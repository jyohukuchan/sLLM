// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Synchronous active1/waiting0 SQ8 serving session contracts.
//!
//! This module is separate from `sq8_generation_runtime`: the P7 fixed request and its audited
//! result schemas remain unchanged while serving gains variable prompt lengths and reusable state.

use crate::decoder::{PagedDecodeShape, PagedDecodeState};
use crate::loader::{read_named_passthrough_f32, verify_named_passthrough_payload};
use crate::scheduler::{
    KvBlockAllocatorStats, Request, RequestId, SchedulerDecodeRequest, SchedulerState,
};
use crate::sq_canonical::Sq8CanonicalArtifact;
use crate::sq8_embedding_runtime::{
    Qwen3Sq8EmbeddingRuntime, Sq8EmbeddingDeviceIdentity, Sq8EmbeddingExecutionReport,
};
use crate::sq8_generation_runtime::{Sq8GenerationTopLogit, greedy_top1_finite};
use crate::sq8_layer_oracle::{
    QWEN3_14B_HEAD_DIM, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_KV_HEADS, QWEN3_14B_Q_HEADS,
    QWEN3_14B_VALUE_DIM,
};
use crate::sq8_layer_runtime::{Qwen3Sq8LayerNormValues, validate_norm_values};
use crate::sq8_model_head_runtime::{
    QWEN3_14B_VOCAB_SIZE, Qwen3Sq8ModelHeadRuntime, Sq8ModelHeadDeviceIdentity,
    validate_qwen3_14b_sq8_r9700_device_info,
};
use crate::sq8_stack_runtime::{
    QWEN3_14B_SQ8_STACK_LAYERS, Qwen3Sq8PagedDecodeRuntime, Qwen3Sq8StackRuntime,
    Sq8PagedStackExecutionReport, Sq8PagedStackPhase,
};
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use ullm_runtime_sys::{DeviceInfo, RuntimeContext, RuntimeStream};

pub const QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS: usize = 4096;
pub const QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS: usize = 16;
pub const QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS: usize =
    QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS / QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS;
pub const QWEN3_14B_SQ8_SERVING_MAX_NEW_TOKENS: usize = 512;
pub const QWEN3_14B_SQ8_SERVING_TOP_K: usize = 20;
pub const QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS: [usize; 2] = [151_645, 151_643];
pub const QWEN3_14B_SQ8_SERVING_ARTIFACT_CONTENT_SHA256: &str =
    "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147";
pub const QWEN3_14B_SQ8_SERVING_PACKAGE_MANIFEST_SHA256: &str =
    "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb";

const SERVING_INTERNAL_REQUEST_ID: RequestId = RequestId(1);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sq8SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: i64,
}

impl Sq8SamplingParams {
    pub const fn greedy(seed: i64) -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: QWEN3_14B_SQ8_SERVING_TOP_K,
            seed,
        }
    }

    fn validate(&self) -> Result<(), Sq8ServingError> {
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(Sq8ServingError::invalid_request(format!(
                "temperature must be finite and in 0..=2, got {}",
                self.temperature
            )));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(Sq8ServingError::invalid_request(format!(
                "top_p must be finite and in 0<top_p<=1, got {}",
                self.top_p
            )));
        }
        if self.top_k != QWEN3_14B_SQ8_SERVING_TOP_K {
            return Err(Sq8ServingError::invalid_request(format!(
                "top_k must be {}, got {}",
                QWEN3_14B_SQ8_SERVING_TOP_K, self.top_k
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8ServingRequest {
    pub request_id: String,
    pub prompt_token_ids: Vec<usize>,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<usize>,
    pub sampling: Sq8SamplingParams,
}

impl Sq8ServingRequest {
    pub fn greedy(
        request_id: impl Into<String>,
        prompt_token_ids: Vec<usize>,
        max_new_tokens: usize,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            prompt_token_ids,
            max_new_tokens,
            eos_token_ids: QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS.to_vec(),
            sampling: Sq8SamplingParams::greedy(0),
        }
    }

    pub fn validate(&self) -> Result<(), Sq8ServingError> {
        validate_request_id(&self.request_id)?;
        if self.prompt_token_ids.is_empty()
            || self.prompt_token_ids.len() > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS
        {
            return Err(Sq8ServingError::invalid_request(format!(
                "prompt token count must be in 1..={}, got {}",
                QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS,
                self.prompt_token_ids.len()
            )));
        }
        if let Some((index, token_id)) = self
            .prompt_token_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token_id)| *token_id >= QWEN3_14B_VOCAB_SIZE)
        {
            return Err(Sq8ServingError::invalid_request(format!(
                "prompt_token_ids[{index}]={token_id} exceeds vocabulary size {QWEN3_14B_VOCAB_SIZE}"
            )));
        }
        if !(1..=QWEN3_14B_SQ8_SERVING_MAX_NEW_TOKENS).contains(&self.max_new_tokens) {
            return Err(Sq8ServingError::invalid_request(format!(
                "max_new_tokens must be in 1..={}, got {}",
                QWEN3_14B_SQ8_SERVING_MAX_NEW_TOKENS, self.max_new_tokens
            )));
        }
        let reserved_tokens = self
            .prompt_token_ids
            .len()
            .checked_add(self.max_new_tokens)
            .ok_or_else(|| Sq8ServingError::invalid_request("context token count overflows"))?;
        if reserved_tokens > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS {
            return Err(Sq8ServingError::invalid_request(format!(
                "prompt plus completion exceeds context: requested={reserved_tokens} context={QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS}"
            )));
        }
        if self.eos_token_ids != QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS {
            return Err(Sq8ServingError::invalid_request(format!(
                "eos_token_ids must be {:?}, got {:?}",
                QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS, self.eos_token_ids
            )));
        }
        self.sampling.validate()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Sq8CancellationToken {
    flag: Arc<AtomicBool>,
}

impl Sq8CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8ReleaseOutcome {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8ReleaseSummary {
    pub request_id: String,
    pub outcome: Sq8ReleaseOutcome,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub reset_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sq8ServingAdvance {
    PromptProgress {
        prompt_tokens_processed: usize,
        cache_len: usize,
        execution_width: usize,
    },
    Token {
        token_id: usize,
        generated_index: usize,
        cache_len: usize,
        terminal_reason: Option<Sq8FinishReason>,
    },
    CancellationObserved,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8ServingOracleCapture {
    pub position: usize,
    pub top1: Sq8GenerationTopLogit,
    pub final_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub final_hidden_f32_le_sha256: String,
    pub logits_f32_le_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8ServingOracleAdvance {
    pub advance: Sq8ServingAdvance,
    pub capture: Option<Sq8ServingOracleCapture>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8ServingRuntimeStatus {
    Ready,
    Prefilling,
    Decoding,
    Finishing,
    Cancelling,
    Resetting,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8ServingErrorKind {
    InvalidRequest,
    InvalidConfiguration,
    InvalidState,
    FatalRuntime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8ServingError {
    pub kind: Sq8ServingErrorKind,
    pub message: String,
}

impl Sq8ServingError {
    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            kind: Sq8ServingErrorKind::InvalidRequest,
            message: message.into(),
        }
    }

    pub(crate) fn invalid_configuration(message: impl Into<String>) -> Self {
        Self {
            kind: Sq8ServingErrorKind::InvalidConfiguration,
            message: message.into(),
        }
    }

    fn invalid_state(message: impl Into<String>) -> Self {
        Self {
            kind: Sq8ServingErrorKind::InvalidState,
            message: message.into(),
        }
    }

    fn fatal_runtime(message: impl Into<String>) -> Self {
        Self {
            kind: Sq8ServingErrorKind::FatalRuntime,
            message: message.into(),
        }
    }
}

impl fmt::Display for Sq8ServingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Sq8ServingError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8ServingLoadReport {
    pub device: Sq8ModelHeadDeviceIdentity,
    pub artifact_content_sha256: String,
    pub package_manifest_sha256: String,
    pub canonical_package_dir: PathBuf,
    pub upload_chunk_bytes: usize,
    pub stack_layers: usize,
    pub cache_layers: usize,
    pub cache_shape: PagedDecodeShape,
    pub block_table_entries: usize,
    pub kv_cache_bytes_per_layer: usize,
    pub total_kv_cache_bytes: usize,
    pub prompt_execution_width: usize,
    pub embedding_payload_sha256: String,
    pub final_norm_payload_sha256: String,
    pub lm_head_payload_sha256: String,
}

impl Sq8ServingLoadReport {
    pub fn validate(&self) -> Result<(), Sq8ServingError> {
        validate_device_identity(&self.device).map_err(Sq8ServingError::invalid_configuration)?;
        if self.artifact_content_sha256 != QWEN3_14B_SQ8_SERVING_ARTIFACT_CONTENT_SHA256 {
            return Err(Sq8ServingError::invalid_configuration(format!(
                "serving artifact identity mismatch: expected={} actual={}",
                QWEN3_14B_SQ8_SERVING_ARTIFACT_CONTENT_SHA256, self.artifact_content_sha256
            )));
        }
        if self.package_manifest_sha256 != QWEN3_14B_SQ8_SERVING_PACKAGE_MANIFEST_SHA256 {
            return Err(Sq8ServingError::invalid_configuration(format!(
                "serving package identity mismatch: expected={} actual={}",
                QWEN3_14B_SQ8_SERVING_PACKAGE_MANIFEST_SHA256, self.package_manifest_sha256
            )));
        }
        if self.upload_chunk_bytes == 0
            || self.stack_layers != QWEN3_14B_SQ8_STACK_LAYERS
            || self.cache_layers != QWEN3_14B_SQ8_STACK_LAYERS
            || self.cache_shape != qwen3_14b_sq8_serving_cache_shape()
            || self.block_table_entries != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
            || self.kv_cache_bytes_per_layer != qwen3_14b_sq8_serving_kv_cache_bytes_per_layer()?
            || self.total_kv_cache_bytes
                != qwen3_14b_sq8_serving_total_kv_cache_bytes(QWEN3_14B_SQ8_STACK_LAYERS)?
            || self.prompt_execution_width != 1
        {
            return Err(Sq8ServingError::invalid_configuration(
                "serving resident geometry/load report mismatch",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8ServingSnapshot {
    pub status: Sq8ServingRuntimeStatus,
    pub active_request_id: Option<String>,
    pub prompt_tokens: usize,
    pub prompt_tokens_processed: usize,
    pub generated_tokens: usize,
    pub cache_lengths: Vec<usize>,
    pub scheduler_active: usize,
    pub scheduler_waiting: usize,
    pub allocator: KvBlockAllocatorStats,
}

#[derive(Debug)]
struct ActiveServingRequest {
    request: Sq8ServingRequest,
    cancel: Sq8CancellationToken,
    prompt_tokens_processed: usize,
    generated_tokens: usize,
    last_generated_token: Option<usize>,
    finish_reason: Option<Sq8FinishReason>,
}

impl ActiveServingRequest {
    fn new(request: Sq8ServingRequest, cancel: Sq8CancellationToken) -> Self {
        Self {
            request,
            cancel,
            prompt_tokens_processed: 0,
            generated_tokens: 0,
            last_generated_token: None,
            finish_reason: None,
        }
    }

    fn expected_cache_len(&self) -> Result<usize, String> {
        if self.generated_tokens == 0 {
            return Ok(self.prompt_tokens_processed);
        }
        self.request
            .prompt_token_ids
            .len()
            .checked_add(self.generated_tokens - 1)
            .ok_or_else(|| "serving expected cache length overflows".to_string())
    }

    fn terminal_reason(&self, token_id: usize) -> Option<Sq8FinishReason> {
        if self.request.eos_token_ids.contains(&token_id) {
            Some(Sq8FinishReason::Stop)
        } else if self.generated_tokens + 1 == self.request.max_new_tokens {
            Some(Sq8FinishReason::Length)
        } else {
            None
        }
    }
}

enum GeneratedTokenCommit<'a> {
    Prefill,
    Decode(&'a [SchedulerDecodeRequest]),
}

/// Owns one resident Qwen3-14B SQ8 model and one reusable active1/waiting0 session.
#[derive(Debug)]
pub struct Qwen3Sq8ServingSession {
    load_report: Sq8ServingLoadReport,
    stack: Qwen3Sq8StackRuntime,
    decode: Qwen3Sq8PagedDecodeRuntime,
    caches: Box<[PagedDecodeState; QWEN3_14B_SQ8_STACK_LAYERS]>,
    embedding: Qwen3Sq8EmbeddingRuntime,
    head: Qwen3Sq8ModelHeadRuntime,
    scheduler: SchedulerState,
    active: Option<ActiveServingRequest>,
    state: Sq8ServingRuntimeStatus,
    failure_reason: Option<String>,
}

impl Qwen3Sq8ServingSession {
    pub fn load(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        artifact: &Sq8CanonicalArtifact,
        package_path: impl AsRef<Path>,
        norms: Vec<Qwen3Sq8LayerNormValues>,
        upload_chunk_bytes: usize,
    ) -> Result<Self, Sq8ServingError> {
        if upload_chunk_bytes == 0 {
            return Err(Sq8ServingError::invalid_configuration(
                "serving upload chunk size must be nonzero",
            ));
        }
        if artifact.manifest().integrity.content_sha256
            != QWEN3_14B_SQ8_SERVING_ARTIFACT_CONTENT_SHA256
        {
            return Err(Sq8ServingError::invalid_configuration(format!(
                "serving artifact identity mismatch: expected={} actual={}",
                QWEN3_14B_SQ8_SERVING_ARTIFACT_CONTENT_SHA256,
                artifact.manifest().integrity.content_sha256
            )));
        }
        let package_path = package_path.as_ref();
        let load_result = (|| {
            let device_info = context.device_info()?;
            validate_qwen3_14b_sq8_r9700_device_info(&device_info)?;
            let stack = Qwen3Sq8StackRuntime::load(
                context,
                stream,
                artifact,
                1,
                norms,
                upload_chunk_bytes,
            )?;
            let embedding =
                Qwen3Sq8EmbeddingRuntime::load(context, stream, package_path, upload_chunk_bytes)?;
            let head =
                Qwen3Sq8ModelHeadRuntime::load(context, stream, package_path, upload_chunk_bytes)?;
            validate_component_device_identity(
                embedding.device_identity(),
                head.device_identity(),
            )?;
            if embedding.load_report().package.manifest_sha256 != head.package_manifest_sha256() {
                return Err(format!(
                    "serving package manifest mismatch: embedding={} head={}",
                    embedding.load_report().package.manifest_sha256,
                    head.package_manifest_sha256()
                ));
            }
            if head.package_manifest_sha256() != QWEN3_14B_SQ8_SERVING_PACKAGE_MANIFEST_SHA256 {
                return Err(format!(
                    "serving package identity mismatch: expected={} actual={}",
                    QWEN3_14B_SQ8_SERVING_PACKAGE_MANIFEST_SHA256,
                    head.package_manifest_sha256()
                ));
            }

            let decode = Qwen3Sq8PagedDecodeRuntime::allocate(context)?;
            let cache_shape = qwen3_14b_sq8_serving_cache_shape();
            cache_shape.validate()?;
            let block_table = qwen3_14b_sq8_serving_block_table().map_err(|err| err.to_string())?;
            let mut cache_values = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
            for layer_index in 0..QWEN3_14B_SQ8_STACK_LAYERS {
                cache_values.push(
                    PagedDecodeState::new(context, stream, cache_shape, block_table.clone())
                        .map_err(|err| {
                            format!(
                                "failed to allocate serving layer {layer_index} KV cache: {err}"
                            )
                        })?,
                );
            }
            let caches: [PagedDecodeState; QWEN3_14B_SQ8_STACK_LAYERS] = cache_values
                .try_into()
                .map_err(|values: Vec<PagedDecodeState>| {
                    format!(
                        "serving cache array length mismatch: expected={} actual={}",
                        QWEN3_14B_SQ8_STACK_LAYERS,
                        values.len()
                    )
                })?;
            let load_report = Sq8ServingLoadReport {
                device: head.device_identity().clone(),
                artifact_content_sha256: stack.artifact_content_sha256().to_string(),
                package_manifest_sha256: head.package_manifest_sha256().to_string(),
                canonical_package_dir: embedding
                    .load_report()
                    .package
                    .canonical_package_dir
                    .clone(),
                upload_chunk_bytes,
                stack_layers: stack.layer_count(),
                cache_layers: caches.len(),
                cache_shape,
                block_table_entries: block_table.len(),
                kv_cache_bytes_per_layer: qwen3_14b_sq8_serving_kv_cache_bytes_per_layer()
                    .map_err(|err| err.to_string())?,
                total_kv_cache_bytes: qwen3_14b_sq8_serving_total_kv_cache_bytes(
                    QWEN3_14B_SQ8_STACK_LAYERS,
                )
                .map_err(|err| err.to_string())?,
                prompt_execution_width: 1,
                embedding_payload_sha256: embedding.load_report().payload.payload_sha256.clone(),
                final_norm_payload_sha256: head.final_norm_identity().payload_sha256.clone(),
                lm_head_payload_sha256: head.lm_head_identity().payload_sha256.clone(),
            };
            load_report.validate().map_err(|err| err.to_string())?;
            let session = Self {
                load_report,
                stack,
                decode,
                caches: Box::new(caches),
                embedding,
                head,
                scheduler: SchedulerState::with_block_size(
                    u32::try_from(QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS)
                        .map_err(|_| "serving cache block count does not fit u32".to_string())?,
                    u32::try_from(QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS)
                        .map_err(|_| "serving block size does not fit u32".to_string())?,
                ),
                active: None,
                state: Sq8ServingRuntimeStatus::Ready,
                failure_reason: None,
            };
            session.validate_ready_baseline()?;
            Ok(session)
        })();
        match load_result {
            Ok(session) => Ok(session),
            Err(operation_error) => Err(Sq8ServingError::fatal_runtime(
                load_error_after_stream_recovery(stream, operation_error),
            )),
        }
    }

    pub fn status(&self) -> Sq8ServingRuntimeStatus {
        self.state
    }

    pub fn failure_reason(&self) -> Option<&str> {
        self.failure_reason.as_deref()
    }

    pub fn load_report(&self) -> &Sq8ServingLoadReport {
        &self.load_report
    }

    pub fn snapshot(&self) -> Sq8ServingSnapshot {
        let (active_request_id, prompt_tokens, prompt_tokens_processed, generated_tokens) = self
            .active
            .as_ref()
            .map(|active| {
                (
                    Some(active.request.request_id.clone()),
                    active.request.prompt_token_ids.len(),
                    active.prompt_tokens_processed,
                    active.generated_tokens,
                )
            })
            .unwrap_or((None, 0, 0, 0));
        Sq8ServingSnapshot {
            status: self.state,
            active_request_id,
            prompt_tokens,
            prompt_tokens_processed,
            generated_tokens,
            cache_lengths: self
                .caches
                .iter()
                .map(PagedDecodeState::written_len)
                .collect(),
            scheduler_active: self.scheduler.active_len(),
            scheduler_waiting: self.scheduler.waiting_len(),
            allocator: self.scheduler.allocator_stats(),
        }
    }

    pub fn start(
        &mut self,
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
        stream: &mut RuntimeStream,
    ) -> Result<(), Sq8ServingError> {
        match self.state {
            Sq8ServingRuntimeStatus::Ready => {}
            Sq8ServingRuntimeStatus::Failed => return Err(self.failed_error()),
            state => {
                return Err(self.fail_runtime(
                    stream,
                    format!("serving start requires Ready, got {state:?}"),
                ));
            }
        }
        request.validate()?;
        validate_p8b_greedy_execution(request.sampling)?;
        if let Err(err) = self.validate_ready_baseline() {
            return Err(
                self.fail_runtime(stream, format!("serving baseline validation failed: {err}"))
            );
        }
        let expected_table = qwen3_14b_sq8_serving_block_table()?;
        let preflight = (|| {
            self.stack.validate_paged_serving_sequence_start(
                &self.decode,
                self.caches.as_ref(),
                false,
            )?;
            self.embedding.validate_serving_preflight()?;
            self.head.validate_serving_preflight()?;
            Ok::<(), String>(())
        })();
        if let Err(err) = preflight {
            return Err(Sq8ServingError::invalid_configuration(format!(
                "serving start preflight failed before mutation: {err}"
            )));
        }

        let scheduler_request = Request {
            id: SERVING_INTERNAL_REQUEST_ID,
            prompt_tokens: request.prompt_token_ids.len(),
            max_new_tokens: request.max_new_tokens,
        };
        let active = ActiveServingRequest::new(request, cancel);
        let allocation = match self
            .scheduler
            .activate_single_request_with_all_blocks(scheduler_request)
        {
            Ok(allocation) => allocation,
            Err(err) => {
                return Err(self.fail_runtime(
                    stream,
                    format!("serving scheduler activation failed: {err}"),
                ));
            }
        };
        if allocation.allocation.blocks != expected_table {
            return Err(self.fail_runtime(
                stream,
                format!(
                    "serving fixed allocation mismatch: {:?}",
                    allocation.allocation.blocks
                ),
            ));
        }
        self.stack.begin_paged_serving_sequence();
        self.active = Some(active);
        self.state = Sq8ServingRuntimeStatus::Prefilling;
        Ok(())
    }

    pub fn advance_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ServingAdvance, Sq8ServingError> {
        match self.state {
            Sq8ServingRuntimeStatus::Prefilling | Sq8ServingRuntimeStatus::Decoding => {}
            Sq8ServingRuntimeStatus::Ready => {
                return Err(Sq8ServingError::invalid_state(
                    "serving advance requires an active request",
                ));
            }
            Sq8ServingRuntimeStatus::Failed => return Err(self.failed_error()),
            state => {
                return Err(self.fail_runtime(
                    stream,
                    format!("serving advance is invalid in state {state:?}"),
                ));
            }
        }
        let cancelled = match self.active_cancelled() {
            Ok(cancelled) => cancelled,
            Err(err) => return Err(self.fail_runtime(stream, err)),
        };
        if cancelled {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Ok(Sq8ServingAdvance::CancellationObserved);
        }

        let result = match self.state {
            Sq8ServingRuntimeStatus::Prefilling => self
                .advance_prefill_synchronized(stream, false)
                .map(|result| result.advance),
            Sq8ServingRuntimeStatus::Decoding => self.advance_decode_synchronized(stream),
            _ => unreachable!("state checked above"),
        };
        result.map_err(|err| self.fail_runtime(stream, err))
    }

    /// Captures final hidden/logits only for the first token oracle gate.
    pub fn advance_prefill_oracle_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ServingOracleAdvance, Sq8ServingError> {
        match self.state {
            Sq8ServingRuntimeStatus::Prefilling => {}
            Sq8ServingRuntimeStatus::Ready => {
                return Err(Sq8ServingError::invalid_state(
                    "serving prefill oracle requires an active request",
                ));
            }
            Sq8ServingRuntimeStatus::Failed => return Err(self.failed_error()),
            state => {
                return Err(self.fail_runtime(
                    stream,
                    format!("serving prefill oracle is invalid in state {state:?}"),
                ));
            }
        }
        let cancelled = match self.active_cancelled() {
            Ok(cancelled) => cancelled,
            Err(err) => return Err(self.fail_runtime(stream, err)),
        };
        if cancelled {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Ok(Sq8ServingOracleAdvance {
                advance: Sq8ServingAdvance::CancellationObserved,
                capture: None,
            });
        }
        self.advance_prefill_synchronized(stream, true)
            .map_err(|err| self.fail_runtime(stream, err))
    }

    pub fn finish_and_reset_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ReleaseSummary, Sq8ServingError> {
        if self.state != Sq8ServingRuntimeStatus::Finishing {
            return self.reject_cleanup_state(stream, "finish", Sq8ServingRuntimeStatus::Finishing);
        }
        let finish_reason = self
            .active
            .as_ref()
            .and_then(|active| active.finish_reason)
            .ok_or_else(|| {
                self.fail_runtime(stream, "serving finishing state has no finish reason")
            })?;
        let outcome = match finish_reason {
            Sq8FinishReason::Stop => Sq8ReleaseOutcome::Stop,
            Sq8FinishReason::Length => Sq8ReleaseOutcome::Length,
        };
        self.reset_active_synchronized(outcome, stream)
    }

    pub fn abort_and_reset_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ReleaseSummary, Sq8ServingError> {
        if self.state != Sq8ServingRuntimeStatus::Cancelling {
            return self.reject_cleanup_state(stream, "abort", Sq8ServingRuntimeStatus::Cancelling);
        }
        self.reset_active_synchronized(Sq8ReleaseOutcome::Cancelled, stream)
    }

    fn reject_cleanup_state<T>(
        &mut self,
        stream: &mut RuntimeStream,
        operation: &str,
        expected: Sq8ServingRuntimeStatus,
    ) -> Result<T, Sq8ServingError> {
        if self.state == Sq8ServingRuntimeStatus::Ready {
            return Err(Sq8ServingError::invalid_state(format!(
                "serving {operation} requires {expected:?}, got Ready"
            )));
        }
        Err(self.fail_runtime(
            stream,
            format!(
                "serving {operation} requires {expected:?}, got {:?}",
                self.state
            ),
        ))
    }
}

impl Qwen3Sq8ServingSession {
    fn advance_prefill_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
        capture_oracle: bool,
    ) -> Result<Sq8ServingOracleAdvance, String> {
        let (position, prompt_tokens, token_id) = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "serving Prefilling state has no active request".to_string())?;
            let position = active.prompt_tokens_processed;
            let token_id = *active
                .request
                .prompt_token_ids
                .get(position)
                .ok_or_else(|| {
                    format!(
                        "serving prompt position {position} exceeds prompt length {}",
                        active.request.prompt_token_ids.len()
                    )
                })?;
            (position, active.request.prompt_token_ids.len(), token_id)
        };
        self.execute_m1_stack_token(token_id, position, stream)?;
        let scheduler_cached = self.commit_prompt_progress(position)?;
        if self.active_cancelled()? {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Ok(Sq8ServingOracleAdvance {
                advance: Sq8ServingAdvance::CancellationObserved,
                capture: None,
            });
        }
        if scheduler_cached < prompt_tokens {
            return Ok(Sq8ServingOracleAdvance {
                advance: Sq8ServingAdvance::PromptProgress {
                    prompt_tokens_processed: scheduler_cached,
                    cache_len: scheduler_cached,
                    execution_width: 1,
                },
                capture: None,
            });
        }

        let (top1, capture) = self.run_m1_head_synchronized(stream, capture_oracle)?;
        let advance =
            self.commit_generated_token(top1, scheduler_cached, GeneratedTokenCommit::Prefill)?;
        Ok(Sq8ServingOracleAdvance {
            capture: if matches!(advance, Sq8ServingAdvance::Token { .. }) {
                capture
            } else {
                None
            },
            advance,
        })
    }

    fn advance_decode_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ServingAdvance, String> {
        let (prompt_tokens, generated_tokens, input_token_id, expected_position) = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "serving Decoding state has no active request".to_string())?;
            if active.prompt_tokens_processed != active.request.prompt_token_ids.len()
                || active.generated_tokens == 0
            {
                return Err("serving decode counters are not initialized".into());
            }
            let expected_position = active.expected_cache_len()?;
            let input_token_id = active
                .last_generated_token
                .ok_or_else(|| "serving decode has no feedback token".to_string())?;
            (
                active.request.prompt_token_ids.len(),
                active.generated_tokens,
                input_token_id,
                expected_position,
            )
        };
        let ready = self.scheduler.ready_decode_batch(1)?;
        if ready.len() != 1 {
            return Err(format!(
                "serving expected one ready decode request, got {}",
                ready.len()
            ));
        }
        let decode_request = &ready[0];
        if decode_request.request.id != SERVING_INTERNAL_REQUEST_ID
            || decode_request.request.prompt_tokens != prompt_tokens
            || decode_request.generated_tokens != generated_tokens
            || decode_request.cached_tokens != expected_position
            || decode_request.cache_position != expected_position
            || decode_request.next_cache_len != expected_position + 1
            || decode_request.allocation.blocks
                != qwen3_14b_sq8_serving_block_table().map_err(|err| err.message)?
        {
            return Err(format!(
                "serving ready decode metadata mismatch: {decode_request:?}"
            ));
        }

        self.execute_m1_stack_token(input_token_id, expected_position, stream)?;
        validate_cache_lengths(self.caches.as_ref(), expected_position + 1)?;
        if self.active_cancelled()? {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Ok(Sq8ServingAdvance::CancellationObserved);
        }
        let (top1, capture) = self.run_m1_head_synchronized(stream, false)?;
        debug_assert!(capture.is_none());
        self.commit_generated_token(
            top1,
            expected_position + 1,
            GeneratedTokenCommit::Decode(&ready),
        )
    }

    fn execute_m1_stack_token(
        &mut self,
        token_id: usize,
        position: usize,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8PagedStackExecutionReport, String> {
        if token_id >= QWEN3_14B_VOCAB_SIZE {
            return Err(format!(
                "serving M=1 input token exceeds vocabulary: {token_id}"
            ));
        }
        if position >= QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS {
            return Err(format!("serving M=1 position exceeds context: {position}"));
        }
        validate_cache_lengths(self.caches.as_ref(), position)?;
        let embedding_report = self.embedding.enqueue_token_resident(token_id, stream)?;
        validate_embedding_report(&embedding_report, token_id, &self.load_report)?;
        let (embedding_output, resident_report) = self.embedding.resident_output()?;
        if resident_report != &embedding_report {
            return Err("serving embedding report changed before M=1 execution".into());
        }
        let report = self
            .stack
            .run_paged_m1_sequence_step_optimized_synchronized(
                &mut self.decode,
                embedding_output,
                position,
                &mut self.caches[..],
                stream,
            )?;
        report.validate_contract()?;
        if report.phase != Sq8PagedStackPhase::Decode
            || report.position != position
            || report.stack.sequence_len != 1
            || report.stack.artifact_content_sha256 != self.load_report.artifact_content_sha256
            || report
                .cache_lengths
                .iter()
                .any(|length| *length != position + 1)
            || report.stack.fallback_used
            || report.stack.host_staging_used
        {
            return Err(format!(
                "serving M=1 stack report failed at position {position}: {report:?}"
            ));
        }
        Ok(report)
    }

    fn run_m1_head_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
        capture_oracle: bool,
    ) -> Result<(Sq8GenerationTopLogit, Option<Sq8ServingOracleCapture>), String> {
        let result = if capture_oracle {
            self.head
                .run_m1_serving_oracle_synchronized(&self.decode, stream)?
        } else {
            self.head
                .run_m1_serving_logits_synchronized(&self.decode, stream)?
        };
        result.validate_contract()?;
        if result.report.binding.device != self.load_report.device
            || result.report.binding.package_manifest_sha256
                != self.load_report.package_manifest_sha256
            || result.report.binding.artifact_content_sha256
                != self.load_report.artifact_content_sha256
            || result.report.final_norm.payload_sha256 != self.load_report.final_norm_payload_sha256
            || result.report.lm_head.payload_sha256 != self.load_report.lm_head_payload_sha256
            || result.report.fallback_used
            || result.report.host_staging_used
        {
            return Err("serving M=1 model-head identity/report mismatch".into());
        }
        let top1 = greedy_top1_finite(&result.logits)?;
        let capture = if capture_oracle {
            let final_hidden = result.final_hidden.ok_or_else(|| {
                "serving oracle head did not return final-hidden capture".to_string()
            })?;
            Some(Sq8ServingOracleCapture {
                position: result.report.position,
                top1,
                final_hidden,
                logits: result.logits,
                final_hidden_f32_le_sha256: result
                    .report
                    .final_hidden_health
                    .as_ref()
                    .ok_or_else(|| {
                        "serving oracle head did not report final-hidden health".to_string()
                    })?
                    .f32_le_sha256
                    .clone(),
                logits_f32_le_sha256: result.report.logits_health.f32_le_sha256.clone(),
            })
        } else {
            if result.final_hidden.is_some() {
                return Err("lean serving head unexpectedly captured final hidden".into());
            }
            None
        };
        Ok((top1, capture))
    }

    fn commit_prompt_progress(&mut self, position: usize) -> Result<usize, String> {
        validate_cache_lengths(self.caches.as_ref(), position + 1)?;
        let active = self
            .active
            .as_ref()
            .ok_or_else(|| "serving prompt commit has no active request".to_string())?;
        let scheduled = self
            .scheduler
            .active_request(SERVING_INTERNAL_REQUEST_ID)
            .ok_or_else(|| "serving prompt commit has no scheduled request".to_string())?;
        if active.prompt_tokens_processed != position
            || active.generated_tokens != 0
            || scheduled.cached_tokens != position
            || scheduled.generated_tokens != 0
            || scheduled.request.prompt_tokens != active.request.prompt_token_ids.len()
            || scheduled.request.max_new_tokens != active.request.max_new_tokens
        {
            return Err("serving prompt commit metadata is stale".into());
        }
        let expected = position
            .checked_add(1)
            .ok_or_else(|| "serving prompt position overflows".to_string())?;
        let actual = self
            .scheduler
            .advance_prefill_token(SERVING_INTERNAL_REQUEST_ID)?;
        if actual != expected {
            return Err(format!(
                "serving scheduler prompt progress mismatch: expected={expected} actual={actual}"
            ));
        }
        self.active
            .as_mut()
            .expect("active request was validated before scheduler prompt commit")
            .prompt_tokens_processed = actual;
        Ok(actual)
    }

    fn commit_generated_token(
        &mut self,
        top1: Sq8GenerationTopLogit,
        cache_len: usize,
        commit: GeneratedTokenCommit<'_>,
    ) -> Result<Sq8ServingAdvance, String> {
        if top1.token_id >= QWEN3_14B_VOCAB_SIZE || !top1.logit.is_finite() {
            return Err(format!("serving sampled invalid top1: {top1:?}"));
        }
        let (generated_index, next_generated_tokens, terminal_reason, cancelled) = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "serving generated token has no active request".to_string())?;
            if active.prompt_tokens_processed != active.request.prompt_token_ids.len()
                || active.finish_reason.is_some()
                || active.generated_tokens >= active.request.max_new_tokens
            {
                return Err("serving generated token metadata is not publishable".into());
            }
            let generated_index = active.generated_tokens;
            let next_generated_tokens = generated_index
                .checked_add(1)
                .ok_or_else(|| "serving generated token counter overflows".to_string())?;
            let expected_cache_len = active
                .request
                .prompt_token_ids
                .len()
                .checked_add(next_generated_tokens.saturating_sub(1))
                .ok_or_else(|| "serving generated token cache length overflows".to_string())?;
            if cache_len != expected_cache_len {
                return Err(format!(
                    "serving emitted token cache mismatch: expected={expected_cache_len} actual={cache_len}"
                ));
            }
            let scheduled = self
                .scheduler
                .active_request(SERVING_INTERNAL_REQUEST_ID)
                .ok_or_else(|| "serving generated token has no scheduled request".to_string())?;
            if scheduled.request.prompt_tokens != active.request.prompt_token_ids.len()
                || scheduled.request.max_new_tokens != active.request.max_new_tokens
                || scheduled.generated_tokens != active.generated_tokens
            {
                return Err("serving generated token scheduler metadata is stale".into());
            }
            match &commit {
                GeneratedTokenCommit::Prefill => {
                    if generated_index != 0
                        || scheduled.cached_tokens != active.request.prompt_token_ids.len()
                    {
                        return Err("serving prefill token commit metadata is stale".into());
                    }
                }
                GeneratedTokenCommit::Decode(ready) => {
                    if ready.len() != 1
                        || ready[0].request != scheduled.request
                        || ready[0].allocation != scheduled.allocation
                        || ready[0].cached_tokens != scheduled.cached_tokens
                        || ready[0].generated_tokens != scheduled.generated_tokens
                        || ready[0].next_cache_len != cache_len
                    {
                        return Err("serving decode token commit metadata is stale".into());
                    }
                }
            }
            (
                generated_index,
                next_generated_tokens,
                active.terminal_reason(top1.token_id),
                active.cancel.is_cancelled(),
            )
        };

        // This is the final cancellation observation before scheduler/token publication.
        if cancelled {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Ok(Sq8ServingAdvance::CancellationObserved);
        }
        match commit {
            GeneratedTokenCommit::Prefill => self
                .scheduler
                .record_prefill_generated_token(SERVING_INTERNAL_REQUEST_ID)?,
            GeneratedTokenCommit::Decode(ready) => self.scheduler.advance_decode_batch(ready)?,
        }
        let active = self
            .active
            .as_mut()
            .expect("active request was validated before scheduler token commit");
        active.generated_tokens = next_generated_tokens;
        active.last_generated_token = Some(top1.token_id);
        active.finish_reason = terminal_reason;
        self.state = if terminal_reason.is_some() {
            Sq8ServingRuntimeStatus::Finishing
        } else {
            Sq8ServingRuntimeStatus::Decoding
        };
        Ok(Sq8ServingAdvance::Token {
            token_id: top1.token_id,
            generated_index,
            cache_len,
            terminal_reason,
        })
    }

    fn reset_active_synchronized(
        &mut self,
        outcome: Sq8ReleaseOutcome,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ReleaseSummary, Sq8ServingError> {
        let (request_id, prompt_tokens, generated_tokens) = match self.active.as_ref() {
            Some(active) => (
                active.request.request_id.clone(),
                active.request.prompt_token_ids.len(),
                active.generated_tokens,
            ),
            None => {
                return Err(self.fail_runtime(stream, "serving reset has no active request"));
            }
        };
        let expected_table = match qwen3_14b_sq8_serving_block_table() {
            Ok(table) => table,
            Err(err) => return Err(self.fail_runtime(stream, err.to_string())),
        };
        let reset_preflight = (|| {
            let scheduled = self
                .scheduler
                .active_request(SERVING_INTERNAL_REQUEST_ID)
                .ok_or_else(|| "serving reset has no scheduled request".to_string())?;
            if self.scheduler.active_len() != 1
                || !self.scheduler.waiting_is_empty()
                || scheduled.allocation.blocks != expected_table
            {
                return Err("serving reset scheduler metadata is inconsistent".into());
            }
            Ok::<(), String>(())
        })();
        if let Err(err) = reset_preflight {
            return Err(self.fail_runtime(stream, err.to_string()));
        }

        self.state = Sq8ServingRuntimeStatus::Resetting;
        let released = self.scheduler.release_request(SERVING_INTERNAL_REQUEST_ID);
        if released != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS {
            return Err(self.fail_runtime(
                stream,
                format!(
                    "serving scheduler released {released} blocks, expected {}",
                    QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
                ),
            ));
        }
        let reset_result = (|| {
            for (layer_index, cache) in self.caches.iter_mut().enumerate() {
                cache.enqueue_serving_reset(stream).map_err(|err| {
                    format!("failed to enqueue serving layer {layer_index} reset: {err}")
                })?;
            }
            self.stack.enqueue_serving_reset(&mut self.decode, stream)?;
            self.embedding.enqueue_serving_reset(stream)?;
            self.head.enqueue_serving_reset(stream)?;
            stream
                .synchronize()
                .map_err(|err| format!("failed to synchronize serving reset: {err}"))?;
            Ok::<(), String>(())
        })();
        if let Err(err) = reset_result {
            return Err(self.fail_runtime(stream, err));
        }

        for cache in self.caches.iter_mut() {
            cache.commit_serving_reset();
        }
        self.stack.commit_serving_reset(&mut self.decode);
        self.embedding.commit_serving_reset();
        self.head.commit_serving_reset();
        validate_scheduler_baseline(&self.scheduler)
            .map_err(|err| self.fail_runtime(stream, err))?;
        self.active = None;
        self.state = Sq8ServingRuntimeStatus::Ready;
        if let Err(err) = self.validate_ready_baseline() {
            return Err(
                self.fail_runtime(stream, format!("serving post-reset baseline failed: {err}"))
            );
        }
        Ok(Sq8ReleaseSummary {
            request_id,
            outcome,
            prompt_tokens,
            generated_tokens,
            reset_complete: true,
        })
    }

    fn active_cancelled(&self) -> Result<bool, String> {
        self.active
            .as_ref()
            .map(|active| active.cancel.is_cancelled())
            .ok_or_else(|| "serving active request is missing".to_string())
    }

    fn failed_error(&self) -> Sq8ServingError {
        Sq8ServingError::fatal_runtime(format!(
            "serving session is failed: {}",
            self.failure_reason.as_deref().unwrap_or("unknown failure")
        ))
    }

    fn validate_ready_baseline(&self) -> Result<(), String> {
        self.load_report.validate().map_err(|err| err.to_string())?;
        if self.state != Sq8ServingRuntimeStatus::Ready
            || self.failure_reason.is_some()
            || self.active.is_some()
        {
            return Err("serving Ready metadata is not at baseline".into());
        }
        if self.stack.config().sequence_len != 1
            || self.stack.layer_count() != QWEN3_14B_SQ8_STACK_LAYERS
            || self.stack.artifact_content_sha256() != self.load_report.artifact_content_sha256
            || self.stack.poison_reason().is_some()
            || self.embedding.poison_reason().is_some()
            || self.head.poison_reason().is_some()
        {
            return Err("serving resident model state is not reusable".into());
        }
        self.stack.validate_serving_baseline(&self.decode)?;
        self.embedding.validate_serving_baseline()?;
        self.head.validate_serving_baseline()?;
        validate_component_device_identity(
            self.embedding.device_identity(),
            self.head.device_identity(),
        )?;
        if self.embedding.load_report().package.manifest_sha256
            != self.load_report.package_manifest_sha256
            || self.head.package_manifest_sha256() != self.load_report.package_manifest_sha256
        {
            return Err("serving resident package identity changed".into());
        }
        let expected_table = qwen3_14b_sq8_serving_block_table().map_err(|err| err.to_string())?;
        if self.caches.len() != QWEN3_14B_SQ8_STACK_LAYERS
            || self.caches.iter().any(|cache| {
                cache.shape() != qwen3_14b_sq8_serving_cache_shape()
                    || cache.block_table() != expected_table
                    || cache.written_len() != 0
            })
        {
            return Err("serving resident KV cache baseline mismatch".into());
        }
        validate_scheduler_baseline(&self.scheduler)
    }

    fn fail_runtime(
        &mut self,
        stream: &mut RuntimeStream,
        operation_error: impl Into<String>,
    ) -> Sq8ServingError {
        let operation_error = operation_error.into();
        let message = match stream.synchronize() {
            Ok(()) => operation_error,
            Err(sync_error) => format!(
                "{operation_error}; subsequent serving stream recovery failed: {sync_error}"
            ),
        };
        self.state = Sq8ServingRuntimeStatus::Failed;
        if self.failure_reason.is_none() {
            self.failure_reason = Some(message.clone());
        }
        Sq8ServingError::fatal_runtime(message)
    }
}

fn validate_embedding_report(
    report: &Sq8EmbeddingExecutionReport,
    token_id: usize,
    load: &Sq8ServingLoadReport,
) -> Result<(), String> {
    report.validate_contract()?;
    if report.token_id != token_id
        || report.load.package.manifest_sha256 != load.package_manifest_sha256
        || report.load.payload.payload_sha256 != load.embedding_payload_sha256
        || report.fallback_used
        || report.host_staging_used
    {
        return Err("serving embedding identity/report mismatch".into());
    }
    validate_component_device_identity(&report.device, &load.device)
}

fn validate_cache_lengths(caches: &[PagedDecodeState], expected: usize) -> Result<(), String> {
    if caches.len() != QWEN3_14B_SQ8_STACK_LAYERS {
        return Err(format!(
            "serving cache layer count mismatch: expected={} actual={}",
            QWEN3_14B_SQ8_STACK_LAYERS,
            caches.len()
        ));
    }
    if let Some((layer_index, actual)) = caches
        .iter()
        .enumerate()
        .map(|(layer_index, cache)| (layer_index, cache.written_len()))
        .find(|(_, actual)| *actual != expected)
    {
        return Err(format!(
            "serving layer {layer_index} cache length mismatch: expected={expected} actual={actual}"
        ));
    }
    Ok(())
}

fn validate_scheduler_baseline(scheduler: &SchedulerState) -> Result<(), String> {
    let stats = scheduler.allocator_stats();
    if scheduler.active_len() != 0
        || !scheduler.waiting_is_empty()
        || stats.block_size_tokens != QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS as u32
        || stats.total_blocks != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS as u32
        || stats.free_blocks != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
        || stats.allocated_blocks != 0
        || stats.free_runs != 1
        || stats.largest_free_run != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
    {
        return Err(format!(
            "serving scheduler/allocator baseline mismatch: active={} waiting={} stats={stats:?}",
            scheduler.active_len(),
            scheduler.waiting_len()
        ));
    }
    Ok(())
}

fn validate_component_device_identity(
    embedding: &Sq8EmbeddingDeviceIdentity,
    head: &Sq8ModelHeadDeviceIdentity,
) -> Result<(), String> {
    if embedding.device_id != head.device_id
        || embedding.backend != head.backend
        || embedding.name != head.name
        || embedding.gcn_arch_name != head.gcn_arch_name
        || embedding.compute_major != head.compute_major
        || embedding.compute_minor != head.compute_minor
        || embedding.total_global_mem != head.total_global_mem
    {
        return Err(format!(
            "serving component device mismatch: embedding={embedding:?} head={head:?}"
        ));
    }
    validate_device_identity(head)
}

fn validate_device_identity(value: &Sq8ModelHeadDeviceIdentity) -> Result<(), String> {
    validate_qwen3_14b_sq8_r9700_device_info(&DeviceInfo {
        device_id: value.device_id,
        backend: value.backend.clone(),
        name: value.name.clone(),
        total_global_mem: value.total_global_mem,
        compute_major: value.compute_major,
        compute_minor: value.compute_minor,
        gcn_arch_name: value.gcn_arch_name.clone(),
        flags: 0,
    })
}

fn load_error_after_stream_recovery(stream: &mut RuntimeStream, operation_error: String) -> String {
    match stream.synchronize() {
        Ok(()) => operation_error,
        Err(sync_error) => format!(
            "{operation_error}; subsequent serving load stream recovery failed: {sync_error}"
        ),
    }
}

pub fn load_qwen3_14b_sq8_serving_norms(
    package_path: impl AsRef<Path>,
    chunk_bytes: usize,
) -> Result<Vec<Qwen3Sq8LayerNormValues>, Sq8ServingError> {
    if chunk_bytes == 0 {
        return Err(Sq8ServingError::invalid_configuration(
            "serving norm chunk size must be nonzero",
        ));
    }
    let package_path = package_path.as_ref();
    let mut norms = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
    for layer_index in 0..QWEN3_14B_SQ8_STACK_LAYERS {
        let prefix = format!("model.layers.{layer_index}");
        let input = read_verified_serving_norm(
            package_path,
            &format!("{prefix}.input_layernorm.weight"),
            QWEN3_14B_HIDDEN_SIZE,
            chunk_bytes,
        )?;
        let post_attention = read_verified_serving_norm(
            package_path,
            &format!("{prefix}.post_attention_layernorm.weight"),
            QWEN3_14B_HIDDEN_SIZE,
            chunk_bytes,
        )?;
        let q = read_verified_serving_norm(
            package_path,
            &format!("{prefix}.self_attn.q_norm.weight"),
            QWEN3_14B_HEAD_DIM,
            chunk_bytes,
        )?;
        let k = read_verified_serving_norm(
            package_path,
            &format!("{prefix}.self_attn.k_norm.weight"),
            QWEN3_14B_HEAD_DIM,
            chunk_bytes,
        )?;
        let values = Qwen3Sq8LayerNormValues {
            input,
            post_attention,
            q,
            k,
        };
        validate_norm_values(&values).map_err(|err| {
            Sq8ServingError::invalid_configuration(format!(
                "serving layer {layer_index} norm validation failed: {err}"
            ))
        })?;
        norms.push(values);
    }
    if norms.len() != QWEN3_14B_SQ8_STACK_LAYERS {
        return Err(Sq8ServingError::invalid_configuration(format!(
            "serving norm layer count mismatch: expected={} actual={}",
            QWEN3_14B_SQ8_STACK_LAYERS,
            norms.len()
        )));
    }
    Ok(norms)
}

fn read_verified_serving_norm(
    package_path: &Path,
    tensor_name: &str,
    elements: usize,
    chunk_bytes: usize,
) -> Result<Vec<f32>, Sq8ServingError> {
    let expected_shape = [u64::try_from(elements).map_err(|_| {
        Sq8ServingError::invalid_configuration(format!(
            "serving norm element count does not fit u64: {elements}"
        ))
    })?];
    let verification = verify_named_passthrough_payload(
        package_path,
        tensor_name,
        "BF16",
        &expected_shape,
        chunk_bytes,
    )
    .map_err(|err| {
        Sq8ServingError::invalid_configuration(format!(
            "failed to verify serving norm {tensor_name}: {err}"
        ))
    })?;
    let data =
        read_named_passthrough_f32(package_path, tensor_name, chunk_bytes).map_err(|err| {
            Sq8ServingError::invalid_configuration(format!(
                "failed to read serving norm {tensor_name}: {err}"
            ))
        })?;
    if data.dtype != "BF16" || data.shape != expected_shape || data.values.len() != elements {
        return Err(Sq8ServingError::invalid_configuration(format!(
            "serving norm {tensor_name} changed after verification"
        )));
    }
    if let Some((index, value)) = data
        .values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Sq8ServingError::invalid_configuration(format!(
            "serving norm {tensor_name} contains non-finite value {value} at {index}"
        )));
    }
    let mut digest = Sha256::new();
    for value in &data.values {
        digest.update(((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    let decoded_sha256 = format!("{:x}", digest.finalize());
    if decoded_sha256 != verification.payload_sha256 {
        return Err(Sq8ServingError::invalid_configuration(format!(
            "serving norm {tensor_name} checksum changed after verification: expected={} actual={decoded_sha256}",
            verification.payload_sha256
        )));
    }
    Ok(data.values)
}

pub fn qwen3_14b_sq8_serving_cache_shape() -> PagedDecodeShape {
    PagedDecodeShape {
        block_size: QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS,
        cache_blocks: QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS,
        q_heads: QWEN3_14B_Q_HEADS,
        kv_heads: QWEN3_14B_KV_HEADS,
        head_dim: QWEN3_14B_HEAD_DIM,
        value_dim: QWEN3_14B_VALUE_DIM,
    }
}

pub fn qwen3_14b_sq8_serving_block_table() -> Result<Vec<u32>, Sq8ServingError> {
    (0..QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS)
        .map(|block| {
            u32::try_from(block).map_err(|_| {
                Sq8ServingError::invalid_configuration(format!(
                    "serving block index does not fit u32: {block}"
                ))
            })
        })
        .collect()
}

pub fn qwen3_14b_sq8_serving_kv_cache_bytes_per_layer() -> Result<usize, Sq8ServingError> {
    let shape = qwen3_14b_sq8_serving_cache_shape();
    shape.validate().map_err(|err| {
        Sq8ServingError::invalid_configuration(format!("invalid serving cache shape: {err}"))
    })?;
    shape
        .k_cache_elements()
        .and_then(|k| {
            shape.v_cache_elements().and_then(|v| {
                k.checked_add(v)
                    .ok_or_else(|| "KV elements overflow".into())
            })
        })
        .and_then(|elements| {
            elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "KV cache bytes overflow".into())
        })
        .map_err(Sq8ServingError::invalid_configuration)
}

pub fn qwen3_14b_sq8_serving_total_kv_cache_bytes(
    layer_count: usize,
) -> Result<usize, Sq8ServingError> {
    if layer_count == 0 {
        return Err(Sq8ServingError::invalid_configuration(
            "serving layer count must be nonzero",
        ));
    }
    qwen3_14b_sq8_serving_kv_cache_bytes_per_layer()?
        .checked_mul(layer_count)
        .ok_or_else(|| Sq8ServingError::invalid_configuration("total KV cache bytes overflow"))
}

pub fn validate_p8b_greedy_execution(sampling: Sq8SamplingParams) -> Result<(), Sq8ServingError> {
    if sampling.temperature.to_bits() != 0.0_f32.to_bits() {
        return Err(Sq8ServingError::invalid_configuration(
            "P8-B lean serving currently enables only temperature=0 greedy sampling",
        ));
    }
    Ok(())
}

fn validate_request_id(value: &str) -> Result<(), Sq8ServingError> {
    let bytes = value.as_bytes();
    if bytes.is_empty() || bytes.len() > 128 {
        return Err(Sq8ServingError::invalid_request(format!(
            "request_id must contain 1..=128 ASCII bytes, got {}",
            bytes.len()
        )));
    }
    if !bytes[0].is_ascii_alphanumeric()
        || bytes[1..].iter().any(|byte| {
            !byte.is_ascii_alphanumeric() && !matches!(*byte, b'.' | b'_' | b':' | b'-')
        })
    {
        return Err(Sq8ServingError::invalid_request(format!(
            "request_id has invalid syntax: {value:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sq8_stack_runtime::QWEN3_14B_SQ8_STACK_LAYERS;

    #[test]
    fn serving_request_accepts_exact_context_boundary() {
        let request = Sq8ServingRequest::greedy(
            "req-1",
            vec![1; QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS - 1],
            1,
        );
        request.validate().unwrap();
    }

    #[test]
    fn serving_request_rejects_context_overflow_before_execution() {
        let request =
            Sq8ServingRequest::greedy("req-1", vec![1; QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS], 1);
        let err = request.validate().unwrap_err();
        assert_eq!(err.kind, Sq8ServingErrorKind::InvalidRequest);
        assert!(err.message.contains("exceeds context"), "{err}");
    }

    #[test]
    fn serving_request_rejects_invalid_tokens_without_partial_validation() {
        for prompt in [vec![], vec![QWEN3_14B_VOCAB_SIZE]] {
            let err = Sq8ServingRequest::greedy("req-1", prompt, 1)
                .validate()
                .unwrap_err();
            assert_eq!(err.kind, Sq8ServingErrorKind::InvalidRequest);
        }
        for maximum in [0, QWEN3_14B_SQ8_SERVING_MAX_NEW_TOKENS + 1] {
            let err = Sq8ServingRequest::greedy("req-1", vec![1], maximum)
                .validate()
                .unwrap_err();
            assert_eq!(err.kind, Sq8ServingErrorKind::InvalidRequest);
        }
    }

    #[test]
    fn serving_request_id_matches_worker_protocol_rule() {
        for valid in ["a", "A0._:-z", &"x".repeat(128)] {
            Sq8ServingRequest::greedy(valid, vec![1], 1)
                .validate()
                .unwrap();
        }
        for invalid in ["", "-bad", "bad/slash", "space bad", &"x".repeat(129)] {
            let err = Sq8ServingRequest::greedy(invalid, vec![1], 1)
                .validate()
                .unwrap_err();
            assert!(err.message.contains("request_id"), "{err}");
        }
    }

    #[test]
    fn serving_request_requires_frozen_eos_and_sampling_ranges() {
        let mut request = Sq8ServingRequest::greedy("req-1", vec![1], 1);
        request.eos_token_ids.reverse();
        assert!(
            request
                .validate()
                .unwrap_err()
                .message
                .contains("eos_token_ids")
        );

        let mut request = Sq8ServingRequest::greedy("req-1", vec![1], 1);
        request.sampling.top_k = QWEN3_14B_SQ8_SERVING_TOP_K - 1;
        assert!(request.validate().unwrap_err().message.contains("top_k"));

        let mut request = Sq8ServingRequest::greedy("req-1", vec![1], 1);
        request.sampling.top_p = f32::NAN;
        assert!(request.validate().unwrap_err().message.contains("top_p"));

        let mut request = Sq8ServingRequest::greedy("req-1", vec![1], 1);
        request.sampling.temperature = 2.01;
        assert!(
            request
                .validate()
                .unwrap_err()
                .message
                .contains("temperature")
        );
    }

    #[test]
    fn p8b_execution_gate_rejects_stochastic_sampling_without_changing_request_contract() {
        let mut request = Sq8ServingRequest::greedy("req-1", vec![1], 1);
        request.sampling.temperature = 0.6;
        request.sampling.top_p = 0.95;
        request.validate().unwrap();
        let err = validate_p8b_greedy_execution(request.sampling).unwrap_err();
        assert_eq!(err.kind, Sq8ServingErrorKind::InvalidConfiguration);
    }

    #[test]
    fn serving_cancellation_is_shared_and_monotonic() {
        let first = Sq8CancellationToken::new();
        let second = first.clone();
        assert!(!first.is_cancelled());
        second.cancel();
        assert!(first.is_cancelled());
        first.cancel();
        assert!(second.is_cancelled());
    }

    #[test]
    fn serving_cache_geometry_is_4096_tokens_with_identity_block_table() {
        let shape = qwen3_14b_sq8_serving_cache_shape();
        shape.validate().unwrap();
        assert_eq!(shape.block_size, 16);
        assert_eq!(shape.cache_blocks, 256);
        assert_eq!(shape.physical_tokens().unwrap(), 4096);
        let table = qwen3_14b_sq8_serving_block_table().unwrap();
        assert_eq!(table.len(), 256);
        assert_eq!(table.first(), Some(&0));
        assert_eq!(table.last(), Some(&255));
    }

    #[test]
    fn serving_cache_byte_count_matches_frozen_f32_layout() {
        assert_eq!(
            qwen3_14b_sq8_serving_kv_cache_bytes_per_layer().unwrap(),
            33_554_432
        );
        assert_eq!(
            qwen3_14b_sq8_serving_total_kv_cache_bytes(QWEN3_14B_SQ8_STACK_LAYERS).unwrap(),
            1_342_177_280
        );
    }

    #[test]
    fn serving_active_metadata_tracks_prompt_and_generated_cache_semantics() {
        let request = Sq8ServingRequest::greedy("req-1", vec![1, 2, 3], 2);
        let mut active = ActiveServingRequest::new(request, Sq8CancellationToken::new());
        assert_eq!(active.expected_cache_len().unwrap(), 0);
        active.prompt_tokens_processed = 3;
        assert_eq!(active.expected_cache_len().unwrap(), 3);
        assert_eq!(active.terminal_reason(10), None);

        active.generated_tokens = 1;
        active.last_generated_token = Some(10);
        assert_eq!(active.expected_cache_len().unwrap(), 3);
        assert_eq!(active.terminal_reason(11), Some(Sq8FinishReason::Length));

        active.request.eos_token_ids = vec![11];
        assert_eq!(active.terminal_reason(11), Some(Sq8FinishReason::Stop));
    }

    #[test]
    fn serving_terminal_policy_stops_on_first_eos_output() {
        let request = Sq8ServingRequest::greedy("req-1", vec![1, 2, 3], 8);
        let mut active = ActiveServingRequest::new(request, Sq8CancellationToken::new());
        active.prompt_tokens_processed = active.request.prompt_token_ids.len();

        assert_eq!(
            active.terminal_reason(QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS[0]),
            Some(Sq8FinishReason::Stop)
        );
        assert_eq!(active.terminal_reason(42), None);
    }

    #[test]
    fn serving_terminal_policy_stops_during_decode_and_caps_non_eos() {
        let request = Sq8ServingRequest::greedy("req-1", vec![1, 2, 3], 8);
        let mut active = ActiveServingRequest::new(request, Sq8CancellationToken::new());
        active.prompt_tokens_processed = active.request.prompt_token_ids.len();
        active.generated_tokens = 3;

        assert_eq!(
            active.terminal_reason(QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS[1]),
            Some(Sq8FinishReason::Stop)
        );
        assert_eq!(active.terminal_reason(42), None);

        active.generated_tokens = active.request.max_new_tokens - 1;
        assert_eq!(active.terminal_reason(42), Some(Sq8FinishReason::Length));
        assert_eq!(
            active.terminal_reason(QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS[0]),
            Some(Sq8FinishReason::Stop)
        );
    }

    #[test]
    fn serving_scheduler_and_active_metadata_share_contiguous_positions() {
        let request = Sq8ServingRequest::greedy("req-1", vec![1, 2, 3], 2);
        let mut active = ActiveServingRequest::new(request.clone(), Sq8CancellationToken::new());
        let mut scheduler = SchedulerState::with_block_size(
            QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS as u32,
            QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS as u32,
        );
        let allocation = scheduler
            .activate_single_request_with_all_blocks(Request {
                id: SERVING_INTERNAL_REQUEST_ID,
                prompt_tokens: request.prompt_token_ids.len(),
                max_new_tokens: request.max_new_tokens,
            })
            .unwrap();
        assert_eq!(
            allocation.allocation.blocks,
            qwen3_14b_sq8_serving_block_table().unwrap()
        );

        for expected in 1..=request.prompt_token_ids.len() {
            assert_eq!(
                scheduler
                    .advance_prefill_token(SERVING_INTERNAL_REQUEST_ID)
                    .unwrap(),
                expected
            );
            active.prompt_tokens_processed = expected;
            assert_eq!(active.expected_cache_len().unwrap(), expected);
        }
        scheduler
            .record_prefill_generated_token(SERVING_INTERNAL_REQUEST_ID)
            .unwrap();
        active.generated_tokens = 1;
        active.last_generated_token = Some(10);
        let ready = scheduler.ready_decode_batch(1).unwrap();
        assert_eq!(ready[0].cache_position, 3);
        assert_eq!(ready[0].next_cache_len, 4);

        scheduler.advance_decode_batch(&ready).unwrap();
        active.generated_tokens = 2;
        active.last_generated_token = Some(11);
        assert_eq!(active.expected_cache_len().unwrap(), 4);
        assert_eq!(scheduler.release_request(SERVING_INTERNAL_REQUEST_ID), 256);
        validate_scheduler_baseline(&scheduler).unwrap();
    }
}
