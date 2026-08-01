// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Synchronous active1/waiting0 SQ8 serving session contracts.
//!
//! This module is separate from `sq8_generation_runtime`: the P7 fixed request and its audited
//! result schemas remain unchanged while serving gains variable prompt lengths and reusable state.

use crate::decoder::{
    PagedDecodeShape, PagedDecodeState, PagedKvCacheReadback,
    SQ8_PAGED_DECODE_SPLIT_MULTITILE_EVALUATION_ENV,
};
use crate::inference_api::ReasoningUsage;
pub use crate::inference_api::{
    CancellationToken as Sq8CancellationToken, FinishReason as Sq8FinishReason,
    InferenceError as Sq8ServingError, InferenceErrorKind as Sq8ServingErrorKind,
    InferenceRequest as Sq8ServingRequest, ReleaseOutcome as Sq8ReleaseOutcome,
    ReleaseSummary as Sq8ReleaseSummary, SamplingParams as Sq8SamplingParams,
};
use crate::loader::{read_named_passthrough_f32, verify_named_passthrough_payload};
use crate::model_config::load_model_config_from_package;
use crate::reasoning::{ReasoningDialect, ReasoningPhase, ReasoningState};
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
use crate::sq8_layer_runtime::{
    QWEN3_14B_SQ8_PREFILL_CHUNK_TOKEN_OPTIONS, QWEN3_14B_SQ8_PREFILL_CHUNK_TOKENS,
    Qwen3Sq8LayerNormValues, Sq8LayerExecutionProfile, Sq8LayerRuntimeTrace,
    is_qwen3_14b_sq8_prefill_chunk_tokens, validate_norm_values,
};
use crate::sq8_model_head_runtime::{
    QWEN3_14B_VOCAB_SIZE, Qwen3Sq8ModelHeadRuntime, Sq8ModelHeadDeviceIdentity,
    Sq8ModelHeadServingSource, validate_qwen3_14b_sq8_r9700_device_info,
};
use crate::sq8_sampling::{Sq8CpuSampler, Sq8SamplingProposal};
use crate::sq8_stack_runtime::{
    QWEN3_14B_SQ8_STACK_LAYERS, Qwen3Sq8PagedDecodeRuntime, Qwen3Sq8StackRuntime,
    Sq8PagedStackExecutionReport, Sq8PagedStackPhase, Sq8ServingChunkExecutionReport,
};
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use ullm_runtime_sys::{DeviceInfo, RuntimeBuffer, RuntimeContext, RuntimeStream};

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
const SQ8_SEQUENTIAL_M1_PREFILL_IMPLEMENTATION: &str = "sq8.sequential-m1.v1";
const SQ8_FIXED_M8_PREFILL_IMPLEMENTATION: &str = "sq8.fixed-m8-cached-prefix.v1";
const SQ8_FIXED_M32_PREFILL_IMPLEMENTATION: &str = "sq8.fixed-m32-cached-prefix.v1";
const SQ8_FIXED_M128_PREFILL_IMPLEMENTATION: &str = "sq8.fixed-m128-cached-prefix.v1";
const SQ8_ADAPTIVE_PREFILL_IMPLEMENTATION: &str = "sq8.adaptive-measured-width.v1";
/// Test-only opt-in for the existing split API.  Absence preserves the
/// ordinary direct paged-decode dispatch exactly.
pub const QWEN3_14B_SQ8_PAGED_DECODE_SPLIT_EXPERIMENT_TILE_ENV: &str =
    "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE";
/// Test-only bypass for the multi-tile containment fallback. It is only read
/// when the source-tile experiment itself has already been explicitly
/// enabled, and must be exactly `1` to take effect.
pub const QWEN3_14B_SQ8_PAGED_DECODE_SPLIT_MULTITILE_EVALUATION_ENV: &str =
    SQ8_PAGED_DECODE_SPLIT_MULTITILE_EVALUATION_ENV;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8ServingPrefillMode {
    /// Selects a measured fixed resident width from the prompt length at the
    /// beginning of each Ready-to-Prefilling transition. The selected width
    /// is then kept fixed for the entire request.
    Adaptive,
    SequentialM1,
    FixedM8Chunks,
    FixedM32Chunks,
    FixedM128Chunks,
    /// A scheduler-selected fixed resident width. Construction through
    /// [`Self::fixed_chunk_tokens`] validates the scheduling contract; the
    /// lower CK/layer/stack execution contract remains separately measured.
    FixedChunkTokens(NonZeroUsize),
}

impl Sq8ServingPrefillMode {
    /// Selects a fixed-width cached-prefix schedule without padding or
    /// synthetic tokens. Runtime execution of a new width is admitted
    /// separately, after the CK/layer/stack contract has measured it.
    pub fn fixed_chunk_tokens(chunk_tokens: usize) -> Result<Self, Sq8ServingError> {
        Self::validate_fixed_chunk_tokens(chunk_tokens)
            .map_err(Sq8ServingError::invalid_configuration)?;
        Ok(match chunk_tokens {
            QWEN3_14B_SQ8_PREFILL_CHUNK_TOKENS => Self::FixedM8Chunks,
            32 => Self::FixedM32Chunks,
            128 => Self::FixedM128Chunks,
            _ => Self::FixedChunkTokens(
                NonZeroUsize::new(chunk_tokens)
                    .expect("validated serving fixed chunk width is nonzero"),
            ),
        })
    }

    /// Returns the requested fixed width, or `None` for the sequential M=1
    /// schedule. This is the scheduler contract; callers that load a model
    /// still receive the lower-runtime admission check.
    pub fn chunk_tokens(self) -> Option<usize> {
        match self {
            Self::Adaptive | Self::SequentialM1 => None,
            Self::FixedM8Chunks => Some(QWEN3_14B_SQ8_PREFILL_CHUNK_TOKENS),
            Self::FixedM32Chunks => Some(32),
            Self::FixedM128Chunks => Some(128),
            Self::FixedChunkTokens(chunk_tokens) => Some(chunk_tokens.get()),
        }
    }

    fn validate_fixed_chunk_tokens(chunk_tokens: usize) -> Result<(), String> {
        // The worker must reserve at least one decode position, so a 4096-row
        // prefill can never be an all-real serving unit. Keep the direct
        // lower-runtime admission separate, but reject this unusable serving
        // override instead of silently falling back to M=1 for every request.
        if !(2..=2048).contains(&chunk_tokens)
            || !chunk_tokens.is_power_of_two()
        {
            return Err(format!(
                "SQ8 serving fixed prefill chunk width must be a power of two in 2..=2048, got M={chunk_tokens}",
            ));
        }
        Ok(())
    }

    fn validate_scheduler_contract(self) -> Result<(), String> {
        if let Some(chunk_tokens) = self.chunk_tokens() {
            Self::validate_fixed_chunk_tokens(chunk_tokens)?;
        }
        Ok(())
    }

    /// Scheduler planning and resident execution have deliberately separate
    /// admissions. Do not allocate a candidate model only to discover that
    /// its unmeasured CK/layer/stack width cannot execute.
    fn validate_runtime_contract(self) -> Result<(), String> {
        self.validate_scheduler_contract()?;
        let validate_width = |chunk_tokens| {
            if !is_qwen3_14b_sq8_prefill_chunk_tokens(chunk_tokens) {
                return Err(format!(
                    "SQ8 serving requested fixed prefill M={chunk_tokens}, but the current CK/layer/stack runtime admits only measured widths {:?}; wide-M scheduling is available but execution requires the lower-layer wide-M contract",
                    QWEN3_14B_SQ8_PREFILL_CHUNK_TOKEN_OPTIONS
                ));
            }
            Ok(())
        };
        match self {
            Self::Adaptive => {
                for chunk_tokens in [128, 512, 1024, 2048] {
                    validate_width(chunk_tokens)?;
                }
            }
            _ => {
                if let Some(chunk_tokens) = self.chunk_tokens() {
                    validate_width(chunk_tokens)?;
                }
            }
        }
        Ok(())
    }

    /// Resolves the width that owns the resident stack and prompt-buffer
    /// allocation before the first request. Adaptive mode deliberately starts
    /// at M=128; it widens only after a long request has been validated.
    fn initial_resident_mode(self) -> Self {
        match self {
            Self::Adaptive => Self::FixedM128Chunks,
            _ => self,
        }
    }

    /// Selects the empirical winner for a valid Qwen3-14B SQ8 prompt length.
    ///
    /// The measured grid makes M=256 deliberately ineligible: at the nearest
    /// measured long prompt it loses to M=128. M=4096 is likewise excluded:
    /// N=4095 has no legal all-real 4096-row unit. The boundaries are the
    /// measured N columns, so this policy does not infer that merely taking
    /// the largest power of two below N is optimal.
    pub fn selected_for_prompt_tokens(self, prompt_tokens: usize) -> Result<Self, String> {
        if !(1..=QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS).contains(&prompt_tokens) {
            return Err(format!(
                "SQ8 adaptive prefill selection requires N in 1..={}, got N={prompt_tokens}",
                QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS
            ));
        }
        Ok(match self {
            Self::Adaptive => match prompt_tokens {
                1..=511 => Self::FixedM128Chunks,
                512..=1023 => Self::fixed_chunk_tokens(512)
                    .expect("adaptive M=512 is a static valid scheduler width"),
                1024..=2047 => Self::fixed_chunk_tokens(1024)
                    .expect("adaptive M=1024 is a static valid scheduler width"),
                2048..=QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS => Self::fixed_chunk_tokens(2048)
                    .expect("adaptive M=2048 is a static valid scheduler width"),
                _ => unreachable!("adaptive prompt length was validated above"),
            },
            fixed => fixed,
        })
    }

    fn resident_stack_width(self) -> usize {
        self.initial_resident_mode()
            .chunk_tokens()
            .unwrap_or(QWEN3_14B_SQ8_PREFILL_CHUNK_TOKENS)
    }

    fn execution_width(self) -> usize {
        self.initial_resident_mode().chunk_tokens().unwrap_or(1)
    }

    fn implementation_id(self) -> String {
        match self {
            Self::Adaptive => SQ8_ADAPTIVE_PREFILL_IMPLEMENTATION.to_string(),
            Self::SequentialM1 => SQ8_SEQUENTIAL_M1_PREFILL_IMPLEMENTATION.to_string(),
            Self::FixedM8Chunks => SQ8_FIXED_M8_PREFILL_IMPLEMENTATION.to_string(),
            Self::FixedM32Chunks => SQ8_FIXED_M32_PREFILL_IMPLEMENTATION.to_string(),
            Self::FixedM128Chunks => SQ8_FIXED_M128_PREFILL_IMPLEMENTATION.to_string(),
            Self::FixedChunkTokens(chunk_tokens) => {
                format!("sq8.fixed-m{}-cached-prefix.v1", chunk_tokens.get())
            }
        }
    }

    fn uses_chunks(self) -> bool {
        matches!(self, Self::Adaptive) || self.chunk_tokens().is_some()
    }
}

/// The production-compatible serving default. Explicit callers can pin a
/// fixed resident width with [`Sq8ServingPrefillMode::fixed_chunk_tokens`].
pub const QWEN3_14B_SQ8_SERVING_DEFAULT_PREFILL_MODE: Sq8ServingPrefillMode =
    Sq8ServingPrefillMode::Adaptive;

impl Sq8ServingRequest {
    pub fn new(
        request_id: impl Into<String>,
        prompt_token_ids: Vec<usize>,
        max_new_tokens: usize,
        sampling: Sq8SamplingParams,
    ) -> Self {
        Self::new_with_eos(
            request_id,
            prompt_token_ids,
            max_new_tokens,
            QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS.to_vec(),
            sampling,
        )
    }

    pub fn greedy(
        request_id: impl Into<String>,
        prompt_token_ids: Vec<usize>,
        max_new_tokens: usize,
    ) -> Self {
        Self::new(
            request_id,
            prompt_token_ids,
            max_new_tokens,
            Sq8SamplingParams::greedy(0),
        )
    }

    /// Constructs the fixed deep-boundary diagnostic request.
    #[doc(hidden)]
    pub fn greedy_ignore_eos_for_testing(
        request_id: impl Into<String>,
        prompt_token_ids: Vec<usize>,
        max_new_tokens: usize,
    ) -> Self {
        Self::greedy(request_id, prompt_token_ids, max_new_tokens).ignore_eos_for_testing()
    }

    pub fn validate(&self) -> Result<(), Sq8ServingError> {
        self.validate_for_worker(
            QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS,
            QWEN3_14B_SQ8_SERVING_MAX_NEW_TOKENS,
            QWEN3_14B_VOCAB_SIZE,
            &QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS,
            QWEN3_14B_SQ8_SERVING_TOP_K,
        )
    }
}

impl Sq8SamplingParams {
    pub const fn greedy(seed: i64) -> Self {
        Self::greedy_with_top_k(seed, QWEN3_14B_SQ8_SERVING_TOP_K)
    }

    pub fn validate(&self) -> Result<(), Sq8ServingError> {
        self.validate_with_top_k(QWEN3_14B_SQ8_SERVING_TOP_K)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum TokenPublication<T> {
    Published(T),
    Cancelled,
}

fn linearize_token_publication<T, P, C>(
    cancel: &Sq8CancellationToken,
    publish: P,
    commit: C,
) -> Result<TokenPublication<T>, String>
where
    P: FnOnce() -> Result<(), String>,
    C: FnOnce() -> Result<T, String>,
{
    let _publication = cancel.publication_guard()?;
    if cancel.is_cancelled() {
        return Ok(TokenPublication::Cancelled);
    }
    publish().map_err(|err| format!("serving token publisher failed before commit: {err}"))?;
    commit().map(TokenPublication::Published)
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8PreparedToken {
    pub token_id: usize,
    pub generated_index: usize,
    pub cache_len: usize,
    pub terminal_reason: Option<Sq8FinishReason>,
    nonce: u64,
}

#[cfg(test)]
impl Sq8PreparedToken {
    pub(crate) fn for_worker_test(
        token_id: usize,
        generated_index: usize,
        cache_len: usize,
        terminal_reason: Option<Sq8FinishReason>,
    ) -> Self {
        Self {
            token_id,
            generated_index,
            cache_len,
            terminal_reason,
            nonce: generated_index as u64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sq8PreparedAdvance {
    PromptProgress {
        prompt_tokens_processed: usize,
        cache_len: usize,
        execution_width: usize,
    },
    Token(Sq8PreparedToken),
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

/// Host readback from every layer of one actual M=1 serving decode. This is
/// intentionally test-only instrumentation: each layer readback synchronizes
/// the stream and is never part of the ordinary serving or timing paths.
#[derive(Debug, Clone, PartialEq)]
#[doc(hidden)]
pub struct Sq8ServingDecodeLayerTraceCapture {
    pub input_token_id: usize,
    pub position: usize,
    pub profile: Sq8LayerExecutionProfile,
    pub layers: Vec<Sq8LayerRuntimeTrace>,
}

/// One host-side record from the isolated v0.2 teacher-forced capture path.
/// `layers`, when requested, holds one post-layer residual vector per
/// transformer layer. This is intentionally separate from normal serving:
/// it can force an externally supplied next token and it resets after the
/// final observed forward rather than publishing a sampled response.
#[derive(Debug, Clone, PartialEq)]
#[doc(hidden)]
pub struct Sq8ServingTeacherForcedCapture {
    pub input_token_id: usize,
    pub position: usize,
    pub top1: Sq8GenerationTopLogit,
    pub final_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub layers: Option<Vec<Vec<f32>>>,
}

/// Logical, written-only K/V state for every layer of one active serving
/// request.  This exists solely for differential diagnostics; it deliberately
/// exposes neither physical cache capacity nor an execution selector.
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8ServingKvCachePrefixCapture {
    pub cache_len: usize,
    pub layer_caches: Vec<PagedKvCacheReadback>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8ServingRuntimeStatus {
    Ready,
    Prefilling,
    Decoding,
    TokenPrepared,
    Finishing,
    Cancelling,
    Resetting,
    Failed,
}

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
    pub prefill_mode: Sq8ServingPrefillMode,
    pub prefill_chunk_tokens: usize,
    pub prefill_implementation: String,
    pub prompt_execution_width: usize,
    pub paged_decode_split_source_tile: Option<usize>,
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
            || self.prefill_chunk_tokens != self.prefill_mode.resident_stack_width()
            || self.prefill_implementation != self.prefill_mode.implementation_id()
            || self.prompt_execution_width != self.prefill_mode.execution_width()
            || self
                .paged_decode_split_source_tile
                .is_some_and(|tile| !matches!(tile, 20 | 128 | 256 | 512))
        {
            return Err(Sq8ServingError::invalid_configuration(
                "serving resident geometry/load report mismatch",
            ));
        }
        Ok(())
    }
}

fn parse_paged_decode_split_source_tile(value: Option<&str>) -> Result<Option<usize>, String> {
    match value {
        None => Ok(None),
        Some("20") => Ok(Some(20)),
        Some("128") => Ok(Some(128)),
        Some("256") => Ok(Some(256)),
        Some("512") => Ok(Some(512)),
        Some(other) => Err(format!(
            "{QWEN3_14B_SQ8_PAGED_DECODE_SPLIT_EXPERIMENT_TILE_ENV} must be exactly 20, 128, 256, or 512, got {other:?}"
        )),
    }
}

fn read_paged_decode_split_source_tile() -> Result<Option<usize>, String> {
    match std::env::var(QWEN3_14B_SQ8_PAGED_DECODE_SPLIT_EXPERIMENT_TILE_ENV) {
        Ok(value) => parse_paged_decode_split_source_tile(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "{QWEN3_14B_SQ8_PAGED_DECODE_SPLIT_EXPERIMENT_TILE_ENV} must be UTF-8"
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8ServingSnapshot {
    pub status: Sq8ServingRuntimeStatus,
    pub active_request_id: Option<String>,
    pub prompt_tokens: usize,
    pub prompt_tokens_processed: usize,
    pub generated_tokens: usize,
    pub sampling_draws: u64,
    pub token_prepared: bool,
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
    sampled_tokens: usize,
    // This counter exists only for the isolated numerical-gate state machine.
    // It records materialized oracle tokens that deliberately bypass sampling
    // and must therefore remain distinct from reasoning forced-end tokens.
    teacher_forced_tokens: usize,
    last_generated_token: Option<usize>,
    finish_reason: Option<Sq8FinishReason>,
    sampler: Sq8CpuSampler,
    reasoning: Option<ReasoningState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingReleaseAccounting {
    request_id: String,
    outcome: Sq8ReleaseOutcome,
    prompt_tokens: usize,
    generated_tokens: usize,
    reasoning_usage: Option<ReasoningUsage>,
}

impl PendingReleaseAccounting {
    fn complete_after_reset(self) -> Sq8ReleaseSummary {
        Sq8ReleaseSummary {
            request_id: self.request_id,
            outcome: self.outcome,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            reasoning_usage: self.reasoning_usage,
            reset_complete: true,
        }
    }
}

impl ActiveServingRequest {
    #[cfg(test)]
    fn new(request: Sq8ServingRequest, cancel: Sq8CancellationToken) -> Self {
        Self::new_with_reasoning_dialect(request, cancel, None)
            .expect("a request without reasoning does not require a loaded dialect")
    }

    fn new_with_reasoning_dialect(
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
        reasoning_dialect: Option<&ReasoningDialect>,
    ) -> Result<Self, String> {
        let reasoning = match (request.reasoning.as_ref(), reasoning_dialect) {
            (Some(execution), Some(dialect)) => {
                if [
                    dialect.start_sequence.len(),
                    dialect.end_sequence.len(),
                    dialect.forced_end_sequence.len(),
                    execution.end_sequence.len(),
                    execution.forced_end_sequence.len(),
                ]
                .into_iter()
                .any(|length| length != 1)
                {
                    return Err(
                        "SQ8 serving v2 reasoning token sequences must contain exactly one token"
                            .into(),
                    );
                }
                if execution.dialect_id != dialect.identity
                    || execution.end_sequence != dialect.end_sequence
                    || execution.forced_end_sequence != dialect.forced_end_sequence
                    || execution.reserved_answer_tokens != dialect.reserved_answer_tokens
                {
                    return Err("SQ8 reasoning execution is not bound to the loaded dialect".into());
                }
                Some(
                    ReasoningState::new(
                        dialect.clone(),
                        execution.enabled,
                        execution.budget_tokens,
                        QWEN3_14B_VOCAB_SIZE,
                    )
                    .map_err(|error| format!("SQ8 reasoning execution is invalid: {error:?}"))?,
                )
            }
            (Some(_), None) => {
                return Err("SQ8 reasoning request has no loaded dialect".into());
            }
            (None, Some(_)) => {
                return Err("SQ8 loaded reasoning dialect requires an execution contract".into());
            }
            (None, None) => None,
        };
        let sampler = Sq8CpuSampler::new(request.sampling.seed);
        Ok(Self {
            request,
            cancel,
            prompt_tokens_processed: 0,
            generated_tokens: 0,
            sampled_tokens: 0,
            teacher_forced_tokens: 0,
            last_generated_token: None,
            finish_reason: None,
            sampler,
            reasoning,
        })
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

    fn reasoning_usage(&self) -> Option<ReasoningUsage> {
        self.reasoning.as_ref().map(|reasoning| ReasoningUsage {
            reasoning_tokens: reasoning.reasoning_tokens,
            forced_end_tokens: reasoning.forced_end_tokens,
        })
    }

    fn snapshot_release_accounting(&self, outcome: Sq8ReleaseOutcome) -> PendingReleaseAccounting {
        PendingReleaseAccounting {
            request_id: self.request.request_id.clone(),
            outcome,
            prompt_tokens: self.request.prompt_token_ids.len(),
            generated_tokens: self.generated_tokens,
            reasoning_usage: self.reasoning_usage(),
        }
    }

    fn forced_reasoning_transition(&self) -> Result<Option<(usize, ReasoningState)>, String> {
        let Some(mut reasoning) = self.reasoning.clone() else {
            return Ok(None);
        };
        if reasoning.phase == ReasoningPhase::Reasoning {
            let execution = self.request.reasoning.as_ref().ok_or_else(|| {
                "SQ8 active reasoning state has no execution contract".to_string()
            })?;
            let remaining = self
                .request
                .max_new_tokens
                .checked_sub(self.generated_tokens)
                .ok_or_else(|| "SQ8 reasoning completion counter exceeds its limit".to_string())?;
            let close_and_answer = execution
                .forced_end_sequence
                .len()
                .checked_add(execution.reserved_answer_tokens)
                .ok_or_else(|| "SQ8 reasoning length reservation overflows".to_string())?;
            if remaining <= close_and_answer {
                reasoning.force_close().map_err(|error| {
                    format!("SQ8 reasoning answer-reservation close failed: {error:?}")
                })?;
            }
        }
        if reasoning.phase != ReasoningPhase::ForcingEndSequence {
            return Ok(None);
        }
        let token_id = reasoning.next_forced_token().ok_or_else(|| {
            "SQ8 reasoning forced-end sequence is exhausted before answer phase".to_string()
        })?;
        reasoning
            .accept_forced(token_id)
            .map_err(|error| format!("SQ8 reasoning forced transition failed: {error:?}"))?;
        Ok(Some((token_id, reasoning)))
    }

    fn sampled_reasoning_transition(
        &self,
        sampled_token_id: usize,
    ) -> Result<(usize, bool, Option<ReasoningState>), String> {
        if let Some((token_id, reasoning_after)) = self.forced_reasoning_transition()? {
            return Ok((token_id, true, Some(reasoning_after)));
        }
        let Some(mut reasoning_after) = self.reasoning.clone() else {
            return Ok((sampled_token_id, false, None));
        };
        if !self.request.test_only_ignores_eos()
            && self.request.eos_token_ids.contains(&sampled_token_id)
        {
            reasoning_after.on_eos();
            if reasoning_after.phase == ReasoningPhase::ForcingEndSequence {
                let token_id = reasoning_after
                    .next_forced_token()
                    .ok_or_else(|| "SQ8 reasoning EOS close has no forced token".to_string())?;
                reasoning_after.accept_forced(token_id).map_err(|error| {
                    format!("SQ8 reasoning EOS forced transition failed: {error:?}")
                })?;
                return Ok((token_id, true, Some(reasoning_after)));
            }
        } else {
            reasoning_after
                .accept_sampled(sampled_token_id)
                .map_err(|error| format!("SQ8 reasoning sampled transition failed: {error:?}"))?;
        }
        Ok((sampled_token_id, false, Some(reasoning_after)))
    }

    fn terminal_reason_after(
        &self,
        token_id: usize,
        reasoning_after: Option<&ReasoningState>,
    ) -> Option<Sq8FinishReason> {
        if reasoning_after.is_some_and(|reasoning| reasoning.phase == ReasoningPhase::Finished) {
            Some(Sq8FinishReason::Stop)
        } else if reasoning_after.is_none()
            && !self.request.test_only_ignores_eos()
            && self.request.eos_token_ids.contains(&token_id)
        {
            Some(Sq8FinishReason::Stop)
        } else if self.generated_tokens + 1 == self.request.max_new_tokens {
            Some(Sq8FinishReason::Length)
        } else {
            None
        }
    }

    #[cfg(test)]
    fn terminal_reason(&self, token_id: usize) -> Option<Sq8FinishReason> {
        self.terminal_reason_after(token_id, None)
    }
}

#[derive(Debug)]
enum GeneratedTokenCommit {
    Prefill,
    Decode(Vec<SchedulerDecodeRequest>),
}

#[derive(Debug)]
enum PendingTokenSource {
    Sampled(Sq8SamplingProposal),
    /// A token injected by the numerical-gate harness.  This is deliberately
    /// distinct from a reasoning close token: it must not affect the serving
    /// request's reasoning forced-end accounting.
    TeacherForced,
    /// A token selected by the reasoning state machine while closing a
    /// reasoning span.  Unlike a teacher-forced diagnostic token, this must
    /// advance `forced_end_tokens` exactly once.
    ReasoningForced,
}

#[derive(Debug)]
struct PendingServingToken {
    prepared: Sq8PreparedToken,
    source: PendingTokenSource,
    reasoning_after: Option<ReasoningState>,
    commit: GeneratedTokenCommit,
}

#[derive(Debug)]
struct PreparedOracleAdvance {
    advance: Sq8PreparedAdvance,
    capture: Option<Sq8ServingOracleCapture>,
}

#[derive(Debug)]
struct DecodeStepPlan {
    input_token_id: usize,
    expected_position: usize,
    ready: Vec<SchedulerDecodeRequest>,
}

#[derive(Debug)]
enum HeadPreparation {
    Prepared {
        proposal: Sq8SamplingProposal,
        capture: Option<Sq8ServingOracleCapture>,
    },
    CancellationObserved,
}

fn publish_prepared_token_transaction<F>(
    state: &mut Sq8ServingRuntimeStatus,
    pending_token: &mut Option<PendingServingToken>,
    active: &mut Option<ActiveServingRequest>,
    scheduler: &mut SchedulerState,
    cancel: &Sq8CancellationToken,
    prepared: &Sq8PreparedToken,
    publish: F,
) -> Result<Sq8ServingAdvance, String>
where
    F: FnOnce(&Sq8PreparedToken) -> Result<(), String>,
{
    let publication = linearize_token_publication(
        cancel,
        || publish(prepared),
        || commit_pending_token_state(pending_token, active, scheduler, state, prepared),
    );
    match publication {
        Ok(TokenPublication::Published(committed)) => Ok(committed),
        Ok(TokenPublication::Cancelled) => {
            *pending_token = None;
            *state = Sq8ServingRuntimeStatus::Cancelling;
            Ok(Sq8ServingAdvance::CancellationObserved)
        }
        Err(err) => {
            *state = Sq8ServingRuntimeStatus::Failed;
            Err(err)
        }
    }
}

fn commit_pending_token_state(
    pending_token: &mut Option<PendingServingToken>,
    active: &mut Option<ActiveServingRequest>,
    scheduler: &mut SchedulerState,
    state: &mut Sq8ServingRuntimeStatus,
    prepared: &Sq8PreparedToken,
) -> Result<Sq8ServingAdvance, String> {
    let pending = pending_token
        .take()
        .ok_or_else(|| "serving token commit has no pending token".to_string())?;
    if &pending.prepared != prepared {
        return Err("serving token commit handle changed after publication".into());
    }
    match &pending.source {
        PendingTokenSource::Sampled(proposal)
            if proposal.sampled().token_id == prepared.token_id => {}
        PendingTokenSource::Sampled(_) => {
            return Err("serving sampling proposal changed before commit".into());
        }
        PendingTokenSource::TeacherForced | PendingTokenSource::ReasoningForced => {}
    }
    let next_generated_tokens = prepared
        .generated_index
        .checked_add(1)
        .ok_or_else(|| "serving generated token counter overflows".to_string())?;
    let active_before = active
        .as_ref()
        .ok_or_else(|| "serving token commit has no active request".to_string())?;
    if active_before.reasoning.is_some() != pending.reasoning_after.is_some() {
        return Err("serving pending reasoning state changed before commit".into());
    }
    let forced_before = active_before
        .reasoning
        .as_ref()
        .map_or(0, |reasoning| reasoning.forced_end_tokens);
    let forced_after = pending
        .reasoning_after
        .as_ref()
        .map_or(0, |reasoning| reasoning.forced_end_tokens);
    match &pending.source {
        PendingTokenSource::Sampled(_) if forced_after != forced_before => {
            return Err("serving sampled token changed forced-end accounting".into());
        }
        PendingTokenSource::ReasoningForced
            if forced_after
                != forced_before
                    .checked_add(1)
                    .ok_or_else(|| "serving forced-end counter overflows".to_string())? =>
        {
            return Err("serving forced token did not advance forced-end accounting".into());
        }
        _ => {}
    }
    match pending.commit {
        GeneratedTokenCommit::Prefill => {
            scheduler.record_prefill_generated_token(SERVING_INTERNAL_REQUEST_ID)?
        }
        GeneratedTokenCommit::Decode(ready) => scheduler.advance_decode_batch(&ready)?,
    }
    let active = active
        .as_mut()
        .ok_or_else(|| "serving token commit has no active request".to_string())?;
    match pending.source {
        PendingTokenSource::Sampled(proposal) => {
            let sampled = proposal.sampled();
            let committed = active.sampler.commit(proposal)?;
            if committed.token_id != prepared.token_id
                || committed.logit.to_bits() != sampled.logit.to_bits()
            {
                return Err("serving sampler commit did not match prepared token".into());
            }
            active.sampled_tokens = active
                .sampled_tokens
                .checked_add(1)
                .ok_or_else(|| "serving sampled token counter overflows".to_string())?;
        }
        PendingTokenSource::TeacherForced => {
            active.teacher_forced_tokens = active
                .teacher_forced_tokens
                .checked_add(1)
                .ok_or_else(|| "serving teacher-forced token counter overflows".to_string())?;
        }
        PendingTokenSource::ReasoningForced => {}
    }
    active.generated_tokens = next_generated_tokens;
    active.last_generated_token = Some(prepared.token_id);
    active.finish_reason = prepared.terminal_reason;
    active.reasoning = pending.reasoning_after;
    validate_active_sampling_progress(active)?;
    *state = if prepared.terminal_reason.is_some() {
        Sq8ServingRuntimeStatus::Finishing
    } else {
        Sq8ServingRuntimeStatus::Decoding
    };
    Ok(Sq8ServingAdvance::Token {
        token_id: prepared.token_id,
        generated_index: prepared.generated_index,
        cache_len: prepared.cache_len,
        terminal_reason: prepared.terminal_reason,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sq8PrefillUnit {
    /// Logical prompt cursor before this unit. Scheduler and request metadata advance from here.
    logical_start_position: usize,
    /// First real token supplied to the fixed-width resident stack.
    execution_start_position: usize,
    /// Resident-stack width used by this execution. It is always 1 or the selected fixed width.
    execution_width: usize,
    /// Number of newly committed prompt tokens. An overlapping tail commits fewer tokens than it
    /// executes because its leading real tokens are deliberately recomputed.
    committed_tokens: usize,
    is_final: bool,
}

impl Sq8PrefillUnit {
    fn rewinds_cache(self) -> bool {
        self.execution_start_position < self.logical_start_position
    }

    fn execution_end(self) -> Result<usize, String> {
        self.execution_start_position
            .checked_add(self.execution_width)
            .ok_or_else(|| "serving prefill execution range overflows".to_string())
    }

    fn logical_end(self) -> Result<usize, String> {
        self.logical_start_position
            .checked_add(self.committed_tokens)
            .ok_or_else(|| "serving prefill logical range overflows".to_string())
    }
}

#[cfg(test)]
fn plan_prefill_units(
    prompt_tokens: usize,
    mode: Sq8ServingPrefillMode,
) -> Result<Vec<Sq8PrefillUnit>, String> {
    let mode = mode.selected_for_prompt_tokens(prompt_tokens)?;
    mode.validate_scheduler_contract()?;
    if prompt_tokens == 0 || prompt_tokens > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS {
        return Err(format!(
            "serving prefill planner prompt length must be in 1..={}, got {prompt_tokens}",
            QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS
        ));
    }
    let mut units = Vec::with_capacity(prompt_tokens);
    let mut logical_start_position = 0_usize;
    while logical_start_position < prompt_tokens {
        let unit = plan_next_prefill_unit(logical_start_position, prompt_tokens, mode)?;
        if unit.logical_start_position != logical_start_position {
            return Err("serving prefill planner returned a stale logical cursor".into());
        }
        logical_start_position = unit.logical_end()?;
        units.push(unit);
    }
    Ok(units)
}

fn plan_next_prefill_unit(
    logical_start_position: usize,
    prompt_tokens: usize,
    mode: Sq8ServingPrefillMode,
) -> Result<Sq8PrefillUnit, String> {
    let mode = mode.selected_for_prompt_tokens(prompt_tokens)?;
    mode.validate_scheduler_contract()?;
    if prompt_tokens == 0
        || prompt_tokens > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS
        || logical_start_position >= prompt_tokens
    {
        return Err(format!(
            "serving prefill planner position {logical_start_position} is invalid for prompt length {prompt_tokens}"
        ));
    }
    let remaining = prompt_tokens - logical_start_position;
    let (execution_start_position, execution_width, committed_tokens) = match mode.chunk_tokens() {
        Some(chunk_tokens) if remaining >= chunk_tokens => {
            (logical_start_position, chunk_tokens, chunk_tokens)
        }
        // The fixed resident stack cannot accept a ragged M. Rather than pad (which would
        // need a separate masking proof), replay a suffix made entirely of real tokens. The
        // cache cursor is rewound only across the overlapping prefix, then the M-wide chunk
        // rewrites it and advances through the prompt boundary.
        Some(chunk_tokens) if logical_start_position >= chunk_tokens => {
            let overlap_tokens = chunk_tokens
                .checked_sub(remaining)
                .ok_or_else(|| "serving prefill tail overlap underflows".to_string())?;
            let execution_start_position = logical_start_position
                .checked_sub(overlap_tokens)
                .ok_or_else(|| {
                    "serving prefill tail overlap exceeds processed prompt".to_string()
                })?;
            (execution_start_position, chunk_tokens, remaining)
        }
        // A prompt shorter than the first fixed chunk has no real-token prefix to overlap.
        // Keep the audited M=1 seed path rather than introduce unmasked padding.
        _ => (logical_start_position, 1, 1),
    };
    let unit = Sq8PrefillUnit {
        logical_start_position,
        execution_start_position,
        execution_width,
        committed_tokens,
        is_final: logical_start_position
            .checked_add(committed_tokens)
            .ok_or_else(|| "serving prefill planner position overflows".to_string())?
            == prompt_tokens,
    };
    if unit.execution_end()? != unit.logical_end()? {
        return Err("serving prefill unit does not end at its logical prompt cursor".into());
    }
    if unit.execution_width == 0 || unit.committed_tokens == 0 {
        return Err("serving prefill unit has zero execution or commit width".into());
    }
    Ok(unit)
}

/// Owns one resident Qwen3-14B SQ8 model and one reusable active1/waiting0 session.
#[derive(Debug)]
pub struct Qwen3Sq8ServingSession {
    /// User-selected policy. For adaptive serving this is distinct from the
    /// fixed mode currently represented by `load_report` and resident buffers.
    prefill_policy: Sq8ServingPrefillMode,
    load_report: Sq8ServingLoadReport,
    stack: Qwen3Sq8StackRuntime,
    decode: Qwen3Sq8PagedDecodeRuntime,
    caches: Box<[PagedDecodeState; QWEN3_14B_SQ8_STACK_LAYERS]>,
    embedding: Qwen3Sq8EmbeddingRuntime,
    prompt_chunk_hidden: RuntimeBuffer,
    head: Qwen3Sq8ModelHeadRuntime,
    scheduler: SchedulerState,
    active: Option<ActiveServingRequest>,
    pending_token: Option<PendingServingToken>,
    next_prepared_nonce: u64,
    state: Sq8ServingRuntimeStatus,
    failure_reason: Option<String>,
    handwritten_wmma_prototype_enabled: bool,
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
        Self::load_with_prefill_mode(
            context,
            stream,
            artifact,
            package_path,
            norms,
            upload_chunk_bytes,
            QWEN3_14B_SQ8_SERVING_DEFAULT_PREFILL_MODE,
        )
    }

    pub fn load_with_prefill_mode(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        artifact: &Sq8CanonicalArtifact,
        package_path: impl AsRef<Path>,
        norms: Vec<Qwen3Sq8LayerNormValues>,
        upload_chunk_bytes: usize,
        prefill_mode: Sq8ServingPrefillMode,
    ) -> Result<Self, Sq8ServingError> {
        prefill_mode
            .validate_runtime_contract()
            .map_err(Sq8ServingError::invalid_configuration)?;
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
        let resident_descriptor = load_model_config_from_package(package_path)
            .and_then(|loaded| loaded.resident_descriptor())
            .and_then(|descriptor| {
                descriptor.require_qwen3_14b_sq8_0()?;
                Ok(descriptor)
            })
            .map_err(|error| {
                Sq8ServingError::invalid_configuration(format!(
                    "Qwen3-14B SQ8_0 model config rejection: {error}"
                ))
            })?;
        let load_result = (|| {
            let device_info = context.device_info()?;
            validate_qwen3_14b_sq8_r9700_device_info(&device_info)?;
            let resident_prefill_mode = prefill_mode.initial_resident_mode();
            let resident_stack_width = resident_prefill_mode.resident_stack_width();
            let stack = Qwen3Sq8StackRuntime::load_for_resident_descriptor(
                context,
                stream,
                artifact,
                &resident_descriptor,
                resident_stack_width,
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

            let prompt_chunk_bytes =
                qwen3_14b_sq8_serving_prompt_chunk_bytes(resident_stack_width)?;
            let mut prompt_chunk_hidden = context
                .alloc_buffer(prompt_chunk_bytes)
                .map_err(|err| format!("failed to allocate serving prompt chunk: {err}"))?;
            prompt_chunk_hidden
                .zero(0, prompt_chunk_bytes, Some(&mut *stream))
                .map_err(|err| format!("failed to initialize serving prompt chunk: {err}"))?;
            stream.synchronize().map_err(|err| {
                format!("failed to synchronize serving prompt chunk setup: {err}")
            })?;

            let decode = Qwen3Sq8PagedDecodeRuntime::allocate(context)?;
            let cache_shape = qwen3_14b_sq8_serving_cache_shape();
            cache_shape.validate()?;
            let paged_decode_split_source_tile = read_paged_decode_split_source_tile()?;
            let block_table = qwen3_14b_sq8_serving_block_table().map_err(|err| err.to_string())?;
            let mut cache_values = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
            for layer_index in 0..QWEN3_14B_SQ8_STACK_LAYERS {
                let mut cache =
                    PagedDecodeState::new(context, stream, cache_shape, block_table.clone())
                        .map_err(|err| {
                            format!(
                                "failed to allocate serving layer {layer_index} KV cache: {err}"
                            )
                        })?;
                if let Some(source_tile) = paged_decode_split_source_tile {
                    cache
                        .enable_source_tiled_decode_experiment(context, source_tile)
                        .map_err(|err| {
                            format!(
                                "failed to configure serving layer {layer_index} paged decode split tile {source_tile}: {err}"
                            )
                        })?;
                }
                cache_values.push(cache);
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
                prefill_mode: resident_prefill_mode,
                prefill_chunk_tokens: resident_stack_width,
                prefill_implementation: resident_prefill_mode.implementation_id(),
                prompt_execution_width: resident_prefill_mode.execution_width(),
                paged_decode_split_source_tile,
                embedding_payload_sha256: embedding.load_report().payload.payload_sha256.clone(),
                final_norm_payload_sha256: head.final_norm_identity().payload_sha256.clone(),
                lm_head_payload_sha256: head.lm_head_identity().payload_sha256.clone(),
            };
            load_report.validate().map_err(|err| err.to_string())?;
            let session = Self {
                prefill_policy: prefill_mode,
                load_report,
                stack,
                decode,
                caches: Box::new(caches),
                embedding,
                prompt_chunk_hidden,
                head,
                scheduler: SchedulerState::with_block_size(
                    u32::try_from(QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS)
                        .map_err(|_| "serving cache block count does not fit u32".to_string())?,
                    u32::try_from(QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS)
                        .map_err(|_| "serving block size does not fit u32".to_string())?,
                ),
                active: None,
                pending_token: None,
                next_prepared_nonce: 0,
                state: Sq8ServingRuntimeStatus::Ready,
                failure_reason: None,
                handwritten_wmma_prototype_enabled: false,
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

    pub fn prefill_mode(&self) -> Sq8ServingPrefillMode {
        self.load_report.prefill_mode
    }

    /// Returns the configured selection policy rather than the fixed width
    /// currently resident for the last (or next) request.
    pub fn prefill_policy(&self) -> Sq8ServingPrefillMode {
        self.prefill_policy
    }

    /// Materializes the width selected by the configured policy without
    /// retaining a larger prompt workspace for a short request. The stack
    /// weights, embedding, head, decode workspace, and K/V caches remain
    /// resident; only the M-dependent prefill workspace and prompt buffer are
    /// exchanged at the Ready baseline.
    fn select_prefill_mode_for_request(
        &mut self,
        context: &mut RuntimeContext,
        prompt_tokens: usize,
        stream: &mut RuntimeStream,
    ) -> Result<(), String> {
        let selected = self.prefill_policy.selected_for_prompt_tokens(prompt_tokens)?;
        selected.validate_runtime_contract()?;
        if selected == self.load_report.prefill_mode {
            return Ok(());
        }
        self.validate_ready_baseline()?;
        let width = selected
            .chunk_tokens()
            .ok_or_else(|| "SQ8 adaptive selection did not resolve a fixed chunk width".to_string())?;
        let prompt_bytes = qwen3_14b_sq8_serving_prompt_chunk_bytes(width)?;
        let mut prompt_chunk_hidden = context
            .alloc_buffer(prompt_bytes)
            .map_err(|error| format!("failed to allocate adaptive serving prompt chunk: {error}"))?;
        prompt_chunk_hidden
            .zero(0, prompt_bytes, Some(&mut *stream))
            .map_err(|error| format!("failed to initialize adaptive serving prompt chunk: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("failed to synchronize adaptive serving prompt chunk: {error}"))?;

        self.stack
            .reconfigure_serving_prefill_width(context, stream, width)?;
        self.prompt_chunk_hidden = prompt_chunk_hidden;
        self.load_report.prefill_mode = selected;
        self.load_report.prefill_chunk_tokens = width;
        self.load_report.prefill_implementation = selected.implementation_id();
        self.load_report.prompt_execution_width = selected.execution_width();
        self.load_report.validate().map_err(|error| error.to_string())
    }

    /// Enables the isolated M=1 handwritten projection probe for one test
    /// session.  This is deliberately unavailable unless the dedicated Cargo
    /// feature was compiled, requires the pristine Ready state, and leaves the
    /// normal CK profile as the default for every caller.
    #[doc(hidden)]
    pub fn enable_handwritten_wmma_projection_prototype(&mut self) -> Result<(), Sq8ServingError> {
        if !cfg!(feature = "rocm-handwritten-projection-gfx1201") {
            return Err(Sq8ServingError::invalid_configuration(
                "SQ8 handwritten WMMA prototype requires Cargo feature rocm-handwritten-projection-gfx1201",
            ));
        }
        if self.state != Sq8ServingRuntimeStatus::Ready
            || self.active.is_some()
            || self.pending_token.is_some()
        {
            return Err(Sq8ServingError::invalid_state(
                "SQ8 handwritten WMMA prototype can only be enabled on a fresh Ready session",
            ));
        }
        self.handwritten_wmma_prototype_enabled = true;
        Ok(())
    }

    #[doc(hidden)]
    pub fn handwritten_wmma_projection_prototype_enabled(&self) -> bool {
        self.handwritten_wmma_prototype_enabled
    }

    pub fn snapshot(&self) -> Sq8ServingSnapshot {
        let (
            active_request_id,
            prompt_tokens,
            prompt_tokens_processed,
            generated_tokens,
            sampling_draws,
        ) = self
            .active
            .as_ref()
            .map(|active| {
                (
                    Some(active.request.request_id.clone()),
                    active.request.prompt_token_ids.len(),
                    active.prompt_tokens_processed,
                    active.generated_tokens,
                    active.sampler.draws(),
                )
            })
            .unwrap_or((None, 0, 0, 0, 0));
        Sq8ServingSnapshot {
            status: self.state,
            active_request_id,
            prompt_tokens,
            prompt_tokens_processed,
            generated_tokens,
            sampling_draws,
            token_prepared: self.pending_token.is_some(),
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

    /// Synchronizes and captures the logical written K/V prefix for every
    /// layer.  It is intended for a one-off differential after an actual
    /// decode step, before `finish_and_reset_synchronized` clears the caches.
    pub fn read_paged_kv_cache_prefix_synchronized(
        &self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ServingKvCachePrefixCapture, Sq8ServingError> {
        let expected_cache_len = self
            .caches
            .first()
            .map(PagedDecodeState::written_len)
            .ok_or_else(|| Sq8ServingError::fatal_runtime("serving KV cache array is empty"))?;
        if expected_cache_len == 0 {
            return Err(Sq8ServingError::invalid_state(
                "serving KV prefix capture requires written cache state",
            ));
        }
        let mut layer_caches = Vec::with_capacity(self.caches.len());
        for (layer_index, cache) in self.caches.iter().enumerate() {
            if cache.written_len() != expected_cache_len {
                return Err(Sq8ServingError::fatal_runtime(format!(
                    "serving KV prefix capture layer {layer_index} length {} differs from expected {expected_cache_len}",
                    cache.written_len()
                )));
            }
            let readback = cache
                .read_written_cache_prefix_to_host(stream)
                .map_err(|err| {
                    Sq8ServingError::fatal_runtime(format!(
                        "failed to read serving KV prefix for layer {layer_index}: {err}"
                    ))
                })?;
            layer_caches.push(readback);
        }
        Ok(Sq8ServingKvCachePrefixCapture {
            cache_len: expected_cache_len,
            layer_caches,
        })
    }

    pub fn start(
        &mut self,
        context: &mut RuntimeContext,
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
        stream: &mut RuntimeStream,
    ) -> Result<(), Sq8ServingError> {
        self.start_with_reasoning_dialect(context, request, cancel, None, stream)
    }

    pub fn start_with_reasoning_dialect(
        &mut self,
        context: &mut RuntimeContext,
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
        reasoning_dialect: Option<&ReasoningDialect>,
        stream: &mut RuntimeStream,
    ) -> Result<(), Sq8ServingError> {
        self.start_internal(context, request, cancel, reasoning_dialect, false, stream)
    }

    /// Starts the isolated teacher-forced capture state machine used only by
    /// the numerical-gate harness. The last observed forward is not committed
    /// as a new input token, so the normal reservation is intentionally one
    /// token larger than the highest forward position. This permits the
    /// frozen `p4095/g1` tail without weakening ordinary request validation.
    #[doc(hidden)]
    pub fn start_teacher_forced_capture_for_testing(
        &mut self,
        context: &mut RuntimeContext,
        request_id: impl Into<String>,
        prompt_token_ids: Vec<usize>,
        decode_positions: usize,
        stream: &mut RuntimeStream,
    ) -> Result<(), Sq8ServingError> {
        if decode_positions == 0 {
            return Err(Sq8ServingError::invalid_request(
                "teacher-forced capture requires at least one decode position",
            ));
        }
        let max_new_tokens = decode_positions.checked_add(1).ok_or_else(|| {
            Sq8ServingError::invalid_request("teacher-forced capture length overflows")
        })?;
        let request = Sq8ServingRequest::greedy(request_id, prompt_token_ids, max_new_tokens)
            .ignore_eos_for_testing();
        self.start_internal(
            context,
            request,
            Sq8CancellationToken::new(),
            None,
            true,
            stream,
        )
    }

    fn start_internal(
        &mut self,
        context: &mut RuntimeContext,
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
        reasoning_dialect: Option<&ReasoningDialect>,
        teacher_forced_capture: bool,
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
        if teacher_forced_capture {
            // Validate all ordinary request fields while permitting exactly
            // one uncommitted final token beyond the serving reservation.
            request.validate_for_worker(
                QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS + 1,
                QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS + 1,
                QWEN3_14B_VOCAB_SIZE,
                &QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS,
                QWEN3_14B_SQ8_SERVING_TOP_K,
            )?;
            let highest_forward_count = request
                .prompt_token_ids
                .len()
                .checked_add(request.max_new_tokens.saturating_sub(1))
                .ok_or_else(|| {
                    Sq8ServingError::invalid_request("teacher-forced context overflows")
                })?;
            if highest_forward_count > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS {
                return Err(Sq8ServingError::invalid_request(format!(
                    "teacher-forced capture exceeds context: forwards={highest_forward_count} context={QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS}"
                )));
            }
        } else {
            request.validate()?;
        }
        if let Err(err) = self.select_prefill_mode_for_request(
            context,
            request.prompt_token_ids.len(),
            stream,
        ) {
            return Err(self.fail_runtime(
                stream,
                format!("serving adaptive prefill selection failed: {err}"),
            ));
        }
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
                self.load_report.prefill_mode.uses_chunks(),
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
        let active =
            ActiveServingRequest::new_with_reasoning_dialect(request, cancel, reasoning_dialect)
                .map_err(Sq8ServingError::invalid_request)?;
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

    /// Advance one unit of the isolated teacher-forced capture state machine.
    ///
    /// `forced_next_token` is the already-materialized artifact-FP32 token to
    /// use on the following forward. Passing `None` records the final forward
    /// and resets the session without publishing an extra input token. This
    /// method is deliberately diagnostic-only and never changes the ordinary
    /// sampling or serving path.
    #[doc(hidden)]
    pub fn advance_teacher_forced_capture_for_testing(
        &mut self,
        forced_next_token: Option<usize>,
        capture_output: bool,
        capture_layers: bool,
        stream: &mut RuntimeStream,
    ) -> Result<Option<Sq8ServingTeacherForcedCapture>, Sq8ServingError> {
        if capture_layers && !capture_output {
            return Err(Sq8ServingError::invalid_configuration(
                "teacher-forced layer capture requires final-hidden/logit capture",
            ));
        }
        match self.state {
            Sq8ServingRuntimeStatus::Prefilling | Sq8ServingRuntimeStatus::Decoding => {}
            Sq8ServingRuntimeStatus::Ready => {
                return Err(Sq8ServingError::invalid_state(
                    "teacher-forced capture requires an active request",
                ));
            }
            Sq8ServingRuntimeStatus::Failed => return Err(self.failed_error()),
            state => {
                return Err(self.fail_runtime(
                    stream,
                    format!("teacher-forced capture is invalid in state {state:?}"),
                ));
            }
        }
        let cancelled = match self.active_cancelled() {
            Ok(cancelled) => cancelled,
            Err(err) => return Err(self.fail_runtime(stream, err)),
        };
        if cancelled {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Err(Sq8ServingError::invalid_state(
                "teacher-forced capture was cancelled",
            ));
        }
        let result = match self.state {
            Sq8ServingRuntimeStatus::Prefilling => self.advance_teacher_forced_prefill(
                forced_next_token,
                capture_output,
                capture_layers,
                stream,
            ),
            Sq8ServingRuntimeStatus::Decoding => self.advance_teacher_forced_decode(
                forced_next_token,
                capture_output,
                capture_layers,
                stream,
            ),
            _ => unreachable!("state was checked above"),
        };
        result.map_err(|err| self.fail_runtime(stream, err))
    }

    fn advance_teacher_forced_prefill(
        &mut self,
        forced_next_token: Option<usize>,
        capture_output: bool,
        capture_layers: bool,
        stream: &mut RuntimeStream,
    ) -> Result<Option<Sq8ServingTeacherForcedCapture>, String> {
        let (unit, prompt_tokens, token_ids) = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "teacher-forced prefill has no active request".to_string())?;
            let position = active.prompt_tokens_processed;
            let prompt_tokens = active.request.prompt_token_ids.len();
            let unit =
                plan_next_prefill_unit(position, prompt_tokens, self.load_report.prefill_mode)?;
            let end = unit.execution_end()?;
            let token_ids = active
                .request
                .prompt_token_ids
                .get(unit.execution_start_position..end)
                .ok_or_else(|| "teacher-forced prompt range exceeds request".to_string())?
                .to_vec();
            (unit, prompt_tokens, token_ids)
        };
        self.rewind_prefill_tail_for_execution(unit)?;
        let (source, input_token_id, position, layer_traces) = if unit.execution_width == 1 {
            let (report, traces) = self.execute_m1_stack_token_inner(
                token_ids[0],
                unit.execution_start_position,
                capture_layers,
                stream,
            )?;
            if report.position != unit.execution_start_position {
                return Err("teacher-forced M=1 prefill report position mismatch".into());
            }
            (
                Sq8ModelHeadServingSource::M1PagedDecode,
                token_ids[0],
                unit.execution_start_position,
                traces,
            )
        } else if Some(unit.execution_width) == self.load_report.prefill_mode.chunk_tokens() {
            let (report, traces) = if capture_layers {
                self.execute_stack_chunk_with_layer_trace(
                    &token_ids,
                    unit.execution_start_position,
                    stream,
                )?
            } else {
                (
                    self.execute_stack_chunk(&token_ids, unit.execution_start_position, stream)?,
                    Vec::new(),
                )
            };
            if report.prefix_position != unit.execution_start_position
                || report.chunk_len != unit.execution_width
            {
                return Err("teacher-forced chunk prefill report geometry mismatch".into());
            }
            (
                Sq8ModelHeadServingSource::CachedPrefixChunk,
                *token_ids
                    .last()
                    .ok_or_else(|| "teacher-forced chunk has no final input token".to_string())?,
                unit.execution_end()? - 1,
                traces,
            )
        } else {
            return Err(format!(
                "teacher-forced prefill planner produced unsupported width {}",
                unit.execution_width
            ));
        };
        let scheduler_cached =
            self.commit_prompt_progress(unit.logical_start_position, unit.committed_tokens)?;
        let capture = self.capture_teacher_forced_head(
            source,
            scheduler_cached,
            input_token_id,
            position,
            unit.execution_width,
            capture_output,
            capture_layers,
            layer_traces,
            stream,
        )?;

        if !unit.is_final {
            if forced_next_token.is_some() {
                return Err("teacher-forced token supplied before final prompt unit".into());
            }
            if scheduler_cached >= prompt_tokens {
                return Err("teacher-forced non-final prefill reached prompt boundary".into());
            }
            return Ok(capture);
        }
        if scheduler_cached != prompt_tokens {
            return Err(format!(
                "teacher-forced final prefill cache mismatch: expected={prompt_tokens} actual={scheduler_cached}"
            ));
        }
        self.commit_teacher_forced_or_finish(
            forced_next_token,
            scheduler_cached,
            GeneratedTokenCommit::Prefill,
            stream,
        )?;
        Ok(capture)
    }

    fn advance_teacher_forced_decode(
        &mut self,
        forced_next_token: Option<usize>,
        capture_output: bool,
        capture_layers: bool,
        stream: &mut RuntimeStream,
    ) -> Result<Option<Sq8ServingTeacherForcedCapture>, String> {
        let DecodeStepPlan {
            input_token_id,
            expected_position,
            ready,
        } = self.decode_step_plan()?;
        let (report, layer_traces) = self.execute_m1_stack_token_inner(
            input_token_id,
            expected_position,
            capture_layers,
            stream,
        )?;
        if report.position != expected_position {
            return Err("teacher-forced decode report position mismatch".into());
        }
        validate_cache_lengths(self.caches.as_ref(), expected_position + 1)?;
        let capture = self.capture_teacher_forced_head(
            Sq8ModelHeadServingSource::M1PagedDecode,
            expected_position + 1,
            input_token_id,
            expected_position,
            1,
            capture_output,
            capture_layers,
            layer_traces,
            stream,
        )?;
        self.commit_teacher_forced_or_finish(
            forced_next_token,
            expected_position + 1,
            GeneratedTokenCommit::Decode(ready),
            stream,
        )?;
        Ok(capture)
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_teacher_forced_head(
        &mut self,
        source: Sq8ModelHeadServingSource,
        expected_cache_len: usize,
        input_token_id: usize,
        position: usize,
        execution_width: usize,
        capture_output: bool,
        capture_layers: bool,
        layer_traces: Vec<Sq8LayerRuntimeTrace>,
        stream: &mut RuntimeStream,
    ) -> Result<Option<Sq8ServingTeacherForcedCapture>, String> {
        if !capture_output {
            if capture_layers || !layer_traces.is_empty() {
                return Err(
                    "teacher-forced layer trace was produced without an output capture".into(),
                );
            }
            return Ok(None);
        }
        let capture = match self.run_head_synchronized(source, expected_cache_len, stream, true)? {
            HeadPreparation::Prepared {
                capture: Some(capture),
                ..
            } => capture,
            HeadPreparation::Prepared { capture: None, .. } => {
                return Err("teacher-forced oracle head omitted its capture".into());
            }
            HeadPreparation::CancellationObserved => {
                return Err("teacher-forced oracle head observed cancellation".into());
            }
        };
        if capture.position != position {
            return Err(format!(
                "teacher-forced head position mismatch: expected={position} actual={}",
                capture.position
            ));
        }
        let layers = if capture_layers {
            if layer_traces.len() != QWEN3_14B_SQ8_STACK_LAYERS {
                return Err(format!(
                    "teacher-forced layer trace count mismatch: expected={} actual={}",
                    QWEN3_14B_SQ8_STACK_LAYERS,
                    layer_traces.len()
                ));
            }
            let row_start = execution_width
                .checked_sub(1)
                .and_then(|row| row.checked_mul(QWEN3_14B_HIDDEN_SIZE))
                .ok_or_else(|| "teacher-forced layer trace row offset overflows".to_string())?;
            let row_end = row_start
                .checked_add(QWEN3_14B_HIDDEN_SIZE)
                .ok_or_else(|| "teacher-forced layer trace row end overflows".to_string())?;
            let mut outputs = Vec::with_capacity(layer_traces.len());
            for (layer_index, trace) in layer_traces.into_iter().enumerate() {
                if trace.output.len() != execution_width * QWEN3_14B_HIDDEN_SIZE {
                    return Err(format!(
                        "teacher-forced layer {layer_index} output length mismatch: expected={} actual={}",
                        execution_width * QWEN3_14B_HIDDEN_SIZE,
                        trace.output.len()
                    ));
                }
                outputs.push(trace.output[row_start..row_end].to_vec());
            }
            Some(outputs)
        } else {
            if !layer_traces.is_empty() {
                return Err("teacher-forced unexpected layer traces".into());
            }
            None
        };
        Ok(Some(Sq8ServingTeacherForcedCapture {
            input_token_id,
            position,
            top1: capture.top1,
            final_hidden: capture.final_hidden,
            logits: capture.logits,
            layers,
        }))
    }

    fn commit_teacher_forced_or_finish(
        &mut self,
        forced_next_token: Option<usize>,
        cache_len: usize,
        commit: GeneratedTokenCommit,
        stream: &mut RuntimeStream,
    ) -> Result<(), String> {
        let Some(token_id) = forced_next_token else {
            // The final output needs no following input. Reset instead of
            // publishing a fabricated token; this is what makes the final
            // context-boundary forward executable without extending KV state.
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            self.reset_active_synchronized(Sq8ReleaseOutcome::Cancelled, stream)
                .map_err(|err| err.to_string())?;
            return Ok(());
        };
        if token_id >= QWEN3_14B_VOCAB_SIZE {
            return Err(format!(
                "teacher-forced next token exceeds vocabulary: {token_id}"
            ));
        }
        let prepared = self.prepare_selected_token(
            token_id,
            PendingTokenSource::TeacherForced,
            None,
            cache_len,
            commit,
        )?;
        let published = self
            .publish_prepared_token(prepared, stream, |_| Ok(()))
            .map_err(|err| err.to_string())?;
        match published {
            Sq8ServingAdvance::Token {
                terminal_reason: None,
                ..
            } => Ok(()),
            Sq8ServingAdvance::Token {
                terminal_reason: Some(reason),
                ..
            } => Err(format!(
                "teacher-forced capture reached terminal state before final forward: {reason:?}"
            )),
            other => Err(format!(
                "teacher-forced token publication did not commit a token: {other:?}"
            )),
        }
    }

    pub fn advance_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ServingAdvance, Sq8ServingError> {
        match self.prepare_advance_synchronized(stream)? {
            Sq8PreparedAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            } => Ok(Sq8ServingAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            }),
            Sq8PreparedAdvance::Token(prepared) => {
                self.publish_prepared_token(prepared, stream, |_| Ok(()))
            }
            Sq8PreparedAdvance::CancellationObserved => Ok(Sq8ServingAdvance::CancellationObserved),
        }
    }

    pub fn prepare_advance_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8PreparedAdvance, Sq8ServingError> {
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
            return Ok(Sq8PreparedAdvance::CancellationObserved);
        }

        let result = match self.state {
            Sq8ServingRuntimeStatus::Prefilling => self
                .prepare_prefill_synchronized(stream, false)
                .map(|result| result.advance),
            Sq8ServingRuntimeStatus::Decoding => self
                .prepare_decode_synchronized(stream, false)
                .map(|result| result.advance),
            _ => unreachable!("state checked above"),
        };
        result.map_err(|err| self.fail_runtime(stream, err))
    }

    pub fn publish_prepared_token<F>(
        &mut self,
        prepared: Sq8PreparedToken,
        stream: &mut RuntimeStream,
        publish: F,
    ) -> Result<Sq8ServingAdvance, Sq8ServingError>
    where
        F: FnOnce(&Sq8PreparedToken) -> Result<(), String>,
    {
        if self.state == Sq8ServingRuntimeStatus::Failed {
            return Err(self.failed_error());
        }
        if self.state != Sq8ServingRuntimeStatus::TokenPrepared {
            return Err(self.fail_runtime(
                stream,
                format!(
                    "serving token publication requires TokenPrepared, got {:?}",
                    self.state
                ),
            ));
        }
        if self.pending_token.as_ref().map(|pending| &pending.prepared) != Some(&prepared) {
            return Err(self.fail_runtime(
                stream,
                "serving token publication handle does not match pending token",
            ));
        }
        let cancel = match self.active.as_ref() {
            Some(active) => active.cancel.clone(),
            None => {
                return Err(
                    self.fail_runtime(stream, "serving token publication has no active request")
                );
            }
        };
        match publish_prepared_token_transaction(
            &mut self.state,
            &mut self.pending_token,
            &mut self.active,
            &mut self.scheduler,
            &cancel,
            &prepared,
            publish,
        ) {
            Ok(committed) => Ok(committed),
            Err(err) => Err(self.fail_runtime(stream, err)),
        }
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
        let prepared = self
            .prepare_prefill_synchronized(stream, true)
            .map_err(|err| self.fail_runtime(stream, err))?;
        let advance = match prepared.advance {
            Sq8PreparedAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            } => Sq8ServingAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            },
            Sq8PreparedAdvance::Token(token) => {
                self.publish_prepared_token(token, stream, |_| Ok(()))?
            }
            Sq8PreparedAdvance::CancellationObserved => Sq8ServingAdvance::CancellationObserved,
        };
        Ok(Sq8ServingOracleAdvance {
            capture: if matches!(advance, Sq8ServingAdvance::Token { .. }) {
                prepared.capture
            } else {
                None
            },
            advance,
        })
    }

    /// Captures final hidden/logits for one actual M=1 decode step.
    ///
    /// This is intentionally separate from the prefill oracle so differential
    /// harnesses can prove the dispatch used after token feedback without
    /// changing the normal lean serving path.
    pub fn advance_decode_oracle_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ServingOracleAdvance, Sq8ServingError> {
        match self.state {
            Sq8ServingRuntimeStatus::Decoding => {}
            Sq8ServingRuntimeStatus::Ready => {
                return Err(Sq8ServingError::invalid_state(
                    "serving decode oracle requires an active request",
                ));
            }
            Sq8ServingRuntimeStatus::Failed => return Err(self.failed_error()),
            state => {
                return Err(self.fail_runtime(
                    stream,
                    format!("serving decode oracle is invalid in state {state:?}"),
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
        let prepared = self
            .prepare_decode_synchronized(stream, true)
            .map_err(|err| self.fail_runtime(stream, err))?;
        let advance = match prepared.advance {
            Sq8PreparedAdvance::PromptProgress { .. } => {
                return Err(self.fail_runtime(
                    stream,
                    "serving decode oracle unexpectedly made prompt progress",
                ));
            }
            Sq8PreparedAdvance::Token(token) => {
                self.publish_prepared_token(token, stream, |_| Ok(()))?
            }
            Sq8PreparedAdvance::CancellationObserved => Sq8ServingAdvance::CancellationObserved,
        };
        Ok(Sq8ServingOracleAdvance {
            capture: if matches!(advance, Sq8ServingAdvance::Token { .. }) {
                prepared.capture
            } else {
                None
            },
            advance,
        })
    }

    /// Executes exactly the next serving M=1 decode and captures each layer
    /// workspace immediately after it runs.
    ///
    /// This terminal diagnostic intentionally does not run the model head or
    /// commit a generated token. The caller must drop the session afterwards;
    /// it must not resume ordinary serving from this partially advanced state.
    #[doc(hidden)]
    pub fn trace_next_decode_layers_for_testing_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ServingDecodeLayerTraceCapture, Sq8ServingError> {
        match self.state {
            Sq8ServingRuntimeStatus::Decoding => {}
            Sq8ServingRuntimeStatus::Ready => {
                return Err(Sq8ServingError::invalid_state(
                    "serving decode layer trace requires an active request",
                ));
            }
            Sq8ServingRuntimeStatus::Failed => return Err(self.failed_error()),
            state => {
                return Err(self.fail_runtime(
                    stream,
                    format!("serving decode layer trace is invalid in state {state:?}"),
                ));
            }
        }
        let cancelled = match self.active_cancelled() {
            Ok(cancelled) => cancelled,
            Err(err) => return Err(self.fail_runtime(stream, err)),
        };
        if cancelled {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Err(Sq8ServingError::invalid_state(
                "serving decode layer trace was cancelled before execution",
            ));
        }
        let plan = self
            .decode_step_plan()
            .map_err(|err| self.fail_runtime(stream, err))?;
        let (report, layers) = self
            .execute_m1_stack_token_with_layer_trace(
                plan.input_token_id,
                plan.expected_position,
                stream,
            )
            .map_err(|err| self.fail_runtime(stream, err))?;
        validate_cache_lengths(self.caches.as_ref(), plan.expected_position + 1)
            .map_err(|err| self.fail_runtime(stream, err))?;
        if layers.len() != QWEN3_14B_SQ8_STACK_LAYERS {
            return Err(self.fail_runtime(
                stream,
                format!(
                    "serving decode layer trace count mismatch: expected={} actual={}",
                    QWEN3_14B_SQ8_STACK_LAYERS,
                    layers.len()
                ),
            ));
        }
        Ok(Sq8ServingDecodeLayerTraceCapture {
            input_token_id: plan.input_token_id,
            position: plan.expected_position,
            profile: report.stack.profile,
            layers,
        })
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
    /// Makes an overlapping fixed-width tail visible as a contiguous scheduler advance.
    ///
    /// The prior chunk has already synchronized, so moving these logical cursors does not reorder
    /// GPU work. The immediately following chunk overwrites every rewound cache entry with real
    /// tokens before cached-prefix attention uses the new logical length.
    fn rewind_prefill_tail_for_execution(&mut self, unit: Sq8PrefillUnit) -> Result<(), String> {
        if !unit.rewinds_cache() {
            return Ok(());
        }
        let chunk_tokens = self
            .load_report
            .prefill_mode
            .chunk_tokens()
            .ok_or_else(|| "serving prefill tail rewind requires a fixed chunk mode".to_string())?;
        if unit.execution_width != chunk_tokens
            || unit.execution_end()? != unit.logical_end()?
            || unit.committed_tokens >= unit.execution_width
        {
            return Err("serving prefill overlap geometry is invalid".into());
        }
        validate_cache_lengths(self.caches.as_ref(), unit.logical_start_position)?;
        self.stack.rewind_paged_serving_cursor(
            unit.logical_start_position,
            unit.execution_start_position,
        )?;
        for (layer_index, cache) in self.caches.iter_mut().enumerate() {
            cache
                .rewind_serving_write_cursor(unit.execution_start_position)
                .map_err(|error| {
                    format!(
                        "serving prefill tail failed to rewind layer {layer_index} cache: {error}"
                    )
                })?;
        }
        validate_cache_lengths(self.caches.as_ref(), unit.execution_start_position)
    }

    fn prepare_prefill_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
        capture_oracle: bool,
    ) -> Result<PreparedOracleAdvance, String> {
        let (unit, prompt_tokens, token_ids) = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "serving Prefilling state has no active request".to_string())?;
            let position = active.prompt_tokens_processed;
            let prompt_tokens = active.request.prompt_token_ids.len();
            let unit =
                plan_next_prefill_unit(position, prompt_tokens, self.load_report.prefill_mode)?;
            let end = unit.execution_end()?;
            let token_ids = active
                .request
                .prompt_token_ids
                .get(unit.execution_start_position..end)
                .ok_or_else(|| {
                    format!(
                        "serving prompt range {}..{end} exceeds prompt length {prompt_tokens}",
                        unit.execution_start_position
                    )
                })?
                .to_vec();
            (unit, prompt_tokens, token_ids)
        };
        self.rewind_prefill_tail_for_execution(unit)?;
        if unit.execution_width == 1 {
            self.execute_m1_stack_token(token_ids[0], unit.execution_start_position, stream)?;
        } else if Some(unit.execution_width) == self.load_report.prefill_mode.chunk_tokens() {
            self.execute_stack_chunk(&token_ids, unit.execution_start_position, stream)?;
        } else {
            return Err(format!(
                "serving prefill planner produced unsupported execution width {}",
                unit.execution_width
            ));
        }
        let scheduler_cached =
            self.commit_prompt_progress(unit.logical_start_position, unit.committed_tokens)?;
        if self.active_cancelled()? {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Ok(PreparedOracleAdvance {
                advance: Sq8PreparedAdvance::CancellationObserved,
                capture: None,
            });
        }
        if !unit.is_final {
            if scheduler_cached >= prompt_tokens {
                return Err("serving non-final prefill unit reached prompt boundary".into());
            }
            return Ok(PreparedOracleAdvance {
                advance: Sq8PreparedAdvance::PromptProgress {
                    prompt_tokens_processed: scheduler_cached,
                    cache_len: scheduler_cached,
                    execution_width: unit.execution_width,
                },
                capture: None,
            });
        }
        if scheduler_cached != prompt_tokens {
            return Err(format!(
                "serving final prefill unit cache mismatch: expected={prompt_tokens} actual={scheduler_cached}"
            ));
        }

        let source = match unit.execution_width {
            1 => Sq8ModelHeadServingSource::M1PagedDecode,
            _ => Sq8ModelHeadServingSource::CachedPrefixChunk,
        };
        if let Some((token_id, reasoning_after)) = self
            .active
            .as_ref()
            .ok_or_else(|| "serving final prefill has no active request".to_string())?
            .forced_reasoning_transition()?
        {
            let prepared = self.prepare_selected_token(
                token_id,
                PendingTokenSource::ReasoningForced,
                Some(reasoning_after),
                scheduler_cached,
                GeneratedTokenCommit::Prefill,
            )?;
            return Ok(PreparedOracleAdvance {
                advance: Sq8PreparedAdvance::Token(prepared),
                capture: None,
            });
        }
        match self.run_head_synchronized(source, scheduler_cached, stream, capture_oracle)? {
            HeadPreparation::Prepared { proposal, capture } => {
                let prepared = self.prepare_generated_token(
                    proposal,
                    scheduler_cached,
                    GeneratedTokenCommit::Prefill,
                )?;
                Ok(PreparedOracleAdvance {
                    advance: Sq8PreparedAdvance::Token(prepared),
                    capture,
                })
            }
            HeadPreparation::CancellationObserved => Ok(PreparedOracleAdvance {
                advance: Sq8PreparedAdvance::CancellationObserved,
                capture: None,
            }),
        }
    }

    fn decode_step_plan(&self) -> Result<DecodeStepPlan, String> {
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

        Ok(DecodeStepPlan {
            input_token_id,
            expected_position,
            ready,
        })
    }

    fn prepare_decode_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
        capture_oracle: bool,
    ) -> Result<PreparedOracleAdvance, String> {
        let DecodeStepPlan {
            input_token_id,
            expected_position,
            ready,
        } = self.decode_step_plan()?;
        self.execute_m1_stack_token(input_token_id, expected_position, stream)?;
        validate_cache_lengths(self.caches.as_ref(), expected_position + 1)?;
        if self.active_cancelled()? {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Ok(PreparedOracleAdvance {
                advance: Sq8PreparedAdvance::CancellationObserved,
                capture: None,
            });
        }
        if let Some((token_id, reasoning_after)) = self
            .active
            .as_ref()
            .ok_or_else(|| "serving decode has no active request".to_string())?
            .forced_reasoning_transition()?
        {
            return self
                .prepare_selected_token(
                    token_id,
                    PendingTokenSource::ReasoningForced,
                    Some(reasoning_after),
                    expected_position + 1,
                    GeneratedTokenCommit::Decode(ready),
                )
                .map(|token| PreparedOracleAdvance {
                    advance: Sq8PreparedAdvance::Token(token),
                    capture: None,
                });
        }
        let head = self.run_head_synchronized(
            Sq8ModelHeadServingSource::M1PagedDecode,
            expected_position + 1,
            stream,
            capture_oracle,
        )?;
        match head {
            HeadPreparation::Prepared { proposal, capture } => self
                .prepare_generated_token(
                    proposal,
                    expected_position + 1,
                    GeneratedTokenCommit::Decode(ready),
                )
                .map(|token| PreparedOracleAdvance {
                    advance: Sq8PreparedAdvance::Token(token),
                    capture,
                }),
            HeadPreparation::CancellationObserved => Ok(PreparedOracleAdvance {
                advance: Sq8PreparedAdvance::CancellationObserved,
                capture: None,
            }),
        }
    }

    fn execute_m1_stack_token(
        &mut self,
        token_id: usize,
        position: usize,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8PagedStackExecutionReport, String> {
        self.execute_m1_stack_token_inner(token_id, position, false, stream)
            .map(|(report, _)| report)
    }

    fn execute_m1_stack_token_with_layer_trace(
        &mut self,
        token_id: usize,
        position: usize,
        stream: &mut RuntimeStream,
    ) -> Result<(Sq8PagedStackExecutionReport, Vec<Sq8LayerRuntimeTrace>), String> {
        self.execute_m1_stack_token_inner(token_id, position, true, stream)
    }

    fn execute_m1_stack_token_inner(
        &mut self,
        token_id: usize,
        position: usize,
        capture_layer_trace: bool,
        stream: &mut RuntimeStream,
    ) -> Result<(Sq8PagedStackExecutionReport, Vec<Sq8LayerRuntimeTrace>), String> {
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
        let expected_profile = if self.handwritten_wmma_prototype_enabled {
            Sq8LayerExecutionProfile::Rdna4W8a8BlockHandwrittenWmmaPrototype
        } else {
            Sq8LayerExecutionProfile::Rdna4W8a8BlockCk
        };
        let (report, layer_traces) = if capture_layer_trace {
            self.stack
                .run_paged_m1_sequence_step_with_layer_trace_synchronized(
                    &mut self.decode,
                    embedding_output,
                    position,
                    &mut self.caches[..],
                    expected_profile,
                    stream,
                )?
        } else if self.handwritten_wmma_prototype_enabled {
            (
                self.stack
                    .run_paged_m1_sequence_step_handwritten_wmma_prototype_synchronized(
                        &mut self.decode,
                        embedding_output,
                        position,
                        &mut self.caches[..],
                        stream,
                    )?,
                Vec::new(),
            )
        } else {
            (
                self.stack
                    .run_paged_m1_sequence_step_optimized_synchronized(
                        &mut self.decode,
                        embedding_output,
                        position,
                        &mut self.caches[..],
                        stream,
                    )?,
                Vec::new(),
            )
        };
        report.validate_contract()?;
        if report.phase != Sq8PagedStackPhase::Decode
            || report.position != position
            || report.stack.sequence_len != 1
            || report.stack.profile != expected_profile
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
        Ok((report, layer_traces))
    }

    fn execute_stack_chunk(
        &mut self,
        token_ids: &[usize],
        prefix_position: usize,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ServingChunkExecutionReport, String> {
        self.execute_stack_chunk_inner(token_ids, prefix_position, false, stream)
            .map(|(report, _)| report)
    }

    fn execute_stack_chunk_with_layer_trace(
        &mut self,
        token_ids: &[usize],
        prefix_position: usize,
        stream: &mut RuntimeStream,
    ) -> Result<(Sq8ServingChunkExecutionReport, Vec<Sq8LayerRuntimeTrace>), String> {
        self.execute_stack_chunk_inner(token_ids, prefix_position, true, stream)
    }

    fn execute_stack_chunk_inner(
        &mut self,
        token_ids: &[usize],
        prefix_position: usize,
        capture_layer_trace: bool,
        stream: &mut RuntimeStream,
    ) -> Result<(Sq8ServingChunkExecutionReport, Vec<Sq8LayerRuntimeTrace>), String> {
        let chunk_tokens = self
            .load_report
            .prefill_mode
            .chunk_tokens()
            .ok_or_else(|| "serving chunk execution requires a fixed chunk mode".to_string())?;
        if token_ids.len() != chunk_tokens {
            return Err(format!(
                "serving chunk requires {chunk_tokens} tokens, got {}",
                token_ids.len()
            ));
        }
        let end = prefix_position
            .checked_add(token_ids.len())
            .ok_or_else(|| "serving chunk position overflows".to_string())?;
        if end > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS {
            return Err(format!(
                "serving chunk exceeds context: range={prefix_position}..{end}"
            ));
        }
        validate_cache_lengths(self.caches.as_ref(), prefix_position)?;
        let hidden_row_bytes = QWEN3_14B_HIDDEN_SIZE
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "serving chunk embedding row byte size overflows".to_string())?;
        let expected_chunk_bytes = qwen3_14b_sq8_serving_prompt_chunk_bytes(chunk_tokens)?;
        if self.prompt_chunk_hidden.size()? != expected_chunk_bytes {
            return Err(format!(
                "serving prompt chunk buffer size mismatch: expected={expected_chunk_bytes} actual={}",
                self.prompt_chunk_hidden.size()?
            ));
        }

        for (row, token_id) in token_ids.iter().copied().enumerate() {
            if token_id >= QWEN3_14B_VOCAB_SIZE {
                return Err(format!(
                    "serving chunk input token at row {row} exceeds vocabulary: {token_id}"
                ));
            }
            let embedding_report = self.embedding.enqueue_token_resident(token_id, stream)?;
            validate_embedding_report(&embedding_report, token_id, &self.load_report)?;
            let (embedding_output, resident_report) = self.embedding.resident_output()?;
            if resident_report != &embedding_report {
                return Err(format!(
                    "serving embedding report changed before chunk row {row} copy"
                ));
            }
            let destination_offset = row.checked_mul(hidden_row_bytes).ok_or_else(|| {
                "serving chunk embedding destination offset overflows".to_string()
            })?;
            self.prompt_chunk_hidden
                .copy_from_buffer(
                    destination_offset,
                    embedding_output,
                    0,
                    hidden_row_bytes,
                    Some(&mut *stream),
                )
                .map_err(|err| {
                    format!("failed to copy serving chunk embedding row {row} D2D: {err}")
                })?;
        }

        let (report, layer_traces) = if capture_layer_trace {
            self.stack
                .run_paged_serving_chunk_with_layer_trace_synchronized(
                    &self.prompt_chunk_hidden,
                    prefix_position,
                    &mut self.caches[..],
                    stream,
                )?
        } else {
            (
                self.stack.run_paged_serving_chunk_optimized_synchronized(
                    &self.prompt_chunk_hidden,
                    prefix_position,
                    &mut self.caches[..],
                    stream,
                )?,
                Vec::new(),
            )
        };
        report.validate_contract()?;
        if report.prefix_position != prefix_position
            || report.chunk_len != chunk_tokens
            || report.stack.artifact_content_sha256 != self.load_report.artifact_content_sha256
            || report.cache_lengths.iter().any(|length| *length != end)
            || report.stack.fallback_used
            || report.stack.host_staging_used
        {
            return Err(format!(
                "serving chunk stack report failed at prefix {prefix_position}: {report:?}"
            ));
        }
        Ok((report, layer_traces))
    }

    fn run_head_synchronized(
        &mut self,
        source: Sq8ModelHeadServingSource,
        expected_cache_len: usize,
        stream: &mut RuntimeStream,
        capture_oracle: bool,
    ) -> Result<HeadPreparation, String> {
        let result = match (source, capture_oracle) {
            (Sq8ModelHeadServingSource::M1PagedDecode, true) => self
                .head
                .run_m1_serving_oracle_synchronized(&self.decode, stream)?,
            (Sq8ModelHeadServingSource::M1PagedDecode, false) => self
                .head
                .run_m1_serving_logits_synchronized(&self.decode, stream)?,
            (Sq8ModelHeadServingSource::CachedPrefixChunk, true) => self
                .head
                .run_chunk_serving_oracle_synchronized(&self.stack, stream)?,
            (Sq8ModelHeadServingSource::CachedPrefixChunk, false) => self
                .head
                .run_chunk_serving_logits_synchronized(&self.stack, stream)?,
        };
        result.validate_contract()?;
        if result.report.source != source
            || result.report.position + 1 != expected_cache_len
            || result.report.cache_len != expected_cache_len
            || result.report.binding.device != self.load_report.device
            || result.report.binding.package_manifest_sha256
                != self.load_report.package_manifest_sha256
            || result.report.binding.artifact_content_sha256
                != self.load_report.artifact_content_sha256
            || result.report.final_norm.payload_sha256 != self.load_report.final_norm_payload_sha256
            || result.report.lm_head.payload_sha256 != self.load_report.lm_head_payload_sha256
            || result.report.fallback_used
            || result.report.host_staging_used
        {
            return Err("serving model-head source/identity/report mismatch".into());
        }
        if self.active_cancelled()? {
            self.state = Sq8ServingRuntimeStatus::Cancelling;
            return Ok(HeadPreparation::CancellationObserved);
        }
        let proposal = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "serving model head has no active sampler".to_string())?;
            validate_active_sampling_progress(active)?;
            active.sampler.propose(
                &result.logits,
                active.request.sampling.temperature,
                active.request.sampling.top_k,
                active.request.sampling.top_p,
            )?
        };
        let sampled = proposal.sampled();
        if sampled.token_id >= QWEN3_14B_VOCAB_SIZE || !sampled.logit.is_finite() {
            return Err(format!(
                "serving sampler proposed invalid token: {sampled:?}"
            ));
        }
        let capture = if capture_oracle {
            let top1 = greedy_top1_finite(&result.logits)?;
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
        Ok(HeadPreparation::Prepared { proposal, capture })
    }

    fn commit_prompt_progress(&mut self, position: usize, width: usize) -> Result<usize, String> {
        let expected = position
            .checked_add(width)
            .ok_or_else(|| "serving prompt position overflows".to_string())?;
        validate_cache_lengths(self.caches.as_ref(), expected)?;
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
            || expected > active.request.prompt_token_ids.len()
        {
            return Err("serving prompt commit metadata is stale".into());
        }
        let actual = self
            .scheduler
            .advance_prefill_tokens(SERVING_INTERNAL_REQUEST_ID, width)?;
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

    fn prepare_generated_token(
        &mut self,
        proposal: Sq8SamplingProposal,
        cache_len: usize,
        commit: GeneratedTokenCommit,
    ) -> Result<Sq8PreparedToken, String> {
        if self.pending_token.is_some() {
            return Err("serving already has a pending token".into());
        }
        let sampled = proposal.sampled();
        if sampled.token_id >= QWEN3_14B_VOCAB_SIZE || !sampled.logit.is_finite() {
            return Err(format!("serving sampled invalid token: {sampled:?}"));
        }
        let (token_id, forced, reasoning_after) = self
            .active
            .as_ref()
            .ok_or_else(|| "serving generated token has no active request".to_string())?
            .sampled_reasoning_transition(sampled.token_id)?;
        let source = if forced {
            PendingTokenSource::ReasoningForced
        } else {
            PendingTokenSource::Sampled(proposal)
        };
        self.prepare_selected_token(token_id, source, reasoning_after, cache_len, commit)
    }

    fn prepare_selected_token(
        &mut self,
        token_id: usize,
        source: PendingTokenSource,
        reasoning_after: Option<ReasoningState>,
        cache_len: usize,
        commit: GeneratedTokenCommit,
    ) -> Result<Sq8PreparedToken, String> {
        if self.pending_token.is_some() {
            return Err("serving already has a pending token".into());
        }
        if token_id >= QWEN3_14B_VOCAB_SIZE {
            return Err(format!("serving selected invalid token: {token_id}"));
        }
        let (generated_index, terminal_reason) = {
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
                active.terminal_reason_after(token_id, reasoning_after.as_ref()),
            )
        };

        let nonce = self.next_prepared_nonce;
        self.next_prepared_nonce = nonce
            .checked_add(1)
            .ok_or_else(|| "serving prepared-token nonce overflows".to_string())?;
        let prepared = Sq8PreparedToken {
            token_id,
            generated_index,
            cache_len,
            terminal_reason,
            nonce,
        };
        self.pending_token = Some(PendingServingToken {
            prepared: prepared.clone(),
            source,
            reasoning_after,
            commit,
        });
        self.state = Sq8ServingRuntimeStatus::TokenPrepared;
        Ok(prepared)
    }

    fn reset_active_synchronized(
        &mut self,
        outcome: Sq8ReleaseOutcome,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ReleaseSummary, Sq8ServingError> {
        let release = match self.active.as_ref() {
            Some(active) => active.snapshot_release_accounting(outcome),
            None => {
                return Err(self.fail_runtime(stream, "serving reset has no active request"));
            }
        };
        let expected_table = match qwen3_14b_sq8_serving_block_table() {
            Ok(table) => table,
            Err(err) => return Err(self.fail_runtime(stream, err.to_string())),
        };
        let reset_preflight = (|| {
            if self.pending_token.is_some() {
                return Err("serving reset cannot discard a pending token".into());
            }
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "serving reset has no active metadata".to_string())?;
            validate_active_sampling_progress(active)?;
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
            let prompt_chunk_bytes =
                qwen3_14b_sq8_serving_prompt_chunk_bytes(self.load_report.prefill_chunk_tokens)?;
            if self.prompt_chunk_hidden.size()? != prompt_chunk_bytes {
                return Err("serving prompt chunk buffer size changed before reset".into());
            }
            self.prompt_chunk_hidden
                .zero(0, prompt_chunk_bytes, Some(&mut *stream))
                .map_err(|err| format!("failed to enqueue serving prompt chunk reset: {err}"))?;
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
        self.pending_token = None;
        self.state = Sq8ServingRuntimeStatus::Ready;
        if let Err(err) = self.validate_ready_baseline() {
            return Err(
                self.fail_runtime(stream, format!("serving post-reset baseline failed: {err}"))
            );
        }
        Ok(release.complete_after_reset())
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
            || self.pending_token.is_some()
        {
            return Err("serving Ready metadata is not at baseline".into());
        }
        if self.stack.config().sequence_len != self.load_report.prefill_chunk_tokens
            || self.stack.layer_count() != QWEN3_14B_SQ8_STACK_LAYERS
            || self.stack.artifact_content_sha256() != self.load_report.artifact_content_sha256
            || self.stack.poison_reason().is_some()
            || self.embedding.poison_reason().is_some()
            || self.head.poison_reason().is_some()
        {
            return Err("serving resident model state is not reusable".into());
        }
        if self.prompt_chunk_hidden.size()?
            != qwen3_14b_sq8_serving_prompt_chunk_bytes(self.load_report.prefill_chunk_tokens)?
        {
            return Err("serving resident prompt chunk buffer size mismatch".into());
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

fn qwen3_14b_sq8_serving_prompt_chunk_bytes(chunk_tokens: usize) -> Result<usize, String> {
    chunk_tokens
        .checked_mul(QWEN3_14B_HIDDEN_SIZE)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "serving prompt chunk byte size overflows".to_string())
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

fn validate_active_sampling_progress(active: &ActiveServingRequest) -> Result<(), String> {
    let expected_draws = if active.request.sampling.temperature == 0.0 {
        0
    } else {
        u64::try_from(active.sampled_tokens)
            .map_err(|_| "serving sampled-token count does not fit RNG draw counter".to_string())?
    };
    let forced_tokens = active
        .reasoning
        .as_ref()
        .map_or(0, |reasoning| reasoning.forced_end_tokens);
    let accounted_tokens = active
        .sampled_tokens
        .checked_add(forced_tokens)
        .and_then(|count| count.checked_add(active.teacher_forced_tokens));
    if accounted_tokens != Some(active.generated_tokens) {
        return Err(format!(
            "serving generated-token accounting mismatch: generated={} sampled={} forced={} teacher_forced={}",
            active.generated_tokens,
            active.sampled_tokens,
            forced_tokens,
            active.teacher_forced_tokens
        ));
    }
    if active.sampler.draws() != expected_draws {
        return Err(format!(
            "serving sampling progress mismatch: sampled_tokens={} expected_draws={expected_draws} actual_draws={}",
            active.sampled_tokens,
            active.sampler.draws()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sq8_stack_runtime::QWEN3_14B_SQ8_STACK_LAYERS;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn sq8_paged_decode_split_tile_parser_is_explicit_and_closed() {
        assert_eq!(parse_paged_decode_split_source_tile(None).unwrap(), None);
        assert_eq!(
            parse_paged_decode_split_source_tile(Some("20")).unwrap(),
            Some(20)
        );
        assert_eq!(
            parse_paged_decode_split_source_tile(Some("128")).unwrap(),
            Some(128)
        );
        assert_eq!(
            parse_paged_decode_split_source_tile(Some("256")).unwrap(),
            Some(256)
        );
        assert_eq!(
            parse_paged_decode_split_source_tile(Some("512")).unwrap(),
            Some(512)
        );
        for invalid in ["", "0", "64", "1024", "256 ", "tile256"] {
            assert!(parse_paged_decode_split_source_tile(Some(invalid)).is_err());
        }
    }

    fn qwen3_reasoning_dialect() -> ReasoningDialect {
        ReasoningDialect {
            identity: "qwen3-thinking-v1".into(),
            start_sequence: vec![151_667],
            end_sequence: vec![151_668],
            forced_end_sequence: vec![151_668],
            max_budget_tokens: 256,
            reserved_answer_tokens: 1,
            enabled_by_default: false,
            effort_budgets: vec![
                ("low".into(), 32),
                ("medium".into(), 128),
                ("high".into(), 256),
            ],
            history_reasoning_policy: crate::reasoning::HistoryReasoningPolicy::Omit,
            initial_phase: crate::reasoning::InitialReasoningPhase::Reasoning,
            eos_policy: crate::reasoning::ReasoningEosPolicy::Close,
        }
    }

    fn reasoning_active(
        enabled: bool,
        budget_tokens: Option<usize>,
        max_new_tokens: usize,
        temperature: f32,
    ) -> ActiveServingRequest {
        let dialect = qwen3_reasoning_dialect();
        let mut request = Sq8ServingRequest::new(
            "req-reasoning",
            vec![1, 2, 151_667],
            max_new_tokens,
            Sq8SamplingParams {
                temperature,
                top_p: 1.0,
                top_k: QWEN3_14B_SQ8_SERVING_TOP_K,
                seed: 9,
            },
        );
        request.reasoning = Some(crate::reasoning::ReasoningExecution {
            enabled,
            budget_tokens,
            dialect_id: dialect.identity.clone(),
            end_sequence: dialect.end_sequence.clone(),
            forced_end_sequence: dialect.forced_end_sequence.clone(),
            reserved_answer_tokens: dialect.reserved_answer_tokens,
        });
        request.validate().unwrap();
        let mut active = ActiveServingRequest::new_with_reasoning_dialect(
            request,
            Sq8CancellationToken::new(),
            Some(&dialect),
        )
        .unwrap();
        active.prompt_tokens_processed = active.request.prompt_token_ids.len();
        active
    }

    fn commit_cpu_reasoning_transition(
        active: &mut ActiveServingRequest,
        token_id: usize,
        forced: bool,
        reasoning_after: ReasoningState,
    ) {
        active.generated_tokens += 1;
        if !forced {
            active.sampled_tokens += 1;
        }
        active.last_generated_token = Some(token_id);
        active.reasoning = Some(reasoning_after);
        validate_active_sampling_progress(active).unwrap();
    }

    struct ServingTokenTransactionFixture {
        state: Sq8ServingRuntimeStatus,
        pending_token: Option<PendingServingToken>,
        active: Option<ActiveServingRequest>,
        scheduler: SchedulerState,
        prepared: Sq8PreparedToken,
        cancel: Sq8CancellationToken,
    }

    fn serving_token_transaction_fixture() -> ServingTokenTransactionFixture {
        let cancel = Sq8CancellationToken::new();
        let request = Sq8ServingRequest::new(
            "req-transaction",
            vec![1, 2, 3],
            2,
            Sq8SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: QWEN3_14B_SQ8_SERVING_TOP_K,
                seed: 9,
            },
        );
        let mut active = ActiveServingRequest::new(request, cancel.clone());
        active.prompt_tokens_processed = 3;
        let logits = (0..QWEN3_14B_SQ8_SERVING_TOP_K)
            .map(|token_id| token_id as f32 / 10.0)
            .collect::<Vec<_>>();
        let proposal = active
            .sampler
            .propose(
                &logits,
                active.request.sampling.temperature,
                active.request.sampling.top_k,
                active.request.sampling.top_p,
            )
            .unwrap();
        let sampled = proposal.sampled();
        let prepared = Sq8PreparedToken {
            token_id: sampled.token_id,
            generated_index: 0,
            cache_len: 3,
            terminal_reason: None,
            nonce: 7,
        };
        let pending_token = Some(PendingServingToken {
            prepared: prepared.clone(),
            source: PendingTokenSource::Sampled(proposal),
            reasoning_after: None,
            commit: GeneratedTokenCommit::Prefill,
        });
        let mut scheduler = SchedulerState::with_block_size(4, 16);
        scheduler
            .activate_single_request_with_all_blocks(Request::new(1, 3, 2))
            .unwrap();
        scheduler
            .advance_prefill_tokens(SERVING_INTERNAL_REQUEST_ID, 3)
            .unwrap();
        ServingTokenTransactionFixture {
            state: Sq8ServingRuntimeStatus::TokenPrepared,
            pending_token,
            active: Some(active),
            scheduler,
            prepared,
            cancel,
        }
    }

    fn eos_forced_token_transaction_fixture() -> ServingTokenTransactionFixture {
        let active = reasoning_active(true, None, 4, 1.0);
        let cancel = active.cancel.clone();
        let mut logits = vec![-100.0; QWEN3_14B_VOCAB_SIZE];
        let eos = QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS[0];
        logits[eos] = 100.0;
        let proposal = active
            .sampler
            .propose(
                &logits,
                active.request.sampling.temperature,
                active.request.sampling.top_k,
                active.request.sampling.top_p,
            )
            .unwrap();
        assert!(proposal.consumes_rng());
        assert_eq!(proposal.sampled().token_id, eos);
        let (token_id, forced, reasoning_after) = active.sampled_reasoning_transition(eos).unwrap();
        assert!(forced);
        assert_eq!(token_id, 151_668);
        let prepared = Sq8PreparedToken {
            token_id,
            generated_index: 0,
            cache_len: active.request.prompt_token_ids.len(),
            terminal_reason: None,
            nonce: 8,
        };
        let pending_token = Some(PendingServingToken {
            prepared: prepared.clone(),
            source: PendingTokenSource::ReasoningForced,
            reasoning_after,
            commit: GeneratedTokenCommit::Prefill,
        });
        let mut scheduler = SchedulerState::with_block_size(4, 16);
        scheduler
            .activate_single_request_with_all_blocks(Request::new(1, 3, 4))
            .unwrap();
        scheduler
            .advance_prefill_tokens(SERVING_INTERNAL_REQUEST_ID, 3)
            .unwrap();
        ServingTokenTransactionFixture {
            state: Sq8ServingRuntimeStatus::TokenPrepared,
            pending_token,
            active: Some(active),
            scheduler,
            prepared,
            cancel,
        }
    }

    fn assert_uncommitted_transaction(fixture: &ServingTokenTransactionFixture) {
        let active = fixture.active.as_ref().unwrap();
        assert_eq!(active.generated_tokens, 0);
        assert_eq!(active.sampler.draws(), 0);
        assert_eq!(active.last_generated_token, None);
        assert_eq!(active.finish_reason, None);
        assert_eq!(
            fixture
                .scheduler
                .active_request(SERVING_INTERNAL_REQUEST_ID)
                .unwrap()
                .generated_tokens,
            0
        );
    }

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
    fn serving_request_accepts_product_stochastic_sampling() {
        let sampling = Sq8SamplingParams {
            temperature: 0.6,
            top_p: 0.95,
            top_k: QWEN3_14B_SQ8_SERVING_TOP_K,
            seed: -17,
        };
        let request = Sq8ServingRequest::new("req-sample", vec![1, 2, 3], 4, sampling);
        request.validate().unwrap();
        let active = ActiveServingRequest::new(request, Sq8CancellationToken::new());
        validate_active_sampling_progress(&active).unwrap();
    }

    #[test]
    fn qwen3_reasoning_contract_fixes_effort_budgets_and_token_ids() {
        let dialect = qwen3_reasoning_dialect();
        dialect.validate(QWEN3_14B_VOCAB_SIZE).unwrap();
        assert_eq!(dialect.identity, "qwen3-thinking-v1");
        assert_eq!(dialect.start_sequence, vec![151_667]);
        assert_eq!(dialect.end_sequence, vec![151_668]);
        assert_eq!(dialect.forced_end_sequence, vec![151_668]);
        assert_eq!(
            dialect.effort_budgets,
            vec![
                ("low".into(), 32),
                ("medium".into(), 128),
                ("high".into(), 256),
            ]
        );
    }

    #[test]
    fn reasoning_request_and_loaded_dialect_must_be_present_together() {
        let dialect = qwen3_reasoning_dialect();
        let plain = Sq8ServingRequest::greedy("req-plain", vec![1], 1);
        assert!(
            ActiveServingRequest::new_with_reasoning_dialect(
                plain,
                Sq8CancellationToken::new(),
                Some(&dialect),
            )
            .is_err()
        );

        let with_reasoning = reasoning_active(true, Some(0), 2, 0.0).request;
        assert!(
            ActiveServingRequest::new_with_reasoning_dialect(
                with_reasoning,
                Sq8CancellationToken::new(),
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn serving_v2_rejects_multi_token_loaded_and_request_sequences() {
        for field in [
            "loaded_start",
            "loaded_end",
            "loaded_forced_end",
            "request_end",
            "request_forced_end",
        ] {
            let mut dialect = qwen3_reasoning_dialect();
            let mut request = reasoning_active(true, Some(0), 2, 0.0).request;
            match field {
                "loaded_start" => dialect.start_sequence.push(1),
                "loaded_end" => {
                    dialect.end_sequence.push(1);
                    request.reasoning.as_mut().unwrap().end_sequence.push(1);
                }
                "loaded_forced_end" => {
                    dialect.forced_end_sequence.push(1);
                    request
                        .reasoning
                        .as_mut()
                        .unwrap()
                        .forced_end_sequence
                        .push(1);
                }
                "request_end" => request.reasoning.as_mut().unwrap().end_sequence.push(1),
                "request_forced_end" => request
                    .reasoning
                    .as_mut()
                    .unwrap()
                    .forced_end_sequence
                    .push(1),
                _ => unreachable!(),
            }
            let error = ActiveServingRequest::new_with_reasoning_dialect(
                request,
                Sq8CancellationToken::new(),
                Some(&dialect),
            )
            .unwrap_err();
            assert!(error.contains("exactly one token"), "field={field}");
        }
    }

    #[test]
    fn reasoning_disabled_retains_v2_zero_usage_accounting() {
        let mut active = reasoning_active(false, None, 1, 0.0);
        assert_eq!(
            active.reasoning.as_ref().unwrap().phase,
            ReasoningPhase::Disabled
        );
        let (token_id, forced, reasoning_after) = active.sampled_reasoning_transition(42).unwrap();
        assert_eq!(token_id, 42);
        assert!(!forced);
        commit_cpu_reasoning_transition(&mut active, token_id, forced, reasoning_after.unwrap());
        assert_eq!(
            active.reasoning_usage(),
            Some(ReasoningUsage {
                reasoning_tokens: 0,
                forced_end_tokens: 0,
            })
        );
    }

    #[test]
    fn bounded_reasoning_budgets_close_exactly_at_zero_low_medium_and_high() {
        for budget in [0_usize, 32, 128, 256] {
            let mut active = reasoning_active(true, Some(budget), budget + 2, 0.0);
            for _ in 0..budget {
                let (token_id, forced, reasoning_after) =
                    active.sampled_reasoning_transition(42).unwrap();
                assert_eq!(token_id, 42, "budget={budget}");
                assert!(!forced, "budget={budget}");
                commit_cpu_reasoning_transition(
                    &mut active,
                    token_id,
                    forced,
                    reasoning_after.unwrap(),
                );
            }
            assert_eq!(
                active.reasoning.as_ref().unwrap().phase,
                ReasoningPhase::ForcingEndSequence,
                "budget={budget}"
            );
            let (token_id, reasoning_after) = active
                .forced_reasoning_transition()
                .unwrap()
                .expect("budget close token");
            assert_eq!(token_id, 151_668);
            commit_cpu_reasoning_transition(&mut active, token_id, true, reasoning_after);
            assert_eq!(
                active.reasoning_usage(),
                Some(ReasoningUsage {
                    reasoning_tokens: budget,
                    forced_end_tokens: 1,
                }),
                "budget={budget}"
            );
            assert_eq!(
                active.reasoning.as_ref().unwrap().phase,
                ReasoningPhase::Answer
            );
            assert_eq!(active.request.max_new_tokens - active.generated_tokens, 1);
        }
    }

    #[test]
    fn unbounded_reasoning_natural_close_is_sampled_and_not_forced() {
        let mut active = reasoning_active(true, None, 6, 0.0);
        for token_id in [42, 151_668] {
            let (selected, forced, reasoning_after) =
                active.sampled_reasoning_transition(token_id).unwrap();
            assert_eq!(selected, token_id);
            assert!(!forced);
            commit_cpu_reasoning_transition(
                &mut active,
                selected,
                forced,
                reasoning_after.unwrap(),
            );
        }
        assert_eq!(
            active.reasoning_usage(),
            Some(ReasoningUsage {
                reasoning_tokens: 1,
                forced_end_tokens: 0,
            })
        );
        assert_eq!(
            active.reasoning.as_ref().unwrap().phase,
            ReasoningPhase::Answer
        );
    }

    #[test]
    fn unbounded_reasoning_reserves_forced_close_and_one_answer_token() {
        let mut active = reasoning_active(true, None, 3, 0.0);
        let (token_id, forced, reasoning_after) = active.sampled_reasoning_transition(42).unwrap();
        assert!(!forced);
        commit_cpu_reasoning_transition(&mut active, token_id, forced, reasoning_after.unwrap());

        let (forced_token, reasoning_after) = active
            .forced_reasoning_transition()
            .unwrap()
            .expect("length guard must close reasoning");
        assert_eq!(forced_token, 151_668);
        commit_cpu_reasoning_transition(&mut active, forced_token, true, reasoning_after);
        assert_eq!(active.request.max_new_tokens - active.generated_tokens, 1);

        let (answer_token, forced, reasoning_after) =
            active.sampled_reasoning_transition(77).unwrap();
        assert!(!forced);
        assert_eq!(
            active.terminal_reason_after(answer_token, reasoning_after.as_ref()),
            Some(Sq8FinishReason::Length)
        );
        commit_cpu_reasoning_transition(
            &mut active,
            answer_token,
            forced,
            reasoning_after.unwrap(),
        );
        assert_eq!(
            active.reasoning_usage(),
            Some(ReasoningUsage {
                reasoning_tokens: 1,
                forced_end_tokens: 1,
            })
        );
    }

    #[test]
    fn reasoning_eos_is_replaced_by_forced_close_without_consuming_rng() {
        let mut fixture = eos_forced_token_transaction_fixture();
        let eos = QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS[0];
        assert_ne!(fixture.prepared.token_id, eos);
        let result = publish_prepared_token_transaction(
            &mut fixture.state,
            &mut fixture.pending_token,
            &mut fixture.active,
            &mut fixture.scheduler,
            &fixture.cancel,
            &fixture.prepared,
            |_| Ok(()),
        )
        .unwrap();
        assert!(matches!(
            result,
            Sq8ServingAdvance::Token {
                token_id: 151_668,
                terminal_reason: None,
                ..
            }
        ));
        let active = fixture.active.as_ref().unwrap();
        assert_eq!(active.generated_tokens, 1);
        assert_eq!(active.sampled_tokens, 0);
        assert_eq!(active.teacher_forced_tokens, 0);
        assert_eq!(active.sampler.draws(), 0);
        assert_eq!(
            active.reasoning_usage(),
            Some(ReasoningUsage {
                reasoning_tokens: 0,
                forced_end_tokens: 1,
            })
        );
        assert_eq!(
            active.reasoning.as_ref().unwrap().phase,
            ReasoningPhase::Answer
        );
    }

    #[test]
    fn teacher_forced_capture_token_does_not_mutate_reasoning_accounting() {
        let mut fixture = serving_token_transaction_fixture();
        fixture.pending_token.as_mut().unwrap().source = PendingTokenSource::TeacherForced;

        let result = publish_prepared_token_transaction(
            &mut fixture.state,
            &mut fixture.pending_token,
            &mut fixture.active,
            &mut fixture.scheduler,
            &fixture.cancel,
            &fixture.prepared,
            |_| Ok(()),
        )
        .unwrap();

        assert!(matches!(result, Sq8ServingAdvance::Token { .. }));
        let active = fixture.active.as_ref().unwrap();
        assert_eq!(active.generated_tokens, 1);
        assert_eq!(active.sampled_tokens, 0);
        assert_eq!(active.teacher_forced_tokens, 1);
        assert_eq!(active.sampler.draws(), 0);
        assert!(active.reasoning.is_none());
    }

    #[test]
    fn reasoning_cancel_and_publication_failure_leave_committed_accounting_unchanged() {
        let mut cancelled = eos_forced_token_transaction_fixture();
        cancelled.cancel.cancel_checked().unwrap();
        let result = publish_prepared_token_transaction(
            &mut cancelled.state,
            &mut cancelled.pending_token,
            &mut cancelled.active,
            &mut cancelled.scheduler,
            &cancelled.cancel,
            &cancelled.prepared,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(result, Sq8ServingAdvance::CancellationObserved);
        let active = cancelled.active.as_ref().unwrap();
        assert_eq!(active.generated_tokens, 0);
        assert_eq!(active.sampled_tokens, 0);
        assert_eq!(active.sampler.draws(), 0);
        assert_eq!(
            active.reasoning_usage(),
            Some(ReasoningUsage {
                reasoning_tokens: 0,
                forced_end_tokens: 0,
            })
        );
        assert_eq!(
            active.reasoning.as_ref().unwrap().phase,
            ReasoningPhase::Reasoning
        );

        let mut failed = eos_forced_token_transaction_fixture();
        assert!(
            publish_prepared_token_transaction(
                &mut failed.state,
                &mut failed.pending_token,
                &mut failed.active,
                &mut failed.scheduler,
                &failed.cancel,
                &failed.prepared,
                |_| Err("flush failed".into()),
            )
            .is_err()
        );
        let active = failed.active.as_ref().unwrap();
        assert_eq!(active.generated_tokens, 0);
        assert_eq!(active.sampled_tokens, 0);
        assert_eq!(active.sampler.draws(), 0);
        assert_eq!(
            active.reasoning_usage(),
            Some(ReasoningUsage {
                reasoning_tokens: 0,
                forced_end_tokens: 0,
            })
        );
        assert_eq!(
            active.reasoning.as_ref().unwrap().phase,
            ReasoningPhase::Reasoning
        );
    }

    #[test]
    fn reasoning_release_summary_keeps_committed_usage_after_active_state_is_cleared() {
        let mut active = reasoning_active(true, Some(0), 2, 0.0);
        let (forced_token, reasoning_after) = active
            .forced_reasoning_transition()
            .unwrap()
            .expect("zero budget forces the close token");
        commit_cpu_reasoning_transition(&mut active, forced_token, true, reasoning_after);
        let release = active.snapshot_release_accounting(Sq8ReleaseOutcome::Cancelled);
        drop(active);

        assert_eq!(
            release.complete_after_reset(),
            Sq8ReleaseSummary {
                request_id: "req-reasoning".into(),
                outcome: Sq8ReleaseOutcome::Cancelled,
                prompt_tokens: 3,
                generated_tokens: 1,
                reasoning_usage: Some(ReasoningUsage {
                    reasoning_tokens: 0,
                    forced_end_tokens: 1,
                }),
                reset_complete: true,
            }
        );
    }

    #[test]
    fn reasoning_reuse_starts_with_zero_accounting_after_prior_release() {
        let mut first = reasoning_active(true, Some(0), 2, 0.0);
        let (forced_token, reasoning_after) = first
            .forced_reasoning_transition()
            .unwrap()
            .expect("zero budget forces the close token");
        commit_cpu_reasoning_transition(&mut first, forced_token, true, reasoning_after);
        let prior_release = first
            .snapshot_release_accounting(Sq8ReleaseOutcome::Cancelled)
            .complete_after_reset();
        assert_eq!(
            prior_release.reasoning_usage,
            Some(ReasoningUsage {
                reasoning_tokens: 0,
                forced_end_tokens: 1,
            })
        );

        let reused = reasoning_active(false, None, 1, 0.0);
        assert_eq!(reused.generated_tokens, 0);
        assert_eq!(reused.sampled_tokens, 0);
        assert_eq!(reused.sampler.draws(), 0);
        assert_eq!(
            reused.reasoning_usage(),
            Some(ReasoningUsage {
                reasoning_tokens: 0,
                forced_end_tokens: 0,
            })
        );
        validate_active_sampling_progress(&reused).unwrap();
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
    fn serving_publication_discards_cancelled_token_without_side_effects() {
        let token = Sq8CancellationToken::new();
        token.cancel_checked().unwrap();
        let mut published = false;
        let mut committed = false;

        let result = linearize_token_publication(
            &token,
            || {
                published = true;
                Ok(())
            },
            || {
                committed = true;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(result, TokenPublication::Cancelled);
        assert!(!published);
        assert!(!committed);
    }

    #[test]
    fn serving_publication_failure_does_not_commit_token() {
        let token = Sq8CancellationToken::new();
        let mut committed = false;

        let err = linearize_token_publication(
            &token,
            || Err("flush failed".into()),
            || {
                committed = true;
                Ok(())
            },
        )
        .unwrap_err();

        assert!(err.contains("publisher failed before commit"), "{err}");
        assert!(!committed);
        assert!(!token.is_cancelled());
    }

    #[test]
    fn serving_transaction_discards_cancelled_pending_token_without_progress() {
        let mut fixture = serving_token_transaction_fixture();
        fixture.cancel.cancel_checked().unwrap();
        let mut published = false;

        let result = publish_prepared_token_transaction(
            &mut fixture.state,
            &mut fixture.pending_token,
            &mut fixture.active,
            &mut fixture.scheduler,
            &fixture.cancel,
            &fixture.prepared,
            |_| {
                published = true;
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(result, Sq8ServingAdvance::CancellationObserved);
        assert_eq!(fixture.state, Sq8ServingRuntimeStatus::Cancelling);
        assert!(fixture.pending_token.is_none());
        assert!(!published);
        assert_uncommitted_transaction(&fixture);
    }

    #[test]
    fn serving_transaction_publisher_failure_is_fatal_and_uncommitted() {
        let mut fixture = serving_token_transaction_fixture();

        let err = publish_prepared_token_transaction(
            &mut fixture.state,
            &mut fixture.pending_token,
            &mut fixture.active,
            &mut fixture.scheduler,
            &fixture.cancel,
            &fixture.prepared,
            |_| Err("flush failed".into()),
        )
        .unwrap_err();

        assert!(err.contains("publisher failed before commit"), "{err}");
        assert_eq!(fixture.state, Sq8ServingRuntimeStatus::Failed);
        assert!(fixture.pending_token.is_some());
        assert_uncommitted_transaction(&fixture);
    }

    #[test]
    fn serving_transaction_commit_failure_after_publish_is_fatal() {
        let mut fixture = serving_token_transaction_fixture();
        let mut stale_scheduler = SchedulerState::with_block_size(4, 16);
        stale_scheduler
            .activate_single_request_with_all_blocks(Request::new(1, 3, 2))
            .unwrap();
        fixture.scheduler = stale_scheduler;
        let mut published = false;

        let err = publish_prepared_token_transaction(
            &mut fixture.state,
            &mut fixture.pending_token,
            &mut fixture.active,
            &mut fixture.scheduler,
            &fixture.cancel,
            &fixture.prepared,
            |_| {
                published = true;
                Ok(())
            },
        )
        .unwrap_err();

        assert!(published);
        assert!(err.contains("prefill"), "{err}");
        assert_eq!(fixture.state, Sq8ServingRuntimeStatus::Failed);
        assert!(fixture.pending_token.is_none());
        assert_uncommitted_transaction(&fixture);
    }

    #[test]
    fn serving_transaction_terminal_cancel_split_is_unambiguous() {
        let mut cancelled = serving_token_transaction_fixture();
        cancelled.prepared.terminal_reason = Some(Sq8FinishReason::Length);
        cancelled
            .pending_token
            .as_mut()
            .unwrap()
            .prepared
            .terminal_reason = Some(Sq8FinishReason::Length);
        cancelled.cancel.cancel_checked().unwrap();
        let result = publish_prepared_token_transaction(
            &mut cancelled.state,
            &mut cancelled.pending_token,
            &mut cancelled.active,
            &mut cancelled.scheduler,
            &cancelled.cancel,
            &cancelled.prepared,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(result, Sq8ServingAdvance::CancellationObserved);
        assert_eq!(cancelled.state, Sq8ServingRuntimeStatus::Cancelling);
        assert_uncommitted_transaction(&cancelled);

        let mut published = serving_token_transaction_fixture();
        published.prepared.terminal_reason = Some(Sq8FinishReason::Length);
        published
            .pending_token
            .as_mut()
            .unwrap()
            .prepared
            .terminal_reason = Some(Sq8FinishReason::Length);
        let result = publish_prepared_token_transaction(
            &mut published.state,
            &mut published.pending_token,
            &mut published.active,
            &mut published.scheduler,
            &published.cancel,
            &published.prepared,
            |_| Ok(()),
        )
        .unwrap();
        published.cancel.cancel_checked().unwrap();
        assert!(matches!(
            result,
            Sq8ServingAdvance::Token {
                terminal_reason: Some(Sq8FinishReason::Length),
                ..
            }
        ));
        assert_eq!(published.state, Sq8ServingRuntimeStatus::Finishing);
        assert_eq!(published.active.as_ref().unwrap().generated_tokens, 1);
        assert_eq!(published.active.as_ref().unwrap().sampler.draws(), 1);
        assert_eq!(
            published
                .scheduler
                .active_request(SERVING_INTERNAL_REQUEST_ID)
                .unwrap()
                .generated_tokens,
            1
        );
    }

    #[test]
    fn serving_cancel_waits_for_transaction_publish_and_commit() {
        let fixture = serving_token_transaction_fixture();
        let token = fixture.cancel.clone();
        let publication_token = token.clone();
        let (publish_entered_tx, publish_entered_rx) = mpsc::channel();
        let (publish_release_tx, publish_release_rx) = mpsc::channel();
        let (publication_done_tx, publication_done_rx) = mpsc::channel();
        let publication_thread = std::thread::spawn(move || {
            let mut fixture = fixture;
            let result = publish_prepared_token_transaction(
                &mut fixture.state,
                &mut fixture.pending_token,
                &mut fixture.active,
                &mut fixture.scheduler,
                &publication_token,
                &fixture.prepared,
                |_| {
                    publish_entered_tx.send(()).unwrap();
                    publish_release_rx.recv().unwrap();
                    Ok(())
                },
            );
            publication_done_tx.send((result, fixture)).unwrap();
        });
        publish_entered_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert!(token.publication_is_locked().unwrap());

        let cancel = token.clone();
        let (cancel_attempted_tx, cancel_attempted_rx) = mpsc::channel();
        let (cancel_done_tx, cancel_done_rx) = mpsc::channel();
        let cancel_thread = std::thread::spawn(move || {
            cancel_attempted_tx.send(()).unwrap();
            cancel.cancel_checked().unwrap();
            cancel_done_tx.send(()).unwrap();
        });
        cancel_attempted_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        assert!(token.publication_is_locked().unwrap());
        assert!(!token.is_cancelled());
        publish_release_tx.send(()).unwrap();
        let (result, fixture) = publication_done_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        let result = result.unwrap();
        assert_eq!(
            result,
            Sq8ServingAdvance::Token {
                token_id: fixture.prepared.token_id,
                generated_index: 0,
                cache_len: 3,
                terminal_reason: None,
            }
        );
        assert_eq!(fixture.state, Sq8ServingRuntimeStatus::Decoding);
        assert!(fixture.pending_token.is_none());
        let active = fixture.active.as_ref().unwrap();
        assert_eq!(active.generated_tokens, 1);
        assert_eq!(active.sampler.draws(), 1);
        assert_eq!(active.last_generated_token, Some(fixture.prepared.token_id));
        assert_eq!(
            fixture
                .scheduler
                .active_request(SERVING_INTERNAL_REQUEST_ID)
                .unwrap()
                .generated_tokens,
            1
        );
        cancel_done_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        publication_thread.join().unwrap();
        cancel_thread.join().unwrap();
        assert!(token.is_cancelled());
    }

    #[test]
    fn serving_sampling_progress_tracks_only_committed_stochastic_tokens() {
        let request = Sq8ServingRequest::new(
            "req-sample",
            vec![1, 2, 3],
            2,
            Sq8SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: QWEN3_14B_SQ8_SERVING_TOP_K,
                seed: 9,
            },
        );
        let mut active = ActiveServingRequest::new(request, Sq8CancellationToken::new());
        let logits = vec![0.0; QWEN3_14B_SQ8_SERVING_TOP_K];
        let proposal = active
            .sampler
            .propose(
                &logits,
                active.request.sampling.temperature,
                active.request.sampling.top_k,
                active.request.sampling.top_p,
            )
            .unwrap();
        active.sampler.commit(proposal).unwrap();
        assert!(validate_active_sampling_progress(&active).is_err());
        active.generated_tokens = 1;
        active.sampled_tokens = 1;
        validate_active_sampling_progress(&active).unwrap();
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
        assert_eq!(
            qwen3_14b_sq8_serving_prompt_chunk_bytes(8).unwrap(),
            163_840
        );
        assert_eq!(
            qwen3_14b_sq8_serving_prompt_chunk_bytes(128).unwrap(),
            2_621_440
        );
    }

    #[test]
    fn serving_prefill_planner_uses_real_token_overlap_for_m8_tails() {
        for prompt_tokens in [1, 7, 8, 9, 15, 16, 17, 32, 128, 512, 4095] {
            let units =
                plan_prefill_units(prompt_tokens, Sq8ServingPrefillMode::FixedM8Chunks).unwrap();
            let chunk_tokens = QWEN3_14B_SQ8_PREFILL_CHUNK_TOKENS;
            let expected_chunks = prompt_tokens / chunk_tokens
                + usize::from(
                    prompt_tokens >= chunk_tokens && !prompt_tokens.is_multiple_of(chunk_tokens),
                );
            let expected_m1 = if prompt_tokens < chunk_tokens {
                prompt_tokens
            } else {
                0
            };
            assert_eq!(
                units
                    .iter()
                    .filter(|unit| unit.execution_width == chunk_tokens)
                    .count(),
                expected_chunks,
                "prompt={prompt_tokens}"
            );
            assert_eq!(
                units
                    .iter()
                    .filter(|unit| unit.execution_width == 1)
                    .count(),
                expected_m1,
                "prompt={prompt_tokens}"
            );
            let mut expected_logical_position = 0_usize;
            for (index, unit) in units.iter().enumerate() {
                assert_eq!(unit.logical_start_position, expected_logical_position);
                assert!(matches!(unit.execution_width, 1 | 8));
                assert!(unit.execution_start_position <= unit.logical_start_position);
                assert_eq!(unit.execution_end().unwrap(), unit.logical_end().unwrap());
                expected_logical_position = unit.logical_end().unwrap();
                assert_eq!(unit.is_final, index + 1 == units.len());
            }
            assert_eq!(expected_logical_position, prompt_tokens);
        }

        let deepest = plan_prefill_units(4095, Sq8ServingPrefillMode::FixedM8Chunks).unwrap();
        assert_eq!(deepest.len(), 512);
        assert!(deepest.iter().all(|unit| unit.execution_width == 8));
        let tail = deepest.last().unwrap();
        assert_eq!(tail.logical_start_position, 4088);
        assert_eq!(tail.execution_start_position, 4087);
        assert_eq!(tail.execution_width, 8);
        assert_eq!(tail.committed_tokens, 7);
        assert!(tail.rewinds_cache());
        assert!(deepest.last().unwrap().is_final);
    }

    #[test]
    fn serving_prefill_planner_covers_m32_and_m128_boundaries_without_m1_tail() {
        for (mode, chunk_tokens) in [
            (Sq8ServingPrefillMode::FixedM32Chunks, 32_usize),
            (Sq8ServingPrefillMode::FixedM128Chunks, 128_usize),
        ] {
            assert_eq!(mode.chunk_tokens(), Some(chunk_tokens));
            assert_eq!(mode.resident_stack_width(), chunk_tokens);
            for prompt_tokens in [31, 32, 33, 127, 128, 129, 255, 256, 257, 4095] {
                let units = plan_prefill_units(prompt_tokens, mode).unwrap();
                let chunks = prompt_tokens / chunk_tokens
                    + usize::from(
                        prompt_tokens >= chunk_tokens
                            && !prompt_tokens.is_multiple_of(chunk_tokens),
                    );
                let m1_units = if prompt_tokens < chunk_tokens {
                    prompt_tokens
                } else {
                    0
                };
                assert_eq!(
                    units
                        .iter()
                        .filter(|unit| unit.execution_width == chunk_tokens)
                        .count(),
                    chunks,
                    "mode={mode:?} prompt={prompt_tokens}"
                );
                assert_eq!(
                    units
                        .iter()
                        .filter(|unit| unit.execution_width == 1)
                        .count(),
                    m1_units,
                    "mode={mode:?} prompt={prompt_tokens}"
                );
                assert_eq!(units.len(), chunks + m1_units);
                let mut expected_logical_position = 0_usize;
                for (index, unit) in units.iter().enumerate() {
                    assert_eq!(unit.logical_start_position, expected_logical_position);
                    assert!(unit.execution_width == 1 || unit.execution_width == chunk_tokens);
                    assert!(unit.execution_start_position <= unit.logical_start_position);
                    assert_eq!(unit.execution_end().unwrap(), unit.logical_end().unwrap());
                    expected_logical_position = unit.logical_end().unwrap();
                    assert_eq!(unit.is_final, index + 1 == units.len());
                }
                assert_eq!(expected_logical_position, prompt_tokens);
            }
        }
    }

    #[test]
    fn serving_m128_overlap_tail_geometry_and_divisible_geometry_are_explicit() {
        for prompt_tokens in [128, 512, 1024, 2048] {
            let units =
                plan_prefill_units(prompt_tokens, Sq8ServingPrefillMode::FixedM128Chunks).unwrap();
            assert_eq!(units.len(), prompt_tokens / 128);
            for (index, unit) in units.iter().enumerate() {
                assert_eq!(unit.logical_start_position, index * 128);
                assert_eq!(unit.execution_start_position, index * 128);
                assert_eq!(unit.execution_width, 128);
                assert_eq!(unit.committed_tokens, 128);
                assert!(!unit.rewinds_cache());
            }
        }

        for (prompt_tokens, logical_start, execution_start, committed_tokens) in [
            (129, 128, 1, 1),
            (1000, 896, 872, 104),
            (4095, 3968, 3967, 127),
        ] {
            let units =
                plan_prefill_units(prompt_tokens, Sq8ServingPrefillMode::FixedM128Chunks).unwrap();
            let tail = units.last().unwrap();
            assert_eq!(tail.logical_start_position, logical_start);
            assert_eq!(tail.execution_start_position, execution_start);
            assert_eq!(tail.execution_width, 128);
            assert_eq!(tail.committed_tokens, committed_tokens);
            assert_eq!(tail.execution_end().unwrap(), prompt_tokens);
            assert!(tail.rewinds_cache());
            assert!(units.iter().all(|unit| unit.execution_width == 128));
        }
    }

    #[test]
    fn serving_prefill_modes_bind_fixed_resident_widths_and_implementation_ids() {
        for (mode, resident_width, execution_width, implementation) in [
            (
                Sq8ServingPrefillMode::Adaptive,
                128,
                128,
                SQ8_ADAPTIVE_PREFILL_IMPLEMENTATION,
            ),
            (
                Sq8ServingPrefillMode::SequentialM1,
                8,
                1,
                SQ8_SEQUENTIAL_M1_PREFILL_IMPLEMENTATION,
            ),
            (
                Sq8ServingPrefillMode::FixedM8Chunks,
                8,
                8,
                SQ8_FIXED_M8_PREFILL_IMPLEMENTATION,
            ),
            (
                Sq8ServingPrefillMode::FixedM32Chunks,
                32,
                32,
                SQ8_FIXED_M32_PREFILL_IMPLEMENTATION,
            ),
            (
                Sq8ServingPrefillMode::FixedM128Chunks,
                128,
                128,
                SQ8_FIXED_M128_PREFILL_IMPLEMENTATION,
            ),
        ] {
            assert_eq!(mode.resident_stack_width(), resident_width);
            assert_eq!(mode.execution_width(), execution_width);
            assert_eq!(mode.implementation_id(), implementation);
            assert_eq!(
                mode.uses_chunks(),
                mode != Sq8ServingPrefillMode::SequentialM1
            );
        }
        assert_eq!(QWEN3_14B_SQ8_SERVING_DEFAULT_PREFILL_MODE, Sq8ServingPrefillMode::Adaptive);
        assert_eq!(
            QWEN3_14B_SQ8_SERVING_DEFAULT_PREFILL_MODE.initial_resident_mode(),
            Sq8ServingPrefillMode::FixedM128Chunks
        );
    }

    #[test]
    fn serving_adaptive_prefill_selects_only_measured_empirical_winners() {
        let adaptive = Sq8ServingPrefillMode::Adaptive;
        for (prompt_tokens, expected_width) in [
            (1_usize, 128_usize),
            (128, 128),
            (256, 128),
            (511, 128),
            (512, 512),
            (1023, 512),
            (1024, 1024),
            (2047, 1024),
            (2048, 2048),
            (4095, 2048),
        ] {
            let selected = adaptive.selected_for_prompt_tokens(prompt_tokens).unwrap();
            assert_eq!(selected.chunk_tokens(), Some(expected_width), "N={prompt_tokens}");
            selected.validate_runtime_contract().unwrap();

            let units = plan_prefill_units(prompt_tokens, adaptive).unwrap();
            assert!(units.iter().all(|unit| {
                unit.execution_width == expected_width
                    || (prompt_tokens < expected_width && unit.execution_width == 1)
            }));
            assert!(units.iter().all(|unit| {
                unit.execution_start_position <= unit.logical_start_position
                    && unit.execution_end().unwrap() == unit.logical_end().unwrap()
            }));
            assert_eq!(units.last().unwrap().logical_end().unwrap(), prompt_tokens);
        }

        let tail = plan_prefill_units(4095, adaptive).unwrap();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].execution_width, 2048);
        assert_eq!(tail[1].execution_width, 2048);
        assert_eq!(tail[1].logical_start_position, 2048);
        assert_eq!(tail[1].execution_start_position, 2047);
        assert_eq!(tail[1].committed_tokens, 2047);
        assert!(tail[1].rewinds_cache());
        assert!(adaptive.selected_for_prompt_tokens(0).is_err());
        assert!(adaptive.selected_for_prompt_tokens(4097).is_err());
    }

    #[test]
    fn serving_wide_chunk_scheduler_preserves_real_token_tail_replay() {
        for (chunk_tokens, expected_units, expected_attention_calls) in [
            (128_usize, 32_usize, 1_280_usize),
            (256, 16, 640),
            (512, 8, 320),
            (1_024, 4, 160),
            (2_048, 2, 80),
        ] {
            let mode = Sq8ServingPrefillMode::fixed_chunk_tokens(chunk_tokens).unwrap();
            assert_eq!(mode.chunk_tokens(), Some(chunk_tokens));
            assert_eq!(mode.resident_stack_width(), chunk_tokens);
            assert_eq!(mode.execution_width(), chunk_tokens);
            assert_eq!(
                mode.implementation_id(),
                format!("sq8.fixed-m{chunk_tokens}-cached-prefix.v1")
            );

            let units = plan_prefill_units(4095, mode).unwrap();
            assert_eq!(units.len(), expected_units, "M={chunk_tokens}");
            assert!(
                units
                    .iter()
                    .all(|unit| unit.execution_width == chunk_tokens)
            );
            assert_eq!(
                units.len() * QWEN3_14B_SQ8_STACK_LAYERS,
                expected_attention_calls,
                "M={chunk_tokens}"
            );

            let tail = units.last().unwrap();
            assert_eq!(
                tail.logical_start_position,
                (4095 / chunk_tokens) * chunk_tokens
            );
            assert_eq!(tail.execution_start_position, 4095 - chunk_tokens);
            assert_eq!(tail.execution_width, chunk_tokens);
            assert_eq!(tail.committed_tokens, chunk_tokens - 1);
            assert_eq!(tail.execution_end().unwrap(), 4095);
            assert!(tail.rewinds_cache());

            mode.validate_runtime_contract().unwrap();
        }
    }

    #[test]
    fn serving_4096_chunk_is_rejected_because_no_4096_row_prompt_is_servable() {
        assert!(Sq8ServingPrefillMode::fixed_chunk_tokens(4096).is_err());
    }

    #[test]
    fn serving_fixed_chunk_selector_rejects_invalid_scheduler_widths() {
        for chunk_tokens in [0, 1, 3, 129, 3_000, 4_096, 4_097] {
            assert!(Sq8ServingPrefillMode::fixed_chunk_tokens(chunk_tokens).is_err());
        }
    }

    #[test]
    fn serving_sequential_prefill_planner_keeps_every_execution_m1() {
        for prompt_tokens in [1, 8, 17, 128] {
            let units =
                plan_prefill_units(prompt_tokens, Sq8ServingPrefillMode::SequentialM1).unwrap();
            assert_eq!(units.len(), prompt_tokens);
            assert!(units.iter().all(|unit| {
                unit.execution_width == 1
                    && unit.execution_start_position == unit.logical_start_position
                    && unit.committed_tokens == 1
            }));
            assert_eq!(
                units.last().unwrap().logical_start_position,
                prompt_tokens - 1
            );
            assert!(units.last().unwrap().is_final);
        }
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
    fn serving_test_only_ignore_eos_runs_until_the_length_cap() {
        let request =
            Sq8ServingRequest::greedy_ignore_eos_for_testing("deep-boundary", vec![1, 2, 3], 4);
        request.validate().unwrap();
        assert!(request.test_only_ignores_eos());
        let mut active = ActiveServingRequest::new(request, Sq8CancellationToken::new());
        active.prompt_tokens_processed = active.request.prompt_token_ids.len();

        for generated_tokens in 0..3 {
            active.generated_tokens = generated_tokens;
            assert_eq!(
                active.terminal_reason(QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS[0]),
                None
            );
        }
        active.generated_tokens = 3;
        assert_eq!(
            active.terminal_reason(QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS[1]),
            Some(Sq8FinishReason::Length)
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
