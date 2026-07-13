// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Strict bounded JSONL contracts for the private SQ8 resident worker protocol.

pub use crate::inference_api::GenerationTimings as Sq8WorkerTimings;
use crate::inference_api::{
    CancellationToken as Sq8CancellationToken, InferenceRequest as Sq8ServingRequest,
    ReasoningUsage, SamplingParams as Sq8SamplingParams,
};
use crate::sq8_model_head_runtime::QWEN3_14B_VOCAB_SIZE;
use crate::sq8_serving_runtime::{
    QWEN3_14B_SQ8_SERVING_ARTIFACT_CONTENT_SHA256, QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS,
    QWEN3_14B_SQ8_SERVING_MAX_NEW_TOKENS, QWEN3_14B_SQ8_SERVING_PACKAGE_MANIFEST_SHA256,
    QWEN3_14B_SQ8_SERVING_TOP_K,
};
use memchr::memchr;
use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::io::{BufRead, Write};
use std::sync::{Mutex, TryLockError};

pub const SQ8_WORKER_SCHEMA_VERSION: &str = "ullm.worker.v1";
pub const SQ8_WORKER_SCHEMA_VERSION_V2: &str = "ullm.worker.v2";
pub const SQ8_WORKER_MAX_RECORD_BYTES: usize = 4_194_304;
pub const SQ8_WORKER_MAX_JSON_DEPTH: usize = 16;
pub const SQ8_WORKER_MAX_ERROR_MESSAGE_BYTES: usize = 1024;
/// Maximum number of prompt tokens that one worker execution unit may cover.
///
/// This is a worker protocol bound rather than a model or quantization-specific
/// execution width. Backends may use any contiguous unit from one token through
/// this bound, while progress events remain coalesced at this cadence.
pub const MAX_WORKER_PROGRESS_EXECUTION_WIDTH: usize = 128;
pub const SQ8_WORKER_MODEL: &str = "ullm-qwen3-14b-sq8";
pub const SQ8_WORKER_MODEL_REVISION: &str = "9a283b4a5efbc09ce247e0ae5b02b744739e525a";
pub const SQ8_WORKER_DEVICE: &str = "gfx1201";
pub const SQ8_WORKER_EXECUTION_PROFILE: &str = "rdna4_w8a8_block_ck";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8WorkerProfile {
    pub worker_schema: String,
    pub model: String,
    pub model_revision: String,
    pub artifact_content_sha256: String,
    pub package_manifest_sha256: String,
    pub device: String,
    pub execution_profile: String,
    pub context_length: usize,
    pub max_new_tokens: usize,
    pub vocab_size: usize,
    pub eos_token_ids: Vec<usize>,
    pub top_k: usize,
    pub reasoning: Option<crate::reasoning::ReasoningDialect>,
}

pub fn configured_worker_profile() -> Sq8WorkerProfile {
    Sq8WorkerProfile::from_environment_with_defaults(&Sq8WorkerProfile::sq8_defaults())
}

impl Sq8WorkerProfile {
    pub fn sq8_defaults() -> Self {
        Self {
            worker_schema: SQ8_WORKER_SCHEMA_VERSION.to_string(),
            model: SQ8_WORKER_MODEL.to_string(),
            model_revision: SQ8_WORKER_MODEL_REVISION.to_string(),
            artifact_content_sha256: QWEN3_14B_SQ8_SERVING_ARTIFACT_CONTENT_SHA256.to_string(),
            package_manifest_sha256: QWEN3_14B_SQ8_SERVING_PACKAGE_MANIFEST_SHA256.to_string(),
            device: SQ8_WORKER_DEVICE.to_string(),
            execution_profile: SQ8_WORKER_EXECUTION_PROFILE.to_string(),
            context_length: QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS,
            max_new_tokens: QWEN3_14B_SQ8_SERVING_MAX_NEW_TOKENS,
            vocab_size: QWEN3_14B_VOCAB_SIZE,
            eos_token_ids: crate::sq8_serving_runtime::QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS.to_vec(),
            top_k: QWEN3_14B_SQ8_SERVING_TOP_K,
            reasoning: None,
        }
    }

    pub fn from_environment_with_defaults(defaults: &Self) -> Self {
        Self {
            worker_schema: env_string("ULLM_WORKER_SCHEMA_VERSION", &defaults.worker_schema),
            model: env_string("ULLM_MODEL_ID", &defaults.model),
            model_revision: env_string("ULLM_MODEL_REVISION", &defaults.model_revision),
            artifact_content_sha256: env_string(
                "ULLM_ARTIFACT_CONTENT_SHA256",
                &defaults.artifact_content_sha256,
            ),
            package_manifest_sha256: env_string(
                "ULLM_PACKAGE_MANIFEST_SHA256",
                &defaults.package_manifest_sha256,
            ),
            device: env_string("ULLM_DEVICE", &defaults.device),
            execution_profile: env_string("ULLM_EXECUTION_PROFILE", &defaults.execution_profile),
            context_length: env_usize("ULLM_MODEL_CONTEXT_LENGTH", defaults.context_length),
            max_new_tokens: env_usize("ULLM_MAX_NEW_TOKENS", defaults.max_new_tokens),
            vocab_size: env_usize("ULLM_VOCAB_SIZE", defaults.vocab_size),
            eos_token_ids: env_usize_csv("ULLM_EOS_TOKEN_IDS", &defaults.eos_token_ids),
            top_k: env_usize("ULLM_TOP_K", defaults.top_k),
            reasoning: defaults.reasoning.clone(),
        }
    }
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_usize_csv(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| {
            value
                .split(',')
                .map(str::parse)
                .collect::<Result<Vec<usize>, _>>()
                .ok()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

const DUPLICATE_KEY_ERROR: &str = "duplicate object key";
const JSON_DEPTH_ERROR: &str = "JSON nesting exceeds protocol limit";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sq8JsonlRead {
    Record(Vec<u8>),
    Oversized,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8JsonlFramingErrorKind {
    Io,
    UnterminatedRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8JsonlFramingError {
    pub kind: Sq8JsonlFramingErrorKind,
    pub message: String,
}

impl fmt::Display for Sq8JsonlFramingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Sq8JsonlFramingError {}

/// Reads LF-terminated records without allowing an input line to grow memory.
pub struct Sq8BoundedJsonlReader<R> {
    inner: R,
    record: Box<[u8]>,
    record_len: usize,
    max_payload_bytes: usize,
}

impl<R: BufRead> Sq8BoundedJsonlReader<R> {
    pub fn new(inner: R) -> Self {
        Self::with_max_payload_bytes(inner, SQ8_WORKER_MAX_RECORD_BYTES)
    }

    fn with_max_payload_bytes(inner: R, max_payload_bytes: usize) -> Self {
        let raw_limit = max_payload_bytes
            .checked_add(1)
            .expect("SQ8 JSONL payload limit must leave room for CR");
        Self {
            inner,
            record: vec![0_u8; raw_limit].into_boxed_slice(),
            record_len: 0,
            max_payload_bytes,
        }
    }

    pub fn next_record(&mut self) -> Result<Sq8JsonlRead, Sq8JsonlFramingError> {
        self.record_len = 0;
        let mut oversized = false;
        loop {
            let (consumed, found_lf) = {
                let available = self.inner.fill_buf().map_err(|_| Sq8JsonlFramingError {
                    kind: Sq8JsonlFramingErrorKind::Io,
                    message: "failed to read SQ8 worker stdin".into(),
                })?;
                if available.is_empty() {
                    if self.record_len == 0 && !oversized {
                        return Ok(Sq8JsonlRead::Eof);
                    }
                    return Err(Sq8JsonlFramingError {
                        kind: Sq8JsonlFramingErrorKind::UnterminatedRecord,
                        message: "SQ8 worker stdin ended inside an unterminated record".into(),
                    });
                }

                let lf = memchr(b'\n', available);
                let segment_len = lf.unwrap_or(available.len());
                if !oversized {
                    let remaining = self.record.len().saturating_sub(self.record_len);
                    let copied = remaining.min(segment_len);
                    self.record[self.record_len..self.record_len + copied]
                        .copy_from_slice(&available[..copied]);
                    self.record_len += copied;
                    if copied != segment_len {
                        oversized = true;
                    }
                }
                (segment_len + usize::from(lf.is_some()), lf.is_some())
            };
            self.inner.consume(consumed);

            if !found_lf {
                continue;
            }
            if oversized {
                return Ok(Sq8JsonlRead::Oversized);
            }
            let payload_len = if self.record_len > 0 && self.record[self.record_len - 1] == b'\r' {
                self.record_len - 1
            } else {
                self.record_len
            };
            if payload_len > self.max_payload_bytes {
                return Ok(Sq8JsonlRead::Oversized);
            }
            return Ok(Sq8JsonlRead::Record(self.record[..payload_len].to_vec()));
        }
    }

    #[cfg(test)]
    fn fixed_buffer_bytes(&self) -> usize {
        self.record.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8WorkerProtocolErrorKind {
    InvalidCommand,
    InvalidRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8WorkerProtocolError {
    pub kind: Sq8WorkerProtocolErrorKind,
    pub message: String,
}

impl Sq8WorkerProtocolError {
    fn invalid_command(message: impl Into<String>) -> Self {
        Self {
            kind: Sq8WorkerProtocolErrorKind::InvalidCommand,
            message: message.into(),
        }
    }

    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            kind: Sq8WorkerProtocolErrorKind::InvalidRequest,
            message: message.into(),
        }
    }
}

impl fmt::Display for Sq8WorkerProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Sq8WorkerProtocolError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sq8CancelReason {
    ClientDisconnect,
    SlowClient,
    Shutdown,
    Operator,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8GenerateCommand {
    pub request_id: String,
    pub prompt_token_ids: Vec<usize>,
    pub max_new_tokens: usize,
    pub sampling: Sq8WorkerSampling,
    pub eos_token_ids: Vec<usize>,
    pub reasoning: Option<crate::reasoning::ReasoningExecution>,
}

impl Sq8GenerateCommand {
    pub fn into_serving_request(self) -> Result<Sq8ServingRequest, Sq8WorkerProtocolError> {
        let profile = configured_worker_profile();
        self.into_serving_request_with_profile(&profile)
    }

    pub fn into_serving_request_with_profile(
        self,
        profile: &Sq8WorkerProfile,
    ) -> Result<Sq8ServingRequest, Sq8WorkerProtocolError> {
        if !self.sampling.temperature.is_finite()
            || !(0.0..=2.0).contains(&self.sampling.temperature)
            || !self.sampling.top_p.is_finite()
            || self.sampling.top_p <= 0.0
            || self.sampling.top_p > 1.0
            || self.sampling.top_k != profile.top_k
        {
            return Err(Sq8WorkerProtocolError::invalid_request(
                "generate sampling violates the fixed SQ8 product limits",
            ));
        }
        let sampling = Sq8SamplingParams {
            temperature: self.sampling.temperature as f32,
            top_p: self.sampling.top_p as f32,
            top_k: self.sampling.top_k,
            seed: self.sampling.seed,
        };
        let mut request = Sq8ServingRequest::new(
            self.request_id,
            self.prompt_token_ids,
            self.max_new_tokens,
            sampling,
        );
        request.eos_token_ids = self.eos_token_ids;
        request.reasoning = self.reasoning;
        if let Some(execution) = request.reasoning.as_ref() {
            let Some(dialect) = profile.reasoning.as_ref() else {
                return Err(Sq8WorkerProtocolError::invalid_request(
                    "reasoning execution is not declared by the loaded worker profile",
                ));
            };
            if execution.dialect_id != dialect.identity
                || execution.end_sequence != dialect.end_sequence
                || execution.forced_end_sequence != dialect.forced_end_sequence
                || execution.reserved_answer_tokens != dialect.reserved_answer_tokens
            {
                return Err(Sq8WorkerProtocolError::invalid_request(
                    "reasoning execution does not match the loaded worker profile",
                ));
            }
            dialect.validate(profile.vocab_size).map_err(|_| {
                Sq8WorkerProtocolError::invalid_request(
                    "loaded worker reasoning dialect is invalid",
                )
            })?;
            if execution
                .budget_tokens
                .is_some_and(|budget| budget > dialect.max_budget_tokens)
            {
                return Err(Sq8WorkerProtocolError::invalid_request(
                    "reasoning budget exceeds the loaded worker profile",
                ));
            }
        }
        request
            .validate_for_worker(
                profile.context_length,
                profile.max_new_tokens,
                profile.vocab_size,
                &profile.eos_token_ids,
                profile.top_k,
            )
            .map_err(|_| {
                Sq8WorkerProtocolError::invalid_request(
                    "generate request violates the fixed SQ8 product limits",
                )
            })?;
        Ok(request)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sq8WorkerSampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub seed: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Sq8WorkerCommand {
    Generate(Sq8GenerateCommand),
    Cancel {
        request_id: String,
        reason: Sq8CancelReason,
    },
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8WorkerCommandKind {
    Generate,
    Cancel,
    Shutdown,
}

pub struct Sq8WorkerCommandInspection<'a> {
    payload: &'a [u8],
    pub kind: Sq8WorkerCommandKind,
    request_id: Option<String>,
}

impl Sq8WorkerCommandInspection<'_> {
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn decode(self) -> Result<Sq8WorkerCommand, Sq8WorkerProtocolError> {
        let profile = configured_worker_profile();
        self.decode_with_profile(&profile)
    }

    pub fn decode_with_profile(
        self,
        profile: &Sq8WorkerProfile,
    ) -> Result<Sq8WorkerCommand, Sq8WorkerProtocolError> {
        decode_inspected_sq8_worker_command(self.payload, self.kind, profile.context_length)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8WorkerLifecycle {
    Loading,
    Ready,
    Closing,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Sq8WorkerAdmission {
    pub generation: u64,
    pub request_id: String,
    pub cancel: Sq8CancellationToken,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8WorkerControlSnapshot {
    pub lifecycle: Sq8WorkerLifecycle,
    pub active_generation: Option<u64>,
    pub active_request_id: Option<String>,
    pub first_cancel_reason: Option<Sq8CancelReason>,
    pub cancelled: bool,
    pub terminal_in_flight: bool,
}

#[derive(Debug)]
pub struct Sq8ActiveTerminalPermit {
    generation: u64,
    request_id: String,
}

#[derive(Debug)]
pub struct Sq8ActiveTerminalFlushAck {
    generation: u64,
    request_id: String,
}

#[derive(Debug)]
pub struct Sq8ReadyFlushAck {
    private: (),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8WorkerControlErrorKind {
    NotReady,
    Busy,
    UnknownRequest,
    Closing,
    Failed,
    StaleGeneration,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8WorkerControlError {
    pub kind: Sq8WorkerControlErrorKind,
    pub message: &'static str,
}

impl fmt::Display for Sq8WorkerControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl std::error::Error for Sq8WorkerControlError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sq8WorkerCancelResult {
    pub generation: u64,
    pub first_reason: Sq8CancelReason,
    pub repeated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8WorkerShutdownDisposition {
    Idle,
    Cancelling(Sq8WorkerCancelResult),
}

#[derive(Debug)]
struct Sq8ActiveControl {
    generation: u64,
    request_id: String,
    cancel: Sq8CancellationToken,
    first_cancel_reason: Option<Sq8CancelReason>,
    terminal_in_flight: bool,
}

#[derive(Debug)]
struct Sq8WorkerControlState {
    lifecycle: Sq8WorkerLifecycle,
    next_generation: u64,
    active: Option<Sq8ActiveControl>,
}

struct Sq8PreparedControlCancel {
    cancel: Sq8CancellationToken,
    result: Sq8WorkerCancelResult,
}

#[derive(Debug)]
pub struct Sq8WorkerControl {
    state: Mutex<Sq8WorkerControlState>,
}

impl Default for Sq8WorkerControl {
    fn default() -> Self {
        Self::new()
    }
}

impl Sq8WorkerControl {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(Sq8WorkerControlState {
                lifecycle: Sq8WorkerLifecycle::Loading,
                next_generation: 1,
                active: None,
            }),
        }
    }

    pub fn mark_ready_after_flush(
        &self,
        acknowledgement: Sq8ReadyFlushAck,
    ) -> Result<(), Sq8WorkerControlError> {
        let Sq8ReadyFlushAck { private: () } = acknowledgement;
        let mut state = self.lock_state()?;
        if state.lifecycle != Sq8WorkerLifecycle::Loading || state.active.is_some() {
            return Err(control_error(
                Sq8WorkerControlErrorKind::Internal,
                "worker readiness transition is invalid",
            ));
        }
        state.lifecycle = Sq8WorkerLifecycle::Ready;
        Ok(())
    }

    /// Runs before semantic generate validation so an occupied active slot always wins as busy.
    pub fn precheck_generate(&self) -> Result<(), Sq8WorkerControlError> {
        let state = self.lock_state()?;
        validate_generate_admission(&state)
    }

    /// Installs the sole active slot after complete reader-side request validation.
    pub fn admit(&self, request_id: &str) -> Result<Sq8WorkerAdmission, Sq8WorkerControlError> {
        validate_worker_request_id(request_id).map_err(|_| {
            control_error(
                Sq8WorkerControlErrorKind::Internal,
                "worker admission requires a prevalidated request ID",
            )
        })?;
        let mut state = self.lock_state()?;
        validate_generate_admission(&state)?;
        let generation = state.next_generation;
        state.next_generation = state.next_generation.checked_add(1).ok_or_else(|| {
            control_error(
                Sq8WorkerControlErrorKind::Internal,
                "worker request generation overflowed",
            )
        })?;
        let cancel = Sq8CancellationToken::new();
        state.active = Some(Sq8ActiveControl {
            generation,
            request_id: request_id.to_string(),
            cancel: cancel.clone(),
            first_cancel_reason: None,
            terminal_in_flight: false,
        });
        Ok(Sq8WorkerAdmission {
            generation,
            request_id: request_id.to_string(),
            cancel,
        })
    }

    pub fn cancel(
        &self,
        request_id: &str,
        reason: Sq8CancelReason,
    ) -> Result<Sq8WorkerCancelResult, Sq8WorkerControlError> {
        self.cancel_with_hook(request_id, reason, || {})
    }

    fn cancel_with_hook<F>(
        &self,
        request_id: &str,
        reason: Sq8CancelReason,
        before_publication_lock: F,
    ) -> Result<Sq8WorkerCancelResult, Sq8WorkerControlError>
    where
        F: FnOnce(),
    {
        let prepared = {
            let mut state = self.lock_state()?;
            if state.lifecycle == Sq8WorkerLifecycle::Failed {
                return Err(control_error(
                    Sq8WorkerControlErrorKind::Failed,
                    "worker control is failed",
                ));
            }
            if state.lifecycle == Sq8WorkerLifecycle::Closing {
                return Err(control_error(
                    Sq8WorkerControlErrorKind::Closing,
                    "worker input is closing",
                ));
            }
            if state.lifecycle == Sq8WorkerLifecycle::Loading {
                return Err(control_error(
                    Sq8WorkerControlErrorKind::NotReady,
                    "worker is not ready",
                ));
            }
            prepare_cancel_active(&mut state, request_id, reason)?
        };
        before_publication_lock();
        self.commit_prepared_cancel(prepared)
    }

    pub fn begin_shutdown(&self) -> Result<Sq8WorkerShutdownDisposition, Sq8WorkerControlError> {
        let prepared = {
            let mut state = self.lock_state()?;
            match state.lifecycle {
                Sq8WorkerLifecycle::Failed => {
                    return Err(control_error(
                        Sq8WorkerControlErrorKind::Failed,
                        "worker control is failed",
                    ));
                }
                Sq8WorkerLifecycle::Loading => {
                    return Err(control_error(
                        Sq8WorkerControlErrorKind::NotReady,
                        "worker is not ready",
                    ));
                }
                Sq8WorkerLifecycle::Closing => {
                    return Err(control_error(
                        Sq8WorkerControlErrorKind::Closing,
                        "worker input is closing",
                    ));
                }
                Sq8WorkerLifecycle::Ready => {}
            }
            state.lifecycle = Sq8WorkerLifecycle::Closing;
            let request_id = match state.active.as_ref() {
                Some(active) => active.request_id.clone(),
                None => return Ok(Sq8WorkerShutdownDisposition::Idle),
            };
            prepare_cancel_active(&mut state, &request_id, Sq8CancelReason::Shutdown)?
        };
        self.commit_prepared_cancel(prepared)
            .map(Sq8WorkerShutdownDisposition::Cancelling)
    }

    pub fn first_cancel_reason(
        &self,
        generation: u64,
    ) -> Result<Option<Sq8CancelReason>, Sq8WorkerControlError> {
        let state = self.lock_state()?;
        let active = matching_active(&state, generation)?;
        Ok(active.first_cancel_reason)
    }

    /// Reserves the sole terminal publication for this active generation.
    pub fn begin_terminal_publication(
        &self,
        generation: u64,
        request_id: &str,
    ) -> Result<Sq8ActiveTerminalPermit, Sq8WorkerControlError> {
        let mut state = self.lock_state()?;
        match state.lifecycle {
            Sq8WorkerLifecycle::Ready | Sq8WorkerLifecycle::Closing => {}
            Sq8WorkerLifecycle::Loading => {
                return Err(control_error(
                    Sq8WorkerControlErrorKind::NotReady,
                    "worker is not ready for terminal publication",
                ));
            }
            Sq8WorkerLifecycle::Failed => {
                return Err(control_error(
                    Sq8WorkerControlErrorKind::Failed,
                    "failed worker cannot publish a terminal release",
                ));
            }
        }
        let active = state
            .active
            .as_mut()
            .filter(|active| active.generation == generation && active.request_id == request_id)
            .ok_or_else(|| {
                control_error(
                    Sq8WorkerControlErrorKind::StaleGeneration,
                    "terminal publication has stale active ownership",
                )
            })?;
        if active.terminal_in_flight {
            return Err(control_error(
                Sq8WorkerControlErrorKind::Internal,
                "terminal publication is already in flight",
            ));
        }
        active.terminal_in_flight = true;
        Ok(Sq8ActiveTerminalPermit {
            generation,
            request_id: request_id.to_string(),
        })
    }

    /// Clears ownership only after the ordered writer has flushed rejection or release.
    pub fn acknowledge_terminal_flush(
        &self,
        acknowledgement: Sq8ActiveTerminalFlushAck,
    ) -> Result<(), Sq8WorkerControlError> {
        let mut state = self.lock_state()?;
        let active = matching_active(&state, acknowledgement.generation)?;
        if active.request_id != acknowledgement.request_id {
            return Err(control_error(
                Sq8WorkerControlErrorKind::StaleGeneration,
                "terminal flush acknowledgement has a stale request ID",
            ));
        }
        if !active.terminal_in_flight {
            return Err(control_error(
                Sq8WorkerControlErrorKind::Internal,
                "terminal flush has no matching in-flight publication",
            ));
        }
        state.active = None;
        Ok(())
    }

    pub fn fail_admission_transfer(&self, generation: u64) -> Result<(), Sq8WorkerControlError> {
        let mut state = self.lock_state()?;
        matching_active(&state, generation)?;
        state.active = None;
        state.lifecycle = Sq8WorkerLifecycle::Failed;
        Ok(())
    }

    pub fn mark_failed(&self) -> Result<(), Sq8WorkerControlError> {
        let mut state = self.lock_state()?;
        state.lifecycle = Sq8WorkerLifecycle::Failed;
        Ok(())
    }

    pub(crate) fn try_mark_failed_best_effort(&self) -> bool {
        match self.state.try_lock() {
            Ok(mut state) => {
                state.lifecycle = Sq8WorkerLifecycle::Failed;
                true
            }
            Err(TryLockError::Poisoned(poisoned)) => {
                poisoned.into_inner().lifecycle = Sq8WorkerLifecycle::Failed;
                true
            }
            Err(TryLockError::WouldBlock) => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_state_lock_for_test<F>(&self, action: F)
    where
        F: FnOnce(),
    {
        let _state = self.state.lock().unwrap();
        action();
    }

    pub fn snapshot(&self) -> Result<Sq8WorkerControlSnapshot, Sq8WorkerControlError> {
        let state = self.lock_state()?;
        Ok(Sq8WorkerControlSnapshot {
            lifecycle: state.lifecycle,
            active_generation: state.active.as_ref().map(|active| active.generation),
            active_request_id: state
                .active
                .as_ref()
                .map(|active| active.request_id.clone()),
            first_cancel_reason: state
                .active
                .as_ref()
                .and_then(|active| active.first_cancel_reason),
            cancelled: state
                .active
                .as_ref()
                .is_some_and(|active| active.cancel.is_cancelled()),
            terminal_in_flight: state
                .active
                .as_ref()
                .is_some_and(|active| active.terminal_in_flight),
        })
    }

    fn lock_state(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Sq8WorkerControlState>, Sq8WorkerControlError> {
        self.state.lock().map_err(|_| {
            control_error(
                Sq8WorkerControlErrorKind::Internal,
                "worker control mutex is poisoned",
            )
        })
    }

    fn commit_prepared_cancel(
        &self,
        prepared: Sq8PreparedControlCancel,
    ) -> Result<Sq8WorkerCancelResult, Sq8WorkerControlError> {
        if prepared.cancel.cancel_checked().is_err() {
            let _ = self.mark_failed();
            return Err(control_error(
                Sq8WorkerControlErrorKind::Internal,
                "worker cancellation publication mutex failed",
            ));
        }
        Ok(prepared.result)
    }
}

fn validate_generate_admission(state: &Sq8WorkerControlState) -> Result<(), Sq8WorkerControlError> {
    match state.lifecycle {
        Sq8WorkerLifecycle::Loading => Err(control_error(
            Sq8WorkerControlErrorKind::NotReady,
            "worker is not ready",
        )),
        Sq8WorkerLifecycle::Closing => Err(control_error(
            Sq8WorkerControlErrorKind::Closing,
            "worker input is closing",
        )),
        Sq8WorkerLifecycle::Failed => Err(control_error(
            Sq8WorkerControlErrorKind::Failed,
            "worker control is failed",
        )),
        Sq8WorkerLifecycle::Ready if state.active.is_some() => Err(control_error(
            Sq8WorkerControlErrorKind::Busy,
            "one request is already active",
        )),
        Sq8WorkerLifecycle::Ready => Ok(()),
    }
}

fn prepare_cancel_active(
    state: &mut Sq8WorkerControlState,
    request_id: &str,
    reason: Sq8CancelReason,
) -> Result<Sq8PreparedControlCancel, Sq8WorkerControlError> {
    let active = state.active.as_mut().ok_or_else(|| {
        control_error(
            Sq8WorkerControlErrorKind::UnknownRequest,
            "cancel request does not match an active request",
        )
    })?;
    if active.request_id != request_id {
        return Err(control_error(
            Sq8WorkerControlErrorKind::UnknownRequest,
            "cancel request does not match an active request",
        ));
    }
    let repeated = active.first_cancel_reason.is_some();
    let first_reason = *active.first_cancel_reason.get_or_insert(reason);
    Ok(Sq8PreparedControlCancel {
        cancel: active.cancel.clone(),
        result: Sq8WorkerCancelResult {
            generation: active.generation,
            first_reason,
            repeated,
        },
    })
}

fn matching_active(
    state: &Sq8WorkerControlState,
    generation: u64,
) -> Result<&Sq8ActiveControl, Sq8WorkerControlError> {
    state
        .active
        .as_ref()
        .filter(|active| active.generation == generation)
        .ok_or_else(|| {
            control_error(
                Sq8WorkerControlErrorKind::StaleGeneration,
                "terminal flush acknowledgement has a stale generation",
            )
        })
}

fn control_error(kind: Sq8WorkerControlErrorKind, message: &'static str) -> Sq8WorkerControlError {
    Sq8WorkerControlError { kind, message }
}

enum RawWorkerCommand {
    Generate {
        schema_version: String,
        request_id: String,
        prompt_token_ids: Vec<u64>,
        max_new_tokens: u64,
        sampling: RawWorkerSampling,
        eos_token_ids: Vec<u64>,
        reasoning: Option<RawWorkerReasoning>,
    },
    Cancel {
        schema_version: String,
        request_id: String,
        reason: Sq8CancelReason,
    },
    Shutdown {
        schema_version: String,
    },
}

#[derive(Default)]
struct RawWorkerCommandFields {
    schema_version: Option<String>,
    command_type: Option<String>,
    request_id: Option<String>,
    prompt_token_ids: Option<Vec<u64>>,
    max_new_tokens: Option<u64>,
    sampling: Option<RawWorkerSampling>,
    eos_token_ids: Option<Vec<u64>>,
    reasoning: Option<RawWorkerReasoning>,
    reason: Option<Sq8CancelReason>,
}

impl<'de> Deserialize<'de> for RawWorkerCommand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(RawWorkerCommandVisitor {
            token_id_limit: configured_worker_profile().context_length,
        })
    }
}

struct RawWorkerCommandSeed {
    token_id_limit: usize,
}

impl<'de> DeserializeSeed<'de> for RawWorkerCommandSeed {
    type Value = RawWorkerCommand;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(RawWorkerCommandVisitor {
            token_id_limit: self.token_id_limit,
        })
    }
}

struct RawWorkerCommandVisitor {
    token_id_limit: usize,
}

impl<'de> Visitor<'de> for RawWorkerCommandVisitor {
    type Value = RawWorkerCommand;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an exact SQ8 worker command object")
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut fields = RawWorkerCommandFields::default();
        while let Some(key) = object.next_key::<String>()? {
            match key.as_str() {
                "schema_version" => {
                    set_once(&mut fields.schema_version, object.next_value::<String>()?)?;
                }
                "type" => {
                    set_once(&mut fields.command_type, object.next_value::<String>()?)?;
                }
                "request_id" => {
                    set_once(&mut fields.request_id, object.next_value::<String>()?)?;
                }
                "prompt_token_ids" => {
                    set_once(
                        &mut fields.prompt_token_ids,
                        object.next_value_seed(BoundedTokenIdsSeed {
                            limit: self.token_id_limit,
                        })?,
                    )?;
                }
                "max_new_tokens" => {
                    set_once(&mut fields.max_new_tokens, object.next_value::<u64>()?)?;
                }
                "sampling" => {
                    set_once(
                        &mut fields.sampling,
                        object.next_value::<RawWorkerSampling>()?,
                    )?;
                }
                "eos_token_ids" => {
                    set_once(
                        &mut fields.eos_token_ids,
                        object.next_value_seed(BoundedTokenIdsSeed {
                            limit: self.token_id_limit,
                        })?,
                    )?;
                }
                "reasoning" => {
                    set_once(
                        &mut fields.reasoning,
                        object.next_value_seed(RawWorkerReasoningSeed {
                            token_id_limit: self.token_id_limit,
                        })?,
                    )?;
                }
                "reason" => {
                    set_once(&mut fields.reason, object.next_value::<Sq8CancelReason>()?)?;
                }
                _ => return Err(de::Error::custom("unknown SQ8 worker command field")),
            }
        }

        let schema_version = fields
            .schema_version
            .ok_or_else(|| de::Error::custom("missing schema_version"))?;
        let command_type = fields
            .command_type
            .ok_or_else(|| de::Error::custom("missing type"))?;
        match command_type.as_str() {
            "generate" => {
                if fields.reason.is_some() {
                    return Err(de::Error::custom("generate contains a cancel field"));
                }
                Ok(RawWorkerCommand::Generate {
                    schema_version,
                    request_id: require_field(fields.request_id, "request_id")?,
                    prompt_token_ids: require_field(fields.prompt_token_ids, "prompt_token_ids")?,
                    max_new_tokens: require_field(fields.max_new_tokens, "max_new_tokens")?,
                    sampling: require_field(fields.sampling, "sampling")?,
                    eos_token_ids: require_field(fields.eos_token_ids, "eos_token_ids")?,
                    reasoning: fields.reasoning,
                })
            }
            "cancel" => {
                if fields.prompt_token_ids.is_some()
                    || fields.max_new_tokens.is_some()
                    || fields.sampling.is_some()
                    || fields.eos_token_ids.is_some()
                    || fields.reasoning.is_some()
                {
                    return Err(de::Error::custom("cancel contains a generate field"));
                }
                Ok(RawWorkerCommand::Cancel {
                    schema_version,
                    request_id: require_field(fields.request_id, "request_id")?,
                    reason: require_field(fields.reason, "reason")?,
                })
            }
            "shutdown" => {
                if fields.request_id.is_some()
                    || fields.prompt_token_ids.is_some()
                    || fields.max_new_tokens.is_some()
                    || fields.sampling.is_some()
                    || fields.eos_token_ids.is_some()
                    || fields.reason.is_some()
                {
                    return Err(de::Error::custom("shutdown contains an extra field"));
                }
                Ok(RawWorkerCommand::Shutdown { schema_version })
            }
            _ => Err(de::Error::custom("unknown SQ8 worker command type")),
        }
    }
}

fn set_once<E: de::Error, T>(slot: &mut Option<T>, value: T) -> Result<(), E> {
    if slot.replace(value).is_some() {
        return Err(E::custom(DUPLICATE_KEY_ERROR));
    }
    Ok(())
}

fn require_field<E: de::Error, T>(value: Option<T>, name: &str) -> Result<T, E> {
    value.ok_or_else(|| E::custom(format_args!("missing {name}")))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWorkerSampling {
    temperature: f64,
    top_p: f64,
    top_k: u64,
    seed: i64,
}

struct RawWorkerReasoning {
    enabled: bool,
    budget_tokens: Option<u64>,
    dialect_id: String,
    end_token_ids: Vec<u64>,
    forced_end_token_ids: Vec<u64>,
    reserved_answer_tokens: u64,
}

struct RawWorkerReasoningSeed {
    token_id_limit: usize,
}

impl<'de> DeserializeSeed<'de> for RawWorkerReasoningSeed {
    type Value = RawWorkerReasoning;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(RawWorkerReasoningVisitor {
            token_id_limit: self.token_id_limit,
        })
    }
}

struct RawWorkerReasoningVisitor {
    token_id_limit: usize,
}

impl<'de> Visitor<'de> for RawWorkerReasoningVisitor {
    type Value = RawWorkerReasoning;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an exact reasoning execution object")
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut enabled = None;
        let mut budget_tokens = None;
        let mut dialect_id = None;
        let mut end_token_ids = None;
        let mut forced_end_token_ids = None;
        let mut reserved_answer_tokens = None;
        while let Some(key) = object.next_key::<String>()? {
            match key.as_str() {
                "enabled" => set_once(&mut enabled, object.next_value::<bool>()?)?,
                "budget_tokens" => {
                    set_once(&mut budget_tokens, object.next_value::<Option<u64>>()?)?
                }
                "dialect_id" => set_once(&mut dialect_id, object.next_value::<String>()?)?,
                "end_token_ids" => set_once(
                    &mut end_token_ids,
                    object.next_value_seed(BoundedTokenIdsSeed {
                        limit: self.token_id_limit,
                    })?,
                )?,
                "forced_end_token_ids" => set_once(
                    &mut forced_end_token_ids,
                    object.next_value_seed(BoundedTokenIdsSeed {
                        limit: self.token_id_limit,
                    })?,
                )?,
                "reserved_answer_tokens" => {
                    set_once(&mut reserved_answer_tokens, object.next_value::<u64>()?)?
                }
                _ => return Err(de::Error::custom("unknown reasoning execution field")),
            }
        }
        Ok(RawWorkerReasoning {
            enabled: require_field(enabled, "enabled")?,
            budget_tokens: require_field(budget_tokens, "budget_tokens")?,
            dialect_id: require_field(dialect_id, "dialect_id")?,
            end_token_ids: require_field(end_token_ids, "end_token_ids")?,
            forced_end_token_ids: require_field(forced_end_token_ids, "forced_end_token_ids")?,
            reserved_answer_tokens: require_field(
                reserved_answer_tokens,
                "reserved_answer_tokens",
            )?,
        })
    }
}

pub fn decode_sq8_worker_command(
    payload: &[u8],
) -> Result<Sq8WorkerCommand, Sq8WorkerProtocolError> {
    inspect_sq8_worker_command(payload)?.decode()
}

pub fn inspect_sq8_worker_command(
    payload: &[u8],
) -> Result<Sq8WorkerCommandInspection<'_>, Sq8WorkerProtocolError> {
    validate_strict_json(payload)?;
    let discriminators =
        serde_json::from_slice::<RawWorkerDiscriminators>(payload).map_err(|_| {
            Sq8WorkerProtocolError::invalid_command(
                "command does not contain valid SQ8 worker discriminators",
            )
        })?;
    validate_schema_version(&discriminators.schema_version)?;
    let kind = match discriminators.command_type.as_str() {
        "generate" => Sq8WorkerCommandKind::Generate,
        "cancel" => Sq8WorkerCommandKind::Cancel,
        "shutdown" => Sq8WorkerCommandKind::Shutdown,
        _ => {
            return Err(Sq8WorkerProtocolError::invalid_command(
                "command type is not valid on SQ8 worker stdin",
            ));
        }
    };
    let request_id = discriminators
        .request_id
        .filter(|request_id| validate_worker_request_id(request_id).is_ok());
    Ok(Sq8WorkerCommandInspection {
        payload,
        kind,
        request_id,
    })
}

fn decode_inspected_sq8_worker_command(
    payload: &[u8],
    inspected_kind: Sq8WorkerCommandKind,
    token_id_limit: usize,
) -> Result<Sq8WorkerCommand, Sq8WorkerProtocolError> {
    let mut deserializer = serde_json::Deserializer::from_slice(payload);
    let raw = RawWorkerCommandSeed { token_id_limit }
        .deserialize(&mut deserializer)
        .and_then(|value| deserializer.end().map(|()| value))
        .map_err(|_| {
            Sq8WorkerProtocolError::invalid_command(
                "command does not match the strict SQ8 worker schema",
            )
        })?;
    match raw {
        RawWorkerCommand::Generate {
            schema_version,
            request_id,
            prompt_token_ids,
            max_new_tokens,
            sampling,
            eos_token_ids,
            reasoning,
        } => {
            validate_schema_version(&schema_version)?;
            if inspected_kind != Sq8WorkerCommandKind::Generate {
                return Err(Sq8WorkerProtocolError::invalid_command(
                    "command type changed after inspection",
                ));
            }
            if schema_version == SQ8_WORKER_SCHEMA_VERSION && reasoning.is_some() {
                return Err(Sq8WorkerProtocolError::invalid_command(
                    "reasoning execution requires ullm.worker.v2",
                ));
            }
            Ok(Sq8WorkerCommand::Generate(Sq8GenerateCommand {
                request_id,
                prompt_token_ids: convert_token_ids(prompt_token_ids)?,
                max_new_tokens: usize::try_from(max_new_tokens).map_err(|_| {
                    Sq8WorkerProtocolError::invalid_command(
                        "max_new_tokens does not fit the worker integer type",
                    )
                })?,
                sampling: Sq8WorkerSampling {
                    temperature: sampling.temperature,
                    top_p: sampling.top_p,
                    top_k: usize::try_from(sampling.top_k).map_err(|_| {
                        Sq8WorkerProtocolError::invalid_command(
                            "top_k does not fit the worker integer type",
                        )
                    })?,
                    seed: sampling.seed,
                },
                eos_token_ids: convert_token_ids(eos_token_ids)?,
                reasoning: reasoning.map(convert_reasoning).transpose()?,
            }))
        }
        RawWorkerCommand::Cancel {
            schema_version,
            request_id,
            reason,
        } => {
            validate_schema_version(&schema_version)?;
            if inspected_kind != Sq8WorkerCommandKind::Cancel {
                return Err(Sq8WorkerProtocolError::invalid_command(
                    "command type changed after inspection",
                ));
            }
            validate_worker_request_id(&request_id).map_err(|_| {
                Sq8WorkerProtocolError::invalid_command(
                    "cancel request_id violates the SQ8 worker syntax",
                )
            })?;
            Ok(Sq8WorkerCommand::Cancel { request_id, reason })
        }
        RawWorkerCommand::Shutdown { schema_version } => {
            validate_schema_version(&schema_version)?;
            if inspected_kind != Sq8WorkerCommandKind::Shutdown {
                return Err(Sq8WorkerProtocolError::invalid_command(
                    "command type changed after inspection",
                ));
            }
            Ok(Sq8WorkerCommand::Shutdown)
        }
    }
}

struct RawWorkerDiscriminators {
    schema_version: String,
    command_type: String,
    request_id: Option<String>,
}

impl<'de> Deserialize<'de> for RawWorkerDiscriminators {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(RawWorkerDiscriminatorVisitor)
    }
}

struct RawWorkerDiscriminatorVisitor;

impl<'de> Visitor<'de> for RawWorkerDiscriminatorVisitor {
    type Value = RawWorkerDiscriminators;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an SQ8 worker command object with discriminators")
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut schema_version = None;
        let mut command_type = None;
        let mut request_id = None;
        while let Some(key) = object.next_key::<String>()? {
            match key.as_str() {
                "schema_version" => {
                    set_once(&mut schema_version, object.next_value::<String>()?)?;
                }
                "type" => {
                    set_once(&mut command_type, object.next_value::<String>()?)?;
                }
                "request_id" => {
                    set_once(&mut request_id, object.next_value::<LenientString>()?.0)?;
                }
                _ => {
                    object.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(RawWorkerDiscriminators {
            schema_version: require_field(schema_version, "schema_version")?,
            command_type: require_field(command_type, "type")?,
            request_id: request_id.flatten(),
        })
    }
}

struct LenientString(Option<String>);

impl<'de> Deserialize<'de> for LenientString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(LenientStringVisitor)
    }
}

struct LenientStringVisitor;

impl<'de> Visitor<'de> for LenientStringVisitor {
    type Value = LenientString;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(LenientString(Some(value.to_string())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(LenientString(Some(value)))
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(LenientString(None))
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(LenientString(None))
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(LenientString(None))
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(LenientString(None))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(LenientString(None))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(LenientString(None))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element::<IgnoredAny>()?.is_some() {}
        Ok(LenientString(None))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while object.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
        Ok(LenientString(None))
    }
}

fn validate_schema_version(value: &str) -> Result<(), Sq8WorkerProtocolError> {
    if value != SQ8_WORKER_SCHEMA_VERSION && value != SQ8_WORKER_SCHEMA_VERSION_V2 {
        return Err(Sq8WorkerProtocolError::invalid_command(
            "command schema_version is not a supported SQ8 worker version",
        ));
    }
    Ok(())
}

fn convert_reasoning(
    raw: RawWorkerReasoning,
) -> Result<crate::reasoning::ReasoningExecution, Sq8WorkerProtocolError> {
    if raw.dialect_id.is_empty() || raw.dialect_id.len() > 256 {
        return Err(Sq8WorkerProtocolError::invalid_command(
            "reasoning dialect_id violates the bounded text contract",
        ));
    }
    Ok(crate::reasoning::ReasoningExecution {
        enabled: raw.enabled,
        budget_tokens: raw
            .budget_tokens
            .map(|value| {
                usize::try_from(value).map_err(|_| {
                    Sq8WorkerProtocolError::invalid_command(
                        "reasoning budget_tokens does not fit the worker integer type",
                    )
                })
            })
            .transpose()?,
        dialect_id: raw.dialect_id,
        end_sequence: convert_token_ids(raw.end_token_ids)?,
        forced_end_sequence: convert_token_ids(raw.forced_end_token_ids)?,
        reserved_answer_tokens: usize::try_from(raw.reserved_answer_tokens).map_err(|_| {
            Sq8WorkerProtocolError::invalid_command(
                "reasoning reserved_answer_tokens does not fit the worker integer type",
            )
        })?,
    })
}

fn convert_token_ids(values: Vec<u64>) -> Result<Vec<usize>, Sq8WorkerProtocolError> {
    values
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                Sq8WorkerProtocolError::invalid_command(
                    "token ID does not fit the worker integer type",
                )
            })
        })
        .collect()
}

pub fn validate_worker_request_id(value: &str) -> Result<(), Sq8WorkerProtocolError> {
    let bytes = value.as_bytes();
    if bytes.is_empty()
        || bytes.len() > 128
        || !bytes[0].is_ascii_alphanumeric()
        || bytes[1..].iter().any(|byte| {
            !byte.is_ascii_alphanumeric() && !matches!(*byte, b'.' | b'_' | b':' | b'-')
        })
    {
        return Err(Sq8WorkerProtocolError::invalid_command(
            "request_id violates the SQ8 worker syntax",
        ));
    }
    Ok(())
}

fn validate_strict_json(payload: &[u8]) -> Result<(), Sq8WorkerProtocolError> {
    let mut deserializer = serde_json::Deserializer::from_slice(payload);
    StrictJsonSeed { composite_depth: 0 }
        .deserialize(&mut deserializer)
        .and_then(|()| deserializer.end())
        .map_err(|err| {
            let error = err.to_string();
            if error.contains(DUPLICATE_KEY_ERROR) {
                Sq8WorkerProtocolError::invalid_command(DUPLICATE_KEY_ERROR)
            } else if error.contains(JSON_DEPTH_ERROR) {
                Sq8WorkerProtocolError::invalid_command(JSON_DEPTH_ERROR)
            } else {
                Sq8WorkerProtocolError::invalid_command(
                    "command is not one complete valid UTF-8 JSON value",
                )
            }
        })
}

#[derive(Clone, Copy)]
struct StrictJsonSeed {
    composite_depth: usize,
}

impl<'de> DeserializeSeed<'de> for StrictJsonSeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor {
            composite_depth: self.composite_depth,
        })
    }
}

struct StrictJsonVisitor {
    composite_depth: usize,
}

impl StrictJsonVisitor {
    fn child_seed<E: de::Error>(&self) -> Result<StrictJsonSeed, E> {
        let composite_depth = self
            .composite_depth
            .checked_add(1)
            .ok_or_else(|| E::custom(JSON_DEPTH_ERROR))?;
        if composite_depth > SQ8_WORKER_MAX_JSON_DEPTH {
            return Err(E::custom(JSON_DEPTH_ERROR));
        }
        Ok(StrictJsonSeed { composite_depth })
    }
}

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded JSON value")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let child = self.child_seed::<A::Error>()?;
        while sequence.next_element_seed(child)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let child = self.child_seed::<A::Error>()?;
        let mut keys = HashSet::new();
        while let Some(key) = object.next_key::<String>()? {
            if !keys.insert(key) {
                return Err(de::Error::custom(DUPLICATE_KEY_ERROR));
            }
            object.next_value_seed(child)?;
        }
        Ok(())
    }
}

struct BoundedTokenIdsSeed {
    limit: usize,
}

impl<'de> DeserializeSeed<'de> for BoundedTokenIdsSeed {
    type Value = Vec<u64>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(BoundedTokenIdsVisitor { limit: self.limit })
    }
}

struct BoundedTokenIdsVisitor {
    limit: usize,
}

impl<'de> Visitor<'de> for BoundedTokenIdsVisitor {
    type Value = Vec<u64>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a bounded array of non-negative integer token IDs")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(self.limit + 1));
        while let Some(value) = sequence.next_element::<u64>()? {
            if values.len() == self.limit + 1 {
                return Err(de::Error::custom("token ID array exceeds protocol limit"));
            }
            values.push(value);
        }
        Ok(values)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sq8ReleaseOutcomeEvent {
    Stop,
    Length,
    Cancelled,
}

impl Sq8WorkerTimings {
    pub fn from_elapsed_millis(
        prompt_n: usize,
        prompt_ms: f64,
        predicted_n: usize,
        predicted_ms: f64,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        let profile = configured_worker_profile();
        Self::from_elapsed_millis_with_profile(
            prompt_n,
            prompt_ms,
            predicted_n,
            predicted_ms,
            &profile,
        )
    }

    pub fn from_elapsed_millis_with_profile(
        prompt_n: usize,
        prompt_ms: f64,
        predicted_n: usize,
        predicted_ms: f64,
        profile: &Sq8WorkerProfile,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        Self::from_elapsed_millis_with_limits(
            prompt_n,
            prompt_ms,
            predicted_n,
            predicted_ms,
            profile.context_length,
            profile.max_new_tokens,
        )
        .ok_or_else(|| {
            Sq8WorkerProtocolError::invalid_command("SQ8 worker timings are out of range")
        })
    }

    fn validate_for_release(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Result<(), String> {
        if !self.validates_release(prompt_tokens, completion_tokens) {
            return Err("SQ8 released timings violate the llama-server contract".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sq8WorkerErrorCode {
    InvalidCommand,
    InvalidRequest,
    Busy,
    UnknownRequest,
    LoadFailed,
    RuntimeFailed,
    InvariantFailed,
    ProtocolFramingFailed,
    CleanupDeadlineExceeded,
}

impl Sq8WorkerErrorCode {
    pub fn recoverable(self) -> bool {
        matches!(
            self,
            Self::InvalidCommand | Self::InvalidRequest | Self::Busy | Self::UnknownRequest
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields, tag = "type")]
pub enum Sq8WorkerEvent {
    #[serde(rename = "ready")]
    Ready {
        schema_version: String,
        model: String,
        model_revision: String,
        artifact_content_sha256: String,
        package_manifest_sha256: String,
        device: String,
        execution_profile: String,
        context_length: usize,
        max_new_tokens: usize,
    },
    #[serde(rename = "started")]
    Started {
        schema_version: String,
        request_id: String,
        prompt_tokens: usize,
    },
    #[serde(rename = "progress")]
    Progress {
        schema_version: String,
        request_id: String,
        phase: &'static str,
        processed_prompt_tokens: usize,
    },
    #[serde(rename = "token")]
    Token {
        schema_version: String,
        request_id: String,
        index: usize,
        token_id: usize,
    },
    #[serde(rename = "released")]
    Released {
        schema_version: String,
        request_id: String,
        outcome: Sq8ReleaseOutcomeEvent,
        #[serde(skip_serializing_if = "Option::is_none")]
        cancel_reason: Option<Sq8CancelReason>,
        prompt_tokens: usize,
        completion_tokens: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_tokens: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        forced_end_tokens: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timings: Option<Sq8WorkerTimings>,
        reset_complete: bool,
    },
    #[serde(rename = "error")]
    Error {
        schema_version: String,
        request_id: Option<String>,
        code: Sq8WorkerErrorCode,
        recoverable: bool,
        message: String,
    },
}

impl Sq8WorkerEvent {
    pub fn ready() -> Self {
        let profile = configured_worker_profile();
        Self::ready_with_profile(&profile)
    }

    pub fn ready_with_profile(profile: &Sq8WorkerProfile) -> Self {
        Self::Ready {
            schema_version: profile.worker_schema.clone(),
            model: profile.model.clone(),
            model_revision: profile.model_revision.clone(),
            artifact_content_sha256: profile.artifact_content_sha256.clone(),
            package_manifest_sha256: profile.package_manifest_sha256.clone(),
            device: profile.device.clone(),
            execution_profile: profile.execution_profile.clone(),
            context_length: profile.context_length,
            max_new_tokens: profile.max_new_tokens,
        }
    }

    pub fn started(request_id: impl Into<String>, prompt_tokens: usize) -> Self {
        Self::started_with_schema(SQ8_WORKER_SCHEMA_VERSION, request_id, prompt_tokens)
    }

    pub fn started_with_schema(
        schema_version: &str,
        request_id: impl Into<String>,
        prompt_tokens: usize,
    ) -> Self {
        Self::Started {
            schema_version: schema_version.to_string(),
            request_id: request_id.into(),
            prompt_tokens,
        }
    }

    pub fn progress(request_id: impl Into<String>, processed_prompt_tokens: usize) -> Self {
        Self::progress_with_schema(
            SQ8_WORKER_SCHEMA_VERSION,
            request_id,
            processed_prompt_tokens,
        )
    }

    pub fn progress_with_schema(
        schema_version: &str,
        request_id: impl Into<String>,
        processed_prompt_tokens: usize,
    ) -> Self {
        Self::Progress {
            schema_version: schema_version.to_string(),
            request_id: request_id.into(),
            phase: "prefill",
            processed_prompt_tokens,
        }
    }

    pub fn token(request_id: impl Into<String>, index: usize, token_id: usize) -> Self {
        Self::token_with_schema(SQ8_WORKER_SCHEMA_VERSION, request_id, index, token_id)
    }

    pub fn token_with_schema(
        schema_version: &str,
        request_id: impl Into<String>,
        index: usize,
        token_id: usize,
    ) -> Self {
        Self::Token {
            schema_version: schema_version.to_string(),
            request_id: request_id.into(),
            index,
            token_id,
        }
    }

    pub fn released(
        request_id: impl Into<String>,
        outcome: Sq8ReleaseOutcomeEvent,
        cancel_reason: Option<Sq8CancelReason>,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        Self::released_with_schema(
            SQ8_WORKER_SCHEMA_VERSION,
            request_id,
            outcome,
            cancel_reason,
            prompt_tokens,
            completion_tokens,
        )
    }

    pub fn released_with_schema(
        schema_version: &str,
        request_id: impl Into<String>,
        outcome: Sq8ReleaseOutcomeEvent,
        cancel_reason: Option<Sq8CancelReason>,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        Self::released_with_schema_and_reasoning(
            schema_version,
            request_id,
            outcome,
            cancel_reason,
            prompt_tokens,
            completion_tokens,
            None,
        )
    }

    pub fn released_with_schema_and_reasoning(
        schema_version: &str,
        request_id: impl Into<String>,
        outcome: Sq8ReleaseOutcomeEvent,
        cancel_reason: Option<Sq8CancelReason>,
        prompt_tokens: usize,
        completion_tokens: usize,
        reasoning_usage: Option<ReasoningUsage>,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        Self::released_inner(
            schema_version,
            request_id,
            outcome,
            cancel_reason,
            prompt_tokens,
            completion_tokens,
            reasoning_usage,
            None,
        )
    }

    pub fn released_with_timings(
        request_id: impl Into<String>,
        outcome: Sq8ReleaseOutcomeEvent,
        cancel_reason: Option<Sq8CancelReason>,
        prompt_tokens: usize,
        completion_tokens: usize,
        timings: Sq8WorkerTimings,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        Self::released_with_timings_schema(
            SQ8_WORKER_SCHEMA_VERSION,
            request_id,
            outcome,
            cancel_reason,
            prompt_tokens,
            completion_tokens,
            timings,
        )
    }

    pub fn released_with_timings_schema(
        schema_version: &str,
        request_id: impl Into<String>,
        outcome: Sq8ReleaseOutcomeEvent,
        cancel_reason: Option<Sq8CancelReason>,
        prompt_tokens: usize,
        completion_tokens: usize,
        timings: Sq8WorkerTimings,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        Self::released_with_timings_schema_and_reasoning(
            schema_version,
            request_id,
            outcome,
            cancel_reason,
            prompt_tokens,
            completion_tokens,
            timings,
            None,
        )
    }

    pub fn released_with_timings_schema_and_reasoning(
        schema_version: &str,
        request_id: impl Into<String>,
        outcome: Sq8ReleaseOutcomeEvent,
        cancel_reason: Option<Sq8CancelReason>,
        prompt_tokens: usize,
        completion_tokens: usize,
        timings: Sq8WorkerTimings,
        reasoning_usage: Option<ReasoningUsage>,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        Self::released_inner(
            schema_version,
            request_id,
            outcome,
            cancel_reason,
            prompt_tokens,
            completion_tokens,
            reasoning_usage,
            Some(timings),
        )
    }

    fn released_inner(
        schema_version: &str,
        request_id: impl Into<String>,
        outcome: Sq8ReleaseOutcomeEvent,
        cancel_reason: Option<Sq8CancelReason>,
        prompt_tokens: usize,
        completion_tokens: usize,
        reasoning_usage: Option<ReasoningUsage>,
        timings: Option<Sq8WorkerTimings>,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        if matches!(outcome, Sq8ReleaseOutcomeEvent::Cancelled) != cancel_reason.is_some() {
            return Err(Sq8WorkerProtocolError::invalid_command(
                "cancel_reason must exist only for a cancelled release",
            ));
        }
        if reasoning_usage.is_some() && schema_version != SQ8_WORKER_SCHEMA_VERSION_V2 {
            return Err(Sq8WorkerProtocolError::invalid_command(
                "reasoning release accounting requires ullm.worker.v2",
            ));
        }
        if let Some(usage) = reasoning_usage.as_ref()
            && usage
                .reasoning_tokens
                .saturating_add(usage.forced_end_tokens)
                > completion_tokens
        {
            return Err(Sq8WorkerProtocolError::invalid_command(
                "reasoning release accounting exceeds completion tokens",
            ));
        }
        if outcome == Sq8ReleaseOutcomeEvent::Cancelled && timings.is_some() {
            return Err(Sq8WorkerProtocolError::invalid_command(
                "timings are forbidden for a cancelled release",
            ));
        }
        if let Some(timings) = timings {
            timings
                .validate_for_release(prompt_tokens, completion_tokens)
                .map_err(Sq8WorkerProtocolError::invalid_command)?;
        }
        Ok(Self::Released {
            schema_version: schema_version.to_string(),
            request_id: request_id.into(),
            outcome,
            cancel_reason,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens: reasoning_usage.as_ref().map(|usage| usage.reasoning_tokens),
            forced_end_tokens: reasoning_usage.map(|usage| usage.forced_end_tokens),
            timings,
            reset_complete: true,
        })
    }

    pub fn error(
        request_id: Option<String>,
        code: Sq8WorkerErrorCode,
        message: impl Into<String>,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        let message = message.into();
        if message.len() > SQ8_WORKER_MAX_ERROR_MESSAGE_BYTES
            || message.chars().any(char::is_control)
        {
            return Err(Sq8WorkerProtocolError::invalid_command(
                "worker error message violates the bounded text contract",
            ));
        }
        if let Some(request_id) = request_id.as_deref() {
            validate_worker_request_id(request_id)?;
        }
        Self::error_with_schema(SQ8_WORKER_SCHEMA_VERSION, request_id, code, message)
    }

    pub fn error_with_schema(
        schema_version: &str,
        request_id: Option<String>,
        code: Sq8WorkerErrorCode,
        message: impl Into<String>,
    ) -> Result<Self, Sq8WorkerProtocolError> {
        let message = message.into();
        if message.len() > SQ8_WORKER_MAX_ERROR_MESSAGE_BYTES
            || message.chars().any(char::is_control)
        {
            return Err(Sq8WorkerProtocolError::invalid_command(
                "worker error message violates the bounded text contract",
            ));
        }
        if let Some(request_id) = request_id.as_deref() {
            validate_worker_request_id(request_id)?;
        }
        Ok(Self::Error {
            schema_version: schema_version.to_string(),
            request_id,
            code,
            recoverable: code.recoverable(),
            message,
        })
    }

    fn validate_with_profile(&self, profile: &Sq8WorkerProfile) -> Result<(), String> {
        match self {
            Self::Ready {
                schema_version,
                model,
                model_revision,
                artifact_content_sha256,
                package_manifest_sha256,
                device,
                execution_profile,
                context_length,
                max_new_tokens,
            } => {
                if schema_version != &profile.worker_schema
                    || *model != profile.model
                    || *model_revision != profile.model_revision
                    || *artifact_content_sha256 != profile.artifact_content_sha256
                    || *package_manifest_sha256 != profile.package_manifest_sha256
                    || *device != profile.device
                    || *execution_profile != profile.execution_profile
                    || *context_length != profile.context_length
                    || *max_new_tokens != profile.max_new_tokens
                {
                    return Err("SQ8 ready event identity changed".into());
                }
            }
            Self::Started {
                schema_version,
                request_id,
                prompt_tokens,
            } => {
                validate_event_common(schema_version, request_id, &profile.worker_schema)?;
                if !(1..=profile.context_length).contains(prompt_tokens) {
                    return Err("SQ8 started prompt count is out of range".into());
                }
            }
            Self::Progress {
                schema_version,
                request_id,
                phase,
                processed_prompt_tokens,
            } => {
                validate_event_common(schema_version, request_id, &profile.worker_schema)?;
                if *phase != "prefill"
                    || !(1..=profile.context_length).contains(processed_prompt_tokens)
                {
                    return Err("SQ8 progress event is out of range".into());
                }
            }
            Self::Token {
                schema_version,
                request_id,
                index,
                token_id,
            } => {
                validate_event_common(schema_version, request_id, &profile.worker_schema)?;
                if *index >= profile.max_new_tokens || *token_id >= profile.vocab_size {
                    return Err("SQ8 token event is out of range".into());
                }
            }
            Self::Released {
                schema_version,
                request_id,
                outcome,
                cancel_reason,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                forced_end_tokens,
                timings,
                reset_complete,
            } => {
                validate_event_common(schema_version, request_id, &profile.worker_schema)?;
                if matches!(outcome, Sq8ReleaseOutcomeEvent::Cancelled) != cancel_reason.is_some()
                    || !(1..=profile.context_length).contains(prompt_tokens)
                    || *completion_tokens > profile.max_new_tokens
                    || (*outcome == Sq8ReleaseOutcomeEvent::Cancelled && timings.is_some())
                    || !reset_complete
                {
                    return Err("SQ8 released event violates its terminal contract".into());
                }
                if reasoning_tokens.is_some() != forced_end_tokens.is_some()
                    || (schema_version != &SQ8_WORKER_SCHEMA_VERSION_V2
                        && (reasoning_tokens.is_some() || forced_end_tokens.is_some()))
                    || reasoning_tokens.zip(*forced_end_tokens).is_some_and(
                        |(reasoning, forced)| reasoning.saturating_add(forced) > *completion_tokens,
                    )
                {
                    return Err("SQ8 released reasoning accounting is invalid".into());
                }
                if let Some(timings) = timings {
                    timings.validate_for_release(*prompt_tokens, *completion_tokens)?;
                }
            }
            Self::Error {
                schema_version,
                request_id,
                code,
                recoverable,
                message,
            } => {
                if schema_version != &profile.worker_schema
                    || *recoverable != code.recoverable()
                    || message.len() > SQ8_WORKER_MAX_ERROR_MESSAGE_BYTES
                    || message.chars().any(char::is_control)
                {
                    return Err("SQ8 error event violates its bounded contract".into());
                }
                if let Some(request_id) = request_id {
                    validate_worker_request_id(request_id).map_err(|_| {
                        "SQ8 error event request ID violates the protocol".to_string()
                    })?;
                }
            }
        }
        Ok(())
    }
}

fn validate_event_common(
    schema_version: &str,
    request_id: &str,
    expected_schema_version: &str,
) -> Result<(), String> {
    if schema_version != expected_schema_version {
        return Err("SQ8 event schema version changed".into());
    }
    validate_worker_request_id(request_id)
        .map_err(|_| "SQ8 event request ID violates the protocol".to_string())
}

pub struct Sq8OrderedJsonlWriter<W> {
    inner: W,
    failed: bool,
    profile: Sq8WorkerProfile,
}

impl<W: Write> Sq8OrderedJsonlWriter<W> {
    pub fn new(inner: W) -> Self {
        Self::with_profile(inner, configured_worker_profile())
    }

    pub fn with_profile(inner: W, profile: Sq8WorkerProfile) -> Self {
        Self {
            inner,
            failed: false,
            profile,
        }
    }

    pub fn write_event(&mut self, event: &Sq8WorkerEvent) -> Result<(), String> {
        if self.failed {
            return Err("SQ8 worker event writer is failed".into());
        }
        let result = (|| {
            event.validate_with_profile(&self.profile)?;
            serde_json::to_writer(&mut self.inner, event)
                .map_err(|_| "failed to serialize SQ8 worker event".to_string())?;
            self.inner
                .write_all(b"\n")
                .map_err(|_| "failed to write SQ8 worker event".to_string())?;
            self.inner
                .flush()
                .map_err(|_| "failed to flush SQ8 worker event".to_string())
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    pub fn write_active_terminal_event(
        &mut self,
        permit: Sq8ActiveTerminalPermit,
        event: &Sq8WorkerEvent,
    ) -> Result<Sq8ActiveTerminalFlushAck, String> {
        let event_request_id = match event {
            Sq8WorkerEvent::Released {
                request_id: event_request_id,
                ..
            } => event_request_id,
            Sq8WorkerEvent::Error {
                request_id: Some(event_request_id),
                code: Sq8WorkerErrorCode::InvalidRequest,
                recoverable: true,
                ..
            } => event_request_id,
            _ => {
                self.failed = true;
                return Err("SQ8 active terminal writer received a nonterminal event".into());
            }
        };
        if event_request_id != &permit.request_id {
            self.failed = true;
            return Err("SQ8 active terminal event request ID does not match ownership".into());
        }
        self.write_event(event)?;
        Ok(Sq8ActiveTerminalFlushAck {
            generation: permit.generation,
            request_id: permit.request_id,
        })
    }

    pub fn write_ready_event(
        &mut self,
        event: &Sq8WorkerEvent,
    ) -> Result<Sq8ReadyFlushAck, String> {
        if !matches!(event, Sq8WorkerEvent::Ready { .. }) {
            self.failed = true;
            return Err("SQ8 ready writer received a non-ready event".into());
        }
        self.write_event(event)?;
        Ok(Sq8ReadyFlushAck { private: () })
    }

    pub fn is_failed(&self) -> bool {
        self.failed
    }

    pub fn into_inner(self) -> W {
        self.inner
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8PromptProgressTracker {
    prompt_tokens: usize,
    last_observed: usize,
    last_emitted: usize,
    transition_emitted: bool,
}

impl Sq8PromptProgressTracker {
    pub fn new(prompt_tokens: usize) -> Result<Self, String> {
        let profile = configured_worker_profile();
        Self::new_with_profile(prompt_tokens, &profile)
    }

    pub fn new_with_profile(
        prompt_tokens: usize,
        profile: &Sq8WorkerProfile,
    ) -> Result<Self, String> {
        if !(1..=profile.context_length).contains(&prompt_tokens) {
            return Err("SQ8 worker progress prompt length is out of range".into());
        }
        Ok(Self {
            prompt_tokens,
            last_observed: 0,
            last_emitted: 0,
            transition_emitted: false,
        })
    }

    pub fn observe_unit(
        &mut self,
        processed_prompt_tokens: usize,
        execution_width: usize,
    ) -> Result<Option<usize>, String> {
        if !(1..=MAX_WORKER_PROGRESS_EXECUTION_WIDTH).contains(&execution_width) {
            return Err("worker prompt progress execution width is out of range".into());
        }
        let remaining = self
            .prompt_tokens
            .checked_sub(self.last_observed)
            .ok_or_else(|| "worker prompt progress remaining token count overflowed".to_string())?;
        if execution_width > remaining {
            return Err("worker prompt progress execution width exceeds remaining prompt".into());
        }
        let expected_processed = self
            .last_observed
            .checked_add(execution_width)
            .ok_or_else(|| "worker prompt progress token count overflowed".to_string())?;
        if processed_prompt_tokens != expected_processed {
            return Err("worker prompt progress is non-contiguous".into());
        }
        let accumulated_since_emit = processed_prompt_tokens
            .checked_sub(self.last_emitted)
            .ok_or_else(|| "worker prompt progress emission count overflowed".to_string())?;
        self.last_observed = processed_prompt_tokens;
        if accumulated_since_emit >= MAX_WORKER_PROGRESS_EXECUTION_WIDTH
            || processed_prompt_tokens == self.prompt_tokens
        {
            self.last_emitted = processed_prompt_tokens;
            if processed_prompt_tokens == self.prompt_tokens {
                self.transition_emitted = true;
            }
            return Ok(Some(processed_prompt_tokens));
        }
        Ok(None)
    }

    pub fn observe_transition(&mut self) -> Result<Option<usize>, String> {
        if self.transition_emitted {
            return Ok(None);
        }
        let final_width = self
            .prompt_tokens
            .checked_sub(self.last_observed)
            .ok_or_else(|| {
                "worker prefill transition remaining token count overflowed".to_string()
            })?;
        if self.last_emitted > self.last_observed {
            return Err("worker prefill transition emission count overflowed".into());
        }
        if !(1..=MAX_WORKER_PROGRESS_EXECUTION_WIDTH).contains(&final_width) {
            return Err("worker prefill transition has an invalid final unit".into());
        }
        self.last_observed = self.prompt_tokens;
        self.last_emitted = self.prompt_tokens;
        self.transition_emitted = true;
        Ok(Some(self.prompt_tokens))
    }

    pub fn transition_emitted(&self) -> bool {
        self.transition_emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::io::{BufReader, Cursor};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    fn valid_generate() -> Vec<u8> {
        br#"{"schema_version":"ullm.worker.v1","type":"generate","request_id":"req-1","prompt_token_ids":[1,2,3],"max_new_tokens":2,"sampling":{"temperature":0.6,"top_p":0.95,"top_k":20,"seed":-7},"eos_token_ids":[151645,151643]}"#.to_vec()
    }

    fn flushed_terminal_ack(
        control: &Sq8WorkerControl,
        generation: u64,
        request_id: &str,
    ) -> Sq8ActiveTerminalFlushAck {
        let event =
            Sq8WorkerEvent::released(request_id, Sq8ReleaseOutcomeEvent::Length, None, 1, 1)
                .unwrap();
        let permit = control
            .begin_terminal_publication(generation, request_id)
            .unwrap();
        Sq8OrderedJsonlWriter::new(FlushCountingWriter::default())
            .write_active_terminal_event(permit, &event)
            .unwrap()
    }

    fn mark_control_ready(control: &Sq8WorkerControl) {
        let acknowledgement = Sq8OrderedJsonlWriter::new(FlushCountingWriter::default())
            .write_ready_event(&Sq8WorkerEvent::ready())
            .unwrap();
        control.mark_ready_after_flush(acknowledgement).unwrap();
    }

    fn read_all_with_capacity(input: &[u8], capacity: usize) -> Vec<Sq8JsonlRead> {
        let reader = BufReader::with_capacity(capacity, Cursor::new(input.to_vec()));
        let mut reader = Sq8BoundedJsonlReader::with_max_payload_bytes(reader, 64);
        let mut records = Vec::new();
        loop {
            let record = reader.next_record().unwrap();
            let eof = record == Sq8JsonlRead::Eof;
            records.push(record);
            if eof {
                return records;
            }
        }
    }

    #[test]
    fn bounded_reader_is_chunking_invariant_and_accepts_crlf() {
        let input = b"{\"a\":1}\n{\"b\":2}\r\n";
        let expected = vec![
            Sq8JsonlRead::Record(br#"{"a":1}"#.to_vec()),
            Sq8JsonlRead::Record(br#"{"b":2}"#.to_vec()),
            Sq8JsonlRead::Eof,
        ];
        for capacity in [1, 2, 7, 64] {
            assert_eq!(read_all_with_capacity(input, capacity), expected);
        }
    }

    #[test]
    fn bounded_reader_handles_limit_cr_and_oversize_drain() {
        let mut input = vec![b'x'; 32];
        input.extend_from_slice(b"\n");
        input.extend(std::iter::repeat_n(b'y', 32));
        input.extend_from_slice(b"\r\n");
        input.extend(std::iter::repeat_n(b'z', 33));
        input.extend_from_slice(b"\n{}\n");
        let reader = BufReader::with_capacity(3, Cursor::new(input));
        let mut reader = Sq8BoundedJsonlReader::with_max_payload_bytes(reader, 32);
        assert_eq!(reader.fixed_buffer_bytes(), 33);
        assert!(
            matches!(reader.next_record().unwrap(), Sq8JsonlRead::Record(value) if value.len() == 32)
        );
        assert!(
            matches!(reader.next_record().unwrap(), Sq8JsonlRead::Record(value) if value.len() == 32)
        );
        assert_eq!(reader.next_record().unwrap(), Sq8JsonlRead::Oversized);
        assert_eq!(
            reader.next_record().unwrap(),
            Sq8JsonlRead::Record(b"{}".to_vec())
        );
        assert_eq!(reader.next_record().unwrap(), Sq8JsonlRead::Eof);
    }

    #[test]
    fn bounded_reader_rejects_unterminated_short_and_oversized_records() {
        for input in [b"{}".as_slice(), b"0123456789".as_slice()] {
            let reader = BufReader::with_capacity(2, Cursor::new(input.to_vec()));
            let mut reader = Sq8BoundedJsonlReader::with_max_payload_bytes(reader, 4);
            let err = reader.next_record().unwrap_err();
            assert_eq!(err.kind, Sq8JsonlFramingErrorKind::UnterminatedRecord);
        }
    }

    #[test]
    fn framed_malformed_record_does_not_prevent_the_next_record() {
        let valid = br#"{"schema_version":"ullm.worker.v1","type":"shutdown"}"#;
        let mut input = b"{malformed}\n".to_vec();
        input.extend_from_slice(valid);
        input.push(b'\n');
        let reader = BufReader::with_capacity(1, Cursor::new(input));
        let mut reader = Sq8BoundedJsonlReader::with_max_payload_bytes(reader, 128);
        let Sq8JsonlRead::Record(malformed) = reader.next_record().unwrap() else {
            panic!("expected malformed framed record")
        };
        assert!(decode_sq8_worker_command(&malformed).is_err());
        let Sq8JsonlRead::Record(next) = reader.next_record().unwrap() else {
            panic!("expected next framed record")
        };
        assert_eq!(
            decode_sq8_worker_command(&next).unwrap(),
            Sq8WorkerCommand::Shutdown
        );
    }

    #[test]
    fn bounded_reader_accepts_the_real_four_mib_limit() {
        let mut input = vec![b' '; SQ8_WORKER_MAX_RECORD_BYTES];
        input.push(b'\n');
        let reader = BufReader::with_capacity(8192, Cursor::new(input));
        let mut reader = Sq8BoundedJsonlReader::new(reader);
        assert_eq!(reader.fixed_buffer_bytes(), SQ8_WORKER_MAX_RECORD_BYTES + 1);
        assert!(matches!(
            reader.next_record().unwrap(),
            Sq8JsonlRead::Record(value) if value.len() == SQ8_WORKER_MAX_RECORD_BYTES
        ));
    }

    #[test]
    fn strict_decoder_accepts_every_command_shape() {
        let generate = decode_sq8_worker_command(&valid_generate()).unwrap();
        let Sq8WorkerCommand::Generate(generate) = generate else {
            panic!("expected generate")
        };
        let request = generate.into_serving_request().unwrap();
        assert_eq!(request.prompt_token_ids, vec![1, 2, 3]);
        assert_eq!(request.sampling.seed, -7);

        assert!(matches!(
            decode_sq8_worker_command(br#"{"schema_version":"ullm.worker.v1","type":"cancel","request_id":"req-1","reason":"operator"}"#).unwrap(),
            Sq8WorkerCommand::Cancel { reason: Sq8CancelReason::Operator, .. }
        ));
        assert_eq!(
            decode_sq8_worker_command(br#"{"schema_version":"ullm.worker.v1","type":"shutdown"}"#)
                .unwrap(),
            Sq8WorkerCommand::Shutdown
        );
        assert_eq!(
            decode_sq8_worker_command(br#"{"schema_version":"ullm.worker.v2","type":"shutdown"}"#)
                .unwrap(),
            Sq8WorkerCommand::Shutdown
        );
    }

    #[test]
    fn strict_decoder_accepts_v2_reasoning_execution() {
        let payload = br#"{"schema_version":"ullm.worker.v2","type":"generate","request_id":"req-1","prompt_token_ids":[1,2,3],"max_new_tokens":16,"sampling":{"temperature":0.6,"top_p":0.95,"top_k":20,"seed":7},"eos_token_ids":[151645,151643],"reasoning":{"enabled":true,"budget_tokens":8,"dialect_id":"qwen3.5-thinking-v1","end_token_ids":[248069],"forced_end_token_ids":[248069],"reserved_answer_tokens":1}}"#;
        let Sq8WorkerCommand::Generate(command) = decode_sq8_worker_command(payload).unwrap()
        else {
            panic!("expected generate")
        };
        let reasoning = command.reasoning.as_ref().expect("reasoning contract");
        assert!(reasoning.enabled);
        assert_eq!(reasoning.budget_tokens, Some(8));
        assert_eq!(reasoning.dialect_id, "qwen3.5-thinking-v1");
        assert_eq!(reasoning.end_sequence, vec![248069]);
        assert_eq!(reasoning.forced_end_sequence, vec![248069]);
        assert_eq!(reasoning.reserved_answer_tokens, 1);
    }

    #[test]
    fn v1_rejects_reasoning_execution_without_hidden_defaults() {
        let payload = br#"{"schema_version":"ullm.worker.v1","type":"generate","request_id":"req-1","prompt_token_ids":[1,2,3],"max_new_tokens":16,"sampling":{"temperature":0.6,"top_p":0.95,"top_k":20,"seed":7},"eos_token_ids":[151645,151643],"reasoning":{"enabled":false,"budget_tokens":null,"dialect_id":"qwen3.5-thinking-v1","end_token_ids":[248069],"forced_end_token_ids":[248069],"reserved_answer_tokens":1}}"#;
        let error = decode_sq8_worker_command(payload).unwrap_err();
        assert_eq!(error.kind, Sq8WorkerProtocolErrorKind::InvalidCommand);
    }

    #[test]
    fn strict_decoder_rejects_duplicate_keys_at_every_depth() {
        for payload in [
            br#"{"schema_version":"ullm.worker.v1","schema_version":"ullm.worker.v1","type":"shutdown"}"#.as_slice(),
            br#"{"schema_version":"ullm.worker.v1","type":"generate","request_id":"r","prompt_token_ids":[1],"max_new_tokens":1,"sampling":{"temperature":0.6,"temperature":0.7,"top_p":1.0,"top_k":20,"seed":0},"eos_token_ids":[151645,151643]}"#.as_slice(),
            br#"{"schema_version":"ullm.worker.v1","type":"shutdown","unknown":{"x":1,"\u0078":2}}"#.as_slice(),
        ] {
            let err = decode_sq8_worker_command(payload).unwrap_err();
            assert_eq!(err.kind, Sq8WorkerProtocolErrorKind::InvalidCommand);
            assert_eq!(err.message, DUPLICATE_KEY_ERROR);
        }
    }

    #[test]
    fn strict_decoder_rejects_wrong_top_level_and_trailing_json() {
        for payload in [
            b"[]".as_slice(),
            b"null".as_slice(),
            b"1".as_slice(),
            b"{} {}".as_slice(),
            b"\xff".as_slice(),
        ] {
            assert_eq!(
                decode_sq8_worker_command(payload).unwrap_err().kind,
                Sq8WorkerProtocolErrorKind::InvalidCommand
            );
        }
    }

    #[test]
    fn strict_decoder_enforces_depth_16_before_typed_schema() {
        let depth_16 = format!("{}0{}", "[".repeat(16), "]".repeat(16));
        let depth_17 = format!("{}0{}", "[".repeat(17), "]".repeat(17));
        let err_16 = decode_sq8_worker_command(depth_16.as_bytes()).unwrap_err();
        assert_ne!(err_16.message, JSON_DEPTH_ERROR);
        let err_17 = decode_sq8_worker_command(depth_17.as_bytes()).unwrap_err();
        assert_eq!(err_17.message, JSON_DEPTH_ERROR);
    }

    #[test]
    fn strict_decoder_rejects_unknown_missing_wrong_version_and_wrong_direction() {
        for payload in [
            br#"{"schema_version":"ullm.worker.v1","type":"shutdown","schema":"bad"}"#.as_slice(),
            br#"{"type":"shutdown"}"#.as_slice(),
            br#"{"schema_version":"ullm.worker.v1","type":"ready"}"#.as_slice(),
        ] {
            assert_eq!(
                decode_sq8_worker_command(payload).unwrap_err().kind,
                Sq8WorkerProtocolErrorKind::InvalidCommand
            );
        }
    }

    #[test]
    fn strict_decoder_rejects_non_integer_and_overflow_integer_fields() {
        for replacement in ["1.0", "1e0", "-1", "18446744073709551616"] {
            let payload = String::from_utf8(valid_generate()).unwrap().replace(
                "\"max_new_tokens\":2",
                &format!("\"max_new_tokens\":{replacement}"),
            );
            assert_eq!(
                decode_sq8_worker_command(payload.as_bytes())
                    .unwrap_err()
                    .kind,
                Sq8WorkerProtocolErrorKind::InvalidCommand
            );
        }
    }

    #[test]
    fn strict_decoder_accepts_i64_seed_edges_and_rejects_overflow() {
        for seed in [i64::MIN.to_string(), i64::MAX.to_string()] {
            let payload = String::from_utf8(valid_generate())
                .unwrap()
                .replace("\"seed\":-7", &format!("\"seed\":{seed}"));
            let Sq8WorkerCommand::Generate(command) =
                decode_sq8_worker_command(payload.as_bytes()).unwrap()
            else {
                panic!("expected generate")
            };
            assert_eq!(command.sampling.seed.to_string(), seed);
        }
        for seed in ["-9223372036854775809", "9223372036854775808"] {
            let payload = String::from_utf8(valid_generate())
                .unwrap()
                .replace("\"seed\":-7", &format!("\"seed\":{seed}"));
            assert_eq!(
                decode_sq8_worker_command(payload.as_bytes())
                    .unwrap_err()
                    .kind,
                Sq8WorkerProtocolErrorKind::InvalidCommand
            );
        }
    }

    #[test]
    fn generate_semantic_validation_is_separate_from_structural_decode() {
        let payload = String::from_utf8(valid_generate())
            .unwrap()
            .replace("\"max_new_tokens\":2", "\"max_new_tokens\":0");
        let Sq8WorkerCommand::Generate(generate) =
            decode_sq8_worker_command(payload.as_bytes()).unwrap()
        else {
            panic!("expected generate")
        };
        let err = generate.into_serving_request().unwrap_err();
        assert_eq!(err.kind, Sq8WorkerProtocolErrorKind::InvalidRequest);
    }

    #[test]
    fn generate_sampling_validates_f64_before_narrowing_to_f32() {
        for (field, value) in [
            ("temperature", "2.0000000001"),
            ("temperature", "-1e-50"),
            ("top_p", "1.0000000001"),
            ("top_p", "1e-500"),
        ] {
            let needle = if field == "temperature" {
                "\"temperature\":0.6"
            } else {
                "\"top_p\":0.95"
            };
            let payload = String::from_utf8(valid_generate())
                .unwrap()
                .replace(needle, &format!("\"{field}\":{value}"));
            match decode_sq8_worker_command(payload.as_bytes()) {
                Ok(Sq8WorkerCommand::Generate(generate)) => assert_eq!(
                    generate.into_serving_request().unwrap_err().kind,
                    Sq8WorkerProtocolErrorKind::InvalidRequest
                ),
                Err(err) => assert_eq!(err.kind, Sq8WorkerProtocolErrorKind::InvalidCommand),
                Ok(command) => panic!("unexpected command: {command:?}"),
            }
        }
        let payload = String::from_utf8(valid_generate())
            .unwrap()
            .replace("\"temperature\":0.6", "\"temperature\":1e400");
        assert_eq!(
            decode_sq8_worker_command(payload.as_bytes())
                .unwrap_err()
                .kind,
            Sq8WorkerProtocolErrorKind::InvalidCommand
        );
    }

    #[test]
    fn prompt_length_4097_is_structural_but_semantically_invalid() {
        let prompt = std::iter::repeat_n("1", 4097).collect::<Vec<_>>().join(",");
        let payload = String::from_utf8(valid_generate()).unwrap().replace(
            "\"prompt_token_ids\":[1,2,3]",
            &format!("\"prompt_token_ids\":[{prompt}]"),
        );
        let Sq8WorkerCommand::Generate(command) =
            decode_sq8_worker_command(payload.as_bytes()).unwrap()
        else {
            panic!("expected generate")
        };
        assert_eq!(command.prompt_token_ids.len(), 4097);
        assert_eq!(
            command.into_serving_request().unwrap_err().kind,
            Sq8WorkerProtocolErrorKind::InvalidRequest
        );
    }

    #[test]
    fn request_id_rule_is_exact_ascii() {
        for valid in ["a", "A0._:-z", &"x".repeat(128)] {
            validate_worker_request_id(valid).unwrap();
        }
        for invalid in [
            "",
            "-bad",
            "bad/slash",
            "space bad",
            "nonascii-\u{3042}",
            &"x".repeat(129),
        ] {
            assert!(validate_worker_request_id(invalid).is_err());
        }
    }

    #[test]
    fn worker_control_admits_exactly_one_and_clears_only_after_flush_ack() {
        let control = Sq8WorkerControl::new();
        assert_eq!(
            control.precheck_generate().unwrap_err().kind,
            Sq8WorkerControlErrorKind::NotReady
        );
        mark_control_ready(&control);
        let first = control.admit("req-1").unwrap();
        assert_eq!(first.generation, 1);
        assert_eq!(
            control.precheck_generate().unwrap_err().kind,
            Sq8WorkerControlErrorKind::Busy
        );
        let before_ack = control.snapshot().unwrap();
        assert_eq!(before_ack.active_request_id.as_deref(), Some("req-1"));
        assert_eq!(
            control
                .begin_terminal_publication(2, "req-1")
                .unwrap_err()
                .kind,
            Sq8WorkerControlErrorKind::StaleGeneration
        );
        assert_eq!(control.snapshot().unwrap(), before_ack);

        control
            .acknowledge_terminal_flush(flushed_terminal_ack(&control, first.generation, "req-1"))
            .unwrap();
        assert!(control.snapshot().unwrap().active_request_id.is_none());
        let reused = control.admit("req-1").unwrap();
        assert_eq!(reused.generation, 2);
    }

    #[test]
    fn worker_control_busy_precedes_generate_semantic_validation() {
        let control = Sq8WorkerControl::new();
        mark_control_ready(&control);
        control.admit("req-active").unwrap();
        let payload = String::from_utf8(valid_generate())
            .unwrap()
            .replace("\"max_new_tokens\":2", "\"max_new_tokens\":0");
        let inspection = inspect_sq8_worker_command(payload.as_bytes()).unwrap();
        assert_eq!(inspection.kind, Sq8WorkerCommandKind::Generate);
        assert_eq!(
            control.precheck_generate().unwrap_err().kind,
            Sq8WorkerControlErrorKind::Busy
        );
        let Sq8WorkerCommand::Generate(invalid_generate) = inspection.decode().unwrap() else {
            panic!("expected generate")
        };
        assert_eq!(
            invalid_generate.into_serving_request().unwrap_err().kind,
            Sq8WorkerProtocolErrorKind::InvalidRequest
        );
        assert_eq!(
            control.snapshot().unwrap().active_request_id.as_deref(),
            Some("req-active")
        );

        let prompt = std::iter::repeat_n("1", 5000).collect::<Vec<_>>().join(",");
        let oversized_semantic = String::from_utf8(valid_generate()).unwrap().replace(
            "\"prompt_token_ids\":[1,2,3]",
            &format!("\"prompt_token_ids\":[{prompt}]"),
        );
        let inspection = inspect_sq8_worker_command(oversized_semantic.as_bytes()).unwrap();
        assert_eq!(inspection.kind, Sq8WorkerCommandKind::Generate);
        assert_eq!(
            control.precheck_generate().unwrap_err().kind,
            Sq8WorkerControlErrorKind::Busy
        );
        assert_eq!(
            inspection.decode().unwrap_err().kind,
            Sq8WorkerProtocolErrorKind::InvalidCommand
        );
    }

    #[test]
    fn worker_control_cancel_is_id_matched_idempotent_and_first_reason_wins() {
        let control = Sq8WorkerControl::new();
        mark_control_ready(&control);
        let admission = control.admit("req-1").unwrap();
        assert_eq!(
            control
                .cancel("req-other", Sq8CancelReason::Operator)
                .unwrap_err()
                .kind,
            Sq8WorkerControlErrorKind::UnknownRequest
        );
        assert!(!control.snapshot().unwrap().cancelled);
        assert_eq!(control.snapshot().unwrap().first_cancel_reason, None);

        let first = control
            .cancel("req-1", Sq8CancelReason::ClientDisconnect)
            .unwrap();
        assert_eq!(first.generation, admission.generation);
        assert_eq!(first.first_reason, Sq8CancelReason::ClientDisconnect);
        assert!(!first.repeated);
        let repeated = control.cancel("req-1", Sq8CancelReason::Operator).unwrap();
        assert!(repeated.repeated);
        assert_eq!(repeated.first_reason, Sq8CancelReason::ClientDisconnect);
        assert_eq!(
            control.first_cancel_reason(admission.generation).unwrap(),
            Some(Sq8CancelReason::ClientDisconnect)
        );
        let snapshot = control.snapshot().unwrap();
        assert!(snapshot.cancelled);
        assert_eq!(
            snapshot.first_cancel_reason,
            Some(Sq8CancelReason::ClientDisconnect)
        );
    }

    #[test]
    fn worker_control_releases_its_mutex_before_waiting_for_token_publication() {
        let control = Arc::new(Sq8WorkerControl::new());
        mark_control_ready(&control);
        let admission = control.admit("req-1").unwrap();
        let publication = admission.cancel.publication_guard_for_testing().unwrap();
        let (prepared_tx, prepared_rx) = mpsc::channel();
        let (cancel_done_tx, cancel_done_rx) = mpsc::channel();
        let cancel_control = Arc::clone(&control);
        let cancel_thread = std::thread::spawn(move || {
            let result =
                cancel_control.cancel_with_hook("req-1", Sq8CancelReason::Operator, || {
                    prepared_tx.send(()).unwrap()
                });
            cancel_done_tx.send(result).unwrap();
        });
        prepared_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let (snapshot_tx, snapshot_rx) = mpsc::channel();
        let snapshot_control = Arc::clone(&control);
        let snapshot_thread = std::thread::spawn(move || {
            snapshot_tx.send(snapshot_control.snapshot()).unwrap();
        });
        let snapshot_before_unlock = snapshot_rx.recv_timeout(Duration::from_millis(250));
        drop(publication);
        let result = cancel_done_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        cancel_thread.join().unwrap();
        snapshot_thread.join().unwrap();

        let snapshot = snapshot_before_unlock
            .expect("control mutex remained locked while cancel waited for publication")
            .unwrap();
        assert_eq!(
            snapshot.first_cancel_reason,
            Some(Sq8CancelReason::Operator)
        );
        assert!(!snapshot.cancelled);
        assert_eq!(result.first_reason, Sq8CancelReason::Operator);
        assert!(control.snapshot().unwrap().cancelled);
    }

    #[test]
    fn worker_control_shutdown_is_clean_when_idle_and_cancels_when_active() {
        let idle = Sq8WorkerControl::new();
        mark_control_ready(&idle);
        assert_eq!(
            idle.begin_shutdown().unwrap(),
            Sq8WorkerShutdownDisposition::Idle
        );
        assert_eq!(
            idle.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Closing
        );
        assert_eq!(
            idle.precheck_generate().unwrap_err().kind,
            Sq8WorkerControlErrorKind::Closing
        );

        let active = Sq8WorkerControl::new();
        mark_control_ready(&active);
        let admission = active.admit("req-1").unwrap();
        let shutdown = active.begin_shutdown().unwrap();
        assert_eq!(
            shutdown,
            Sq8WorkerShutdownDisposition::Cancelling(Sq8WorkerCancelResult {
                generation: admission.generation,
                first_reason: Sq8CancelReason::Shutdown,
                repeated: false,
            })
        );
        let snapshot = active.snapshot().unwrap();
        assert_eq!(snapshot.lifecycle, Sq8WorkerLifecycle::Closing);
        assert!(snapshot.cancelled);
        assert_eq!(
            snapshot.first_cancel_reason,
            Some(Sq8CancelReason::Shutdown)
        );
        assert_eq!(
            active.begin_shutdown().unwrap_err().kind,
            Sq8WorkerControlErrorKind::Closing
        );
        assert_eq!(
            active
                .cancel("req-1", Sq8CancelReason::Operator)
                .unwrap_err()
                .kind,
            Sq8WorkerControlErrorKind::Closing
        );
        active
            .acknowledge_terminal_flush(flushed_terminal_ack(
                &active,
                admission.generation,
                "req-1",
            ))
            .unwrap();
        assert_eq!(
            active.begin_shutdown().unwrap_err().kind,
            Sq8WorkerControlErrorKind::Closing
        );
    }

    #[test]
    fn worker_control_failure_retains_active_ownership() {
        assert_eq!(
            Sq8WorkerControl::new()
                .begin_terminal_publication(1, "req-1")
                .unwrap_err()
                .kind,
            Sq8WorkerControlErrorKind::NotReady
        );
        let control = Sq8WorkerControl::new();
        mark_control_ready(&control);
        let admission = control.admit("req-1").unwrap();
        control.mark_failed().unwrap();
        let snapshot = control.snapshot().unwrap();
        assert_eq!(snapshot.lifecycle, Sq8WorkerLifecycle::Failed);
        assert_eq!(snapshot.active_generation, Some(admission.generation));
        assert_eq!(
            control.precheck_generate().unwrap_err().kind,
            Sq8WorkerControlErrorKind::Failed
        );
        assert_eq!(
            control
                .begin_terminal_publication(admission.generation, &admission.request_id)
                .unwrap_err()
                .kind,
            Sq8WorkerControlErrorKind::Failed
        );
    }

    #[test]
    fn released_flush_boundary_keeps_the_slot_busy_until_ack() {
        let control = Sq8WorkerControl::new();
        mark_control_ready(&control);
        let admission = control.admit("req-1").unwrap();
        let released =
            Sq8WorkerEvent::released("req-1", Sq8ReleaseOutcomeEvent::Length, None, 3, 1).unwrap();

        assert_eq!(
            control.precheck_generate().unwrap_err().kind,
            Sq8WorkerControlErrorKind::Busy
        );
        let mut writer = Sq8OrderedJsonlWriter::new(FlushCountingWriter::default());
        let permit = control
            .begin_terminal_publication(admission.generation, "req-1")
            .unwrap();
        assert_eq!(
            control
                .begin_terminal_publication(admission.generation, "req-1")
                .unwrap_err()
                .kind,
            Sq8WorkerControlErrorKind::Internal
        );
        assert!(control.snapshot().unwrap().terminal_in_flight);
        let acknowledgement = writer
            .write_active_terminal_event(permit, &released)
            .unwrap();
        assert_eq!(
            control.precheck_generate().unwrap_err().kind,
            Sq8WorkerControlErrorKind::Busy
        );
        control.acknowledge_terminal_flush(acknowledgement).unwrap();
        control.precheck_generate().unwrap();
    }

    #[test]
    fn event_serialization_has_exact_release_field_sets() {
        let timings = Sq8WorkerTimings::from_elapsed_millis(3, 12.0, 2, 8.0).unwrap();
        let normal =
            Sq8WorkerEvent::released("req-1", Sq8ReleaseOutcomeEvent::Length, None, 3, 2).unwrap();
        let timed = Sq8WorkerEvent::released_with_timings(
            "req-timed",
            Sq8ReleaseOutcomeEvent::Length,
            None,
            3,
            2,
            timings,
        )
        .unwrap();
        let cancelled = Sq8WorkerEvent::released(
            "req-2",
            Sq8ReleaseOutcomeEvent::Cancelled,
            Some(Sq8CancelReason::SlowClient),
            3,
            1,
        )
        .unwrap();
        let normal = serde_json::to_value(normal).unwrap();
        let timed = serde_json::to_value(timed).unwrap();
        let cancelled = serde_json::to_value(cancelled).unwrap();
        assert!(normal.get("cancel_reason").is_none());
        assert!(normal.get("timings").is_none());
        assert_eq!(cancelled["cancel_reason"], "slow_client");
        assert!(cancelled.get("timings").is_none());
        assert_eq!(timed["timings"]["cache_n"], 0);
        assert_eq!(timed["timings"]["prompt_n"], 3);
        assert_eq!(timed["timings"]["prompt_ms"], 12.0);
        assert_eq!(timed["timings"]["prompt_per_token_ms"], 4.0);
        assert_eq!(timed["timings"]["prompt_per_second"], 250.0);
        assert_eq!(timed["timings"]["predicted_n"], 2);
        assert_eq!(timed["timings"]["predicted_ms"], 8.0);
        assert_eq!(timed["timings"]["predicted_per_token_ms"], 4.0);
        assert_eq!(timed["timings"]["predicted_per_second"], 250.0);
        assert_eq!(normal["reset_complete"], true);
        assert_eq!(normal["schema_version"], SQ8_WORKER_SCHEMA_VERSION);
    }

    #[test]
    fn release_timings_reject_cancelled_mismatch_and_nonfinite_values() {
        let timings = Sq8WorkerTimings::from_elapsed_millis(3, 12.0, 2, 0.001).unwrap();
        assert!(
            Sq8WorkerEvent::released_with_timings(
                "req-cancelled",
                Sq8ReleaseOutcomeEvent::Cancelled,
                Some(Sq8CancelReason::Operator),
                3,
                2,
                timings,
            )
            .is_err()
        );
        assert!(
            Sq8WorkerEvent::released_with_timings(
                "req-mismatch",
                Sq8ReleaseOutcomeEvent::Length,
                None,
                3,
                1,
                timings,
            )
            .is_err()
        );
        assert!(Sq8WorkerTimings::from_elapsed_millis(3, f64::NAN, 2, 1.0).is_err());
        assert!(Sq8WorkerTimings::from_elapsed_millis(3, 1.0, 2, f64::INFINITY).is_err());
        assert!(Sq8WorkerTimings::from_elapsed_millis(3, 1.0, 2, 0.000_999).is_err());
    }

    #[derive(Default)]
    struct FlushCountingWriter {
        bytes: Vec<u8>,
        flushes: usize,
    }

    impl Write for FlushCountingWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.bytes.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    struct FailingWriter {
        fail_flush: bool,
        remaining_bytes: usize,
    }

    impl Write for FailingWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            if self.remaining_bytes == 0 {
                return Err(std::io::Error::other("injected write failure"));
            }
            let written = buffer.len().min(self.remaining_bytes);
            self.remaining_bytes -= written;
            Ok(written)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            if self.fail_flush {
                Err(std::io::Error::other("injected flush failure"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn ordered_writer_emits_one_lf_and_flushes_every_event() {
        let mut writer = Sq8OrderedJsonlWriter::new(FlushCountingWriter::default());
        writer.write_event(&Sq8WorkerEvent::ready()).unwrap();
        writer
            .write_event(&Sq8WorkerEvent::started("req-1", 3))
            .unwrap();
        let writer = writer.into_inner();
        assert_eq!(writer.flushes, 2);
        assert_eq!(
            writer.bytes.iter().filter(|byte| **byte == b'\n').count(),
            2
        );
        for line in writer.bytes.split(|byte| *byte == b'\n').take(2) {
            let value: Value = serde_json::from_slice(line).unwrap();
            assert_eq!(value["schema_version"], SQ8_WORKER_SCHEMA_VERSION);
        }
    }

    #[test]
    fn ordered_writer_poison_is_permanent_after_write_or_flush_failure() {
        for inner in [
            FailingWriter {
                fail_flush: false,
                remaining_bytes: 4,
            },
            FailingWriter {
                fail_flush: true,
                remaining_bytes: usize::MAX,
            },
        ] {
            let mut writer = Sq8OrderedJsonlWriter::new(inner);
            assert!(writer.write_event(&Sq8WorkerEvent::ready()).is_err());
            assert!(writer.is_failed());
            assert_eq!(
                writer.write_event(&Sq8WorkerEvent::ready()).unwrap_err(),
                "SQ8 worker event writer is failed"
            );
        }
    }

    #[test]
    fn ready_flush_failure_cannot_open_admission() {
        let control = Sq8WorkerControl::new();
        let mut writer = Sq8OrderedJsonlWriter::new(FailingWriter {
            fail_flush: true,
            remaining_bytes: usize::MAX,
        });
        assert!(writer.write_ready_event(&Sq8WorkerEvent::ready()).is_err());
        assert!(writer.is_failed());
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Loading
        );
        assert_eq!(
            control.precheck_generate().unwrap_err().kind,
            Sq8WorkerControlErrorKind::NotReady
        );
    }

    #[test]
    fn busy_error_flush_failure_does_not_clear_the_active_slot() {
        let control = Sq8WorkerControl::new();
        mark_control_ready(&control);
        let admission = control.admit("req-active").unwrap();
        let busy = Sq8WorkerEvent::error(
            Some("req-second".into()),
            Sq8WorkerErrorCode::Busy,
            "one request is already active",
        )
        .unwrap();
        let mut writer = Sq8OrderedJsonlWriter::new(FailingWriter {
            fail_flush: true,
            remaining_bytes: usize::MAX,
        });
        assert!(writer.write_event(&busy).is_err());
        assert_eq!(
            control.snapshot().unwrap().active_generation,
            Some(admission.generation)
        );
    }

    #[test]
    fn released_flush_failure_cannot_produce_a_clear_ack() {
        let control = Sq8WorkerControl::new();
        mark_control_ready(&control);
        let admission = control.admit("req-active").unwrap();
        let released =
            Sq8WorkerEvent::released("req-active", Sq8ReleaseOutcomeEvent::Length, None, 3, 1)
                .unwrap();
        let mut writer = Sq8OrderedJsonlWriter::new(FailingWriter {
            fail_flush: true,
            remaining_bytes: usize::MAX,
        });
        assert!(
            writer
                .write_active_terminal_event(
                    control
                        .begin_terminal_publication(admission.generation, &admission.request_id,)
                        .unwrap(),
                    &released,
                )
                .is_err()
        );
        assert!(writer.is_failed());
        assert_eq!(
            control.snapshot().unwrap().active_generation,
            Some(admission.generation)
        );
    }

    #[test]
    fn ordered_writer_rejects_invalid_event_before_output() {
        let invalid = Sq8WorkerEvent::Token {
            schema_version: SQ8_WORKER_SCHEMA_VERSION.to_string(),
            request_id: "req-1".into(),
            index: 0,
            token_id: QWEN3_14B_VOCAB_SIZE,
        };
        let mut writer = Sq8OrderedJsonlWriter::new(FlushCountingWriter::default());
        assert!(writer.write_event(&invalid).is_err());
        assert!(writer.is_failed());
        assert!(writer.into_inner().bytes.is_empty());
    }

    #[test]
    fn worker_error_code_fixes_recoverability_and_bounds_message() {
        let event = Sq8WorkerEvent::error(
            Some("req-1".into()),
            Sq8WorkerErrorCode::Busy,
            "one request is already active",
        )
        .unwrap();
        let value = serde_json::to_value(event).unwrap();
        assert_eq!(value["recoverable"], true);
        assert!(
            Sq8WorkerEvent::error(
                None,
                Sq8WorkerErrorCode::RuntimeFailed,
                "x".repeat(SQ8_WORKER_MAX_ERROR_MESSAGE_BYTES + 1),
            )
            .is_err()
        );
    }

    fn batched_progress_points(prompt_tokens: usize) -> Vec<usize> {
        let mut tracker = Sq8PromptProgressTracker::new(prompt_tokens).unwrap();
        let mut points = Vec::new();
        let mut processed = 0;
        while processed < prompt_tokens {
            let remaining = prompt_tokens - processed;
            let width = if remaining >= 128 { 128 } else { 1 };
            if processed + width == prompt_tokens {
                break;
            }
            processed += width;
            if let Some(point) = tracker.observe_unit(processed, width).unwrap() {
                points.push(point);
            }
        }
        if let Some(point) = tracker.observe_transition().unwrap() {
            points.push(point);
        }
        points
    }

    fn tokenwise_progress_points(prompt_tokens: usize) -> Vec<usize> {
        let mut tracker = Sq8PromptProgressTracker::new(prompt_tokens).unwrap();
        let mut points = Vec::new();
        for processed in 1..=prompt_tokens {
            if let Some(point) = tracker.observe_unit(processed, 1).unwrap() {
                points.push(point);
            }
        }
        assert_eq!(tracker.observe_transition().unwrap(), None);
        points
    }

    #[test]
    fn tokenwise_and_batched_prefill_have_the_same_wire_progress() {
        for (prompt_tokens, expected) in [
            (127, vec![127]),
            (128, vec![128]),
            (129, vec![128, 129]),
            (1011, vec![128, 256, 384, 512, 640, 768, 896, 1011]),
        ] {
            assert_eq!(batched_progress_points(prompt_tokens), expected);
            assert_eq!(tokenwise_progress_points(prompt_tokens), expected);
        }

        assert_eq!(batched_progress_points(256), vec![128, 256]);
        assert_eq!(batched_progress_points(257), vec![128, 256, 257]);
        let points = batched_progress_points(4095);
        assert_eq!(points.first(), Some(&128));
        assert_eq!(points.get(points.len() - 2), Some(&3968));
        assert_eq!(points.last(), Some(&4095));
        assert_eq!(points.len(), 32);
    }

    #[test]
    fn prompt_progress_accepts_bounded_execution_widths() {
        for execution_width in [2, 3, 8, 127, 128] {
            let mut tracker = Sq8PromptProgressTracker::new(execution_width + 1).unwrap();
            assert_eq!(
                tracker
                    .observe_unit(execution_width, execution_width)
                    .unwrap(),
                (execution_width == 128).then_some(execution_width)
            );
            assert_eq!(
                tracker.observe_transition().unwrap(),
                Some(execution_width + 1)
            );
        }
    }

    #[test]
    fn prompt_progress_unit_sequences_preserve_wire_cadence() {
        for (prompt_tokens, units, expected_events) in [
            (127, [127].as_slice(), vec![127]),
            (129, [128, 1].as_slice(), vec![128, 129]),
            (255, [128, 127].as_slice(), vec![128, 255]),
        ] {
            let mut tracker = Sq8PromptProgressTracker::new(prompt_tokens).unwrap();
            let mut processed = 0;
            let mut events = Vec::new();
            for &width in units {
                processed += width;
                if let Some(event) = tracker.observe_unit(processed, width).unwrap() {
                    events.push(event);
                }
            }
            assert_eq!(tracker.observe_transition().unwrap(), None);
            assert_eq!(events, expected_events);
        }
    }

    #[test]
    fn prompt_progress_transition_accepts_any_bounded_final_tail() {
        for final_tail in [1, 2, 3, 8, 127, 128] {
            let mut tracker = Sq8PromptProgressTracker::new(final_tail).unwrap();
            assert_eq!(tracker.observe_transition().unwrap(), Some(final_tail));
        }
    }

    #[test]
    fn prompt_progress_rejects_non_contiguous_invalid_and_overshooting_units() {
        let mut zero_width = Sq8PromptProgressTracker::new(129).unwrap();
        assert!(zero_width.observe_unit(0, 0).is_err());

        let mut oversized_width = Sq8PromptProgressTracker::new(129).unwrap();
        assert!(
            oversized_width
                .observe_unit(
                    MAX_WORKER_PROGRESS_EXECUTION_WIDTH + 1,
                    MAX_WORKER_PROGRESS_EXECUTION_WIDTH + 1
                )
                .is_err()
        );

        let mut non_contiguous = Sq8PromptProgressTracker::new(129).unwrap();
        assert!(non_contiguous.observe_unit(2, 1).is_err());

        let mut unaligned_batch = Sq8PromptProgressTracker::new(256).unwrap();
        assert_eq!(unaligned_batch.observe_unit(1, 1).unwrap(), None);
        assert_eq!(unaligned_batch.observe_unit(129, 128).unwrap(), Some(129));

        let mut overshoot = Sq8PromptProgressTracker::new(127).unwrap();
        for processed in 1..127 {
            overshoot.observe_unit(processed, 1).unwrap();
        }
        assert!(overshoot.observe_unit(128, 1).is_err());

        let mut short_batch = Sq8PromptProgressTracker::new(127).unwrap();
        assert!(short_batch.observe_unit(128, 128).is_err());

        let mut remaining_overflow = Sq8PromptProgressTracker {
            prompt_tokens: 1,
            last_observed: 2,
            last_emitted: 0,
            transition_emitted: false,
        };
        assert!(remaining_overflow.observe_unit(3, 1).is_err());
        assert!(remaining_overflow.observe_transition().is_err());

        let mut emission_overflow = Sq8PromptProgressTracker {
            prompt_tokens: 1,
            last_observed: 0,
            last_emitted: usize::MAX,
            transition_emitted: false,
        };
        assert!(emission_overflow.observe_unit(1, 1).is_err());

        let mut transition_without_tail = Sq8PromptProgressTracker {
            prompt_tokens: 1,
            last_observed: 1,
            last_emitted: 0,
            transition_emitted: false,
        };
        assert!(transition_without_tail.observe_transition().is_err());

        let mut transition_emission_overflow = Sq8PromptProgressTracker {
            prompt_tokens: 2,
            last_observed: 0,
            last_emitted: 1,
            transition_emitted: false,
        };
        assert!(transition_emission_overflow.observe_transition().is_err());
    }
}
