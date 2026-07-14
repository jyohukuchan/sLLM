// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Quantization-independent inference request, cancellation, and release contracts.

use serde::Serialize;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: i64,
}

impl SamplingParams {
    pub const fn greedy_with_top_k(seed: i64, top_k: usize) -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k,
            seed,
        }
    }

    pub fn validate_with_top_k(&self, top_k: usize) -> Result<(), InferenceError> {
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(InferenceError::invalid_request(format!(
                "temperature must be finite and in 0..=2, got {}",
                self.temperature
            )));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(InferenceError::invalid_request(format!(
                "top_p must be finite and in 0<top_p<=1, got {}",
                self.top_p
            )));
        }
        if self.top_k != top_k {
            return Err(InferenceError::invalid_request(format!(
                "top_k must be {top_k}, got {}",
                self.top_k
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt_token_ids: Vec<usize>,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<usize>,
    pub sampling: SamplingParams,
    pub reasoning: Option<crate::reasoning::ReasoningExecution>,
    test_only_ignore_eos: bool,
}

impl InferenceRequest {
    pub fn new_with_eos(
        request_id: impl Into<String>,
        prompt_token_ids: Vec<usize>,
        max_new_tokens: usize,
        eos_token_ids: Vec<usize>,
        sampling: SamplingParams,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            prompt_token_ids,
            max_new_tokens,
            eos_token_ids,
            sampling,
            reasoning: None,
            test_only_ignore_eos: false,
        }
    }

    pub fn test_only_ignores_eos(&self) -> bool {
        self.test_only_ignore_eos
    }

    pub(crate) fn ignore_eos_for_testing(mut self) -> Self {
        self.test_only_ignore_eos = true;
        self
    }

    pub fn validate_for_worker(
        &self,
        context_tokens: usize,
        max_new_tokens: usize,
        vocab_size: usize,
        eos_token_ids: &[usize],
        top_k: usize,
    ) -> Result<(), InferenceError> {
        validate_request_id(&self.request_id)?;
        if context_tokens == 0
            || self.prompt_token_ids.is_empty()
            || self.prompt_token_ids.len() > context_tokens
        {
            return Err(InferenceError::invalid_request(format!(
                "prompt token count must be in 1..={}, got {}",
                context_tokens,
                self.prompt_token_ids.len()
            )));
        }
        if let Some((index, token_id)) = self
            .prompt_token_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token_id)| *token_id >= vocab_size)
        {
            return Err(InferenceError::invalid_request(format!(
                "prompt_token_ids[{index}]={token_id} exceeds vocabulary size {vocab_size}"
            )));
        }
        if !(1..=max_new_tokens).contains(&self.max_new_tokens) {
            return Err(InferenceError::invalid_request(format!(
                "max_new_tokens must be in 1..={}, got {}",
                max_new_tokens, self.max_new_tokens
            )));
        }
        let reserved_tokens = self
            .prompt_token_ids
            .len()
            .checked_add(self.max_new_tokens)
            .ok_or_else(|| InferenceError::invalid_request("context token count overflows"))?;
        if reserved_tokens > context_tokens {
            return Err(InferenceError::invalid_request(format!(
                "prompt plus completion exceeds context: requested={reserved_tokens} context={context_tokens}"
            )));
        }
        if self.eos_token_ids != eos_token_ids {
            return Err(InferenceError::invalid_request(format!(
                "eos_token_ids must be {:?}, got {:?}",
                eos_token_ids, self.eos_token_ids
            )));
        }
        self.sampling
            .validate_with_top_k(top_k)
            .and_then(|()| self.validate_reasoning_reservation())
    }

    /// Validates the AQ4 benchmark's prefill-only request without weakening the
    /// ordinary generation contract, which continues to require at least one token.
    pub fn validate_prefill_only_for_worker(
        &self,
        context_tokens: usize,
        vocab_size: usize,
        eos_token_ids: &[usize],
        top_k: usize,
    ) -> Result<(), InferenceError> {
        validate_request_id(&self.request_id)?;
        if context_tokens == 0
            || self.prompt_token_ids.is_empty()
            || self.prompt_token_ids.len() > context_tokens
        {
            return Err(InferenceError::invalid_request(format!(
                "prefill-only prompt token count must be in 1..={}, got {}",
                context_tokens,
                self.prompt_token_ids.len()
            )));
        }
        if let Some((index, token_id)) = self
            .prompt_token_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token_id)| *token_id >= vocab_size)
        {
            return Err(InferenceError::invalid_request(format!(
                "prompt_token_ids[{index}]={token_id} exceeds vocabulary size {vocab_size}"
            )));
        }
        if self.max_new_tokens != 0 || self.reasoning.is_some() {
            return Err(InferenceError::invalid_request(
                "prefill-only request requires max_new_tokens=0 and no reasoning execution",
            ));
        }
        if self.eos_token_ids != eos_token_ids {
            return Err(InferenceError::invalid_request(format!(
                "eos_token_ids must be {:?}, got {:?}",
                eos_token_ids, self.eos_token_ids
            )));
        }
        self.sampling.validate_with_top_k(top_k)
    }

    fn validate_reasoning_reservation(&self) -> Result<(), InferenceError> {
        let Some(reasoning) = self.reasoning.as_ref().filter(|value| value.enabled) else {
            return Ok(());
        };
        if reasoning.dialect_id.is_empty()
            || reasoning.end_sequence.is_empty()
            || reasoning.forced_end_sequence.is_empty()
            || reasoning.end_sequence != reasoning.forced_end_sequence
            || reasoning.reserved_answer_tokens == 0
        {
            return Err(InferenceError::invalid_request(
                "reasoning execution contract is invalid",
            ));
        }
        let reserved = reasoning
            .budget_tokens
            .unwrap_or(0)
            .checked_add(reasoning.forced_end_sequence.len())
            .and_then(|value| value.checked_add(reasoning.reserved_answer_tokens))
            .ok_or_else(|| InferenceError::invalid_request("reasoning reservation overflows"))?;
        if reserved > self.max_new_tokens {
            return Err(InferenceError::invalid_request(
                "reasoning reservation exceeds max_new_tokens",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct CancellationState {
    flag: AtomicBool,
    publication: Mutex<()>,
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    inner: Arc<CancellationState>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        match self.cancel_checked() {
            Ok(()) => {}
            Err(_) => self.inner.flag.store(true, Ordering::Release),
        }
    }

    pub fn cancel_checked(&self) -> Result<(), String> {
        let _publication = self.publication_guard()?;
        self.inner.flag.store(true, Ordering::Release);
        Ok(())
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.flag.load(Ordering::Acquire)
    }

    pub(crate) fn publication_guard(&self) -> Result<MutexGuard<'_, ()>, String> {
        self.inner
            .publication
            .lock()
            .map_err(|_| "SQ8 cancellation publication mutex is poisoned".to_string())
    }

    #[cfg(test)]
    pub(crate) fn publication_is_locked(&self) -> Result<bool, String> {
        match self.inner.publication.try_lock() {
            Ok(_publication) => Ok(false),
            Err(std::sync::TryLockError::WouldBlock) => Ok(true),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                Err("SQ8 cancellation publication mutex is poisoned".to_string())
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn publication_guard_for_testing(&self) -> Result<MutexGuard<'_, ()>, String> {
        self.publication_guard()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseOutcome {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningUsage {
    pub reasoning_tokens: usize,
    pub forced_end_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseSummary {
    pub request_id: String,
    pub outcome: ReleaseOutcome,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub reasoning_usage: Option<ReasoningUsage>,
    pub reset_complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GenerationTimings {
    pub cache_n: usize,
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
}

impl GenerationTimings {
    pub fn from_elapsed_millis_with_limits(
        prompt_n: usize,
        prompt_ms: f64,
        predicted_n: usize,
        predicted_ms: f64,
        max_prompt_tokens: usize,
        max_predicted_tokens: usize,
    ) -> Option<Self> {
        if !(1..=max_prompt_tokens).contains(&prompt_n)
            || !(1..=max_predicted_tokens).contains(&predicted_n)
            || !prompt_ms.is_finite()
            || prompt_ms <= 0.0
            || !predicted_ms.is_finite()
            || predicted_ms < 0.001
        {
            return None;
        }
        let value = Self {
            cache_n: 0,
            prompt_n,
            prompt_ms,
            prompt_per_token_ms: prompt_ms / prompt_n as f64,
            prompt_per_second: 1e3 / prompt_ms * prompt_n as f64,
            predicted_n,
            predicted_ms,
            predicted_per_token_ms: predicted_ms / predicted_n as f64,
            predicted_per_second: 1e3 / predicted_ms * predicted_n as f64,
        };
        value
            .validates_release(prompt_n, predicted_n)
            .then_some(value)
    }

    pub fn validates_release(&self, prompt_tokens: usize, completion_tokens: usize) -> bool {
        let positive_finite = [
            self.prompt_ms,
            self.prompt_per_token_ms,
            self.prompt_per_second,
            self.predicted_ms,
            self.predicted_per_token_ms,
            self.predicted_per_second,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0);
        self.cache_n == 0
            && self.prompt_n == prompt_tokens
            && self.predicted_n == completion_tokens
            && self.predicted_ms >= 0.001
            && positive_finite
            && timing_value_matches(
                self.prompt_per_token_ms,
                self.prompt_ms / self.prompt_n as f64,
            )
            && timing_value_matches(
                self.prompt_per_second,
                1e3 / self.prompt_ms * self.prompt_n as f64,
            )
            && timing_value_matches(
                self.predicted_per_token_ms,
                self.predicted_ms / self.predicted_n as f64,
            )
            && timing_value_matches(
                self.predicted_per_second,
                1e3 / self.predicted_ms * self.predicted_n as f64,
            )
    }
}

fn timing_value_matches(actual: f64, expected: f64) -> bool {
    let tolerance = (expected.abs() * 1e-12).max(1e-12);
    (actual - expected).abs() <= tolerance
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceErrorKind {
    InvalidRequest,
    InvalidConfiguration,
    InvalidState,
    FatalRuntime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceError {
    pub kind: InferenceErrorKind,
    pub message: String,
}

impl InferenceError {
    pub(crate) fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            kind: InferenceErrorKind::InvalidRequest,
            message: message.into(),
        }
    }

    pub(crate) fn invalid_configuration(message: impl Into<String>) -> Self {
        Self {
            kind: InferenceErrorKind::InvalidConfiguration,
            message: message.into(),
        }
    }

    pub(crate) fn invalid_state(message: impl Into<String>) -> Self {
        Self {
            kind: InferenceErrorKind::InvalidState,
            message: message.into(),
        }
    }

    pub(crate) fn fatal_runtime(message: impl Into<String>) -> Self {
        Self {
            kind: InferenceErrorKind::FatalRuntime,
            message: message.into(),
        }
    }
}

impl fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for InferenceError {}

fn validate_request_id(value: &str) -> Result<(), InferenceError> {
    let bytes = value.as_bytes();
    if bytes.is_empty() || bytes.len() > 128 {
        return Err(InferenceError::invalid_request(format!(
            "request_id must contain 1..=128 ASCII bytes, got {}",
            bytes.len()
        )));
    }
    if !bytes[0].is_ascii_alphanumeric()
        || bytes[1..].iter().any(|byte| {
            !byte.is_ascii_alphanumeric() && !matches!(*byte, b'.' | b'_' | b':' | b'-')
        })
    {
        return Err(InferenceError::invalid_request(format!(
            "request_id has invalid syntax: {value:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_model_contract_is_not_tied_to_legacy_sq8_defaults() {
        let sampling = SamplingParams::greedy_with_top_k(7, 1);
        let request = InferenceRequest::new_with_eos(
            "aq4-request",
            vec![248_000],
            4,
            vec![248_044, 248_046],
            sampling,
        );
        request
            .validate_for_worker(4096, 512, 248_320, &[248_044, 248_046], 1)
            .unwrap();
    }

    #[test]
    fn cancellation_is_shared_and_monotonic() {
        let first = CancellationToken::new();
        let second = first.clone();
        assert!(!second.is_cancelled());
        first.cancel();
        assert!(first.is_cancelled());
        assert!(second.is_cancelled());
    }

    #[test]
    fn generation_timings_use_llama_server_counts_with_explicit_limits() {
        let timings =
            GenerationTimings::from_elapsed_millis_with_limits(4, 2.0, 3, 10.0, 4096, 512).unwrap();
        assert_eq!(timings.prompt_per_second, 2000.0);
        assert_eq!(timings.predicted_per_second, 300.0);
        assert!(timings.validates_release(4, 3));
        assert!(!timings.validates_release(4, 2));
    }

    #[test]
    fn reasoning_reservation_is_checked_before_worker_execution() {
        let sampling = SamplingParams::greedy_with_top_k(7, 1);
        let mut request = InferenceRequest::new_with_eos(
            "reasoning-request",
            vec![248_000],
            6,
            vec![248_044, 248_046],
            sampling,
        );
        request.reasoning = Some(crate::reasoning::ReasoningExecution {
            enabled: true,
            budget_tokens: Some(4),
            dialect_id: "synthetic.multi-token.v1".into(),
            end_sequence: vec![248_068, 248_069],
            forced_end_sequence: vec![248_068, 248_069],
            reserved_answer_tokens: 1,
        });
        assert!(
            request
                .validate_for_worker(4096, 512, 248_320, &[248_044, 248_046], 1)
                .is_err()
        );
        request.max_new_tokens = 7;
        assert!(
            request
                .validate_for_worker(4096, 512, 248_320, &[248_044, 248_046], 1)
                .is_ok()
        );
    }
}
