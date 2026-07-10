// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Synchronous active1/waiting0 SQ8 serving session contracts.
//!
//! This module is separate from `sq8_generation_runtime`: the P7 fixed request and its audited
//! result schemas remain unchanged while serving gains variable prompt lengths and reusable state.

use crate::decoder::PagedDecodeShape;
use crate::sq8_layer_oracle::{
    QWEN3_14B_HEAD_DIM, QWEN3_14B_KV_HEADS, QWEN3_14B_Q_HEADS, QWEN3_14B_VALUE_DIM,
};
use crate::sq8_model_head_runtime::QWEN3_14B_VOCAB_SIZE;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
}

impl fmt::Display for Sq8ServingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Sq8ServingError {}

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
}
