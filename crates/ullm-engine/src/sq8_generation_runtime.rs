// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Fixed, fail-closed Qwen3-14B SQ8 generation loop for the first P7 workload.
//!
//! The runtime intentionally supports one B=1 request with raw prompt token IDs `1..=8`,
//! greedy sampling, a 16-token context, and at most eight generated tokens. The narrow contract
//! keeps the first real generation path auditable: prefill produces token 0, every later token is
//! produced by an M=1 paged decode whose input is the preceding generated token, and all scheduler
//! and KV-cache progress is checked before the sole allocation is released.

use crate::decoder::{PagedDecodeShape, PagedDecodeState};
use crate::scheduler::{
    KvBlockAllocatorStats, Request, RequestId, SchedulerDecodeRequest, SchedulerState,
};
use crate::sq_canonical::Sq8CanonicalArtifact;
use crate::sq8_embedding_runtime::{
    Qwen3Sq8EmbeddingRuntime, Sq8EmbeddingDeviceIdentity, Sq8EmbeddingExecutionReport,
};
use crate::sq8_layer_oracle::{
    QWEN3_14B_HEAD_DIM, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_KV_HEADS, QWEN3_14B_Q_HEADS,
    QWEN3_14B_VALUE_DIM,
};
use crate::sq8_layer_runtime::{
    QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS, QWEN3_14B_SQ8_LAYER_PROJECTIONS,
    Qwen3Sq8LayerNormValues, Sq8LayerExecutionProfile,
};
use crate::sq8_model_head_runtime::{
    QWEN3_14B_VOCAB_SIZE, Qwen3Sq8ModelHeadRuntime, Sq8ModelHeadDeviceIdentity,
    Sq8ModelHeadM1Result, Sq8ModelHeadResult, validate_qwen3_14b_sq8_r9700_device_info,
    validate_sq8_model_head_tensor_health,
};
use crate::sq8_stack_runtime::{
    QWEN3_14B_SQ8_STACK_LAYERS, Qwen3Sq8PagedDecodeRuntime, Qwen3Sq8StackRuntime,
    Sq8PagedStackExecutionReport, Sq8PagedStackPhase,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;
use ullm_runtime_sys::{DeviceInfo, RuntimeBuffer, RuntimeContext, RuntimeStream};

pub const QWEN3_14B_SQ8_GENERATION_PROMPT_TOKEN_IDS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
pub const QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS: usize =
    QWEN3_14B_SQ8_GENERATION_PROMPT_TOKEN_IDS.len();
pub const QWEN3_14B_SQ8_GENERATION_CONTEXT_TOKENS: usize = 16;
pub const QWEN3_14B_SQ8_GENERATION_BLOCK_TOKENS: usize = 16;
pub const QWEN3_14B_SQ8_GENERATION_MAX_NEW_TOKENS: usize = 8;
pub const QWEN3_14B_SQ8_GENERATION_EOS_TOKEN_ID: usize = 151_645;
pub const QWEN3_14B_SQ8_GENERATION_REQUEST_ID: RequestId = RequestId(1);
pub const QWEN3_14B_SQ8_GENERATION_TOP_LOGITS: usize = 10;

const GENERATION_KV_BLOCKS: u32 = 1;
const SHA256_HEX_LEN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sq8GenerationSampling {
    pub temperature: f32,
    pub greedy: bool,
}

impl Sq8GenerationSampling {
    pub const fn greedy_temperature_zero() -> Self {
        Self {
            temperature: 0.0,
            greedy: true,
        }
    }

    pub fn validate_contract(&self) -> Result<(), String> {
        if !self.greedy || self.temperature.to_bits() != 0.0_f32.to_bits() {
            return Err(format!(
                "SQ8 generation requires greedy temperature=0 sampling, got greedy={} temperature={}",
                self.greedy, self.temperature
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8GenerationRequest {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<usize>,
    pub sampling: Sq8GenerationSampling,
    pub eos_token_id: usize,
    pub max_new_tokens: usize,
}

impl Sq8GenerationRequest {
    pub fn fixed(max_new_tokens: usize) -> Self {
        Self {
            request_id: QWEN3_14B_SQ8_GENERATION_REQUEST_ID,
            prompt_token_ids: QWEN3_14B_SQ8_GENERATION_PROMPT_TOKEN_IDS.to_vec(),
            sampling: Sq8GenerationSampling::greedy_temperature_zero(),
            eos_token_id: QWEN3_14B_SQ8_GENERATION_EOS_TOKEN_ID,
            max_new_tokens,
        }
    }

    pub fn validate_contract(&self) -> Result<(), String> {
        if self.request_id != QWEN3_14B_SQ8_GENERATION_REQUEST_ID {
            return Err(format!(
                "SQ8 generation request ID mismatch: expected={:?} actual={:?}",
                QWEN3_14B_SQ8_GENERATION_REQUEST_ID, self.request_id
            ));
        }
        if self.prompt_token_ids != QWEN3_14B_SQ8_GENERATION_PROMPT_TOKEN_IDS {
            return Err(format!(
                "SQ8 generation requires raw prompt IDs {:?}, got {:?}",
                QWEN3_14B_SQ8_GENERATION_PROMPT_TOKEN_IDS, self.prompt_token_ids
            ));
        }
        self.sampling.validate_contract()?;
        if self.eos_token_id != QWEN3_14B_SQ8_GENERATION_EOS_TOKEN_ID {
            return Err(format!(
                "SQ8 generation EOS token mismatch: expected={} actual={}",
                QWEN3_14B_SQ8_GENERATION_EOS_TOKEN_ID, self.eos_token_id
            ));
        }
        if !(1..=QWEN3_14B_SQ8_GENERATION_MAX_NEW_TOKENS).contains(&self.max_new_tokens) {
            return Err(format!(
                "SQ8 generation max_new_tokens must be in 1..={}, got {}",
                QWEN3_14B_SQ8_GENERATION_MAX_NEW_TOKENS, self.max_new_tokens
            ));
        }
        let context_tokens = self
            .prompt_token_ids
            .len()
            .checked_add(self.max_new_tokens)
            .ok_or_else(|| "SQ8 generation context length overflows".to_string())?;
        if context_tokens > QWEN3_14B_SQ8_GENERATION_CONTEXT_TOKENS {
            return Err(format!(
                "SQ8 generation request exceeds context: requested={context_tokens} context={QWEN3_14B_SQ8_GENERATION_CONTEXT_TOKENS}"
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8GenerationStepPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sq8GenerationTopLogit {
    pub token_id: usize,
    pub logit: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8GenerationStepResult {
    pub generated_index: usize,
    pub phase: Sq8GenerationStepPhase,
    pub input_token_id: Option<usize>,
    pub cache_position: Option<usize>,
    pub cache_len_after: usize,
    pub output_token_id: usize,
    pub output_logit: f32,
    pub top_logits: Vec<Sq8GenerationTopLogit>,
    pub final_hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub final_hidden_f32_le_sha256: String,
    pub logits_f32_le_sha256: String,
    pub started_at_ns: u128,
    pub completed_at_ns: u128,
    pub latency_ns: u128,
}

impl Sq8GenerationStepResult {
    pub fn validate_contract(&self) -> Result<(), String> {
        let final_hidden_health = validate_sq8_model_head_tensor_health(
            &self.final_hidden,
            QWEN3_14B_HIDDEN_SIZE,
            "generation step final hidden",
        )?;
        let logits_health = validate_sq8_model_head_tensor_health(
            &self.logits,
            QWEN3_14B_VOCAB_SIZE,
            "generation step logits",
        )?;
        if final_hidden_health.f32_le_sha256 != self.final_hidden_f32_le_sha256
            || logits_health.f32_le_sha256 != self.logits_f32_le_sha256
        {
            return Err("SQ8 generation step tensor hash mismatch".into());
        }
        if self.output_token_id >= QWEN3_14B_VOCAB_SIZE || !self.output_logit.is_finite() {
            return Err(format!(
                "SQ8 generation step has invalid output: token={} logit={}",
                self.output_token_id, self.output_logit
            ));
        }
        validate_sha256(&self.final_hidden_f32_le_sha256)?;
        validate_sha256(&self.logits_f32_le_sha256)?;
        let recomputed_top1 = greedy_top1_finite(&self.logits)?;
        if recomputed_top1.token_id != self.output_token_id
            || recomputed_top1.logit.to_bits() != self.output_logit.to_bits()
        {
            return Err("SQ8 generation step stored logits/top1 mismatch".into());
        }
        let recomputed_top_logits =
            top_finite_logits(&self.logits, QWEN3_14B_SQ8_GENERATION_TOP_LOGITS)?;
        if recomputed_top_logits != self.top_logits {
            return Err("SQ8 generation step stored logits/top-logit mismatch".into());
        }
        validate_top_logits(&self.top_logits, self.output_token_id, self.output_logit)?;
        let expected_latency = self
            .completed_at_ns
            .checked_sub(self.started_at_ns)
            .ok_or_else(|| "SQ8 generation step timing runs backwards".to_string())?;
        if self.latency_ns != expected_latency || self.latency_ns == 0 {
            return Err(format!(
                "SQ8 generation step latency mismatch: expected={expected_latency} actual={}",
                self.latency_ns
            ));
        }
        match self.phase {
            Sq8GenerationStepPhase::Prefill => {
                if self.generated_index != 0
                    || self.input_token_id.is_some()
                    || self.cache_position.is_some()
                    || self.cache_len_after != QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
                    || self.started_at_ns != 0
                {
                    return Err("SQ8 generation prefill step metadata is inconsistent".into());
                }
            }
            Sq8GenerationStepPhase::Decode => {
                if self.generated_index == 0 {
                    return Err("SQ8 generation decode step index must be nonzero".into());
                }
                let expected_position = QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
                    .checked_add(self.generated_index - 1)
                    .ok_or_else(|| "SQ8 generation decode position overflows".to_string())?;
                if self.input_token_id.is_none()
                    || self.cache_position != Some(expected_position)
                    || self.cache_len_after != expected_position + 1
                {
                    return Err(format!(
                        "SQ8 generation decode step {} metadata is inconsistent",
                        self.generated_index
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8GenerationCompletionReason {
    Eos,
    MaxNewTokens,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8GenerationCompletion {
    pub reason: Sq8GenerationCompletionReason,
    pub generated_token_ids: Vec<usize>,
    pub decode_input_token_ids: Vec<usize>,
    pub decode_positions: Vec<usize>,
    pub final_kv_len: usize,
    pub released_kv_blocks: usize,
    pub allocation_released: bool,
}

impl Sq8GenerationCompletion {
    pub fn validate_contract(&self, request: &Sq8GenerationRequest) -> Result<(), String> {
        validate_feedback_contract(
            request,
            &self.generated_token_ids,
            &self.decode_input_token_ids,
            &self.decode_positions,
            self.final_kv_len,
        )?;
        let eos_observed = self.generated_token_ids.last().copied() == Some(request.eos_token_id);
        match self.reason {
            Sq8GenerationCompletionReason::Eos if !eos_observed => {
                return Err("SQ8 generation completion says EOS but final token is not EOS".into());
            }
            Sq8GenerationCompletionReason::MaxNewTokens => {
                if eos_observed || self.generated_token_ids.len() != request.max_new_tokens {
                    return Err(
                        "SQ8 generation max-token completion has inconsistent token count/EOS"
                            .into(),
                    );
                }
            }
            Sq8GenerationCompletionReason::Eos => {}
        }
        if self.released_kv_blocks != GENERATION_KV_BLOCKS as usize || !self.allocation_released {
            return Err(format!(
                "SQ8 generation allocation release mismatch: expected={} actual={} released={}",
                GENERATION_KV_BLOCKS, self.released_kv_blocks, self.allocation_released
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8OfflineGenerationMetrics {
    pub request_count: usize,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub time_to_first_token_ns: u128,
    pub request_latency_ns: u128,
    pub decode_elapsed_ns: u128,
    pub requests_per_second: f64,
    pub generated_tokens_per_second: f64,
    pub total_tokens_per_second: f64,
    pub decode_tokens_per_second: Option<f64>,
}

impl Sq8OfflineGenerationMetrics {
    fn measured(
        generated_tokens: usize,
        time_to_first_token_ns: u128,
        request_latency_ns: u128,
    ) -> Result<Self, String> {
        if generated_tokens == 0 || time_to_first_token_ns == 0 || request_latency_ns == 0 {
            return Err("SQ8 offline metrics require nonzero tokens and timings".into());
        }
        let decode_elapsed_ns = request_latency_ns
            .checked_sub(time_to_first_token_ns)
            .ok_or_else(|| "SQ8 offline TTFT exceeds request latency".to_string())?;
        let seconds = request_latency_ns as f64 / 1_000_000_000.0;
        let decode_tokens = generated_tokens - 1;
        let decode_tokens_per_second = if decode_tokens == 0 {
            None
        } else {
            if decode_elapsed_ns == 0 {
                return Err("SQ8 offline decode timing is zero with decode tokens".into());
            }
            Some(decode_tokens as f64 * 1_000_000_000.0 / decode_elapsed_ns as f64)
        };
        let metrics = Self {
            request_count: 1,
            prompt_tokens: QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS,
            generated_tokens,
            time_to_first_token_ns,
            request_latency_ns,
            decode_elapsed_ns,
            requests_per_second: 1.0 / seconds,
            generated_tokens_per_second: generated_tokens as f64 / seconds,
            total_tokens_per_second: (QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS + generated_tokens)
                as f64
                / seconds,
            decode_tokens_per_second,
        };
        metrics.validate_contract()?;
        Ok(metrics)
    }

    pub fn validate_contract(&self) -> Result<(), String> {
        if self.request_count != 1
            || self.prompt_tokens != QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
            || self.generated_tokens == 0
            || self.generated_tokens > QWEN3_14B_SQ8_GENERATION_MAX_NEW_TOKENS
            || self.time_to_first_token_ns == 0
            || self.request_latency_ns < self.time_to_first_token_ns
            || self.decode_elapsed_ns != self.request_latency_ns - self.time_to_first_token_ns
        {
            return Err("SQ8 offline metric counts/timings are inconsistent".into());
        }
        for (label, value) in [
            ("requests_per_second", self.requests_per_second),
            (
                "generated_tokens_per_second",
                self.generated_tokens_per_second,
            ),
            ("total_tokens_per_second", self.total_tokens_per_second),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("SQ8 offline metric {label} is invalid: {value}"));
            }
        }
        let seconds = self.request_latency_ns as f64 / 1_000_000_000.0;
        let expected_rates = [
            1.0 / seconds,
            self.generated_tokens as f64 / seconds,
            (self.prompt_tokens + self.generated_tokens) as f64 / seconds,
        ];
        let actual_rates = [
            self.requests_per_second,
            self.generated_tokens_per_second,
            self.total_tokens_per_second,
        ];
        if actual_rates
            .iter()
            .zip(expected_rates.iter())
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err("SQ8 offline throughput is not derived from measured latency".into());
        }
        match (self.generated_tokens, self.decode_tokens_per_second) {
            (1, None) => {}
            (1, Some(_)) => {
                return Err("SQ8 offline one-token result must not report decode TPS".into());
            }
            (_, Some(value))
                if value.is_finite()
                    && value > 0.0
                    && value.to_bits()
                        == ((self.generated_tokens - 1) as f64 * 1_000_000_000.0
                            / self.decode_elapsed_ns as f64)
                            .to_bits() => {}
            _ => return Err("SQ8 offline decode TPS is missing or invalid".into()),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8GenerationStackStepReport {
    pub phase: Sq8PagedStackPhase,
    pub profile: Sq8LayerExecutionProfile,
    pub artifact_content_sha256: String,
    pub position: usize,
    pub sequence_len: usize,
    pub cache_len: usize,
    pub projection_calls: usize,
    pub activation_quantizations: usize,
    pub layer_d2d_copies: usize,
    pub kv_write_calls: usize,
    pub paged_attention_calls: usize,
    pub input_d2d_copies: usize,
    pub all_ck: bool,
    pub fallback_used: bool,
    pub host_staging_used: bool,
}

impl Sq8GenerationStackStepReport {
    fn from_runtime(value: &Sq8PagedStackExecutionReport) -> Result<Self, String> {
        value.validate_contract()?;
        let cache_len = value.cache_lengths[0];
        if value
            .cache_lengths
            .iter()
            .any(|actual| *actual != cache_len)
        {
            return Err("SQ8 generation stack step has divergent layer KV lengths".into());
        }
        Ok(Self {
            phase: value.phase,
            profile: value.stack.profile,
            artifact_content_sha256: value.stack.artifact_content_sha256.clone(),
            position: value.position,
            sequence_len: value.stack.sequence_len,
            cache_len,
            projection_calls: value.stack.projection_calls,
            activation_quantizations: value.stack.activation_quantizations,
            layer_d2d_copies: value.stack.d2d_copy_count,
            kv_write_calls: value.kv_write_calls,
            paged_attention_calls: value.paged_attention_calls,
            input_d2d_copies: value.input_d2d_copy_count,
            all_ck: value.stack.all_ck(),
            fallback_used: value.stack.fallback_used,
            host_staging_used: value.stack.host_staging_used,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sq8GenerationLoadReport {
    pub device: Sq8ModelHeadDeviceIdentity,
    pub artifact_content_sha256: String,
    pub package_manifest_sha256: String,
    pub canonical_package_dir: PathBuf,
    pub upload_chunk_bytes: usize,
    pub stack_layers: usize,
    pub cache_layers: usize,
    pub cache_shape: PagedDecodeShape,
    pub prompt_buffer_bytes: usize,
    pub embedding_payload_sha256: String,
    pub final_norm_payload_sha256: String,
    pub lm_head_payload_sha256: String,
}

impl Sq8GenerationLoadReport {
    pub fn validate_contract(&self) -> Result<(), String> {
        validate_device_identity(&self.device)?;
        validate_sha256(&self.artifact_content_sha256)?;
        validate_sha256(&self.package_manifest_sha256)?;
        validate_sha256(&self.embedding_payload_sha256)?;
        validate_sha256(&self.final_norm_payload_sha256)?;
        validate_sha256(&self.lm_head_payload_sha256)?;
        if !self.canonical_package_dir.is_absolute() || self.upload_chunk_bytes == 0 {
            return Err("SQ8 generation load path/chunk contract failed".into());
        }
        if self.stack_layers != QWEN3_14B_SQ8_STACK_LAYERS
            || self.cache_layers != QWEN3_14B_SQ8_STACK_LAYERS
            || self.cache_shape != generation_cache_shape()
            || self.prompt_buffer_bytes != prompt_buffer_bytes()?
        {
            return Err("SQ8 generation load shape/count contract failed".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8GenerationExecutionReport {
    pub request_id: RequestId,
    pub device: Sq8ModelHeadDeviceIdentity,
    pub artifact_content_sha256: String,
    pub package_manifest_sha256: String,
    pub profile: Sq8LayerExecutionProfile,
    pub stack_steps: Vec<Sq8GenerationStackStepReport>,
    pub embedding_gather_calls: usize,
    pub prompt_embedding_d2d_copies: usize,
    pub stack_input_d2d_copies: usize,
    pub projection_calls: usize,
    pub activation_quantizations: usize,
    pub layer_d2d_copies: usize,
    pub kv_write_calls: usize,
    pub paged_attention_calls: usize,
    pub model_head_calls: usize,
    pub model_head_d2d_copies: usize,
    pub result_readback_count: usize,
    pub execution_synchronization_count: usize,
    pub scheduler_prefill_completions: usize,
    pub scheduler_prefill_token_records: usize,
    pub scheduler_decode_advances: usize,
    pub scheduler_release_calls: usize,
    pub identity_check_count: usize,
    pub final_cache_lengths: [usize; QWEN3_14B_SQ8_STACK_LAYERS],
    pub generated_token_ids_sha256: String,
    pub feedback_verified: bool,
    pub allocation_released: bool,
    pub fallback_used: bool,
    pub host_staging_used: bool,
}

impl Sq8GenerationExecutionReport {
    pub fn validate_contract(
        &self,
        request: &Sq8GenerationRequest,
        completion: &Sq8GenerationCompletion,
    ) -> Result<(), String> {
        request.validate_contract()?;
        completion.validate_contract(request)?;
        validate_device_identity(&self.device)?;
        validate_sha256(&self.artifact_content_sha256)?;
        validate_sha256(&self.package_manifest_sha256)?;
        validate_sha256(&self.generated_token_ids_sha256)?;
        if self.request_id != request.request_id
            || self.profile != Sq8LayerExecutionProfile::Rdna4W8a8BlockCk
        {
            return Err("SQ8 generation execution identity/profile mismatch".into());
        }
        let generated = completion.generated_token_ids.len();
        let decode_steps = generated - 1;
        if self.stack_steps.len() != generated
            || self.stack_steps[0].phase != Sq8PagedStackPhase::Prefill
            || self.stack_steps[0].sequence_len != QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
            || self
                .stack_steps
                .iter()
                .skip(1)
                .any(|step| step.phase != Sq8PagedStackPhase::Decode || step.sequence_len != 1)
        {
            return Err("SQ8 generation stack-step sequence is invalid".into());
        }
        for (index, step) in self.stack_steps.iter().enumerate() {
            let (
                expected_position,
                expected_cache_len,
                expected_kv_writes,
                expected_attention,
                expected_input_d2d,
            ) = if index == 0 {
                (
                    0,
                    QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS,
                    QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS * QWEN3_14B_SQ8_STACK_LAYERS,
                    0,
                    0,
                )
            } else {
                (
                    QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS + index - 1,
                    QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS + index,
                    QWEN3_14B_SQ8_STACK_LAYERS,
                    QWEN3_14B_SQ8_STACK_LAYERS,
                    1,
                )
            };
            if step.profile != self.profile
                || step.artifact_content_sha256 != self.artifact_content_sha256
                || step.position != expected_position
                || step.cache_len != expected_cache_len
                || step.projection_calls
                    != QWEN3_14B_SQ8_STACK_LAYERS * QWEN3_14B_SQ8_LAYER_PROJECTIONS
                || step.activation_quantizations
                    != QWEN3_14B_SQ8_STACK_LAYERS * QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS
                || step.layer_d2d_copies != QWEN3_14B_SQ8_STACK_LAYERS
                || step.kv_write_calls != expected_kv_writes
                || step.paged_attention_calls != expected_attention
                || step.input_d2d_copies != expected_input_d2d
                || !step.all_ck
                || step.fallback_used
                || step.host_staging_used
            {
                return Err(format!("SQ8 generation stack step {index} contract failed"));
            }
        }
        let expected_projection_calls = generated
            .checked_mul(QWEN3_14B_SQ8_STACK_LAYERS)
            .and_then(|value| value.checked_mul(QWEN3_14B_SQ8_LAYER_PROJECTIONS))
            .ok_or_else(|| "SQ8 generation projection count overflows".to_string())?;
        let expected_activation_quantizations = generated
            .checked_mul(QWEN3_14B_SQ8_STACK_LAYERS)
            .and_then(|value| value.checked_mul(QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS))
            .ok_or_else(|| "SQ8 generation activation count overflows".to_string())?;
        let expected_layer_d2d = generated
            .checked_mul(QWEN3_14B_SQ8_STACK_LAYERS)
            .ok_or_else(|| "SQ8 generation layer D2D count overflows".to_string())?;
        let expected_kv_writes = (QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS + decode_steps)
            .checked_mul(QWEN3_14B_SQ8_STACK_LAYERS)
            .ok_or_else(|| "SQ8 generation KV write count overflows".to_string())?;
        let expected_paged_attention = decode_steps
            .checked_mul(QWEN3_14B_SQ8_STACK_LAYERS)
            .ok_or_else(|| "SQ8 generation attention count overflows".to_string())?;
        if self.embedding_gather_calls != QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS + decode_steps
            || self.prompt_embedding_d2d_copies != QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
            || self.stack_input_d2d_copies != 1 + decode_steps
            || self.projection_calls != expected_projection_calls
            || self.activation_quantizations != expected_activation_quantizations
            || self.layer_d2d_copies != expected_layer_d2d
            || self.kv_write_calls != expected_kv_writes
            || self.paged_attention_calls != expected_paged_attention
            || self.model_head_calls != generated
            || self.model_head_d2d_copies != 1
            || self.result_readback_count != 2 + 4 * decode_steps
            || self.execution_synchronization_count != 3 + 2 * decode_steps
        {
            return Err("SQ8 generation execution operation counts are inconsistent".into());
        }
        if self.scheduler_prefill_completions != 1
            || self.scheduler_prefill_token_records != 1
            || self.scheduler_decode_advances != decode_steps
            || self.scheduler_release_calls != 1
            || self.identity_check_count != 3
            || !self.feedback_verified
            || !self.allocation_released
            || self.fallback_used
            || self.host_staging_used
        {
            return Err("SQ8 generation scheduler/health/fallback contract failed".into());
        }
        if self
            .final_cache_lengths
            .iter()
            .any(|actual| *actual != completion.final_kv_len)
        {
            return Err("SQ8 generation final layer KV lengths diverge".into());
        }
        if self.generated_token_ids_sha256 != token_ids_sha256(&completion.generated_token_ids)? {
            return Err("SQ8 generation token ID hash mismatch".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sq8OfflineGenerationResult {
    pub request: Sq8GenerationRequest,
    pub steps: Vec<Sq8GenerationStepResult>,
    pub completion: Sq8GenerationCompletion,
    pub metrics: Sq8OfflineGenerationMetrics,
    pub allocator_before: KvBlockAllocatorStats,
    pub allocator_after_release: KvBlockAllocatorStats,
    pub report: Sq8GenerationExecutionReport,
}

impl Sq8OfflineGenerationResult {
    pub fn validate_contract(&self) -> Result<(), String> {
        self.request.validate_contract()?;
        self.completion.validate_contract(&self.request)?;
        self.metrics.validate_contract()?;
        self.report
            .validate_contract(&self.request, &self.completion)?;
        if self.steps.len() != self.completion.generated_token_ids.len()
            || self.steps.len() != self.metrics.generated_tokens
        {
            return Err("SQ8 generation result step/token count mismatch".into());
        }
        for (index, step) in self.steps.iter().enumerate() {
            step.validate_contract()?;
            if step.generated_index != index
                || step.output_token_id != self.completion.generated_token_ids[index]
                || (index > 0
                    && step.input_token_id
                        != Some(self.completion.decode_input_token_ids[index - 1]))
            {
                return Err(format!(
                    "SQ8 generation result step {index} is not bound to completion tokens"
                ));
            }
            if index > 0 && step.started_at_ns != self.steps[index - 1].completed_at_ns {
                return Err(format!(
                    "SQ8 generation result step {index} timing is not contiguous"
                ));
            }
        }
        if self.steps[0].completed_at_ns != self.metrics.time_to_first_token_ns
            || self.steps.last().map(|step| step.completed_at_ns)
                != Some(self.metrics.request_latency_ns)
        {
            return Err("SQ8 generation metrics are not bound to step timings".into());
        }
        validate_allocator_before(self.allocator_before)?;
        validate_allocator_after_release(self.allocator_after_release)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8GenerationRuntimeStatus {
    Ready,
    Running,
    Completed,
    Poisoned,
}

#[derive(Debug)]
enum Sq8GenerationRuntimeState {
    Ready,
    Running,
    Completed,
    Poisoned(String),
}

impl Sq8GenerationRuntimeState {
    fn status(&self) -> Sq8GenerationRuntimeStatus {
        match self {
            Self::Ready => Sq8GenerationRuntimeStatus::Ready,
            Self::Running => Sq8GenerationRuntimeStatus::Running,
            Self::Completed => Sq8GenerationRuntimeStatus::Completed,
            Self::Poisoned(_) => Sq8GenerationRuntimeStatus::Poisoned,
        }
    }

    fn require_ready(&self) -> Result<(), String> {
        match self {
            Self::Ready => Ok(()),
            Self::Running => Err("Qwen3-14B SQ8 generation is already running".into()),
            Self::Completed => Err("Qwen3-14B SQ8 generation runtime is single-use".into()),
            Self::Poisoned(reason) => Err(format!(
                "Qwen3-14B SQ8 generation runtime is permanently poisoned: {reason}"
            )),
        }
    }
}

/// Owns every resident object needed for the fixed B=1 SQ8 generation workload.
#[derive(Debug)]
pub struct Qwen3Sq8GenerationRuntime {
    load_report: Sq8GenerationLoadReport,
    stack: Qwen3Sq8StackRuntime,
    decode: Qwen3Sq8PagedDecodeRuntime,
    caches: Box<[PagedDecodeState; QWEN3_14B_SQ8_STACK_LAYERS]>,
    embedding: Qwen3Sq8EmbeddingRuntime,
    head: Qwen3Sq8ModelHeadRuntime,
    prompt_hidden: RuntimeBuffer,
    scheduler: SchedulerState,
    state: Sq8GenerationRuntimeState,
}

impl Qwen3Sq8GenerationRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        context: &mut RuntimeContext,
        stream: &mut RuntimeStream,
        artifact: &Sq8CanonicalArtifact,
        package_path: impl AsRef<Path>,
        norms: Vec<Qwen3Sq8LayerNormValues>,
        upload_chunk_bytes: usize,
    ) -> Result<Self, String> {
        if upload_chunk_bytes == 0 {
            return Err("Qwen3-14B SQ8 generation upload chunk size must be nonzero".into());
        }
        let package_path = package_path.as_ref();
        let load_result = (|| {
            let device_info = context.device_info()?;
            validate_qwen3_14b_sq8_r9700_device_info(&device_info)?;
            let stack = Qwen3Sq8StackRuntime::load(
                context,
                stream,
                artifact,
                QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS,
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
                    "SQ8 generation package manifest mismatch: embedding={} head={}",
                    embedding.load_report().package.manifest_sha256,
                    head.package_manifest_sha256()
                ));
            }
            if stack.artifact_content_sha256() != artifact.manifest().integrity.content_sha256 {
                return Err("SQ8 generation stack artifact identity changed while loading".into());
            }

            let decode = Qwen3Sq8PagedDecodeRuntime::allocate(context)?;
            let cache_shape = generation_cache_shape();
            let mut cache_values = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
            for layer_index in 0..QWEN3_14B_SQ8_STACK_LAYERS {
                cache_values.push(
                    PagedDecodeState::new(context, stream, cache_shape, vec![0]).map_err(
                        |err| format!("failed to allocate SQ8 generation layer {layer_index} KV cache: {err}"),
                    )?,
                );
            }
            let caches: [PagedDecodeState; QWEN3_14B_SQ8_STACK_LAYERS] = cache_values
                .try_into()
                .map_err(|values: Vec<PagedDecodeState>| {
                    format!(
                        "SQ8 generation cache array length mismatch: expected={} actual={}",
                        QWEN3_14B_SQ8_STACK_LAYERS,
                        values.len()
                    )
                })?;
            let prompt_hidden = context
                .alloc_buffer(prompt_buffer_bytes()?)
                .map_err(|err| format!("failed to allocate SQ8 generation prompt buffer: {err}"))?;
            let package_manifest_sha256 = head.package_manifest_sha256().to_string();
            let artifact_content_sha256 = stack.artifact_content_sha256().to_string();
            let load_report = Sq8GenerationLoadReport {
                device: head.device_identity().clone(),
                artifact_content_sha256,
                package_manifest_sha256,
                canonical_package_dir: embedding
                    .load_report()
                    .package
                    .canonical_package_dir
                    .clone(),
                upload_chunk_bytes,
                stack_layers: stack.layer_count(),
                cache_layers: caches.len(),
                cache_shape,
                prompt_buffer_bytes: prompt_hidden.size()?,
                embedding_payload_sha256: embedding.load_report().payload.payload_sha256.clone(),
                final_norm_payload_sha256: head.final_norm_identity().payload_sha256.clone(),
                lm_head_payload_sha256: head.lm_head_identity().payload_sha256.clone(),
            };
            load_report.validate_contract()?;
            let runtime = Self {
                load_report,
                stack,
                decode,
                caches: Box::new(caches),
                embedding,
                head,
                prompt_hidden,
                scheduler: SchedulerState::with_block_size(
                    GENERATION_KV_BLOCKS,
                    QWEN3_14B_SQ8_GENERATION_BLOCK_TOKENS as u32,
                ),
                state: Sq8GenerationRuntimeState::Ready,
            };
            runtime.validate_runtime_contract()?;
            Ok(runtime)
        })();
        match load_result {
            Ok(runtime) => Ok(runtime),
            Err(operation_error) => Err(load_error_after_stream_recovery(stream, operation_error)),
        }
    }

    pub fn status(&self) -> Sq8GenerationRuntimeStatus {
        self.state.status()
    }

    pub fn poison_reason(&self) -> Option<&str> {
        match &self.state {
            Sq8GenerationRuntimeState::Poisoned(reason) => Some(reason),
            Sq8GenerationRuntimeState::Ready
            | Sq8GenerationRuntimeState::Running
            | Sq8GenerationRuntimeState::Completed => None,
        }
    }

    pub fn load_report(&self) -> &Sq8GenerationLoadReport {
        &self.load_report
    }

    pub fn run_fixed_synchronized(
        &mut self,
        max_new_tokens: usize,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8OfflineGenerationResult, String> {
        self.run_request_synchronized(Sq8GenerationRequest::fixed(max_new_tokens), stream)
    }

    /// Resets completed request state outside the measured generation interval.
    pub fn reset_synchronized(&mut self, stream: &mut RuntimeStream) -> Result<(), String> {
        match &self.state {
            Sq8GenerationRuntimeState::Completed => {}
            Sq8GenerationRuntimeState::Ready => {
                return Err("Qwen3-14B SQ8 generation runtime is already ready".into());
            }
            Sq8GenerationRuntimeState::Running => {
                return Err("Qwen3-14B SQ8 generation cannot reset while running".into());
            }
            Sq8GenerationRuntimeState::Poisoned(reason) => {
                return Err(format!(
                    "Qwen3-14B SQ8 generation runtime is permanently poisoned: {reason}"
                ));
            }
        }
        if self.scheduler.active_len() != 0 || !self.scheduler.waiting_is_empty() {
            return Err(self.poison_after_failure(
                stream,
                "SQ8 generation reset found retained scheduler requests".into(),
            ));
        }
        if let Err(err) = validate_allocator_before(self.scheduler.allocator_stats()) {
            return Err(self.poison_after_failure(stream, err));
        }
        for layer_index in 0..self.caches.len() {
            if let Err(err) = self.caches[layer_index].reset(stream) {
                return Err(self.poison_after_failure(
                    stream,
                    format!("failed to reset SQ8 generation layer {layer_index} KV cache: {err}"),
                ));
            }
        }
        if let Err(err) = validate_cache_lengths(self.caches.as_ref(), 0) {
            return Err(self.poison_after_failure(stream, err));
        }
        self.state = Sq8GenerationRuntimeState::Ready;
        if let Err(err) = self.validate_runtime_contract() {
            return Err(self.poison_after_failure(
                stream,
                format!("SQ8 generation reset contract failed: {err}"),
            ));
        }
        Ok(())
    }

    pub fn run_request_synchronized(
        &mut self,
        request: Sq8GenerationRequest,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8OfflineGenerationResult, String> {
        self.state.require_ready()?;
        request.validate_contract()?;
        self.validate_runtime_contract()?;
        self.state = Sq8GenerationRuntimeState::Running;

        let result = self.run_request_inner(request, stream);
        match result {
            Ok(result) => {
                if let Err(err) = result.validate_contract() {
                    let error = self.poison_after_failure(
                        stream,
                        format!("SQ8 generation result validation failed: {err}"),
                    );
                    return Err(error);
                }
                self.state = Sq8GenerationRuntimeState::Completed;
                Ok(result)
            }
            Err(err) => Err(self.poison_after_failure(stream, err)),
        }
    }

    fn run_request_inner(
        &mut self,
        request: Sq8GenerationRequest,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8OfflineGenerationResult, String> {
        let request_start = Instant::now();
        let allocator_before = self.scheduler.allocator_stats();
        validate_allocator_before(allocator_before)?;
        self.scheduler.enqueue(Request {
            id: request.request_id,
            prompt_tokens: request.prompt_token_ids.len(),
            max_new_tokens: request.max_new_tokens,
        });
        let allocated = self
            .scheduler
            .pop_prefill_batch_with_allocation(request.prompt_token_ids.len())?;
        if allocated.len() != 1
            || allocated[0].request.id != request.request_id
            || allocated[0].allocation.blocks != [0]
        {
            return Err(format!(
                "SQ8 generation scheduler allocation mismatch: {allocated:?}"
            ));
        }
        validate_cache_block_tables(self.caches.as_ref(), &allocated[0].allocation.blocks)?;

        let mut embedding_reports =
            Vec::with_capacity(QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS + request.max_new_tokens - 1);
        for (row, token_id) in request.prompt_token_ids.iter().copied().enumerate() {
            let report = self.embedding.enqueue_token_resident(token_id, stream)?;
            let (embedding_output, resident_report) = self.embedding.resident_output()?;
            if resident_report != &report {
                return Err(format!(
                    "SQ8 generation prompt embedding report changed at row {row}"
                ));
            }
            self.prompt_hidden.copy_from_buffer(
                row.checked_mul(hidden_row_bytes()?)
                    .ok_or_else(|| "SQ8 generation prompt row offset overflows".to_string())?,
                embedding_output,
                0,
                hidden_row_bytes()?,
                Some(&mut *stream),
            )?;
            embedding_reports.push(report);
        }
        self.stack
            .upload_device_input_synchronized(&self.prompt_hidden, stream)?;
        let prefill_stack = self
            .stack
            .run_uploaded_paged_prefill_optimized_synchronized(&mut self.caches[..], stream)?;
        self.scheduler.complete_prefill(request.request_id)?;

        let first_head = self.head.run_synchronized(&self.stack, stream)?;
        if first_head.report.stack_execution != prefill_stack.stack
            || first_head.report.device != self.load_report.device
            || first_head.report.stack_artifact_content_sha256
                != self.load_report.artifact_content_sha256
            || first_head.report.final_norm.payload_sha256
                != self.load_report.final_norm_payload_sha256
            || first_head.report.lm_head.payload_sha256 != self.load_report.lm_head_payload_sha256
        {
            return Err(
                "SQ8 generation prefill head identity is not bound to paged prefill/load".into(),
            );
        }
        let first_top1 = greedy_top1_finite(&first_head.logits)?;
        let first_step_completed = elapsed_ns_nonzero(request_start);
        let first_step =
            generation_step_from_p6_head(&first_head, first_top1, first_step_completed)?;
        self.scheduler
            .record_prefill_generated_token(request.request_id)?;

        let mut generated_token_ids = vec![first_top1.token_id];
        let mut decode_input_token_ids = Vec::with_capacity(request.max_new_tokens - 1);
        let mut decode_positions = Vec::with_capacity(request.max_new_tokens - 1);
        let mut steps = vec![first_step];
        let mut stack_steps = vec![Sq8GenerationStackStepReport::from_runtime(&prefill_stack)?];
        let mut final_cache_lengths = prefill_stack.cache_lengths;
        let mut previous_completed_ns = first_step_completed;

        while generated_token_ids.len() < request.max_new_tokens
            && generated_token_ids.last().copied() != Some(request.eos_token_id)
        {
            let ready = self.scheduler.ready_decode_batch(1)?;
            let decode_request = validate_ready_batch(&ready, &request, generated_token_ids.len())?;
            let input_token_id = *generated_token_ids
                .last()
                .ok_or_else(|| "SQ8 generation decode has no feedback token".to_string())?;
            if input_token_id >= QWEN3_14B_VOCAB_SIZE {
                return Err(format!(
                    "SQ8 generation decode input token is out of range: {input_token_id}"
                ));
            }
            let embedding_report = self
                .embedding
                .enqueue_token_resident(input_token_id, stream)?;
            let (embedding_output, resident_report) = self.embedding.resident_output()?;
            if resident_report != &embedding_report {
                return Err("SQ8 generation decode embedding report changed".into());
            }
            let stack_report = self.stack.run_paged_decode_optimized_synchronized(
                &mut self.decode,
                embedding_output,
                decode_request.cache_position,
                &mut self.caches[..],
                stream,
            )?;
            let decode_head = self
                .head
                .run_m1_paged_decode_top1_synchronized(&self.decode, stream)?;
            decode_head.validate_contract()?;
            if decode_head.report.paged_decode != stack_report
                || decode_head.report.binding.device != self.load_report.device
                || decode_head.report.binding.package_manifest_sha256
                    != self.load_report.package_manifest_sha256
                || decode_head.report.binding.artifact_content_sha256
                    != self.load_report.artifact_content_sha256
                || decode_head.report.final_norm.payload_sha256
                    != self.load_report.final_norm_payload_sha256
                || decode_head.report.lm_head.payload_sha256
                    != self.load_report.lm_head_payload_sha256
            {
                return Err(format!(
                    "SQ8 generation M=1 head identity is not bound to decode position {}",
                    decode_request.cache_position
                ));
            }
            let host_top1 = greedy_top1_finite(&decode_head.logits)?;
            if host_top1.token_id != decode_head.top1.token_id
                || host_top1.logit.to_bits() != decode_head.top1.logit.to_bits()
            {
                return Err(format!(
                    "SQ8 generation M=1 top1 mismatch: generation=({}, {}) head=({}, {})",
                    host_top1.token_id,
                    host_top1.logit,
                    decode_head.top1.token_id,
                    decode_head.top1.logit
                ));
            }
            self.scheduler.advance_decode_batch(&ready)?;

            decode_input_token_ids.push(input_token_id);
            decode_positions.push(decode_request.cache_position);
            generated_token_ids.push(host_top1.token_id);
            embedding_reports.push(embedding_report);
            final_cache_lengths = stack_report.cache_lengths;
            stack_steps.push(Sq8GenerationStackStepReport::from_runtime(&stack_report)?);
            let completed_ns = elapsed_ns_strictly_after(request_start, previous_completed_ns);
            steps.push(generation_step_from_m1_head(
                generated_token_ids.len() - 1,
                input_token_id,
                decode_request.cache_position,
                previous_completed_ns,
                completed_ns,
                &decode_head,
                host_top1,
            )?);
            previous_completed_ns = completed_ns;
        }

        let final_kv_len = QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
            .checked_add(generated_token_ids.len() - 1)
            .ok_or_else(|| "SQ8 generation final KV length overflows".to_string())?;
        validate_feedback_contract(
            &request,
            &generated_token_ids,
            &decode_input_token_ids,
            &decode_positions,
            final_kv_len,
        )?;
        validate_scheduler_final_state(
            &self.scheduler,
            &request,
            generated_token_ids.len(),
            final_kv_len,
        )?;
        validate_cache_lengths(self.caches.as_ref(), final_kv_len)?;

        let reason = if generated_token_ids.last().copied() == Some(request.eos_token_id) {
            Sq8GenerationCompletionReason::Eos
        } else {
            Sq8GenerationCompletionReason::MaxNewTokens
        };
        let released_kv_blocks = self.scheduler.release_request(request.request_id);
        let allocator_after_release = self.scheduler.allocator_stats();
        validate_allocator_after_release(allocator_after_release)?;
        if self.scheduler.active_len() != 0 || !self.scheduler.waiting_is_empty() {
            return Err("SQ8 generation scheduler retained request state after release".into());
        }
        let completion = Sq8GenerationCompletion {
            reason,
            generated_token_ids,
            decode_input_token_ids,
            decode_positions,
            final_kv_len,
            released_kv_blocks,
            allocation_released: released_kv_blocks == GENERATION_KV_BLOCKS as usize,
        };
        completion.validate_contract(&request)?;

        let request_latency_ns = previous_completed_ns;
        let metrics = Sq8OfflineGenerationMetrics::measured(
            completion.generated_token_ids.len(),
            first_step_completed,
            request_latency_ns,
        )?;
        validate_embedding_reports(&embedding_reports, &request, &completion, &self.load_report)?;
        let decode_steps = completion.generated_token_ids.len() - 1;
        let report = Sq8GenerationExecutionReport {
            request_id: request.request_id,
            device: self.load_report.device.clone(),
            artifact_content_sha256: self.load_report.artifact_content_sha256.clone(),
            package_manifest_sha256: self.load_report.package_manifest_sha256.clone(),
            profile: Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
            embedding_gather_calls: embedding_reports.len(),
            prompt_embedding_d2d_copies: QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS,
            stack_input_d2d_copies: 1 + decode_steps,
            projection_calls: stack_steps.iter().map(|step| step.projection_calls).sum(),
            activation_quantizations: stack_steps
                .iter()
                .map(|step| step.activation_quantizations)
                .sum(),
            layer_d2d_copies: stack_steps.iter().map(|step| step.layer_d2d_copies).sum(),
            kv_write_calls: stack_steps.iter().map(|step| step.kv_write_calls).sum(),
            paged_attention_calls: stack_steps
                .iter()
                .map(|step| step.paged_attention_calls)
                .sum(),
            model_head_calls: completion.generated_token_ids.len(),
            model_head_d2d_copies: first_head.report.d2d_copy_count,
            result_readback_count: first_head.report.result_readback_count + 4 * decode_steps,
            execution_synchronization_count: 3 + 2 * decode_steps,
            scheduler_prefill_completions: 1,
            scheduler_prefill_token_records: 1,
            scheduler_decode_advances: decode_steps,
            scheduler_release_calls: 1,
            identity_check_count: 3,
            final_cache_lengths,
            generated_token_ids_sha256: token_ids_sha256(&completion.generated_token_ids)?,
            feedback_verified: true,
            allocation_released: true,
            fallback_used: false,
            host_staging_used: false,
            stack_steps,
        };
        report.validate_contract(&request, &completion)?;
        Ok(Sq8OfflineGenerationResult {
            request,
            steps,
            completion,
            metrics,
            allocator_before,
            allocator_after_release,
            report,
        })
    }

    fn validate_runtime_contract(&self) -> Result<(), String> {
        self.load_report.validate_contract()?;
        if self.stack.config().sequence_len != QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
            || self.stack.layer_count() != QWEN3_14B_SQ8_STACK_LAYERS
            || self.stack.artifact_content_sha256() != self.load_report.artifact_content_sha256
            || self.prompt_hidden.size()? != prompt_buffer_bytes()?
        {
            return Err("SQ8 generation resident stack/prompt contract failed".into());
        }
        validate_component_device_identity(
            self.embedding.device_identity(),
            self.head.device_identity(),
        )?;
        if self.embedding.load_report().package.manifest_sha256
            != self.load_report.package_manifest_sha256
            || self.head.package_manifest_sha256() != self.load_report.package_manifest_sha256
        {
            return Err("SQ8 generation resident package identity mismatch".into());
        }
        if self.caches.len() != QWEN3_14B_SQ8_STACK_LAYERS
            || self.caches.iter().any(|cache| {
                cache.shape() != generation_cache_shape() || cache.block_table() != [0]
            })
        {
            return Err("SQ8 generation resident KV cache contract failed".into());
        }
        if matches!(self.state, Sq8GenerationRuntimeState::Ready) {
            validate_cache_lengths(self.caches.as_ref(), 0)?;
            validate_allocator_before(self.scheduler.allocator_stats())?;
        }
        Ok(())
    }

    fn poison_after_failure(
        &mut self,
        stream: &mut RuntimeStream,
        operation_error: String,
    ) -> String {
        let recovery_error = stream.synchronize().err();
        for request_id in self.scheduler.active_request_ids() {
            self.scheduler.release_request(request_id);
        }
        let error = match recovery_error {
            Some(sync_error) => format!(
                "{operation_error}; subsequent SQ8 generation stream recovery failed: {sync_error}"
            ),
            None => operation_error,
        };
        self.state = Sq8GenerationRuntimeState::Poisoned(error.clone());
        error
    }
}

fn generation_step_from_p6_head(
    result: &Sq8ModelHeadResult,
    top1: Sq8GenerationTopLogit,
    completed_at_ns: u128,
) -> Result<Sq8GenerationStepResult, String> {
    result.validate_contract()?;
    let step = Sq8GenerationStepResult {
        generated_index: 0,
        phase: Sq8GenerationStepPhase::Prefill,
        input_token_id: None,
        cache_position: None,
        cache_len_after: QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS,
        output_token_id: top1.token_id,
        output_logit: top1.logit,
        top_logits: top_finite_logits(&result.logits, QWEN3_14B_SQ8_GENERATION_TOP_LOGITS)?,
        final_hidden: result.final_hidden.clone(),
        logits: result.logits.clone(),
        final_hidden_f32_le_sha256: result.report.final_hidden_health.f32_le_sha256.clone(),
        logits_f32_le_sha256: result.report.logits_health.f32_le_sha256.clone(),
        started_at_ns: 0,
        completed_at_ns,
        latency_ns: completed_at_ns,
    };
    step.validate_contract()?;
    Ok(step)
}

#[allow(clippy::too_many_arguments)]
fn generation_step_from_m1_head(
    generated_index: usize,
    input_token_id: usize,
    cache_position: usize,
    started_at_ns: u128,
    completed_at_ns: u128,
    result: &Sq8ModelHeadM1Result,
    top1: Sq8GenerationTopLogit,
) -> Result<Sq8GenerationStepResult, String> {
    result.validate_contract()?;
    let step = Sq8GenerationStepResult {
        generated_index,
        phase: Sq8GenerationStepPhase::Decode,
        input_token_id: Some(input_token_id),
        cache_position: Some(cache_position),
        cache_len_after: cache_position + 1,
        output_token_id: top1.token_id,
        output_logit: top1.logit,
        top_logits: top_finite_logits(&result.logits, QWEN3_14B_SQ8_GENERATION_TOP_LOGITS)?,
        final_hidden: result.final_hidden.clone(),
        logits: result.logits.clone(),
        final_hidden_f32_le_sha256: result.report.final_hidden_health.f32_le_sha256.clone(),
        logits_f32_le_sha256: result.report.logits_health.f32_le_sha256.clone(),
        started_at_ns,
        completed_at_ns,
        latency_ns: completed_at_ns
            .checked_sub(started_at_ns)
            .ok_or_else(|| "SQ8 generation decode timing runs backwards".to_string())?,
    };
    step.validate_contract()?;
    Ok(step)
}

fn validate_ready_batch<'a>(
    ready: &'a [SchedulerDecodeRequest],
    request: &Sq8GenerationRequest,
    generated_tokens: usize,
) -> Result<&'a SchedulerDecodeRequest, String> {
    if ready.len() != 1 {
        return Err(format!(
            "SQ8 generation expected one ready decode request, got {}",
            ready.len()
        ));
    }
    let value = &ready[0];
    let expected_position = QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
        .checked_add(generated_tokens - 1)
        .ok_or_else(|| "SQ8 generation ready position overflows".to_string())?;
    if value.request.id != request.request_id
        || value.request.prompt_tokens != request.prompt_token_ids.len()
        || value.request.max_new_tokens != request.max_new_tokens
        || value.allocation.blocks != [0]
        || value.cached_tokens != expected_position
        || value.generated_tokens != generated_tokens
        || value.cache_position != expected_position
        || value.next_cache_len != expected_position + 1
        || value.remaining_new_tokens != request.max_new_tokens - generated_tokens
    {
        return Err(format!(
            "SQ8 generation ready decode metadata mismatch at generated={generated_tokens}: {value:?}"
        ));
    }
    Ok(value)
}

fn validate_scheduler_final_state(
    scheduler: &SchedulerState,
    request: &Sq8GenerationRequest,
    generated_tokens: usize,
    final_kv_len: usize,
) -> Result<(), String> {
    let active = scheduler
        .active_request(request.request_id)
        .ok_or_else(|| "SQ8 generation request disappeared before release".to_string())?;
    if active.request.prompt_tokens != request.prompt_token_ids.len()
        || active.request.max_new_tokens != request.max_new_tokens
        || active.allocation.blocks != [0]
        || active.cached_tokens != final_kv_len
        || active.generated_tokens != generated_tokens
    {
        return Err(format!(
            "SQ8 generation final scheduler state mismatch: {active:?}"
        ));
    }
    Ok(())
}

fn validate_feedback_contract(
    request: &Sq8GenerationRequest,
    generated_token_ids: &[usize],
    decode_input_token_ids: &[usize],
    decode_positions: &[usize],
    final_kv_len: usize,
) -> Result<(), String> {
    request.validate_contract()?;
    if generated_token_ids.is_empty() || generated_token_ids.len() > request.max_new_tokens {
        return Err(format!(
            "SQ8 generation generated token count is invalid: {}",
            generated_token_ids.len()
        ));
    }
    if generated_token_ids
        .iter()
        .any(|token_id| *token_id >= QWEN3_14B_VOCAB_SIZE)
    {
        return Err("SQ8 generation produced an out-of-range token ID".into());
    }
    if decode_input_token_ids != &generated_token_ids[..generated_token_ids.len() - 1] {
        return Err(format!(
            "SQ8 generation feedback mismatch: decode_inputs={decode_input_token_ids:?} generated_prefix={:?}",
            &generated_token_ids[..generated_token_ids.len() - 1]
        ));
    }
    let expected_positions = (QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
        ..QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS + decode_input_token_ids.len())
        .collect::<Vec<_>>();
    if decode_positions != expected_positions {
        return Err(format!(
            "SQ8 generation decode positions mismatch: expected={expected_positions:?} actual={decode_positions:?}"
        ));
    }
    let expected_kv_len = QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
        .checked_add(generated_token_ids.len() - 1)
        .ok_or_else(|| "SQ8 generation expected KV length overflows".to_string())?;
    if final_kv_len != expected_kv_len {
        return Err(format!(
            "SQ8 generation final KV length mismatch: expected={expected_kv_len} actual={final_kv_len}"
        ));
    }
    if let Some(eos_index) = generated_token_ids
        .iter()
        .position(|token_id| *token_id == request.eos_token_id)
        && eos_index + 1 != generated_token_ids.len()
    {
        return Err("SQ8 generation continued after EOS".into());
    }
    Ok(())
}

fn validate_embedding_reports(
    reports: &[Sq8EmbeddingExecutionReport],
    request: &Sq8GenerationRequest,
    completion: &Sq8GenerationCompletion,
    load: &Sq8GenerationLoadReport,
) -> Result<(), String> {
    let expected_tokens = request
        .prompt_token_ids
        .iter()
        .copied()
        .chain(completion.decode_input_token_ids.iter().copied())
        .collect::<Vec<_>>();
    if reports.len() != expected_tokens.len() {
        return Err(format!(
            "SQ8 generation embedding count mismatch: expected={} actual={}",
            expected_tokens.len(),
            reports.len()
        ));
    }
    for (index, (report, expected_token)) in reports.iter().zip(expected_tokens.iter()).enumerate()
    {
        report.validate_contract()?;
        if report.token_id != *expected_token
            || report.load.package.manifest_sha256 != load.package_manifest_sha256
            || report.load.payload.payload_sha256 != load.embedding_payload_sha256
            || report.fallback_used
            || report.host_staging_used
        {
            return Err(format!(
                "SQ8 generation embedding report {index} contract failed"
            ));
        }
        validate_component_device_identity(&report.device, &load.device)?;
    }
    Ok(())
}

fn validate_cache_block_tables(
    caches: &[PagedDecodeState],
    expected: &[u32],
) -> Result<(), String> {
    if caches.len() != QWEN3_14B_SQ8_STACK_LAYERS {
        return Err(format!(
            "SQ8 generation cache layer count mismatch: expected={} actual={}",
            QWEN3_14B_SQ8_STACK_LAYERS,
            caches.len()
        ));
    }
    if let Some((layer, actual)) = caches
        .iter()
        .enumerate()
        .map(|(layer, cache)| (layer, cache.block_table()))
        .find(|(_, actual)| *actual != expected)
    {
        return Err(format!(
            "SQ8 generation layer {layer} block table mismatch: expected={expected:?} actual={actual:?}"
        ));
    }
    Ok(())
}

fn validate_cache_lengths(caches: &[PagedDecodeState], expected: usize) -> Result<(), String> {
    if caches.len() != QWEN3_14B_SQ8_STACK_LAYERS {
        return Err("SQ8 generation cache layer count changed".into());
    }
    if let Some((layer, actual)) = caches
        .iter()
        .enumerate()
        .map(|(layer, cache)| (layer, cache.written_len()))
        .find(|(_, actual)| *actual != expected)
    {
        return Err(format!(
            "SQ8 generation layer {layer} cache length mismatch: expected={expected} actual={actual}"
        ));
    }
    Ok(())
}

fn validate_allocator_before(stats: KvBlockAllocatorStats) -> Result<(), String> {
    if stats.block_size_tokens != QWEN3_14B_SQ8_GENERATION_BLOCK_TOKENS as u32
        || stats.total_blocks != GENERATION_KV_BLOCKS
        || stats.free_blocks != GENERATION_KV_BLOCKS as usize
        || stats.allocated_blocks != 0
    {
        return Err(format!(
            "SQ8 generation allocator is not empty before request: {stats:?}"
        ));
    }
    Ok(())
}

fn validate_allocator_after_release(stats: KvBlockAllocatorStats) -> Result<(), String> {
    validate_allocator_before(stats)
        .map_err(|err| format!("SQ8 generation allocator was not restored after release: {err}"))
}

pub(crate) fn greedy_top1_finite(logits: &[f32]) -> Result<Sq8GenerationTopLogit, String> {
    let first = logits
        .first()
        .copied()
        .ok_or_else(|| "SQ8 generation logits must not be empty".to_string())?;
    if !first.is_finite() {
        return Err(format!(
            "SQ8 generation logits contain non-finite value {first} at index 0"
        ));
    }
    let mut best = Sq8GenerationTopLogit {
        token_id: 0,
        logit: first,
    };
    for (token_id, logit) in logits.iter().copied().enumerate().skip(1) {
        if !logit.is_finite() {
            return Err(format!(
                "SQ8 generation logits contain non-finite value {logit} at index {token_id}"
            ));
        }
        if logit > best.logit {
            best = Sq8GenerationTopLogit { token_id, logit };
        }
    }
    Ok(best)
}

fn top_finite_logits(logits: &[f32], count: usize) -> Result<Vec<Sq8GenerationTopLogit>, String> {
    if count == 0 || logits.len() < count {
        return Err(format!(
            "SQ8 generation top-logit request is invalid: logits={} count={count}",
            logits.len()
        ));
    }
    let mut top = Vec::with_capacity(count);
    for (token_id, logit) in logits.iter().copied().enumerate() {
        if !logit.is_finite() {
            return Err(format!(
                "SQ8 generation logits contain non-finite value {logit} at index {token_id}"
            ));
        }
        let candidate = Sq8GenerationTopLogit { token_id, logit };
        let insertion = top
            .iter()
            .position(|current: &Sq8GenerationTopLogit| {
                candidate.logit > current.logit
                    || (candidate.logit == current.logit && candidate.token_id < current.token_id)
            })
            .unwrap_or(top.len());
        if insertion < count {
            top.insert(insertion, candidate);
            if top.len() > count {
                top.pop();
            }
        }
    }
    Ok(top)
}

fn validate_top_logits(
    values: &[Sq8GenerationTopLogit],
    expected_token_id: usize,
    expected_logit: f32,
) -> Result<(), String> {
    if values.len() != QWEN3_14B_SQ8_GENERATION_TOP_LOGITS {
        return Err(format!(
            "SQ8 generation top-logit count mismatch: expected={} actual={}",
            QWEN3_14B_SQ8_GENERATION_TOP_LOGITS,
            values.len()
        ));
    }
    if values[0].token_id != expected_token_id
        || values[0].logit.to_bits() != expected_logit.to_bits()
    {
        return Err("SQ8 generation top-logit list is not bound to selected token".into());
    }
    for (index, value) in values.iter().enumerate() {
        if value.token_id >= QWEN3_14B_VOCAB_SIZE || !value.logit.is_finite() {
            return Err(format!(
                "SQ8 generation top-logit {index} is invalid: {value:?}"
            ));
        }
        if values[..index]
            .iter()
            .any(|previous| previous.token_id == value.token_id)
        {
            return Err(format!(
                "SQ8 generation top-logit token {} is duplicated",
                value.token_id
            ));
        }
        if index > 0 {
            let previous = values[index - 1];
            if value.logit > previous.logit
                || (value.logit == previous.logit && value.token_id < previous.token_id)
            {
                return Err("SQ8 generation top-logit ordering is invalid".into());
            }
        }
    }
    Ok(())
}

fn generation_cache_shape() -> PagedDecodeShape {
    PagedDecodeShape {
        block_size: QWEN3_14B_SQ8_GENERATION_BLOCK_TOKENS,
        cache_blocks: GENERATION_KV_BLOCKS as usize,
        q_heads: QWEN3_14B_Q_HEADS,
        kv_heads: QWEN3_14B_KV_HEADS,
        head_dim: QWEN3_14B_HEAD_DIM,
        value_dim: QWEN3_14B_VALUE_DIM,
    }
}

fn hidden_row_bytes() -> Result<usize, String> {
    QWEN3_14B_HIDDEN_SIZE
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "SQ8 generation hidden row byte size overflows".to_string())
}

fn prompt_buffer_bytes() -> Result<usize, String> {
    QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS
        .checked_mul(hidden_row_bytes()?)
        .ok_or_else(|| "SQ8 generation prompt buffer byte size overflows".to_string())
}

fn elapsed_ns_nonzero(start: Instant) -> u128 {
    start.elapsed().as_nanos().max(1)
}

fn elapsed_ns_strictly_after(start: Instant, previous: u128) -> u128 {
    start.elapsed().as_nanos().max(previous.saturating_add(1))
}

fn token_ids_sha256(token_ids: &[usize]) -> Result<String, String> {
    if token_ids.is_empty() {
        return Err("SQ8 generation cannot hash an empty token sequence".into());
    }
    let mut hasher = Sha256::new();
    for token_id in token_ids {
        let token_id = u32::try_from(*token_id)
            .map_err(|_| format!("SQ8 generation token ID does not fit u32: {token_id}"))?;
        hasher.update(token_id.to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() != SHA256_HEX_LEN
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("invalid lowercase SHA-256: {value}"));
    }
    Ok(())
}

fn validate_component_device_identity(
    embedding: &Sq8EmbeddingDeviceIdentity,
    head: &Sq8ModelHeadDeviceIdentity,
) -> Result<(), String> {
    let matches = embedding.device_id == head.device_id
        && embedding.backend == head.backend
        && embedding.name == head.name
        && embedding.gcn_arch_name == head.gcn_arch_name
        && embedding.compute_major == head.compute_major
        && embedding.compute_minor == head.compute_minor
        && embedding.total_global_mem == head.total_global_mem;
    if !matches {
        return Err(format!(
            "SQ8 generation component device mismatch: embedding={embedding:?} head={head:?}"
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
            "{operation_error}; subsequent SQ8 generation load stream recovery failed: {sync_error}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_request_accepts_only_one_to_eight_tokens() {
        for max_new_tokens in 1..=QWEN3_14B_SQ8_GENERATION_MAX_NEW_TOKENS {
            Sq8GenerationRequest::fixed(max_new_tokens)
                .validate_contract()
                .unwrap();
        }
        assert!(Sq8GenerationRequest::fixed(0).validate_contract().is_err());
        assert!(Sq8GenerationRequest::fixed(9).validate_contract().is_err());
    }

    #[test]
    fn fixed_request_rejects_changed_prompt_and_sampling() {
        let mut request = Sq8GenerationRequest::fixed(4);
        request.prompt_token_ids[0] = 0;
        assert!(request.validate_contract().is_err());

        let mut request = Sq8GenerationRequest::fixed(4);
        request.sampling.temperature = 1.0;
        assert!(request.validate_contract().is_err());
    }

    #[test]
    fn greedy_top1_uses_smallest_token_for_exact_tie() {
        let result = greedy_top1_finite(&[-3.0, 4.5, 4.5, 2.0]).unwrap();
        assert_eq!(result.token_id, 1);
        assert_eq!(result.logit, 4.5);
    }

    #[test]
    fn greedy_top1_rejects_empty_and_nonfinite_logits() {
        assert!(greedy_top1_finite(&[]).is_err());
        assert!(greedy_top1_finite(&[0.0, f32::NAN]).is_err());
        assert!(greedy_top1_finite(&[f32::INFINITY]).is_err());
    }

    #[test]
    fn top_logits_are_sorted_with_stable_ties() {
        let logits = [0.0, 3.0, 3.0, -1.0, 2.0, 1.0, 4.0, 4.0, 2.5, -2.0];
        let top = top_finite_logits(&logits, 5).unwrap();
        assert_eq!(
            top.iter().map(|value| value.token_id).collect::<Vec<_>>(),
            vec![6, 7, 1, 2, 8]
        );
    }

    #[test]
    fn step_contract_recomputes_tensor_hashes_and_top_logits() {
        let final_hidden = vec![0.0_f32; QWEN3_14B_HIDDEN_SIZE];
        let mut logits = vec![0.0_f32; QWEN3_14B_VOCAB_SIZE];
        logits[42] = 1.0;
        let final_hidden_health = validate_sq8_model_head_tensor_health(
            &final_hidden,
            QWEN3_14B_HIDDEN_SIZE,
            "test final hidden",
        )
        .unwrap();
        let logits_health =
            validate_sq8_model_head_tensor_health(&logits, QWEN3_14B_VOCAB_SIZE, "test logits")
                .unwrap();
        let mut step = Sq8GenerationStepResult {
            generated_index: 0,
            phase: Sq8GenerationStepPhase::Prefill,
            input_token_id: None,
            cache_position: None,
            cache_len_after: QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS,
            output_token_id: 42,
            output_logit: 1.0,
            top_logits: top_finite_logits(&logits, QWEN3_14B_SQ8_GENERATION_TOP_LOGITS).unwrap(),
            final_hidden,
            logits,
            final_hidden_f32_le_sha256: final_hidden_health.f32_le_sha256,
            logits_f32_le_sha256: logits_health.f32_le_sha256,
            started_at_ns: 0,
            completed_at_ns: 1,
            latency_ns: 1,
        };
        step.validate_contract().unwrap();
        step.logits[0] = 2.0;
        assert!(step.validate_contract().is_err());
    }

    #[test]
    fn feedback_contract_requires_generated_prefix_and_contiguous_positions() {
        let request = Sq8GenerationRequest::fixed(4);
        validate_feedback_contract(&request, &[10, 20, 30], &[10, 20], &[8, 9], 10).unwrap();
        assert!(
            validate_feedback_contract(&request, &[10, 20, 30], &[10, 30], &[8, 9], 10).is_err()
        );
        assert!(
            validate_feedback_contract(&request, &[10, 20, 30], &[10, 20], &[8, 10], 10).is_err()
        );
        assert!(
            validate_feedback_contract(&request, &[10, 20, 30], &[10, 20], &[8, 9], 11).is_err()
        );
    }

    #[test]
    fn feedback_contract_rejects_generation_after_eos() {
        let request = Sq8GenerationRequest::fixed(4);
        assert!(
            validate_feedback_contract(
                &request,
                &[10, QWEN3_14B_SQ8_GENERATION_EOS_TOKEN_ID, 30],
                &[10, QWEN3_14B_SQ8_GENERATION_EOS_TOKEN_ID],
                &[8, 9],
                10,
            )
            .is_err()
        );
    }

    #[test]
    fn offline_metrics_distinguish_ttft_and_decode() {
        let one = Sq8OfflineGenerationMetrics::measured(1, 10, 10).unwrap();
        assert_eq!(one.decode_elapsed_ns, 0);
        assert_eq!(one.decode_tokens_per_second, None);

        let three = Sq8OfflineGenerationMetrics::measured(3, 10, 30).unwrap();
        assert_eq!(three.decode_elapsed_ns, 20);
        assert!(three.decode_tokens_per_second.unwrap().is_finite());
    }

    #[test]
    fn token_hash_is_stable_little_endian_u32() {
        assert_eq!(
            token_ids_sha256(&[1, 2, 3]).unwrap(),
            "4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d"
        );
    }
}
