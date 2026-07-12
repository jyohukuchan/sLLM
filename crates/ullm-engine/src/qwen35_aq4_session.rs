// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Load-once Qwen3.5 AQ4 inference session.
//!
//! The session owns one resident model and only resets request-owned state between requests.
//! Preparing a token is deliberately separate from publishing and committing it, so cancellation
//! and publisher failures cannot make an unobserved token part of the decode history.

use crate::backend_operation_registry::{
    OperationExecutionAudit, OperationExecutionCount, OperationExecutionRecord,
    OperationExecutionStatus, OperationResolutionTrace,
};
use crate::execution_batch::ExecutionPhase;
use crate::inference_api::{
    CancellationToken, FinishReason, InferenceRequest, ReleaseOutcome, ReleaseSummary,
};
use crate::qwen35_aq4_model_runtime::{Qwen35Aq4ModelLoadConfig, Qwen35Aq4ModelRuntime};
use crate::worker_driver::{InferenceSession, PublishedAdvance, SessionAdvance};
use sha2::{Digest, Sha256};

pub const QWEN35_AQ4_ROTARY_DIM: usize = 64;
pub const QWEN35_AQ4_ROPE_BASE: f32 = 10_000_000.0;
pub const QWEN35_AQ4_MAX_PREFILL_CHUNK: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35PrefillExecutionStep {
    pub layer_index: usize,
    pub execution_width: usize,
    pub phase: ExecutionPhase,
    pub records: [OperationExecutionRecord; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35FailedPrefillExecutionStep {
    pub layer_index: usize,
    pub execution_width: usize,
    pub phase: ExecutionPhase,
    pub records: [Option<OperationExecutionRecord>; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Aq4SessionStatus {
    Ready,
    Prefilling,
    PreparedToken,
    Decoding,
    Terminal,
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Aq4SessionConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<usize>,
    pub rotary_dim: usize,
    pub rope_base: f32,
    pub sync_each_layer_for_timing: bool,
}

impl Qwen35Aq4SessionConfig {
    pub fn greedy(max_new_tokens: usize, eos_token_ids: Vec<usize>) -> Self {
        Self {
            max_new_tokens,
            eos_token_ids,
            rotary_dim: QWEN35_AQ4_ROTARY_DIM,
            rope_base: QWEN35_AQ4_ROPE_BASE,
            sync_each_layer_for_timing: false,
        }
    }
}

/// Minimal model boundary used by the serving state machine and CPU-only contract tests.
pub trait Qwen35Aq4SessionModel {
    fn context_length(&self) -> usize;

    fn vocab_size(&self) -> usize;

    fn operation_resolution_traces(&self) -> Vec<Vec<OperationResolutionTrace>> {
        Vec::new()
    }

    fn dispatch_token(
        &mut self,
        token_id: usize,
        rotary_dim: usize,
        rope_base: f32,
        position: usize,
        _phase: ExecutionPhase,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<Vec<[OperationExecutionRecord; 2]>, String>;

    fn take_failed_operation_executions(&mut self) -> Vec<[Option<OperationExecutionRecord>; 2]> {
        Vec::new()
    }

    fn take_failed_prefill_executions(&mut self) -> Vec<Qwen35FailedPrefillExecutionStep> {
        Vec::new()
    }

    fn dispatch_prefill_chunk(
        &mut self,
        token_ids: &[usize],
        rotary_dim: usize,
        rope_base: f32,
        absolute_start: usize,
        _phase: ExecutionPhase,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<Vec<Qwen35PrefillExecutionStep>, String> {
        let mut executions = Vec::new();
        for (offset, token_id) in token_ids.iter().copied().enumerate() {
            let position = absolute_start
                .checked_add(offset)
                .ok_or_else(|| "prefill chunk position overflows".to_string())?;
            let token_phase = if position == 0 {
                ExecutionPhase::ColdPrefill
            } else {
                ExecutionPhase::CachedPrefixPrefill
            };
            let records = self.dispatch_token(
                token_id,
                rotary_dim,
                rope_base,
                position,
                token_phase,
                sync_each_layer_for_timing,
                label,
            )?;
            executions.reserve(records.len());
            executions.extend(
                records
                    .into_iter()
                    .enumerate()
                    .map(|(layer_index, records)| Qwen35PrefillExecutionStep {
                        layer_index,
                        execution_width: 1,
                        phase: token_phase,
                        records,
                    }),
            );
        }
        Ok(executions)
    }

    fn synchronize_after_prefill_chunk(&mut self) -> Result<(), String> {
        Ok(())
    }

    fn mark_prefill_chunk_uncommitted(&mut self) {}

    fn top_token_from_last_layer(&mut self, label: &str) -> Result<usize, String>;

    fn reset_all_request_state_synchronized(&mut self) -> Result<(), String>;

    fn shutdown_synchronized(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl Qwen35Aq4SessionModel for Qwen35Aq4ModelRuntime {
    fn context_length(&self) -> usize {
        self.geometry().context_length
    }

    fn vocab_size(&self) -> usize {
        self.geometry().vocab
    }

    fn operation_resolution_traces(&self) -> Vec<Vec<OperationResolutionTrace>> {
        Qwen35Aq4ModelRuntime::operation_resolution_traces(self)
    }

    fn dispatch_token(
        &mut self,
        token_id: usize,
        rotary_dim: usize,
        rope_base: f32,
        position: usize,
        phase: ExecutionPhase,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<Vec<[OperationExecutionRecord; 2]>, String> {
        let step = Qwen35Aq4ModelRuntime::dispatch_token_for_phase(
            self,
            token_id,
            rotary_dim,
            rope_base,
            position,
            position,
            phase,
            sync_each_layer_for_timing,
            label,
        )?;
        Ok(step.operation_executions)
    }

    fn dispatch_prefill_chunk(
        &mut self,
        token_ids: &[usize],
        rotary_dim: usize,
        rope_base: f32,
        absolute_start: usize,
        phase: ExecutionPhase,
        sync_each_layer_for_timing: bool,
        label: &str,
    ) -> Result<Vec<Qwen35PrefillExecutionStep>, String> {
        if token_ids.len() == 1 {
            let records = Qwen35Aq4ModelRuntime::dispatch_token_for_phase(
                self,
                token_ids[0],
                rotary_dim,
                rope_base,
                absolute_start,
                absolute_start,
                phase,
                sync_each_layer_for_timing,
                label,
            )?
            .operation_executions;
            return Ok(records
                .into_iter()
                .enumerate()
                .map(|(layer_index, records)| Qwen35PrefillExecutionStep {
                    layer_index,
                    execution_width: 1,
                    phase,
                    records,
                })
                .collect());
        }
        let step = Qwen35Aq4ModelRuntime::dispatch_prefill_chunk_for_phase(
            self,
            token_ids,
            rotary_dim,
            rope_base,
            absolute_start,
            phase,
            sync_each_layer_for_timing,
            label,
        )?;
        Ok(step
            .invocations
            .into_iter()
            .map(|invocation| Qwen35PrefillExecutionStep {
                layer_index: invocation.layer_index,
                execution_width: invocation.execution_width,
                phase: invocation.phase,
                records: invocation.records,
            })
            .collect())
    }

    fn top_token_from_last_layer(&mut self, label: &str) -> Result<usize, String> {
        let logits = self.top_logits_from_last_layer(1, label)?;
        let top = logits
            .first()
            .ok_or_else(|| "Qwen3.5 AQ4 top-1 returned no token".to_string())?;
        if !top.logit.is_finite() {
            return Err("Qwen3.5 AQ4 top-1 returned a non-finite logit".to_string());
        }
        Ok(top.token_id)
    }

    fn take_failed_operation_executions(&mut self) -> Vec<[Option<OperationExecutionRecord>; 2]> {
        self.take_last_partial_operation_executions()
    }

    fn take_failed_prefill_executions(&mut self) -> Vec<Qwen35FailedPrefillExecutionStep> {
        self.take_last_partial_prefill_invocations()
            .into_iter()
            .map(|invocation| Qwen35FailedPrefillExecutionStep {
                layer_index: invocation.layer_index,
                execution_width: invocation.execution_width,
                phase: invocation.phase,
                records: invocation.records,
            })
            .collect()
    }

    fn synchronize_after_prefill_chunk(&mut self) -> Result<(), String> {
        self.synchronize()
    }

    fn mark_prefill_chunk_uncommitted(&mut self) {
        Qwen35Aq4ModelRuntime::mark_prefill_chunk_uncommitted(self)
    }

    fn reset_all_request_state_synchronized(&mut self) -> Result<(), String> {
        Qwen35Aq4ModelRuntime::reset_all_request_state_synchronized(self)
    }

    fn shutdown_synchronized(&mut self) -> Result<(), String> {
        self.synchronize()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4PreparedToken {
    pub token_id: usize,
    pub generated_index: usize,
    pub cache_len: usize,
    pub terminal_reason: Option<FinishReason>,
    nonce: u64,
}

const EXECUTION_IMPLEMENTATIONS: [(&str, &str); 8] = [
    (
        "linear_attention_qkv_prepare",
        "hip.linear-attention-qkv-prepare-f32.m1",
    ),
    (
        "gated_delta_rule_scan",
        "hip.linear-attention-recurrent-f32.m1",
    ),
    ("paged_kv_write", "hip.paged-kv-write-f32.m1"),
    (
        "fused_qk_norm_rope_paged_kv_write",
        "hip.fused-qk-norm-rope-paged-kv-write-f32.m1",
    ),
    (
        "paged_causal_gqa_read",
        "hip.paged-decode-attention-f32.m1-gqa",
    ),
    (
        "paged_causal_gqa_read",
        "hip.paged-decode-attention-sigmoid-gate-f32.m1-gqa",
    ),
    (
        "linear_attention_qkv_prepare",
        "hip.linear-attention-qkv-prepare-batch-f32.m2-m128",
    ),
    (
        "gated_delta_rule_scan",
        "hip.linear-attention-recurrent-sequence-f32.m2-m128",
    ),
];

type LayerExecutionContract = [[&'static str; 2]; 3];

struct OperationAuditAccumulator {
    cold_prefill_steps: u64,
    cached_prefix_prefill_steps: u64,
    decode_steps: u64,
    total_steps: u64,
    total_records: u64,
    token_equivalent_operation_coverage: u64,
    prefill_chunks_executed: u64,
    prefill_tokens_executed: u64,
    prefill_tokens_committed: u64,
    prefill_width_histogram: Vec<u64>,
    implementation_counts: [u64; 8],
    digest: Sha256,
}

impl OperationAuditAccumulator {
    fn new() -> Self {
        Self {
            cold_prefill_steps: 0,
            cached_prefix_prefill_steps: 0,
            decode_steps: 0,
            total_steps: 0,
            total_records: 0,
            token_equivalent_operation_coverage: 0,
            prefill_chunks_executed: 0,
            prefill_tokens_executed: 0,
            prefill_tokens_committed: 0,
            prefill_width_histogram: vec![0; QWEN35_AQ4_MAX_PREFILL_CHUNK + 1],
            implementation_counts: [0; 8],
            digest: Sha256::new(),
        }
    }

    fn observe(
        &mut self,
        phase: ExecutionPhase,
        contract: &[LayerExecutionContract],
        records: &[[OperationExecutionRecord; 2]],
    ) -> Result<(), String> {
        if records.len() != contract.len() || records.is_empty() {
            return Err(format!(
                "operation execution trace layer coverage mismatch: expected={} actual={}",
                contract.len(),
                records.len()
            ));
        }
        let phase_index = execution_phase_index(phase);
        for (layer_index, layer_records) in records.iter().enumerate() {
            for (record_index, record) in layer_records.iter().enumerate() {
                let expected = contract[layer_index][phase_index][record_index];
                if record.phase != phase
                    || record.status != OperationExecutionStatus::Succeeded
                    || record.implementation_id != expected
                {
                    return Err(format!(
                        "operation execution trace mismatch at layer={layer_index} record={record_index}"
                    ));
                }
                let slot = EXECUTION_IMPLEMENTATIONS
                    .iter()
                    .position(|(_, implementation_id)| {
                        *implementation_id == record.implementation_id
                    })
                    .ok_or_else(|| {
                        format!(
                            "operation execution trace contains unknown implementation {}",
                            record.implementation_id
                        )
                    })?;
                self.implementation_counts[slot] = self.implementation_counts[slot]
                    .checked_add(1)
                    .ok_or_else(|| {
                        "operation execution implementation count overflows".to_string()
                    })?;
                self.digest.update([phase_index as u8]);
                self.digest.update(self.total_steps.to_le_bytes());
                self.digest.update((layer_index as u64).to_le_bytes());
                self.digest.update([record_index as u8]);
                self.digest.update(record.implementation_id.as_bytes());
                self.digest.update([0]);
            }
        }
        match phase {
            ExecutionPhase::ColdPrefill => self.cold_prefill_steps += 1,
            ExecutionPhase::CachedPrefixPrefill => self.cached_prefix_prefill_steps += 1,
            ExecutionPhase::Decode => self.decode_steps += 1,
        }
        self.total_steps = self
            .total_steps
            .checked_add(1)
            .ok_or_else(|| "operation execution step count overflows".to_string())?;
        self.total_records = self
            .total_records
            .checked_add(
                u64::try_from(records.len() * 2)
                    .map_err(|_| "operation execution record count does not fit u64".to_string())?,
            )
            .ok_or_else(|| "operation execution record count overflows".to_string())?;
        self.token_equivalent_operation_coverage =
            self.token_equivalent_operation_coverage
                .checked_add(u64::try_from(records.len() * 2).map_err(|_| {
                    "token-equivalent operation coverage does not fit u64".to_string()
                })?)
                .ok_or_else(|| "token-equivalent operation coverage overflows".to_string())?;
        Ok(())
    }

    fn observe_prefill_chunk(
        &mut self,
        phase: ExecutionPhase,
        execution_width: usize,
        contract: &[LayerExecutionContract],
        invocations: &[Qwen35PrefillExecutionStep],
    ) -> Result<(), String> {
        if !(1..=QWEN35_AQ4_MAX_PREFILL_CHUNK).contains(&execution_width) {
            return Err(format!(
                "prefill execution width is outside 1..={QWEN35_AQ4_MAX_PREFILL_CHUNK}: {execution_width}"
            ));
        }
        if phase == ExecutionPhase::Decode {
            return Err("decode phase cannot be recorded as a prefill chunk".into());
        }
        if contract.is_empty() || invocations.is_empty() {
            return Err("prefill execution trace must contain layers and invocations".into());
        }
        let mut layer_widths = vec![0_usize; contract.len()];
        for (invocation_index, invocation) in invocations.iter().enumerate() {
            if invocation.layer_index >= contract.len() {
                return Err(format!(
                    "prefill invocation {invocation_index} layer {} exceeds contract layers {}",
                    invocation.layer_index,
                    contract.len()
                ));
            }
            if !(1..=execution_width).contains(&invocation.execution_width) {
                return Err(format!(
                    "prefill invocation {invocation_index} width {} is outside 1..={execution_width}",
                    invocation.execution_width
                ));
            }
            if invocation.phase == ExecutionPhase::Decode {
                return Err(format!(
                    "prefill invocation {invocation_index} cannot use decode phase"
                ));
            }
            layer_widths[invocation.layer_index] = layer_widths[invocation.layer_index]
                .checked_add(invocation.execution_width)
                .ok_or_else(|| "prefill layer coverage width overflows".to_string())?;
            let phase_index = execution_phase_index(invocation.phase);
            for (operation_index, record) in invocation.records.iter().enumerate() {
                let expected = prefill_expected_implementation(
                    contract[invocation.layer_index][phase_index][operation_index],
                    invocation.execution_width,
                );
                if record.phase != invocation.phase
                    || record.status != OperationExecutionStatus::Succeeded
                    || record.implementation_id != expected
                {
                    return Err(format!(
                        "prefill operation execution trace mismatch at invocation={invocation_index} layer={} operation={operation_index}",
                        invocation.layer_index
                    ));
                }
                let slot = EXECUTION_IMPLEMENTATIONS
                    .iter()
                    .position(|(_, implementation_id)| {
                        *implementation_id == record.implementation_id
                    })
                    .ok_or_else(|| {
                        format!(
                            "prefill operation trace contains unknown implementation {}",
                            record.implementation_id
                        )
                    })?;
                self.implementation_counts[slot] = self.implementation_counts[slot]
                    .checked_add(1)
                    .ok_or_else(|| {
                        "operation execution implementation count overflows".to_string()
                    })?;
                self.total_records = self
                    .total_records
                    .checked_add(1)
                    .ok_or_else(|| "operation execution record count overflows".to_string())?;
                self.token_equivalent_operation_coverage = self
                    .token_equivalent_operation_coverage
                    .checked_add(
                        u64::try_from(invocation.execution_width)
                            .map_err(|_| "prefill invocation width does not fit u64".to_string())?,
                    )
                    .ok_or_else(|| "token-equivalent operation coverage overflows".to_string())?;
                self.digest.update(b"prefill-invocation-v2\0");
                self.digest.update((invocation_index as u64).to_le_bytes());
                self.digest
                    .update((invocation.layer_index as u64).to_le_bytes());
                self.digest
                    .update((invocation.execution_width as u64).to_le_bytes());
                self.digest.update([phase_index as u8]);
                self.digest.update([operation_index as u8]);
                self.digest.update(record.implementation_id.as_bytes());
                self.digest.update([0]);
            }
        }
        if let Some((layer_index, covered)) = layer_widths
            .iter()
            .copied()
            .enumerate()
            .find(|(_, covered)| *covered != execution_width)
        {
            return Err(format!(
                "prefill layer coverage mismatch at layer={layer_index}: expected={execution_width} actual={covered}"
            ));
        }
        let width_u64 = u64::try_from(execution_width)
            .map_err(|_| "prefill execution width does not fit u64".to_string())?;
        match phase {
            ExecutionPhase::ColdPrefill => {
                self.cold_prefill_steps = self
                    .cold_prefill_steps
                    .checked_add(1)
                    .ok_or_else(|| "cold-prefill step count overflows".to_string())?;
                self.cached_prefix_prefill_steps = self
                    .cached_prefix_prefill_steps
                    .checked_add(width_u64 - 1)
                    .ok_or_else(|| "cached-prefix step count overflows".to_string())?;
            }
            ExecutionPhase::CachedPrefixPrefill => {
                self.cached_prefix_prefill_steps = self
                    .cached_prefix_prefill_steps
                    .checked_add(width_u64)
                    .ok_or_else(|| "cached-prefix step count overflows".to_string())?;
            }
            ExecutionPhase::Decode => unreachable!("decode rejected above"),
        }
        self.total_steps = self
            .total_steps
            .checked_add(width_u64)
            .ok_or_else(|| "operation execution step count overflows".to_string())?;
        self.prefill_chunks_executed = self
            .prefill_chunks_executed
            .checked_add(1)
            .ok_or_else(|| "prefill chunk count overflows".to_string())?;
        self.prefill_tokens_executed = self
            .prefill_tokens_executed
            .checked_add(width_u64)
            .ok_or_else(|| "prefill executed-token count overflows".to_string())?;
        self.prefill_width_histogram[execution_width] = self.prefill_width_histogram
            [execution_width]
            .checked_add(1)
            .ok_or_else(|| "prefill width histogram count overflows".to_string())?;
        self.digest.update(b"prefill-chunk-v2\0");
        self.digest.update([execution_phase_index(phase) as u8]);
        self.digest
            .update(self.prefill_chunks_executed.to_le_bytes());
        self.digest.update((execution_width as u64).to_le_bytes());
        Ok(())
    }

    fn commit_prefill_chunk(&mut self, execution_width: usize) -> Result<(), String> {
        self.prefill_tokens_committed = self
            .prefill_tokens_committed
            .checked_add(
                u64::try_from(execution_width)
                    .map_err(|_| "prefill commit width does not fit u64".to_string())?,
            )
            .ok_or_else(|| "prefill committed-token count overflows".to_string())?;
        Ok(())
    }

    fn finish(
        &self,
        layers: usize,
        expected_cold: u64,
        expected_cached: u64,
        expected_decode: u64,
        outcome: &'static str,
    ) -> Result<OperationExecutionAudit, String> {
        let coverage_complete = self.cold_prefill_steps == expected_cold
            && self.cached_prefix_prefill_steps == expected_cached
            && self.decode_steps == expected_decode
            && self.total_steps == expected_cold + expected_cached + expected_decode
            && self.token_equivalent_operation_coverage
                == self
                    .total_steps
                    .checked_mul(u64::try_from(layers * 2).map_err(|_| {
                        "expected token-equivalent operation coverage does not fit u64".to_string()
                    })?)
                    .ok_or_else(|| {
                        "expected token-equivalent operation coverage overflows".to_string()
                    })?
            && self.prefill_tokens_executed == expected_cold + expected_cached
            && self.prefill_tokens_committed == expected_cold + expected_cached;
        if !coverage_complete {
            return Err("operation execution terminal coverage is incomplete".into());
        }
        let deterministic_digest_sha256 = self.digest.clone().finalize().into();
        Ok(OperationExecutionAudit {
            schema_version: "ullm.backend_operation.request.v2",
            outcome,
            expected_layers_per_step: layers,
            expected_records_per_layer: 2,
            cold_prefill_steps: self.cold_prefill_steps,
            cached_prefix_prefill_steps: self.cached_prefix_prefill_steps,
            decode_steps: self.decode_steps,
            total_steps: self.total_steps,
            total_records: self.total_records,
            physical_operation_invocations: self.total_records,
            token_equivalent_operation_coverage: self.token_equivalent_operation_coverage,
            prefill_chunks_executed: self.prefill_chunks_executed,
            prefill_tokens_executed: self.prefill_tokens_executed,
            prefill_tokens_committed: self.prefill_tokens_committed,
            prefill_width_histogram: self.prefill_width_histogram.clone(),
            implementation_counts: std::array::from_fn(|index| OperationExecutionCount {
                kind: EXECUTION_IMPLEMENTATIONS[index].0,
                implementation_id: EXECUTION_IMPLEMENTATIONS[index].1,
                count: self.implementation_counts[index],
            }),
            deterministic_digest_sha256,
            coverage_complete,
            failed_phase: None,
            failed_layer: None,
            failed_operation: None,
        })
    }

    fn observe_failed_step(
        &mut self,
        phase: ExecutionPhase,
        contract: &[LayerExecutionContract],
        records: &[[Option<OperationExecutionRecord>; 2]],
    ) -> Result<(Option<usize>, Option<usize>), String> {
        if records.len() > contract.len() {
            return Err("failed operation trace exceeds layer contract".into());
        }
        let phase_index = execution_phase_index(phase);
        let mut failure = (None, None);
        for (layer_index, layer) in records.iter().enumerate() {
            for (operation_index, record) in layer.iter().enumerate() {
                let Some(record) = record else { continue };
                if record.phase != phase
                    || record.implementation_id
                        != contract[layer_index][phase_index][operation_index]
                {
                    return Err("failed operation trace does not match resolved contract".into());
                }
                match record.status {
                    OperationExecutionStatus::Succeeded => {
                        let slot = EXECUTION_IMPLEMENTATIONS
                            .iter()
                            .position(|(_, id)| *id == record.implementation_id)
                            .ok_or_else(|| {
                                "unknown successful partial implementation".to_string()
                            })?;
                        self.implementation_counts[slot] = self.implementation_counts[slot]
                            .checked_add(1)
                            .ok_or_else(|| "partial implementation count overflows".to_string())?;
                        self.total_records =
                            self.total_records.checked_add(1).ok_or_else(|| {
                                "partial operation record count overflows".to_string()
                            })?;
                        self.token_equivalent_operation_coverage = self
                            .token_equivalent_operation_coverage
                            .checked_add(1)
                            .ok_or_else(|| {
                                "partial token-equivalent coverage overflows".to_string()
                            })?;
                        self.digest.update([phase_index as u8]);
                        self.digest.update(self.total_steps.to_le_bytes());
                        self.digest.update((layer_index as u64).to_le_bytes());
                        self.digest.update([operation_index as u8]);
                        self.digest.update(record.implementation_id.as_bytes());
                        self.digest.update([0]);
                    }
                    OperationExecutionStatus::Failed => {
                        if failure.0.is_some() {
                            return Err("failed operation trace contains multiple failures".into());
                        }
                        failure = (Some(layer_index), Some(operation_index));
                    }
                    OperationExecutionStatus::Started => {
                        return Err("failed operation trace retained an unclassified start".into());
                    }
                }
            }
        }
        Ok(failure)
    }

    fn observe_failed_prefill(
        &mut self,
        contract: &[LayerExecutionContract],
        invocations: &[Qwen35FailedPrefillExecutionStep],
    ) -> Result<(Option<usize>, Option<usize>), String> {
        let mut failure = (None, None);
        for (invocation_index, invocation) in invocations.iter().enumerate() {
            if invocation.layer_index >= contract.len()
                || !(1..=QWEN35_AQ4_MAX_PREFILL_CHUNK).contains(&invocation.execution_width)
                || invocation.phase == ExecutionPhase::Decode
            {
                return Err(format!(
                    "failed prefill invocation {invocation_index} has invalid metadata"
                ));
            }
            let phase_index = execution_phase_index(invocation.phase);
            for (operation_index, record) in invocation.records.iter().enumerate() {
                let Some(record) = record else { continue };
                let expected = prefill_expected_implementation(
                    contract[invocation.layer_index][phase_index][operation_index],
                    invocation.execution_width,
                );
                if record.phase != invocation.phase || record.implementation_id != expected {
                    return Err(format!(
                        "failed prefill invocation {invocation_index} does not match resolved contract"
                    ));
                }
                match record.status {
                    OperationExecutionStatus::Succeeded => {
                        let slot = EXECUTION_IMPLEMENTATIONS
                            .iter()
                            .position(|(_, id)| *id == record.implementation_id)
                            .ok_or_else(|| {
                                "unknown successful partial prefill implementation".to_string()
                            })?;
                        self.implementation_counts[slot] = self.implementation_counts[slot]
                            .checked_add(1)
                            .ok_or_else(|| "partial implementation count overflows".to_string())?;
                        self.total_records =
                            self.total_records.checked_add(1).ok_or_else(|| {
                                "partial operation record count overflows".to_string()
                            })?;
                        self.token_equivalent_operation_coverage = self
                            .token_equivalent_operation_coverage
                            .checked_add(u64::try_from(invocation.execution_width).map_err(
                                |_| "partial prefill width does not fit u64".to_string(),
                            )?)
                            .ok_or_else(|| {
                                "partial token-equivalent coverage overflows".to_string()
                            })?;
                        self.digest.update(b"failed-prefill-invocation-v2\0");
                        self.digest.update((invocation_index as u64).to_le_bytes());
                        self.digest
                            .update((invocation.layer_index as u64).to_le_bytes());
                        self.digest
                            .update((invocation.execution_width as u64).to_le_bytes());
                        self.digest
                            .update([phase_index as u8, operation_index as u8]);
                        self.digest.update(record.implementation_id.as_bytes());
                        self.digest.update([0]);
                    }
                    OperationExecutionStatus::Failed => {
                        if failure.0.is_some() {
                            return Err("failed prefill trace contains multiple failures".into());
                        }
                        failure = (Some(invocation.layer_index), Some(operation_index));
                    }
                    OperationExecutionStatus::Started => {
                        return Err("failed prefill trace retained an unclassified start".into());
                    }
                }
            }
        }
        Ok(failure)
    }

    fn partial(
        &self,
        layers: usize,
        outcome: &'static str,
        failed_phase: Option<ExecutionPhase>,
        failed_layer: Option<usize>,
        failed_operation: Option<usize>,
    ) -> OperationExecutionAudit {
        OperationExecutionAudit {
            schema_version: "ullm.backend_operation.request.v2",
            outcome,
            expected_layers_per_step: layers,
            expected_records_per_layer: 2,
            cold_prefill_steps: self.cold_prefill_steps,
            cached_prefix_prefill_steps: self.cached_prefix_prefill_steps,
            decode_steps: self.decode_steps,
            total_steps: self.total_steps,
            total_records: self.total_records,
            physical_operation_invocations: self.total_records,
            token_equivalent_operation_coverage: self.token_equivalent_operation_coverage,
            prefill_chunks_executed: self.prefill_chunks_executed,
            prefill_tokens_executed: self.prefill_tokens_executed,
            prefill_tokens_committed: self.prefill_tokens_committed,
            prefill_width_histogram: self.prefill_width_histogram.clone(),
            implementation_counts: std::array::from_fn(|index| OperationExecutionCount {
                kind: EXECUTION_IMPLEMENTATIONS[index].0,
                implementation_id: EXECUTION_IMPLEMENTATIONS[index].1,
                count: self.implementation_counts[index],
            }),
            deterministic_digest_sha256: self.digest.clone().finalize().into(),
            coverage_complete: false,
            failed_phase: failed_phase.map(execution_phase_name),
            failed_layer,
            failed_operation,
        }
    }
}

fn prefill_expected_implementation(
    m1_implementation: &'static str,
    execution_width: usize,
) -> &'static str {
    if execution_width <= 1 {
        return m1_implementation;
    }
    match m1_implementation {
        "hip.linear-attention-qkv-prepare-f32.m1" => {
            "hip.linear-attention-qkv-prepare-batch-f32.m2-m128"
        }
        "hip.linear-attention-recurrent-f32.m1" => {
            "hip.linear-attention-recurrent-sequence-f32.m2-m128"
        }
        _ => m1_implementation,
    }
}

fn execution_phase_index(phase: ExecutionPhase) -> usize {
    match phase {
        ExecutionPhase::ColdPrefill => 0,
        ExecutionPhase::CachedPrefixPrefill => 1,
        ExecutionPhase::Decode => 2,
    }
}

fn execution_phase_name(phase: ExecutionPhase) -> &'static str {
    match phase {
        ExecutionPhase::ColdPrefill => "cold_prefill",
        ExecutionPhase::CachedPrefixPrefill => "cached_prefix_prefill",
        ExecutionPhase::Decode => "decode",
    }
}

fn build_execution_contract(
    traces: Vec<Vec<OperationResolutionTrace>>,
) -> Result<Option<Vec<LayerExecutionContract>>, String> {
    if traces.is_empty() {
        return Ok(None);
    }
    if traces.len() != 32 {
        return Err(format!(
            "Qwen3.5 AQ4 operation execution audit requires exactly 32 layers, got {}",
            traces.len()
        ));
    }
    let mut contract = Vec::with_capacity(32);
    for (layer_index, layer_traces) in traces.into_iter().enumerate() {
        if layer_traces.is_empty() {
            return Err(format!(
                "Qwen3.5 AQ4 operation trace layer {layer_index} is empty"
            ));
        }
        let mut phases = [[""; 2]; 3];
        for phase in [
            ExecutionPhase::ColdPrefill,
            ExecutionPhase::CachedPrefixPrefill,
            ExecutionPhase::Decode,
        ] {
            let selected = layer_traces
                .iter()
                .filter(|trace| trace.phase == phase)
                .collect::<Vec<_>>();
            if selected.len() != 2 {
                return Err(format!(
                    "Qwen3.5 AQ4 operation trace layer {layer_index} phase {phase:?} must contain exactly two operations"
                ));
            }
            phases[execution_phase_index(phase)] =
                [selected[0].implementation_id, selected[1].implementation_id];
        }
        contract.push(phases);
    }
    Ok(Some(contract))
}

#[derive(Debug)]
struct ActiveRequest {
    request_id: String,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
    cancel: CancellationToken,
    prompt_tokens_processed: usize,
    generated_tokens: usize,
    decode_input: Option<usize>,
    terminal_outcome: Option<ReleaseOutcome>,
}

pub struct Qwen35Aq4InferenceSession<M = Qwen35Aq4ModelRuntime> {
    model: M,
    config: Qwen35Aq4SessionConfig,
    status: Qwen35Aq4SessionStatus,
    active: Option<ActiveRequest>,
    pending: Option<Qwen35Aq4PreparedToken>,
    next_nonce: u64,
    execution_contract: Option<Vec<LayerExecutionContract>>,
    active_operation_audit: Option<OperationAuditAccumulator>,
    last_terminal_operation_audit: Option<OperationExecutionAudit>,
}

impl Qwen35Aq4InferenceSession<Qwen35Aq4ModelRuntime> {
    /// Loads the resident model exactly once and transfers it into the reusable session.
    pub fn load(
        model_config: Qwen35Aq4ModelLoadConfig,
        session_config: Qwen35Aq4SessionConfig,
    ) -> Result<Self, String> {
        let model = Qwen35Aq4ModelRuntime::load(model_config)?;
        if !model.has_resident_embedding() || !model.supports_device_logits() {
            return Err(
                "Qwen3.5 AQ4 session requires resident embedding, final norm, and LM head"
                    .to_string(),
            );
        }
        Self::from_model(model, session_config)
    }
}

impl<M: Qwen35Aq4SessionModel> Qwen35Aq4InferenceSession<M> {
    pub fn from_model(model: M, config: Qwen35Aq4SessionConfig) -> Result<Self, String> {
        validate_config(&model, &config)?;
        let execution_contract = build_execution_contract(model.operation_resolution_traces())?;
        Ok(Self {
            model,
            config,
            status: Qwen35Aq4SessionStatus::Ready,
            active: None,
            pending: None,
            next_nonce: 0,
            execution_contract,
            active_operation_audit: None,
            last_terminal_operation_audit: None,
        })
    }

    pub fn status(&self) -> Qwen35Aq4SessionStatus {
        self.status
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn operation_resolution_traces(&self) -> Vec<Vec<OperationResolutionTrace>> {
        self.model.operation_resolution_traces()
    }

    pub fn last_terminal_operation_audit(&self) -> Option<&OperationExecutionAudit> {
        self.last_terminal_operation_audit.as_ref()
    }

    fn fail<T>(&mut self, message: impl Into<String>) -> Result<T, String> {
        self.status = Qwen35Aq4SessionStatus::Failed;
        Err(message.into())
    }

    fn prepare_prefill_chunk(&mut self) -> Result<SessionAdvance<Qwen35Aq4PreparedToken>, String> {
        let (absolute_start, token_ids, cancel) = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "Qwen3.5 AQ4 prefill chunk has no active request".to_string())?;
            let absolute_start = active.prompt_tokens_processed;
            let end = absolute_start
                .checked_add(QWEN35_AQ4_MAX_PREFILL_CHUNK)
                .unwrap_or(usize::MAX)
                .min(active.prompt_token_ids.len());
            (
                absolute_start,
                active.prompt_token_ids[absolute_start..end].to_vec(),
                active.cancel.clone(),
            )
        };
        if token_ids.is_empty() {
            return self.prepare_token("Qwen3.5 AQ4 prefill");
        }
        let phase = if absolute_start == 0 {
            ExecutionPhase::ColdPrefill
        } else {
            ExecutionPhase::CachedPrefixPrefill
        };
        let execution_width = token_ids.len();
        let execution_steps = match self.model.dispatch_prefill_chunk(
            &token_ids,
            self.config.rotary_dim,
            self.config.rope_base,
            absolute_start,
            phase,
            self.config.sync_each_layer_for_timing,
            "Qwen3.5 AQ4 prefill chunk",
        ) {
            Ok(steps) => steps,
            Err(error) => {
                let partial_prefill = self.model.take_failed_prefill_executions();
                let partial_records = self.model.take_failed_operation_executions();
                let failure = match (
                    self.active_operation_audit.as_mut(),
                    self.execution_contract.as_deref(),
                ) {
                    (Some(audit), Some(contract)) if !partial_prefill.is_empty() => audit
                        .observe_failed_prefill(contract, &partial_prefill)
                        .unwrap_or((None, None)),
                    (Some(audit), Some(contract)) => audit
                        .observe_failed_step(phase, contract, &partial_records)
                        .unwrap_or((None, None)),
                    _ => (None, None),
                };
                self.last_terminal_operation_audit =
                    self.active_operation_audit.as_ref().map(|audit| {
                        audit.partial(
                            self.execution_contract.as_ref().map_or(0, Vec::len),
                            "execution_failed",
                            Some(phase),
                            failure.0,
                            failure.1,
                        )
                    });
                return self.fail(format!(
                    "Qwen3.5 AQ4 prefill chunk dispatch failed: {error}"
                ));
            }
        };
        // Dispatch returns only after every requested invocation has been launched and its
        // successful operation records have been collected.  Account for that physical work
        // before the stream synchronization below: synchronization can fail after dispatch, and
        // a partial terminal audit must still retain the invocation and token-equivalent counts.
        // Commit/progress stay below synchronization and the post-sync cancellation check.
        if let (Some(contract), Some(audit)) = (
            self.execution_contract.as_deref(),
            self.active_operation_audit.as_mut(),
        ) {
            if let Err(error) =
                audit.observe_prefill_chunk(phase, execution_width, contract, &execution_steps)
            {
                self.model.mark_prefill_chunk_uncommitted();
                return self.fail(format!(
                    "Qwen3.5 AQ4 prefill chunk operation audit failed: {error}"
                ));
            }
        }
        if let Err(error) = self.model.synchronize_after_prefill_chunk() {
            self.model.mark_prefill_chunk_uncommitted();
            self.last_terminal_operation_audit =
                self.active_operation_audit.as_ref().map(|audit| {
                    audit.partial(
                        self.execution_contract.as_ref().map_or(0, Vec::len),
                        "synchronization_failed",
                        Some(phase),
                        None,
                        None,
                    )
                });
            return self.fail(format!(
                "Qwen3.5 AQ4 prefill chunk synchronization failed: {error}"
            ));
        }
        if cancel.is_cancelled() {
            self.model.mark_prefill_chunk_uncommitted();
            let active = self.active.as_mut().expect("active request checked above");
            active.terminal_outcome = Some(ReleaseOutcome::Cancelled);
            self.status = Qwen35Aq4SessionStatus::Terminal;
            return Ok(SessionAdvance::CancellationObserved);
        }
        let next_prompt_tokens_processed = self
            .active
            .as_ref()
            .expect("active request checked above")
            .prompt_tokens_processed
            .checked_add(execution_width)
            .ok_or_else(|| "Qwen3.5 AQ4 prefill progress overflows".to_string())?;
        if let Some(audit) = self.active_operation_audit.as_mut() {
            if let Err(error) = audit.commit_prefill_chunk(execution_width) {
                return self.fail(format!(
                    "Qwen3.5 AQ4 prefill chunk commit audit failed: {error}"
                ));
            }
        }
        let active = self.active.as_mut().expect("active request checked above");
        active.prompt_tokens_processed = next_prompt_tokens_processed;
        Ok(SessionAdvance::PromptProgress {
            prompt_tokens_processed: active.prompt_tokens_processed,
            cache_len: active.prompt_tokens_processed,
            execution_width,
        })
    }

    fn prepare_token(
        &mut self,
        label: &str,
    ) -> Result<SessionAdvance<Qwen35Aq4PreparedToken>, String> {
        let token_id = match self.model.top_token_from_last_layer(label) {
            Ok(token_id) => token_id,
            Err(error) => return self.fail(format!("{label} top-1 failed: {error}")),
        };
        if token_id >= self.model.vocab_size() {
            return self.fail(format!(
                "{label} top-1 token {token_id} exceeds vocabulary size {}",
                self.model.vocab_size()
            ));
        }
        let (generated_index, cache_len, next_generated, max_new_tokens) =
            match self.active.as_ref() {
                Some(active) => {
                    let generated_index = active.generated_tokens;
                    let Some(cache_len) =
                        active.prompt_token_ids.len().checked_add(generated_index)
                    else {
                        return self.fail("Qwen3.5 AQ4 prepared cache length overflows");
                    };
                    let Some(next_generated) = generated_index.checked_add(1) else {
                        return self.fail("Qwen3.5 AQ4 generated token count overflows");
                    };
                    (
                        generated_index,
                        cache_len,
                        next_generated,
                        active.max_new_tokens,
                    )
                }
                None => {
                    return self.fail("Qwen3.5 AQ4 token preparation has no active request");
                }
            };
        let terminal_reason = if self.config.eos_token_ids.contains(&token_id) {
            Some(FinishReason::Stop)
        } else if next_generated == max_new_tokens {
            Some(FinishReason::Length)
        } else {
            None
        };
        let Some(next_nonce) = self.next_nonce.checked_add(1) else {
            return self.fail("Qwen3.5 AQ4 prepared token nonce overflows");
        };
        let prepared = Qwen35Aq4PreparedToken {
            token_id,
            generated_index,
            cache_len,
            terminal_reason,
            nonce: self.next_nonce,
        };
        self.next_nonce = next_nonce;
        self.pending = Some(prepared.clone());
        self.status = Qwen35Aq4SessionStatus::PreparedToken;
        Ok(SessionAdvance::Token {
            token_id,
            generated_index,
            cache_len,
            terminal_reason,
            prepared,
        })
    }

    fn reset_with_outcome(&mut self, outcome: ReleaseOutcome) -> Result<ReleaseSummary, String> {
        let active = self
            .active
            .as_ref()
            .ok_or_else(|| "Qwen3.5 AQ4 reset has no active request".to_string())?;
        let summary = ReleaseSummary {
            request_id: active.request_id.clone(),
            outcome,
            prompt_tokens: active.prompt_token_ids.len(),
            generated_tokens: active.generated_tokens,
            reset_complete: true,
        };
        let audit_layers = self.execution_contract.as_ref().map_or(0, Vec::len);
        // A failed dispatch/synchronization already produced the authoritative terminal audit.
        // Resetting that request is allowed as a cleanup-only transition, but must not relabel
        // the failure as an ordinary cancellation in the retained audit.
        let failed_terminal_audit = (self.status == Qwen35Aq4SessionStatus::Failed)
            .then(|| self.last_terminal_operation_audit.clone())
            .flatten();
        let terminal_audit = if let Some(audit) = failed_terminal_audit {
            Some(audit)
        } else if let Some(audit) = self.active_operation_audit.as_ref() {
            if outcome == ReleaseOutcome::Cancelled {
                Some(audit.partial(audit_layers, "cancelled", None, None, None))
            } else {
                let expected_cold = u64::from(!active.prompt_token_ids.is_empty());
                let expected_cached = u64::try_from(
                    active.prompt_token_ids.len().saturating_sub(1),
                )
                .map_err(|_| "expected cached-prefix step count does not fit u64".to_string())?;
                let expected_decode = u64::try_from(active.generated_tokens.saturating_sub(1))
                    .map_err(|_| "expected decode step count does not fit u64".to_string())?;
                Some(audit.finish(
                    audit_layers,
                    expected_cold,
                    expected_cached,
                    expected_decode,
                    match outcome {
                        ReleaseOutcome::Stop => "stop",
                        ReleaseOutcome::Length => "length",
                        ReleaseOutcome::Cancelled => unreachable!(),
                    },
                )?)
            }
        } else {
            None
        };
        if let Err(error) = self.model.reset_all_request_state_synchronized() {
            self.last_terminal_operation_audit = self
                .active_operation_audit
                .as_ref()
                .map(|audit| audit.partial(audit_layers, "reset_failed", None, None, None));
            self.status = Qwen35Aq4SessionStatus::Failed;
            return Err(format!("Qwen3.5 AQ4 request reset failed: {error}"));
        }
        self.active = None;
        self.active_operation_audit = None;
        self.last_terminal_operation_audit = terminal_audit;
        self.pending = None;
        self.status = Qwen35Aq4SessionStatus::Ready;
        Ok(summary)
    }
}

impl<M: Qwen35Aq4SessionModel> InferenceSession for Qwen35Aq4InferenceSession<M> {
    type Prepared = Qwen35Aq4PreparedToken;

    fn start_request(
        &mut self,
        request: InferenceRequest,
        cancel: CancellationToken,
    ) -> Result<(), String> {
        if self.status != Qwen35Aq4SessionStatus::Ready {
            return Err(format!(
                "Qwen3.5 AQ4 start requires Ready, got {:?}",
                self.status
            ));
        }
        request
            .validate_for_worker(
                self.model.context_length(),
                self.config.max_new_tokens,
                self.model.vocab_size(),
                &self.config.eos_token_ids,
                1,
            )
            .map_err(|error| error.to_string())?;
        if request.sampling.temperature != 0.0 || request.sampling.top_p != 1.0 {
            return Err(
                "Qwen3.5 AQ4 session supports greedy sampling only (temperature=0, top_p=1, top_k=1)"
                    .to_string(),
            );
        }
        self.active = Some(ActiveRequest {
            request_id: request.request_id,
            prompt_token_ids: request.prompt_token_ids,
            max_new_tokens: request.max_new_tokens,
            cancel,
            prompt_tokens_processed: 0,
            generated_tokens: 0,
            decode_input: None,
            terminal_outcome: None,
        });
        self.active_operation_audit = self
            .execution_contract
            .as_ref()
            .map(|_| OperationAuditAccumulator::new());
        self.last_terminal_operation_audit = None;
        self.pending = None;
        self.status = Qwen35Aq4SessionStatus::Prefilling;
        Ok(())
    }

    fn prepare_advance(&mut self) -> Result<SessionAdvance<Self::Prepared>, String> {
        if !matches!(
            self.status,
            Qwen35Aq4SessionStatus::Prefilling | Qwen35Aq4SessionStatus::Decoding
        ) {
            return Err(format!(
                "Qwen3.5 AQ4 prepare requires Prefilling or Decoding, got {:?}",
                self.status
            ));
        }
        let cancelled = self
            .active
            .as_ref()
            .ok_or_else(|| "Qwen3.5 AQ4 prepare has no active request".to_string())?
            .cancel
            .is_cancelled();
        if cancelled {
            let active = self.active.as_mut().expect("active request checked above");
            active.terminal_outcome = Some(ReleaseOutcome::Cancelled);
            self.status = Qwen35Aq4SessionStatus::Terminal;
            return Ok(SessionAdvance::CancellationObserved);
        }

        if self.status == Qwen35Aq4SessionStatus::Prefilling {
            return self.prepare_prefill_chunk();
        }

        let (token_id, position) = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "Qwen3.5 AQ4 prepare has no active request".to_string())?;
            match self.status {
                Qwen35Aq4SessionStatus::Decoding => {
                    let token_id = active.decode_input.ok_or_else(|| {
                        "Qwen3.5 AQ4 decode has no committed input token".to_string()
                    })?;
                    let position = active
                        .prompt_token_ids
                        .len()
                        .checked_add(active.generated_tokens)
                        .and_then(|value| value.checked_sub(1))
                        .ok_or_else(|| "Qwen3.5 AQ4 decode position overflows".to_string())?;
                    (token_id, position)
                }
                _ => unreachable!("status checked above"),
            }
        };
        let label = "Qwen3.5 AQ4 decode";
        let phase = ExecutionPhase::Decode;
        let operation_records = match self.model.dispatch_token(
            token_id,
            self.config.rotary_dim,
            self.config.rope_base,
            position,
            phase,
            self.config.sync_each_layer_for_timing,
            label,
        ) {
            Ok(records) => records,
            Err(error) => {
                let partial_records = self.model.take_failed_operation_executions();
                let failure = match (
                    self.active_operation_audit.as_mut(),
                    self.execution_contract.as_deref(),
                ) {
                    (Some(audit), Some(contract)) => audit
                        .observe_failed_step(phase, contract, &partial_records)
                        .unwrap_or((None, None)),
                    _ => (None, None),
                };
                self.last_terminal_operation_audit =
                    self.active_operation_audit.as_ref().map(|audit| {
                        audit.partial(
                            self.execution_contract.as_ref().map_or(0, Vec::len),
                            "execution_failed",
                            Some(phase),
                            failure.0,
                            failure.1,
                        )
                    });
                return self.fail(format!("{label} token dispatch failed: {error}"));
            }
        };
        if let (Some(contract), Some(audit)) = (
            self.execution_contract.as_deref(),
            self.active_operation_audit.as_mut(),
        ) {
            if let Err(error) = audit.observe(phase, contract, &operation_records) {
                return self.fail(format!("{label} operation execution audit failed: {error}"));
            }
        }
        self.prepare_token(label)
    }

    fn publish_prepared<F>(
        &mut self,
        prepared: Self::Prepared,
        publish: F,
    ) -> Result<PublishedAdvance, String>
    where
        F: FnOnce(usize) -> Result<(), String>,
    {
        if self.status != Qwen35Aq4SessionStatus::PreparedToken {
            return Err(format!(
                "Qwen3.5 AQ4 publish requires PreparedToken, got {:?}",
                self.status
            ));
        }
        if self.pending.as_ref() != Some(&prepared) {
            return Err("Qwen3.5 AQ4 publication handle does not match pending token".to_string());
        }
        let cancel = self
            .active
            .as_ref()
            .ok_or_else(|| "Qwen3.5 AQ4 publication has no active request".to_string())?
            .cancel
            .clone();
        let next_generated = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "Qwen3.5 AQ4 publication has no active request".to_string())?;
            if active.generated_tokens != prepared.generated_index {
                return Err(
                    "Qwen3.5 AQ4 prepared token index does not match active request".to_string(),
                );
            }
            active
                .generated_tokens
                .checked_add(1)
                .ok_or_else(|| "Qwen3.5 AQ4 generated token count overflows".to_string())?
        };
        let publication = cancel.publication_guard()?;
        if cancel.is_cancelled() {
            drop(publication);
            self.pending = None;
            let active = self.active.as_mut().expect("active request checked above");
            active.terminal_outcome = Some(ReleaseOutcome::Cancelled);
            self.status = Qwen35Aq4SessionStatus::Terminal;
            return Ok(PublishedAdvance::CancellationObserved);
        }
        if let Err(error) = publish(prepared.token_id) {
            drop(publication);
            self.pending = None;
            // A publisher failure does not poison resident model state. The caller must abort it.
            self.status = Qwen35Aq4SessionStatus::Terminal;
            return Err(format!(
                "Qwen3.5 AQ4 token publisher failed before commit: {error}"
            ));
        }
        // The callback cannot mutate the session, so every fallible commit precondition was
        // checked before publication. No error may be introduced after the public side effect.
        let active = self
            .active
            .as_mut()
            .expect("active request validated before publication");
        active.generated_tokens = next_generated;
        active.decode_input = Some(prepared.token_id);
        if let Some(reason) = prepared.terminal_reason {
            active.terminal_outcome = Some(match reason {
                FinishReason::Stop => ReleaseOutcome::Stop,
                FinishReason::Length => ReleaseOutcome::Length,
            });
            self.status = Qwen35Aq4SessionStatus::Terminal;
        } else {
            self.status = Qwen35Aq4SessionStatus::Decoding;
        }
        self.pending = None;
        drop(publication);
        Ok(PublishedAdvance::Token {
            token_id: prepared.token_id,
            generated_index: prepared.generated_index,
            cache_len: prepared.cache_len,
            terminal_reason: prepared.terminal_reason,
        })
    }

    fn finish_and_reset(&mut self) -> Result<ReleaseSummary, String> {
        if self.status != Qwen35Aq4SessionStatus::Terminal {
            return Err(format!(
                "Qwen3.5 AQ4 finish requires Terminal, got {:?}",
                self.status
            ));
        }
        let outcome = self
            .active
            .as_ref()
            .and_then(|active| active.terminal_outcome)
            .filter(|outcome| matches!(outcome, ReleaseOutcome::Stop | ReleaseOutcome::Length))
            .ok_or_else(|| "Qwen3.5 AQ4 finish has no completed outcome".to_string())?;
        self.reset_with_outcome(outcome)
    }

    fn abort_and_reset(&mut self) -> Result<ReleaseSummary, String> {
        if self.status == Qwen35Aq4SessionStatus::Ready {
            return Err(format!(
                "Qwen3.5 AQ4 abort requires an active reusable request, got {:?}",
                self.status
            ));
        }
        self.reset_with_outcome(ReleaseOutcome::Cancelled)
    }

    fn shutdown(&mut self) -> Result<(), String> {
        if let Err(error) = self.model.shutdown_synchronized() {
            self.status = Qwen35Aq4SessionStatus::Failed;
            return Err(format!("Qwen3.5 AQ4 session shutdown sync failed: {error}"));
        }
        Ok(())
    }

    fn terminal_operation_execution_audit(&self) -> Option<&OperationExecutionAudit> {
        self.last_terminal_operation_audit()
    }
}

fn validate_config<M: Qwen35Aq4SessionModel>(
    model: &M,
    config: &Qwen35Aq4SessionConfig,
) -> Result<(), String> {
    if model.context_length() == 0 || model.vocab_size() == 0 {
        return Err("Qwen3.5 AQ4 session model geometry must be nonzero".to_string());
    }
    if config.max_new_tokens == 0 || config.max_new_tokens > model.context_length() {
        return Err(format!(
            "Qwen3.5 AQ4 session max_new_tokens must be in 1..={}, got {}",
            model.context_length(),
            config.max_new_tokens
        ));
    }
    if config.eos_token_ids.is_empty() {
        return Err("Qwen3.5 AQ4 session requires at least one EOS token".to_string());
    }
    let mut eos = config.eos_token_ids.clone();
    eos.sort_unstable();
    eos.dedup();
    if eos.len() != config.eos_token_ids.len()
        || eos.iter().any(|token_id| *token_id >= model.vocab_size())
    {
        return Err(
            "Qwen3.5 AQ4 session EOS tokens must be unique and inside the vocabulary".to_string(),
        );
    }
    if config.rotary_dim == 0 || config.rotary_dim % 2 != 0 {
        return Err("Qwen3.5 AQ4 session rotary_dim must be a positive even number".to_string());
    }
    if !config.rope_base.is_finite() || config.rope_base <= 0.0 {
        return Err("Qwen3.5 AQ4 session rope_base must be positive and finite".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_api::SamplingParams;
    use std::collections::VecDeque;

    fn audited_contract() -> Vec<LayerExecutionContract> {
        (0..32)
            .map(|layer| {
                let pair = if layer % 4 == 3 {
                    [
                        "hip.fused-qk-norm-rope-paged-kv-write-f32.m1",
                        "hip.paged-decode-attention-sigmoid-gate-f32.m1-gqa",
                    ]
                } else {
                    [
                        "hip.linear-attention-qkv-prepare-f32.m1",
                        "hip.linear-attention-recurrent-f32.m1",
                    ]
                };
                [pair, pair, pair]
            })
            .collect()
    }

    fn native_hybrid_contract(execution_width: usize) -> Vec<LayerExecutionContract> {
        let mut contract = audited_contract();
        if execution_width > 1 {
            for (layer_index, phases) in contract.iter_mut().enumerate() {
                if layer_index % 4 == 3 {
                    continue;
                }
                for phase in [0, 1] {
                    phases[phase] = [
                        "hip.linear-attention-qkv-prepare-batch-f32.m2-m128",
                        "hip.linear-attention-recurrent-sequence-f32.m2-m128",
                    ];
                }
            }
        }
        contract
    }

    fn audited_records(
        contract: &[LayerExecutionContract],
        phase: ExecutionPhase,
    ) -> Vec<[OperationExecutionRecord; 2]> {
        contract
            .iter()
            .map(|layer| {
                std::array::from_fn(|record| OperationExecutionRecord {
                    implementation_id: layer[execution_phase_index(phase)][record],
                    phase,
                    status: OperationExecutionStatus::Succeeded,
                })
            })
            .collect()
    }

    fn audited_failed_records(
        phase: ExecutionPhase,
        failed_layer: usize,
        failed_operation: usize,
    ) -> Vec<[Option<OperationExecutionRecord>; 2]> {
        let contract = audited_contract();
        (0..=failed_layer)
            .map(|layer| {
                std::array::from_fn(|operation| {
                    if layer == failed_layer && operation > failed_operation {
                        return None;
                    }
                    Some(OperationExecutionRecord {
                        implementation_id: contract[layer][execution_phase_index(phase)][operation],
                        phase,
                        status: if layer == failed_layer && operation == failed_operation {
                            OperationExecutionStatus::Failed
                        } else {
                            OperationExecutionStatus::Succeeded
                        },
                    })
                })
            })
            .collect()
    }

    fn fallback_prefill_invocations(
        contract: &[LayerExecutionContract],
        execution_width: usize,
    ) -> Vec<Qwen35PrefillExecutionStep> {
        (0..execution_width)
            .flat_map(|offset| {
                let phase = if offset == 0 {
                    ExecutionPhase::ColdPrefill
                } else {
                    ExecutionPhase::CachedPrefixPrefill
                };
                audited_records(contract, phase)
                    .into_iter()
                    .enumerate()
                    .map(move |(layer_index, records)| Qwen35PrefillExecutionStep {
                        layer_index,
                        execution_width: 1,
                        phase,
                        records,
                    })
            })
            .collect()
    }

    #[test]
    fn fixed_operation_audit_covers_32_layer_cold_cached_and_decode_steps() {
        let contract = audited_contract();
        let mut audit = OperationAuditAccumulator::new();
        audit
            .observe_prefill_chunk(
                ExecutionPhase::ColdPrefill,
                3,
                &contract,
                &fallback_prefill_invocations(&contract, 3),
            )
            .unwrap();
        audit
            .observe(
                ExecutionPhase::Decode,
                &contract,
                &audited_records(&contract, ExecutionPhase::Decode),
            )
            .unwrap();
        audit.commit_prefill_chunk(3).unwrap();
        let finished = audit.finish(32, 1, 2, 1, "length").unwrap();
        assert_eq!(finished.schema_version, "ullm.backend_operation.request.v2");
        assert_eq!(finished.total_steps, 4);
        assert_eq!(finished.total_records, 256);
        assert_eq!(finished.physical_operation_invocations, 256);
        assert_eq!(finished.token_equivalent_operation_coverage, 256);
        assert_eq!(finished.prefill_chunks_executed, 1);
        assert_eq!(finished.prefill_tokens_executed, 3);
        assert_eq!(finished.prefill_tokens_committed, 3);
        assert_eq!(finished.prefill_width_histogram[3], 1);
        assert_eq!(finished.implementation_counts[0].count, 96);
        assert_eq!(finished.implementation_counts[1].count, 96);
        assert_eq!(finished.implementation_counts[2].count, 0);
        assert_eq!(finished.implementation_counts[3].count, 32);
        assert_eq!(finished.implementation_counts[4].count, 0);
        assert_eq!(finished.implementation_counts[5].count, 32);
        assert_eq!(finished.implementation_counts[6].count, 0);
        assert_eq!(finished.implementation_counts[7].count, 0);
        assert_ne!(finished.deterministic_digest_sha256, [0; 32]);
        assert!(finished.coverage_complete);

        let mut second_request = OperationAuditAccumulator::new();
        second_request
            .observe_prefill_chunk(
                ExecutionPhase::ColdPrefill,
                3,
                &contract,
                &fallback_prefill_invocations(&contract, 3),
            )
            .unwrap();
        second_request
            .observe(
                ExecutionPhase::Decode,
                &contract,
                &audited_records(&contract, ExecutionPhase::Decode),
            )
            .unwrap();
        second_request.commit_prefill_chunk(3).unwrap();
        assert_eq!(
            second_request
                .finish(32, 1, 2, 1, "length")
                .unwrap()
                .deterministic_digest_sha256,
            finished.deterministic_digest_sha256
        );
    }

    #[test]
    fn operation_audit_rejects_empty_missing_and_out_of_order_records() {
        let contract = audited_contract();
        let mut audit = OperationAuditAccumulator::new();
        assert!(
            audit
                .observe(ExecutionPhase::ColdPrefill, &contract, &[])
                .is_err()
        );
        let mut records = audited_records(&contract, ExecutionPhase::ColdPrefill);
        records[0].swap(0, 1);
        assert!(
            audit
                .observe(ExecutionPhase::ColdPrefill, &contract, &records)
                .is_err()
        );
        assert!(build_execution_contract(vec![Vec::new(); 32]).is_err());
    }

    #[test]
    fn failed_native_prefill_retains_physical_width_and_layer_location() {
        let contract = audited_contract();
        let mut audit = OperationAuditAccumulator::new();
        let failure = audit
            .observe_failed_prefill(
                &contract,
                &[
                    Qwen35FailedPrefillExecutionStep {
                        layer_index: 0,
                        execution_width: 128,
                        phase: ExecutionPhase::ColdPrefill,
                        records: [
                            Some(OperationExecutionRecord {
                                implementation_id:
                                    "hip.linear-attention-qkv-prepare-batch-f32.m2-m128",
                                phase: ExecutionPhase::ColdPrefill,
                                status: OperationExecutionStatus::Succeeded,
                            }),
                            Some(OperationExecutionRecord {
                                implementation_id:
                                    "hip.linear-attention-recurrent-sequence-f32.m2-m128",
                                phase: ExecutionPhase::ColdPrefill,
                                status: OperationExecutionStatus::Succeeded,
                            }),
                        ],
                    },
                    Qwen35FailedPrefillExecutionStep {
                        layer_index: 1,
                        execution_width: 128,
                        phase: ExecutionPhase::ColdPrefill,
                        records: [
                            Some(OperationExecutionRecord {
                                implementation_id:
                                    "hip.linear-attention-qkv-prepare-batch-f32.m2-m128",
                                phase: ExecutionPhase::ColdPrefill,
                                status: OperationExecutionStatus::Succeeded,
                            }),
                            Some(OperationExecutionRecord {
                                implementation_id:
                                    "hip.linear-attention-recurrent-sequence-f32.m2-m128",
                                phase: ExecutionPhase::ColdPrefill,
                                status: OperationExecutionStatus::Failed,
                            }),
                        ],
                    },
                ],
            )
            .unwrap();
        assert_eq!(failure, (Some(1), Some(1)));
        assert_eq!(audit.total_records, 3);
        assert_eq!(audit.token_equivalent_operation_coverage, 384);
        assert_eq!(audit.implementation_counts[6], 2);
        assert_eq!(audit.implementation_counts[7], 1);
    }

    #[derive(Default)]
    struct ScriptedModel {
        context: usize,
        vocab: usize,
        logits: VecDeque<Result<usize, String>>,
        dispatches: Vec<(usize, usize, usize, u32)>,
        dispatch_phases: Vec<ExecutionPhase>,
        resets: usize,
        fail_reset: bool,
        shutdowns: usize,
        fail_shutdown: bool,
        audited: bool,
        fail_dispatch_phase: Option<ExecutionPhase>,
        failed_operation: Option<(usize, usize)>,
        cancel_on_prefill_sync: Option<CancellationToken>,
        fail_prefill_sync: bool,
        native_hybrid_prefill: bool,
    }

    impl Qwen35Aq4SessionModel for ScriptedModel {
        fn context_length(&self) -> usize {
            self.context
        }

        fn vocab_size(&self) -> usize {
            self.vocab
        }

        fn dispatch_token(
            &mut self,
            token_id: usize,
            rotary_dim: usize,
            rope_base: f32,
            position: usize,
            phase: ExecutionPhase,
            _: bool,
            _: &str,
        ) -> Result<Vec<[OperationExecutionRecord; 2]>, String> {
            self.dispatches
                .push((token_id, position, rotary_dim, rope_base.to_bits()));
            self.dispatch_phases.push(phase);
            if self.fail_dispatch_phase == Some(phase) {
                return Err("scripted operation failure".into());
            }
            if self.audited {
                Ok(audited_records(&audited_contract(), phase))
            } else {
                Ok(Vec::new())
            }
        }

        fn dispatch_prefill_chunk(
            &mut self,
            token_ids: &[usize],
            rotary_dim: usize,
            rope_base: f32,
            absolute_start: usize,
            phase: ExecutionPhase,
            sync_each_layer_for_timing: bool,
            label: &str,
        ) -> Result<Vec<Qwen35PrefillExecutionStep>, String> {
            if !self.native_hybrid_prefill {
                let mut invocations = Vec::new();
                for (offset, token_id) in token_ids.iter().copied().enumerate() {
                    let position = absolute_start + offset;
                    let token_phase = if position == 0 {
                        ExecutionPhase::ColdPrefill
                    } else {
                        ExecutionPhase::CachedPrefixPrefill
                    };
                    let records = self.dispatch_token(
                        token_id,
                        rotary_dim,
                        rope_base,
                        position,
                        token_phase,
                        sync_each_layer_for_timing,
                        label,
                    )?;
                    invocations.extend(records.into_iter().enumerate().map(
                        |(layer_index, records)| Qwen35PrefillExecutionStep {
                            layer_index,
                            execution_width: 1,
                            phase: token_phase,
                            records,
                        },
                    ));
                }
                return Ok(invocations);
            }
            let contract = native_hybrid_contract(token_ids.len());
            let mut invocations = Vec::with_capacity(24 + 8 * token_ids.len());
            for layer_index in 0..contract.len() {
                if layer_index % 4 != 3 {
                    invocations.push(Qwen35PrefillExecutionStep {
                        layer_index,
                        execution_width: token_ids.len(),
                        phase,
                        records: audited_records(&contract, phase)[layer_index],
                    });
                    continue;
                }
                for offset in 0..token_ids.len() {
                    let token_phase = if absolute_start + offset == 0 {
                        ExecutionPhase::ColdPrefill
                    } else {
                        ExecutionPhase::CachedPrefixPrefill
                    };
                    invocations.push(Qwen35PrefillExecutionStep {
                        layer_index,
                        execution_width: 1,
                        phase: token_phase,
                        records: audited_records(&contract, token_phase)[layer_index],
                    });
                }
            }
            Ok(invocations)
        }

        fn top_token_from_last_layer(&mut self, _: &str) -> Result<usize, String> {
            self.logits
                .pop_front()
                .unwrap_or_else(|| Err("script exhausted".to_string()))
        }

        fn take_failed_operation_executions(
            &mut self,
        ) -> Vec<[Option<OperationExecutionRecord>; 2]> {
            match (self.fail_dispatch_phase, self.failed_operation) {
                (Some(phase), Some((layer, operation))) => {
                    audited_failed_records(phase, layer, operation)
                }
                _ => Vec::new(),
            }
        }

        fn synchronize_after_prefill_chunk(&mut self) -> Result<(), String> {
            if let Some(cancel) = self.cancel_on_prefill_sync.take() {
                cancel.cancel();
            }
            if self.fail_prefill_sync {
                Err("scripted prefill synchronization failure".to_string())
            } else {
                Ok(())
            }
        }

        fn reset_all_request_state_synchronized(&mut self) -> Result<(), String> {
            self.resets += 1;
            if self.fail_reset {
                Err("scripted reset failure".to_string())
            } else {
                Ok(())
            }
        }

        fn shutdown_synchronized(&mut self) -> Result<(), String> {
            self.shutdowns += 1;
            if self.fail_shutdown {
                Err("scripted shutdown failure".to_string())
            } else {
                Ok(())
            }
        }
    }

    fn model(tokens: &[usize]) -> ScriptedModel {
        ScriptedModel {
            context: 16,
            vocab: 32,
            logits: tokens.iter().copied().map(Ok).collect(),
            ..ScriptedModel::default()
        }
    }

    fn session(tokens: &[usize]) -> Qwen35Aq4InferenceSession<ScriptedModel> {
        Qwen35Aq4InferenceSession::from_model(
            model(tokens),
            Qwen35Aq4SessionConfig::greedy(8, vec![2]),
        )
        .unwrap()
    }

    fn audited_session(tokens: &[usize]) -> Qwen35Aq4InferenceSession<ScriptedModel> {
        let mut model = model(tokens);
        model.audited = true;
        let mut session = Qwen35Aq4InferenceSession::from_model(
            model,
            Qwen35Aq4SessionConfig::greedy(8, vec![2]),
        )
        .unwrap();
        session.execution_contract = Some(audited_contract());
        session
    }

    fn request(id: &str, prompt: &[usize], max_new_tokens: usize) -> InferenceRequest {
        InferenceRequest::new_with_eos(
            id,
            prompt.to_vec(),
            max_new_tokens,
            vec![2],
            SamplingParams::greedy_with_top_k(0, 1),
        )
    }

    fn prepared(advance: SessionAdvance<Qwen35Aq4PreparedToken>) -> Qwen35Aq4PreparedToken {
        match advance {
            SessionAdvance::Token { prepared, .. } => prepared,
            other => panic!("expected token, got {other:?}"),
        }
    }

    fn next_prepared(
        session: &mut Qwen35Aq4InferenceSession<ScriptedModel>,
    ) -> Qwen35Aq4PreparedToken {
        loop {
            match session.prepare_advance().unwrap() {
                SessionAdvance::PromptProgress { .. } => {}
                advance => return prepared(advance),
            }
        }
    }

    #[test]
    fn prompt_progresses_in_one_bounded_chunk_and_uses_explicit_rope_config() {
        let mut session = session(&[9]);
        session
            .start_request(request("r1", &[4, 5, 6], 2), CancellationToken::new())
            .unwrap();
        assert_eq!(
            session.prepare_advance().unwrap(),
            SessionAdvance::PromptProgress {
                prompt_tokens_processed: 3,
                cache_len: 3,
                execution_width: 3,
            }
        );
        let token = next_prepared(&mut session);
        assert_eq!(token.token_id, 9);
        assert_eq!(
            session.model().dispatches,
            vec![
                (4, 0, 64, QWEN35_AQ4_ROPE_BASE.to_bits()),
                (5, 1, 64, QWEN35_AQ4_ROPE_BASE.to_bits()),
                (6, 2, 64, QWEN35_AQ4_ROPE_BASE.to_bits()),
            ]
        );
        assert_eq!(
            session.model().dispatch_phases,
            vec![
                ExecutionPhase::ColdPrefill,
                ExecutionPhase::CachedPrefixPrefill,
                ExecutionPhase::CachedPrefixPrefill,
            ]
        );
    }

    #[test]
    fn prefill_chunk_widths_cover_boundaries_and_tail_without_partial_progress() {
        for (prompt_len, expected_widths) in [
            (1, vec![1]),
            (2, vec![2]),
            (3, vec![3]),
            (127, vec![127]),
            (128, vec![128]),
            (129, vec![128, 1]),
            (255, vec![128, 127]),
            (256, vec![128, 128]),
        ] {
            let mut scripted = model(&[2]);
            scripted.context = 512;
            let mut session = Qwen35Aq4InferenceSession::from_model(
                scripted,
                Qwen35Aq4SessionConfig::greedy(8, vec![2]),
            )
            .unwrap();
            session
                .start_request(
                    request("chunk-boundary", &vec![4; prompt_len], 1),
                    CancellationToken::new(),
                )
                .unwrap();
            let mut widths = Vec::new();
            loop {
                match session.prepare_advance().unwrap() {
                    SessionAdvance::PromptProgress {
                        execution_width, ..
                    } => widths.push(execution_width),
                    SessionAdvance::Token { .. } => break,
                    SessionAdvance::CancellationObserved => panic!("unexpected cancellation"),
                }
            }
            assert_eq!(widths, expected_widths, "prompt_len={prompt_len}");
            assert_eq!(
                session.model().dispatch_phases[0],
                ExecutionPhase::ColdPrefill
            );
            assert!(
                session.model().dispatch_phases[1..]
                    .iter()
                    .all(|phase| *phase == ExecutionPhase::CachedPrefixPrefill)
            );
        }
    }

    #[test]
    fn native_hybrid_prefill_audit_counts_physical_and_token_equivalent_coverage() {
        for execution_width in [1, 2, 128] {
            let mut scripted = model(&[2]);
            scripted.context = 256;
            scripted.audited = true;
            scripted.native_hybrid_prefill = true;
            let mut session = Qwen35Aq4InferenceSession::from_model(
                scripted,
                Qwen35Aq4SessionConfig::greedy(8, vec![2]),
            )
            .unwrap();
            session.execution_contract = Some(native_hybrid_contract(execution_width));
            session
                .start_request(
                    request("native-hybrid", &vec![4; execution_width], 1),
                    CancellationToken::new(),
                )
                .unwrap();
            assert_eq!(
                session.prepare_advance().unwrap(),
                SessionAdvance::PromptProgress {
                    prompt_tokens_processed: execution_width,
                    cache_len: execution_width,
                    execution_width,
                }
            );
            let token = next_prepared(&mut session);
            session.publish_prepared(token, |_| Ok(())).unwrap();
            session.finish_and_reset().unwrap();

            let audit = session.last_terminal_operation_audit().unwrap();
            assert_eq!(audit.cold_prefill_steps, 1);
            assert_eq!(
                audit.cached_prefix_prefill_steps,
                u64::try_from(execution_width - 1).unwrap()
            );
            assert_eq!(audit.prefill_chunks_executed, 1);
            assert_eq!(
                audit.physical_operation_invocations,
                u64::try_from(48 + 16 * execution_width).unwrap()
            );
            assert_eq!(
                audit.token_equivalent_operation_coverage,
                u64::try_from(64 * execution_width).unwrap()
            );
            let expected_m1_linear = u64::from(execution_width == 1) * 24;
            let expected_native_linear = u64::from(execution_width > 1) * 24;
            assert_eq!(audit.implementation_counts[0].count, expected_m1_linear);
            assert_eq!(audit.implementation_counts[1].count, expected_m1_linear);
            assert_eq!(
                audit.implementation_counts[3].count,
                u64::try_from(8 * execution_width).unwrap()
            );
            assert_eq!(
                audit.implementation_counts[5].count,
                u64::try_from(8 * execution_width).unwrap()
            );
            assert_eq!(audit.implementation_counts[6].count, expected_native_linear);
            assert_eq!(audit.implementation_counts[7].count, expected_native_linear);
            assert_eq!(
                audit
                    .implementation_counts
                    .iter()
                    .map(|entry| entry.count)
                    .sum::<u64>(),
                audit.physical_operation_invocations
            );
            assert_eq!(audit.prefill_width_histogram[execution_width], 1);
            assert!(audit.coverage_complete);
        }
    }

    #[test]
    fn eos_commits_stop_and_reset_summary() {
        let mut session = session(&[2]);
        session
            .start_request(request("eos", &[4], 4), CancellationToken::new())
            .unwrap();
        let token = next_prepared(&mut session);
        assert_eq!(token.terminal_reason, Some(FinishReason::Stop));
        let published = session.publish_prepared(token, |_| Ok(())).unwrap();
        assert!(matches!(published, PublishedAdvance::Token { .. }));
        assert_eq!(
            session.finish_and_reset().unwrap(),
            ReleaseSummary {
                request_id: "eos".to_string(),
                outcome: ReleaseOutcome::Stop,
                prompt_tokens: 1,
                generated_tokens: 1,
                reset_complete: true,
            }
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Ready);
    }

    #[test]
    fn max_new_tokens_commits_length() {
        let mut session = session(&[7, 8]);
        session
            .start_request(request("length", &[4], 2), CancellationToken::new())
            .unwrap();
        let first = next_prepared(&mut session);
        session.publish_prepared(first, |_| Ok(())).unwrap();
        let second = next_prepared(&mut session);
        assert_eq!(second.terminal_reason, Some(FinishReason::Length));
        session.publish_prepared(second, |_| Ok(())).unwrap();
        assert_eq!(
            session.finish_and_reset().unwrap().outcome,
            ReleaseOutcome::Length
        );
        assert_eq!(session.model().dispatches[1].0, 7);
        assert_eq!(
            session.model().dispatch_phases,
            vec![ExecutionPhase::ColdPrefill, ExecutionPhase::Decode]
        );
    }

    #[test]
    fn same_resident_model_serves_two_requests_with_reset_between_them() {
        let mut session = session(&[2, 2]);
        for id in ["one", "two"] {
            session
                .start_request(request(id, &[4], 1), CancellationToken::new())
                .unwrap();
            let token = next_prepared(&mut session);
            session.publish_prepared(token, |_| Ok(())).unwrap();
            assert_eq!(session.finish_and_reset().unwrap().request_id, id);
        }
        assert_eq!(session.model().resets, 2);
        assert_eq!(session.model().dispatches.len(), 2);
        assert_eq!(
            session.model().dispatch_phases,
            vec![ExecutionPhase::ColdPrefill, ExecutionPhase::ColdPrefill]
        );
    }

    #[test]
    fn two_request_reset_restarts_fixed_operation_audit_without_cumulative_growth() {
        let mut session = audited_session(&[7, 8, 7, 8]);
        let mut digest = None;
        for id in ["audit-one", "audit-two"] {
            session
                .start_request(request(id, &[4, 5, 6], 2), CancellationToken::new())
                .unwrap();
            let first = next_prepared(&mut session);
            session.publish_prepared(first, |_| Ok(())).unwrap();
            let second = next_prepared(&mut session);
            session.publish_prepared(second, |_| Ok(())).unwrap();
            session.finish_and_reset().unwrap();
            let audit = session.last_terminal_operation_audit().unwrap();
            assert_eq!(
                (audit.cold_prefill_steps, audit.cached_prefix_prefill_steps),
                (1, 2)
            );
            assert_eq!(audit.decode_steps, 1);
            assert_eq!(audit.total_records, 256);
            if let Some(previous) = digest {
                assert_eq!(audit.deterministic_digest_sha256, previous);
            }
            digest = Some(audit.deterministic_digest_sha256);
        }
        assert_eq!(session.model().resets, 2);
    }

    #[test]
    fn cancellation_after_prepare_skips_callback_and_does_not_commit() {
        let mut session = session(&[7]);
        let cancel = CancellationToken::new();
        session
            .start_request(request("cancel", &[4], 2), cancel.clone())
            .unwrap();
        let token = next_prepared(&mut session);
        cancel.cancel();
        let mut called = false;
        assert_eq!(
            session
                .publish_prepared(token, |_| {
                    called = true;
                    Ok(())
                })
                .unwrap(),
            PublishedAdvance::CancellationObserved
        );
        assert!(!called);
        assert_eq!(session.abort_and_reset().unwrap().generated_tokens, 0);
    }

    #[test]
    fn cancellation_after_prompt_and_before_any_operation_publish_partial_audits() {
        let mut after_prompt = audited_session(&[7]);
        let cancel = CancellationToken::new();
        after_prompt
            .start_request(request("partial-cancel", &[4, 5], 2), cancel.clone())
            .unwrap();
        assert!(matches!(
            after_prompt.prepare_advance().unwrap(),
            SessionAdvance::PromptProgress { .. }
        ));
        cancel.cancel();
        assert_eq!(
            after_prompt.prepare_advance().unwrap(),
            SessionAdvance::CancellationObserved
        );
        after_prompt.abort_and_reset().unwrap();
        let audit = after_prompt.last_terminal_operation_audit().unwrap();
        assert_eq!(audit.outcome, "cancelled");
        assert_eq!(audit.total_steps, 2);
        assert_eq!(audit.total_records, 128);
        assert!(!audit.coverage_complete);

        let mut before_operation = audited_session(&[]);
        let cancel = CancellationToken::new();
        before_operation
            .start_request(request("zero-cancel", &[4], 1), cancel.clone())
            .unwrap();
        cancel.cancel();
        assert_eq!(
            before_operation.prepare_advance().unwrap(),
            SessionAdvance::CancellationObserved
        );
        before_operation.abort_and_reset().unwrap();
        assert_eq!(
            before_operation
                .last_terminal_operation_audit()
                .unwrap()
                .total_records,
            0
        );
    }

    #[test]
    fn cancellation_observed_after_chunk_sync_does_not_commit_prompt_progress() {
        let mut session = audited_session(&[7]);
        let cancel = CancellationToken::new();
        session.model.native_hybrid_prefill = true;
        session.execution_contract = Some(native_hybrid_contract(2));
        session.model.cancel_on_prefill_sync = Some(cancel.clone());
        session
            .start_request(request("sync-cancel", &[4, 5], 2), cancel)
            .unwrap();

        assert_eq!(
            session.prepare_advance().unwrap(),
            SessionAdvance::CancellationObserved
        );
        assert_eq!(session.active.as_ref().unwrap().prompt_tokens_processed, 0);
        session.abort_and_reset().unwrap();
        assert_eq!(session.model.resets, 1);
        let audit = session.last_terminal_operation_audit().unwrap();
        assert_eq!(audit.prefill_chunks_executed, 1);
        assert_eq!(audit.prefill_tokens_executed, 2);
        assert_eq!(audit.prefill_tokens_committed, 0);
        assert_eq!(audit.prefill_width_histogram[2], 1);
        assert_eq!(audit.physical_operation_invocations, 48 + 16 * 2);
        assert_eq!(audit.token_equivalent_operation_coverage, 128);
        assert!(!audit.coverage_complete);
    }

    #[test]
    fn native_prefill_sync_failure_retains_audit_before_and_after_reset() {
        let execution_width = 2;
        let mut session = audited_session(&[]);
        session.model.context = 256;
        session.model.native_hybrid_prefill = true;
        session.model.fail_prefill_sync = true;
        session.execution_contract = Some(native_hybrid_contract(execution_width));
        session
            .start_request(
                request("sync-fail", &vec![4; execution_width], 1),
                CancellationToken::new(),
            )
            .unwrap();

        assert!(
            session
                .prepare_advance()
                .unwrap_err()
                .contains("prefill chunk synchronization failed")
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Failed);
        assert_eq!(
            session
                .active
                .as_ref()
                .expect("failed sync keeps the active request for cleanup")
                .prompt_tokens_processed,
            0
        );
        let audit = session.last_terminal_operation_audit().unwrap();
        assert_eq!(audit.outcome, "synchronization_failed");
        assert_eq!(
            audit.physical_operation_invocations,
            u64::try_from(48 + 16 * execution_width).unwrap()
        );
        assert_eq!(
            audit.token_equivalent_operation_coverage,
            u64::try_from(64 * execution_width).unwrap()
        );
        assert_eq!(audit.prefill_chunks_executed, 1);
        assert_eq!(audit.prefill_tokens_executed, execution_width as u64);
        assert_eq!(audit.prefill_tokens_committed, 0);
        assert_eq!(audit.prefill_width_histogram[execution_width], 1);
        assert!(!audit.coverage_complete);

        let summary = session.abort_and_reset().unwrap();
        assert_eq!(summary.outcome, ReleaseOutcome::Cancelled);
        assert_eq!(session.model.resets, 1);
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Ready);
        assert_eq!(
            session.last_terminal_operation_audit().unwrap().outcome,
            "synchronization_failed"
        );
    }

    #[test]
    fn execution_and_reset_failures_retain_partial_terminal_audits() {
        let mut execution = audited_session(&[]);
        execution.model.context = 256;
        execution.model.failed_operation = Some((3, 0));
        execution
            .start_request(
                request("op-fail", &vec![4; 129], 1),
                CancellationToken::new(),
            )
            .unwrap();
        assert!(matches!(
            execution.prepare_advance().unwrap(),
            SessionAdvance::PromptProgress { .. }
        ));
        execution.model.fail_dispatch_phase = Some(ExecutionPhase::CachedPrefixPrefill);
        assert!(
            execution
                .prepare_advance()
                .unwrap_err()
                .contains("dispatch failed")
        );
        let audit = execution.last_terminal_operation_audit().unwrap();
        assert_eq!(audit.outcome, "execution_failed");
        assert_eq!(audit.failed_phase, Some("cached_prefix_prefill"));
        assert_eq!(audit.failed_layer, Some(3));
        assert_eq!(audit.failed_operation, Some(0));
        assert_eq!(audit.total_records, 128 * 64 + 6);

        let mut reset = audited_session(&[2]);
        reset.model.fail_reset = true;
        reset
            .start_request(
                request("reset-fail-audit", &[4], 1),
                CancellationToken::new(),
            )
            .unwrap();
        let token = next_prepared(&mut reset);
        reset.publish_prepared(token, |_| Ok(())).unwrap();
        assert!(reset.finish_and_reset().is_err());
        let audit = reset.last_terminal_operation_audit().unwrap();
        assert_eq!(audit.outcome, "reset_failed");
        assert!(!audit.coverage_complete);
    }

    #[test]
    fn publisher_failure_does_not_commit_and_can_be_aborted() {
        let mut session = session(&[7]);
        session
            .start_request(request("publish", &[4], 2), CancellationToken::new())
            .unwrap();
        let token = next_prepared(&mut session);
        let error = session
            .publish_prepared(token, |_| Err("closed".to_string()))
            .unwrap_err();
        assert!(error.contains("before commit"));
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Terminal);
        assert_eq!(session.abort_and_reset().unwrap().generated_tokens, 0);
    }

    #[test]
    fn prepared_handle_cannot_be_committed_twice() {
        let mut session = session(&[7, 2]);
        session
            .start_request(request("twice", &[4], 2), CancellationToken::new())
            .unwrap();
        let token = next_prepared(&mut session);
        session.publish_prepared(token.clone(), |_| Ok(())).unwrap();
        let error = session.publish_prepared(token, |_| Ok(())).unwrap_err();
        assert!(error.contains("requires PreparedToken"));
    }

    #[test]
    fn reset_failure_poisons_session_and_rejects_reuse() {
        let mut session = session(&[2]);
        session.model.fail_reset = true;
        session
            .start_request(request("poison", &[4], 1), CancellationToken::new())
            .unwrap();
        let token = next_prepared(&mut session);
        session.publish_prepared(token, |_| Ok(())).unwrap();
        assert!(
            session
                .finish_and_reset()
                .unwrap_err()
                .contains("reset failed")
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Failed);
        assert!(
            session
                .start_request(request("reuse", &[4], 1), CancellationToken::new())
                .unwrap_err()
                .contains("got Failed")
        );
    }

    #[test]
    fn start_rejects_non_greedy_and_context_overflow() {
        let mut session = session(&[2]);
        let mut non_greedy = request("sample", &[4], 1);
        non_greedy.sampling.temperature = 0.5;
        assert!(
            session
                .start_request(non_greedy, CancellationToken::new())
                .is_err()
        );
        assert!(
            session
                .start_request(request("large", &[4; 15], 2), CancellationToken::new())
                .unwrap_err()
                .contains("exceeds context")
        );
    }

    #[test]
    fn start_rejects_prompt_vocabulary_and_eos_contract_mismatches() {
        let mut session = session(&[2]);
        assert!(
            session
                .start_request(request("vocab", &[32], 1), CancellationToken::new())
                .unwrap_err()
                .contains("exceeds vocabulary")
        );
        let wrong_eos = InferenceRequest::new_with_eos(
            "eos-contract",
            vec![4],
            1,
            vec![3],
            SamplingParams::greedy_with_top_k(0, 1),
        );
        assert!(
            session
                .start_request(wrong_eos, CancellationToken::new())
                .unwrap_err()
                .contains("eos_token_ids must be")
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Ready);
    }

    #[test]
    fn prepare_counter_failure_poisons_session_before_proposal() {
        let mut session = session(&[7]);
        session.next_nonce = u64::MAX;
        session
            .start_request(request("overflow", &[4], 2), CancellationToken::new())
            .unwrap();
        assert!(matches!(
            session.prepare_advance().unwrap(),
            SessionAdvance::PromptProgress {
                prompt_tokens_processed: 1,
                ..
            }
        ));
        assert!(
            session
                .prepare_advance()
                .unwrap_err()
                .contains("nonce overflows")
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Failed);
    }

    #[test]
    fn graceful_shutdown_synchronizes_model_and_reports_failure() {
        let mut session = session(&[]);
        session.shutdown().unwrap();
        assert_eq!(session.model().shutdowns, 1);
        session.model.fail_shutdown = true;
        assert!(
            session
                .shutdown()
                .unwrap_err()
                .contains("shutdown sync failed")
        );
        assert_eq!(session.model().shutdowns, 2);
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Failed);
    }
}
