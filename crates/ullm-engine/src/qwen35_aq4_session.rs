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
    CancellationToken, FinishReason, InferenceRequest, ReasoningUsage, ReleaseOutcome,
    ReleaseSummary,
};
use crate::qwen35_aq4_model_runtime::{Qwen35Aq4ModelLoadConfig, Qwen35Aq4ModelRuntime};
use crate::reasoning::{ReasoningPhase, ReasoningState};
use crate::worker_driver::{InferenceSession, PublishedAdvance, SessionAdvance};
use sha2::{Digest, Sha256};

pub const QWEN35_AQ4_ROTARY_DIM: usize = 64;
pub const QWEN35_AQ4_ROPE_BASE: f32 = 10_000_000.0;
pub const QWEN35_AQ4_MAX_PREFILL_CHUNK: usize = 128;
/// Supported requested prefill widths for AQ4 M dispatches.
///
/// The grid is intentionally finite: every requested value is either represented by a
/// production implementation or rejected before a request can execute.
pub const QWEN35_AQ4_PREFILL_CHUNK_GRID: &[usize] = &[1, 8, 16, 32, 64, 128];
pub const QWEN35_AQ4_SUPPORTED_PREFILL_CHUNK_TOKENS: &[usize] = QWEN35_AQ4_PREFILL_CHUNK_GRID;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35Aq4TransactionCounts {
    pub prepare: u64,
    pub commit: u64,
    pub discard: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35Aq4ResetCounts {
    pub attempted: u64,
    pub complete: u64,
    pub failed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35Aq4LifecycleCounts {
    pub prepare: u64,
    pub commit: u64,
    pub discard: u64,
    pub error: u64,
    pub cancel: u64,
    pub prefill: Qwen35Aq4TransactionCounts,
    pub publication: Qwen35Aq4TransactionCounts,
    pub reset: Qwen35Aq4ResetCounts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35Aq4PhaseBatchCounts {
    pub cold_prefill: u64,
    pub cached_prefix_prefill: u64,
    pub decode: u64,
}

/// Privacy-safe, request-terminal execution summary.
///
/// This deliberately contains only dimensions and counters. It has no request identifier,
/// token ids, prompt text, or model output. `operation_audit` remains the existing detailed
/// registry audit and is retained as an optional nested value when the registry contract is
/// active.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Qwen35Aq4RequestExecutionAudit {
    pub schema_version: &'static str,
    pub requested_m: usize,
    pub resolved_m: Option<usize>,
    pub actual_token_batch_width: Option<usize>,
    pub actual_request_batch_width: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_batch_counts: Option<Qwen35Aq4PhaseBatchCounts>,
    pub internal_batch_count: Option<u64>,
    pub lifecycle: Qwen35Aq4LifecycleCounts,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_audit: Option<OperationExecutionAudit>,
}

impl Qwen35Aq4RequestExecutionAudit {
    pub fn requested_prefill_chunk_tokens(&self) -> usize {
        self.requested_m
    }

    pub fn resolved_prefill_chunk_tokens(&self) -> Option<usize> {
        self.resolved_m
    }
}

#[derive(Debug, Clone, Copy)]
struct Qwen35Aq4LifecycleObserver {
    requested_m: usize,
    resolved_m: Option<usize>,
    actual_token_batch_width: Option<usize>,
    actual_request_batch_width: Option<usize>,
    phase_batch_counts: Option<Qwen35Aq4PhaseBatchCounts>,
    internal_batch_count: Option<u64>,
    lifecycle: Qwen35Aq4LifecycleCounts,
    prefill_open: bool,
    publication_open: bool,
    cancel_observed: bool,
}

impl Qwen35Aq4LifecycleObserver {
    fn new(requested_m: usize) -> Self {
        Self {
            requested_m,
            resolved_m: None,
            actual_token_batch_width: None,
            actual_request_batch_width: None,
            phase_batch_counts: None,
            internal_batch_count: None,
            lifecycle: Qwen35Aq4LifecycleCounts {
                prepare: 0,
                commit: 0,
                discard: 0,
                error: 0,
                cancel: 0,
                prefill: Qwen35Aq4TransactionCounts {
                    prepare: 0,
                    commit: 0,
                    discard: 0,
                },
                publication: Qwen35Aq4TransactionCounts {
                    prepare: 0,
                    commit: 0,
                    discard: 0,
                },
                reset: Qwen35Aq4ResetCounts {
                    attempted: 0,
                    complete: 0,
                    failed: 0,
                },
            },
            prefill_open: false,
            publication_open: false,
            cancel_observed: false,
        }
    }

    fn increment(counter: &mut u64, label: &str) -> Result<(), String> {
        *counter = counter
            .checked_add(1)
            .ok_or_else(|| format!("Qwen3.5 AQ4 lifecycle {label} count overflows"))?;
        Ok(())
    }

    fn prepare_prefill(&mut self) -> Result<(), String> {
        if self.prefill_open {
            return Err("Qwen3.5 AQ4 lifecycle prefill transaction is already open".into());
        }
        Self::increment(&mut self.lifecycle.prepare, "prepare")?;
        Self::increment(&mut self.lifecycle.prefill.prepare, "prefill prepare")?;
        self.prefill_open = true;
        Ok(())
    }

    fn commit_prefill(&mut self) -> Result<(), String> {
        if !self.prefill_open {
            return Err("Qwen3.5 AQ4 lifecycle prefill commit has no prepare".into());
        }
        Self::increment(&mut self.lifecycle.commit, "commit")?;
        Self::increment(&mut self.lifecycle.prefill.commit, "prefill commit")?;
        self.prefill_open = false;
        Ok(())
    }

    fn discard_prefill(&mut self) -> Result<(), String> {
        if !self.prefill_open {
            return Err("Qwen3.5 AQ4 lifecycle prefill discard has no prepare".into());
        }
        Self::increment(&mut self.lifecycle.discard, "discard")?;
        Self::increment(&mut self.lifecycle.prefill.discard, "prefill discard")?;
        self.prefill_open = false;
        Ok(())
    }

    fn prepare_publication(&mut self) -> Result<(), String> {
        if self.publication_open {
            return Err("Qwen3.5 AQ4 lifecycle publication transaction is already open".into());
        }
        Self::increment(&mut self.lifecycle.prepare, "prepare")?;
        Self::increment(
            &mut self.lifecycle.publication.prepare,
            "publication prepare",
        )?;
        self.publication_open = true;
        Ok(())
    }

    fn commit_publication(&mut self) -> Result<(), String> {
        if !self.publication_open {
            return Err("Qwen3.5 AQ4 lifecycle publication commit has no prepare".into());
        }
        Self::increment(&mut self.lifecycle.commit, "commit")?;
        Self::increment(&mut self.lifecycle.publication.commit, "publication commit")?;
        self.publication_open = false;
        Ok(())
    }

    fn discard_publication(&mut self) -> Result<(), String> {
        if !self.publication_open {
            return Err("Qwen3.5 AQ4 lifecycle publication discard has no prepare".into());
        }
        Self::increment(&mut self.lifecycle.discard, "discard")?;
        Self::increment(
            &mut self.lifecycle.publication.discard,
            "publication discard",
        )?;
        self.publication_open = false;
        Ok(())
    }

    fn observe_prefill_execution(
        &mut self,
        observation: ObservedPrefillExecution,
    ) -> Result<(), String> {
        self.resolved_m = Some(
            self.resolved_m
                .unwrap_or(0)
                .max(observation.actual_token_batch_width),
        );
        self.actual_token_batch_width = Some(
            self.actual_token_batch_width
                .unwrap_or(0)
                .max(observation.actual_token_batch_width),
        );
        self.actual_request_batch_width = Some(1);
        let counts = self
            .phase_batch_counts
            .get_or_insert(Qwen35Aq4PhaseBatchCounts {
                cold_prefill: 0,
                cached_prefix_prefill: 0,
                decode: 0,
            });
        counts.cold_prefill = counts
            .cold_prefill
            .checked_add(observation.phase_batch_counts.cold_prefill)
            .ok_or_else(|| "Qwen3.5 AQ4 cold-prefill batch count overflows".to_string())?;
        counts.cached_prefix_prefill = counts
            .cached_prefix_prefill
            .checked_add(observation.phase_batch_counts.cached_prefix_prefill)
            .ok_or_else(|| "Qwen3.5 AQ4 cached-prefix-prefill batch count overflows".to_string())?;
        let batches = self.internal_batch_count.get_or_insert(0);
        *batches = batches
            .checked_add(observation.internal_batch_count)
            .ok_or_else(|| "Qwen3.5 AQ4 internal batch count overflows".to_string())?;
        Ok(())
    }

    fn observe_decode_execution(&mut self) -> Result<(), String> {
        self.actual_token_batch_width = Some(self.actual_token_batch_width.unwrap_or(0).max(1));
        self.actual_request_batch_width = Some(1);
        let counts = self
            .phase_batch_counts
            .get_or_insert(Qwen35Aq4PhaseBatchCounts {
                cold_prefill: 0,
                cached_prefix_prefill: 0,
                decode: 0,
            });
        counts.decode = counts
            .decode
            .checked_add(1)
            .ok_or_else(|| "Qwen3.5 AQ4 decode batch count overflows".to_string())?;
        let batches = self.internal_batch_count.get_or_insert(0);
        *batches = batches
            .checked_add(1)
            .ok_or_else(|| "Qwen3.5 AQ4 internal batch count overflows".to_string())?;
        Ok(())
    }

    fn observe_error(&mut self) -> Result<(), String> {
        Self::increment(&mut self.lifecycle.error, "error")
    }

    fn observe_cancel(&mut self) -> Result<(), String> {
        if !self.cancel_observed {
            Self::increment(&mut self.lifecycle.cancel, "cancel")?;
            self.cancel_observed = true;
        }
        Ok(())
    }

    fn begin_reset(&mut self) -> Result<(), String> {
        if self.prefill_open || self.publication_open {
            return Err("Qwen3.5 AQ4 reset cannot begin with an open transaction".into());
        }
        Self::increment(&mut self.lifecycle.reset.attempted, "reset attempted")
    }

    fn complete_reset(&mut self) -> Result<(), String> {
        Self::increment(&mut self.lifecycle.reset.complete, "reset complete")
    }

    fn fail_reset(&mut self) -> Result<(), String> {
        Self::increment(&mut self.lifecycle.reset.failed, "reset failed")
    }

    fn snapshot(
        &self,
        operation_audit: Option<OperationExecutionAudit>,
    ) -> Result<Qwen35Aq4RequestExecutionAudit, String> {
        if self.prefill_open || self.publication_open {
            return Err("Qwen3.5 AQ4 terminal lifecycle retains an open transaction".into());
        }
        let closed = self
            .lifecycle
            .commit
            .checked_add(self.lifecycle.discard)
            .ok_or_else(|| "Qwen3.5 AQ4 lifecycle closed count overflows".to_string())?;
        if self.lifecycle.prepare != closed
            || self.lifecycle.prefill.prepare
                != self.lifecycle.prefill.commit + self.lifecycle.prefill.discard
            || self.lifecycle.publication.prepare
                != self.lifecycle.publication.commit + self.lifecycle.publication.discard
            || self.lifecycle.reset.complete + self.lifecycle.reset.failed
                > self.lifecycle.reset.attempted
        {
            return Err("Qwen3.5 AQ4 terminal lifecycle counters do not reconcile".into());
        }
        Ok(Qwen35Aq4RequestExecutionAudit {
            schema_version: "ullm.qwen35_aq4.request_execution.v1",
            requested_m: self.requested_m,
            resolved_m: self.resolved_m,
            actual_token_batch_width: self.actual_token_batch_width,
            actual_request_batch_width: self.actual_request_batch_width,
            phase_batch_counts: self.phase_batch_counts,
            internal_batch_count: self.internal_batch_count,
            lifecycle: self.lifecycle,
            operation_audit,
        })
    }
}

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
    pub reasoning_dialect: Option<crate::reasoning::ReasoningDialect>,
    /// Requested AQ4 prefill execution width. Only `QWEN35_AQ4_PREFILL_CHUNK_GRID` values
    /// are accepted; the default preserves the historic width-128 path.
    pub prefill_chunk_tokens: usize,
}

fn validate_prefill_chunk_tokens(requested_m: usize) -> Result<(), String> {
    if QWEN35_AQ4_PREFILL_CHUNK_GRID.contains(&requested_m) {
        Ok(())
    } else {
        Err(format!(
            "Qwen3.5 AQ4 prefill_chunk_tokens must be one of {:?}, got {requested_m}",
            QWEN35_AQ4_PREFILL_CHUNK_GRID
        ))
    }
}

impl Qwen35Aq4SessionConfig {
    pub fn greedy(max_new_tokens: usize, eos_token_ids: Vec<usize>) -> Self {
        Self {
            max_new_tokens,
            eos_token_ids,
            rotary_dim: QWEN35_AQ4_ROTARY_DIM,
            rope_base: QWEN35_AQ4_ROPE_BASE,
            sync_each_layer_for_timing: false,
            reasoning_dialect: None,
            prefill_chunk_tokens: QWEN35_AQ4_MAX_PREFILL_CHUNK,
        }
    }

    pub fn with_prefill_chunk_tokens(mut self, requested_m: usize) -> Result<Self, String> {
        validate_prefill_chunk_tokens(requested_m)?;
        self.prefill_chunk_tokens = requested_m;
        Ok(self)
    }

    pub fn prefill_chunk_tokens(&self) -> usize {
        self.prefill_chunk_tokens
    }

    pub fn requested_m(&self) -> usize {
        self.prefill_chunk_tokens
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
    reasoning_tokens_before: usize,
    forced_end_tokens_before: usize,
    nonce: u64,
}

/// Compares one runtime implementation with the canonical load-time contract entry.
///
/// Split paged-decode readers are typed alternates of their matching single-reader family. No
/// writer, chunk reader, linear-attention operation, unknown id, or plain/gated cross-family
/// substitution is accepted.
fn operation_implementation_matches_contract(expected: &str, actual: &str) -> bool {
    if expected == actual {
        return EXECUTION_IMPLEMENTATIONS
            .iter()
            .any(|(_, implementation_id)| *implementation_id == expected);
    }
    match expected {
        "hip.paged-decode-attention-f32.m1-gqa" => matches!(
            actual,
            "hip.paged-decode-attention-split-f32.tile128"
                | "hip.paged-decode-attention-split-f32.tile256"
        ),
        "hip.paged-decode-attention-sigmoid-gate-f32.m1-gqa" => matches!(
            actual,
            "hip.paged-decode-attention-split-sigmoid-gate-f32.tile128"
                | "hip.paged-decode-attention-split-sigmoid-gate-f32.tile256"
        ),
        _ => false,
    }
}

const EXECUTION_IMPLEMENTATIONS: [(&str, &str); 14] = [
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
    ("paged_kv_write", "hip.paged-kv-write-chunk-f32.m2-m128"),
    (
        "paged_causal_gqa_read",
        "hip.paged-causal-gqa-chunk-sigmoid-gate-f32.m2-m128",
    ),
    (
        "paged_causal_gqa_read",
        "hip.paged-decode-attention-split-f32.tile128",
    ),
    (
        "paged_causal_gqa_read",
        "hip.paged-decode-attention-split-f32.tile256",
    ),
    (
        "paged_causal_gqa_read",
        "hip.paged-decode-attention-split-sigmoid-gate-f32.tile128",
    ),
    (
        "paged_causal_gqa_read",
        "hip.paged-decode-attention-split-sigmoid-gate-f32.tile256",
    ),
];

type LayerExecutionContract = [[&'static str; 2]; 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ObservedPrefillExecution {
    actual_token_batch_width: usize,
    phase_batch_counts: Qwen35Aq4PhaseBatchCounts,
    internal_batch_count: u64,
}

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
    implementation_counts: [u64; 14],
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
            implementation_counts: [0; 14],
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
                    || !operation_implementation_matches_contract(
                        expected,
                        record.implementation_id,
                    )
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
    ) -> Result<ObservedPrefillExecution, String> {
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
        let mut layer_batches = vec![Vec::new(); contract.len()];
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
            layer_batches[invocation.layer_index]
                .push((invocation.phase, invocation.execution_width));
            let phase_index = execution_phase_index(invocation.phase);
            for (operation_index, record) in invocation.records.iter().enumerate() {
                let expected = prefill_expected_implementation(
                    contract[invocation.layer_index][phase_index][operation_index],
                    invocation.execution_width,
                );
                if record.phase != invocation.phase
                    || record.status != OperationExecutionStatus::Succeeded
                    || !operation_implementation_matches_contract(
                        expected,
                        record.implementation_id,
                    )
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
        let physical_batches = layer_batches
            .first()
            .ok_or_else(|| "prefill execution has no layer batch evidence".to_string())?;
        if physical_batches.is_empty() {
            return Err("prefill execution has no physical batch evidence".into());
        }
        if let Some((layer_index, _)) = layer_batches
            .iter()
            .enumerate()
            .find(|(_, batches)| batches.as_slice() != physical_batches.as_slice())
        {
            return Err(format!(
                "prefill physical batch sequence differs at layer={layer_index}"
            ));
        }
        match phase {
            ExecutionPhase::ColdPrefill => {
                if physical_batches[0].0 != ExecutionPhase::ColdPrefill
                    || physical_batches[1..]
                        .iter()
                        .any(|(batch_phase, _)| *batch_phase != ExecutionPhase::CachedPrefixPrefill)
                {
                    return Err("cold prefill physical phase sequence is inconsistent".into());
                }
            }
            ExecutionPhase::CachedPrefixPrefill => {
                if physical_batches
                    .iter()
                    .any(|(batch_phase, _)| *batch_phase != ExecutionPhase::CachedPrefixPrefill)
                {
                    return Err(
                        "cached-prefix prefill physical phase sequence is inconsistent".into(),
                    );
                }
            }
            ExecutionPhase::Decode => unreachable!("decode rejected above"),
        }
        let actual_token_batch_width = physical_batches
            .iter()
            .map(|(_, width)| *width)
            .max()
            .ok_or_else(|| "prefill physical width evidence is empty".to_string())?;
        let phase_batch_counts = Qwen35Aq4PhaseBatchCounts {
            cold_prefill: u64::try_from(
                physical_batches
                    .iter()
                    .filter(|(batch_phase, _)| *batch_phase == ExecutionPhase::ColdPrefill)
                    .count(),
            )
            .map_err(|_| "cold-prefill physical batch count does not fit u64".to_string())?,
            cached_prefix_prefill: u64::try_from(
                physical_batches
                    .iter()
                    .filter(|(batch_phase, _)| *batch_phase == ExecutionPhase::CachedPrefixPrefill)
                    .count(),
            )
            .map_err(|_| {
                "cached-prefix-prefill physical batch count does not fit u64".to_string()
            })?,
            decode: 0,
        };
        let internal_batch_count = u64::try_from(physical_batches.len())
            .map_err(|_| "prefill physical batch count does not fit u64".to_string())?;
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
        Ok(ObservedPrefillExecution {
            actual_token_batch_width,
            phase_batch_counts,
            internal_batch_count,
        })
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
            failed_execution_width: None,
            failed_operation: None,
        })
    }

    fn observe_failed_step(
        &mut self,
        phase: ExecutionPhase,
        contract: &[LayerExecutionContract],
        records: &[[Option<OperationExecutionRecord>; 2]],
    ) -> Result<(Option<usize>, Option<usize>, Option<usize>), String> {
        if records.len() > contract.len() {
            return Err("failed operation trace exceeds layer contract".into());
        }
        let phase_index = execution_phase_index(phase);
        let mut failure = (None, None, None);
        for (layer_index, layer) in records.iter().enumerate() {
            for (operation_index, record) in layer.iter().enumerate() {
                let Some(record) = record else { continue };
                if record.phase != phase
                    || !operation_implementation_matches_contract(
                        contract[layer_index][phase_index][operation_index],
                        record.implementation_id,
                    )
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
                        failure = (Some(layer_index), Some(1), Some(operation_index));
                    }
                    OperationExecutionStatus::Started => {
                        return Err("failed operation trace retained an unclassified start".into());
                    }
                }
            }
        }
        if failure.0.is_none() {
            if let Some((layer_index, _)) = records.iter().enumerate().next_back() {
                // A downstream copy/synchronization failure can leave only successful operation
                // records.  Retain the last invocation location even when no operation reported
                // an explicit Failed status.
                failure = (Some(layer_index), Some(1), None);
            }
        }
        Ok(failure)
    }

    fn observe_failed_prefill(
        &mut self,
        contract: &[LayerExecutionContract],
        invocations: &[Qwen35FailedPrefillExecutionStep],
    ) -> Result<(Option<usize>, Option<usize>, Option<usize>), String> {
        let mut failure = (None, None, None);
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
                if record.phase != invocation.phase
                    || !operation_implementation_matches_contract(
                        expected,
                        record.implementation_id,
                    )
                {
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
                        failure = (
                            Some(invocation.layer_index),
                            Some(invocation.execution_width),
                            Some(operation_index),
                        );
                    }
                    OperationExecutionStatus::Started => {
                        return Err("failed prefill trace retained an unclassified start".into());
                    }
                }
            }
        }
        if failure.0.is_none() {
            if let Some(invocation) = invocations.last() {
                // Preserve the failed invocation width for post-operation failures whose
                // operation records all completed successfully.
                failure = (
                    Some(invocation.layer_index),
                    Some(invocation.execution_width),
                    None,
                );
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
        failed_execution_width: Option<usize>,
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
            failed_execution_width,
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
        // Native Qwen3.5 gated self-attention uses one width-M cache writer and one width-M
        // sigmoid-gated causal reader per layer. Plain projection remains on the M1 path, but
        // either M1 writer variant promotes to the generic native chunk writer.
        "hip.paged-kv-write-f32.m1" | "hip.fused-qk-norm-rope-paged-kv-write-f32.m1" => {
            "hip.paged-kv-write-chunk-f32.m2-m128"
        }
        "hip.paged-decode-attention-sigmoid-gate-f32.m1-gqa" => {
            "hip.paged-causal-gqa-chunk-sigmoid-gate-f32.m2-m128"
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
    reasoning: Option<ReasoningState>,
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
    active_lifecycle_observer: Option<Qwen35Aq4LifecycleObserver>,
    last_terminal_request_audit: Option<Qwen35Aq4RequestExecutionAudit>,
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
            active_lifecycle_observer: None,
            last_terminal_request_audit: None,
        })
    }

    pub fn status(&self) -> Qwen35Aq4SessionStatus {
        self.status
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn config(&self) -> &Qwen35Aq4SessionConfig {
        &self.config
    }

    pub fn requested_prefill_chunk_tokens(&self) -> usize {
        self.config.prefill_chunk_tokens
    }

    pub fn operation_resolution_traces(&self) -> Vec<Vec<OperationResolutionTrace>> {
        self.model.operation_resolution_traces()
    }

    pub fn last_terminal_operation_audit(&self) -> Option<&OperationExecutionAudit> {
        self.last_terminal_operation_audit.as_ref()
    }

    /// Returns the privacy-safe terminal request audit, including the existing registry audit
    /// when operation-contract accounting was enabled for the model.
    pub fn last_terminal_request_execution_audit(&self) -> Option<&Qwen35Aq4RequestExecutionAudit> {
        self.last_terminal_request_audit.as_ref()
    }

    fn fail<T>(&mut self, message: impl Into<String>) -> Result<T, String> {
        let mut message = message.into();
        if let Some(observer) = self.active_lifecycle_observer.as_mut() {
            if let Err(error) = observer.observe_error() {
                message.push_str(&format!("; lifecycle error observation failed: {error}"));
            }
        }
        if let Err(error) = self.snapshot_terminal_request_audit() {
            message.push_str(&format!("; terminal lifecycle audit failed: {error}"));
        }
        self.status = Qwen35Aq4SessionStatus::Failed;
        Err(message)
    }

    fn snapshot_terminal_request_audit(&mut self) -> Result<(), String> {
        let operation_audit = self.last_terminal_operation_audit.clone();
        self.last_terminal_request_audit = match self.active_lifecycle_observer.as_ref() {
            Some(observer) => Some(observer.snapshot(operation_audit)?),
            None => None,
        };
        Ok(())
    }

    fn prepare_prefill_chunk(&mut self) -> Result<SessionAdvance<Qwen35Aq4PreparedToken>, String> {
        let (absolute_start, token_ids, cancel) = {
            let active = self
                .active
                .as_ref()
                .ok_or_else(|| "Qwen3.5 AQ4 prefill chunk has no active request".to_string())?;
            let absolute_start = active.prompt_tokens_processed;
            let end = absolute_start
                .checked_add(self.config.prefill_chunk_tokens)
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
                        .unwrap_or((None, None, None)),
                    (Some(audit), Some(contract)) => audit
                        .observe_failed_step(phase, contract, &partial_records)
                        .unwrap_or((None, None, None)),
                    _ => (None, None, None),
                };
                self.last_terminal_operation_audit =
                    self.active_operation_audit.as_ref().map(|audit| {
                        audit.partial(
                            self.execution_contract.as_ref().map_or(0, Vec::len),
                            "execution_failed",
                            Some(phase),
                            failure.0,
                            failure.1,
                            failure.2,
                        )
                    });
                return self.fail(format!(
                    "Qwen3.5 AQ4 prefill chunk dispatch failed: {error}"
                ));
            }
        };
        if let Some(observer) = self.active_lifecycle_observer.as_mut() {
            if let Err(error) = observer.prepare_prefill() {
                return self.fail(error);
            }
        }
        // Dispatch returns only after every requested invocation has been launched and its
        // successful operation records have been collected.  Account for that physical work
        // before the stream synchronization below: synchronization can fail after dispatch, and
        // a partial terminal audit must still retain the invocation and token-equivalent counts.
        // Commit/progress stay below synchronization and the post-sync cancellation check.
        let physical_observation = if let (Some(contract), Some(audit)) = (
            self.execution_contract.as_deref(),
            self.active_operation_audit.as_mut(),
        ) {
            match audit.observe_prefill_chunk(phase, execution_width, contract, &execution_steps) {
                Ok(observation) => Some(observation),
                Err(error) => {
                    self.model.mark_prefill_chunk_uncommitted();
                    if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                        if let Err(lifecycle_error) = observer.discard_prefill() {
                            return self.fail(format!(
                                "Qwen3.5 AQ4 prefill chunk operation audit failed: {error}; lifecycle discard failed: {lifecycle_error}"
                            ));
                        }
                    }
                    return self.fail(format!(
                        "Qwen3.5 AQ4 prefill chunk operation audit failed: {error}"
                    ));
                }
            }
        } else {
            None
        };
        if let (Some(observer), Some(observation)) = (
            self.active_lifecycle_observer.as_mut(),
            physical_observation,
        ) {
            if let Err(error) = observer.observe_prefill_execution(observation) {
                self.model.mark_prefill_chunk_uncommitted();
                let discard_error = observer.discard_prefill().err();
                return self.fail(match discard_error {
                    Some(discard_error) => format!(
                        "Qwen3.5 AQ4 physical prefill observation failed: {error}; lifecycle discard failed: {discard_error}"
                    ),
                    None => format!("Qwen3.5 AQ4 physical prefill observation failed: {error}"),
                });
            }
        }
        if let Err(error) = self.model.synchronize_after_prefill_chunk() {
            self.model.mark_prefill_chunk_uncommitted();
            if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                if let Err(lifecycle_error) = observer.discard_prefill() {
                    return self.fail(format!(
                        "Qwen3.5 AQ4 prefill chunk synchronization failed: {error}; lifecycle discard failed: {lifecycle_error}"
                    ));
                }
            }
            self.last_terminal_operation_audit =
                self.active_operation_audit.as_ref().map(|audit| {
                    audit.partial(
                        self.execution_contract.as_ref().map_or(0, Vec::len),
                        "synchronization_failed",
                        Some(phase),
                        None,
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
            if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                if let Err(error) = observer
                    .discard_prefill()
                    .and_then(|_| observer.observe_cancel())
                {
                    return self.fail(error);
                }
            }
            let active = self.active.as_mut().expect("active request checked above");
            active.terminal_outcome = Some(ReleaseOutcome::Cancelled);
            self.status = Qwen35Aq4SessionStatus::Terminal;
            if let Err(error) = self.snapshot_terminal_request_audit() {
                return self.fail(error);
            }
            return Ok(SessionAdvance::CancellationObserved);
        }
        let next_prompt_tokens_processed = match self
            .active
            .as_ref()
            .expect("active request checked above")
            .prompt_tokens_processed
            .checked_add(execution_width)
        {
            Some(value) => value,
            None => {
                self.model.mark_prefill_chunk_uncommitted();
                if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                    let _ = observer.discard_prefill();
                }
                return self.fail("Qwen3.5 AQ4 prefill progress overflows");
            }
        };
        if let Some(audit) = self.active_operation_audit.as_mut() {
            if let Err(error) = audit.commit_prefill_chunk(execution_width) {
                if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                    if let Err(lifecycle_error) = observer.discard_prefill() {
                        return self.fail(format!(
                            "Qwen3.5 AQ4 prefill chunk commit audit failed: {error}; lifecycle discard failed: {lifecycle_error}"
                        ));
                    }
                }
                return self.fail(format!(
                    "Qwen3.5 AQ4 prefill chunk commit audit failed: {error}"
                ));
            }
        }
        if let Some(observer) = self.active_lifecycle_observer.as_mut() {
            if let Err(error) = observer.commit_prefill() {
                return self.fail(error);
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
        let force_for_length = self
            .active
            .as_ref()
            .zip(self.config.reasoning_dialect.as_ref())
            .and_then(|(active, dialect)| {
                active.reasoning.as_ref().map(|reasoning| {
                    reasoning.phase == ReasoningPhase::Reasoning
                        && active
                            .max_new_tokens
                            .saturating_sub(active.generated_tokens)
                            <= dialect
                                .forced_end_sequence
                                .len()
                                .saturating_add(dialect.reserved_answer_tokens)
                })
            })
            .unwrap_or(false);
        if force_for_length {
            let reasoning = self
                .active
                .as_mut()
                .and_then(|active| active.reasoning.as_mut())
                .ok_or_else(|| {
                    format!("{label} reasoning state disappeared before length guard")
                })?;
            reasoning
                .force_close()
                .map_err(|error| format!("{label} reasoning length guard failed: {error:?}"))?;
        }
        let forced_token = self
            .active
            .as_ref()
            .and_then(|active| active.reasoning.as_ref())
            .and_then(ReasoningState::next_forced_token);
        if self
            .active
            .as_ref()
            .and_then(|active| active.reasoning.as_ref())
            .is_some_and(|reasoning| {
                reasoning.phase == ReasoningPhase::ForcingEndSequence && forced_token.is_none()
            })
        {
            return self.fail(format!("{label} reasoning forced sequence is exhausted"));
        }
        let token_id = match forced_token {
            Some(token_id) => token_id,
            None => match self.model.top_token_from_last_layer(label) {
                Ok(token_id) => token_id,
                Err(error) => return self.fail(format!("{label} top-1 failed: {error}")),
            },
        };
        if token_id >= self.model.vocab_size() {
            return self.fail(format!(
                "{label} top-1 token {token_id} exceeds vocabulary size {}",
                self.model.vocab_size()
            ));
        }
        let (generated_index, cache_len, next_generated) = match self.active.as_ref() {
            Some(active) => {
                let generated_index = active.generated_tokens;
                let Some(cache_len) = active.prompt_token_ids.len().checked_add(generated_index)
                else {
                    return self.fail("Qwen3.5 AQ4 prepared cache length overflows");
                };
                let Some(next_generated) = generated_index.checked_add(1) else {
                    return self.fail("Qwen3.5 AQ4 generated token count overflows");
                };
                (generated_index, cache_len, next_generated)
            }
            None => {
                return self.fail("Qwen3.5 AQ4 token preparation has no active request");
            }
        };
        let (reasoning_tokens_before, forced_end_tokens_before) = self
            .active
            .as_ref()
            .and_then(|active| active.reasoning.as_ref())
            .map(|reasoning| (reasoning.reasoning_tokens, reasoning.forced_end_tokens))
            .unwrap_or((0, 0));
        let terminal_reason = if let Some(active) = self.active.as_mut() {
            if let Some(reasoning) = active.reasoning.as_mut() {
                if forced_token.is_some() {
                    reasoning.accept_forced(token_id)
                } else if self.config.eos_token_ids.contains(&token_id) {
                    Ok(reasoning.on_eos())
                } else {
                    reasoning.accept_sampled(token_id)
                }
                .map_err(|error| format!("{label} reasoning transition failed: {error:?}"))?;
                if reasoning.phase == ReasoningPhase::Finished {
                    Some(FinishReason::Stop)
                } else if next_generated == active.max_new_tokens {
                    Some(FinishReason::Length)
                } else {
                    None
                }
            } else if self.config.eos_token_ids.contains(&token_id) {
                Some(FinishReason::Stop)
            } else if next_generated == active.max_new_tokens {
                Some(FinishReason::Length)
            } else {
                None
            }
        } else {
            return self.fail("Qwen3.5 AQ4 token preparation has no active request");
        };
        let Some(next_nonce) = self.next_nonce.checked_add(1) else {
            return self.fail("Qwen3.5 AQ4 prepared token nonce overflows");
        };
        let prepared = Qwen35Aq4PreparedToken {
            token_id,
            generated_index,
            cache_len,
            terminal_reason,
            reasoning_tokens_before,
            forced_end_tokens_before,
            nonce: self.next_nonce,
        };
        if let Some(observer) = self.active_lifecycle_observer.as_mut() {
            if let Err(error) = observer.prepare_publication() {
                return self.fail(error);
            }
        }
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
        if let Some(prepared) = self.pending.as_ref() {
            if let Some(reasoning) = self
                .active
                .as_mut()
                .and_then(|active| active.reasoning.as_mut())
            {
                reasoning.reasoning_tokens = prepared.reasoning_tokens_before;
                reasoning.forced_end_tokens = prepared.forced_end_tokens_before;
            }
        }
        if let Some(observer) = self.active_lifecycle_observer.as_mut() {
            if observer.publication_open {
                if let Err(error) = observer.discard_publication() {
                    return self.fail(error);
                }
            }
        }
        let active = self
            .active
            .as_ref()
            .ok_or_else(|| "Qwen3.5 AQ4 reset has no active request".to_string())?;
        let reasoning_usage = active.reasoning.as_ref().map(|reasoning| ReasoningUsage {
            reasoning_tokens: reasoning.reasoning_tokens,
            forced_end_tokens: reasoning.forced_end_tokens,
        });
        let summary = ReleaseSummary {
            request_id: active.request_id.clone(),
            outcome,
            prompt_tokens: active.prompt_token_ids.len(),
            generated_tokens: active.generated_tokens,
            reasoning_usage,
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
                Some(audit.partial(audit_layers, "cancelled", None, None, None, None))
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
        if let Some(observer) = self.active_lifecycle_observer.as_mut() {
            if let Err(error) = observer.begin_reset() {
                return self.fail(error);
            }
        }
        if let Err(error) = self.model.reset_all_request_state_synchronized() {
            self.last_terminal_operation_audit = self
                .active_operation_audit
                .as_ref()
                .map(|audit| audit.partial(audit_layers, "reset_failed", None, None, None, None));
            if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                let lifecycle_result = observer.fail_reset().and_then(|_| observer.observe_error());
                if let Err(lifecycle_error) = lifecycle_result {
                    self.status = Qwen35Aq4SessionStatus::Failed;
                    return Err(format!(
                        "Qwen3.5 AQ4 request reset failed: {error}; lifecycle reset failure observation failed: {lifecycle_error}"
                    ));
                }
            }
            if let Err(lifecycle_error) = self.snapshot_terminal_request_audit() {
                self.status = Qwen35Aq4SessionStatus::Failed;
                return Err(format!(
                    "Qwen3.5 AQ4 request reset failed: {error}; terminal lifecycle audit failed: {lifecycle_error}"
                ));
            }
            self.status = Qwen35Aq4SessionStatus::Failed;
            return Err(format!("Qwen3.5 AQ4 request reset failed: {error}"));
        }
        if let Some(observer) = self.active_lifecycle_observer.as_mut() {
            if let Err(error) = observer.complete_reset() {
                return self.fail(error);
            }
        }
        self.active = None;
        self.active_operation_audit = None;
        self.last_terminal_operation_audit = terminal_audit;
        if let Err(error) = self.snapshot_terminal_request_audit() {
            return self.fail(error);
        }
        self.pending = None;
        self.active_lifecycle_observer = None;
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
        let reasoning = match (request.reasoning, self.config.reasoning_dialect.as_ref()) {
            (Some(execution), Some(dialect)) => {
                if execution.dialect_id != dialect.identity
                    || execution.end_sequence != dialect.end_sequence
                    || execution.forced_end_sequence != dialect.forced_end_sequence
                    || execution.reserved_answer_tokens != dialect.reserved_answer_tokens
                {
                    return Err("Qwen3.5 AQ4 reasoning dialect is not bound to the session".into());
                }
                Some(
                    ReasoningState::new(
                        dialect.clone(),
                        execution.enabled,
                        execution.budget_tokens,
                        self.model.vocab_size(),
                    )
                    .map_err(|error| {
                        format!("Qwen3.5 AQ4 reasoning contract is invalid: {error:?}")
                    })?,
                )
            }
            (Some(_), None) => {
                return Err("Qwen3.5 AQ4 reasoning request has no loaded dialect".into());
            }
            (None, _) => None,
        };
        self.active = Some(ActiveRequest {
            request_id: request.request_id,
            prompt_token_ids: request.prompt_token_ids,
            max_new_tokens: request.max_new_tokens,
            cancel,
            prompt_tokens_processed: 0,
            generated_tokens: 0,
            decode_input: None,
            terminal_outcome: None,
            reasoning,
        });
        self.active_operation_audit = self
            .execution_contract
            .as_ref()
            .map(|_| OperationAuditAccumulator::new());
        self.last_terminal_operation_audit = None;
        self.active_lifecycle_observer = Some(Qwen35Aq4LifecycleObserver::new(
            self.config.prefill_chunk_tokens,
        ));
        self.last_terminal_request_audit = None;
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
            if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                if let Err(error) = observer.observe_cancel() {
                    return self.fail(error);
                }
            }
            let active = self.active.as_mut().expect("active request checked above");
            active.terminal_outcome = Some(ReleaseOutcome::Cancelled);
            self.status = Qwen35Aq4SessionStatus::Terminal;
            if let Err(error) = self.snapshot_terminal_request_audit() {
                return self.fail(error);
            }
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
                        .unwrap_or((None, None, None)),
                    _ => (None, None, None),
                };
                self.last_terminal_operation_audit =
                    self.active_operation_audit.as_ref().map(|audit| {
                        audit.partial(
                            self.execution_contract.as_ref().map_or(0, Vec::len),
                            "execution_failed",
                            Some(phase),
                            failure.0,
                            failure.1,
                            failure.2,
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
            if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                if let Err(error) = observer.observe_decode_execution() {
                    return self.fail(error);
                }
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
        let publication = match cancel.publication_guard() {
            Ok(publication) => publication,
            Err(error) => {
                self.pending = None;
                if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                    if let Err(lifecycle_error) = observer
                        .discard_publication()
                        .and_then(|_| observer.observe_error())
                    {
                        return self.fail(format!(
                            "{error}; publication guard lifecycle observation failed: {lifecycle_error}"
                        ));
                    }
                }
                self.status = Qwen35Aq4SessionStatus::Terminal;
                if let Err(lifecycle_error) = self.snapshot_terminal_request_audit() {
                    return self.fail(format!(
                        "{error}; terminal lifecycle audit failed: {lifecycle_error}"
                    ));
                }
                return Err(error);
            }
        };
        if cancel.is_cancelled() {
            if let Some(reasoning) = self
                .active
                .as_mut()
                .and_then(|active| active.reasoning.as_mut())
            {
                reasoning.reasoning_tokens = prepared.reasoning_tokens_before;
                reasoning.forced_end_tokens = prepared.forced_end_tokens_before;
            }
            drop(publication);
            self.pending = None;
            if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                if let Err(error) = observer
                    .discard_publication()
                    .and_then(|_| observer.observe_cancel())
                {
                    return self.fail(error);
                }
            }
            let active = self.active.as_mut().expect("active request checked above");
            active.terminal_outcome = Some(ReleaseOutcome::Cancelled);
            self.status = Qwen35Aq4SessionStatus::Terminal;
            if let Err(error) = self.snapshot_terminal_request_audit() {
                return self.fail(error);
            }
            return Ok(PublishedAdvance::CancellationObserved);
        }
        if let Err(error) = publish(prepared.token_id) {
            if let Some(reasoning) = self
                .active
                .as_mut()
                .and_then(|active| active.reasoning.as_mut())
            {
                reasoning.reasoning_tokens = prepared.reasoning_tokens_before;
                reasoning.forced_end_tokens = prepared.forced_end_tokens_before;
            }
            drop(publication);
            self.pending = None;
            if let Some(observer) = self.active_lifecycle_observer.as_mut() {
                if let Err(lifecycle_error) = observer
                    .discard_publication()
                    .and_then(|_| observer.observe_error())
                {
                    return self.fail(format!(
                        "Qwen3.5 AQ4 token publisher failed before commit: {error}; lifecycle observation failed: {lifecycle_error}"
                    ));
                }
            }
            // A publisher failure does not poison resident model state. The caller must abort it.
            self.status = Qwen35Aq4SessionStatus::Terminal;
            if let Err(lifecycle_error) = self.snapshot_terminal_request_audit() {
                return self.fail(format!(
                    "Qwen3.5 AQ4 token publisher failed before commit: {error}; terminal lifecycle audit failed: {lifecycle_error}"
                ));
            }
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
        if let Some(observer) = self.active_lifecycle_observer.as_mut() {
            if let Err(error) = observer.commit_publication() {
                return self.fail(error);
            }
        }
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
    validate_prefill_chunk_tokens(config.prefill_chunk_tokens)?;
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
                phases[0] = if layer_index % 4 == 3 {
                    [
                        "hip.paged-kv-write-chunk-f32.m2-m128",
                        "hip.paged-causal-gqa-chunk-sigmoid-gate-f32.m2-m128",
                    ]
                } else {
                    [
                        "hip.linear-attention-qkv-prepare-batch-f32.m2-m128",
                        "hip.linear-attention-recurrent-sequence-f32.m2-m128",
                    ]
                };
                phases[1] = phases[0];
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

    #[test]
    fn operation_implementation_contract_accepts_only_matching_split_reader_family() {
        assert!(operation_implementation_matches_contract(
            "hip.paged-decode-attention-f32.m1-gqa",
            "hip.paged-decode-attention-f32.m1-gqa"
        ));
        assert!(operation_implementation_matches_contract(
            "hip.paged-decode-attention-f32.m1-gqa",
            "hip.paged-decode-attention-split-f32.tile128"
        ));
        assert!(operation_implementation_matches_contract(
            "hip.paged-decode-attention-f32.m1-gqa",
            "hip.paged-decode-attention-split-f32.tile256"
        ));
        assert!(!operation_implementation_matches_contract(
            "hip.paged-decode-attention-f32.m1-gqa",
            "hip.paged-decode-attention-split-sigmoid-gate-f32.tile128"
        ));
        assert!(operation_implementation_matches_contract(
            "hip.paged-decode-attention-sigmoid-gate-f32.m1-gqa",
            "hip.paged-decode-attention-split-sigmoid-gate-f32.tile256"
        ));
        assert!(!operation_implementation_matches_contract(
            "hip.paged-decode-attention-sigmoid-gate-f32.m1-gqa",
            "hip.paged-decode-attention-split-f32.tile256"
        ));
        for actual in [
            "hip.paged-kv-write-f32.m1",
            "hip.linear-attention-recurrent-f32.m1",
            "hip.paged-causal-gqa-chunk-sigmoid-gate-f32.m2-m128",
            "unknown",
        ] {
            assert!(!operation_implementation_matches_contract(
                "hip.paged-decode-attention-f32.m1-gqa",
                actual
            ));
        }
        assert!(!operation_implementation_matches_contract(
            "unknown",
            "hip.paged-decode-attention-split-f32.tile128"
        ));
        assert!(!operation_implementation_matches_contract(
            "unknown", "unknown"
        ));
    }

    #[test]
    fn operation_audit_accepts_split_reader_actual_and_counts_actual_id() {
        let contract = audited_contract();
        let mut records = audited_records(&contract, ExecutionPhase::Decode);
        records[3][1].implementation_id =
            "hip.paged-decode-attention-split-sigmoid-gate-f32.tile128";

        let mut audit = OperationAuditAccumulator::new();
        audit
            .observe(ExecutionPhase::Decode, &contract, &records)
            .unwrap();
        assert_eq!(audit.implementation_counts[12], 1);
        assert_eq!(audit.implementation_counts[5], 7);

        let mut second_records = audited_records(&contract, ExecutionPhase::Decode);
        audit
            .observe(ExecutionPhase::Decode, &contract, &second_records)
            .unwrap();
        assert_eq!(audit.implementation_counts[5], 15);
        assert_eq!(audit.implementation_counts[12], 1);
        second_records[3][1].implementation_id = "hip.paged-decode-attention-split-f32.tile128";
        assert!(
            audit
                .observe(ExecutionPhase::Decode, &contract, &second_records)
                .is_err()
        );
    }

    #[test]
    fn failed_split_reader_record_matches_contract_and_retains_failure_location() {
        let contract = audited_contract();
        let mut records = audited_failed_records(ExecutionPhase::Decode, 3, 1);
        records[3][1] = Some(OperationExecutionRecord {
            implementation_id: "hip.paged-decode-attention-split-sigmoid-gate-f32.tile256",
            phase: ExecutionPhase::Decode,
            status: OperationExecutionStatus::Failed,
        });
        assert_eq!(
            OperationAuditAccumulator::new()
                .observe_failed_step(ExecutionPhase::Decode, &contract, &records)
                .unwrap(),
            (Some(3), Some(1), Some(1))
        );
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
    fn prefill_physical_width_requires_identical_observed_batches_across_layers() {
        let contract = audited_contract();
        let fallback = fallback_prefill_invocations(&contract, 2);
        let observation = OperationAuditAccumulator::new()
            .observe_prefill_chunk(ExecutionPhase::ColdPrefill, 2, &contract, &fallback)
            .unwrap();
        assert_eq!(observation.actual_token_batch_width, 1);
        assert_eq!(observation.internal_batch_count, 2);
        assert_eq!(
            observation.phase_batch_counts,
            Qwen35Aq4PhaseBatchCounts {
                cold_prefill: 1,
                cached_prefix_prefill: 1,
                decode: 0,
            }
        );

        let mut mismatched = fallback;
        mismatched.retain(|invocation| invocation.layer_index != 0);
        let native_contract = native_hybrid_contract(2);
        mismatched.push(Qwen35PrefillExecutionStep {
            layer_index: 0,
            execution_width: 2,
            phase: ExecutionPhase::ColdPrefill,
            records: audited_records(&native_contract, ExecutionPhase::ColdPrefill)[0],
        });
        assert!(
            OperationAuditAccumulator::new()
                .observe_prefill_chunk(ExecutionPhase::ColdPrefill, 2, &contract, &mismatched)
                .unwrap_err()
                .contains("physical batch sequence differs")
        );
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
        assert_eq!(failure, (Some(1), Some(128), Some(1)));
        assert_eq!(audit.total_records, 3);
        assert_eq!(audit.token_equivalent_operation_coverage, 384);
        assert_eq!(audit.implementation_counts[6], 2);
        assert_eq!(audit.implementation_counts[7], 1);
    }

    #[test]
    fn failed_native_self_invocation_retains_layer_width_and_reader_operation() {
        let contract = audited_contract();
        let mut audit = OperationAuditAccumulator::new();
        let failure = audit
            .observe_failed_prefill(
                &contract,
                &[Qwen35FailedPrefillExecutionStep {
                    layer_index: 3,
                    execution_width: 127,
                    phase: ExecutionPhase::CachedPrefixPrefill,
                    records: [
                        Some(OperationExecutionRecord {
                            implementation_id: "hip.paged-kv-write-chunk-f32.m2-m128",
                            phase: ExecutionPhase::CachedPrefixPrefill,
                            status: OperationExecutionStatus::Succeeded,
                        }),
                        Some(OperationExecutionRecord {
                            implementation_id:
                                "hip.paged-causal-gqa-chunk-sigmoid-gate-f32.m2-m128",
                            phase: ExecutionPhase::CachedPrefixPrefill,
                            status: OperationExecutionStatus::Failed,
                        }),
                    ],
                }],
            )
            .unwrap();
        assert_eq!(failure, (Some(3), Some(127), Some(1)));
        assert_eq!(audit.total_records, 1);
        assert_eq!(audit.token_equivalent_operation_coverage, 127);
        assert_eq!(audit.implementation_counts[8], 1);
        assert_eq!(audit.implementation_counts[9], 0);
    }

    #[test]
    fn failed_native_self_post_operation_retains_last_invocation_width_without_operation() {
        let contract = audited_contract();
        let mut audit = OperationAuditAccumulator::new();
        let failure = audit
            .observe_failed_prefill(
                &contract,
                &[Qwen35FailedPrefillExecutionStep {
                    layer_index: 3,
                    execution_width: 3,
                    phase: ExecutionPhase::ColdPrefill,
                    records: [
                        Some(OperationExecutionRecord {
                            implementation_id: "hip.paged-kv-write-chunk-f32.m2-m128",
                            phase: ExecutionPhase::ColdPrefill,
                            status: OperationExecutionStatus::Succeeded,
                        }),
                        Some(OperationExecutionRecord {
                            implementation_id:
                                "hip.paged-causal-gqa-chunk-sigmoid-gate-f32.m2-m128",
                            phase: ExecutionPhase::ColdPrefill,
                            status: OperationExecutionStatus::Succeeded,
                        }),
                    ],
                }],
            )
            .unwrap();
        assert_eq!(failure, (Some(3), Some(3), None));
        assert_eq!(audit.total_records, 2);
        assert_eq!(audit.token_equivalent_operation_coverage, 6);
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
            let mut invocations = Vec::with_capacity(contract.len());
            for layer_index in 0..contract.len() {
                invocations.push(Qwen35PrefillExecutionStep {
                    layer_index,
                    execution_width: token_ids.len(),
                    phase,
                    records: audited_records(&contract, phase)[layer_index],
                });
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

    fn reasoning_dialect() -> crate::reasoning::ReasoningDialect {
        crate::reasoning::ReasoningDialect {
            identity: "synthetic.qwen35-thinking.v1".into(),
            start_sequence: vec![10],
            end_sequence: vec![20, 21],
            forced_end_sequence: vec![20, 21],
            max_budget_tokens: 8,
            reserved_answer_tokens: 1,
            enabled_by_default: false,
            effort_budgets: vec![("low".into(), 2), ("medium".into(), 4), ("high".into(), 8)],
            history_reasoning_policy: crate::reasoning::HistoryReasoningPolicy::Omit,
            initial_phase: crate::reasoning::InitialReasoningPhase::Reasoning,
            eos_policy: crate::reasoning::ReasoningEosPolicy::Close,
        }
    }

    fn prepared(advance: SessionAdvance<Qwen35Aq4PreparedToken>) -> Qwen35Aq4PreparedToken {
        match advance {
            SessionAdvance::Token { prepared, .. } => prepared,
            other => panic!("expected token, got {other:?}"),
        }
    }

    #[test]
    fn prefill_chunk_config_accepts_only_bounded_grid_and_keeps_128_default() {
        let default = Qwen35Aq4SessionConfig::greedy(1, vec![2]);
        assert_eq!(default.prefill_chunk_tokens(), 128);
        assert_eq!(QWEN35_AQ4_PREFILL_CHUNK_GRID, &[1, 8, 16, 32, 64, 128]);
        for width in QWEN35_AQ4_PREFILL_CHUNK_GRID {
            assert_eq!(
                default
                    .clone()
                    .with_prefill_chunk_tokens(*width)
                    .unwrap()
                    .prefill_chunk_tokens(),
                *width
            );
        }
        for width in [0, 2, 7, 129, usize::MAX] {
            assert!(default.clone().with_prefill_chunk_tokens(width).is_err());
        }
    }

    #[test]
    fn configured_prefill_chunk_width_controls_tail_boundaries() {
        let config = Qwen35Aq4SessionConfig::greedy(1, vec![2])
            .with_prefill_chunk_tokens(64)
            .unwrap();
        let mut scripted = model(&[2]);
        scripted.context = 256;
        scripted.audited = true;
        let mut session = Qwen35Aq4InferenceSession::from_model(scripted, config).unwrap();
        session.execution_contract = Some(audited_contract());
        session
            .start_request(
                request("m64-boundary", &vec![4; 130], 1),
                CancellationToken::new(),
            )
            .unwrap();
        let mut widths = Vec::new();
        let token = loop {
            match session.prepare_advance().unwrap() {
                SessionAdvance::PromptProgress {
                    execution_width, ..
                } => widths.push(execution_width),
                SessionAdvance::Token { prepared, .. } => break prepared,
                SessionAdvance::CancellationObserved => panic!("unexpected cancellation"),
            }
        };
        assert_eq!(widths, vec![64, 64, 2]);
        session.publish_prepared(token, |_| Ok(())).unwrap();
        session.finish_and_reset().unwrap();
        let audit = session.last_terminal_request_execution_audit().unwrap();
        assert_eq!(audit.requested_m, 64);
        // The fallback model physically executed 130 M1 batches even though the logical
        // scheduler chunks were 64/64/2. The terminal facts must report physical M1.
        assert_eq!(audit.resolved_m, Some(1));
        assert_eq!(audit.actual_token_batch_width, Some(1));
        assert_eq!(audit.actual_request_batch_width, Some(1));
        assert_eq!(audit.internal_batch_count, Some(130));
        assert_eq!(
            audit.phase_batch_counts,
            Some(Qwen35Aq4PhaseBatchCounts {
                cold_prefill: 1,
                cached_prefix_prefill: 129,
                decode: 0,
            })
        );
    }

    #[test]
    fn lifecycle_observer_records_publish_failure_cancel_and_reset() {
        let mut publish = session(&[7]);
        publish
            .start_request(
                request("observer-publish", &[4], 2),
                CancellationToken::new(),
            )
            .unwrap();
        let token = next_prepared(&mut publish);
        assert!(
            publish
                .publish_prepared(token, |_| Err("closed".to_string()))
                .is_err()
        );
        publish.abort_and_reset().unwrap();
        let publish_audit = publish.last_terminal_request_execution_audit().unwrap();
        assert_eq!(publish_audit.resolved_m, None);
        assert_eq!(publish_audit.actual_token_batch_width, None);
        assert_eq!(publish_audit.actual_request_batch_width, None);
        let lifecycle = publish_audit.lifecycle;
        assert_eq!(lifecycle.prepare, 2);
        assert_eq!(lifecycle.commit, 1);
        assert_eq!(lifecycle.discard, 1);
        assert_eq!(lifecycle.error, 1);
        assert_eq!(lifecycle.cancel, 0);
        assert_eq!(lifecycle.prepare, lifecycle.commit + lifecycle.discard);
        assert_eq!(lifecycle.prefill.prepare, 1);
        assert_eq!(lifecycle.prefill.commit, 1);
        assert_eq!(lifecycle.publication.prepare, 1);
        assert_eq!(lifecycle.publication.discard, 1);
        assert_eq!(
            lifecycle.reset,
            Qwen35Aq4ResetCounts {
                attempted: 1,
                complete: 1,
                failed: 0,
            }
        );

        let mut cleanup_abort = session(&[7]);
        cleanup_abort
            .start_request(
                request("observer-cleanup-abort", &[4], 1),
                CancellationToken::new(),
            )
            .unwrap();
        let _prepared_but_unpublished = next_prepared(&mut cleanup_abort);
        cleanup_abort.abort_and_reset().unwrap();
        let lifecycle = cleanup_abort
            .last_terminal_request_execution_audit()
            .unwrap()
            .lifecycle;
        assert_eq!(lifecycle.cancel, 0);
        assert_eq!(lifecycle.prepare, 2);
        assert_eq!(lifecycle.commit, 1);
        assert_eq!(lifecycle.discard, 1);
        assert_eq!(lifecycle.publication.discard, 1);
        assert_eq!(lifecycle.prepare, lifecycle.commit + lifecycle.discard);

        let mut cancel = session(&[7]);
        let cancellation = CancellationToken::new();
        cancel
            .start_request(request("observer-cancel", &[4], 1), cancellation.clone())
            .unwrap();
        cancellation.cancel();
        assert_eq!(
            cancel.prepare_advance().unwrap(),
            SessionAdvance::CancellationObserved
        );
        cancel.abort_and_reset().unwrap();
        let cancel_audit = cancel.last_terminal_request_execution_audit().unwrap();
        assert_eq!(cancel_audit.resolved_m, None);
        assert_eq!(cancel_audit.actual_token_batch_width, None);
        assert_eq!(cancel_audit.actual_request_batch_width, None);
        let lifecycle = cancel_audit.lifecycle;
        assert_eq!(lifecycle.prepare, 0);
        assert_eq!(lifecycle.commit, 0);
        assert_eq!(lifecycle.discard, 0);
        assert_eq!(lifecycle.error, 0);
        assert_eq!(lifecycle.cancel, 1);
        assert_eq!(lifecycle.prepare, lifecycle.commit + lifecycle.discard);
        assert_eq!(
            lifecycle.reset,
            Qwen35Aq4ResetCounts {
                attempted: 1,
                complete: 1,
                failed: 0,
            }
        );

        let mut reset = session(&[2]);
        reset.model.fail_reset = true;
        reset
            .start_request(request("observer-reset", &[4], 1), CancellationToken::new())
            .unwrap();
        let token = next_prepared(&mut reset);
        reset.publish_prepared(token, |_| Ok(())).unwrap();
        assert!(reset.finish_and_reset().is_err());
        let lifecycle = reset
            .last_terminal_request_execution_audit()
            .unwrap()
            .lifecycle;
        assert_eq!(lifecycle.prepare, 2);
        assert_eq!(lifecycle.commit, 2);
        assert_eq!(lifecycle.error, 1);
        assert_eq!(lifecycle.prepare, lifecycle.commit + lifecycle.discard);
        assert_eq!(
            lifecycle.reset,
            Qwen35Aq4ResetCounts {
                attempted: 1,
                complete: 0,
                failed: 1,
            }
        );
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
    fn forced_reasoning_sequence_uses_the_prepare_publish_commit_boundary() {
        let dialect = reasoning_dialect();
        let mut config = Qwen35Aq4SessionConfig::greedy(8, vec![2]);
        config.reasoning_dialect = Some(dialect.clone());
        let mut session = Qwen35Aq4InferenceSession::from_model(model(&[7, 9, 2]), config).unwrap();
        let mut request = request("forced-reasoning", &[4], 6);
        request.reasoning = Some(crate::reasoning::ReasoningExecution {
            enabled: true,
            budget_tokens: Some(1),
            dialect_id: dialect.identity.clone(),
            end_sequence: dialect.end_sequence.clone(),
            forced_end_sequence: dialect.forced_end_sequence.clone(),
            reserved_answer_tokens: dialect.reserved_answer_tokens,
        });
        session
            .start_request(request, CancellationToken::new())
            .unwrap();

        let first = next_prepared(&mut session);
        assert_eq!(first.token_id, 7);
        session.publish_prepared(first, |_| Ok(())).unwrap();

        let forced_first = next_prepared(&mut session);
        assert_eq!(forced_first.token_id, 20);
        session.publish_prepared(forced_first, |_| Ok(())).unwrap();
        let forced_second = next_prepared(&mut session);
        assert_eq!(forced_second.token_id, 21);
        session.publish_prepared(forced_second, |_| Ok(())).unwrap();

        let answer = next_prepared(&mut session);
        assert_eq!(answer.token_id, 9);
        session.publish_prepared(answer, |_| Ok(())).unwrap();
        let eos = next_prepared(&mut session);
        assert_eq!(eos.token_id, 2);
        assert_eq!(eos.terminal_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn unbounded_reasoning_keeps_a_minimum_answer_reservation_at_length() {
        let dialect = reasoning_dialect();
        let mut config = Qwen35Aq4SessionConfig::greedy(8, vec![2]);
        config.reasoning_dialect = Some(dialect.clone());
        let mut session = Qwen35Aq4InferenceSession::from_model(model(&[7, 2]), config).unwrap();
        let mut request = request("unbounded-reasoning", &[4], 4);
        request.reasoning = Some(crate::reasoning::ReasoningExecution {
            enabled: true,
            budget_tokens: None,
            dialect_id: dialect.identity.clone(),
            end_sequence: dialect.end_sequence.clone(),
            forced_end_sequence: dialect.forced_end_sequence.clone(),
            reserved_answer_tokens: dialect.reserved_answer_tokens,
        });
        session
            .start_request(request, CancellationToken::new())
            .unwrap();

        let body = next_prepared(&mut session);
        assert_eq!(body.token_id, 7);
        session.publish_prepared(body, |_| Ok(())).unwrap();
        let forced_first = next_prepared(&mut session);
        assert_eq!(forced_first.token_id, 20);
        session.publish_prepared(forced_first, |_| Ok(())).unwrap();
        let forced_second = next_prepared(&mut session);
        assert_eq!(forced_second.token_id, 21);
        session.publish_prepared(forced_second, |_| Ok(())).unwrap();
        let answer_eos = next_prepared(&mut session);
        assert_eq!(answer_eos.token_id, 2);
        assert_eq!(answer_eos.terminal_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn cancellation_during_reasoning_forced_close_resets_for_reuse() {
        let dialect = reasoning_dialect();
        let mut config = Qwen35Aq4SessionConfig::greedy(8, vec![2]);
        config.reasoning_dialect = Some(dialect.clone());
        let mut session = Qwen35Aq4InferenceSession::from_model(model(&[7, 9, 2]), config).unwrap();
        let cancel = CancellationToken::new();
        let mut reasoning_request = request("reasoning-cancel", &[4], 6);
        reasoning_request.reasoning = Some(crate::reasoning::ReasoningExecution {
            enabled: true,
            budget_tokens: Some(1),
            dialect_id: dialect.identity.clone(),
            end_sequence: dialect.end_sequence.clone(),
            forced_end_sequence: dialect.forced_end_sequence.clone(),
            reserved_answer_tokens: dialect.reserved_answer_tokens,
        });
        session
            .start_request(reasoning_request, cancel.clone())
            .unwrap();
        let body = next_prepared(&mut session);
        session.publish_prepared(body, |_| Ok(())).unwrap();
        let forced = next_prepared(&mut session);
        assert_eq!(forced.token_id, 20);
        cancel.cancel();
        assert_eq!(
            session
                .publish_prepared(forced, |_| panic!("cancelled publication must not run"))
                .unwrap(),
            PublishedAdvance::CancellationObserved
        );
        let cancelled = session.abort_and_reset().unwrap();
        assert_eq!(cancelled.generated_tokens, 1);
        assert_eq!(
            cancelled.reasoning_usage,
            Some(ReasoningUsage {
                reasoning_tokens: 1,
                forced_end_tokens: 0,
            })
        );
        assert_eq!(session.status(), Qwen35Aq4SessionStatus::Ready);

        session
            .start_request(
                request("after-reasoning-cancel", &[4], 1),
                CancellationToken::new(),
            )
            .unwrap();
        let next = next_prepared(&mut session);
        session.publish_prepared(next, |_| Ok(())).unwrap();
        assert_eq!(
            session.finish_and_reset().unwrap().outcome,
            ReleaseOutcome::Length
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
        for execution_width in [1, 2, 3, 127, 128] {
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
            assert_eq!(audit.physical_operation_invocations, 64);
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
                u64::from(execution_width == 1) * 8
            );
            assert_eq!(
                audit.implementation_counts[5].count,
                u64::from(execution_width == 1) * 8
            );
            assert_eq!(audit.implementation_counts[6].count, expected_native_linear);
            assert_eq!(audit.implementation_counts[7].count, expected_native_linear);
            assert_eq!(
                audit.implementation_counts[8].count,
                u64::from(execution_width > 1) * 8
            );
            assert_eq!(
                audit.implementation_counts[9].count,
                u64::from(execution_width > 1) * 8
            );
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
    fn native_prefill_two_chunks_count_each_layer_pair_once_and_sum_widths() {
        let mut audit = OperationAuditAccumulator::new();
        let first_width = 2;
        let second_width = 3;
        let first_contract = native_hybrid_contract(first_width);
        let second_contract = native_hybrid_contract(second_width);
        let native_invocations = |contract: &[LayerExecutionContract],
                                  phase: ExecutionPhase,
                                  width: usize|
         -> Vec<Qwen35PrefillExecutionStep> {
            audited_records(contract, phase)
                .into_iter()
                .enumerate()
                .map(|(layer_index, records)| Qwen35PrefillExecutionStep {
                    layer_index,
                    execution_width: width,
                    phase,
                    records,
                })
                .collect()
        };
        audit
            .observe_prefill_chunk(
                ExecutionPhase::ColdPrefill,
                first_width,
                &first_contract,
                &native_invocations(&first_contract, ExecutionPhase::ColdPrefill, first_width),
            )
            .unwrap();
        audit.commit_prefill_chunk(first_width).unwrap();
        audit
            .observe_prefill_chunk(
                ExecutionPhase::CachedPrefixPrefill,
                second_width,
                &second_contract,
                &native_invocations(
                    &second_contract,
                    ExecutionPhase::CachedPrefixPrefill,
                    second_width,
                ),
            )
            .unwrap();
        audit.commit_prefill_chunk(second_width).unwrap();

        assert_eq!(audit.total_records, 128);
        assert_eq!(
            audit.token_equivalent_operation_coverage,
            64 * (first_width + second_width) as u64
        );
        assert_eq!(audit.prefill_chunks_executed, 2);
        assert_eq!(audit.prefill_tokens_executed, 5);
        assert_eq!(audit.prefill_tokens_committed, 5);
        assert_eq!(audit.prefill_width_histogram[first_width], 1);
        assert_eq!(audit.prefill_width_histogram[second_width], 1);
        let finished = audit.finish(32, 1, 4, 0, "length").unwrap();
        assert_eq!(finished.physical_operation_invocations, 128);
        assert!(finished.coverage_complete);
    }

    #[test]
    fn native_session_commits_two_chunks_with_nonzero_cached_prefix() {
        let mut scripted = model(&[2]);
        scripted.context = 256;
        scripted.audited = true;
        scripted.native_hybrid_prefill = true;
        let mut session = Qwen35Aq4InferenceSession::from_model(
            scripted,
            Qwen35Aq4SessionConfig::greedy(1, vec![2]),
        )
        .unwrap();
        session.execution_contract = Some(native_hybrid_contract(128));
        session
            .start_request(
                request("native-two-chunks", &vec![4; 130], 1),
                CancellationToken::new(),
            )
            .unwrap();
        assert_eq!(
            session.prepare_advance().unwrap(),
            SessionAdvance::PromptProgress {
                prompt_tokens_processed: 128,
                cache_len: 128,
                execution_width: 128,
            }
        );
        assert_eq!(
            session.prepare_advance().unwrap(),
            SessionAdvance::PromptProgress {
                prompt_tokens_processed: 130,
                cache_len: 130,
                execution_width: 2,
            }
        );
        let token = next_prepared(&mut session);
        session.publish_prepared(token, |_| Ok(())).unwrap();
        session.finish_and_reset().unwrap();
        let audit = session.last_terminal_operation_audit().unwrap();
        assert_eq!(audit.prefill_chunks_executed, 2);
        assert_eq!(audit.cold_prefill_steps, 1);
        assert_eq!(audit.cached_prefix_prefill_steps, 129);
        assert_eq!(audit.physical_operation_invocations, 128);
        assert_eq!(audit.token_equivalent_operation_coverage, 64 * 130);
        assert_eq!(audit.prefill_tokens_committed, 130);
        assert_eq!(audit.prefill_width_histogram[128], 1);
        assert_eq!(audit.prefill_width_histogram[2], 1);
        assert!(audit.coverage_complete);
        let request_audit = session.last_terminal_request_execution_audit().unwrap();
        assert_eq!(request_audit.requested_m, 128);
        assert_eq!(request_audit.resolved_m, Some(128));
        assert_eq!(request_audit.actual_token_batch_width, Some(128));
        assert_eq!(request_audit.actual_request_batch_width, Some(1));
        assert_eq!(request_audit.internal_batch_count, Some(2));
        assert_eq!(
            request_audit.phase_batch_counts,
            Some(Qwen35Aq4PhaseBatchCounts {
                cold_prefill: 1,
                cached_prefix_prefill: 1,
                decode: 0,
            })
        );
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
                reasoning_usage: None,
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
        assert_eq!(audit.physical_operation_invocations, 64);
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
        assert_eq!(audit.physical_operation_invocations, 64);
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
