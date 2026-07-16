// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only counters for candidate-A direct sequence-output evidence.
//!
//! The counters are fed by the production route transition after an operation has completed.
//! They do not estimate bytes from a requested route: a copy is counted only when the route
//! transition actually executes the workspace-to-destination copy.  The default model runtime
//! keeps this collector disabled; callers must explicitly enable the diagnostic gate and then
//! pass the bounded counters to the hash-bound Python assembler.

use serde::Serialize;
use sha2::{Digest, Sha256};

pub const RUNTIME_OBSERVATION_SCHEMA: &str =
    "ullm.aq4_p3_candidate_a_direct_runtime_observation.v1";
pub const CANDIDATE_ID: &str = "sequence-output-direct-v1";
pub const RUNTIME_EVIDENCE_LANE: &str = "instrumented_diagnostic";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35Aq4DirectTraceRoute {
    Copy,
    Direct,
    CopyFallback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qwen35Aq4DirectTraceCounters {
    pub invocation_count: u64,
    pub d2d_bytes: u64,
    pub d2d_copy_count: u64,
    pub launch_count: u64,
    pub workspace_bytes: u64,
    pub fallback_count: u64,
    pub fallback_reasons: Vec<String>,
    pub direct_alias_safe: bool,
    pub direct_size_safe: bool,
    pub direct_admission_safe: bool,
    pub failed_invocation_count: u64,
    pub failure_reasons: Vec<String>,
}

impl Default for Qwen35Aq4DirectTraceCounters {
    fn default() -> Self {
        Self {
            invocation_count: 0,
            d2d_bytes: 0,
            d2d_copy_count: 0,
            launch_count: 0,
            workspace_bytes: 0,
            fallback_count: 0,
            fallback_reasons: Vec::new(),
            direct_alias_safe: true,
            direct_size_safe: true,
            direct_admission_safe: true,
            failed_invocation_count: 0,
            failure_reasons: Vec::new(),
        }
    }
}

impl Qwen35Aq4DirectTraceCounters {
    /// Records one completed route transition and its actual operation launch count.
    pub fn record_invocation(
        &mut self,
        route: Qwen35Aq4DirectTraceRoute,
        sequence_bytes: u64,
        launch_count: u64,
        workspace_bytes: u64,
    ) -> Result<(), String> {
        let mut next = self.clone();
        next.invocation_count = next
            .invocation_count
            .checked_add(1)
            .ok_or_else(|| "direct trace invocation count overflows".to_string())?;
        next.launch_count = next
            .launch_count
            .checked_add(launch_count)
            .ok_or_else(|| "direct trace launch count overflows".to_string())?;
        next.workspace_bytes = next.workspace_bytes.max(workspace_bytes);
        match route {
            Qwen35Aq4DirectTraceRoute::Direct => {}
            Qwen35Aq4DirectTraceRoute::Copy => {
                next.record_copy(sequence_bytes)?;
            }
            Qwen35Aq4DirectTraceRoute::CopyFallback => {
                next.record_copy(sequence_bytes)?;
                next.fallback_count = next
                    .fallback_count
                    .checked_add(1)
                    .ok_or_else(|| "direct trace fallback count overflows".to_string())?;
                next.direct_alias_safe = false;
                next.direct_size_safe = false;
                next.direct_admission_safe = false;
                next.push_reason("direct_admission_failed")?;
            }
        }
        *self = next;
        Ok(())
    }

    /// Records a failed route attempt without pretending that a copy or launch completed.
    pub fn record_failure(&mut self, reason: &str) -> Result<(), String> {
        if reason.is_empty() || reason.len() > 128 || !reason.is_ascii() {
            return Err("direct trace failure reason is invalid".to_string());
        }
        let mut next = self.clone();
        next.failed_invocation_count = next
            .failed_invocation_count
            .checked_add(1)
            .ok_or_else(|| "direct trace failed invocation count overflows".to_string())?;
        next.push_failure_reason(reason)?;
        *self = next;
        Ok(())
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn record_copy(&mut self, sequence_bytes: u64) -> Result<(), String> {
        self.d2d_bytes = self
            .d2d_bytes
            .checked_add(sequence_bytes)
            .ok_or_else(|| "direct trace D2D bytes overflow".to_string())?;
        self.d2d_copy_count = self
            .d2d_copy_count
            .checked_add(1)
            .ok_or_else(|| "direct trace D2D copy count overflows".to_string())?;
        Ok(())
    }

    fn push_reason(&mut self, reason: &str) -> Result<(), String> {
        if !self.fallback_reasons.iter().any(|item| item == reason) {
            if self.fallback_reasons.len() >= 16 {
                return Err("direct trace fallback reason count exceeds bound".to_string());
            }
            self.fallback_reasons.push(reason.to_string());
            self.fallback_reasons.sort();
        }
        Ok(())
    }

    fn push_failure_reason(&mut self, reason: &str) -> Result<(), String> {
        if !self.failure_reasons.iter().any(|item| item == reason) {
            if self.failure_reasons.len() >= 16 {
                return Err("direct trace failure reason count exceeds bound".to_string());
            }
            self.failure_reasons.push(reason.to_string());
            self.failure_reasons.sort();
        }
        Ok(())
    }
}

fn bounded_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
        })
}

fn sha256_text(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(value: &[u8]) -> String {
    let digest = Sha256::digest(value);
    let mut result = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing SHA-256 to String cannot fail");
    }
    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Aq4DirectTraceBinding {
    pub side: String,
    pub binding_kind: String,
    pub binding_id: String,
    pub request_id: String,
    pub implementation_id: String,
    pub source_id: String,
    pub source_sha256: String,
    pub case_id: String,
    pub case_sha256: String,
    pub identity_sha256: String,
    pub direct_sequence_output_enabled: bool,
}

impl Qwen35Aq4DirectTraceBinding {
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.side.as_str(), "baseline" | "candidate")
            || !matches!(self.binding_kind.as_str(), "run" | "pair")
            || ![
                self.binding_id.as_str(),
                self.request_id.as_str(),
                self.implementation_id.as_str(),
                self.source_id.as_str(),
                self.case_id.as_str(),
            ]
            .into_iter()
            .all(bounded_identifier)
            || ![
                self.source_sha256.as_str(),
                self.case_sha256.as_str(),
                self.identity_sha256.as_str(),
            ]
            .into_iter()
            .all(sha256_text)
            || self.direct_sequence_output_enabled != (self.side == "candidate")
        {
            return Err("direct trace request binding is invalid".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Qwen35Aq4DirectRuntimeObservation {
    pub schema_version: &'static str,
    pub status: String,
    pub record_sha256: Option<String>,
    pub side: String,
    pub binding_kind: String,
    pub binding_id: String,
    pub request_id: String,
    pub implementation_id: String,
    pub source_id: String,
    pub source_sha256: String,
    pub candidate_id: &'static str,
    pub case_id: String,
    pub case_sha256: String,
    pub identity_sha256: String,
    pub diagnostic_gate: bool,
    pub direct_sequence_output_enabled: bool,
    pub evidence_lane: &'static str,
    pub measurement_eligible: bool,
    pub terminal_status: String,
    pub counters: Qwen35Aq4DirectTraceCounters,
}

impl Qwen35Aq4DirectRuntimeObservation {
    fn new(
        binding: Qwen35Aq4DirectTraceBinding,
        terminal_status: &str,
        counters: Qwen35Aq4DirectTraceCounters,
    ) -> Result<Self, String> {
        binding.validate()?;
        if !matches!(terminal_status, "completed" | "cancelled" | "error") {
            return Err("direct trace terminal status is invalid".to_string());
        }
        let status = if terminal_status == "completed" && counters.failed_invocation_count == 0 {
            "complete"
        } else {
            "failed"
        };
        let mut result = Self {
            schema_version: RUNTIME_OBSERVATION_SCHEMA,
            status: status.to_string(),
            record_sha256: None,
            side: binding.side,
            binding_kind: binding.binding_kind,
            binding_id: binding.binding_id,
            request_id: binding.request_id,
            implementation_id: binding.implementation_id,
            source_id: binding.source_id,
            source_sha256: binding.source_sha256,
            candidate_id: CANDIDATE_ID,
            case_id: binding.case_id,
            case_sha256: binding.case_sha256,
            identity_sha256: binding.identity_sha256,
            diagnostic_gate: true,
            direct_sequence_output_enabled: binding.direct_sequence_output_enabled,
            evidence_lane: RUNTIME_EVIDENCE_LANE,
            measurement_eligible: false,
            terminal_status: terminal_status.to_string(),
            counters,
        };
        let canonical_value = serde_json::to_value(&result)
            .map_err(|error| format!("failed to serialize direct trace observation: {error}"))?;
        let canonical = serde_json::to_vec(&canonical_value)
            .map_err(|error| format!("failed to canonicalize direct trace observation: {error}"))?;
        result.record_sha256 = Some(sha256_hex(&canonical));
        Ok(result)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, String> {
        let mut raw = serde_json::to_vec(self)
            .map_err(|error| format!("failed to serialize direct trace observation: {error}"))?;
        raw.push(b'\n');
        Ok(raw)
    }
}

#[derive(Debug, Default)]
pub struct Qwen35Aq4DirectTraceCollector {
    active_binding: Option<Qwen35Aq4DirectTraceBinding>,
    counters: Qwen35Aq4DirectTraceCounters,
}

impl Qwen35Aq4DirectTraceCollector {
    pub fn begin_request(&mut self, binding: Qwen35Aq4DirectTraceBinding) -> Result<(), String> {
        binding.validate()?;
        if self.active_binding.is_some() {
            return Err("direct trace request is already active".to_string());
        }
        self.counters.reset();
        self.active_binding = Some(binding);
        Ok(())
    }

    pub fn request_active(&self) -> bool {
        self.active_binding.is_some()
    }

    pub fn failure_recorded(&self) -> bool {
        self.counters.failed_invocation_count != 0
    }

    pub fn record_invocation(
        &mut self,
        route: Qwen35Aq4DirectTraceRoute,
        sequence_bytes: u64,
        launch_count: u64,
        workspace_bytes: u64,
    ) -> Result<(), String> {
        if self.active_binding.is_none() {
            return Err("direct trace request was not started".to_string());
        }
        self.counters
            .record_invocation(route, sequence_bytes, launch_count, workspace_bytes)
    }

    pub fn record_failure(&mut self, reason: &str) -> Result<(), String> {
        if self.active_binding.is_none() {
            return Err("direct trace request was not started".to_string());
        }
        self.counters.record_failure(reason)
    }

    pub fn finish_request(
        &mut self,
        terminal_status: &str,
    ) -> Result<Qwen35Aq4DirectRuntimeObservation, String> {
        let binding = self
            .active_binding
            .take()
            .ok_or_else(|| "direct trace request was not started".to_string())?;
        let counters = std::mem::take(&mut self.counters);
        Qwen35Aq4DirectRuntimeObservation::new(binding, terminal_status, counters)
    }

    pub fn reset_without_emission(&mut self) {
        self.active_binding = None;
        self.counters.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(side: &str, binding_id: &str) -> Qwen35Aq4DirectTraceBinding {
        Qwen35Aq4DirectTraceBinding {
            side: side.to_string(),
            binding_kind: "run".to_string(),
            binding_id: binding_id.to_string(),
            request_id: format!("request-{binding_id}"),
            implementation_id: "qwen35-aq4-direct-v1".to_string(),
            source_id: "qwen35_aq4_model_runtime".to_string(),
            source_sha256: "a".repeat(64),
            case_id: "case-1".to_string(),
            case_sha256: "b".repeat(64),
            identity_sha256: "c".repeat(64),
            direct_sequence_output_enabled: side == "candidate",
        }
    }

    #[test]
    fn direct_route_counts_launches_without_a_copy() {
        let mut counters = Qwen35Aq4DirectTraceCounters::default();
        counters
            .record_invocation(Qwen35Aq4DirectTraceRoute::Direct, 512, 2, 4096)
            .unwrap();
        assert_eq!(counters.invocation_count, 1);
        assert_eq!(counters.launch_count, 2);
        assert_eq!(counters.d2d_bytes, 0);
        assert_eq!(counters.d2d_copy_count, 0);
        assert_eq!(counters.workspace_bytes, 4096);
        assert_eq!(counters.fallback_count, 0);
        assert!(counters.direct_alias_safe);
        assert!(counters.direct_size_safe);
        assert!(counters.direct_admission_safe);
    }

    #[test]
    fn copy_fallback_counts_one_copy_and_reason() {
        let mut counters = Qwen35Aq4DirectTraceCounters::default();
        counters
            .record_invocation(Qwen35Aq4DirectTraceRoute::CopyFallback, 512, 2, 4096)
            .unwrap();
        assert_eq!(counters.d2d_bytes, 512);
        assert_eq!(counters.d2d_copy_count, 1);
        assert_eq!(counters.fallback_count, 1);
        assert_eq!(counters.fallback_reasons, vec!["direct_admission_failed"]);
        assert!(!counters.direct_alias_safe);
        assert!(!counters.direct_size_safe);
        assert!(!counters.direct_admission_safe);
    }

    #[test]
    fn failure_does_not_count_copy_and_reset_clears_state() {
        let mut counters = Qwen35Aq4DirectTraceCounters::default();
        counters.record_failure("destination_copy_failed").unwrap();
        assert_eq!(counters.failed_invocation_count, 1);
        assert_eq!(counters.d2d_bytes, 0);
        assert_eq!(counters.d2d_copy_count, 0);
        assert_eq!(counters.failure_reasons, vec!["destination_copy_failed"]);
        counters.reset();
        assert_eq!(counters, Qwen35Aq4DirectTraceCounters::default());
    }

    #[test]
    fn request_observation_is_exact_once_hash_bound_and_request_scoped() {
        let mut collector = Qwen35Aq4DirectTraceCollector::default();
        collector
            .begin_request(binding("candidate", "run-2"))
            .unwrap();
        collector
            .record_invocation(Qwen35Aq4DirectTraceRoute::Direct, 512, 2, 4096)
            .unwrap();
        let observation = collector.finish_request("completed").unwrap();
        assert_eq!(observation.status, "complete");
        assert_eq!(observation.request_id, "request-run-2");
        assert_eq!(observation.evidence_lane, RUNTIME_EVIDENCE_LANE);
        assert!(!observation.measurement_eligible);
        assert_eq!(observation.counters.invocation_count, 1);
        assert_eq!(observation.counters.d2d_copy_count, 0);
        assert!(collector.finish_request("completed").is_err());

        let mut value = serde_json::to_value(&observation).unwrap();
        let object = value.as_object_mut().unwrap();
        assert_eq!(object.len(), 20);
        let recorded = object["record_sha256"].as_str().unwrap().to_string();
        object.insert("record_sha256".to_string(), serde_json::Value::Null);
        assert_eq!(recorded, sha256_hex(&serde_json::to_vec(&value).unwrap()));
    }

    #[test]
    fn error_cancel_and_new_request_never_carry_prior_counters() {
        let mut collector = Qwen35Aq4DirectTraceCollector::default();
        collector
            .begin_request(binding("baseline", "run-2"))
            .unwrap();
        collector
            .record_invocation(Qwen35Aq4DirectTraceRoute::Copy, 128, 2, 1024)
            .unwrap();
        collector.record_failure("prefill_dispatch_failed").unwrap();
        let failed = collector.finish_request("error").unwrap();
        assert_eq!(failed.status, "failed");
        assert_eq!(failed.counters.failed_invocation_count, 1);

        collector
            .begin_request(binding("baseline", "run-3"))
            .unwrap();
        collector.record_failure("request_cancelled").unwrap();
        let cancelled = collector.finish_request("cancelled").unwrap();
        assert_eq!(cancelled.status, "failed");
        assert_eq!(cancelled.counters.invocation_count, 0);
        assert_eq!(cancelled.counters.d2d_bytes, 0);
        assert_eq!(cancelled.counters.failed_invocation_count, 1);

        collector
            .begin_request(binding("baseline", "run-4"))
            .unwrap();
        collector.reset_without_emission();
        collector
            .begin_request(binding("baseline", "run-5"))
            .unwrap();
        let empty = collector.finish_request("completed").unwrap();
        assert_eq!(empty.counters, Qwen35Aq4DirectTraceCounters::default());
    }

    #[test]
    fn counter_overflow_is_transactional() {
        let mut counters = Qwen35Aq4DirectTraceCounters {
            d2d_copy_count: u64::MAX,
            ..Qwen35Aq4DirectTraceCounters::default()
        };
        let before = counters.clone();
        assert!(
            counters
                .record_invocation(Qwen35Aq4DirectTraceRoute::Copy, 1, 1, 1)
                .is_err()
        );
        assert_eq!(counters, before);
    }
}
