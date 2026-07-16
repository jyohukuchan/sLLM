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

pub const RUNTIME_OBSERVATION_SCHEMA: &str =
    "ullm.aq4_p3_candidate_a_direct_runtime_observation.v1";

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
        self.invocation_count = self
            .invocation_count
            .checked_add(1)
            .ok_or_else(|| "direct trace invocation count overflows".to_string())?;
        self.launch_count = self
            .launch_count
            .checked_add(launch_count)
            .ok_or_else(|| "direct trace launch count overflows".to_string())?;
        self.workspace_bytes = self.workspace_bytes.max(workspace_bytes);
        match route {
            Qwen35Aq4DirectTraceRoute::Direct => {}
            Qwen35Aq4DirectTraceRoute::Copy => {
                self.record_copy(sequence_bytes)?;
            }
            Qwen35Aq4DirectTraceRoute::CopyFallback => {
                self.record_copy(sequence_bytes)?;
                self.fallback_count = self
                    .fallback_count
                    .checked_add(1)
                    .ok_or_else(|| "direct trace fallback count overflows".to_string())?;
                self.direct_alias_safe = false;
                self.direct_size_safe = false;
                self.direct_admission_safe = false;
                self.push_reason("direct_admission_failed")?;
            }
        }
        Ok(())
    }

    /// Records a failed route attempt without pretending that a copy or launch completed.
    pub fn record_failure(&mut self, reason: &str) -> Result<(), String> {
        if reason.is_empty() || reason.len() > 128 || !reason.is_ascii() {
            return Err("direct trace failure reason is invalid".to_string());
        }
        self.failed_invocation_count = self
            .failed_invocation_count
            .checked_add(1)
            .ok_or_else(|| "direct trace failed invocation count overflows".to_string())?;
        self.push_failure_reason(reason)
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
