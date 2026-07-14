// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Quantization-independent adapter from an inference session to the JSONL worker runtime.

use crate::backend_operation_registry::OperationExecutionAudit;
use crate::inference_api::{GenerationTimings, InferenceRequest, ReasoningUsage, ReleaseOutcome};
use crate::worker_driver::{InferenceSession, RequestPublications, drive_worker_request};
use crate::worker_protocol::{ReleaseOutcomeEvent, WorkerAdmission};
use crate::worker_runtime::{InferenceBackend, RequestEventPublisher};
use serde::Serialize;
use std::io::Write;
use std::time::Instant;

/// A JSONL worker backend whose model-specific behavior is entirely supplied by `S`.
#[derive(Debug)]
pub struct SessionInferenceBackend<S> {
    session: S,
}

impl<S> SessionInferenceBackend<S> {
    pub fn new(session: S) -> Self {
        Self { session }
    }

    pub fn session(&self) -> &S {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut S {
        &mut self.session
    }

    pub fn into_session(self) -> S {
        self.session
    }
}

impl<S: InferenceSession> InferenceBackend for SessionInferenceBackend<S> {
    fn execute(
        &mut self,
        request: InferenceRequest,
        admission: WorkerAdmission,
        publications: &mut RequestEventPublisher<'_>,
    ) -> Result<(), String> {
        let request_id = request.request_id.clone();
        let prompt_tokens = request.prompt_token_ids.len();
        let started = Instant::now();
        write_backend_log(BackendLog {
            schema_version: "ullm.worker.log.v1",
            level: "info",
            event: "request_admitted",
            request_id: &request_id,
            phase: "start",
            prompt_tokens,
            completion_tokens: 0,
            elapsed_ms: 0,
            outcome: None,
            error_code: None,
            operation_execution_audit: None,
            request_execution_audit: None,
        });

        let result =
            drive_worker_request(&mut self.session, request, admission.cancel, publications);
        let request_execution_audit = self.session.terminal_sanitized_execution_audit();
        match result {
            Ok(outcome) => {
                write_backend_log(BackendLog {
                    schema_version: "ullm.worker.log.v1",
                    level: "info",
                    event: "request_released",
                    request_id: &request_id,
                    phase: "reset_complete",
                    prompt_tokens,
                    completion_tokens: publications.completion_tokens(),
                    elapsed_ms: elapsed_millis(started),
                    outcome: Some(release_outcome_name(outcome)),
                    error_code: None,
                    operation_execution_audit: self.session.terminal_operation_execution_audit(),
                    request_execution_audit: request_execution_audit.as_ref(),
                });
                Ok(())
            }
            Err(error) => {
                write_backend_log(BackendLog {
                    schema_version: "ullm.worker.log.v1",
                    level: "error",
                    event: "request_failed",
                    request_id: &request_id,
                    phase: "execute",
                    prompt_tokens,
                    completion_tokens: publications.completion_tokens(),
                    elapsed_ms: elapsed_millis(started),
                    outcome: None,
                    error_code: Some("runtime_failed"),
                    operation_execution_audit: self.session.terminal_operation_execution_audit(),
                    request_execution_audit: request_execution_audit.as_ref(),
                });
                Err(error)
            }
        }
    }

    fn shutdown(&mut self) -> Result<(), String> {
        self.session.shutdown()
    }
}

impl RequestPublications for RequestEventPublisher<'_> {
    fn publish_started(&mut self) -> Result<(), String> {
        RequestEventPublisher::publish_started(self)
    }

    fn observe_prompt_unit(
        &mut self,
        prompt_tokens_processed: usize,
        execution_width: usize,
    ) -> Result<(), String> {
        RequestEventPublisher::observe_prompt_unit(self, prompt_tokens_processed, execution_width)
    }

    fn observe_prefill_transition(&mut self) -> Result<(), String> {
        RequestEventPublisher::observe_prefill_transition(self)
    }

    fn publish_token(&mut self, token_id: usize) -> Result<(), String> {
        RequestEventPublisher::publish_token(self, token_id)
    }

    fn publish_released(
        &mut self,
        outcome: ReleaseOutcome,
        timings: Option<GenerationTimings>,
    ) -> Result<(), String> {
        let event_outcome = release_outcome_event(outcome);
        match timings {
            Some(timings) => self.publish_released_with_timings(event_outcome, timings),
            None => RequestEventPublisher::publish_released(self, event_outcome),
        }
    }

    fn set_reasoning_usage(&mut self, usage: Option<ReasoningUsage>) {
        RequestEventPublisher::set_reasoning_usage(self, usage)
    }

    fn run_terminal_cleanup<T, F>(&mut self, cleanup: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
    {
        RequestEventPublisher::run_terminal_cleanup(self, cleanup)
    }

    fn completion_tokens(&self) -> usize {
        RequestEventPublisher::completion_tokens(self)
    }
}

pub fn release_outcome_event(outcome: ReleaseOutcome) -> ReleaseOutcomeEvent {
    match outcome {
        ReleaseOutcome::Stop => ReleaseOutcomeEvent::Stop,
        ReleaseOutcome::Length => ReleaseOutcomeEvent::Length,
        ReleaseOutcome::Cancelled => ReleaseOutcomeEvent::Cancelled,
    }
}

pub fn release_outcome_name(outcome: ReleaseOutcome) -> &'static str {
    match outcome {
        ReleaseOutcome::Stop => "stop",
        ReleaseOutcome::Length => "length",
        ReleaseOutcome::Cancelled => "cancelled",
    }
}

#[derive(Serialize)]
struct BackendLog<'a> {
    schema_version: &'static str,
    level: &'static str,
    event: &'static str,
    request_id: &'a str,
    phase: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
    elapsed_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_code: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation_execution_audit: Option<&'a OperationExecutionAudit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_execution_audit: Option<&'a serde_json::Value>,
}

fn write_backend_log(record: BackendLog<'_>) {
    let mut stderr = std::io::stderr().lock();
    let _ = serde_json::to_writer(&mut stderr, &record);
    let _ = stderr.write_all(b"\n");
    let _ = stderr.flush();
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_outcomes_and_log_names_are_stable() {
        for (outcome, event, name) in [
            (ReleaseOutcome::Stop, ReleaseOutcomeEvent::Stop, "stop"),
            (
                ReleaseOutcome::Length,
                ReleaseOutcomeEvent::Length,
                "length",
            ),
            (
                ReleaseOutcome::Cancelled,
                ReleaseOutcomeEvent::Cancelled,
                "cancelled",
            ),
        ] {
            assert_eq!(release_outcome_event(outcome), event);
            assert_eq!(release_outcome_name(outcome), name);
        }
    }

    #[test]
    fn structured_backend_log_contains_counts_but_no_prompt_or_token_content() {
        const IMPLEMENTATION_COUNT_FIXTURE: [(&str, &str, u64); 14] = [
            ("kind", "a", 0),
            ("kind", "b", 1),
            ("kind", "c", 2),
            ("kind", "d", 3),
            ("kind", "e", 4),
            ("kind", "f", 5),
            ("kind", "g", 6),
            ("kind", "h", 7),
            ("kind", "i", 8),
            ("kind", "j", 9),
            (
                "paged_causal_gqa_read",
                "hip.paged-decode-attention-split-f32.tile128",
                10,
            ),
            (
                "paged_causal_gqa_read",
                "hip.paged-decode-attention-split-f32.tile256",
                11,
            ),
            (
                "paged_causal_gqa_read",
                "hip.paged-decode-attention-split-sigmoid-gate-f32.tile128",
                12,
            ),
            (
                "paged_causal_gqa_read",
                "hip.paged-decode-attention-split-sigmoid-gate-f32.tile256",
                13,
            ),
        ];
        let audit = OperationExecutionAudit {
            schema_version: "ullm.backend_operation.request.v2",
            outcome: "stop",
            expected_layers_per_step: 32,
            expected_records_per_layer: 2,
            cold_prefill_steps: 1,
            cached_prefix_prefill_steps: 2,
            decode_steps: 1,
            total_steps: 4,
            total_records: 256,
            physical_operation_invocations: 256,
            token_equivalent_operation_coverage: 256,
            prefill_chunks_executed: 1,
            prefill_tokens_executed: 3,
            prefill_tokens_committed: 3,
            prefill_width_histogram: {
                let mut histogram = vec![0; 129];
                histogram[3] = 1;
                histogram
            },
            implementation_counts: IMPLEMENTATION_COUNT_FIXTURE.map(
                |(kind, implementation_id, count)| {
                    crate::backend_operation_registry::OperationExecutionCount {
                        kind,
                        implementation_id,
                        count,
                    }
                },
            ),
            deterministic_digest_sha256: [0xab; 32],
            coverage_complete: true,
            failed_phase: None,
            failed_layer: None,
            failed_execution_width: None,
            failed_operation: None,
        };
        let request_audit = serde_json::json!({
            "schema_version": "ullm.qwen35_aq4.request_execution.v1",
            "requested_m": 64,
            "resolved_m": 1,
            "actual_token_batch_width": 1,
            "actual_request_batch_width": 1,
            "lifecycle": {
                "prepare": 2,
                "commit": 2,
                "discard": 0,
                "error": 0,
                "cancel": 0,
                "reset": {"attempted": 1, "complete": 1, "failed": 0}
            }
        });
        let value = serde_json::to_value(BackendLog {
            schema_version: "ullm.worker.log.v1",
            level: "info",
            event: "request_released",
            request_id: "req-log",
            phase: "execute",
            prompt_tokens: 128,
            completion_tokens: 3,
            elapsed_ms: 42,
            outcome: Some("stop"),
            error_code: None,
            operation_execution_audit: Some(&audit),
            request_execution_audit: Some(&request_audit),
        })
        .unwrap();
        assert_eq!(value["schema_version"], "ullm.worker.log.v1");
        assert_eq!(value["request_id"], "req-log");
        assert_eq!(value["prompt_tokens"], 128);
        assert_eq!(value["completion_tokens"], 3);
        assert_eq!(value["outcome"], "stop");
        assert_eq!(
            value["operation_execution_audit"]["schema_version"],
            "ullm.backend_operation.request.v2"
        );
        assert_eq!(value["operation_execution_audit"]["total_records"], 256);
        assert_eq!(
            value["operation_execution_audit"]["physical_operation_invocations"],
            256
        );
        assert_eq!(
            value["operation_execution_audit"]["token_equivalent_operation_coverage"],
            256
        );
        assert_eq!(
            value["operation_execution_audit"]["prefill_width_histogram"][3],
            1
        );
        assert_eq!(
            value["operation_execution_audit"]["deterministic_digest_sha256"],
            "ab".repeat(32)
        );
        assert_eq!(
            value["operation_execution_audit"]["implementation_counts"]
                .as_array()
                .unwrap()
                .len(),
            14
        );
        assert_eq!(
            value["operation_execution_audit"]["coverage_complete"],
            true
        );
        assert_eq!(value["request_execution_audit"]["requested_m"], 64);
        assert_eq!(value["request_execution_audit"]["resolved_m"], 1);
        assert_eq!(
            value["request_execution_audit"]["lifecycle"]["reset"]["complete"],
            1
        );
        assert!(value.get("prompt_token_ids").is_none());
        assert!(value.get("token_id").is_none());
        assert!(value.get("message").is_none());
    }
}
