// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Quantization-independent adapter from an inference session to the JSONL worker runtime.

use crate::inference_api::{GenerationTimings, InferenceRequest, ReleaseOutcome};
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
        });

        let result =
            drive_worker_request(&mut self.session, request, admission.cancel, publications);
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
        let value = serde_json::to_value(BackendLog {
            schema_version: "ullm.worker.log.v1",
            level: "error",
            event: "request_failed",
            request_id: "req-log",
            phase: "execute",
            prompt_tokens: 128,
            completion_tokens: 3,
            elapsed_ms: 42,
            outcome: None,
            error_code: Some("runtime_failed"),
        })
        .unwrap();
        assert_eq!(value["schema_version"], "ullm.worker.log.v1");
        assert_eq!(value["request_id"], "req-log");
        assert_eq!(value["prompt_tokens"], 128);
        assert_eq!(value["completion_tokens"], 3);
        assert_eq!(value["error_code"], "runtime_failed");
        assert!(value.get("prompt_token_ids").is_none());
        assert!(value.get("token_id").is_none());
        assert!(value.get("message").is_none());
    }
}
