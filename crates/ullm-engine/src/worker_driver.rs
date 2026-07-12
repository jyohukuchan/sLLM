// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Quantization-independent inference session driver.

use crate::inference_api::{
    CancellationToken, FinishReason, GenerationTimings, InferenceRequest, ReleaseOutcome,
    ReleaseSummary,
};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionAdvance<P> {
    PromptProgress {
        prompt_tokens_processed: usize,
        cache_len: usize,
        execution_width: usize,
    },
    Token {
        prepared: P,
        token_id: usize,
        generated_index: usize,
        cache_len: usize,
        terminal_reason: Option<FinishReason>,
    },
    CancellationObserved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublishedAdvance {
    Token {
        token_id: usize,
        generated_index: usize,
        cache_len: usize,
        terminal_reason: Option<FinishReason>,
    },
    CancellationObserved,
}

pub trait InferenceSession {
    type Prepared;

    fn start_request(
        &mut self,
        request: InferenceRequest,
        cancel: CancellationToken,
    ) -> Result<(), String>;

    fn prepare_advance(&mut self) -> Result<SessionAdvance<Self::Prepared>, String>;

    fn publish_prepared<F>(
        &mut self,
        prepared: Self::Prepared,
        publish: F,
    ) -> Result<PublishedAdvance, String>
    where
        F: FnOnce(usize) -> Result<(), String>;

    fn finish_and_reset(&mut self) -> Result<ReleaseSummary, String>;

    fn abort_and_reset(&mut self) -> Result<ReleaseSummary, String>;

    /// Validate the reusable idle baseline before the worker exits.
    ///
    /// Sessions without resident resources may keep the default implementation.
    fn shutdown(&mut self) -> Result<(), String> {
        Ok(())
    }
}

pub trait RequestPublications {
    fn publish_started(&mut self) -> Result<(), String>;

    fn observe_prompt_unit(
        &mut self,
        prompt_tokens_processed: usize,
        execution_width: usize,
    ) -> Result<(), String>;

    fn observe_prefill_transition(&mut self) -> Result<(), String>;

    fn publish_token(&mut self, token_id: usize) -> Result<(), String>;

    fn publish_released(
        &mut self,
        outcome: ReleaseOutcome,
        timings: Option<GenerationTimings>,
    ) -> Result<(), String>;

    fn run_terminal_cleanup<T, F>(&mut self, cleanup: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>;

    fn completion_tokens(&self) -> usize;
}

pub const LLAMA_SERVER_MIN_TIMING_MS: f64 = 0.001;

#[derive(Debug)]
pub struct RequestTimingTracker {
    prompt_started_at: Instant,
    first_sample_at: Option<Instant>,
    last_sample_at: Option<Instant>,
    sampled_tokens: usize,
}

impl RequestTimingTracker {
    pub fn new(prompt_started_at: Instant) -> Self {
        Self {
            prompt_started_at,
            first_sample_at: None,
            last_sample_at: None,
            sampled_tokens: 0,
        }
    }

    pub fn observe_sample(&mut self, sampled_at: Instant) -> Result<(), String> {
        if sampled_at
            .checked_duration_since(self.prompt_started_at)
            .is_none()
            || self
                .last_sample_at
                .is_some_and(|last_sample_at| sampled_at < last_sample_at)
        {
            return Err("worker sample timing moved backwards".into());
        }
        if self.first_sample_at.is_none() {
            self.first_sample_at = Some(sampled_at);
        }
        self.last_sample_at = Some(sampled_at);
        self.sampled_tokens = self
            .sampled_tokens
            .checked_add(1)
            .ok_or_else(|| "worker sample timing count overflowed".to_string())?;
        Ok(())
    }

    pub fn finish(
        &self,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Result<GenerationTimings, String> {
        if self.sampled_tokens != completion_tokens || completion_tokens == 0 {
            return Err("worker timing count does not match completion tokens".into());
        }
        let first_sample_at = self
            .first_sample_at
            .ok_or_else(|| "worker timing has no first sample".to_string())?;
        let last_sample_at = self
            .last_sample_at
            .ok_or_else(|| "worker timing has no final sample".to_string())?;
        let prompt_elapsed = first_sample_at
            .checked_duration_since(self.prompt_started_at)
            .ok_or_else(|| "worker prompt timing moved backwards".to_string())?;
        let predicted_elapsed = last_sample_at
            .checked_duration_since(first_sample_at)
            .ok_or_else(|| "worker generation timing moved backwards".to_string())?;
        GenerationTimings::from_elapsed_millis_with_limits(
            prompt_tokens,
            positive_elapsed_millis(prompt_elapsed),
            completion_tokens,
            positive_elapsed_millis(predicted_elapsed),
            usize::MAX,
            usize::MAX,
        )
        .ok_or_else(|| "worker timings are out of range".to_string())
    }
}

fn positive_elapsed_millis(elapsed: std::time::Duration) -> f64 {
    (elapsed.as_secs_f64() * 1e3).max(LLAMA_SERVER_MIN_TIMING_MS)
}

pub fn drive_worker_request<S: InferenceSession, P: RequestPublications>(
    session: &mut S,
    request: InferenceRequest,
    cancel: CancellationToken,
    publications: &mut P,
) -> Result<ReleaseOutcome, String> {
    let expected_request_id = request.request_id.clone();
    let expected_prompt_tokens = request.prompt_token_ids.len();
    session.start_request(request, cancel)?;
    publications.publish_started()?;
    let mut timings = RequestTimingTracker::new(Instant::now());

    loop {
        let advance = session.prepare_advance()?;
        if matches!(&advance, SessionAdvance::Token { .. }) {
            timings.observe_sample(Instant::now())?;
        }
        match advance {
            SessionAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            } => {
                if cache_len != prompt_tokens_processed {
                    return Err("prompt progress cache length is inconsistent".into());
                }
                publications.observe_prompt_unit(prompt_tokens_processed, execution_width)?;
            }
            SessionAdvance::Token {
                prepared,
                token_id,
                generated_index,
                cache_len,
                terminal_reason,
            } => {
                if generated_index == 0 {
                    publications.observe_prefill_transition()?;
                }
                if generated_index != publications.completion_tokens() {
                    return Err("prepared token index does not match publication state".into());
                }
                let expected_cache_len = expected_prompt_tokens
                    .checked_add(generated_index)
                    .ok_or_else(|| "prepared token cache length overflows".to_string())?;
                if cache_len != expected_cache_len {
                    return Err("prepared token cache length is inconsistent".into());
                }
                match session.publish_prepared(prepared, |published_token_id| {
                    if published_token_id != token_id {
                        return Err("prepared token changed before publication".into());
                    }
                    publications.publish_token(published_token_id)
                })? {
                    PublishedAdvance::CancellationObserved => {
                        return finish_cancelled_request(
                            session,
                            &expected_request_id,
                            expected_prompt_tokens,
                            publications,
                        );
                    }
                    PublishedAdvance::Token {
                        token_id: committed_token_id,
                        generated_index: committed_index,
                        cache_len: committed_cache_len,
                        terminal_reason: committed_terminal,
                    } => {
                        if committed_token_id != token_id
                            || committed_index != generated_index
                            || committed_cache_len != cache_len
                            || committed_terminal != terminal_reason
                        {
                            return Err(
                                "committed token does not match its prepared proposal".into()
                            );
                        }
                        if let Some(reason) = committed_terminal {
                            let timings = timings
                                .finish(expected_prompt_tokens, publications.completion_tokens())?;
                            return finish_completed_request(
                                session,
                                &expected_request_id,
                                expected_prompt_tokens,
                                reason,
                                timings,
                                publications,
                            );
                        }
                    }
                }
            }
            SessionAdvance::CancellationObserved => {
                return finish_cancelled_request(
                    session,
                    &expected_request_id,
                    expected_prompt_tokens,
                    publications,
                );
            }
        }
    }
}

fn finish_completed_request<S: InferenceSession, P: RequestPublications>(
    session: &mut S,
    request_id: &str,
    prompt_tokens: usize,
    reason: FinishReason,
    timings: GenerationTimings,
    publications: &mut P,
) -> Result<ReleaseOutcome, String> {
    let summary = publications.run_terminal_cleanup(|| session.finish_and_reset())?;
    let outcome = match reason {
        FinishReason::Stop => ReleaseOutcome::Stop,
        FinishReason::Length => ReleaseOutcome::Length,
    };
    validate_release_summary(
        &summary,
        request_id,
        prompt_tokens,
        publications.completion_tokens(),
        outcome,
    )?;
    publications.publish_released(outcome, Some(timings))?;
    Ok(outcome)
}

fn finish_cancelled_request<S: InferenceSession, P: RequestPublications>(
    session: &mut S,
    request_id: &str,
    prompt_tokens: usize,
    publications: &mut P,
) -> Result<ReleaseOutcome, String> {
    let summary = publications.run_terminal_cleanup(|| session.abort_and_reset())?;
    validate_release_summary(
        &summary,
        request_id,
        prompt_tokens,
        publications.completion_tokens(),
        ReleaseOutcome::Cancelled,
    )?;
    publications.publish_released(ReleaseOutcome::Cancelled, None)?;
    Ok(ReleaseOutcome::Cancelled)
}

pub fn validate_release_summary(
    summary: &ReleaseSummary,
    request_id: &str,
    prompt_tokens: usize,
    generated_tokens: usize,
    outcome: ReleaseOutcome,
) -> Result<(), String> {
    if summary.request_id != request_id
        || summary.outcome != outcome
        || summary.prompt_tokens != prompt_tokens
        || summary.generated_tokens != generated_tokens
        || !summary.reset_complete
    {
        return Err(format!(
            "worker release summary does not match the completed request: {summary:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sq8_serving_runtime::Sq8ServingRequest;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct ScriptedSession {
        advances: VecDeque<SessionAdvance<usize>>,
        published: VecDeque<PublishedAdvance>,
        finish: Option<ReleaseSummary>,
        abort: Option<ReleaseSummary>,
        cancel: Option<CancellationToken>,
        trace: Arc<Mutex<Vec<&'static str>>>,
    }

    impl ScriptedSession {
        fn record(&self, event: &'static str) {
            self.trace.lock().unwrap().push(event);
        }
    }

    impl InferenceSession for ScriptedSession {
        type Prepared = usize;

        fn start_request(
            &mut self,
            _request: InferenceRequest,
            cancel: CancellationToken,
        ) -> Result<(), String> {
            self.record("start");
            self.cancel = Some(cancel);
            Ok(())
        }

        fn prepare_advance(&mut self) -> Result<SessionAdvance<Self::Prepared>, String> {
            self.record("prepare");
            self.advances
                .pop_front()
                .ok_or_else(|| "scripted advances exhausted".into())
        }

        fn publish_prepared<F>(
            &mut self,
            prepared: Self::Prepared,
            publish: F,
        ) -> Result<PublishedAdvance, String>
        where
            F: FnOnce(usize) -> Result<(), String>,
        {
            self.record("publish");
            if self
                .cancel
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
            {
                return Ok(PublishedAdvance::CancellationObserved);
            }
            let advance = self
                .published
                .pop_front()
                .ok_or_else(|| "scripted publications exhausted".to_string())?;
            if let PublishedAdvance::Token {
                token_id,
                generated_index,
                ..
            } = advance
            {
                if prepared != generated_index {
                    return Err("prepared handle differs from token index".into());
                }
                publish(token_id)?;
                Ok(PublishedAdvance::Token {
                    token_id,
                    generated_index,
                    cache_len: match prepared {
                        0 => 3,
                        _ => 4,
                    },
                    terminal_reason: (prepared == 1).then_some(FinishReason::Length),
                })
            } else {
                Ok(advance)
            }
        }

        fn finish_and_reset(&mut self) -> Result<ReleaseSummary, String> {
            self.record("finish_reset");
            self.finish.take().ok_or_else(|| "missing finish".into())
        }

        fn abort_and_reset(&mut self) -> Result<ReleaseSummary, String> {
            self.record("abort_reset");
            self.abort.take().ok_or_else(|| "missing abort".into())
        }
    }

    struct TracingPublications {
        trace: Arc<Mutex<Vec<&'static str>>>,
        completion_tokens: usize,
        outcome: Option<ReleaseOutcome>,
        timings: Option<GenerationTimings>,
    }

    impl RequestPublications for TracingPublications {
        fn publish_started(&mut self) -> Result<(), String> {
            self.trace.lock().unwrap().push("started");
            Ok(())
        }

        fn observe_prompt_unit(&mut self, _: usize, _: usize) -> Result<(), String> {
            self.trace.lock().unwrap().push("progress");
            Ok(())
        }

        fn observe_prefill_transition(&mut self) -> Result<(), String> {
            self.trace.lock().unwrap().push("prefill_transition");
            Ok(())
        }

        fn publish_token(&mut self, _: usize) -> Result<(), String> {
            self.trace.lock().unwrap().push("token");
            self.completion_tokens += 1;
            Ok(())
        }

        fn publish_released(
            &mut self,
            outcome: ReleaseOutcome,
            timings: Option<GenerationTimings>,
        ) -> Result<(), String> {
            self.trace.lock().unwrap().push("released");
            self.outcome = Some(outcome);
            self.timings = timings;
            Ok(())
        }

        fn run_terminal_cleanup<T, F>(&mut self, cleanup: F) -> Result<T, String>
        where
            F: FnOnce() -> Result<T, String>,
        {
            cleanup()
        }

        fn completion_tokens(&self) -> usize {
            self.completion_tokens
        }
    }

    #[test]
    fn timing_tracker_uses_first_to_last_sample_with_llama_server_counts() {
        let prompt_started_at = Instant::now();
        let mut tracker = RequestTimingTracker::new(prompt_started_at);
        tracker
            .observe_sample(prompt_started_at + Duration::from_millis(12))
            .unwrap();
        tracker
            .observe_sample(prompt_started_at + Duration::from_millis(20))
            .unwrap();
        let timings = tracker.finish(3, 2).unwrap();
        assert_eq!(timings.prompt_ms, 12.0);
        assert_eq!(timings.predicted_ms, 8.0);
        assert_eq!(timings.predicted_per_second, 250.0);
    }

    #[test]
    fn scripted_session_preserves_prepare_publish_commit_reset_release_order() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let mut session = ScriptedSession {
            advances: VecDeque::from([
                SessionAdvance::PromptProgress {
                    prompt_tokens_processed: 3,
                    cache_len: 3,
                    execution_width: 1,
                },
                SessionAdvance::Token {
                    prepared: 0,
                    token_id: 7,
                    generated_index: 0,
                    cache_len: 3,
                    terminal_reason: None,
                },
                SessionAdvance::Token {
                    prepared: 1,
                    token_id: 8,
                    generated_index: 1,
                    cache_len: 4,
                    terminal_reason: Some(FinishReason::Length),
                },
            ]),
            published: VecDeque::from([
                PublishedAdvance::Token {
                    token_id: 7,
                    generated_index: 0,
                    cache_len: 3,
                    terminal_reason: None,
                },
                PublishedAdvance::Token {
                    token_id: 8,
                    generated_index: 1,
                    cache_len: 4,
                    terminal_reason: Some(FinishReason::Length),
                },
            ]),
            finish: Some(ReleaseSummary {
                request_id: "req-order".into(),
                outcome: ReleaseOutcome::Length,
                prompt_tokens: 3,
                generated_tokens: 2,
                reset_complete: true,
            }),
            abort: None,
            cancel: None,
            trace: Arc::clone(&trace),
        };
        let mut publications = TracingPublications {
            trace: Arc::clone(&trace),
            completion_tokens: 0,
            outcome: None,
            timings: None,
        };
        let outcome = drive_worker_request(
            &mut session,
            Sq8ServingRequest::greedy("req-order", vec![1, 2, 3], 2),
            CancellationToken::new(),
            &mut publications,
        )
        .unwrap();
        assert_eq!(outcome, ReleaseOutcome::Length);
        assert_eq!(publications.outcome, Some(ReleaseOutcome::Length));
        assert!(publications.timings.is_some());
        assert_eq!(
            *trace.lock().unwrap(),
            [
                "start",
                "started",
                "prepare",
                "progress",
                "prepare",
                "prefill_transition",
                "publish",
                "token",
                "prepare",
                "publish",
                "token",
                "finish_reset",
                "released",
            ]
        );
    }

    #[test]
    fn cancel_after_prepare_skips_publish_callback_and_aborts_before_release() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let mut session = ScriptedSession {
            advances: VecDeque::from([SessionAdvance::Token {
                prepared: 0,
                token_id: 7,
                generated_index: 0,
                cache_len: 3,
                terminal_reason: None,
            }]),
            published: VecDeque::new(),
            finish: None,
            abort: Some(ReleaseSummary {
                request_id: "req-cancel".into(),
                outcome: ReleaseOutcome::Cancelled,
                prompt_tokens: 3,
                generated_tokens: 0,
                reset_complete: true,
            }),
            cancel: None,
            trace: Arc::clone(&trace),
        };
        let mut publications = TracingPublications {
            trace: Arc::clone(&trace),
            completion_tokens: 0,
            outcome: None,
            timings: None,
        };
        let outcome = drive_worker_request(
            &mut session,
            Sq8ServingRequest::greedy("req-cancel", vec![1, 2, 3], 2),
            cancel,
            &mut publications,
        )
        .unwrap();
        assert_eq!(outcome, ReleaseOutcome::Cancelled);
        assert_eq!(publications.completion_tokens, 0);
        assert!(publications.timings.is_none());
        assert_eq!(
            *trace.lock().unwrap(),
            [
                "start",
                "started",
                "prepare",
                "prefill_transition",
                "publish",
                "abort_reset",
                "released"
            ]
        );
    }
}
