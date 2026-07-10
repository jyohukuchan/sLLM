// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Resident Qwen3 SQ8 session backend for the JSONL worker.

use crate::sq_canonical::read_sq8_canonical_artifact;
use crate::sq8_embedding_runtime::QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV;
use crate::sq8_layer_runtime::{
    QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV,
    QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
};
use crate::sq8_model_head_runtime::{
    QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV, validate_qwen3_14b_sq8_r9700_device_info,
};
use crate::sq8_serving_runtime::{
    Qwen3Sq8ServingSession, Sq8CancellationToken, Sq8FinishReason, Sq8PreparedAdvance,
    Sq8PreparedToken, Sq8ReleaseOutcome, Sq8ReleaseSummary, Sq8ServingAdvance,
    Sq8ServingPrefillMode, Sq8ServingRequest, Sq8ServingRuntimeStatus,
    load_qwen3_14b_sq8_serving_norms,
};
use crate::sq8_worker_protocol::{Sq8ReleaseOutcomeEvent, Sq8WorkerAdmission};
use crate::sq8_worker_runtime::{Sq8InferenceBackend, Sq8RequestEventPublisher};
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use ullm_runtime_sys::{RuntimeContext, RuntimeStream, device_count, device_info};

pub const SQ8_WORKER_UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3Sq8WorkerBackendConfig {
    artifact: PathBuf,
    package: PathBuf,
}

impl Qwen3Sq8WorkerBackendConfig {
    pub fn new(artifact: impl Into<PathBuf>, package: impl Into<PathBuf>) -> Result<Self, String> {
        let config = Self {
            artifact: artifact.into(),
            package: package.into(),
        };
        if config.artifact.as_os_str().is_empty() || config.package.as_os_str().is_empty() {
            return Err("SQ8 worker artifact and package paths must be nonempty".into());
        }
        Ok(config)
    }

    pub fn artifact(&self) -> &Path {
        &self.artifact
    }

    pub fn package(&self) -> &Path {
        &self.package
    }
}

#[derive(Debug)]
pub struct Qwen3Sq8WorkerBackend {
    driver: Sq8SessionDriver<Qwen3Sq8SessionOwner>,
}

/// Field order keeps the session and stream alive until before the context is destroyed.
#[derive(Debug)]
struct Qwen3Sq8SessionOwner {
    session: Qwen3Sq8ServingSession,
    stream: RuntimeStream,
    _context: RuntimeContext,
}

impl Qwen3Sq8WorkerBackend {
    pub fn load(config: Qwen3Sq8WorkerBackendConfig) -> Result<Self, String> {
        require_sq8_worker_build_feature()?;
        require_sq8_worker_hip_guards()?;
        let artifact = read_sq8_canonical_artifact(config.artifact())?;
        let norms =
            load_qwen3_14b_sq8_serving_norms(config.package(), SQ8_WORKER_UPLOAD_CHUNK_BYTES)
                .map_err(|error| error.to_string())?;
        let runtime_index = isolated_sq8_worker_device()?;
        let mut context = RuntimeContext::create(runtime_index)?;
        let mut stream = context.create_stream()?;
        let session = Qwen3Sq8ServingSession::load_with_prefill_mode(
            &mut context,
            &mut stream,
            &artifact,
            config.package(),
            norms,
            SQ8_WORKER_UPLOAD_CHUNK_BYTES,
            Sq8ServingPrefillMode::FixedM128Chunks,
        )
        .map_err(|error| error.to_string())?;
        let owner = Qwen3Sq8SessionOwner {
            session,
            stream,
            _context: context,
        };
        Ok(Self {
            driver: Sq8SessionDriver { ops: owner },
        })
    }

    fn validate_ready_baseline(&mut self) -> Result<(), String> {
        let owner = &mut self.driver.ops;
        owner.stream.synchronize()?;
        let snapshot = owner.session.snapshot();
        if snapshot.status != Sq8ServingRuntimeStatus::Ready
            || snapshot.active_request_id.is_some()
            || snapshot.prompt_tokens != 0
            || snapshot.prompt_tokens_processed != 0
            || snapshot.generated_tokens != 0
            || snapshot.sampling_draws != 0
            || snapshot.token_prepared
            || snapshot.cache_lengths.iter().any(|length| *length != 0)
            || snapshot.scheduler_active != 0
            || snapshot.scheduler_waiting != 0
            || snapshot.allocator.allocated_blocks != 0
        {
            return Err(format!(
                "SQ8 worker backend is not at the reusable baseline: {snapshot:?}"
            ));
        }
        Ok(())
    }
}

fn require_sq8_worker_build_feature() -> Result<(), String> {
    if cfg!(feature = "rocm-ck-gfx1201") {
        Ok(())
    } else {
        Err("SQ8 worker binary requires the rocm-ck-gfx1201 build feature".into())
    }
}

impl Sq8InferenceBackend for Qwen3Sq8WorkerBackend {
    fn execute(
        &mut self,
        request: Sq8ServingRequest,
        admission: Sq8WorkerAdmission,
        publications: &mut Sq8RequestEventPublisher<'_>,
    ) -> Result<(), String> {
        let request_id = request.request_id.clone();
        let prompt_tokens = request.prompt_token_ids.len();
        let started = Instant::now();
        write_backend_log(Sq8BackendLog {
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
        let result = drive_sq8_worker_request(&mut self.driver, request, admission, publications);
        match result {
            Ok(outcome) => {
                write_backend_log(Sq8BackendLog {
                    schema_version: "ullm.worker.log.v1",
                    level: "info",
                    event: "request_released",
                    request_id: &request_id,
                    phase: "reset_complete",
                    prompt_tokens,
                    completion_tokens: publications.completion_tokens(),
                    elapsed_ms: elapsed_millis(started),
                    outcome: Some(release_event_name(outcome)),
                    error_code: None,
                });
                Ok(())
            }
            Err(error) => {
                write_backend_log(Sq8BackendLog {
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
        self.validate_ready_baseline()
    }
}

#[derive(Serialize)]
struct Sq8BackendLog<'a> {
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

fn write_backend_log(record: Sq8BackendLog<'_>) {
    let mut stderr = std::io::stderr().lock();
    let _ = serde_json::to_writer(&mut stderr, &record);
    let _ = stderr.write_all(b"\n");
    let _ = stderr.flush();
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn release_event_name(outcome: Sq8ReleaseOutcomeEvent) -> &'static str {
    match outcome {
        Sq8ReleaseOutcomeEvent::Stop => "stop",
        Sq8ReleaseOutcomeEvent::Length => "length",
        Sq8ReleaseOutcomeEvent::Cancelled => "cancelled",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DriverAdvance<P> {
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
        terminal_reason: Option<Sq8FinishReason>,
    },
    CancellationObserved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DriverPublished {
    Token {
        token_id: usize,
        generated_index: usize,
        cache_len: usize,
        terminal_reason: Option<Sq8FinishReason>,
    },
    CancellationObserved,
}

trait Sq8WorkerSessionDriver {
    type Prepared;

    fn start_request(
        &mut self,
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
    ) -> Result<(), String>;

    fn prepare_advance(&mut self) -> Result<DriverAdvance<Self::Prepared>, String>;

    fn publish_prepared<F>(
        &mut self,
        prepared: Self::Prepared,
        publish: F,
    ) -> Result<DriverPublished, String>
    where
        F: FnOnce(usize) -> Result<(), String>;

    fn finish_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String>;

    fn abort_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String>;
}

trait Sq8WorkerRequestPublications {
    fn publish_started(&mut self) -> Result<(), String>;

    fn observe_prompt_unit(
        &mut self,
        prompt_tokens_processed: usize,
        execution_width: usize,
    ) -> Result<(), String>;

    fn observe_prefill_transition(&mut self) -> Result<(), String>;

    fn publish_token(&mut self, token_id: usize) -> Result<(), String>;

    fn publish_released(&mut self, outcome: Sq8ReleaseOutcomeEvent) -> Result<(), String>;

    fn completion_tokens(&self) -> usize;
}

impl Sq8WorkerRequestPublications for Sq8RequestEventPublisher<'_> {
    fn publish_started(&mut self) -> Result<(), String> {
        Sq8RequestEventPublisher::publish_started(self)
    }

    fn observe_prompt_unit(
        &mut self,
        prompt_tokens_processed: usize,
        execution_width: usize,
    ) -> Result<(), String> {
        Sq8RequestEventPublisher::observe_prompt_unit(
            self,
            prompt_tokens_processed,
            execution_width,
        )
    }

    fn observe_prefill_transition(&mut self) -> Result<(), String> {
        Sq8RequestEventPublisher::observe_prefill_transition(self)
    }

    fn publish_token(&mut self, token_id: usize) -> Result<(), String> {
        Sq8RequestEventPublisher::publish_token(self, token_id)
    }

    fn publish_released(&mut self, outcome: Sq8ReleaseOutcomeEvent) -> Result<(), String> {
        Sq8RequestEventPublisher::publish_released(self, outcome)
    }

    fn completion_tokens(&self) -> usize {
        Sq8RequestEventPublisher::completion_tokens(self)
    }
}

trait Sq8SessionOps {
    fn start(
        &mut self,
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
    ) -> Result<(), String>;

    fn prepare(&mut self) -> Result<Sq8PreparedAdvance, String>;

    fn publish<F>(
        &mut self,
        prepared: Sq8PreparedToken,
        publish: F,
    ) -> Result<Sq8ServingAdvance, String>
    where
        F: FnOnce(&Sq8PreparedToken) -> Result<(), String>;

    fn finish_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String>;

    fn abort_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String>;
}

impl Sq8SessionOps for Qwen3Sq8SessionOwner {
    fn start(
        &mut self,
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
    ) -> Result<(), String> {
        self.session
            .start(request, cancel, &mut self.stream)
            .map_err(|error| error.to_string())
    }

    fn prepare(&mut self) -> Result<Sq8PreparedAdvance, String> {
        self.session
            .prepare_advance_synchronized(&mut self.stream)
            .map_err(|error| error.to_string())
    }

    fn publish<F>(
        &mut self,
        prepared: Sq8PreparedToken,
        publish: F,
    ) -> Result<Sq8ServingAdvance, String>
    where
        F: FnOnce(&Sq8PreparedToken) -> Result<(), String>,
    {
        self.session
            .publish_prepared_token(prepared, &mut self.stream, publish)
            .map_err(|error| error.to_string())
    }

    fn finish_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String> {
        self.session
            .finish_and_reset_synchronized(&mut self.stream)
            .map_err(|error| error.to_string())
    }

    fn abort_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String> {
        self.session
            .abort_and_reset_synchronized(&mut self.stream)
            .map_err(|error| error.to_string())
    }
}

#[derive(Debug)]
struct Sq8SessionDriver<O> {
    ops: O,
}

impl<O: Sq8SessionOps> Sq8WorkerSessionDriver for Sq8SessionDriver<O> {
    type Prepared = Sq8PreparedToken;

    fn start_request(
        &mut self,
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
    ) -> Result<(), String> {
        self.ops.start(request, cancel)
    }

    fn prepare_advance(&mut self) -> Result<DriverAdvance<Self::Prepared>, String> {
        match self.ops.prepare()? {
            Sq8PreparedAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            } => Ok(DriverAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            }),
            Sq8PreparedAdvance::Token(prepared) => Ok(DriverAdvance::Token {
                token_id: prepared.token_id,
                generated_index: prepared.generated_index,
                cache_len: prepared.cache_len,
                terminal_reason: prepared.terminal_reason,
                prepared,
            }),
            Sq8PreparedAdvance::CancellationObserved => Ok(DriverAdvance::CancellationObserved),
        }
    }

    fn publish_prepared<F>(
        &mut self,
        prepared: Self::Prepared,
        publish: F,
    ) -> Result<DriverPublished, String>
    where
        F: FnOnce(usize) -> Result<(), String>,
    {
        match self
            .ops
            .publish(prepared, |token| publish(token.token_id))?
        {
            Sq8ServingAdvance::Token {
                token_id,
                generated_index,
                cache_len,
                terminal_reason,
            } => Ok(DriverPublished::Token {
                token_id,
                generated_index,
                cache_len,
                terminal_reason,
            }),
            Sq8ServingAdvance::CancellationObserved => Ok(DriverPublished::CancellationObserved),
            Sq8ServingAdvance::PromptProgress { .. } => {
                Err("SQ8 prepared token publication returned prompt progress".into())
            }
        }
    }

    fn finish_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String> {
        self.ops.finish_and_reset()
    }

    fn abort_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String> {
        self.ops.abort_and_reset()
    }
}

fn drive_sq8_worker_request<D: Sq8WorkerSessionDriver, P: Sq8WorkerRequestPublications>(
    driver: &mut D,
    request: Sq8ServingRequest,
    admission: Sq8WorkerAdmission,
    publications: &mut P,
) -> Result<Sq8ReleaseOutcomeEvent, String> {
    let expected_request_id = request.request_id.clone();
    let expected_prompt_tokens = request.prompt_token_ids.len();
    driver.start_request(request, admission.cancel)?;
    publications.publish_started()?;

    loop {
        match driver.prepare_advance()? {
            DriverAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            } => {
                if cache_len != prompt_tokens_processed {
                    return Err("SQ8 prompt progress cache length is inconsistent".into());
                }
                publications.observe_prompt_unit(prompt_tokens_processed, execution_width)?;
            }
            DriverAdvance::Token {
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
                    return Err("SQ8 prepared token index does not match publication state".into());
                }
                let expected_cache_len = expected_prompt_tokens
                    .checked_add(generated_index)
                    .ok_or_else(|| "SQ8 prepared token cache length overflows".to_string())?;
                if cache_len != expected_cache_len {
                    return Err("SQ8 prepared token cache length is inconsistent".into());
                }
                match driver.publish_prepared(prepared, |published_token_id| {
                    if published_token_id != token_id {
                        return Err("SQ8 prepared token changed before publication".into());
                    }
                    publications.publish_token(published_token_id)
                })? {
                    DriverPublished::CancellationObserved => {
                        return finish_cancelled_request(
                            driver,
                            &expected_request_id,
                            expected_prompt_tokens,
                            publications,
                        );
                    }
                    DriverPublished::Token {
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
                                "SQ8 committed token does not match its prepared proposal".into()
                            );
                        }
                        if let Some(reason) = committed_terminal {
                            return finish_completed_request(
                                driver,
                                &expected_request_id,
                                expected_prompt_tokens,
                                reason,
                                publications,
                            );
                        }
                    }
                }
            }
            DriverAdvance::CancellationObserved => {
                return finish_cancelled_request(
                    driver,
                    &expected_request_id,
                    expected_prompt_tokens,
                    publications,
                );
            }
        }
    }
}

fn finish_completed_request<D: Sq8WorkerSessionDriver, P: Sq8WorkerRequestPublications>(
    driver: &mut D,
    request_id: &str,
    prompt_tokens: usize,
    reason: Sq8FinishReason,
    publications: &mut P,
) -> Result<Sq8ReleaseOutcomeEvent, String> {
    let summary = driver.finish_and_reset()?;
    let (expected_outcome, event_outcome) = match reason {
        Sq8FinishReason::Stop => (Sq8ReleaseOutcome::Stop, Sq8ReleaseOutcomeEvent::Stop),
        Sq8FinishReason::Length => (Sq8ReleaseOutcome::Length, Sq8ReleaseOutcomeEvent::Length),
    };
    validate_release_summary(
        &summary,
        request_id,
        prompt_tokens,
        publications.completion_tokens(),
        expected_outcome,
    )?;
    publications.publish_released(event_outcome)?;
    Ok(event_outcome)
}

fn finish_cancelled_request<D: Sq8WorkerSessionDriver, P: Sq8WorkerRequestPublications>(
    driver: &mut D,
    request_id: &str,
    prompt_tokens: usize,
    publications: &mut P,
) -> Result<Sq8ReleaseOutcomeEvent, String> {
    let summary = driver.abort_and_reset()?;
    validate_release_summary(
        &summary,
        request_id,
        prompt_tokens,
        publications.completion_tokens(),
        Sq8ReleaseOutcome::Cancelled,
    )?;
    publications.publish_released(Sq8ReleaseOutcomeEvent::Cancelled)?;
    Ok(Sq8ReleaseOutcomeEvent::Cancelled)
}

fn validate_release_summary(
    summary: &Sq8ReleaseSummary,
    request_id: &str,
    prompt_tokens: usize,
    generated_tokens: usize,
    outcome: Sq8ReleaseOutcome,
) -> Result<(), String> {
    if summary.request_id != request_id
        || summary.outcome != outcome
        || summary.prompt_tokens != prompt_tokens
        || summary.generated_tokens != generated_tokens
        || !summary.reset_complete
    {
        return Err(format!(
            "SQ8 worker release summary does not match the completed request: {summary:?}"
        ));
    }
    Ok(())
}

fn require_sq8_worker_hip_guards() -> Result<(), String> {
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV)
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    let invalid = names
        .into_iter()
        .filter(|name| std::env::var(name).ok().as_deref() != Some("1"))
        .collect::<Vec<_>>();
    if invalid.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "SQ8 worker requires these HIP guards to equal 1: {}",
            invalid.join(",")
        ))
    }
}

fn isolated_sq8_worker_device() -> Result<u32, String> {
    let mut devices = Vec::new();
    for index in 0..device_count()? {
        let info = device_info(index)
            .map_err(|error| format!("failed to inspect runtime device {index}: {error}"))?;
        if info.backend == "hip" {
            devices.push((index, info));
        }
    }
    if devices.len() != 1 {
        return Err(format!(
            "SQ8 worker requires exactly one visible HIP device, found {}",
            devices.len()
        ));
    }
    let (runtime_index, device) = devices.pop().expect("one device was checked");
    validate_qwen3_14b_sq8_r9700_device_info(&device)?;
    if device.device_id != 0 {
        return Err(format!(
            "SQ8 worker requires isolated HIP device 0, got {}",
            device.device_id
        ));
    }
    Ok(runtime_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sq8_worker_protocol::{
        Sq8CancelReason, Sq8OrderedJsonlWriter, Sq8WorkerControl, Sq8WorkerEvent,
    };
    use crate::sq8_worker_runtime::{
        Sq8InferenceCommand, Sq8WorkerEventPublisher, spawn_sq8_inference_thread,
        spawn_sq8_ordered_writer,
    };
    use serde_json::Value;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Receiver, SyncSender, sync_channel};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct ScriptedPublishBarrier {
        prepared: mpsc::Sender<()>,
        release: Receiver<()>,
    }

    struct ScriptedDriver {
        advances: VecDeque<DriverAdvance<usize>>,
        published: VecDeque<DriverPublished>,
        finish: Option<Sq8ReleaseSummary>,
        abort: Option<Sq8ReleaseSummary>,
        cancel: Option<Sq8CancellationToken>,
        publish_barrier: Option<ScriptedPublishBarrier>,
        callback_calls: Arc<AtomicUsize>,
        trace: Option<Arc<Mutex<Vec<&'static str>>>>,
    }

    impl ScriptedDriver {
        fn record(&self, event: &'static str) {
            if let Some(trace) = &self.trace {
                trace.lock().unwrap().push(event);
            }
        }
    }

    impl Sq8WorkerSessionDriver for ScriptedDriver {
        type Prepared = usize;

        fn start_request(
            &mut self,
            request: Sq8ServingRequest,
            cancel: Sq8CancellationToken,
        ) -> Result<(), String> {
            request.validate().map_err(|error| error.to_string())?;
            self.record("start_request");
            self.cancel = Some(cancel);
            Ok(())
        }

        fn prepare_advance(&mut self) -> Result<DriverAdvance<Self::Prepared>, String> {
            self.advances
                .pop_front()
                .ok_or_else(|| "scripted driver exhausted advances".to_string())
        }

        fn publish_prepared<F>(
            &mut self,
            prepared: Self::Prepared,
            publish: F,
        ) -> Result<DriverPublished, String>
        where
            F: FnOnce(usize) -> Result<(), String>,
        {
            self.record("publish_prepared");
            if let Some(barrier) = self.publish_barrier.take() {
                barrier
                    .prepared
                    .send(())
                    .map_err(|_| "scripted prepare barrier receiver closed".to_string())?;
                barrier
                    .release
                    .recv()
                    .map_err(|_| "scripted prepare barrier sender closed".to_string())?;
            }
            if self
                .cancel
                .as_ref()
                .is_some_and(Sq8CancellationToken::is_cancelled)
            {
                return Ok(DriverPublished::CancellationObserved);
            }
            let result = self
                .published
                .pop_front()
                .ok_or_else(|| "scripted driver exhausted publications".to_string())?;
            if let DriverPublished::Token {
                token_id,
                generated_index,
                ..
            } = &result
            {
                if prepared != *generated_index {
                    return Err("scripted prepared handle does not match its index".into());
                }
                self.callback_calls.fetch_add(1, Ordering::SeqCst);
                publish(*token_id)?;
            }
            Ok(result)
        }

        fn finish_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String> {
            self.record("finish_reset");
            self.finish
                .take()
                .ok_or_else(|| "scripted finish summary is missing".to_string())
        }

        fn abort_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String> {
            self.record("abort_reset");
            self.abort
                .take()
                .ok_or_else(|| "scripted abort summary is missing".to_string())
        }
    }

    struct TraceSessionOps {
        advances: VecDeque<Sq8PreparedAdvance>,
        published: VecDeque<Sq8ServingAdvance>,
        finish: Option<Sq8ReleaseSummary>,
        abort: Option<Sq8ReleaseSummary>,
        cancel: Option<Sq8CancellationToken>,
        callback_calls: Arc<AtomicUsize>,
        trace: Arc<Mutex<Vec<&'static str>>>,
    }

    impl TraceSessionOps {
        fn record(&self, event: &'static str) {
            self.trace.lock().unwrap().push(event);
        }
    }

    impl Sq8SessionOps for TraceSessionOps {
        fn start(
            &mut self,
            request: Sq8ServingRequest,
            cancel: Sq8CancellationToken,
        ) -> Result<(), String> {
            request.validate().map_err(|error| error.to_string())?;
            self.record("ops_start");
            self.cancel = Some(cancel);
            Ok(())
        }

        fn prepare(&mut self) -> Result<Sq8PreparedAdvance, String> {
            let advance = self
                .advances
                .pop_front()
                .ok_or_else(|| "trace session exhausted advances".to_string())?;
            self.record(match &advance {
                Sq8PreparedAdvance::PromptProgress { .. } => "ops_prepare_prompt",
                Sq8PreparedAdvance::Token(_) => "ops_prepare_token",
                Sq8PreparedAdvance::CancellationObserved => "ops_prepare_cancelled",
            });
            Ok(advance)
        }

        fn publish<F>(
            &mut self,
            prepared: Sq8PreparedToken,
            publish: F,
        ) -> Result<Sq8ServingAdvance, String>
        where
            F: FnOnce(&Sq8PreparedToken) -> Result<(), String>,
        {
            self.record("ops_publish");
            if self
                .cancel
                .as_ref()
                .is_some_and(Sq8CancellationToken::is_cancelled)
            {
                return Ok(Sq8ServingAdvance::CancellationObserved);
            }
            let advance = self
                .published
                .pop_front()
                .ok_or_else(|| "trace session exhausted publications".to_string())?;
            if let Sq8ServingAdvance::Token {
                token_id,
                generated_index,
                cache_len,
                terminal_reason,
            } = advance
            {
                if token_id != prepared.token_id
                    || generated_index != prepared.generated_index
                    || cache_len != prepared.cache_len
                    || terminal_reason != prepared.terminal_reason
                {
                    return Err("trace session publication differs from prepared token".into());
                }
                self.record("ops_callback");
                self.callback_calls.fetch_add(1, Ordering::SeqCst);
                publish(&prepared)?;
                Ok(Sq8ServingAdvance::Token {
                    token_id,
                    generated_index,
                    cache_len,
                    terminal_reason,
                })
            } else {
                Ok(advance)
            }
        }

        fn finish_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String> {
            self.record("ops_finish_reset");
            self.finish
                .take()
                .ok_or_else(|| "trace session finish summary is missing".to_string())
        }

        fn abort_and_reset(&mut self) -> Result<Sq8ReleaseSummary, String> {
            self.record("ops_abort_reset");
            self.abort
                .take()
                .ok_or_else(|| "trace session abort summary is missing".to_string())
        }
    }

    struct TracingPublications {
        trace: Arc<Mutex<Vec<&'static str>>>,
        completion_tokens: usize,
        tokens: Vec<usize>,
    }

    impl TracingPublications {
        fn record(&self, event: &'static str) {
            self.trace.lock().unwrap().push(event);
        }
    }

    impl Sq8WorkerRequestPublications for TracingPublications {
        fn publish_started(&mut self) -> Result<(), String> {
            self.record("started");
            Ok(())
        }

        fn observe_prompt_unit(
            &mut self,
            _prompt_tokens_processed: usize,
            _execution_width: usize,
        ) -> Result<(), String> {
            self.record("prompt_progress");
            Ok(())
        }

        fn observe_prefill_transition(&mut self) -> Result<(), String> {
            self.record("prefill_transition");
            Ok(())
        }

        fn publish_token(&mut self, token_id: usize) -> Result<(), String> {
            self.record("token");
            self.tokens.push(token_id);
            self.completion_tokens += 1;
            Ok(())
        }

        fn publish_released(&mut self, _outcome: Sq8ReleaseOutcomeEvent) -> Result<(), String> {
            self.record("released");
            Ok(())
        }

        fn completion_tokens(&self) -> usize {
            self.completion_tokens
        }
    }

    fn ready_control() -> Sq8WorkerControl {
        let control = Sq8WorkerControl::new();
        let acknowledgement = Sq8OrderedJsonlWriter::new(Vec::new())
            .write_ready_event(&Sq8WorkerEvent::ready())
            .unwrap();
        control.mark_ready_after_flush(acknowledgement).unwrap();
        control
    }

    struct ScriptedBackend {
        driver: ScriptedDriver,
        completed: mpsc::Sender<Result<(), String>>,
    }

    impl Sq8InferenceBackend for ScriptedBackend {
        fn execute(
            &mut self,
            request: Sq8ServingRequest,
            admission: Sq8WorkerAdmission,
            publications: &mut Sq8RequestEventPublisher<'_>,
        ) -> Result<(), String> {
            let result =
                drive_sq8_worker_request(&mut self.driver, request, admission, publications)
                    .map(|_| ());
            let _ = self.completed.send(result.clone());
            result
        }
    }

    fn start_scripted_backend(
        driver: ScriptedDriver,
    ) -> (
        Arc<Sq8WorkerControl>,
        Sq8WorkerEventPublisher,
        crate::sq8_worker_runtime::Sq8WriterThread<Vec<u8>>,
        SyncSender<Sq8InferenceCommand>,
        crate::sq8_worker_runtime::Sq8InferenceThread,
        Receiver<Result<(), String>>,
    ) {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let (completed, completion) = mpsc::channel();
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || Ok(ScriptedBackend { driver, completed }),
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        (control, events, writer, commands, inference, completion)
    }

    fn finish_scripted_backend(
        control: &Sq8WorkerControl,
        commands: SyncSender<Sq8InferenceCommand>,
        inference: crate::sq8_worker_runtime::Sq8InferenceThread,
        events: Sq8WorkerEventPublisher,
        writer: crate::sq8_worker_runtime::Sq8WriterThread<Vec<u8>>,
    ) -> Vec<Value> {
        assert_eq!(
            control.begin_shutdown().unwrap(),
            crate::sq8_worker_protocol::Sq8WorkerShutdownDisposition::Idle
        );
        commands.send(Sq8InferenceCommand::Shutdown).unwrap();
        inference.join().unwrap();
        let bytes = writer.close_and_join().unwrap();
        drop(events);
        bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect::<Vec<_>>()
    }

    #[test]
    fn scripted_driver_publishes_two_tokens_then_resets_before_release() {
        let driver = ScriptedDriver {
            advances: VecDeque::from([
                DriverAdvance::PromptProgress {
                    prompt_tokens_processed: 1,
                    cache_len: 1,
                    execution_width: 1,
                },
                DriverAdvance::PromptProgress {
                    prompt_tokens_processed: 2,
                    cache_len: 2,
                    execution_width: 1,
                },
                DriverAdvance::Token {
                    prepared: 0,
                    token_id: 7,
                    generated_index: 0,
                    cache_len: 3,
                    terminal_reason: None,
                },
                DriverAdvance::Token {
                    prepared: 1,
                    token_id: 8,
                    generated_index: 1,
                    cache_len: 4,
                    terminal_reason: Some(Sq8FinishReason::Length),
                },
            ]),
            published: VecDeque::from([
                DriverPublished::Token {
                    token_id: 7,
                    generated_index: 0,
                    cache_len: 3,
                    terminal_reason: None,
                },
                DriverPublished::Token {
                    token_id: 8,
                    generated_index: 1,
                    cache_len: 4,
                    terminal_reason: Some(Sq8FinishReason::Length),
                },
            ]),
            finish: Some(Sq8ReleaseSummary {
                request_id: "req-two".into(),
                outcome: Sq8ReleaseOutcome::Length,
                prompt_tokens: 3,
                generated_tokens: 2,
                reset_complete: true,
            }),
            abort: None,
            cancel: None,
            publish_barrier: None,
            callback_calls: Arc::new(AtomicUsize::new(0)),
            trace: None,
        };
        let (control, events, writer, commands, inference, completion) =
            start_scripted_backend(driver);
        let request = Sq8ServingRequest::greedy("req-two", vec![1, 2, 3], 2);
        let admission = control.admit("req-two").unwrap();
        commands
            .send(Sq8InferenceCommand::Generate { request, admission })
            .unwrap();
        completion
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        let lines = finish_scripted_backend(&control, commands, inference, events, writer);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "started", "progress", "token", "token", "released"]
        );
        assert_eq!(lines[3]["index"], 0);
        assert_eq!(lines[4]["index"], 1);
        assert_eq!(lines[5]["outcome"], "length");
    }

    #[test]
    fn scripted_driver_cancel_after_prepare_skips_token_callback_and_aborts() {
        let (prepared, prepared_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let callback_calls = Arc::new(AtomicUsize::new(0));
        let driver = ScriptedDriver {
            advances: VecDeque::from([
                DriverAdvance::PromptProgress {
                    prompt_tokens_processed: 1,
                    cache_len: 1,
                    execution_width: 1,
                },
                DriverAdvance::PromptProgress {
                    prompt_tokens_processed: 2,
                    cache_len: 2,
                    execution_width: 1,
                },
                DriverAdvance::Token {
                    prepared: 0,
                    token_id: 7,
                    generated_index: 0,
                    cache_len: 3,
                    terminal_reason: None,
                },
            ]),
            published: VecDeque::new(),
            finish: None,
            abort: Some(Sq8ReleaseSummary {
                request_id: "req-cancel".into(),
                outcome: Sq8ReleaseOutcome::Cancelled,
                prompt_tokens: 3,
                generated_tokens: 0,
                reset_complete: true,
            }),
            cancel: None,
            publish_barrier: Some(ScriptedPublishBarrier {
                prepared,
                release: release_rx,
            }),
            callback_calls: Arc::clone(&callback_calls),
            trace: None,
        };
        let (control, events, writer, commands, inference, completion) =
            start_scripted_backend(driver);
        let request = Sq8ServingRequest::greedy("req-cancel", vec![1, 2, 3], 2);
        let admission = control.admit("req-cancel").unwrap();
        commands
            .send(Sq8InferenceCommand::Generate { request, admission })
            .unwrap();
        prepared_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        control
            .cancel("req-cancel", Sq8CancelReason::Operator)
            .unwrap();
        release.send(()).unwrap();
        completion
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        let lines = finish_scripted_backend(&control, commands, inference, events, writer);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "started", "progress", "released"]
        );
        assert_eq!(lines[3]["outcome"], "cancelled");
        assert_eq!(lines[3]["cancel_reason"], "operator");
        assert_eq!(callback_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn generic_session_adapter_maps_two_tokens_and_finish_cleanup() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let callback_calls = Arc::new(AtomicUsize::new(0));
        let prepared_zero = Sq8PreparedToken::for_worker_test(7, 0, 3, None);
        let prepared_one =
            Sq8PreparedToken::for_worker_test(8, 1, 4, Some(Sq8FinishReason::Length));
        let ops = TraceSessionOps {
            advances: VecDeque::from([
                Sq8PreparedAdvance::PromptProgress {
                    prompt_tokens_processed: 3,
                    cache_len: 3,
                    execution_width: 1,
                },
                Sq8PreparedAdvance::Token(prepared_zero),
                Sq8PreparedAdvance::Token(prepared_one),
            ]),
            published: VecDeque::from([
                Sq8ServingAdvance::Token {
                    token_id: 7,
                    generated_index: 0,
                    cache_len: 3,
                    terminal_reason: None,
                },
                Sq8ServingAdvance::Token {
                    token_id: 8,
                    generated_index: 1,
                    cache_len: 4,
                    terminal_reason: Some(Sq8FinishReason::Length),
                },
            ]),
            finish: Some(Sq8ReleaseSummary {
                request_id: "req-adapter".into(),
                outcome: Sq8ReleaseOutcome::Length,
                prompt_tokens: 3,
                generated_tokens: 2,
                reset_complete: true,
            }),
            abort: None,
            cancel: None,
            callback_calls: Arc::clone(&callback_calls),
            trace: Arc::clone(&trace),
        };
        let mut driver = Sq8SessionDriver { ops };
        let control = ready_control();
        let admission = control.admit("req-adapter").unwrap();
        let mut publications = TracingPublications {
            trace: Arc::clone(&trace),
            completion_tokens: 0,
            tokens: Vec::new(),
        };

        let outcome = drive_sq8_worker_request(
            &mut driver,
            Sq8ServingRequest::greedy("req-adapter", vec![1, 2, 3], 2),
            admission,
            &mut publications,
        )
        .unwrap();

        assert_eq!(outcome, Sq8ReleaseOutcomeEvent::Length);
        assert_eq!(publications.tokens, [7, 8]);
        assert_eq!(callback_calls.load(Ordering::SeqCst), 2);
        assert!(!driver.ops.cancel.as_ref().unwrap().is_cancelled());
        assert_eq!(
            *trace.lock().unwrap(),
            [
                "ops_start",
                "started",
                "ops_prepare_prompt",
                "prompt_progress",
                "ops_prepare_token",
                "prefill_transition",
                "ops_publish",
                "ops_callback",
                "token",
                "ops_prepare_token",
                "ops_publish",
                "ops_callback",
                "token",
                "ops_finish_reset",
                "released",
            ]
        );
    }

    #[test]
    fn generic_session_adapter_forwards_cancel_and_maps_abort_cleanup() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let callback_calls = Arc::new(AtomicUsize::new(0));
        let ops = TraceSessionOps {
            advances: VecDeque::from([
                Sq8PreparedAdvance::PromptProgress {
                    prompt_tokens_processed: 3,
                    cache_len: 3,
                    execution_width: 1,
                },
                Sq8PreparedAdvance::Token(Sq8PreparedToken::for_worker_test(7, 0, 3, None)),
            ]),
            published: VecDeque::new(),
            finish: None,
            abort: Some(Sq8ReleaseSummary {
                request_id: "req-adapter-cancel".into(),
                outcome: Sq8ReleaseOutcome::Cancelled,
                prompt_tokens: 3,
                generated_tokens: 0,
                reset_complete: true,
            }),
            cancel: None,
            callback_calls: Arc::clone(&callback_calls),
            trace: Arc::clone(&trace),
        };
        let mut driver = Sq8SessionDriver { ops };
        let control = ready_control();
        let admission = control.admit("req-adapter-cancel").unwrap();
        control
            .cancel("req-adapter-cancel", Sq8CancelReason::Operator)
            .unwrap();
        let mut publications = TracingPublications {
            trace: Arc::clone(&trace),
            completion_tokens: 0,
            tokens: Vec::new(),
        };

        let outcome = drive_sq8_worker_request(
            &mut driver,
            Sq8ServingRequest::greedy("req-adapter-cancel", vec![1, 2, 3], 2),
            admission,
            &mut publications,
        )
        .unwrap();

        assert_eq!(outcome, Sq8ReleaseOutcomeEvent::Cancelled);
        assert!(driver.ops.cancel.as_ref().unwrap().is_cancelled());
        assert_eq!(callback_calls.load(Ordering::SeqCst), 0);
        assert!(publications.tokens.is_empty());
        assert_eq!(
            *trace.lock().unwrap(),
            [
                "ops_start",
                "started",
                "ops_prepare_prompt",
                "prompt_progress",
                "ops_prepare_token",
                "prefill_transition",
                "ops_publish",
                "ops_abort_reset",
                "released",
            ]
        );
    }

    #[test]
    fn stop_release_follows_finish_reset_and_maps_eos_outcome() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let callback_calls = Arc::new(AtomicUsize::new(0));
        let mut driver = ScriptedDriver {
            advances: VecDeque::from([
                DriverAdvance::PromptProgress {
                    prompt_tokens_processed: 3,
                    cache_len: 3,
                    execution_width: 1,
                },
                DriverAdvance::Token {
                    prepared: 0,
                    token_id: 151_645,
                    generated_index: 0,
                    cache_len: 3,
                    terminal_reason: Some(Sq8FinishReason::Stop),
                },
            ]),
            published: VecDeque::from([DriverPublished::Token {
                token_id: 151_645,
                generated_index: 0,
                cache_len: 3,
                terminal_reason: Some(Sq8FinishReason::Stop),
            }]),
            finish: Some(Sq8ReleaseSummary {
                request_id: "req-stop".into(),
                outcome: Sq8ReleaseOutcome::Stop,
                prompt_tokens: 3,
                generated_tokens: 1,
                reset_complete: true,
            }),
            abort: None,
            cancel: None,
            publish_barrier: None,
            callback_calls: Arc::clone(&callback_calls),
            trace: Some(Arc::clone(&trace)),
        };
        let control = ready_control();
        let admission = control.admit("req-stop").unwrap();
        let mut publications = TracingPublications {
            trace: Arc::clone(&trace),
            completion_tokens: 0,
            tokens: Vec::new(),
        };

        let outcome = drive_sq8_worker_request(
            &mut driver,
            Sq8ServingRequest::greedy("req-stop", vec![1, 2, 3], 2),
            admission,
            &mut publications,
        )
        .unwrap();

        assert_eq!(outcome, Sq8ReleaseOutcomeEvent::Stop);
        assert_eq!(publications.completion_tokens(), 1);
        assert_eq!(callback_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            *trace.lock().unwrap(),
            [
                "start_request",
                "started",
                "prompt_progress",
                "prefill_transition",
                "publish_prepared",
                "token",
                "finish_reset",
                "released",
            ]
        );
    }

    #[test]
    fn cancelled_release_follows_abort_reset_without_token_publication() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let callback_calls = Arc::new(AtomicUsize::new(0));
        let mut driver = ScriptedDriver {
            advances: VecDeque::from([
                DriverAdvance::PromptProgress {
                    prompt_tokens_processed: 3,
                    cache_len: 3,
                    execution_width: 1,
                },
                DriverAdvance::Token {
                    prepared: 0,
                    token_id: 7,
                    generated_index: 0,
                    cache_len: 3,
                    terminal_reason: None,
                },
            ]),
            published: VecDeque::new(),
            finish: None,
            abort: Some(Sq8ReleaseSummary {
                request_id: "req-cancel-order".into(),
                outcome: Sq8ReleaseOutcome::Cancelled,
                prompt_tokens: 3,
                generated_tokens: 0,
                reset_complete: true,
            }),
            cancel: None,
            publish_barrier: None,
            callback_calls: Arc::clone(&callback_calls),
            trace: Some(Arc::clone(&trace)),
        };
        let control = ready_control();
        let admission = control.admit("req-cancel-order").unwrap();
        control
            .cancel("req-cancel-order", Sq8CancelReason::Operator)
            .unwrap();
        let mut publications = TracingPublications {
            trace: Arc::clone(&trace),
            completion_tokens: 0,
            tokens: Vec::new(),
        };

        let outcome = drive_sq8_worker_request(
            &mut driver,
            Sq8ServingRequest::greedy("req-cancel-order", vec![1, 2, 3], 2),
            admission,
            &mut publications,
        )
        .unwrap();

        assert_eq!(outcome, Sq8ReleaseOutcomeEvent::Cancelled);
        assert_eq!(publications.completion_tokens(), 0);
        assert_eq!(callback_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            *trace.lock().unwrap(),
            [
                "start_request",
                "started",
                "prompt_progress",
                "prefill_transition",
                "publish_prepared",
                "abort_reset",
                "released",
            ]
        );
    }

    #[test]
    fn release_summary_validation_is_exact() {
        let summary = Sq8ReleaseSummary {
            request_id: "req-1".into(),
            outcome: Sq8ReleaseOutcome::Stop,
            prompt_tokens: 3,
            generated_tokens: 1,
            reset_complete: true,
        };
        validate_release_summary(&summary, "req-1", 3, 1, Sq8ReleaseOutcome::Stop).unwrap();
        for invalid in [
            Sq8ReleaseSummary {
                request_id: "wrong".into(),
                ..summary.clone()
            },
            Sq8ReleaseSummary {
                outcome: Sq8ReleaseOutcome::Length,
                ..summary.clone()
            },
            Sq8ReleaseSummary {
                prompt_tokens: 2,
                ..summary.clone()
            },
            Sq8ReleaseSummary {
                generated_tokens: 0,
                ..summary.clone()
            },
            Sq8ReleaseSummary {
                reset_complete: false,
                ..summary.clone()
            },
        ] {
            assert!(
                validate_release_summary(&invalid, "req-1", 3, 1, Sq8ReleaseOutcome::Stop).is_err()
            );
        }
    }

    #[test]
    fn backend_config_rejects_empty_paths_and_freezes_inputs() {
        assert!(Qwen3Sq8WorkerBackendConfig::new("", "package").is_err());
        assert!(Qwen3Sq8WorkerBackendConfig::new("artifact", "").is_err());
        let config = Qwen3Sq8WorkerBackendConfig::new("artifact", "package").unwrap();
        assert_eq!(config.artifact(), Path::new("artifact"));
        assert_eq!(config.package(), Path::new("package"));
        assert_eq!(SQ8_WORKER_UPLOAD_CHUNK_BYTES, 16 * 1024 * 1024);
        assert_eq!(
            require_sq8_worker_build_feature().is_ok(),
            cfg!(feature = "rocm-ck-gfx1201")
        );
    }

    #[test]
    fn structured_backend_log_contains_counts_but_no_prompt_or_token_content() {
        let value = serde_json::to_value(Sq8BackendLog {
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
