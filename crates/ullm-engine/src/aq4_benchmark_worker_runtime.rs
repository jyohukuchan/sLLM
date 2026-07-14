// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Resident CPU-side topology for the AQ4 production-server benchmark wire.

use crate::aq4_benchmark_worker_protocol::{
    AQ4_BENCHMARK_TERMINAL_EVIDENCE_SCHEMA_VERSION, AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
    Aq4BenchmarkExecutionEvidence, Aq4BenchmarkPrefillCommand, Aq4BenchmarkReuse,
    Aq4BenchmarkTerminalStatus, Aq4BenchmarkWorkerCommand, Aq4BenchmarkWorkerEvent,
    decode_aq4_benchmark_worker_command,
};
use crate::inference_api::CancellationToken;
use crate::qwen35_aq4_session::{Qwen35Aq4InferenceSession, Qwen35Aq4SessionModel};
use crate::session_worker_backend::SessionInferenceBackend;
use crate::sq8_worker_protocol::{Sq8BoundedJsonlReader, Sq8JsonlRead, Sq8WorkerProfile};
use crate::worker_driver::{InferenceSession, SessionAdvance};
use std::io::{BufRead, Write};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aq4BenchmarkCommandReaderExit {
    IdleShutdown,
    ActiveShutdown,
}

pub trait Aq4BenchmarkProgressPublisher {
    fn publish_prompt_progress(&mut self, processed_prompt_tokens: usize) -> Result<(), String>;
}

pub trait Aq4BenchmarkInferenceBackend: 'static {
    fn execute_benchmark_prefill(
        &mut self,
        command: &Aq4BenchmarkPrefillCommand,
        cancel: CancellationToken,
        progress: &mut dyn Aq4BenchmarkProgressPublisher,
    ) -> Aq4BenchmarkExecutionEvidence;

    fn shutdown(&mut self) -> Result<(), String> {
        Ok(())
    }
}

impl<M> Aq4BenchmarkInferenceBackend for SessionInferenceBackend<Qwen35Aq4InferenceSession<M>>
where
    M: Qwen35Aq4SessionModel + 'static,
{
    fn execute_benchmark_prefill(
        &mut self,
        command: &Aq4BenchmarkPrefillCommand,
        cancel: CancellationToken,
        progress: &mut dyn Aq4BenchmarkProgressPublisher,
    ) -> Aq4BenchmarkExecutionEvidence {
        let profile = benchmark_profile_for_session(self.session());
        let request = command.into_inference_request(&profile, i64::from(command.run_index));
        if let Err(error) = self.session_mut().start_benchmark_prefill_request(
            request,
            cancel,
            command.requested_m,
            command.resolved_m,
        ) {
            return session_failure_evidence(command, self.session(), &error, false);
        }
        loop {
            match self.session_mut().prepare_advance() {
                Ok(SessionAdvance::PromptProgress {
                    prompt_tokens_processed,
                    ..
                }) => {
                    if let Err(error) = progress.publish_prompt_progress(prompt_tokens_processed) {
                        let reset_ok = self.session_mut().abort_and_reset().is_ok();
                        return session_failure_evidence(command, self.session(), &error, reset_ok);
                    }
                    if prompt_tokens_processed == command.prompt_token_ids.len() {
                        return match self.session_mut().finish_benchmark_prefill_and_reset() {
                            Ok(_) => session_evidence(
                                command,
                                self.session(),
                                Aq4BenchmarkTerminalStatus::Ok,
                                Aq4BenchmarkReuse::Allowed,
                                None,
                            ),
                            Err(error) => {
                                session_failure_evidence(command, self.session(), &error, false)
                            }
                        };
                    }
                }
                Ok(SessionAdvance::CancellationObserved) => {
                    return match self.session_mut().abort_and_reset() {
                        Ok(_) => session_evidence(
                            command,
                            self.session(),
                            Aq4BenchmarkTerminalStatus::Cancelled,
                            Aq4BenchmarkReuse::Allowed,
                            None,
                        ),
                        Err(error) => {
                            session_failure_evidence(command, self.session(), &error, false)
                        }
                    };
                }
                Ok(SessionAdvance::Token { .. }) => {
                    let reset_ok = self.session_mut().abort_and_reset().is_ok();
                    return session_failure_evidence(
                        command,
                        self.session(),
                        "prefill-only session attempted token publication",
                        reset_ok,
                    );
                }
                Err(error) => {
                    let reset_ok = self.session_mut().abort_and_reset().is_ok();
                    return session_failure_evidence(command, self.session(), &error, reset_ok);
                }
            }
        }
    }

    fn shutdown(&mut self) -> Result<(), String> {
        self.session_mut().shutdown()
    }
}

fn benchmark_profile_for_session<M: Qwen35Aq4SessionModel>(
    session: &Qwen35Aq4InferenceSession<M>,
) -> Sq8WorkerProfile {
    Sq8WorkerProfile {
        worker_schema: AQ4_BENCHMARK_WORKER_SCHEMA_VERSION.into(),
        model: "aq4-benchmark-resident".into(),
        model_revision: "request-scoped-m-v1".into(),
        artifact_content_sha256: "0".repeat(64),
        package_manifest_sha256: "0".repeat(64),
        device: "benchmark".into(),
        execution_profile: "aq4-p2-prefill-only".into(),
        context_length: session.model().context_length(),
        max_new_tokens: 1,
        vocab_size: session.model().vocab_size(),
        eos_token_ids: session.config().eos_token_ids.clone(),
        top_k: 1,
        reasoning: None,
    }
}

fn session_failure_evidence<M: Qwen35Aq4SessionModel>(
    command: &Aq4BenchmarkPrefillCommand,
    session: &Qwen35Aq4InferenceSession<M>,
    error: &str,
    _reset_ok: bool,
) -> Aq4BenchmarkExecutionEvidence {
    let lowered = error.to_ascii_lowercase();
    let code = if lowered.contains("out of memory") || lowered.contains("oom") {
        "runtime_out_of_memory"
    } else if lowered.contains("hip") {
        "hip_fault"
    } else if lowered.contains("reset") {
        "reset_failed"
    } else if lowered.contains("publish") || lowered.contains("stdout") {
        "publish_failed"
    } else {
        "execution_failed"
    };
    session_evidence(
        command,
        session,
        Aq4BenchmarkTerminalStatus::Failed,
        Aq4BenchmarkReuse::Forbidden,
        Some(code.into()),
    )
}

fn session_evidence<M: Qwen35Aq4SessionModel>(
    command: &Aq4BenchmarkPrefillCommand,
    session: &Qwen35Aq4InferenceSession<M>,
    status: Aq4BenchmarkTerminalStatus,
    reuse: Aq4BenchmarkReuse,
    failure_code: Option<String>,
) -> Aq4BenchmarkExecutionEvidence {
    use crate::aq4_benchmark_worker_protocol::{
        Aq4BenchmarkEvidenceLinks, Aq4BenchmarkFallbackEvidence, Aq4BenchmarkResetEvidence,
        sha256_json,
    };
    let audit = session
        .last_terminal_request_execution_audit()
        .and_then(|audit| serde_json::to_value(audit).ok());
    let lifecycle = audit
        .as_ref()
        .and_then(|audit| audit.get("lifecycle"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let reset = lifecycle.get("reset");
    let reset = Aq4BenchmarkResetEvidence {
        attempted: reset
            .and_then(|value| value.get("attempted"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        complete: reset
            .and_then(|value| value.get("complete"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        failed: reset
            .and_then(|value| value.get("failed"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
    };
    let actual_m = audit
        .as_ref()
        .and_then(|value| value.get("resolved_m"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let actual_token_batch_width = audit
        .as_ref()
        .and_then(|value| value.get("actual_token_batch_width"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let actual_request_batch_width = audit
        .as_ref()
        .and_then(|value| value.get("actual_request_batch_width"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let operation_audit_sha256 = audit
        .as_ref()
        .and_then(|value| value.get("operation_audit"))
        .and_then(|value| sha256_json(value).ok());
    Aq4BenchmarkExecutionEvidence {
        status,
        reuse,
        failure_code,
        requested_m: command.requested_m,
        resolved_m: command.resolved_m,
        actual_m,
        actual_token_batch_width,
        actual_request_batch_width,
        fallback: Aq4BenchmarkFallbackEvidence {
            used: command.requested_m != command.resolved_m,
            reason: (command.requested_m != command.resolved_m).then_some("all_m1"),
        },
        lifecycle,
        reset,
        sanitized_audit: audit,
        links: Aq4BenchmarkEvidenceLinks {
            fixture_sha256: command.fixture_sha256.clone(),
            input_sha256: command.input_sha256.clone(),
            operation_audit_sha256,
            resource_observation_key: command.request_id.clone(),
            resource_samples_embedded: false,
        },
    }
}

struct ActiveRequest {
    request_id: String,
    cancel: CancellationToken,
}

#[derive(Default)]
struct RuntimeState {
    active: Option<ActiveRequest>,
    reuse_forbidden: bool,
    writer_failed: bool,
}

enum InferenceCommand {
    Prefill {
        command: Aq4BenchmarkPrefillCommand,
        cancel: CancellationToken,
    },
    Shutdown,
}

struct SharedWriter<W> {
    output: Mutex<W>,
}

impl<W: Write> SharedWriter<W> {
    fn publish(&self, event: &Aq4BenchmarkWorkerEvent<'_>) -> Result<(), String> {
        let mut output = self
            .output
            .lock()
            .map_err(|_| "AQ4 benchmark stdout mutex is poisoned".to_string())?;
        serde_json::to_writer(&mut *output, event)
            .map_err(|error| format!("failed to encode AQ4 benchmark event: {error}"))?;
        output
            .write_all(b"\n")
            .and_then(|()| output.flush())
            .map_err(|error| format!("failed to flush AQ4 benchmark event: {error}"))
    }
}

struct RuntimeProgress<'a, W> {
    writer: &'a SharedWriter<W>,
    request_id: &'a str,
    last_processed: usize,
    prompt_tokens: usize,
}

impl<W: Write> Aq4BenchmarkProgressPublisher for RuntimeProgress<'_, W> {
    fn publish_prompt_progress(&mut self, processed_prompt_tokens: usize) -> Result<(), String> {
        if processed_prompt_tokens <= self.last_processed
            || processed_prompt_tokens > self.prompt_tokens
        {
            return Err("AQ4 benchmark prompt progress is out of order or range".into());
        }
        self.writer.publish(&Aq4BenchmarkWorkerEvent::Progress {
            schema_version: AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
            request_id: self.request_id,
            phase: "prefill",
            processed_prompt_tokens,
        })?;
        self.last_processed = processed_prompt_tokens;
        Ok(())
    }
}

pub fn run_aq4_benchmark_worker_process<R, W, B, F>(
    input: R,
    output: W,
    profile: Sq8WorkerProfile,
    build_backend: F,
) -> Result<Aq4BenchmarkCommandReaderExit, String>
where
    R: BufRead,
    W: Write + Send + 'static,
    B: Aq4BenchmarkInferenceBackend,
    F: FnOnce() -> Result<B, String> + Send + 'static,
{
    let writer = Arc::new(SharedWriter {
        output: Mutex::new(output),
    });
    writer.publish(&Aq4BenchmarkWorkerEvent::Ready {
        schema_version: AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
        model: &profile.model,
        model_revision: &profile.model_revision,
        artifact_content_sha256: &profile.artifact_content_sha256,
        package_manifest_sha256: &profile.package_manifest_sha256,
        device: &profile.device,
        execution_profile: &profile.execution_profile,
        context_length: profile.context_length,
        capability: Default::default(),
    })?;

    let state = Arc::new(Mutex::new(RuntimeState::default()));
    let (sender, receiver) = sync_channel(1);
    let inference_writer = Arc::clone(&writer);
    let inference_state = Arc::clone(&state);
    let inference = thread::Builder::new()
        .name("ullm-aq4-benchmark-inference".into())
        .spawn(move || run_inference(receiver, inference_writer, inference_state, build_backend))
        .map_err(|_| "failed to spawn AQ4 benchmark inference thread".to_string())?;

    let reader_result = run_reader(input, &profile, &writer, &state, &sender);
    drop(sender);
    let inference_result = inference
        .join()
        .map_err(|_| "AQ4 benchmark inference thread panicked".to_string())?;
    match (reader_result, inference_result) {
        (Ok(exit), Ok(())) => {
            let state = state
                .lock()
                .map_err(|_| "AQ4 benchmark runtime state is poisoned".to_string())?;
            if state.writer_failed {
                Err("AQ4 benchmark stdout failed; worker reuse is forbidden".into())
            } else if state.reuse_forbidden {
                Err("AQ4 benchmark terminal evidence forbids worker reuse".into())
            } else {
                Ok(exit)
            }
        }
        (Err(reader), Ok(())) => Err(reader),
        (Ok(_), Err(inference)) => Err(inference),
        (Err(reader), Err(inference)) => Err(format!(
            "AQ4 benchmark reader failed: {reader}; inference failed: {inference}"
        )),
    }
}

fn run_reader<R: BufRead, W: Write>(
    input: R,
    profile: &Sq8WorkerProfile,
    writer: &SharedWriter<W>,
    state: &Mutex<RuntimeState>,
    inference: &SyncSender<InferenceCommand>,
) -> Result<Aq4BenchmarkCommandReaderExit, String> {
    let mut reader = Sq8BoundedJsonlReader::new(input);
    loop {
        let payload = match reader.next_record() {
            Ok(Sq8JsonlRead::Record(payload)) => payload,
            Ok(Sq8JsonlRead::Oversized) => {
                publish_error(
                    writer,
                    None,
                    "invalid_command",
                    "benchmark command exceeds the record bound",
                )?;
                continue;
            }
            Ok(Sq8JsonlRead::Eof) => return begin_shutdown(state, inference),
            Err(_) => return Err("AQ4 benchmark stdin framing failed".into()),
        };
        let command = match decode_aq4_benchmark_worker_command(&payload, profile) {
            Ok(command) => command,
            Err(_) => {
                publish_error(
                    writer,
                    None,
                    "invalid_command",
                    "benchmark command failed exact validation",
                )?;
                continue;
            }
        };
        match command {
            Aq4BenchmarkWorkerCommand::Prefill(command) => {
                let cancel = {
                    let mut state = state
                        .lock()
                        .map_err(|_| "AQ4 benchmark runtime state is poisoned".to_string())?;
                    if state.writer_failed || state.reuse_forbidden {
                        drop(state);
                        publish_error(
                            writer,
                            Some(&command.request_id),
                            "reuse_forbidden",
                            "worker baseline is not reusable",
                        )?;
                        return Err(
                            "AQ4 benchmark request arrived after reuse was forbidden".into()
                        );
                    }
                    if state.active.is_some() {
                        drop(state);
                        publish_error(
                            writer,
                            Some(&command.request_id),
                            "busy",
                            "one benchmark request is already active",
                        )?;
                        continue;
                    }
                    let cancel = CancellationToken::new();
                    state.active = Some(ActiveRequest {
                        request_id: command.request_id.clone(),
                        cancel: cancel.clone(),
                    });
                    cancel
                };
                if inference
                    .send(InferenceCommand::Prefill { command, cancel })
                    .is_err()
                {
                    return Err("AQ4 benchmark inference channel closed during admission".into());
                }
            }
            Aq4BenchmarkWorkerCommand::Cancel { request_id } => {
                let cancel = state
                    .lock()
                    .map_err(|_| "AQ4 benchmark runtime state is poisoned".to_string())?
                    .active
                    .as_ref()
                    .filter(|active| active.request_id == request_id)
                    .map(|active| active.cancel.clone());
                match cancel {
                    Some(cancel) => cancel.cancel_checked()?,
                    None => publish_error(
                        writer,
                        Some(&request_id),
                        "unknown_request",
                        "cancel does not match the active request",
                    )?,
                }
            }
            Aq4BenchmarkWorkerCommand::Shutdown => return begin_shutdown(state, inference),
        }
    }
}

fn begin_shutdown(
    state: &Mutex<RuntimeState>,
    inference: &SyncSender<InferenceCommand>,
) -> Result<Aq4BenchmarkCommandReaderExit, String> {
    let active = {
        let state = state
            .lock()
            .map_err(|_| "AQ4 benchmark runtime state is poisoned".to_string())?;
        state.active.as_ref().map(|active| active.cancel.clone())
    };
    if let Some(cancel) = active.as_ref() {
        cancel.cancel_checked()?;
    }
    inference
        .send(InferenceCommand::Shutdown)
        .map_err(|_| "AQ4 benchmark inference channel closed during shutdown".to_string())?;
    Ok(if active.is_some() {
        Aq4BenchmarkCommandReaderExit::ActiveShutdown
    } else {
        Aq4BenchmarkCommandReaderExit::IdleShutdown
    })
}

fn run_inference<W, B, F>(
    commands: Receiver<InferenceCommand>,
    writer: Arc<SharedWriter<W>>,
    state: Arc<Mutex<RuntimeState>>,
    build_backend: F,
) -> Result<(), String>
where
    W: Write,
    B: Aq4BenchmarkInferenceBackend,
    F: FnOnce() -> Result<B, String>,
{
    let mut backend = build_backend()?;
    while let Ok(command) = commands.recv() {
        match command {
            InferenceCommand::Prefill { command, cancel } => {
                if let Err(error) = execute_one(&mut backend, &command, cancel, &writer, &state) {
                    if let Ok(mut state) = state.lock() {
                        state.writer_failed = true;
                        state.reuse_forbidden = true;
                    }
                    return Err(error);
                }
            }
            InferenceCommand::Shutdown => return backend.shutdown(),
        }
    }
    Err("AQ4 benchmark inference channel closed without shutdown".into())
}

fn execute_one<W: Write, B: Aq4BenchmarkInferenceBackend>(
    backend: &mut B,
    command: &Aq4BenchmarkPrefillCommand,
    cancel: CancellationToken,
    writer: &SharedWriter<W>,
    state: &Mutex<RuntimeState>,
) -> Result<(), String> {
    writer.publish(&Aq4BenchmarkWorkerEvent::Started {
        schema_version: AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
        request_id: &command.request_id,
        case_id: &command.case_id,
        run_kind: command.run_kind,
        run_index: command.run_index,
        prompt_tokens: command.prompt_token_ids.len(),
    })?;
    let mut progress = RuntimeProgress {
        writer,
        request_id: &command.request_id,
        last_processed: 0,
        prompt_tokens: command.prompt_token_ids.len(),
    };
    let evidence = backend.execute_benchmark_prefill(command, cancel, &mut progress);
    validate_terminal_evidence(command, progress.last_processed, &evidence)?;
    writer.publish(&Aq4BenchmarkWorkerEvent::TerminalEvidence {
        schema_version: AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
        evidence_schema_version: AQ4_BENCHMARK_TERMINAL_EVIDENCE_SCHEMA_VERSION,
        request_id: &command.request_id,
        case_id: &command.case_id,
        case_sha256: &command.case_sha256,
        run_kind: command.run_kind,
        run_index: command.run_index,
        generated_tokens: 0,
        evidence: &evidence,
    })?;
    writer.publish(&Aq4BenchmarkWorkerEvent::Released {
        schema_version: AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
        request_id: &command.request_id,
        status: evidence.status,
        reuse: evidence.reuse,
        reset_complete: evidence.reset
            == crate::aq4_benchmark_worker_protocol::Aq4BenchmarkResetEvidence {
                attempted: 1,
                complete: 1,
                failed: 0,
            },
    })?;
    let mut state = state
        .lock()
        .map_err(|_| "AQ4 benchmark runtime state is poisoned".to_string())?;
    if state
        .active
        .as_ref()
        .is_none_or(|active| active.request_id != command.request_id)
    {
        return Err("AQ4 benchmark terminal request ownership changed".into());
    }
    state.active = None;
    if evidence.reuse == Aq4BenchmarkReuse::Forbidden {
        state.reuse_forbidden = true;
    }
    Ok(())
}

fn validate_terminal_evidence(
    command: &Aq4BenchmarkPrefillCommand,
    processed_prompt_tokens: usize,
    evidence: &Aq4BenchmarkExecutionEvidence,
) -> Result<(), String> {
    if evidence.requested_m != command.requested_m
        || evidence.resolved_m != command.resolved_m
        || evidence.fallback.used != (command.requested_m != command.resolved_m)
        || evidence.links.fixture_sha256 != command.fixture_sha256
        || evidence.links.input_sha256 != command.input_sha256
        || evidence.links.resource_observation_key != command.request_id
        || evidence.links.resource_samples_embedded
    {
        return Err("AQ4 benchmark terminal evidence identity differs from its command".into());
    }
    let reset_complete =
        evidence.reset.attempted == 1 && evidence.reset.complete == 1 && evidence.reset.failed == 0;
    match evidence.status {
        Aq4BenchmarkTerminalStatus::Ok
            if processed_prompt_tokens == command.prompt_token_ids.len()
                && evidence.failure_code.is_none()
                && evidence.actual_m == Some(command.resolved_m)
                && reset_complete
                && evidence.reuse == Aq4BenchmarkReuse::Allowed => {}
        Aq4BenchmarkTerminalStatus::Cancelled
            if evidence.failure_code.is_none() && reset_complete => {}
        Aq4BenchmarkTerminalStatus::Failed if evidence.failure_code.is_some() => {}
        _ => {
            return Err("AQ4 benchmark terminal status, reset, or actual M is inconsistent".into());
        }
    }
    Ok(())
}

fn publish_error<W: Write>(
    writer: &SharedWriter<W>,
    request_id: Option<&str>,
    code: &'static str,
    message: &'static str,
) -> Result<(), String> {
    writer.publish(&Aq4BenchmarkWorkerEvent::Error {
        schema_version: AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
        request_id,
        code,
        recoverable: matches!(code, "invalid_command" | "busy" | "unknown_request"),
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aq4_benchmark_worker_protocol::{
        Aq4BenchmarkEvidenceLinks, Aq4BenchmarkFallbackEvidence, Aq4BenchmarkResetEvidence,
        aq4_benchmark_input_sha256,
    };
    use std::io::Cursor;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Clone, Default)]
    struct Output(Arc<Mutex<Vec<u8>>>);

    impl Write for Output {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    struct MockBackend {
        cancel_seen: Arc<AtomicBool>,
        reset_failed: bool,
        honor_cancel: bool,
    }

    impl Aq4BenchmarkInferenceBackend for MockBackend {
        fn execute_benchmark_prefill(
            &mut self,
            command: &Aq4BenchmarkPrefillCommand,
            cancel: CancellationToken,
            progress: &mut dyn Aq4BenchmarkProgressPublisher,
        ) -> Aq4BenchmarkExecutionEvidence {
            let cancel_requested = cancel.is_cancelled();
            self.cancel_seen.store(cancel_requested, Ordering::SeqCst);
            let cancelled = self.honor_cancel && cancel_requested;
            if !cancelled {
                progress
                    .publish_prompt_progress(command.prompt_token_ids.len())
                    .unwrap();
            }
            let status = if self.reset_failed {
                Aq4BenchmarkTerminalStatus::Failed
            } else if cancelled {
                Aq4BenchmarkTerminalStatus::Cancelled
            } else {
                Aq4BenchmarkTerminalStatus::Ok
            };
            Aq4BenchmarkExecutionEvidence {
                status,
                reuse: if self.reset_failed {
                    Aq4BenchmarkReuse::Forbidden
                } else {
                    Aq4BenchmarkReuse::Allowed
                },
                failure_code: self.reset_failed.then(|| "reset_failed".into()),
                requested_m: command.requested_m,
                resolved_m: command.resolved_m,
                actual_m: (!cancelled && !self.reset_failed).then_some(command.resolved_m),
                actual_token_batch_width: (!cancelled && !self.reset_failed)
                    .then_some(command.resolved_m),
                actual_request_batch_width: (!cancelled && !self.reset_failed).then_some(1),
                fallback: Aq4BenchmarkFallbackEvidence {
                    used: command.requested_m != command.resolved_m,
                    reason: (command.requested_m != command.resolved_m).then_some("all_m1"),
                },
                lifecycle: serde_json::json!({"reset": {"attempted": 1, "complete": usize::from(!self.reset_failed), "failed": usize::from(self.reset_failed)}}),
                reset: Aq4BenchmarkResetEvidence {
                    attempted: 1,
                    complete: u64::from(!self.reset_failed),
                    failed: u64::from(self.reset_failed),
                },
                sanitized_audit: Some(serde_json::json!({"schema_version": "mock.v1"})),
                links: Aq4BenchmarkEvidenceLinks {
                    fixture_sha256: command.fixture_sha256.clone(),
                    input_sha256: command.input_sha256.clone(),
                    operation_audit_sha256: Some("d".repeat(64)),
                    resource_observation_key: command.request_id.clone(),
                    resource_samples_embedded: false,
                },
            }
        }
    }

    fn profile() -> Sq8WorkerProfile {
        let mut profile = Sq8WorkerProfile::sq8_defaults();
        profile.context_length = 32;
        profile.vocab_size = 64;
        profile
    }

    fn prefill(request_id: &str, requested_m: usize, resolved_m: usize) -> String {
        let tokens = [1, 2, 3, 4];
        serde_json::json!({
            "schema_version": AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
            "type": "benchmark_prefill",
            "request_id": request_id,
            "case_id": "case-1",
            "case_sha256": "a".repeat(64),
            "run_kind": "measured",
            "run_index": 2,
            "requested_m": requested_m,
            "resolved_m": resolved_m,
            "generated_tokens": 0,
            "fixture_sha256": "b".repeat(64),
            "input_sha256": aq4_benchmark_input_sha256(&tokens),
            "prompt_token_ids": tokens,
        })
        .to_string()
    }

    #[test]
    fn m_grid_prefill_orders_ready_started_progress_evidence_release() {
        for &m in crate::aq4_benchmark_worker_protocol::AQ4_BENCHMARK_PREFILL_M_GRID {
            let input = format!(
                "{}\n{{\"schema_version\":\"{}\",\"type\":\"shutdown\"}}\n",
                prefill("req-grid", m, m),
                AQ4_BENCHMARK_WORKER_SCHEMA_VERSION
            );
            let output = Output::default();
            let captured = output.clone();
            let result =
                run_aq4_benchmark_worker_process(Cursor::new(input), output, profile(), || {
                    Ok(MockBackend {
                        cancel_seen: Default::default(),
                        reset_failed: false,
                        honor_cancel: false,
                    })
                });
            assert!(matches!(
                result,
                Ok(Aq4BenchmarkCommandReaderExit::ActiveShutdown)
                    | Ok(Aq4BenchmarkCommandReaderExit::IdleShutdown)
            ));
            let events = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
            let types = events
                .lines()
                .map(|line| {
                    serde_json::from_str::<serde_json::Value>(line).unwrap()["type"]
                        .as_str()
                        .unwrap()
                        .to_string()
                })
                .collect::<Vec<_>>();
            assert_eq!(
                types,
                [
                    "ready",
                    "started",
                    "progress",
                    "terminal_evidence",
                    "released"
                ]
            );
        }
    }

    #[test]
    fn cancel_is_terminal_and_reset_failure_forbids_reuse() {
        let cancel_seen = Arc::new(AtomicBool::new(false));
        let input = format!(
            "{}\n{{\"schema_version\":\"{}\",\"type\":\"cancel\",\"request_id\":\"req-cancel\"}}\n{{\"schema_version\":\"{}\",\"type\":\"shutdown\"}}\n",
            prefill("req-cancel", 64, 1),
            AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
            AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
        );
        let output = Output::default();
        let captured = output.clone();
        run_aq4_benchmark_worker_process(Cursor::new(input), output, profile(), {
            let cancel_seen = Arc::clone(&cancel_seen);
            move || {
                Ok(MockBackend {
                    cancel_seen,
                    reset_failed: false,
                    honor_cancel: true,
                })
            }
        })
        .unwrap();
        assert!(cancel_seen.load(Ordering::SeqCst));
        let events = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
        assert!(events.contains("\"status\":\"cancelled\""));

        let input = format!(
            "{}\n{{\"schema_version\":\"{}\",\"type\":\"shutdown\"}}\n",
            prefill("req-reset", 1, 1),
            AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
        );
        let result = run_aq4_benchmark_worker_process(
            Cursor::new(input),
            Output::default(),
            profile(),
            || {
                Ok(MockBackend {
                    cancel_seen: Default::default(),
                    reset_failed: true,
                    honor_cancel: false,
                })
            },
        );
        assert!(result.unwrap_err().contains("forbids worker reuse"));
    }
}
