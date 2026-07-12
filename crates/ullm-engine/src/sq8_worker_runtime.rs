// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU-side thread and channel topology for the resident SQ8 worker.

use crate::sq8_model_head_runtime::QWEN3_14B_VOCAB_SIZE;
use crate::sq8_serving_runtime::Sq8ServingRequest;
use crate::sq8_worker_protocol::{
    Sq8ActiveTerminalFlushAck, Sq8ActiveTerminalPermit, Sq8BoundedJsonlReader,
    Sq8JsonlFramingError, Sq8JsonlRead, Sq8OrderedJsonlWriter, Sq8PromptProgressTracker,
    Sq8ReadyFlushAck, Sq8ReleaseOutcomeEvent, Sq8WorkerAdmission, Sq8WorkerCommand,
    Sq8WorkerCommandKind, Sq8WorkerControl, Sq8WorkerControlErrorKind, Sq8WorkerErrorCode,
    Sq8WorkerEvent, Sq8WorkerLifecycle, Sq8WorkerProtocolErrorKind, Sq8WorkerShutdownDisposition,
    Sq8WorkerTimings, inspect_sq8_worker_command,
};
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const SQ8_INFERENCE_POISON_POLL: Duration = Duration::from_millis(50);
pub const SQ8_TERMINAL_CLEANUP_DEADLINE: Duration = Duration::from_secs(5);

enum Sq8WriterPublication {
    Regular,
    Ready,
    ActiveTerminal(Sq8ActiveTerminalPermit),
    FatalBestEffort,
    Close,
}

enum Sq8WriterReceipt {
    Regular,
    Ready(Sq8ReadyFlushAck),
    ActiveTerminal(Sq8ActiveTerminalFlushAck),
    Closed,
}

struct Sq8WriterEnvelope {
    publication: Sq8WriterPublication,
    event: Option<Sq8WorkerEvent>,
    acknowledgement: Option<SyncSender<Result<Sq8WriterReceipt, String>>>,
}

#[derive(Clone)]
pub struct Sq8WorkerEventPublisher {
    sender: SyncSender<Sq8WriterEnvelope>,
    poisoned: Arc<AtomicBool>,
}

impl Sq8WorkerEventPublisher {
    fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    fn publish(&self, event: Sq8WorkerEvent) -> Result<(), String> {
        match self.publish_inner(Sq8WriterPublication::Regular, event)? {
            Sq8WriterReceipt::Regular => Ok(()),
            _ => Err("SQ8 writer returned the wrong regular-event acknowledgement".into()),
        }
    }

    fn publish_ready(&self, event: Sq8WorkerEvent) -> Result<Sq8ReadyFlushAck, String> {
        match self.publish_inner(Sq8WriterPublication::Ready, event)? {
            Sq8WriterReceipt::Ready(acknowledgement) => Ok(acknowledgement),
            _ => Err("SQ8 writer returned the wrong ready acknowledgement".into()),
        }
    }

    fn publish_active_terminal(
        &self,
        permit: Sq8ActiveTerminalPermit,
        event: Sq8WorkerEvent,
    ) -> Result<Sq8ActiveTerminalFlushAck, String> {
        match self.publish_inner(Sq8WriterPublication::ActiveTerminal(permit), event)? {
            Sq8WriterReceipt::ActiveTerminal(acknowledgement) => Ok(acknowledgement),
            _ => Err("SQ8 writer returned the wrong terminal acknowledgement".into()),
        }
    }

    fn try_publish_fatal(&self, event: Sq8WorkerEvent) -> bool {
        self.poisoned.store(true, Ordering::Release);
        let envelope = Sq8WriterEnvelope {
            publication: Sq8WriterPublication::FatalBestEffort,
            event: Some(event),
            acknowledgement: None,
        };
        match self.sender.try_send(envelope) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    fn publish_inner(
        &self,
        publication: Sq8WriterPublication,
        event: Sq8WorkerEvent,
    ) -> Result<Sq8WriterReceipt, String> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err("SQ8 ordered writer is poisoned".into());
        }
        let (acknowledgement, result) = sync_channel(0);
        self.sender
            .send(Sq8WriterEnvelope {
                publication,
                event: Some(event),
                acknowledgement: Some(acknowledgement),
            })
            .map_err(|_| "SQ8 ordered writer channel is closed".to_string())?;
        result
            .recv()
            .map_err(|_| "SQ8 ordered writer exited before acknowledging an event".to_string())?
    }
}

pub struct Sq8WriterThread<W> {
    closer: SyncSender<Sq8WriterEnvelope>,
    poisoned: Arc<AtomicBool>,
    join: JoinHandle<Result<W, String>>,
}

impl<W> Sq8WriterThread<W> {
    pub fn close_and_join(self) -> Result<W, String> {
        let Self {
            closer,
            poisoned,
            join,
        } = self;
        let (acknowledgement, result) = sync_channel(0);
        let close_sent = closer
            .send(Sq8WriterEnvelope {
                publication: Sq8WriterPublication::Close,
                event: None,
                acknowledgement: Some(acknowledgement),
            })
            .is_ok();
        let close_error = close_sent
            .then(|| match result.recv() {
                Ok(Ok(Sq8WriterReceipt::Closed)) | Err(_) => None,
                Ok(Ok(_)) => Some("SQ8 writer returned the wrong close acknowledgement".into()),
                Ok(Err(error)) => Some(error),
            })
            .flatten();
        drop(closer);
        let was_poisoned = poisoned.load(Ordering::Acquire);
        let output = join
            .join()
            .map_err(|_| "SQ8 ordered writer thread panicked".to_string())??;
        if let Some(error) = close_error {
            return Err(error);
        }
        if was_poisoned || poisoned.load(Ordering::Acquire) {
            return Err("SQ8 ordered writer closed after process poison".into());
        }
        Ok(output)
    }
}

pub fn spawn_sq8_ordered_writer<W>(
    output: W,
) -> Result<(Sq8WorkerEventPublisher, Sq8WriterThread<W>), String>
where
    W: Write + Send + 'static,
{
    let (sender, receiver) = sync_channel(1);
    let poisoned = Arc::new(AtomicBool::new(false));
    let writer_poisoned = Arc::clone(&poisoned);
    let closer = sender.clone();
    let join = thread::Builder::new()
        .name("ullm-sq8-writer".into())
        .spawn(move || run_sq8_ordered_writer(output, receiver, writer_poisoned))
        .map_err(|_| "failed to spawn SQ8 ordered writer thread".to_string())?;
    let thread_poisoned = Arc::clone(&poisoned);
    Ok((
        Sq8WorkerEventPublisher { sender, poisoned },
        Sq8WriterThread {
            closer,
            poisoned: thread_poisoned,
            join,
        },
    ))
}

fn run_sq8_ordered_writer<W: Write>(
    output: W,
    receiver: Receiver<Sq8WriterEnvelope>,
    poisoned: Arc<AtomicBool>,
) -> Result<W, String> {
    let mut writer = Sq8OrderedJsonlWriter::new(output);
    while let Ok(envelope) = receiver.recv() {
        if matches!(envelope.publication, Sq8WriterPublication::Close) {
            let Some(acknowledgement) = envelope.acknowledgement else {
                return Err("SQ8 writer close is missing its acknowledgement".into());
            };
            let _ = acknowledgement.send(Ok(Sq8WriterReceipt::Closed));
            return Ok(writer.into_inner());
        }
        if matches!(envelope.publication, Sq8WriterPublication::FatalBestEffort) {
            let Some(event) = envelope.event else {
                return Err("SQ8 fatal writer envelope is missing its event".into());
            };
            if is_fatal_event(&event) {
                let _ = writer.write_event(&event);
            }
            return Err("SQ8 ordered writer received fatal process poison".into());
        }
        if poisoned.load(Ordering::Acquire) {
            if let Some(acknowledgement) = envelope.acknowledgement {
                let _ = acknowledgement.send(Err("SQ8 ordered writer is poisoned".into()));
            }
            return Err("SQ8 ordered writer stopped after process poison".into());
        }
        let Some(event) = envelope.event else {
            poisoned.store(true, Ordering::Release);
            return Err("SQ8 nonterminal writer envelope is missing its event".into());
        };
        let result = match envelope.publication {
            Sq8WriterPublication::Regular if is_regular_event(&event) => writer
                .write_event(&event)
                .map(|()| Sq8WriterReceipt::Regular),
            Sq8WriterPublication::Regular => {
                Err("SQ8 regular writer received a boundary event".into())
            }
            Sq8WriterPublication::Ready => writer
                .write_ready_event(&event)
                .map(Sq8WriterReceipt::Ready),
            Sq8WriterPublication::ActiveTerminal(permit) => writer
                .write_active_terminal_event(permit, &event)
                .map(Sq8WriterReceipt::ActiveTerminal),
            Sq8WriterPublication::FatalBestEffort | Sq8WriterPublication::Close => {
                unreachable!("handled before event dispatch")
            }
        };
        let failed = result.is_err();
        let failure = result.as_ref().err().cloned();
        let Some(acknowledgement) = envelope.acknowledgement else {
            poisoned.store(true, Ordering::Release);
            return Err("SQ8 nonfatal event is missing its acknowledgement channel".into());
        };
        if acknowledgement.send(result).is_err() {
            poisoned.store(true, Ordering::Release);
            return Err("SQ8 event caller disappeared before flush acknowledgement".into());
        }
        if failed {
            poisoned.store(true, Ordering::Release);
            return Err(failure.unwrap_or_else(|| "SQ8 ordered writer failed".into()));
        }
    }
    Ok(writer.into_inner())
}

fn is_regular_event(event: &Sq8WorkerEvent) -> bool {
    matches!(
        event,
        Sq8WorkerEvent::Started { .. }
            | Sq8WorkerEvent::Progress { .. }
            | Sq8WorkerEvent::Token { .. }
            | Sq8WorkerEvent::Error {
                recoverable: true,
                ..
            }
    )
}

fn is_fatal_event(event: &Sq8WorkerEvent) -> bool {
    matches!(
        event,
        Sq8WorkerEvent::Error {
            recoverable: false,
            ..
        }
    )
}

#[derive(Debug)]
pub enum Sq8InferenceCommand {
    Generate {
        request: Sq8ServingRequest,
        admission: Sq8WorkerAdmission,
    },
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sq8CommandReaderExit {
    IdleShutdown,
    ActiveShutdown { generation: u64 },
}

pub struct Sq8RequestEventPublisher<'a> {
    control: &'a Sq8WorkerControl,
    events: &'a Sq8WorkerEventPublisher,
    generation: u64,
    request_id: String,
    prompt_tokens: usize,
    max_new_tokens: usize,
    eos_token_ids: Vec<usize>,
    progress: Sq8PromptProgressTracker,
    started: bool,
    completion_tokens: usize,
    last_token_was_eos: bool,
    released: bool,
}

impl<'a> Sq8RequestEventPublisher<'a> {
    fn new(
        control: &'a Sq8WorkerControl,
        events: &'a Sq8WorkerEventPublisher,
        request: &Sq8ServingRequest,
        admission: &Sq8WorkerAdmission,
    ) -> Result<Self, String> {
        if request.request_id != admission.request_id {
            return Err("SQ8 request and admission IDs do not match".into());
        }
        let snapshot = control.snapshot().map_err(|error| error.to_string())?;
        if snapshot.active_generation != Some(admission.generation)
            || snapshot.active_request_id.as_deref() != Some(admission.request_id.as_str())
            || !matches!(
                snapshot.lifecycle,
                Sq8WorkerLifecycle::Ready | Sq8WorkerLifecycle::Closing
            )
        {
            return Err("SQ8 publication state does not own the active generation".into());
        }
        let prompt_tokens = request.prompt_token_ids.len();
        Ok(Self {
            control,
            events,
            generation: admission.generation,
            request_id: admission.request_id.clone(),
            prompt_tokens,
            max_new_tokens: request.max_new_tokens,
            eos_token_ids: request.eos_token_ids.clone(),
            progress: Sq8PromptProgressTracker::new(prompt_tokens)?,
            started: false,
            completion_tokens: 0,
            last_token_was_eos: false,
            released: false,
        })
    }

    pub fn publish_started(&mut self) -> Result<(), String> {
        if self.started || self.released {
            return Err("SQ8 request started event is out of order".into());
        }
        self.events.publish(Sq8WorkerEvent::started(
            self.request_id.clone(),
            self.prompt_tokens,
        ))?;
        self.started = true;
        Ok(())
    }

    pub fn observe_prompt_unit(
        &mut self,
        processed_prompt_tokens: usize,
        execution_width: usize,
    ) -> Result<(), String> {
        self.require_prefill_event("progress")?;
        if let Some(processed) = self
            .progress
            .observe_unit(processed_prompt_tokens, execution_width)?
        {
            self.events
                .publish(Sq8WorkerEvent::progress(self.request_id.clone(), processed))?;
        }
        Ok(())
    }

    pub fn observe_prefill_transition(&mut self) -> Result<(), String> {
        self.require_prefill_event("prefill transition")?;
        if let Some(processed) = self.progress.observe_transition()? {
            self.events
                .publish(Sq8WorkerEvent::progress(self.request_id.clone(), processed))?;
        }
        Ok(())
    }

    pub fn publish_token(&mut self, token_id: usize) -> Result<(), String> {
        if !self.started
            || self.released
            || !self.progress.transition_emitted()
            || self.completion_tokens >= self.max_new_tokens
            || self.last_token_was_eos
            || token_id >= QWEN3_14B_VOCAB_SIZE
        {
            return Err("SQ8 token publication is out of order or range".into());
        }
        self.events.publish(Sq8WorkerEvent::token(
            self.request_id.clone(),
            self.completion_tokens,
            token_id,
        ))?;
        self.completion_tokens += 1;
        self.last_token_was_eos = self.eos_token_ids.contains(&token_id);
        Ok(())
    }

    pub fn publish_released(&mut self, outcome: Sq8ReleaseOutcomeEvent) -> Result<(), String> {
        self.publish_released_inner(outcome, None)
    }

    pub fn publish_released_with_timings(
        &mut self,
        outcome: Sq8ReleaseOutcomeEvent,
        timings: Sq8WorkerTimings,
    ) -> Result<(), String> {
        self.publish_released_inner(outcome, Some(timings))
    }

    fn publish_released_inner(
        &mut self,
        outcome: Sq8ReleaseOutcomeEvent,
        timings: Option<Sq8WorkerTimings>,
    ) -> Result<(), String> {
        if !self.started || self.released {
            return Err("SQ8 released event is out of order".into());
        }
        match outcome {
            Sq8ReleaseOutcomeEvent::Stop
                if self.progress.transition_emitted()
                    && self.completion_tokens > 0
                    && self.last_token_was_eos => {}
            Sq8ReleaseOutcomeEvent::Length
                if self.progress.transition_emitted()
                    && self.completion_tokens == self.max_new_tokens
                    && !self.last_token_was_eos => {}
            Sq8ReleaseOutcomeEvent::Cancelled
                if self.completion_tokens < self.max_new_tokens && !self.last_token_was_eos => {}
            _ => return Err("SQ8 released outcome does not match request progress".into()),
        }
        let cancel_reason = if outcome == Sq8ReleaseOutcomeEvent::Cancelled {
            self.control
                .first_cancel_reason(self.generation)
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "SQ8 cancelled release has no control reason".to_string())?
                .into()
        } else {
            None
        };
        let event = match timings {
            Some(timings) => Sq8WorkerEvent::released_with_timings(
                self.request_id.clone(),
                outcome,
                cancel_reason,
                self.prompt_tokens,
                self.completion_tokens,
                timings,
            ),
            None => Sq8WorkerEvent::released(
                self.request_id.clone(),
                outcome,
                cancel_reason,
                self.prompt_tokens,
                self.completion_tokens,
            ),
        }
        .map_err(|error| error.to_string())?;
        let permit = self
            .control
            .begin_terminal_publication(self.generation, &self.request_id)
            .map_err(|error| error.to_string())?;
        let acknowledgement = self.events.publish_active_terminal(permit, event)?;
        self.control
            .acknowledge_terminal_flush(acknowledgement)
            .map_err(|error| error.to_string())?;
        self.released = true;
        Ok(())
    }

    pub fn run_terminal_cleanup<T, F>(&mut self, cleanup: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
    {
        self.run_terminal_cleanup_with(
            SQ8_TERMINAL_CLEANUP_DEADLINE,
            || std::process::exit(1),
            cleanup,
        )
    }

    fn run_terminal_cleanup_with<T, F, X>(
        &mut self,
        deadline: Duration,
        terminate: X,
        cleanup: F,
    ) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
        X: FnOnce() + Send,
    {
        self.run_terminal_cleanup_with_arm_hook(deadline, terminate, || {}, cleanup)
    }

    fn run_terminal_cleanup_with_arm_hook<T, F, X, A>(
        &mut self,
        deadline: Duration,
        terminate: X,
        before_arm: A,
        cleanup: F,
    ) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
        X: FnOnce() + Send,
        A: FnOnce() + Send,
    {
        let request_id = self.request_id.clone();
        let control = self.control;
        let events = self.events;
        let expires_at = Instant::now()
            .checked_add(deadline)
            .ok_or_else(|| "SQ8 terminal cleanup deadline overflowed".to_string())?;
        let (completed, completion) = std::sync::mpsc::channel();
        let (armed_sender, armed_receiver) = sync_channel(1);
        let (cleanup_result, timed_out) = thread::scope(|scope| {
            let watchdog = thread::Builder::new()
                .name("ullm-sq8-cleanup-watchdog".into())
                .spawn_scoped(scope, move || {
                    before_arm();
                    if armed_sender.send(()).is_err() {
                        return false;
                    }
                    let completion_result = expires_at
                        .checked_duration_since(Instant::now())
                        .filter(|remaining| !remaining.is_zero())
                        .map_or(Err(RecvTimeoutError::Timeout), |remaining| {
                            completion.recv_timeout(remaining)
                        });
                    if terminal_cleanup_completed_before_deadline(&completion_result, expires_at) {
                        false
                    } else {
                        publish_fatal_best_effort(
                            events,
                            Some(request_id),
                            Sq8WorkerErrorCode::CleanupDeadlineExceeded,
                            "SQ8 terminal cleanup exceeded the 5 second deadline",
                        );
                        control.try_mark_failed_best_effort();
                        terminate();
                        true
                    }
                })
                .map_err(|_| "failed to spawn SQ8 terminal cleanup watchdog".to_string())?;
            armed_receiver
                .recv()
                .map_err(|_| "SQ8 terminal cleanup watchdog failed before arming".to_string())?;
            let cleanup_result = if Instant::now() < expires_at {
                Some(cleanup())
            } else {
                None
            };
            if cleanup_result.is_some() {
                let _ = completed.send(Instant::now());
            }
            let timed_out = watchdog
                .join()
                .map_err(|_| "SQ8 terminal cleanup watchdog panicked".to_string())?;
            Ok::<_, String>((cleanup_result, timed_out))
        })?;
        if timed_out {
            Err("SQ8 terminal cleanup deadline exceeded".into())
        } else {
            cleanup_result.ok_or_else(|| {
                "SQ8 terminal cleanup watchdog expired before cleanup started".to_string()
            })?
        }
    }

    pub fn completion_tokens(&self) -> usize {
        self.completion_tokens
    }

    fn require_prefill_event(&self, event: &str) -> Result<(), String> {
        if !self.started || self.released || self.completion_tokens != 0 {
            return Err(format!("SQ8 request {event} event is out of order"));
        }
        Ok(())
    }
}

fn terminal_cleanup_completed_before_deadline(
    completion: &Result<Instant, RecvTimeoutError>,
    expires_at: Instant,
) -> bool {
    matches!(completion, Ok(completed_at) if *completed_at < expires_at)
}

pub trait Sq8InferenceBackend {
    fn execute(
        &mut self,
        request: Sq8ServingRequest,
        admission: Sq8WorkerAdmission,
        publications: &mut Sq8RequestEventPublisher<'_>,
    ) -> Result<(), String>;

    fn shutdown(&mut self) -> Result<(), String> {
        Ok(())
    }
}

pub struct Sq8InferenceThread {
    startup: std::sync::Mutex<Option<Receiver<Result<(), String>>>>,
    join: JoinHandle<Result<(), String>>,
}

impl Sq8InferenceThread {
    pub fn wait_until_ready(&self) -> Result<(), String> {
        let startup = self
            .startup
            .lock()
            .map_err(|_| "SQ8 startup acknowledgement mutex is poisoned".to_string())?
            .take()
            .ok_or_else(|| "SQ8 startup acknowledgement was already consumed".to_string())?;
        startup
            .recv()
            .map_err(|_| "SQ8 inference thread exited before startup acknowledgement".to_string())?
    }

    pub fn join(self) -> Result<(), String> {
        let Self { startup, join } = self;
        drop(startup);
        join.join()
            .map_err(|_| "SQ8 inference thread panicked".to_string())?
    }
}

pub fn spawn_sq8_inference_thread<B, F>(
    control: Arc<Sq8WorkerControl>,
    events: Sq8WorkerEventPublisher,
    commands: Receiver<Sq8InferenceCommand>,
    build_backend: F,
) -> Result<Sq8InferenceThread, String>
where
    B: Sq8InferenceBackend + 'static,
    F: FnOnce() -> Result<B, String> + Send + 'static,
{
    let (startup_sender, startup) = sync_channel(0);
    let join = thread::Builder::new()
        .name("ullm-sq8-inference".into())
        .spawn(move || {
            run_sq8_inference_thread(control, events, commands, build_backend, startup_sender)
        })
        .map_err(|_| "failed to spawn SQ8 inference thread".to_string())?;
    Ok(Sq8InferenceThread {
        startup: std::sync::Mutex::new(Some(startup)),
        join,
    })
}

fn run_sq8_inference_thread<B, F>(
    control: Arc<Sq8WorkerControl>,
    events: Sq8WorkerEventPublisher,
    commands: Receiver<Sq8InferenceCommand>,
    build_backend: F,
    startup: SyncSender<Result<(), String>>,
) -> Result<(), String>
where
    B: Sq8InferenceBackend,
    F: FnOnce() -> Result<B, String>,
{
    let mut backend = match build_backend() {
        Ok(backend) => backend,
        Err(error) => {
            let _ = control.mark_failed();
            publish_fatal_best_effort(
                &events,
                None,
                Sq8WorkerErrorCode::LoadFailed,
                "SQ8 resident backend failed to load",
            );
            let _ = startup.send(Err(error.clone()));
            return Err(error);
        }
    };
    let ready = match events.publish_ready(Sq8WorkerEvent::ready()) {
        Ok(acknowledgement) => acknowledgement,
        Err(error) => {
            let _ = control.mark_failed();
            let _ = startup.send(Err(error.clone()));
            return Err(error);
        }
    };
    if let Err(error) = control.mark_ready_after_flush(ready) {
        let message = error.to_string();
        let _ = control.mark_failed();
        publish_fatal_best_effort(
            &events,
            None,
            Sq8WorkerErrorCode::InvariantFailed,
            "SQ8 readiness transition failed",
        );
        let _ = startup.send(Err(message.clone()));
        return Err(message);
    }
    if startup.send(Ok(())).is_err() {
        return Err(fail_inference(
            &control,
            &events,
            "SQ8 startup caller disappeared after readiness",
        ));
    }

    loop {
        let command = loop {
            match commands.recv_timeout(SQ8_INFERENCE_POISON_POLL) {
                Ok(command) => break command,
                Err(RecvTimeoutError::Timeout) if events.is_poisoned() => {
                    let _ = control.mark_failed();
                    return Err("SQ8 inference thread observed process poison".into());
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(fail_inference(
                        &control,
                        &events,
                        "SQ8 inference command channel closed unexpectedly",
                    ));
                }
            }
        };
        match command {
            Sq8InferenceCommand::Generate { request, admission } => {
                let completed_generation = admission.generation;
                let mut publications =
                    Sq8RequestEventPublisher::new(control.as_ref(), &events, &request, &admission)
                        .map_err(|_| {
                            fail_inference(
                                &control,
                                &events,
                                "SQ8 admitted request publication state is invalid",
                            )
                        })?;
                if let Err(error) = backend.execute(request, admission, &mut publications) {
                    let _ =
                        fail_inference(&control, &events, "SQ8 admitted request execution failed");
                    return Err(error);
                }
                let snapshot = control.snapshot().map_err(|_| {
                    fail_inference(
                        &control,
                        &events,
                        "SQ8 control snapshot failed after request execution",
                    )
                })?;
                if snapshot.active_generation == Some(completed_generation) {
                    return Err(fail_inference(
                        &control,
                        &events,
                        "SQ8 backend returned before terminal flush acknowledgement",
                    ));
                }
                if snapshot.active_generation.is_some() {
                    match snapshot.lifecycle {
                        Sq8WorkerLifecycle::Ready | Sq8WorkerLifecycle::Closing => continue,
                        Sq8WorkerLifecycle::Failed => {
                            return Err("SQ8 control failed after request execution".into());
                        }
                        Sq8WorkerLifecycle::Loading => {
                            return Err(fail_inference(
                                &control,
                                &events,
                                "SQ8 control returned to Loading with a queued request",
                            ));
                        }
                    }
                }
                match snapshot.lifecycle {
                    Sq8WorkerLifecycle::Ready => {}
                    Sq8WorkerLifecycle::Closing => {
                        return shutdown_inference_backend(&mut backend, &control, &events);
                    }
                    Sq8WorkerLifecycle::Failed => {
                        return Err("SQ8 control failed during request execution".into());
                    }
                    Sq8WorkerLifecycle::Loading => {
                        return Err(fail_inference(
                            &control,
                            &events,
                            "SQ8 control returned to Loading after request execution",
                        ));
                    }
                }
            }
            Sq8InferenceCommand::Shutdown => {
                let snapshot = control.snapshot().map_err(|_| {
                    fail_inference(
                        &control,
                        &events,
                        "SQ8 control snapshot failed during shutdown",
                    )
                })?;
                if snapshot.lifecycle != Sq8WorkerLifecycle::Closing
                    || snapshot.active_generation.is_some()
                {
                    return Err(fail_inference(
                        &control,
                        &events,
                        "SQ8 idle shutdown arrived outside Closing baseline",
                    ));
                }
                return shutdown_inference_backend(&mut backend, &control, &events);
            }
        }
    }
}

fn shutdown_inference_backend<B: Sq8InferenceBackend>(
    backend: &mut B,
    control: &Arc<Sq8WorkerControl>,
    events: &Sq8WorkerEventPublisher,
) -> Result<(), String> {
    backend.shutdown().inspect_err(|_| {
        let _ = fail_inference(control, events, "SQ8 resident backend shutdown failed");
    })
}

fn fail_inference(
    control: &Arc<Sq8WorkerControl>,
    events: &Sq8WorkerEventPublisher,
    message: &'static str,
) -> String {
    let request_id = active_request_id(control);
    let _ = control.mark_failed();
    publish_fatal_best_effort(
        events,
        request_id,
        Sq8WorkerErrorCode::RuntimeFailed,
        message,
    );
    message.into()
}

pub fn run_sq8_worker_process<R, W, B, F>(
    input: R,
    output: W,
    build_backend: F,
) -> Result<Sq8CommandReaderExit, String>
where
    R: BufRead + Send + 'static,
    W: Write + Send + 'static,
    B: Sq8InferenceBackend + 'static,
    F: FnOnce() -> Result<B, String> + Send + 'static,
{
    let control = Arc::new(Sq8WorkerControl::new());
    let (events, writer) = spawn_sq8_ordered_writer(output)?;
    let (commands, command_receiver) = sync_channel(1);
    let inference = match spawn_sq8_inference_thread(
        Arc::clone(&control),
        events.clone(),
        command_receiver,
        build_backend,
    ) {
        Ok(inference) => inference,
        Err(error) => {
            drop(commands);
            drop(events);
            let writer_result = writer.close_and_join().map(|_| ());
            return Err(join_worker_process_errors([
                ("inference spawn", Err(error)),
                ("writer close", writer_result),
            ]));
        }
    };

    if let Err(startup_error) = inference.wait_until_ready() {
        drop(commands);
        let inference_result = inference.join();
        drop(events);
        let writer_result = writer.close_and_join().map(|_| ());
        return Err(join_worker_process_errors([
            ("startup", Err(startup_error)),
            ("inference join", inference_result),
            ("writer close", writer_result),
        ]));
    }

    let reader_control = Arc::clone(&control);
    let reader_events = events.clone();
    let reader = match thread::Builder::new()
        .name("ullm-sq8-reader".into())
        .spawn(move || run_sq8_command_reader(input, &reader_control, &reader_events, &commands))
    {
        Ok(reader) => reader,
        Err(_) => {
            let inference_result = inference.join();
            drop(events);
            let writer_result = writer.close_and_join().map(|_| ());
            return Err(join_worker_process_errors([
                (
                    "reader spawn",
                    Err("failed to spawn SQ8 reader thread".into()),
                ),
                ("inference join", inference_result),
                ("writer close", writer_result),
            ]));
        }
    };

    let inference_result = inference.join();
    let reader_result = if inference_result.is_ok() || reader.is_finished() {
        Some(
            reader
                .join()
                .map_err(|_| "SQ8 reader thread panicked".to_string())
                .and_then(|result| result),
        )
    } else {
        drop(reader);
        None
    };
    drop(events);
    let writer_result = writer.close_and_join().map(|_| ());

    if inference_result.is_ok()
        && reader_result.as_ref().is_some_and(Result::is_ok)
        && writer_result.is_ok()
    {
        return reader_result.expect("successful reader result was checked");
    }

    let mut failures = vec![("inference join", inference_result)];
    if let Some(reader_result) = reader_result {
        failures.push(("reader join", reader_result.map(|_| ())));
    }
    failures.push(("writer close", writer_result));
    Err(join_worker_process_errors(failures))
}

fn join_worker_process_errors<I>(results: I) -> String
where
    I: IntoIterator<Item = (&'static str, Result<(), String>)>,
{
    let failures = results
        .into_iter()
        .filter_map(|(stage, result)| result.err().map(|error| format!("{stage}: {error}")))
        .collect::<Vec<_>>();
    if failures.is_empty() {
        "SQ8 worker process failed without an error detail".into()
    } else {
        failures.join("; ")
    }
}

pub fn run_sq8_command_reader<R: BufRead>(
    input: R,
    control: &Arc<Sq8WorkerControl>,
    events: &Sq8WorkerEventPublisher,
    inference: &SyncSender<Sq8InferenceCommand>,
) -> Result<Sq8CommandReaderExit, String> {
    let mut reader = Sq8BoundedJsonlReader::new(input);
    loop {
        match reader.next_record() {
            Ok(Sq8JsonlRead::Record(payload)) => {
                if let Some(exit) = dispatch_record(&payload, control, events, inference)? {
                    return Ok(exit);
                }
            }
            Ok(Sq8JsonlRead::Oversized) => publish_recoverable(
                control,
                events,
                None,
                Sq8WorkerErrorCode::InvalidCommand,
                "worker command exceeds the 4 MiB record limit",
            )?,
            Ok(Sq8JsonlRead::Eof) => {
                return begin_reader_shutdown(control, events, inference);
            }
            Err(error) => {
                return Err(fail_reader_framing(control, events, error));
            }
        }
    }
}

fn dispatch_record(
    payload: &[u8],
    control: &Arc<Sq8WorkerControl>,
    events: &Sq8WorkerEventPublisher,
    inference: &SyncSender<Sq8InferenceCommand>,
) -> Result<Option<Sq8CommandReaderExit>, String> {
    let inspection = match inspect_sq8_worker_command(payload) {
        Ok(inspection) => inspection,
        Err(_) => {
            publish_recoverable(
                control,
                events,
                None,
                Sq8WorkerErrorCode::InvalidCommand,
                "worker command does not match the strict protocol schema",
            )?;
            return Ok(None);
        }
    };
    let inspected_request_id = inspection.request_id().map(str::to_string);

    if inspection.kind == Sq8WorkerCommandKind::Generate {
        match control.precheck_generate() {
            Ok(()) => {}
            Err(error) if error.kind == Sq8WorkerControlErrorKind::Busy => {
                publish_recoverable(
                    control,
                    events,
                    inspected_request_id,
                    Sq8WorkerErrorCode::Busy,
                    "one request is already active",
                )?;
                return Ok(None);
            }
            Err(_) => {
                return Err(fail_reader_internal(
                    control,
                    events,
                    "generate admission control failed",
                ));
            }
        }
    }

    let command = match inspection.decode() {
        Ok(command) => command,
        Err(error) => {
            let code = match error.kind {
                Sq8WorkerProtocolErrorKind::InvalidCommand => Sq8WorkerErrorCode::InvalidCommand,
                Sq8WorkerProtocolErrorKind::InvalidRequest => Sq8WorkerErrorCode::InvalidRequest,
            };
            publish_recoverable(
                control,
                events,
                inspected_request_id,
                code,
                "worker command failed protocol validation",
            )?;
            return Ok(None);
        }
    };

    match command {
        Sq8WorkerCommand::Generate(generate) => {
            let request_id = generate.request_id.clone();
            let event_request_id = inspected_request_id;
            let request = match generate.into_serving_request() {
                Ok(request) => request,
                Err(_) => {
                    publish_recoverable(
                        control,
                        events,
                        event_request_id,
                        Sq8WorkerErrorCode::InvalidRequest,
                        "generate request violates the fixed SQ8 product limits",
                    )?;
                    return Ok(None);
                }
            };
            let admission = control.admit(&request_id).map_err(|_| {
                fail_reader_internal(control, events, "validated generate admission failed")
            })?;
            let generation = admission.generation;
            if inference
                .send(Sq8InferenceCommand::Generate { request, admission })
                .is_err()
            {
                let error = fail_reader_internal(
                    control,
                    events,
                    "SQ8 inference channel closed during admission",
                );
                let _ = control.fail_admission_transfer(generation);
                return Err(error);
            }
            Ok(None)
        }
        Sq8WorkerCommand::Cancel { request_id, reason } => {
            match control.cancel(&request_id, reason) {
                Ok(_) => {}
                Err(error) if error.kind == Sq8WorkerControlErrorKind::UnknownRequest => {
                    publish_recoverable(
                        control,
                        events,
                        Some(request_id),
                        Sq8WorkerErrorCode::UnknownRequest,
                        "cancel request does not match the active request",
                    )?;
                }
                Err(_) => {
                    return Err(fail_reader_internal(
                        control,
                        events,
                        "cancel admission control failed",
                    ));
                }
            }
            Ok(None)
        }
        Sq8WorkerCommand::Shutdown => begin_reader_shutdown(control, events, inference).map(Some),
    }
}

fn begin_reader_shutdown(
    control: &Arc<Sq8WorkerControl>,
    events: &Sq8WorkerEventPublisher,
    inference: &SyncSender<Sq8InferenceCommand>,
) -> Result<Sq8CommandReaderExit, String> {
    match control.begin_shutdown() {
        Ok(Sq8WorkerShutdownDisposition::Idle) => {
            inference.send(Sq8InferenceCommand::Shutdown).map_err(|_| {
                fail_reader_internal(
                    control,
                    events,
                    "SQ8 inference channel closed during idle shutdown",
                )
            })?;
            Ok(Sq8CommandReaderExit::IdleShutdown)
        }
        Ok(Sq8WorkerShutdownDisposition::Cancelling(cancel)) => {
            Ok(Sq8CommandReaderExit::ActiveShutdown {
                generation: cancel.generation,
            })
        }
        Err(_) => Err(fail_reader_internal(
            control,
            events,
            "shutdown admission control failed",
        )),
    }
}

fn publish_recoverable(
    control: &Sq8WorkerControl,
    events: &Sq8WorkerEventPublisher,
    request_id: Option<String>,
    code: Sq8WorkerErrorCode,
    message: &'static str,
) -> Result<(), String> {
    if !code.recoverable() {
        return Err("SQ8 reader attempted to publish a fatal code as recoverable".into());
    }
    let event = Sq8WorkerEvent::error(request_id, code, message)
        .map_err(|_| "failed to construct SQ8 recoverable error event".to_string())?;
    events.publish(event).inspect_err(|_| {
        let _ = control.mark_failed();
    })
}

fn fail_reader_framing(
    control: &Arc<Sq8WorkerControl>,
    events: &Sq8WorkerEventPublisher,
    _error: Sq8JsonlFramingError,
) -> String {
    let request_id = active_request_id(control);
    let _ = control.mark_failed();
    publish_fatal_best_effort(
        events,
        request_id,
        Sq8WorkerErrorCode::ProtocolFramingFailed,
        "worker stdin ended inside an invalid JSONL frame",
    );
    "SQ8 worker stdin framing failed".into()
}

fn fail_reader_internal(
    control: &Arc<Sq8WorkerControl>,
    events: &Sq8WorkerEventPublisher,
    message: &'static str,
) -> String {
    let request_id = active_request_id(control);
    let _ = control.mark_failed();
    publish_fatal_best_effort(
        events,
        request_id,
        Sq8WorkerErrorCode::InvariantFailed,
        message,
    );
    message.into()
}

fn active_request_id(control: &Sq8WorkerControl) -> Option<String> {
    control
        .snapshot()
        .ok()
        .and_then(|snapshot| snapshot.active_request_id)
}

fn publish_fatal_best_effort(
    events: &Sq8WorkerEventPublisher,
    request_id: Option<String>,
    code: Sq8WorkerErrorCode,
    message: &'static str,
) {
    if let Ok(event) = Sq8WorkerEvent::error(request_id, code, message) {
        events.try_publish_fatal(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sq8_worker_protocol::{
        SQ8_WORKER_MAX_RECORD_BYTES, Sq8CancelReason, Sq8ReleaseOutcomeEvent, Sq8WorkerLifecycle,
    };
    use serde_json::Value;
    use std::io::{BufReader, Cursor, Write};
    use std::os::unix::net::UnixStream;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Condvar, Mutex, mpsc};
    use std::time::{Duration, Instant};

    fn valid_generate(request_id: &str) -> String {
        format!(
            "{{\"schema_version\":\"ullm.worker.v1\",\"type\":\"generate\",\"request_id\":\"{request_id}\",\"prompt_token_ids\":[1,2,3],\"max_new_tokens\":2,\"sampling\":{{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"seed\":-7}},\"eos_token_ids\":[151645,151643]}}"
        )
    }

    fn start_ready_writer() -> (
        Arc<Sq8WorkerControl>,
        Sq8WorkerEventPublisher,
        Sq8WriterThread<Vec<u8>>,
    ) {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
        let acknowledgement = events.publish_ready(Sq8WorkerEvent::ready()).unwrap();
        control.mark_ready_after_flush(acknowledgement).unwrap();
        (control, events, writer)
    }

    fn finish_writer(
        events: Sq8WorkerEventPublisher,
        writer: Sq8WriterThread<Vec<u8>>,
    ) -> Vec<Value> {
        let bytes = writer.close_and_join().unwrap();
        drop(events);
        bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect()
    }

    struct FakeInferenceBackend {
        fail: bool,
        fail_shutdown: bool,
        barrier: Option<(SyncSender<()>, Receiver<()>)>,
        after_release: Option<(SyncSender<()>, Receiver<()>)>,
        execution_completed: Option<mpsc::Sender<String>>,
        completed: Option<SyncSender<()>>,
        _thread_local_only: Rc<()>,
    }

    impl Sq8InferenceBackend for FakeInferenceBackend {
        fn execute(
            &mut self,
            request: Sq8ServingRequest,
            admission: Sq8WorkerAdmission,
            publications: &mut Sq8RequestEventPublisher<'_>,
        ) -> Result<(), String> {
            publications.publish_started()?;
            if let Some((entered, release)) = self.barrier.take() {
                entered
                    .send(())
                    .map_err(|_| "fake backend barrier receiver closed".to_string())?;
                release
                    .recv()
                    .map_err(|_| "fake backend barrier release closed".to_string())?;
            }
            if self.fail {
                return Err("injected fake backend failure".into());
            }

            let outcome = if admission.cancel.is_cancelled() {
                Sq8ReleaseOutcomeEvent::Cancelled
            } else {
                for processed in 1..=request.prompt_token_ids.len() {
                    publications.observe_prompt_unit(processed, 1)?;
                }
                for _ in 0..request.max_new_tokens {
                    publications.publish_token(1)?;
                }
                Sq8ReleaseOutcomeEvent::Length
            };
            publications.publish_released(outcome)?;
            if let Some((entered, release)) = self.after_release.take() {
                entered
                    .send(())
                    .map_err(|_| "fake post-release barrier receiver closed".to_string())?;
                release
                    .recv()
                    .map_err(|_| "fake post-release barrier release closed".to_string())?;
            }
            if let Some(completed) = &self.execution_completed {
                let _ = completed.send(request.request_id.clone());
            }
            if let Some(completed) = self.completed.take() {
                let _ = completed.send(());
            }
            Ok(())
        }

        fn shutdown(&mut self) -> Result<(), String> {
            if self.fail_shutdown {
                Err("injected fake backend shutdown failure".into())
            } else {
                Ok(())
            }
        }
    }

    fn fake_backend(
        fail: bool,
        barrier: Option<(SyncSender<()>, Receiver<()>)>,
        completed: Option<SyncSender<()>>,
    ) -> FakeInferenceBackend {
        FakeInferenceBackend {
            fail,
            fail_shutdown: false,
            barrier,
            after_release: None,
            execution_completed: None,
            completed,
            _thread_local_only: Rc::new(()),
        }
    }

    fn fake_backend_with_shutdown_failure(
        barrier: Option<(SyncSender<()>, Receiver<()>)>,
        completed: Option<SyncSender<()>>,
    ) -> FakeInferenceBackend {
        let mut backend = fake_backend(false, barrier, completed);
        backend.fail_shutdown = true;
        backend
    }

    #[derive(Default)]
    struct SharedProcessOutput {
        bytes: Mutex<Vec<u8>>,
        changed: Condvar,
        thread_ids: Mutex<Vec<thread::ThreadId>>,
        flushes: AtomicUsize,
    }

    impl SharedProcessOutput {
        fn lines(&self) -> Vec<Value> {
            self.bytes
                .lock()
                .unwrap()
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap())
                .collect()
        }

        fn record_thread(&self) {
            self.thread_ids.lock().unwrap().push(thread::current().id());
        }

        fn wait_for_line_count(&self, expected: usize, timeout: Duration) -> bool {
            let deadline = Instant::now() + timeout;
            let mut bytes = self.bytes.lock().unwrap();
            loop {
                if bytes.iter().filter(|byte| **byte == b'\n').count() >= expected {
                    return true;
                }
                let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let (next, wait) = self.changed.wait_timeout(bytes, remaining).unwrap();
                bytes = next;
                if wait.timed_out() {
                    return bytes.iter().filter(|byte| **byte == b'\n').count() >= expected;
                }
            }
        }
    }

    struct SharedProcessWriter {
        output: Arc<SharedProcessOutput>,
        fail_after_flushes: Option<usize>,
    }

    impl SharedProcessWriter {
        fn new(output: Arc<SharedProcessOutput>) -> Self {
            Self {
                output,
                fail_after_flushes: None,
            }
        }

        fn fail_after_flushes(output: Arc<SharedProcessOutput>, flushes: usize) -> Self {
            Self {
                output,
                fail_after_flushes: Some(flushes),
            }
        }
    }

    impl Write for SharedProcessWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.output.record_thread();
            if self
                .fail_after_flushes
                .is_some_and(|limit| self.output.flushes.load(AtomicOrdering::SeqCst) >= limit)
            {
                return Err(std::io::Error::other("injected process stdout failure"));
            }
            self.output.bytes.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.output.record_thread();
            self.output.flushes.fetch_add(1, AtomicOrdering::SeqCst);
            self.output.changed.notify_all();
            Ok(())
        }
    }

    struct WaitForCancellationBackend;

    impl Sq8InferenceBackend for WaitForCancellationBackend {
        fn execute(
            &mut self,
            _request: Sq8ServingRequest,
            admission: Sq8WorkerAdmission,
            publications: &mut Sq8RequestEventPublisher<'_>,
        ) -> Result<(), String> {
            publications.publish_started()?;
            let deadline = Instant::now() + Duration::from_secs(1);
            while !admission.cancel.is_cancelled() {
                if Instant::now() >= deadline {
                    return Err("process test did not observe shutdown cancellation".into());
                }
                thread::yield_now();
            }
            publications.publish_released(Sq8ReleaseOutcomeEvent::Cancelled)
        }
    }

    #[test]
    fn terminal_cleanup_watchdog_allows_fast_cleanup() {
        assert_eq!(SQ8_TERMINAL_CLEANUP_DEADLINE, Duration::from_secs(5));
        let (control, events, writer) = start_ready_writer();
        let request = Sq8ServingRequest::greedy("req-cleanup-fast", vec![1], 1);
        let admission = control.admit("req-cleanup-fast").unwrap();
        let mut publications =
            Sq8RequestEventPublisher::new(control.as_ref(), &events, &request, &admission).unwrap();
        publications.publish_started().unwrap();
        let armed = Arc::new(AtomicBool::new(false));
        let watchdog_armed = Arc::clone(&armed);

        let value = publications
            .run_terminal_cleanup_with_arm_hook(
                Duration::from_secs(1),
                || panic!("fast cleanup must not invoke the terminator"),
                move || watchdog_armed.store(true, Ordering::Release),
                || {
                    assert!(armed.load(Ordering::Acquire));
                    Ok(42)
                },
            )
            .unwrap();

        assert_eq!(value, 42);
        drop(publications);
        assert!(writer.close_and_join().is_ok());
        drop(events);
    }

    #[test]
    fn terminal_cleanup_completion_timestamp_must_precede_deadline() {
        let expires_at = Instant::now() + Duration::from_secs(1);
        let before = expires_at.checked_sub(Duration::from_nanos(1)).unwrap();
        let after = expires_at.checked_add(Duration::from_nanos(1)).unwrap();

        assert!(terminal_cleanup_completed_before_deadline(
            &Ok(before),
            expires_at
        ));
        assert!(!terminal_cleanup_completed_before_deadline(
            &Ok(expires_at),
            expires_at
        ));
        assert!(!terminal_cleanup_completed_before_deadline(
            &Ok(after),
            expires_at
        ));
        assert!(!terminal_cleanup_completed_before_deadline(
            &Err(RecvTimeoutError::Timeout),
            expires_at
        ));
    }

    #[test]
    fn terminal_cleanup_watchdog_poison_prevents_release_after_timeout() {
        let output = Arc::new(SharedProcessOutput::default());
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) =
            spawn_sq8_ordered_writer(SharedProcessWriter::new(Arc::clone(&output))).unwrap();
        let ready = events.publish_ready(Sq8WorkerEvent::ready()).unwrap();
        control.mark_ready_after_flush(ready).unwrap();
        let request = Sq8ServingRequest::greedy("req-cleanup-timeout", vec![1], 1);
        let admission = control.admit("req-cleanup-timeout").unwrap();
        let mut publications =
            Sq8RequestEventPublisher::new(control.as_ref(), &events, &request, &admission).unwrap();
        publications.publish_started().unwrap();
        control
            .cancel("req-cleanup-timeout", Sq8CancelReason::Operator)
            .unwrap();
        let holder_control = Arc::clone(&control);
        let (locked_sender, locked_receiver) = sync_channel(0);
        let (release_sender, release_receiver) = sync_channel::<()>(0);
        let holder = thread::spawn(move || {
            holder_control.with_state_lock_for_test(|| {
                locked_sender.send(()).unwrap();
                let _ = release_receiver.recv();
            });
        });
        locked_receiver.recv().unwrap();
        let terminated = Arc::new(AtomicBool::new(false));
        let watchdog_terminated = Arc::clone(&terminated);

        let error = publications
            .run_terminal_cleanup_with(
                Duration::from_millis(10),
                move || watchdog_terminated.store(true, Ordering::SeqCst),
                || {
                    thread::sleep(Duration::from_millis(30));
                    Ok(())
                },
            )
            .unwrap_err();
        release_sender.send(()).unwrap();
        holder.join().unwrap();

        assert!(error.contains("deadline exceeded"), "{error}");
        assert!(terminated.load(Ordering::SeqCst));
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Ready
        );
        assert!(
            publications
                .publish_released(Sq8ReleaseOutcomeEvent::Cancelled)
                .is_err()
        );
        drop(publications);
        assert!(writer.close_and_join().is_err());
        drop(events);
        let lines = output.lines();
        assert!(lines.iter().all(|line| line["type"] != "released"));
        assert!(lines.iter().any(|line| {
            line["type"] == "error" && line["code"] == "cleanup_deadline_exceeded"
        }));
    }

    #[test]
    fn terminal_cleanup_watchdog_counts_delayed_arming_against_deadline() {
        let (control, events, writer) = start_ready_writer();
        let request = Sq8ServingRequest::greedy("req-cleanup-delayed-arm", vec![1], 1);
        let admission = control.admit("req-cleanup-delayed-arm").unwrap();
        let mut publications =
            Sq8RequestEventPublisher::new(control.as_ref(), &events, &request, &admission).unwrap();
        publications.publish_started().unwrap();
        let terminated = Arc::new(AtomicBool::new(false));
        let watchdog_terminated = Arc::clone(&terminated);
        let cleanup_called = Arc::new(AtomicBool::new(false));
        let watchdog_cleanup_called = Arc::clone(&cleanup_called);

        let error = publications
            .run_terminal_cleanup_with_arm_hook(
                Duration::from_millis(10),
                move || watchdog_terminated.store(true, Ordering::Release),
                || thread::sleep(Duration::from_millis(30)),
                move || {
                    watchdog_cleanup_called.store(true, Ordering::Release);
                    Ok(())
                },
            )
            .unwrap_err();

        assert!(error.contains("deadline exceeded"), "{error}");
        assert!(terminated.load(Ordering::Acquire));
        assert!(!cleanup_called.load(Ordering::Acquire));
        drop(publications);
        assert!(writer.close_and_join().is_err());
        drop(events);
    }

    #[test]
    fn terminal_cleanup_watchdog_exits_child_process_nonzero() {
        const CHILD_ENV: &str = "ULLM_TEST_CLEANUP_WATCHDOG_CHILD";
        if std::env::var_os(CHILD_ENV).is_some() {
            let (control, events, _writer) = start_ready_writer();
            let request = Sq8ServingRequest::greedy("req-cleanup-child", vec![1], 1);
            let admission = control.admit("req-cleanup-child").unwrap();
            let mut publications =
                Sq8RequestEventPublisher::new(control.as_ref(), &events, &request, &admission)
                    .unwrap();
            publications.publish_started().unwrap();
            let holder_control = Arc::clone(&control);
            let (locked_sender, locked_receiver) = sync_channel(0);
            let (release_sender, release_receiver) = sync_channel::<()>(0);
            let _holder = thread::spawn(move || {
                holder_control.with_state_lock_for_test(|| {
                    locked_sender.send(()).unwrap();
                    let _ = release_receiver.recv();
                });
            });
            locked_receiver.recv().unwrap();
            let _lock_release = release_sender;
            let _ = publications.run_terminal_cleanup_with(
                Duration::from_millis(20),
                || std::process::exit(1),
                || {
                    thread::sleep(Duration::from_secs(10));
                    Ok(())
                },
            );
            std::process::exit(99);
        }

        let started = Instant::now();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "sq8_worker_runtime::tests::terminal_cleanup_watchdog_exits_child_process_nonzero",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD_ENV, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if started.elapsed() >= Duration::from_secs(2) {
                let _ = child.kill();
                let _ = child.wait();
                panic!("SQ8 cleanup watchdog child did not exit within two seconds");
            }
            thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(status.code(), Some(1));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn process_runner_idle_eof_joins_all_owners_and_uses_one_writer_thread() {
        let output = Arc::new(SharedProcessOutput::default());
        let caller_thread = thread::current().id();
        let exit = run_sq8_worker_process(
            Cursor::new(Vec::<u8>::new()),
            SharedProcessWriter::new(Arc::clone(&output)),
            || Ok(fake_backend(false, None, None)),
        )
        .unwrap();

        assert_eq!(exit, Sq8CommandReaderExit::IdleShutdown);
        let lines = output.lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "ready");
        let thread_ids = output.thread_ids.lock().unwrap();
        assert!(!thread_ids.is_empty());
        assert!(thread_ids.iter().all(|id| *id == thread_ids[0]));
        assert_ne!(thread_ids[0], caller_thread);
    }

    #[test]
    fn process_runner_explicit_idle_shutdown_exits_cleanly() {
        let output = Arc::new(SharedProcessOutput::default());
        let input = Cursor::new(
            b"{\"schema_version\":\"ullm.worker.v1\",\"type\":\"shutdown\"}\n".to_vec(),
        );
        let exit =
            run_sq8_worker_process(input, SharedProcessWriter::new(Arc::clone(&output)), || {
                Ok(fake_backend(false, None, None))
            })
            .unwrap();

        assert_eq!(exit, Sq8CommandReaderExit::IdleShutdown);
        assert_eq!(output.lines()[0]["type"], "ready");
    }

    #[test]
    fn process_runner_handles_two_sequential_requests_with_one_backend() {
        let (mut input_writer, input_reader) = UnixStream::pair().unwrap();
        let output = Arc::new(SharedProcessOutput::default());
        let process_output = Arc::clone(&output);
        let build_count = Arc::new(AtomicUsize::new(0));
        let process_build_count = Arc::clone(&build_count);
        let (completed, completion) = mpsc::channel();
        let process = thread::spawn(move || {
            let result = run_sq8_worker_process(
                BufReader::new(input_reader),
                SharedProcessWriter::new(process_output),
                move || {
                    process_build_count.fetch_add(1, AtomicOrdering::SeqCst);
                    Ok(fake_backend(false, None, None))
                },
            );
            completed.send(result).unwrap();
        });
        assert!(output.wait_for_line_count(1, Duration::from_secs(1)));

        writeln!(input_writer, "{}", valid_generate("req-process-a")).unwrap();
        input_writer.flush().unwrap();
        assert!(output.wait_for_line_count(6, Duration::from_secs(1)));
        assert_eq!(output.lines()[5]["request_id"], "req-process-a");

        writeln!(input_writer, "{}", valid_generate("req-process-b")).unwrap();
        input_writer.flush().unwrap();
        assert!(output.wait_for_line_count(11, Duration::from_secs(1)));
        assert_eq!(output.lines()[10]["request_id"], "req-process-b");

        writeln!(
            input_writer,
            "{{\"schema_version\":\"ullm.worker.v1\",\"type\":\"shutdown\"}}"
        )
        .unwrap();
        input_writer.flush().unwrap();
        let exit = completion
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        assert_eq!(exit, Sq8CommandReaderExit::IdleShutdown);
        assert_eq!(build_count.load(AtomicOrdering::SeqCst), 1);

        let lines = output.lines();
        let tokens = lines
            .iter()
            .filter(|line| line["type"] == "token")
            .map(|line| {
                (
                    line["request_id"].as_str().unwrap(),
                    line["index"].as_u64().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            tokens,
            [
                ("req-process-a", 0),
                ("req-process-a", 1),
                ("req-process-b", 0),
                ("req-process-b", 1),
            ]
        );
        drop(input_writer);
        process.join().unwrap();
    }

    #[test]
    fn process_runner_active_eof_cancels_releases_and_then_shuts_down() {
        let output = Arc::new(SharedProcessOutput::default());
        let input = Cursor::new(format!("{}\n", valid_generate("req-eof-process")).into_bytes());
        let exit =
            run_sq8_worker_process(input, SharedProcessWriter::new(Arc::clone(&output)), || {
                Ok(WaitForCancellationBackend)
            })
            .unwrap();

        assert_eq!(exit, Sq8CommandReaderExit::ActiveShutdown { generation: 1 });
        let lines = output.lines();
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["ready", "started", "released"]
        );
        assert_eq!(lines[2]["outcome"], "cancelled");
        assert_eq!(lines[2]["cancel_reason"], "shutdown");
    }

    #[test]
    fn process_runner_load_failure_is_nonzero_without_ready() {
        let output = Arc::new(SharedProcessOutput::default());
        let error = run_sq8_worker_process(
            Cursor::new(Vec::<u8>::new()),
            SharedProcessWriter::new(Arc::clone(&output)),
            || Err::<FakeInferenceBackend, _>("injected process load failure".into()),
        )
        .unwrap_err();

        assert!(error.contains("startup"), "{error}");
        let lines = output.lines();
        assert!(lines.iter().all(|line| line["type"] != "ready"));
        assert!(lines.iter().any(|line| line["type"] == "error"));
    }

    #[test]
    fn process_runner_shutdown_failure_changes_clean_eof_to_nonzero() {
        let output = Arc::new(SharedProcessOutput::default());
        let error = run_sq8_worker_process(
            Cursor::new(Vec::<u8>::new()),
            SharedProcessWriter::new(Arc::clone(&output)),
            || Ok(fake_backend_with_shutdown_failure(None, None)),
        )
        .unwrap_err();

        assert!(error.contains("shutdown failure"), "{error}");
        let lines = output.lines();
        assert_eq!(lines[0]["type"], "ready");
        assert!(lines.iter().any(|line| line["type"] == "error"));
    }

    #[test]
    fn process_runner_framing_fatal_wakes_idle_inference() {
        let output = Arc::new(SharedProcessOutput::default());
        let started = Instant::now();
        let error = run_sq8_worker_process(
            Cursor::new(b"{".to_vec()),
            SharedProcessWriter::new(Arc::clone(&output)),
            || Ok(fake_backend(false, None, None)),
        )
        .unwrap_err();

        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(
            error.contains("framing") || error.contains("poison"),
            "{error}"
        );
        let lines = output.lines();
        assert_eq!(lines[0]["type"], "ready");
        assert!(lines.iter().any(|line| line["type"] == "error"));
    }

    #[test]
    fn process_runner_inference_fatal_does_not_join_blocked_stdin_reader() {
        let (mut input_writer, input_reader) = UnixStream::pair().unwrap();
        let output = Arc::new(SharedProcessOutput::default());
        let process_output = Arc::clone(&output);
        let (completed, completion) = mpsc::channel();
        let process = thread::spawn(move || {
            let result = run_sq8_worker_process(
                BufReader::new(input_reader),
                SharedProcessWriter::new(process_output),
                || Ok(fake_backend(true, None, None)),
            );
            completed.send(result).unwrap();
        });
        writeln!(input_writer, "{}", valid_generate("req-blocked-fatal")).unwrap();
        input_writer.flush().unwrap();

        let result = completion.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(result.is_err());
        let lines = output.lines();
        assert!(lines.iter().any(|line| line["type"] == "ready"));
        assert!(lines.iter().any(|line| line["type"] == "started"));
        assert!(lines.iter().all(|line| line["type"] != "released"));
        drop(input_writer);
        process.join().unwrap();
    }

    #[test]
    fn process_runner_stdout_failure_does_not_join_blocked_stdin_reader() {
        let (mut input_writer, input_reader) = UnixStream::pair().unwrap();
        let output = Arc::new(SharedProcessOutput::default());
        let process_output = Arc::clone(&output);
        let (completed, completion) = mpsc::channel();
        let process = thread::spawn(move || {
            let result = run_sq8_worker_process(
                BufReader::new(input_reader),
                SharedProcessWriter::fail_after_flushes(process_output, 1),
                || Ok(fake_backend(false, None, None)),
            );
            completed.send(result).unwrap();
        });
        writeln!(input_writer, "{}", valid_generate("req-blocked-writer")).unwrap();
        input_writer.flush().unwrap();

        let result = completion.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(result.is_err());
        let lines = output.lines();
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["ready"]
        );
        drop(input_writer);
        process.join().unwrap();
    }

    #[test]
    fn writer_thread_flushes_ready_regular_and_terminal_events_in_order() {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
        let ready = events.publish_ready(Sq8WorkerEvent::ready()).unwrap();
        control.mark_ready_after_flush(ready).unwrap();
        let admission = control.admit("req-1").unwrap();
        events.publish(Sq8WorkerEvent::started("req-1", 3)).unwrap();
        let released = Sq8WorkerEvent::released(
            "req-1",
            crate::sq8_worker_protocol::Sq8ReleaseOutcomeEvent::Length,
            None,
            3,
            1,
        )
        .unwrap();
        let permit = control
            .begin_terminal_publication(admission.generation, &admission.request_id)
            .unwrap();
        let terminal = events.publish_active_terminal(permit, released).unwrap();
        control.acknowledge_terminal_flush(terminal).unwrap();
        let lines = finish_writer(events, writer);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["type"], "ready");
        assert_eq!(lines[1]["type"], "started");
        assert_eq!(lines[2]["type"], "released");
        assert!(control.snapshot().unwrap().active_request_id.is_none());
    }

    #[test]
    fn request_publisher_enforces_identity_order_progress_and_token_indices() {
        let (control, events, writer) = start_ready_writer();
        let request = Sq8ServingRequest::greedy("req-scoped", vec![1; 129], 2);
        let admission = control.admit("req-scoped").unwrap();
        {
            let mut publications =
                Sq8RequestEventPublisher::new(control.as_ref(), &events, &request, &admission)
                    .unwrap();
            assert!(publications.observe_prompt_unit(128, 128).is_err());
            assert!(publications.publish_token(1).is_err());
            assert!(
                publications
                    .publish_released(Sq8ReleaseOutcomeEvent::Length)
                    .is_err()
            );
            publications.publish_started().unwrap();
            assert!(publications.publish_started().is_err());
            assert!(publications.observe_prompt_unit(1, 1).is_err());
            publications.observe_prompt_unit(128, 128).unwrap();
            assert!(publications.observe_prompt_unit(129, 128).is_err());
            publications.observe_prompt_unit(129, 1).unwrap();
            assert!(publications.publish_token(QWEN3_14B_VOCAB_SIZE).is_err());
            publications.publish_token(7).unwrap();
            assert!(publications.observe_prefill_transition().is_err());
            assert!(
                publications
                    .publish_released(Sq8ReleaseOutcomeEvent::Length)
                    .is_err()
            );
            publications.publish_token(8).unwrap();
            publications
                .publish_released(Sq8ReleaseOutcomeEvent::Length)
                .unwrap();
            assert!(
                publications
                    .publish_released(Sq8ReleaseOutcomeEvent::Length)
                    .is_err()
            );
            assert_eq!(publications.completion_tokens(), 2);
        }
        let lines = finish_writer(events, writer);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "ready", "started", "progress", "progress", "token", "token", "released"
            ]
        );
        assert_eq!(lines[2]["processed_prompt_tokens"], 128);
        assert_eq!(lines[3]["processed_prompt_tokens"], 129);
        assert_eq!(lines[4]["index"], 0);
        assert_eq!(lines[5]["index"], 1);
    }

    #[test]
    fn request_publisher_gives_flushed_eos_precedence_over_later_cancel() {
        let (control, events, writer) = start_ready_writer();
        let request = Sq8ServingRequest::greedy("req-eos", vec![1], 3);
        let admission = control.admit("req-eos").unwrap();
        {
            let mut publications =
                Sq8RequestEventPublisher::new(control.as_ref(), &events, &request, &admission)
                    .unwrap();
            publications.publish_started().unwrap();
            publications.observe_prompt_unit(1, 1).unwrap();
            publications.publish_token(7).unwrap();
            assert!(
                publications
                    .publish_released(Sq8ReleaseOutcomeEvent::Stop)
                    .is_err()
            );
            publications.publish_token(151_645).unwrap();
            control
                .cancel("req-eos", Sq8CancelReason::Operator)
                .unwrap();
            assert!(publications.publish_token(8).is_err());
            assert!(
                publications
                    .publish_released(Sq8ReleaseOutcomeEvent::Length)
                    .is_err()
            );
            assert!(
                publications
                    .publish_released(Sq8ReleaseOutcomeEvent::Cancelled)
                    .is_err()
            );
            publications
                .publish_released(Sq8ReleaseOutcomeEvent::Stop)
                .unwrap();
        }
        let lines = finish_writer(events, writer);
        assert_eq!(lines.last().unwrap()["type"], "released");
        assert_eq!(lines.last().unwrap()["outcome"], "stop");
        assert_eq!(lines.last().unwrap()["completion_tokens"], 2);
        assert!(lines.last().unwrap().get("cancel_reason").is_none());
    }

    #[test]
    fn inference_thread_runs_normal_request_and_clean_idle_shutdown() {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let (completed, completion) = sync_channel(0);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || Ok(fake_backend(false, None, Some(completed))),
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        assert!(inference.wait_until_ready().is_err());
        let request = Sq8ServingRequest::greedy("req-1", vec![1, 2, 3], 1);
        let admission = control.admit("req-1").unwrap();
        commands
            .send(Sq8InferenceCommand::Generate { request, admission })
            .unwrap();
        completion.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(control.snapshot().unwrap().active_generation.is_none());
        assert_eq!(
            control.begin_shutdown().unwrap(),
            Sq8WorkerShutdownDisposition::Idle
        );
        commands.send(Sq8InferenceCommand::Shutdown).unwrap();
        inference.join().unwrap();
        let lines = finish_writer(events, writer);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "started", "progress", "token", "released"]
        );
        assert_eq!(lines[4]["outcome"], "length");
        assert_eq!(lines[4]["completion_tokens"], 1);
    }

    #[test]
    fn released_request_can_queue_the_next_generation_before_execute_returns() {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let (first_released, first_released_rx) = sync_channel(0);
        let (resume, resume_rx) = sync_channel(0);
        let (completed, completion) = mpsc::channel();
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || {
                let mut backend = fake_backend(false, None, None);
                backend.after_release = Some((first_released, resume_rx));
                backend.execution_completed = Some(completed);
                Ok(backend)
            },
        )
        .unwrap();
        inference.wait_until_ready().unwrap();

        for request_id in ["req-a", "req-b"] {
            let request = Sq8ServingRequest::greedy(request_id, vec![1, 2, 3], 1);
            let admission = if request_id == "req-a" {
                control.admit(request_id).unwrap()
            } else {
                first_released_rx
                    .recv_timeout(Duration::from_secs(1))
                    .unwrap();
                let admission = control.admit(request_id).unwrap();
                resume.send(()).unwrap();
                admission
            };
            commands
                .send(Sq8InferenceCommand::Generate { request, admission })
                .unwrap();
        }
        assert_eq!(
            completion.recv_timeout(Duration::from_secs(1)).unwrap(),
            "req-a"
        );
        assert_eq!(
            completion.recv_timeout(Duration::from_secs(1)).unwrap(),
            "req-b"
        );
        assert_eq!(
            control.begin_shutdown().unwrap(),
            Sq8WorkerShutdownDisposition::Idle
        );
        commands.send(Sq8InferenceCommand::Shutdown).unwrap();
        inference.join().unwrap();
        let lines = finish_writer(events, writer);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "ready", "started", "progress", "token", "released", "started", "progress",
                "token", "released"
            ]
        );
    }

    #[test]
    fn queued_next_generation_is_drained_when_eof_closes_the_worker() {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let (first_released, first_released_rx) = sync_channel(0);
        let (resume, resume_rx) = sync_channel(0);
        let (completed, completion) = mpsc::channel();
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || {
                let mut backend = fake_backend(false, None, None);
                backend.after_release = Some((first_released, resume_rx));
                backend.execution_completed = Some(completed);
                Ok(backend)
            },
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        let request_a = Sq8ServingRequest::greedy("req-a", vec![1, 2, 3], 1);
        let admission_a = control.admit("req-a").unwrap();
        commands
            .send(Sq8InferenceCommand::Generate {
                request: request_a,
                admission: admission_a,
            })
            .unwrap();
        first_released_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        let request_b = Sq8ServingRequest::greedy("req-b", vec![1, 2, 3], 1);
        let admission_b = control.admit("req-b").unwrap();
        commands
            .send(Sq8InferenceCommand::Generate {
                request: request_b,
                admission: admission_b,
            })
            .unwrap();
        assert!(matches!(
            run_sq8_command_reader(
                BufReader::new(Cursor::new(Vec::<u8>::new())),
                &control,
                &events,
                &commands,
            )
            .unwrap(),
            Sq8CommandReaderExit::ActiveShutdown { .. }
        ));
        resume.send(()).unwrap();
        assert_eq!(
            completion.recv_timeout(Duration::from_secs(1)).unwrap(),
            "req-a"
        );
        assert_eq!(
            completion.recv_timeout(Duration::from_secs(1)).unwrap(),
            "req-b"
        );
        inference.join().unwrap();
        let lines = finish_writer(events, writer);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![
                "ready", "started", "progress", "token", "released", "started", "released"
            ]
        );
        assert_eq!(lines[6]["outcome"], "cancelled");
        assert_eq!(lines[6]["cancel_reason"], "shutdown");
    }

    #[test]
    fn inference_thread_keeps_started_before_cancelled_release() {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let (completed, completion) = sync_channel(0);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || Ok(fake_backend(false, None, Some(completed))),
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        let request = Sq8ServingRequest::greedy("req-cancel", vec![1, 2, 3], 1);
        let admission = control.admit("req-cancel").unwrap();
        control
            .cancel("req-cancel", Sq8CancelReason::Operator)
            .unwrap();
        commands
            .send(Sq8InferenceCommand::Generate { request, admission })
            .unwrap();
        completion.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(
            control.begin_shutdown().unwrap(),
            Sq8WorkerShutdownDisposition::Idle
        );
        commands.send(Sq8InferenceCommand::Shutdown).unwrap();
        inference.join().unwrap();
        let lines = finish_writer(events, writer);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "started", "released"]
        );
        assert_eq!(lines[2]["outcome"], "cancelled");
        assert_eq!(lines[2]["cancel_reason"], "operator");
        assert_eq!(lines[2]["completion_tokens"], 0);
    }

    #[test]
    fn active_eof_drains_release_and_joins_the_inference_thread() {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let (entered, entered_rx) = sync_channel(0);
        let (release, release_rx) = sync_channel(0);
        let (completed, completion) = sync_channel(0);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || {
                Ok(fake_backend(
                    false,
                    Some((entered, release_rx)),
                    Some(completed),
                ))
            },
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        let input = format!("{}\n", valid_generate("req-eof"));
        let exit = run_sq8_command_reader(
            BufReader::with_capacity(1, Cursor::new(input.into_bytes())),
            &control,
            &events,
            &commands,
        )
        .unwrap();
        assert!(matches!(exit, Sq8CommandReaderExit::ActiveShutdown { .. }));
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        release.send(()).unwrap();
        completion.recv_timeout(Duration::from_secs(1)).unwrap();
        inference.join().unwrap();
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Closing
        );
        assert!(control.snapshot().unwrap().active_generation.is_none());
        let lines = finish_writer(events, writer);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "started", "released"]
        );
        assert_eq!(lines[2]["outcome"], "cancelled");
        assert_eq!(lines[2]["cancel_reason"], "shutdown");
    }

    #[test]
    fn operator_cancel_then_eof_preserves_the_first_cancel_reason() {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let (entered, entered_rx) = sync_channel(0);
        let (release, release_rx) = sync_channel(0);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || Ok(fake_backend(false, Some((entered, release_rx)), None)),
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        let input = format!(
            "{}\n{{\"schema_version\":\"ullm.worker.v1\",\"type\":\"cancel\",\"request_id\":\"req-first-reason\",\"reason\":\"operator\"}}\n",
            valid_generate("req-first-reason")
        );
        assert!(matches!(
            run_sq8_command_reader(
                BufReader::new(Cursor::new(input.into_bytes())),
                &control,
                &events,
                &commands,
            )
            .unwrap(),
            Sq8CommandReaderExit::ActiveShutdown { .. }
        ));
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        release.send(()).unwrap();
        inference.join().unwrap();
        let lines = finish_writer(events, writer);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "started", "released"]
        );
        assert_eq!(lines[2]["outcome"], "cancelled");
        assert_eq!(lines[2]["cancel_reason"], "operator");
    }

    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn direct_join_before_startup_ack_is_bounded_and_fatal() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(SharedWriter(Arc::clone(&output))).unwrap();
        let (_commands, command_receiver) = sync_channel(1);
        let inference =
            spawn_sq8_inference_thread(Arc::clone(&control), events, command_receiver, move || {
                Ok(fake_backend(false, None, None))
            })
            .unwrap();
        assert!(inference.join().is_err());
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Failed
        );
        assert!(writer.close_and_join().is_err());
        let lines = shared_output_lines(&output);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "error"]
        );
        assert_eq!(lines[1]["code"], "runtime_failed");
    }

    #[test]
    fn backend_load_failure_emits_fatal_without_ready() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(SharedWriter(Arc::clone(&output))).unwrap();
        let (_commands, command_receiver) = sync_channel(1);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events,
            command_receiver,
            move || -> Result<FakeInferenceBackend, String> {
                Err("injected backend load failure".into())
            },
        )
        .unwrap();
        assert!(inference.wait_until_ready().is_err());
        assert!(inference.join().is_err());
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Failed
        );
        assert!(writer.close_and_join().is_err());
        let lines = shared_output_lines(&output);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["type"], "error");
        assert_eq!(lines[0]["code"], "load_failed");
    }

    #[test]
    fn idle_framing_fatal_wakes_and_joins_the_inference_thread() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(SharedWriter(Arc::clone(&output))).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || Ok(fake_backend(false, None, None)),
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        assert!(
            run_sq8_command_reader(
                BufReader::new(Cursor::new(b"{".to_vec())),
                &control,
                &events,
                &commands,
            )
            .is_err()
        );
        assert!(inference.join().is_err());
        assert!(writer.close_and_join().is_err());
        let lines = shared_output_lines(&output);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "error"]
        );
        assert_eq!(lines[1]["code"], "protocol_framing_failed");
    }

    #[test]
    fn active_framing_fatal_prevents_later_token_and_release() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(SharedWriter(Arc::clone(&output))).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let (entered, entered_rx) = sync_channel(0);
        let (release, release_rx) = sync_channel(0);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || Ok(fake_backend(false, Some((entered, release_rx)), None)),
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        assert!(
            dispatch_record(
                valid_generate("req-framing").as_bytes(),
                &control,
                &events,
                &commands,
            )
            .unwrap()
            .is_none()
        );
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            run_sq8_command_reader(
                BufReader::new(Cursor::new(b"{".to_vec())),
                &control,
                &events,
                &commands,
            )
            .is_err()
        );
        release.send(()).unwrap();
        assert!(inference.join().is_err());
        assert!(control.snapshot().unwrap().active_generation.is_some());
        assert!(writer.close_and_join().is_err());
        let lines = shared_output_lines(&output);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "started", "error"]
        );
        assert!(!lines.iter().any(|line| line["type"] == "token"));
        assert!(!lines.iter().any(|line| line["type"] == "released"));
    }

    #[test]
    fn inference_backend_failure_is_fatal_and_emits_no_release() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(SharedWriter(Arc::clone(&output))).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || Ok(fake_backend(true, None, None)),
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        let request = Sq8ServingRequest::greedy("req-fail", vec![1, 2, 3], 1);
        let admission = control.admit("req-fail").unwrap();
        commands
            .send(Sq8InferenceCommand::Generate { request, admission })
            .unwrap();
        assert!(inference.join().is_err());
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Failed
        );
        assert!(control.snapshot().unwrap().active_generation.is_some());
        assert!(writer.close_and_join().is_err());
        let lines = shared_output_lines(&output);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "started", "error"]
        );
        assert_eq!(lines[2]["code"], "runtime_failed");
        assert!(!lines.iter().any(|line| line["type"] == "released"));
    }

    #[test]
    fn idle_backend_shutdown_failure_is_fatal() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(SharedWriter(Arc::clone(&output))).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let inference =
            spawn_sq8_inference_thread(Arc::clone(&control), events, command_receiver, move || {
                Ok(fake_backend_with_shutdown_failure(None, None))
            })
            .unwrap();
        inference.wait_until_ready().unwrap();
        assert_eq!(
            control.begin_shutdown().unwrap(),
            Sq8WorkerShutdownDisposition::Idle
        );
        commands.send(Sq8InferenceCommand::Shutdown).unwrap();
        assert!(inference.join().is_err());
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Failed
        );
        assert!(writer.close_and_join().is_err());
        let lines = shared_output_lines(&output);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "error"]
        );
        assert_eq!(lines[1]["code"], "runtime_failed");
    }

    #[test]
    fn active_eof_backend_shutdown_failure_is_fatal_after_release() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(SharedWriter(Arc::clone(&output))).unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let (entered, entered_rx) = sync_channel(0);
        let (release, release_rx) = sync_channel(0);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || {
                Ok(fake_backend_with_shutdown_failure(
                    Some((entered, release_rx)),
                    None,
                ))
            },
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        let input = format!("{}\n", valid_generate("req-shutdown-fail"));
        assert!(matches!(
            run_sq8_command_reader(
                BufReader::with_capacity(1, Cursor::new(input.into_bytes())),
                &control,
                &events,
                &commands,
            )
            .unwrap(),
            Sq8CommandReaderExit::ActiveShutdown { .. }
        ));
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        release.send(()).unwrap();
        assert!(inference.join().is_err());
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Failed
        );
        assert!(control.snapshot().unwrap().active_generation.is_none());
        assert!(writer.close_and_join().is_err());
        let lines = shared_output_lines(&output);
        assert_eq!(
            lines
                .iter()
                .map(|line| line["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["ready", "started", "released", "error"]
        );
        assert_eq!(lines[2]["outcome"], "cancelled");
        assert_eq!(lines[3]["code"], "runtime_failed");
    }

    struct BlockingFlushWriter {
        entered: SyncSender<()>,
        release: Receiver<()>,
    }

    struct NthBlockingFlushWriter {
        output: Arc<Mutex<Vec<u8>>>,
        block_on: usize,
        flushes: usize,
        entered: SyncSender<()>,
        release: Receiver<()>,
    }

    impl Write for NthBlockingFlushWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.output.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.flushes += 1;
            if self.flushes == self.block_on {
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
            }
            Ok(())
        }
    }

    #[test]
    fn eof_during_terminal_flush_keeps_the_committed_normal_outcome() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let (entered, entered_rx) = sync_channel(0);
        let (release, release_rx) = sync_channel(0);
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(NthBlockingFlushWriter {
            output: Arc::clone(&output),
            block_on: 5,
            flushes: 0,
            entered,
            release: release_rx,
        })
        .unwrap();
        let (commands, command_receiver) = sync_channel(1);
        let inference = spawn_sq8_inference_thread(
            Arc::clone(&control),
            events.clone(),
            command_receiver,
            move || Ok(fake_backend(false, None, None)),
        )
        .unwrap();
        inference.wait_until_ready().unwrap();
        let request = Sq8ServingRequest::greedy("req-terminal-race", vec![1, 2, 3], 1);
        let admission = control.admit("req-terminal-race").unwrap();
        commands
            .send(Sq8InferenceCommand::Generate { request, admission })
            .unwrap();
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(matches!(
            run_sq8_command_reader(
                BufReader::new(Cursor::new(Vec::<u8>::new())),
                &control,
                &events,
                &commands,
            )
            .unwrap(),
            Sq8CommandReaderExit::ActiveShutdown { .. }
        ));
        release.send(()).unwrap();
        inference.join().unwrap();
        writer.close_and_join().unwrap();
        let lines = shared_output_lines(&output);
        assert_eq!(lines.last().unwrap()["type"], "released");
        assert_eq!(lines.last().unwrap()["outcome"], "length");
        assert!(lines.last().unwrap().get("cancel_reason").is_none());
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Closing
        );
        assert!(control.snapshot().unwrap().active_generation.is_none());
    }

    impl Write for BlockingFlushWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            Ok(())
        }
    }

    #[test]
    fn publisher_acknowledgement_waits_for_the_real_flush_boundary() {
        let (entered_tx, entered_rx) = sync_channel(0);
        let (release_tx, release_rx) = sync_channel(0);
        let (events, writer) = spawn_sq8_ordered_writer(BlockingFlushWriter {
            entered: entered_tx,
            release: release_rx,
        })
        .unwrap();
        let publisher = events.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let publication = std::thread::spawn(move || {
            done_tx
                .send(publisher.publish_ready(Sq8WorkerEvent::ready()))
                .unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(done_rx.try_recv().is_err());
        release_tx.send(()).unwrap();
        done_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        publication.join().unwrap();
        writer.close_and_join().unwrap();
        drop(events);
    }

    struct BlockSecondFlushWriter {
        output: Arc<Mutex<Vec<u8>>>,
        flush_count: usize,
        blocked: SyncSender<()>,
        release: Receiver<()>,
    }

    impl Write for BlockSecondFlushWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.output.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            if self.flush_count == 1 {
                self.blocked.send(()).unwrap();
                self.release.recv().unwrap();
            }
            self.flush_count += 1;
            Ok(())
        }
    }

    fn shared_output_lines(output: &Arc<Mutex<Vec<u8>>>) -> Vec<Value> {
        output
            .lock()
            .unwrap()
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect()
    }

    #[test]
    fn fatal_publication_is_nonblocking_and_rejects_later_events() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let (blocked_tx, blocked_rx) = sync_channel(0);
        let (release_tx, release_rx) = sync_channel(0);
        let (events, writer) = spawn_sq8_ordered_writer(BlockSecondFlushWriter {
            output: Arc::clone(&output),
            flush_count: 0,
            blocked: blocked_tx,
            release: release_rx,
        })
        .unwrap();
        events.publish_ready(Sq8WorkerEvent::ready()).unwrap();
        let publisher = events.clone();
        let (published_tx, published_rx) = mpsc::channel();
        let publication = std::thread::spawn(move || {
            published_tx
                .send(publisher.publish(Sq8WorkerEvent::started("req-1", 3)))
                .unwrap();
        });
        blocked_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let fatal = Sq8WorkerEvent::error(
            Some("req-1".into()),
            Sq8WorkerErrorCode::RuntimeFailed,
            "runtime failed",
        )
        .unwrap();
        assert!(events.try_publish_fatal(fatal));
        assert!(
            events
                .publish(Sq8WorkerEvent::token("req-1", 0, 1))
                .is_err()
        );
        assert!(published_rx.try_recv().is_err());
        release_tx.send(()).unwrap();
        published_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        publication.join().unwrap();
        assert!(writer.close_and_join().is_err());
        let lines = shared_output_lines(&output);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["type"], "ready");
        assert_eq!(lines[1]["type"], "started");
        assert_eq!(lines[2]["code"], "runtime_failed");
    }

    #[test]
    fn queue_full_fatal_poison_discards_the_queued_regular_event() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let (blocked_tx, blocked_rx) = sync_channel(0);
        let (release_tx, release_rx) = sync_channel(0);
        let (events, writer) = spawn_sq8_ordered_writer(BlockSecondFlushWriter {
            output: Arc::clone(&output),
            flush_count: 0,
            blocked: blocked_tx,
            release: release_rx,
        })
        .unwrap();
        events.publish_ready(Sq8WorkerEvent::ready()).unwrap();
        let publisher = events.clone();
        let (first_tx, first_rx) = mpsc::channel();
        let first = std::thread::spawn(move || {
            first_tx
                .send(publisher.publish(Sq8WorkerEvent::started("req-1", 3)))
                .unwrap();
        });
        blocked_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let (queued_ack, queued_result) = sync_channel(0);
        events
            .sender
            .try_send(Sq8WriterEnvelope {
                publication: Sq8WriterPublication::Regular,
                event: Some(Sq8WorkerEvent::progress("req-1", 1)),
                acknowledgement: Some(queued_ack),
            })
            .unwrap();
        let fatal = Sq8WorkerEvent::error(
            Some("req-1".into()),
            Sq8WorkerErrorCode::RuntimeFailed,
            "runtime failed",
        )
        .unwrap();
        assert!(!events.try_publish_fatal(fatal));
        release_tx.send(()).unwrap();
        first_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        first.join().unwrap();
        assert!(
            queued_result
                .recv_timeout(Duration::from_secs(1))
                .unwrap()
                .is_err()
        );
        assert!(writer.close_and_join().is_err());
        let lines = shared_output_lines(&output);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["type"], "ready");
        assert_eq!(lines[1]["type"], "started");
    }

    #[test]
    fn generic_publication_rejects_ready_released_and_fatal_events() {
        let boundaries = [
            Sq8WorkerEvent::ready(),
            Sq8WorkerEvent::released(
                "req-1",
                crate::sq8_worker_protocol::Sq8ReleaseOutcomeEvent::Length,
                None,
                3,
                1,
            )
            .unwrap(),
            Sq8WorkerEvent::error(
                Some("req-1".into()),
                Sq8WorkerErrorCode::RuntimeFailed,
                "runtime failed",
            )
            .unwrap(),
        ];
        for event in boundaries {
            let (events, writer) = spawn_sq8_ordered_writer(Vec::new()).unwrap();
            assert!(events.publish(event).is_err());
            assert!(events.publish(Sq8WorkerEvent::started("req-1", 3)).is_err());
            assert!(writer.close_and_join().is_err());
        }
    }

    #[test]
    fn clean_idle_eof_requests_zero_exit_shutdown() {
        let (control, events, writer) = start_ready_writer();
        let (inference_tx, inference_rx) = sync_channel(1);
        let exit = run_sq8_command_reader(
            BufReader::new(Cursor::new(Vec::<u8>::new())),
            &control,
            &events,
            &inference_tx,
        )
        .unwrap();
        assert_eq!(exit, Sq8CommandReaderExit::IdleShutdown);
        assert!(matches!(
            inference_rx.recv().unwrap(),
            Sq8InferenceCommand::Shutdown
        ));
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Closing
        );
        let lines = finish_writer(events, writer);
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn active_eof_cancels_admitted_request_without_abandoning_it() {
        let (control, events, writer) = start_ready_writer();
        let (inference_tx, inference_rx) = sync_channel(1);
        let input = format!("{}\n", valid_generate("req-1"));
        let exit = run_sq8_command_reader(
            BufReader::with_capacity(1, Cursor::new(input.into_bytes())),
            &control,
            &events,
            &inference_tx,
        )
        .unwrap();
        let Sq8InferenceCommand::Generate { admission, .. } = inference_rx.recv().unwrap() else {
            panic!("expected admitted request")
        };
        assert_eq!(
            exit,
            Sq8CommandReaderExit::ActiveShutdown {
                generation: admission.generation
            }
        );
        assert!(admission.cancel.is_cancelled());
        let snapshot = control.snapshot().unwrap();
        assert_eq!(snapshot.lifecycle, Sq8WorkerLifecycle::Closing);
        assert_eq!(snapshot.active_request_id.as_deref(), Some("req-1"));
        assert_eq!(
            snapshot.first_cancel_reason,
            Some(Sq8CancelReason::Shutdown)
        );
        finish_writer(events, writer);
    }

    #[test]
    fn busy_unknown_and_repeated_cancel_leave_the_active_generation_intact() {
        let (control, events, writer) = start_ready_writer();
        let (inference_tx, inference_rx) = sync_channel(1);
        let huge_prompt = std::iter::repeat_n("1", 5000).collect::<Vec<_>>().join(",");
        let second = valid_generate("req-2").replace(
            "\"prompt_token_ids\":[1,2,3]",
            &format!("\"prompt_token_ids\":[{huge_prompt}]"),
        );
        let input = format!(
            "{}\n{}\n{{\"schema_version\":\"ullm.worker.v1\",\"type\":\"cancel\",\"request_id\":\"wrong\",\"reason\":\"operator\"}}\n{{\"schema_version\":\"ullm.worker.v1\",\"type\":\"cancel\",\"request_id\":\"req-1\",\"reason\":\"client_disconnect\"}}\n{{\"schema_version\":\"ullm.worker.v1\",\"type\":\"cancel\",\"request_id\":\"req-1\",\"reason\":\"operator\"}}\n",
            valid_generate("req-1"),
            second,
        );
        let exit = run_sq8_command_reader(
            BufReader::with_capacity(3, Cursor::new(input.into_bytes())),
            &control,
            &events,
            &inference_tx,
        )
        .unwrap();
        let Sq8InferenceCommand::Generate { admission, .. } = inference_rx.recv().unwrap() else {
            panic!("expected admitted request")
        };
        assert_eq!(
            exit,
            Sq8CommandReaderExit::ActiveShutdown {
                generation: admission.generation
            }
        );
        let snapshot = control.snapshot().unwrap();
        assert_eq!(snapshot.active_generation, Some(admission.generation));
        assert_eq!(
            snapshot.first_cancel_reason,
            Some(Sq8CancelReason::ClientDisconnect)
        );
        let lines = finish_writer(events, writer);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1]["code"], "busy");
        assert_eq!(lines[1]["request_id"], "req-2");
        assert_eq!(lines[2]["code"], "unknown_request");
        assert_eq!(lines[2]["request_id"], "wrong");
    }

    #[test]
    fn malformed_and_oversized_records_recover_before_shutdown() {
        let (control, events, writer) = start_ready_writer();
        let (inference_tx, inference_rx) = sync_channel(1);
        let mut input = b"{malformed}\n".to_vec();
        input.extend(std::iter::repeat_n(b'x', SQ8_WORKER_MAX_RECORD_BYTES + 1));
        input.extend_from_slice(
            b"\n{\"schema_version\":\"ullm.worker.v1\",\"type\":\"shutdown\"}\n",
        );
        let exit = run_sq8_command_reader(
            BufReader::with_capacity(8192, Cursor::new(input)),
            &control,
            &events,
            &inference_tx,
        )
        .unwrap();
        assert_eq!(exit, Sq8CommandReaderExit::IdleShutdown);
        assert!(matches!(
            inference_rx.recv().unwrap(),
            Sq8InferenceCommand::Shutdown
        ));
        let lines = finish_writer(events, writer);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1]["code"], "invalid_command");
        assert_eq!(lines[2]["code"], "invalid_command");
    }

    #[test]
    fn semantic_invalid_request_is_flushed_without_claiming_inference() {
        let (control, events, writer) = start_ready_writer();
        let (inference_tx, inference_rx) = sync_channel(1);
        let invalid =
            valid_generate("req-invalid").replace("\"max_new_tokens\":2", "\"max_new_tokens\":0");
        let input =
            format!("{invalid}\n{{\"schema_version\":\"ullm.worker.v1\",\"type\":\"shutdown\"}}\n");
        let exit = run_sq8_command_reader(
            BufReader::new(Cursor::new(input.into_bytes())),
            &control,
            &events,
            &inference_tx,
        )
        .unwrap();
        assert_eq!(exit, Sq8CommandReaderExit::IdleShutdown);
        assert!(matches!(
            inference_rx.recv().unwrap(),
            Sq8InferenceCommand::Shutdown
        ));
        assert!(inference_rx.try_recv().is_err());
        let lines = finish_writer(events, writer);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["code"], "invalid_request");
        assert_eq!(lines[1]["request_id"], "req-invalid");
    }

    #[test]
    fn inference_send_failure_removes_the_matching_admission_and_is_fatal() {
        let (control, events, writer) = start_ready_writer();
        let (inference_tx, inference_rx) = sync_channel(1);
        drop(inference_rx);
        let input = format!("{}\n", valid_generate("req-1"));
        assert!(
            run_sq8_command_reader(
                BufReader::new(Cursor::new(input.into_bytes())),
                &control,
                &events,
                &inference_tx,
            )
            .is_err()
        );
        let snapshot = control.snapshot().unwrap();
        assert_eq!(snapshot.lifecycle, Sq8WorkerLifecycle::Failed);
        assert!(snapshot.active_generation.is_none());
        assert!(writer.close_and_join().is_err());
        drop(events);
    }

    struct FailAfterFlushWriter {
        bytes: Vec<u8>,
        successful_flushes_remaining: usize,
    }

    impl Write for FailAfterFlushWriter {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.bytes.extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            if self.successful_flushes_remaining == 0 {
                return Err(std::io::Error::other("injected flush failure"));
            }
            self.successful_flushes_remaining -= 1;
            Ok(())
        }
    }

    #[test]
    fn recoverable_error_flush_failure_poisoned_the_reader_control() {
        let control = Arc::new(Sq8WorkerControl::new());
        let (events, writer) = spawn_sq8_ordered_writer(FailAfterFlushWriter {
            bytes: Vec::new(),
            successful_flushes_remaining: 1,
        })
        .unwrap();
        let ready = events.publish_ready(Sq8WorkerEvent::ready()).unwrap();
        control.mark_ready_after_flush(ready).unwrap();
        let (inference_tx, inference_rx) = sync_channel(1);
        let input = b"{malformed}\n".to_vec();
        assert!(
            run_sq8_command_reader(
                BufReader::new(Cursor::new(input)),
                &control,
                &events,
                &inference_tx,
            )
            .is_err()
        );
        assert!(inference_rx.try_recv().is_err());
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Failed
        );
        drop(events);
        assert!(writer.close_and_join().is_err());
    }

    #[test]
    fn unterminated_record_is_fatal_and_never_reaches_inference() {
        let (control, events, writer) = start_ready_writer();
        let (inference_tx, inference_rx) = sync_channel(1);
        let error = run_sq8_command_reader(
            BufReader::with_capacity(1, Cursor::new(b"{".to_vec())),
            &control,
            &events,
            &inference_tx,
        )
        .unwrap_err();
        assert!(error.contains("framing"), "{error}");
        assert!(inference_rx.try_recv().is_err());
        assert_eq!(
            control.snapshot().unwrap().lifecycle,
            Sq8WorkerLifecycle::Failed
        );
        drop(events);
        assert!(writer.close_and_join().is_err());
    }
}
