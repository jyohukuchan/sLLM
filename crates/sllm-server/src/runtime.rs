use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sllm_frontend::GenerationCancellationV1;
use tokio::sync::mpsc;

use crate::api::{ApiErrorV1, ChatCompletionRequestV1, FinishReasonV1, TokenUsageV1};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendErrorV1 {
    message: String,
}

impl BackendErrorV1 {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for BackendErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for BackendErrorV1 {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendCompletionV1 {
    pub finish_reason: FinishReasonV1,
    pub usage: TokenUsageV1,
}

pub trait GenerationDeltaSinkV1 {
    fn publish(&mut self, delta: &str) -> Result<(), BackendErrorV1>;
}

pub trait ChatGenerationBackendV1: Send + Sync + 'static {
    fn generate(
        &self,
        request: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1>;
}

#[derive(Clone)]
pub struct ModelRegistryEntryV1 {
    alias: String,
    created: u64,
    owned_by: String,
    lock_fingerprint: String,
    backend: Arc<dyn ChatGenerationBackendV1>,
}

impl ModelRegistryEntryV1 {
    pub fn new(
        alias: impl Into<String>,
        created: u64,
        owned_by: impl Into<String>,
        lock_fingerprint: impl Into<String>,
        backend: Arc<dyn ChatGenerationBackendV1>,
    ) -> Result<Self, ApiErrorV1> {
        let alias = alias.into();
        let owned_by = owned_by.into();
        let lock_fingerprint = lock_fingerprint.into();
        if alias.is_empty() || alias.len() > 256 {
            return Err(ApiErrorV1::invalid_value(
                "model",
                "served alias must be nonempty and at most 256 bytes",
            ));
        }
        if owned_by.is_empty() || !is_sha256_fingerprint(&lock_fingerprint) {
            return Err(ApiErrorV1::generation_failed(
                "model registry metadata is invalid",
            ));
        }
        Ok(Self {
            alias,
            created,
            owned_by,
            lock_fingerprint,
            backend,
        })
    }

    pub fn alias(&self) -> &str {
        &self.alias
    }

    pub const fn created(&self) -> u64 {
        self.created
    }

    pub fn owned_by(&self) -> &str {
        &self.owned_by
    }

    pub fn lock_fingerprint(&self) -> &str {
        &self.lock_fingerprint
    }
}

fn is_sha256_fingerprint(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Clone)]
pub struct ModelRegistryV1 {
    entries: Arc<Vec<Arc<ModelRegistryEntryV1>>>,
}

impl ModelRegistryV1 {
    pub fn new(entries: Vec<ModelRegistryEntryV1>) -> Result<Self, ApiErrorV1> {
        if entries.is_empty() {
            return Err(ApiErrorV1::generation_failed(
                "model registry must contain at least one served alias",
            ));
        }
        let mut entries = entries.into_iter().map(Arc::new).collect::<Vec<_>>();
        entries.sort_by(|left, right| left.alias.cmp(&right.alias));
        if entries
            .windows(2)
            .any(|pair| pair[0].alias == pair[1].alias)
        {
            return Err(ApiErrorV1::generation_failed(
                "model registry aliases must be unique",
            ));
        }
        Ok(Self {
            entries: Arc::new(entries),
        })
    }

    pub fn get(&self, alias: &str) -> Option<Arc<ModelRegistryEntryV1>> {
        self.entries
            .binary_search_by(|entry| entry.alias.as_str().cmp(alias))
            .ok()
            .map(|index| Arc::clone(&self.entries[index]))
    }

    pub fn entries(&self) -> &[Arc<ModelRegistryEntryV1>] {
        &self.entries
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerConfigV1 {
    pub queue_capacity: usize,
    pub event_capacity: usize,
    pub request_timeout: Duration,
}

impl SchedulerConfigV1 {
    pub fn new(
        queue_capacity: usize,
        event_capacity: usize,
        request_timeout: Duration,
    ) -> Result<Self, ApiErrorV1> {
        if queue_capacity == 0 || event_capacity == 0 || request_timeout.is_zero() {
            return Err(ApiErrorV1::generation_failed(
                "scheduler bounds and timeout must be nonzero",
            ));
        }
        Ok(Self {
            queue_capacity,
            event_capacity,
            request_timeout,
        })
    }
}

#[derive(Clone)]
pub struct SchedulerV1 {
    sender: mpsc::Sender<JobV1>,
    event_capacity: usize,
    shutdown: Arc<AtomicBool>,
    active: Arc<Mutex<Option<GenerationCancellationV1>>>,
}

struct JobV1 {
    model: Arc<ModelRegistryEntryV1>,
    request: ChatCompletionRequestV1,
    events: mpsc::Sender<SchedulerEventV1>,
    cancellation: GenerationCancellationV1,
}

#[derive(Debug)]
pub(crate) enum SchedulerEventV1 {
    Delta(String),
    Finished(BackendCompletionV1),
    Failed(ApiErrorV1),
}

pub(crate) struct GenerationReceiverV1 {
    receiver: mpsc::Receiver<SchedulerEventV1>,
    cancellation: GenerationCancellationV1,
}

impl GenerationReceiverV1 {
    pub async fn recv(&mut self) -> Option<SchedulerEventV1> {
        self.receiver.recv().await
    }
}

impl Drop for GenerationReceiverV1 {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

impl SchedulerV1 {
    pub fn new(config: SchedulerConfigV1) -> Self {
        let (sender, receiver) = mpsc::channel(config.queue_capacity);
        let shutdown = Arc::new(AtomicBool::new(false));
        let active = Arc::new(Mutex::new(None));
        tokio::spawn(worker_loop(
            receiver,
            config.request_timeout,
            Arc::clone(&shutdown),
            Arc::clone(&active),
        ));
        Self {
            sender,
            event_capacity: config.event_capacity,
            shutdown,
            active,
        }
    }

    pub(crate) fn submit(
        &self,
        model: Arc<ModelRegistryEntryV1>,
        request: ChatCompletionRequestV1,
    ) -> Result<GenerationReceiverV1, ApiErrorV1> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ApiErrorV1::server_shutdown());
        }
        let cancellation = GenerationCancellationV1::new();
        let (events, receiver) = mpsc::channel(self.event_capacity);
        let job = JobV1 {
            model,
            request,
            events,
            cancellation: cancellation.clone(),
        };
        self.sender.try_send(job).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => ApiErrorV1::rate_limited(),
            mpsc::error::TrySendError::Closed(_) => ApiErrorV1::server_shutdown(),
        })?;
        Ok(GenerationReceiverV1 {
            receiver,
            cancellation,
        })
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(cancellation) = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            cancellation.cancel();
        }
    }
}

async fn worker_loop(
    mut receiver: mpsc::Receiver<JobV1>,
    timeout: Duration,
    shutdown: Arc<AtomicBool>,
    active: Arc<Mutex<Option<GenerationCancellationV1>>>,
) {
    while let Some(job) = receiver.recv().await {
        if shutdown.load(Ordering::Acquire) {
            job.cancellation.cancel();
            let _ = job
                .events
                .send(SchedulerEventV1::Failed(ApiErrorV1::server_shutdown()))
                .await;
            continue;
        }
        {
            let mut guard = active
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *guard = Some(job.cancellation.clone());
        }
        let cancellation = job.cancellation.clone();
        let events = job.events.clone();
        let backend = Arc::clone(&job.model.backend);
        let request = job.request;
        let mut task = tokio::task::spawn_blocking(move || {
            let mut sink = ChannelDeltaSinkV1 {
                events: events.clone(),
                cancellation: cancellation.clone(),
            };
            let result = backend.generate(&request, &cancellation, &mut sink);
            (events, cancellation, result)
        });
        let outcome = match tokio::time::timeout(timeout, &mut task).await {
            Ok(joined) => joined,
            Err(_) => {
                job.cancellation.cancel();
                task.await
            }
        };
        {
            let mut guard = active
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *guard = None;
        }
        match outcome {
            Ok((events, cancellation, Ok(completion))) => {
                if !cancellation.is_cancelled() {
                    let _ = events.send(SchedulerEventV1::Finished(completion)).await;
                }
            }
            Ok((events, cancellation, Err(error))) => {
                if !cancellation.is_cancelled() || !events.is_closed() {
                    let _ = events
                        .send(SchedulerEventV1::Failed(ApiErrorV1::generation_failed(
                            error.to_string(),
                        )))
                        .await;
                }
            }
            Err(error) => {
                let _ = job
                    .events
                    .send(SchedulerEventV1::Failed(ApiErrorV1::generation_failed(
                        format!("generation worker failed: {error}"),
                    )))
                    .await;
            }
        }
    }
}

struct ChannelDeltaSinkV1 {
    events: mpsc::Sender<SchedulerEventV1>,
    cancellation: GenerationCancellationV1,
}

impl GenerationDeltaSinkV1 for ChannelDeltaSinkV1 {
    fn publish(&mut self, delta: &str) -> Result<(), BackendErrorV1> {
        if self.cancellation.is_cancelled() {
            return Err(BackendErrorV1::new("generation was cancelled"));
        }
        if delta.is_empty() {
            return Ok(());
        }
        self.events
            .blocking_send(SchedulerEventV1::Delta(delta.to_owned()))
            .map_err(|_| {
                self.cancellation.cancel();
                BackendErrorV1::new("generation consumer disconnected")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::parse_chat_completion_request;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    struct NeverBackend;

    impl ChatGenerationBackendV1 for NeverBackend {
        fn generate(
            &self,
            _: &ChatCompletionRequestV1,
            _: &GenerationCancellationV1,
            _: &mut dyn GenerationDeltaSinkV1,
        ) -> Result<BackendCompletionV1, BackendErrorV1> {
            Err(BackendErrorV1::new("not used"))
        }
    }

    #[test]
    fn registry_rejects_floating_duplicate_and_invalid_identity() {
        let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(NeverBackend);
        assert!(ModelRegistryEntryV1::new("", 0, "sllm", "main", Arc::clone(&backend)).is_err());
        let entry = ModelRegistryEntryV1::new(
            "qwen",
            1,
            "sllm",
            format!("sha256:{}", "0".repeat(64)),
            Arc::clone(&backend),
        )
        .unwrap();
        let duplicate = ModelRegistryEntryV1::new(
            "qwen",
            2,
            "sllm",
            format!("sha256:{}", "1".repeat(64)),
            backend,
        )
        .unwrap();
        assert!(ModelRegistryV1::new(vec![entry, duplicate]).is_err());
    }

    struct CountingBackend {
        publish_attempts: Arc<AtomicUsize>,
        cancelled: Arc<AtomicBool>,
    }

    impl ChatGenerationBackendV1 for CountingBackend {
        fn generate(
            &self,
            _: &ChatCompletionRequestV1,
            cancellation: &GenerationCancellationV1,
            sink: &mut dyn GenerationDeltaSinkV1,
        ) -> Result<BackendCompletionV1, BackendErrorV1> {
            for _ in 0..100 {
                self.publish_attempts.fetch_add(1, Ordering::Relaxed);
                if let Err(error) = sink.publish("x") {
                    self.cancelled.store(true, Ordering::Release);
                    return Err(error);
                }
                if cancellation.is_cancelled() {
                    self.cancelled.store(true, Ordering::Release);
                    return Err(BackendErrorV1::new("cancelled"));
                }
            }
            Ok(BackendCompletionV1 {
                finish_reason: FinishReasonV1::Length,
                usage: TokenUsageV1::new(1, 100).unwrap(),
            })
        }
    }

    #[tokio::test]
    async fn event_channel_is_bounded_and_receiver_drop_cancels_generation() {
        let publish_attempts = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(CountingBackend {
            publish_attempts: Arc::clone(&publish_attempts),
            cancelled: Arc::clone(&cancelled),
        });
        let entry = ModelRegistryEntryV1::new(
            "qwen",
            1,
            "sllm",
            format!("sha256:{}", "0".repeat(64)),
            backend,
        )
        .unwrap();
        let registry = ModelRegistryV1::new(vec![entry]).unwrap();
        let scheduler =
            SchedulerV1::new(SchedulerConfigV1::new(1, 1, Duration::from_secs(2)).unwrap());
        let request = parse_chat_completion_request(
            br#"{"model":"qwen","messages":[{"role":"user","content":"x"}]}"#,
        )
        .unwrap();
        let receiver = scheduler
            .submit(registry.get("qwen").unwrap(), request)
            .unwrap();
        for _ in 0..100 {
            if publish_attempts.load(Ordering::Acquire) >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert_eq!(publish_attempts.load(Ordering::Acquire), 2);
        drop(receiver);
        for _ in 0..100 {
            if cancelled.load(Ordering::Acquire) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert!(cancelled.load(Ordering::Acquire));
        scheduler.shutdown();
        thread::yield_now();
    }

    struct PollingBackend {
        started: Arc<AtomicUsize>,
        cancelled: Arc<AtomicBool>,
    }

    impl ChatGenerationBackendV1 for PollingBackend {
        fn generate(
            &self,
            _: &ChatCompletionRequestV1,
            cancellation: &GenerationCancellationV1,
            _: &mut dyn GenerationDeltaSinkV1,
        ) -> Result<BackendCompletionV1, BackendErrorV1> {
            self.started.fetch_add(1, Ordering::Release);
            while !cancellation.is_cancelled() {
                thread::sleep(Duration::from_millis(1));
            }
            self.cancelled.store(true, Ordering::Release);
            Err(BackendErrorV1::new("cancelled"))
        }
    }

    async fn wait_for_count(counter: &AtomicUsize, expected: usize) {
        for _ in 0..200 {
            if counter.load(Ordering::Acquire) >= expected {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        panic!("counter did not reach {expected}");
    }

    #[tokio::test]
    async fn request_timeout_cancels_the_active_backend() {
        let started = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(PollingBackend {
            started: Arc::clone(&started),
            cancelled: Arc::clone(&cancelled),
        });
        let entry = ModelRegistryEntryV1::new(
            "qwen",
            1,
            "sllm",
            format!("sha256:{}", "0".repeat(64)),
            backend,
        )
        .unwrap();
        let registry = ModelRegistryV1::new(vec![entry]).unwrap();
        let scheduler =
            SchedulerV1::new(SchedulerConfigV1::new(1, 1, Duration::from_millis(20)).unwrap());
        let request = parse_chat_completion_request(
            br#"{"model":"qwen","messages":[{"role":"user","content":"x"}]}"#,
        )
        .unwrap();
        let mut receiver = scheduler
            .submit(registry.get("qwen").unwrap(), request)
            .unwrap();

        let event = tokio::time::timeout(Duration::from_secs(1), receiver.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            event,
            SchedulerEventV1::Failed(ref error)
                if error.code() == crate::api::ErrorCodeV1::GenerationFailed
        ));
        assert_eq!(started.load(Ordering::Acquire), 1);
        assert!(cancelled.load(Ordering::Acquire));
        scheduler.shutdown();
    }

    #[tokio::test]
    async fn shutdown_cancels_active_rejects_queued_and_refuses_new_work() {
        let started = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(PollingBackend {
            started: Arc::clone(&started),
            cancelled: Arc::clone(&cancelled),
        });
        let entry = ModelRegistryEntryV1::new(
            "qwen",
            1,
            "sllm",
            format!("sha256:{}", "0".repeat(64)),
            backend,
        )
        .unwrap();
        let registry = ModelRegistryV1::new(vec![entry]).unwrap();
        let scheduler =
            SchedulerV1::new(SchedulerConfigV1::new(2, 1, Duration::from_secs(2)).unwrap());
        let request = parse_chat_completion_request(
            br#"{"model":"qwen","messages":[{"role":"user","content":"x"}]}"#,
        )
        .unwrap();
        let mut active_receiver = scheduler
            .submit(registry.get("qwen").unwrap(), request.clone())
            .unwrap();
        wait_for_count(&started, 1).await;
        let mut queued_receiver = scheduler
            .submit(registry.get("qwen").unwrap(), request.clone())
            .unwrap();

        scheduler.shutdown();
        let new_error = match scheduler.submit(registry.get("qwen").unwrap(), request) {
            Ok(_) => panic!("scheduler accepted work after shutdown"),
            Err(error) => error,
        };
        assert_eq!(new_error.code(), crate::api::ErrorCodeV1::ServerShutdown);
        let active_event = tokio::time::timeout(Duration::from_secs(1), active_receiver.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(active_event, SchedulerEventV1::Failed(_)));
        let queued_event = tokio::time::timeout(Duration::from_secs(1), queued_receiver.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            queued_event,
            SchedulerEventV1::Failed(ref error)
                if error.code() == crate::api::ErrorCodeV1::ServerShutdown
        ));
        assert!(cancelled.load(Ordering::Acquire));
        assert_eq!(started.load(Ordering::Acquire), 1);
    }
}
