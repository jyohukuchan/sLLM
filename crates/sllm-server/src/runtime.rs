use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
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

/// One redacted, bounded memory-accounting category exposed to operational
/// observers.  It contains no allocation identity, prompt, token, or
/// credential data.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct BackendMemoryCategorySnapshotV1 {
    pub current_bytes: u64,
    pub high_water_bytes: u64,
}

/// Runtime-tracked backend memory accounting with a fixed category set.
///
/// `request_kv` covers request-local KV/state allocations and
/// `workspace_arena` covers transient execution/workspace allocations.  A
/// backend that cannot expose this accounting returns the all-zero default.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct BackendObservabilitySnapshotV1 {
    pub model_resident: BackendMemoryCategorySnapshotV1,
    pub request_kv: BackendMemoryCategorySnapshotV1,
    pub workspace_arena: BackendMemoryCategorySnapshotV1,
    pub total: BackendMemoryCategorySnapshotV1,
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

    /// Returns redacted, bounded runtime accounting for operational metrics.
    /// Fixture and third-party backends remain safe by inheriting zeroes.
    fn observability_snapshot(&self) -> BackendObservabilitySnapshotV1 {
        BackendObservabilitySnapshotV1::default()
    }
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

    pub fn observability_snapshot(&self) -> BackendObservabilitySnapshotV1 {
        self.backend.observability_snapshot()
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

/// The externally visible lifecycle of a bounded scheduler slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SchedulerSlotStateV1 {
    Queued,
    Active,
    Cancelled,
}

/// A redacted slot description suitable for read-only operational endpoints.
///
/// Prompt text, token IDs, credentials, and backend-specific state are
/// intentionally not represented here.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SchedulerSlotSnapshotV1 {
    pub id: u64,
    pub model_alias: String,
    pub state: SchedulerSlotStateV1,
}

/// A bounded, redacted scheduler snapshot for operational observability.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SchedulerSnapshotV1 {
    pub accepting: bool,
    pub queue_depth: usize,
    pub queue_capacity: usize,
    pub active_requests: usize,
    pub slots: Vec<SchedulerSlotSnapshotV1>,
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
    queue_capacity: usize,
    shutdown: Arc<AtomicBool>,
    active: Arc<Mutex<Option<GenerationCancellationV1>>>,
    next_slot_id: Arc<std::sync::atomic::AtomicU64>,
    slots: Arc<Mutex<SlotRegistryV1>>,
}

struct JobV1 {
    slot_id: u64,
    model: Arc<ModelRegistryEntryV1>,
    request: ChatCompletionRequestV1,
    events: mpsc::Sender<SchedulerEventV1>,
    cancellation: GenerationCancellationV1,
}

struct SlotRecordV1 {
    model_alias: String,
    state: SchedulerSlotStateV1,
    cancellation: GenerationCancellationV1,
    in_queue: bool,
    executing: bool,
}

struct SlotRegistryV1 {
    records: BTreeMap<u64, SlotRecordV1>,
    capacity: usize,
}

#[derive(Debug)]
pub(crate) enum SchedulerEventV1 {
    Delta(String),
    Finished(BackendCompletionV1),
    Failed(ApiErrorV1),
}

pub(crate) struct GenerationReceiverV1 {
    pub(crate) slot_id: u64,
    receiver: mpsc::Receiver<SchedulerEventV1>,
    cancellation: GenerationCancellationV1,
}

impl GenerationReceiverV1 {
    #[allow(dead_code)]
    pub const fn slot_id(&self) -> u64 {
        self.slot_id
    }

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
        let slots = Arc::new(Mutex::new(SlotRegistryV1 {
            records: BTreeMap::new(),
            capacity: config.queue_capacity.saturating_add(1),
        }));
        tokio::spawn(worker_loop(
            receiver,
            config.request_timeout,
            Arc::clone(&shutdown),
            Arc::clone(&active),
            Arc::clone(&slots),
        ));
        Self {
            sender,
            event_capacity: config.event_capacity,
            queue_capacity: config.queue_capacity,
            shutdown,
            active,
            next_slot_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
            slots,
        }
    }

    pub(crate) fn submit(
        &self,
        model: Arc<ModelRegistryEntryV1>,
        request: ChatCompletionRequestV1,
    ) -> Result<GenerationReceiverV1, ApiErrorV1> {
        let slot_id = self.allocate_slot_id();
        let cancellation = GenerationCancellationV1::new();
        let (events, receiver) = mpsc::channel(self.event_capacity);
        let model_alias = model.alias().to_owned();
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ApiErrorV1::server_shutdown());
        }
        if slots.records.len() >= slots.capacity {
            return Err(ApiErrorV1::rate_limited());
        }
        slots.records.insert(
            slot_id,
            SlotRecordV1 {
                model_alias,
                state: SchedulerSlotStateV1::Queued,
                cancellation: cancellation.clone(),
                in_queue: true,
                executing: false,
            },
        );
        let job = JobV1 {
            slot_id,
            model,
            request,
            events,
            cancellation: cancellation.clone(),
        };
        if let Err(error) = self.sender.try_send(job) {
            slots.records.remove(&slot_id);
            return Err(match error {
                mpsc::error::TrySendError::Full(_) => ApiErrorV1::rate_limited(),
                mpsc::error::TrySendError::Closed(_) => ApiErrorV1::server_shutdown(),
            });
        }
        drop(slots);
        Ok(GenerationReceiverV1 {
            slot_id,
            receiver,
            cancellation,
        })
    }

    /// Returns a redacted bounded snapshot of queued and active work.
    pub fn snapshot(&self) -> SchedulerSnapshotV1 {
        let slots = self
            .slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let queue_depth = slots
            .records
            .values()
            .filter(|record| record.in_queue)
            .count();
        let active_requests = slots
            .records
            .values()
            .filter(|record| record.executing)
            .count();
        let slot_snapshots = slots
            .records
            .iter()
            .map(|(&id, record)| SchedulerSlotSnapshotV1 {
                id,
                model_alias: record.model_alias.clone(),
                state: record.state,
            })
            .collect();
        SchedulerSnapshotV1 {
            accepting: self.is_accepting(),
            queue_depth,
            queue_capacity: self.queue_capacity,
            active_requests,
            slots: slot_snapshots,
        }
    }

    /// Requests cancellation for a queued or active slot.
    ///
    /// A queued cancellation is converted to a terminal cancellation event by
    /// the worker when it dequeues the job.  Active cancellation is observed by
    /// the backend through its existing cancellation token.
    pub fn cancel_slot(&self, slot_id: u64) -> bool {
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(record) = slots.records.get_mut(&slot_id) else {
            return false;
        };
        if record.state == SchedulerSlotStateV1::Cancelled {
            return false;
        }
        record.cancellation.cancel();
        record.state = SchedulerSlotStateV1::Cancelled;
        true
    }

    pub fn is_accepting(&self) -> bool {
        !self.shutdown.load(Ordering::Acquire)
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        let slots = self
            .slots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for record in slots.records.values() {
            record.cancellation.cancel();
        }
        drop(slots);
        if let Some(cancellation) = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            cancellation.cancel();
        }
    }

    fn allocate_slot_id(&self) -> u64 {
        loop {
            let current = self.next_slot_id.fetch_add(1, Ordering::Relaxed);
            if current != 0 {
                return current;
            }
        }
    }
}

async fn worker_loop(
    mut receiver: mpsc::Receiver<JobV1>,
    timeout: Duration,
    shutdown: Arc<AtomicBool>,
    active: Arc<Mutex<Option<GenerationCancellationV1>>>,
    slots: Arc<Mutex<SlotRegistryV1>>,
) {
    while let Some(job) = receiver.recv().await {
        let slot_id = job.slot_id;
        if shutdown.load(Ordering::Acquire) {
            job.cancellation.cancel();
            mark_slot_cancelled(&slots, slot_id);
            let _ = job
                .events
                .send(SchedulerEventV1::Failed(ApiErrorV1::server_shutdown()))
                .await;
            remove_slot(&slots, slot_id);
            continue;
        }
        if job.cancellation.is_cancelled() {
            mark_slot_cancelled(&slots, slot_id);
            let _ = job
                .events
                .send(SchedulerEventV1::Failed(ApiErrorV1::request_cancelled()))
                .await;
            remove_slot(&slots, slot_id);
            continue;
        }
        if !mark_slot_active(&slots, slot_id) {
            let _ = job
                .events
                .send(SchedulerEventV1::Failed(ApiErrorV1::request_cancelled()))
                .await;
            remove_slot(&slots, slot_id);
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
                } else {
                    let terminal_error = if shutdown.load(Ordering::Acquire) {
                        ApiErrorV1::server_shutdown()
                    } else if slot_is_cancelled(&slots, slot_id) {
                        ApiErrorV1::request_cancelled()
                    } else {
                        ApiErrorV1::generation_failed("generation timed out")
                    };
                    let _ = events.send(SchedulerEventV1::Failed(terminal_error)).await;
                }
            }
            Ok((events, cancellation, Err(error))) => {
                if !cancellation.is_cancelled() || !events.is_closed() {
                    let terminal_error = if shutdown.load(Ordering::Acquire) {
                        ApiErrorV1::server_shutdown()
                    } else if slot_is_cancelled(&slots, slot_id) {
                        ApiErrorV1::request_cancelled()
                    } else {
                        ApiErrorV1::generation_failed(error.to_string())
                    };
                    let _ = events.send(SchedulerEventV1::Failed(terminal_error)).await;
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
        remove_slot(&slots, slot_id);
    }
}

fn mark_slot_cancelled(slots: &Arc<Mutex<SlotRegistryV1>>, slot_id: u64) {
    let mut slots = slots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(record) = slots.records.get_mut(&slot_id) {
        record.state = SchedulerSlotStateV1::Cancelled;
        record.in_queue = false;
        record.executing = false;
    }
}

fn mark_slot_active(slots: &Arc<Mutex<SlotRegistryV1>>, slot_id: u64) -> bool {
    let mut slots = slots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(record) = slots.records.get_mut(&slot_id) {
        if record.state == SchedulerSlotStateV1::Cancelled || record.cancellation.is_cancelled() {
            return false;
        }
        record.state = SchedulerSlotStateV1::Active;
        record.in_queue = false;
        record.executing = true;
        return true;
    }
    false
}

fn remove_slot(slots: &Arc<Mutex<SlotRegistryV1>>, slot_id: u64) {
    let mut slots = slots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    slots.records.remove(&slot_id);
}

fn slot_is_cancelled(slots: &Arc<Mutex<SlotRegistryV1>>, slot_id: u64) -> bool {
    slots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .records
        .get(&slot_id)
        .is_some_and(|record| record.state == SchedulerSlotStateV1::Cancelled)
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

    #[test]
    fn fixture_backend_observability_is_redacted_and_default_safe() {
        let backend: Arc<dyn ChatGenerationBackendV1> = Arc::new(NeverBackend);
        let entry = ModelRegistryEntryV1::new(
            "qwen",
            1,
            "sllm",
            format!("sha256:{}", "0".repeat(64)),
            backend,
        )
        .unwrap();
        let snapshot = entry.observability_snapshot();
        assert_eq!(snapshot, BackendObservabilitySnapshotV1::default());
        let encoded = serde_json::to_value(snapshot).unwrap();
        let fields = encoded.as_object().unwrap();
        assert_eq!(fields.len(), 4);
        for field in ["model_resident", "request_kv", "workspace_arena", "total"] {
            assert!(fields.contains_key(field));
        }
        assert!(!encoded.to_string().contains("prompt"));
        assert!(!encoded.to_string().contains("token"));
        assert!(!encoded.to_string().contains("credential"));
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

    #[tokio::test]
    async fn slots_have_monotonic_ids_snapshot_state_and_queued_cancel_terminal() {
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
            SchedulerV1::new(SchedulerConfigV1::new(1, 1, Duration::from_secs(2)).unwrap());
        let request = parse_chat_completion_request(
            br#"{"model":"qwen","messages":[{"role":"user","content":"x"}]}"#,
        )
        .unwrap();
        let mut active_receiver = scheduler
            .submit(registry.get("qwen").unwrap(), request.clone())
            .unwrap();
        wait_for_count(&started, 1).await;
        let mut queued_receiver = scheduler
            .submit(registry.get("qwen").unwrap(), request)
            .unwrap();
        let active_id = active_receiver.slot_id();
        let queued_id = queued_receiver.slot_id();
        assert_ne!(active_id, queued_id);
        assert!(active_id < queued_id);

        let snapshot = scheduler.snapshot();
        assert!(snapshot.accepting);
        assert_eq!(snapshot.queue_capacity, 1);
        assert_eq!(snapshot.queue_depth, 1);
        assert_eq!(snapshot.active_requests, 1);
        assert_eq!(snapshot.slots.len(), 2);
        assert_eq!(snapshot.slots[0].id, active_id);
        assert_eq!(snapshot.slots[0].state, SchedulerSlotStateV1::Active);
        assert_eq!(snapshot.slots[1].id, queued_id);
        assert_eq!(snapshot.slots[1].state, SchedulerSlotStateV1::Queued);
        assert!(!scheduler.cancel_slot(99_999));
        assert!(scheduler.cancel_slot(queued_id));
        assert!(!scheduler.cancel_slot(queued_id));
        let cancelled_snapshot = scheduler.snapshot();
        assert_eq!(
            cancelled_snapshot
                .slots
                .iter()
                .find(|slot| slot.id == queued_id)
                .map(|slot| slot.state),
            Some(SchedulerSlotStateV1::Cancelled)
        );

        assert!(scheduler.cancel_slot(active_id));
        let active_end = tokio::time::timeout(Duration::from_secs(1), active_receiver.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            active_end,
            SchedulerEventV1::Failed(ref error)
                if error.code() == crate::api::ErrorCodeV1::RequestCancelled
        ));
        let queued_event = tokio::time::timeout(Duration::from_secs(1), queued_receiver.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            queued_event,
            SchedulerEventV1::Failed(ref error)
                if error.code() == crate::api::ErrorCodeV1::RequestCancelled
        ));
        for _ in 0..100 {
            if scheduler.snapshot().slots.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(scheduler.snapshot().slots.is_empty());
        assert!(cancelled.load(Ordering::Acquire));
        scheduler.shutdown();
        assert!(!scheduler.is_accepting());
    }
}
