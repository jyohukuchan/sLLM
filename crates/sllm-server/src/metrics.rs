//! Bounded, dependency-free Prometheus metrics for the server.
//!
//! The metrics registry is deliberately built from the served model aliases at
//! startup.  No request value is ever used as a label.  In particular, a
//! request id, prompt, generated token, API key, path, or backend error string
//! must not reach this module.  This keeps both the number of series and the
//! amount of memory used by the exporter bounded independently of traffic.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::runtime::{
    BackendMemoryCategorySnapshotV1, BackendObservabilitySnapshotV1, SchedulerSlotStateV1,
    SchedulerSnapshotV1,
};

/// Maximum number of model aliases represented by one metrics registry.
pub const MAX_METRIC_MODELS: usize = 16;

const MAX_MODEL_ALIAS_BYTES: usize = 256;
const HISTOGRAM_BUCKET_COUNT: usize = 15;
const OUTCOME_COUNT: usize = 6;
const TOKEN_DIRECTION_COUNT: usize = 2;
const CANCELLATION_REASON_COUNT: usize = 4;
const HTTP_ENDPOINT_COUNT: usize = 10;
const STATUS_CLASS_COUNT: usize = 6;

// Shared latency buckets in seconds.  They are fixed at compile time so a
// caller cannot create a new time series by choosing a bucket boundary.
const LATENCY_BUCKETS_SECONDS: [f64; HISTOGRAM_BUCKET_COUNT] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];

const OUTCOMES: [RequestOutcomeV1; OUTCOME_COUNT] = [
    RequestOutcomeV1::Admitted,
    RequestOutcomeV1::Success,
    RequestOutcomeV1::Error,
    RequestOutcomeV1::Cancelled,
    RequestOutcomeV1::Timeout,
    RequestOutcomeV1::Rejected,
];

const TOKEN_DIRECTIONS: [TokenDirectionV1; TOKEN_DIRECTION_COUNT] =
    [TokenDirectionV1::Prompt, TokenDirectionV1::Completion];

const CANCELLATION_REASONS: [CancellationReasonV1; CANCELLATION_REASON_COUNT] = [
    CancellationReasonV1::ClientDisconnect,
    CancellationReasonV1::SchedulerTimeout,
    CancellationReasonV1::Shutdown,
    CancellationReasonV1::Backend,
];

const HTTP_ENDPOINTS: [HttpEndpointV1; HTTP_ENDPOINT_COUNT] = [
    HttpEndpointV1::Models,
    HttpEndpointV1::ChatCompletions,
    HttpEndpointV1::Healthz,
    HttpEndpointV1::Readyz,
    HttpEndpointV1::Metrics,
    HttpEndpointV1::Props,
    HttpEndpointV1::Slots,
    HttpEndpointV1::SlotCancel,
    HttpEndpointV1::ChatReplay,
    HttpEndpointV1::KeysReload,
];

const STATUS_CLASSES: [StatusClassV1; STATUS_CLASS_COUNT] = [
    StatusClassV1::Informational,
    StatusClassV1::Success,
    StatusClassV1::Redirection,
    StatusClassV1::ClientError,
    StatusClassV1::ServerError,
    StatusClassV1::Other,
];

/// Configuration errors are returned before the server starts listening.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetricsConfigError {
    EmptyModelList,
    TooManyModels { count: usize, maximum: usize },
    EmptyModelAlias,
    ModelAliasTooLong { bytes: usize, maximum: usize },
    NulInModelAlias,
    DuplicateModelAlias(String),
}

impl fmt::Display for MetricsConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModelList => {
                formatter.write_str("metrics requires at least one model alias")
            }
            Self::TooManyModels { count, maximum } => {
                write!(
                    formatter,
                    "metrics received {count} model aliases; maximum is {maximum}"
                )
            }
            Self::EmptyModelAlias => formatter.write_str("metrics model alias must be nonempty"),
            Self::ModelAliasTooLong { bytes, maximum } => {
                write!(
                    formatter,
                    "metrics model alias is {bytes} bytes; maximum is {maximum}"
                )
            }
            Self::NulInModelAlias => formatter.write_str("metrics model alias contains NUL"),
            Self::DuplicateModelAlias(alias) => {
                write!(formatter, "metrics model aliases must be unique: {alias}")
            }
        }
    }
}

impl std::error::Error for MetricsConfigError {}

/// Terminal state recorded for one admitted request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestOutcomeV1 {
    Admitted,
    Success,
    Error,
    Cancelled,
    Timeout,
    Rejected,
}

impl RequestOutcomeV1 {
    const fn index(self) -> usize {
        match self {
            Self::Admitted => 0,
            Self::Success => 1,
            Self::Error => 2,
            Self::Cancelled => 3,
            Self::Timeout => 4,
            Self::Rejected => 5,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::Success => "success",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
            Self::Timeout => "timeout",
            Self::Rejected => "rejected",
        }
    }
}

/// Direction of token accounting.  The values are counts, never token ids.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenDirectionV1 {
    Prompt,
    Completion,
}

impl TokenDirectionV1 {
    const fn index(self) -> usize {
        match self {
            Self::Prompt => 0,
            Self::Completion => 1,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Completion => "completion",
        }
    }
}

/// Cancellation reasons intentionally have a closed vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancellationReasonV1 {
    ClientDisconnect,
    SchedulerTimeout,
    Shutdown,
    Backend,
}

impl CancellationReasonV1 {
    const fn index(self) -> usize {
        match self {
            Self::ClientDisconnect => 0,
            Self::SchedulerTimeout => 1,
            Self::Shutdown => 2,
            Self::Backend => 3,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::ClientDisconnect => "client_disconnect",
            Self::SchedulerTimeout => "scheduler_timeout",
            Self::Shutdown => "shutdown",
            Self::Backend => "backend",
        }
    }
}

/// Fixed endpoint vocabulary used by the HTTP counter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpEndpointV1 {
    Models,
    ChatCompletions,
    Healthz,
    Readyz,
    Metrics,
    Props,
    Slots,
    SlotCancel,
    ChatReplay,
    KeysReload,
}

impl HttpEndpointV1 {
    const fn index(self) -> usize {
        match self {
            Self::Models => 0,
            Self::ChatCompletions => 1,
            Self::Healthz => 2,
            Self::Readyz => 3,
            Self::Metrics => 4,
            Self::Props => 5,
            Self::Slots => 6,
            Self::SlotCancel => 7,
            Self::ChatReplay => 8,
            Self::KeysReload => 9,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Models => "models",
            Self::ChatCompletions => "chat_completions",
            Self::Healthz => "healthz",
            Self::Readyz => "readyz",
            Self::Metrics => "metrics",
            Self::Props => "props",
            Self::Slots => "slots",
            Self::SlotCancel => "slot_cancel",
            Self::ChatReplay => "chat_replay",
            Self::KeysReload => "keys_reload",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StatusClassV1 {
    Informational,
    Success,
    Redirection,
    ClientError,
    ServerError,
    Other,
}

impl StatusClassV1 {
    const fn index(self) -> usize {
        match self {
            Self::Informational => 0,
            Self::Success => 1,
            Self::Redirection => 2,
            Self::ClientError => 3,
            Self::ServerError => 4,
            Self::Other => 5,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Informational => "1xx",
            Self::Success => "2xx",
            Self::Redirection => "3xx",
            Self::ClientError => "4xx",
            Self::ServerError => "5xx",
            Self::Other => "other",
        }
    }
}

fn status_class(status: u16) -> StatusClassV1 {
    match status {
        100..=199 => StatusClassV1::Informational,
        200..=299 => StatusClassV1::Success,
        300..=399 => StatusClassV1::Redirection,
        400..=499 => StatusClassV1::ClientError,
        500..=599 => StatusClassV1::ServerError,
        _ => StatusClassV1::Other,
    }
}

struct Histogram {
    buckets: [AtomicU64; HISTOGRAM_BUCKET_COUNT],
    count: AtomicU64,
    sum_nanoseconds: AtomicU64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_nanoseconds: AtomicU64::new(0),
        }
    }

    fn observe(&self, duration: Duration) {
        let seconds = duration.as_secs_f64();
        for (index, boundary) in LATENCY_BUCKETS_SECONDS.iter().enumerate() {
            if seconds <= *boundary {
                self.buckets[index].fetch_add(1, Ordering::Relaxed);
            }
        }
        let nanoseconds = duration.as_nanos().min(u128::from(u64::MAX)) as u64;
        self.sum_nanoseconds
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(nanoseconds))
            })
            .ok();
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

struct StreamSeries {
    outcomes: [AtomicU64; OUTCOME_COUNT],
    tokens: [AtomicU64; TOKEN_DIRECTION_COUNT],
    cancellations: [AtomicU64; CANCELLATION_REASON_COUNT],
    ttft: Histogram,
    e2e: Histogram,
}

impl StreamSeries {
    fn new() -> Self {
        Self {
            outcomes: std::array::from_fn(|_| AtomicU64::new(0)),
            tokens: std::array::from_fn(|_| AtomicU64::new(0)),
            cancellations: std::array::from_fn(|_| AtomicU64::new(0)),
            ttft: Histogram::new(),
            e2e: Histogram::new(),
        }
    }
}

struct ModelSeries {
    streams: [StreamSeries; 2],
    ready: AtomicU64,
}

impl ModelSeries {
    fn new() -> Self {
        Self {
            streams: std::array::from_fn(|_| StreamSeries::new()),
            ready: AtomicU64::new(0),
        }
    }
}

struct MetricsInner {
    aliases: Vec<String>,
    models: Vec<ModelSeries>,
    http: [AtomicU64; HTTP_ENDPOINT_COUNT * STATUS_CLASS_COUNT],
}

/// Shared metrics registry.  Cloning this value only clones an `Arc`; all
/// counters remain in one bounded registry.
#[derive(Clone)]
pub struct ServerMetricsV1 {
    inner: Arc<MetricsInner>,
}

impl fmt::Debug for ServerMetricsV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServerMetricsV1")
            .field("model_aliases", &self.inner.aliases)
            .finish_non_exhaustive()
    }
}

impl ServerMetricsV1 {
    /// Create a registry for the complete, fixed served-alias set.
    pub fn new<I, S>(model_aliases: I) -> Result<Self, MetricsConfigError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let aliases = model_aliases
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>();
        if aliases.is_empty() {
            return Err(MetricsConfigError::EmptyModelList);
        }
        if aliases.len() > MAX_METRIC_MODELS {
            return Err(MetricsConfigError::TooManyModels {
                count: aliases.len(),
                maximum: MAX_METRIC_MODELS,
            });
        }
        for alias in &aliases {
            if alias.is_empty() {
                return Err(MetricsConfigError::EmptyModelAlias);
            }
            if alias.len() > MAX_MODEL_ALIAS_BYTES {
                return Err(MetricsConfigError::ModelAliasTooLong {
                    bytes: alias.len(),
                    maximum: MAX_MODEL_ALIAS_BYTES,
                });
            }
            if alias.contains('\0') {
                return Err(MetricsConfigError::NulInModelAlias);
            }
        }
        for (index, alias) in aliases.iter().enumerate() {
            if aliases[index + 1..].iter().any(|other| other == alias) {
                return Err(MetricsConfigError::DuplicateModelAlias(alias.clone()));
            }
        }
        let model_count = aliases.len();
        Ok(Self {
            inner: Arc::new(MetricsInner {
                aliases,
                models: (0..model_count).map(|_| ModelSeries::new()).collect(),
                http: std::array::from_fn(|_| AtomicU64::new(0)),
            }),
        })
    }

    /// Return the immutable model set represented by this registry.
    pub fn model_aliases(&self) -> &[String] {
        &self.inner.aliases
    }

    fn model_index(&self, alias: &str) -> Option<usize> {
        self.inner.aliases.iter().position(|known| known == alias)
    }

    fn stream_index(stream: bool) -> usize {
        usize::from(stream)
    }

    fn series(&self, model_alias: &str, stream: bool) -> Option<&StreamSeries> {
        let model = self.model_index(model_alias)?;
        Some(&self.inner.models[model].streams[Self::stream_index(stream)])
    }

    /// Record admission and return a handle for terminal/timing accounting.
    /// Unknown aliases are ignored and return `None`; no unbounded series are
    /// created for them.
    pub fn admit(&self, model_alias: &str, stream: bool) -> Option<MetricsRequestHandleV1> {
        let model = self.model_index(model_alias)?;
        let series = &self.inner.models[model].streams[Self::stream_index(stream)];
        series.outcomes[RequestOutcomeV1::Admitted.index()].fetch_add(1, Ordering::Relaxed);
        Some(MetricsRequestHandleV1 {
            metrics: self.clone(),
            model,
            stream,
            started: Instant::now(),
            ttft_recorded: false,
            terminal: false,
        })
    }

    /// Record a queue/admission rejection for a known alias.
    pub fn record_rejected(&self, model_alias: &str, stream: bool) {
        if let Some(series) = self.series(model_alias, stream) {
            series.outcomes[RequestOutcomeV1::Rejected.index()].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record an HTTP response using only the fixed endpoint/status classes.
    pub fn record_http(&self, endpoint: HttpEndpointV1, status: u16) {
        let index = endpoint.index() * STATUS_CLASS_COUNT + status_class(status).index();
        self.inner.http[index].fetch_add(1, Ordering::Relaxed);
    }

    /// Set the model-resident/readiness gauge.  Values are always 0 or 1.
    pub fn set_model_ready(&self, model_alias: &str, ready: bool) {
        if let Some(model) = self.model_index(model_alias) {
            self.inner.models[model]
                .ready
                .store(u64::from(ready), Ordering::Release);
        }
    }

    /// Record cancellation independent of request terminal accounting.  The
    /// caller should also finish the request handle with `Cancelled` or
    /// `Timeout`.
    pub fn record_cancellation(
        &self,
        model_alias: &str,
        stream: bool,
        reason: CancellationReasonV1,
    ) {
        if let Some(series) = self.series(model_alias, stream) {
            series.cancellations[reason.index()].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Convenience API for adapters that do not need a request handle.
    #[allow(clippy::too_many_arguments)]
    pub fn record_request(
        &self,
        model_alias: &str,
        stream: bool,
        outcome: RequestOutcomeV1,
        ttft: Option<Duration>,
        e2e: Option<Duration>,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) -> bool {
        let Some(series) = self.series(model_alias, stream) else {
            return false;
        };
        series.outcomes[outcome.index()].fetch_add(1, Ordering::Relaxed);
        if let Some(value) = ttft {
            series.ttft.observe(value);
        }
        if let Some(value) = e2e {
            series.e2e.observe(value);
        }
        series.tokens[TokenDirectionV1::Prompt.index()].fetch_add(prompt_tokens, Ordering::Relaxed);
        series.tokens[TokenDirectionV1::Completion.index()]
            .fetch_add(completion_tokens, Ordering::Relaxed);
        true
    }

    /// Render a complete, deterministic Prometheus text exposition.
    pub fn render(&self, scheduler: &SchedulerSnapshotV1) -> String {
        self.render_with_memory(scheduler, &[])
    }

    /// Render metrics with fixed-category backend memory observations.
    ///
    /// Snapshots whose aliases are not part of this registry are ignored.  The
    /// output always contains the complete known-alias × fixed-category ×
    /// current/high-water series, even when no backend supplies an observation.
    /// This keeps scrape cardinality independent of request or backend input.
    pub fn render_with_memory(
        &self,
        scheduler: &SchedulerSnapshotV1,
        snapshots: &[(&str, BackendObservabilitySnapshotV1)],
    ) -> String {
        let mut output = String::new();
        write_help_type(
            &mut output,
            "sllm_requests_total",
            "HTTP generation request lifecycle counts.",
            "counter",
        );
        for (model, model_series) in self.inner.aliases.iter().zip(&self.inner.models) {
            for stream in [false, true] {
                let series = &model_series.streams[Self::stream_index(stream)];
                for outcome in OUTCOMES {
                    sample(
                        &mut output,
                        "sllm_requests_total",
                        &[
                            ("model", model),
                            ("stream", bool_label(stream)),
                            ("outcome", outcome.as_str()),
                        ],
                        series.outcomes[outcome.index()].load(Ordering::Relaxed),
                    );
                }
            }
        }

        write_help_type(
            &mut output,
            "sllm_tokens_total",
            "Input and generated token counts.",
            "counter",
        );
        for (model, model_series) in self.inner.aliases.iter().zip(&self.inner.models) {
            for stream in [false, true] {
                let series = &model_series.streams[Self::stream_index(stream)];
                for direction in TOKEN_DIRECTIONS {
                    sample(
                        &mut output,
                        "sllm_tokens_total",
                        &[
                            ("model", model),
                            ("stream", bool_label(stream)),
                            ("direction", direction.as_str()),
                        ],
                        series.tokens[direction.index()].load(Ordering::Relaxed),
                    );
                }
            }
        }

        render_histograms(
            &mut output,
            "sllm_request_ttft_seconds",
            "Time to first generated delta in seconds.",
            &self.inner.aliases,
            &self.inner.models,
            |series| &series.ttft,
        );
        render_histograms(
            &mut output,
            "sllm_request_e2e_seconds",
            "End-to-end generation time in seconds.",
            &self.inner.aliases,
            &self.inner.models,
            |series| &series.e2e,
        );

        write_help_type(
            &mut output,
            "sllm_cancellations_total",
            "Generation cancellation counts by bounded reason.",
            "counter",
        );
        for (model, model_series) in self.inner.aliases.iter().zip(&self.inner.models) {
            for stream in [false, true] {
                let series = &model_series.streams[Self::stream_index(stream)];
                for reason in CANCELLATION_REASONS {
                    sample(
                        &mut output,
                        "sllm_cancellations_total",
                        &[
                            ("model", model),
                            ("stream", bool_label(stream)),
                            ("reason", reason.as_str()),
                        ],
                        series.cancellations[reason.index()].load(Ordering::Relaxed),
                    );
                }
            }
        }

        write_help_type(
            &mut output,
            "sllm_http_responses_total",
            "HTTP response counts by fixed endpoint and status class.",
            "counter",
        );
        for endpoint in HTTP_ENDPOINTS {
            for class in STATUS_CLASSES {
                let index = endpoint.index() * STATUS_CLASS_COUNT + class.index();
                sample(
                    &mut output,
                    "sllm_http_responses_total",
                    &[
                        ("endpoint", endpoint.as_str()),
                        ("status_class", class.as_str()),
                    ],
                    self.inner.http[index].load(Ordering::Relaxed),
                );
            }
        }

        write_help_type(
            &mut output,
            "sllm_model_ready",
            "Whether the model is resident and ready to accept generation.",
            "gauge",
        );
        for (model, model_series) in self.inner.aliases.iter().zip(&self.inner.models) {
            sample(
                &mut output,
                "sllm_model_ready",
                &[("model", model)],
                model_series.ready.load(Ordering::Acquire),
            );
        }

        write_help_type(
            &mut output,
            "sllm_scheduler_accepting",
            "Whether the scheduler accepts new requests.",
            "gauge",
        );
        sample(
            &mut output,
            "sllm_scheduler_accepting",
            &[],
            u64::from(scheduler.accepting),
        );
        write_help_type(
            &mut output,
            "sllm_scheduler_queue_depth",
            "Current bounded scheduler queue depth.",
            "gauge",
        );
        sample(
            &mut output,
            "sllm_scheduler_queue_depth",
            &[],
            scheduler.queue_depth as u64,
        );
        write_help_type(
            &mut output,
            "sllm_scheduler_queue_capacity",
            "Configured bounded scheduler queue capacity.",
            "gauge",
        );
        sample(
            &mut output,
            "sllm_scheduler_queue_capacity",
            &[],
            scheduler.queue_capacity as u64,
        );
        write_help_type(
            &mut output,
            "sllm_scheduler_active_requests",
            "Current number of active generation requests.",
            "gauge",
        );
        sample(
            &mut output,
            "sllm_scheduler_active_requests",
            &[],
            scheduler.active_requests as u64,
        );

        write_help_type(
            &mut output,
            "sllm_scheduler_slots",
            "Current scheduler slots by known model and bounded state.",
            "gauge",
        );
        for model in &self.inner.aliases {
            for state in [
                SchedulerSlotStateV1::Queued,
                SchedulerSlotStateV1::Active,
                SchedulerSlotStateV1::Cancelled,
            ] {
                let count = scheduler
                    .slots
                    .iter()
                    .filter(|slot| {
                        slot.model_alias == *model
                            && slot_state_name(state) == slot_state_name(slot.state)
                    })
                    .count() as u64;
                sample(
                    &mut output,
                    "sllm_scheduler_slots",
                    &[("model", model), ("state", slot_state_name(state))],
                    count,
                );
            }
        }
        render_backend_memory(&mut output, &self.inner.aliases, snapshots);
        output
    }
}

#[derive(Clone, Copy)]
enum MemoryCategoryV1 {
    ModelResident,
    RequestKv,
    WorkspaceArena,
    Total,
}

const MEMORY_CATEGORIES: [MemoryCategoryV1; 4] = [
    MemoryCategoryV1::ModelResident,
    MemoryCategoryV1::RequestKv,
    MemoryCategoryV1::WorkspaceArena,
    MemoryCategoryV1::Total,
];

impl MemoryCategoryV1 {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ModelResident => "model_resident",
            Self::RequestKv => "request_kv",
            Self::WorkspaceArena => "workspace_arena",
            Self::Total => "total",
        }
    }

    const fn snapshot(
        self,
        value: &BackendObservabilitySnapshotV1,
    ) -> BackendMemoryCategorySnapshotV1 {
        match self {
            Self::ModelResident => value.model_resident,
            Self::RequestKv => value.request_kv,
            Self::WorkspaceArena => value.workspace_arena,
            Self::Total => value.total,
        }
    }
}

fn render_backend_memory(
    output: &mut String,
    aliases: &[String],
    snapshots: &[(&str, BackendObservabilitySnapshotV1)],
) {
    write_help_type(
        output,
        "sllm_backend_memory_bytes",
        "Backend device memory by fixed category and state.",
        "gauge",
    );
    for model in aliases {
        let snapshot = snapshots
            .iter()
            .find(|(alias, _)| *alias == model.as_str())
            .map_or_else(BackendObservabilitySnapshotV1::default, |(_, value)| *value);
        for category in MEMORY_CATEGORIES {
            let values = category.snapshot(&snapshot);
            sample(
                output,
                "sllm_backend_memory_bytes",
                &[
                    ("model", model),
                    ("category", category.as_str()),
                    ("state", "current"),
                ],
                values.current_bytes,
            );
            sample(
                output,
                "sllm_backend_memory_bytes",
                &[
                    ("model", model),
                    ("category", category.as_str()),
                    ("state", "high_water"),
                ],
                values.high_water_bytes,
            );
        }
    }
}

/// Request-local accounting handle.  It carries no request content or id.
pub struct MetricsRequestHandleV1 {
    metrics: ServerMetricsV1,
    model: usize,
    stream: bool,
    started: Instant,
    ttft_recorded: bool,
    terminal: bool,
}

impl MetricsRequestHandleV1 {
    fn series(&self) -> &StreamSeries {
        &self.metrics.inner.models[self.model].streams[ServerMetricsV1::stream_index(self.stream)]
    }

    /// Record TTFT once.  Repeated calls are ignored to keep the first-token
    /// boundary stable.
    pub fn observe_ttft(&mut self, elapsed: Duration) {
        if !self.ttft_recorded {
            self.series().ttft.observe(elapsed);
            self.ttft_recorded = true;
        }
    }

    /// Record TTFT from the request start, returning whether this call won the
    /// one-shot observation.
    pub fn observe_ttft_since_start(&mut self) -> bool {
        if self.ttft_recorded {
            return false;
        }
        self.observe_ttft(self.started.elapsed());
        true
    }

    /// Add token counts; the values are counts and not token identities.
    pub fn record_tokens(&self, prompt_tokens: u64, completion_tokens: u64) {
        self.series().tokens[TokenDirectionV1::Prompt.index()]
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.series().tokens[TokenDirectionV1::Completion.index()]
            .fetch_add(completion_tokens, Ordering::Relaxed);
    }

    /// Finish the request and record terminal outcome plus E2E latency.
    pub fn finish(&mut self, outcome: RequestOutcomeV1) {
        self.finish_with_elapsed(outcome, self.started.elapsed());
    }

    /// Deterministic/testing variant of [`Self::finish`].
    pub fn finish_with_elapsed(&mut self, outcome: RequestOutcomeV1, elapsed: Duration) {
        if self.terminal {
            return;
        }
        if !self.ttft_recorded {
            self.observe_ttft(elapsed);
        }
        self.series().outcomes[outcome.index()].fetch_add(1, Ordering::Relaxed);
        self.series().e2e.observe(elapsed);
        self.terminal = true;
    }
}

impl Drop for MetricsRequestHandleV1 {
    fn drop(&mut self) {
        if !self.terminal {
            self.finish(RequestOutcomeV1::Cancelled);
            self.metrics.record_cancellation(
                &self.metrics.inner.aliases[self.model],
                self.stream,
                CancellationReasonV1::ClientDisconnect,
            );
        }
    }
}

fn bool_label(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn slot_state_name(state: SchedulerSlotStateV1) -> &'static str {
    match state {
        SchedulerSlotStateV1::Queued => "queued",
        SchedulerSlotStateV1::Active => "active",
        SchedulerSlotStateV1::Cancelled => "cancelled",
    }
}

fn write_help_type(output: &mut String, name: &str, help: &str, kind: &str) {
    output.push_str("# HELP ");
    output.push_str(name);
    output.push(' ');
    output.push_str(help);
    output.push('\n');
    output.push_str("# TYPE ");
    output.push_str(name);
    output.push(' ');
    output.push_str(kind);
    output.push('\n');
}

fn sample(output: &mut String, name: &str, labels: &[(&str, &str)], value: u64) {
    output.push_str(name);
    if !labels.is_empty() {
        output.push('{');
        for (index, (key, value)) in labels.iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            output.push_str(key);
            output.push_str("=\"");
            escape_label_value(output, value);
            output.push('"');
        }
        output.push('}');
    }
    output.push(' ');
    output.push_str(&value.to_string());
    output.push('\n');
}

fn sample_float(output: &mut String, name: &str, labels: &[(&str, &str)], value: f64) {
    output.push_str(name);
    if !labels.is_empty() {
        output.push('{');
        for (index, (key, value)) in labels.iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            output.push_str(key);
            output.push_str("=\"");
            escape_label_value(output, value);
            output.push('"');
        }
        output.push('}');
    }
    output.push(' ');
    if value.is_finite() {
        output.push_str(&value.to_string());
    } else {
        output.push('0');
    }
    output.push('\n');
}

fn escape_label_value(output: &mut String, value: &str) {
    for character in value.chars() {
        match character {
            '\\' => output.push_str("\\\\"),
            '"' => output.push_str("\\\""),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\n"),
            character => output.push(character),
        }
    }
}

fn render_histograms<F>(
    output: &mut String,
    name: &str,
    help: &str,
    aliases: &[String],
    models: &[ModelSeries],
    histogram: F,
) where
    F: Fn(&StreamSeries) -> &Histogram,
{
    write_help_type(output, name, help, "histogram");
    for (model, model_series) in aliases.iter().zip(models) {
        for stream in [false, true] {
            let series = &model_series.streams[ServerMetricsV1::stream_index(stream)];
            let histogram = histogram(series);
            for (index, boundary) in LATENCY_BUCKETS_SECONDS.iter().enumerate() {
                sample_float(
                    output,
                    &format!("{name}_bucket"),
                    &[
                        ("model", model),
                        ("stream", bool_label(stream)),
                        ("le", &boundary.to_string()),
                    ],
                    histogram.buckets[index].load(Ordering::Relaxed) as f64,
                );
            }
            sample_float(
                output,
                &format!("{name}_bucket"),
                &[
                    ("model", model),
                    ("stream", bool_label(stream)),
                    ("le", "+Inf"),
                ],
                histogram.count.load(Ordering::Relaxed) as f64,
            );
            sample_float(
                output,
                &format!("{name}_sum"),
                &[("model", model), ("stream", bool_label(stream))],
                histogram.sum_nanoseconds.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
            );
            sample_float(
                output,
                &format!("{name}_count"),
                &[("model", model), ("stream", bool_label(stream))],
                histogram.count.load(Ordering::Relaxed) as f64,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{
        BackendMemoryCategorySnapshotV1, BackendObservabilitySnapshotV1, SchedulerSlotSnapshotV1,
        SchedulerSlotStateV1, SchedulerSnapshotV1,
    };

    fn snapshot() -> SchedulerSnapshotV1 {
        SchedulerSnapshotV1 {
            accepting: true,
            queue_depth: 2,
            queue_capacity: 8,
            active_requests: 1,
            slots: vec![SchedulerSlotSnapshotV1 {
                id: 7,
                model_alias: "safe".to_owned(),
                state: SchedulerSlotStateV1::Active,
            }],
        }
    }

    #[test]
    fn bounded_model_set_rejects_empty_duplicate_and_seventeen_aliases() {
        assert!(matches!(
            ServerMetricsV1::new(Vec::<String>::new()),
            Err(MetricsConfigError::EmptyModelList)
        ));
        assert!(matches!(
            ServerMetricsV1::new(["same", "same"]),
            Err(MetricsConfigError::DuplicateModelAlias(_))
        ));
        let too_many = (0..=MAX_METRIC_MODELS)
            .map(|index| format!("model-{index}"))
            .collect::<Vec<_>>();
        assert!(matches!(
            ServerMetricsV1::new(too_many),
            Err(MetricsConfigError::TooManyModels { .. })
        ));
    }

    #[test]
    fn request_http_cancellation_and_scheduler_values_render() {
        let metrics = ServerMetricsV1::new(["safe"]).unwrap();
        let mut handle = metrics.admit("safe", false).unwrap();
        handle.observe_ttft(Duration::from_millis(10));
        handle.record_tokens(3, 2);
        handle.finish_with_elapsed(RequestOutcomeV1::Success, Duration::from_millis(25));
        metrics.record_rejected("safe", true);
        metrics.record_cancellation("safe", false, CancellationReasonV1::SchedulerTimeout);
        metrics.record_http(HttpEndpointV1::ChatCompletions, 200);
        metrics.record_http(HttpEndpointV1::ChatCompletions, 429);
        metrics.set_model_ready("safe", true);
        let rendered = metrics.render(&snapshot());
        assert!(rendered.contains(
            "sllm_requests_total{model=\"safe\",stream=\"false\",outcome=\"admitted\"} 1\n"
        ));
        assert!(rendered.contains("outcome=\"success\"} 1\n"));
        assert!(rendered.contains("outcome=\"rejected\"} 1\n"));
        assert!(rendered.contains(
            "sllm_tokens_total{model=\"safe\",stream=\"false\",direction=\"prompt\"} 3\n"
        ));
        assert!(rendered.contains("sllm_request_ttft_seconds_bucket"));
        assert!(rendered.contains("le=\"+Inf\"} 1"));
        assert!(rendered.contains(
            "sllm_http_responses_total{endpoint=\"chat_completions\",status_class=\"2xx\"} 1\n"
        ));
        assert!(rendered.contains("status_class=\"4xx\"} 1\n"));
        assert!(rendered.contains("sllm_cancellations_total{model=\"safe\",stream=\"false\",reason=\"scheduler_timeout\"} 1\n"));
        assert!(rendered.contains("sllm_scheduler_queue_depth 2\n"));
        assert!(rendered.contains("sllm_scheduler_slots{model=\"safe\",state=\"active\"} 1\n"));
    }

    #[test]
    fn backend_memory_render_is_fixed_and_ignores_unknown_aliases() {
        let metrics = ServerMetricsV1::new(["safe"]).unwrap();
        let memory = BackendObservabilitySnapshotV1 {
            model_resident: BackendMemoryCategorySnapshotV1 {
                current_bytes: 11,
                high_water_bytes: 22,
            },
            request_kv: BackendMemoryCategorySnapshotV1 {
                current_bytes: 33,
                high_water_bytes: 44,
            },
            workspace_arena: BackendMemoryCategorySnapshotV1 {
                current_bytes: 55,
                high_water_bytes: 66,
            },
            total: BackendMemoryCategorySnapshotV1 {
                current_bytes: 77,
                high_water_bytes: 88,
            },
        };
        let rendered = metrics.render_with_memory(
            &snapshot(),
            &[
                ("safe", memory),
                (
                    "unknown",
                    BackendObservabilitySnapshotV1 {
                        total: BackendMemoryCategorySnapshotV1 {
                            current_bytes: 999,
                            high_water_bytes: 999,
                        },
                        ..BackendObservabilitySnapshotV1::default()
                    },
                ),
            ],
        );
        assert!(rendered.contains(
            "sllm_backend_memory_bytes{model=\"safe\",category=\"model_resident\",state=\"current\"} 11\n"
        ));
        assert!(rendered.contains(
            "sllm_backend_memory_bytes{model=\"safe\",category=\"total\",state=\"high_water\"} 88\n"
        ));
        assert!(!rendered.contains("model=\"unknown\""));
        assert_eq!(
            rendered
                .matches("sllm_backend_memory_bytes{model=\"safe\"")
                .count(),
            8
        );
        assert!(!rendered.contains("credential"));
    }

    #[test]
    fn unknown_aliases_are_ignored_and_label_values_are_escaped() {
        let metrics = ServerMetricsV1::new(["model\"\\\nname"]).unwrap();
        assert!(metrics.admit("hidden prompt payload", false).is_none());
        assert!(!metrics.record_request(
            "credential:secret-key",
            false,
            RequestOutcomeV1::Success,
            Some(Duration::from_millis(1)),
            Some(Duration::from_millis(2)),
            9,
            1,
        ));
        assert!(!metrics.record_request(
            "token-123",
            false,
            RequestOutcomeV1::Success,
            None,
            None,
            0,
            0,
        ));
        let rendered = metrics.render(&snapshot());
        assert!(rendered.contains("model=\"model\\\"\\\\\\nname\""));
        assert!(!rendered.contains("hidden prompt payload"));
        assert!(!rendered.contains("hidden prompt payload"));
        assert!(!rendered.contains("Bearer secret-key"));
        assert!(!rendered.contains("token-123"));
    }

    #[test]
    fn dropped_handle_records_bounded_client_disconnect_once() {
        let metrics = ServerMetricsV1::new(["safe"]).unwrap();
        let handle = metrics.admit("safe", true).unwrap();
        drop(handle);
        let rendered = metrics.render(&snapshot());
        assert!(rendered.contains("outcome=\"cancelled\"} 1\n"));
        assert!(rendered.contains("reason=\"client_disconnect\"} 1\n"));
    }
}
