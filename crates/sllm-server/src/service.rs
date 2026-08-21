use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{Path, State};
use axum::http::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, HeaderName};
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::stream::{self, Stream};
use serde::Serialize;
use tower_http::cors::CorsLayer;

use crate::api::{
    ApiErrorV1, ChatCompatibilityProfileV1, ErrorCodeV1, FinishReasonV1, MAX_REQUEST_BODY_BYTES,
    ReasoningOptionsV1, TokenUsageV1, parse_chat_completion_request_for_profile,
};
use crate::lifecycle::ServerLifecycleV1;
use crate::metrics::{HttpEndpointV1, MetricsRequestHandleV1, RequestOutcomeV1, ServerMetricsV1};
use crate::resume::{ReplayErrorV1, ResumableStoreV1};
use crate::runtime::{
    BackendTokenLogprobV1, GenerationReceiverV1, ModelRegistryV1, SchedulerEventV1, SchedulerV1,
};
use crate::security::CredentialStoreV1;

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

const LAST_EVENT_ID: HeaderName = HeaderName::from_static("last-event-id");
const MAX_CORS_ORIGINS: usize = 32;

#[derive(Clone)]
pub struct ServerConfigV1 {
    credentials: CredentialStoreV1,
    compatibility_profile: ChatCompatibilityProfileV1,
    lifecycle: ServerLifecycleV1,
    metrics: Option<ServerMetricsV1>,
    replay: Option<ResumableStoreV1>,
    cors_origins: Vec<HeaderValue>,
}

impl ServerConfigV1 {
    pub fn new(bearer_token: Option<String>) -> Result<Self, ApiErrorV1> {
        let credentials = bearer_token
            .map(CredentialStoreV1::from_user_key)
            .transpose()
            .map_err(|_| ApiErrorV1::generation_failed("bearer token configuration is invalid"))?
            .unwrap_or_else(CredentialStoreV1::open);
        Ok(Self {
            credentials,
            compatibility_profile: ChatCompatibilityProfileV1::Strict,
            lifecycle: ServerLifecycleV1::default(),
            metrics: None,
            replay: None,
            cors_origins: Vec::new(),
        })
    }

    pub fn openwebui_compatible(bearer_token: Option<String>) -> Result<Self, ApiErrorV1> {
        let mut config = Self::new(bearer_token)?;
        config.compatibility_profile = ChatCompatibilityProfileV1::OpenWebUi;
        Ok(config)
    }

    pub fn with_credentials(mut self, credentials: CredentialStoreV1) -> Self {
        self.credentials = credentials;
        self
    }

    pub fn with_lifecycle(mut self, lifecycle: ServerLifecycleV1) -> Self {
        self.lifecycle = lifecycle;
        self
    }

    pub fn with_metrics(mut self, metrics: ServerMetricsV1) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn with_resumable_store(mut self, replay: ResumableStoreV1) -> Self {
        self.replay = Some(replay);
        self
    }

    pub fn with_cors_origins<I, S>(mut self, origins: I) -> Result<Self, ApiErrorV1>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut parsed = Vec::new();
        for origin in origins {
            if parsed.len() == MAX_CORS_ORIGINS {
                return Err(ApiErrorV1::generation_failed(
                    "CORS origin count exceeds the bounded limit",
                ));
            }
            let origin = origin.as_ref();
            let uri = origin
                .parse::<axum::http::Uri>()
                .map_err(|_| ApiErrorV1::generation_failed("CORS origin is invalid"))?;
            let scheme = uri.scheme_str();
            let authority = uri.authority();
            let exact = scheme.zip(authority).is_some_and(|(scheme, authority)| {
                matches!(scheme, "http" | "https")
                    && !authority.as_str().contains('@')
                    && origin == format!("{scheme}://{authority}")
            });
            if !exact {
                return Err(ApiErrorV1::generation_failed(
                    "CORS origins must be exact HTTP(S) origins",
                ));
            }
            let value = HeaderValue::from_str(origin)
                .map_err(|_| ApiErrorV1::generation_failed("CORS origin is invalid"))?;
            if parsed.contains(&value) {
                return Err(ApiErrorV1::generation_failed("CORS origins must be unique"));
            }
            parsed.push(value);
        }
        self.cors_origins = parsed;
        Ok(self)
    }
}

impl Default for ServerConfigV1 {
    fn default() -> Self {
        Self::new(None).expect("open server configuration is valid")
    }
}

#[derive(Clone)]
struct AppStateV1 {
    registry: ModelRegistryV1,
    scheduler: SchedulerV1,
    config: ServerConfigV1,
}

pub fn build_router_v1(
    registry: ModelRegistryV1,
    scheduler: SchedulerV1,
    config: ServerConfigV1,
) -> Router {
    if let Some(metrics) = &config.metrics {
        let ready = config.lifecycle.is_ready() && scheduler.is_accepting();
        for entry in registry.entries() {
            metrics.set_model_ready(entry.alias(), ready);
        }
    }
    let cors_origins = config.cors_origins.clone();
    let state = Arc::new(AppStateV1 {
        registry,
        scheduler,
        config,
    });
    let router = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .route("/props", get(props))
        .route("/slots", get(slots))
        .route("/admin/slots/{id}/cancel", post(cancel_slot))
        .route("/admin/keys/reload", post(reload_keys))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(create_chat_completion))
        .route(
            "/v1/chat/completions/{id}/events",
            get(resume_chat_completion),
        )
        .with_state(state);
    if cors_origins.is_empty() {
        router
    } else {
        router.layer(
            CorsLayer::new()
                .allow_origin(cors_origins)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([AUTHORIZATION, CONTENT_TYPE, LAST_EVENT_ID]),
        )
    }
}

async fn healthz(State(state): State<Arc<AppStateV1>>) -> Response {
    let response = axum::Json(serde_json::json!({
        "status": "ok",
        "state": state.config.lifecycle.state(),
    }))
    .into_response();
    record_http(&state, HttpEndpointV1::Healthz, &response);
    response
}

async fn readyz(State(state): State<Arc<AppStateV1>>) -> Response {
    let ready = state.config.lifecycle.is_ready() && state.scheduler.is_accepting();
    if let Some(metrics) = &state.config.metrics {
        for entry in state.registry.entries() {
            metrics.set_model_ready(entry.alias(), ready);
        }
    }
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let response = (
        status,
        axum::Json(serde_json::json!({
            "status": if ready { "ready" } else { "not_ready" },
            "state": state.config.lifecycle.state(),
            "scheduler_accepting": state.scheduler.is_accepting(),
        })),
    )
        .into_response();
    record_http(&state, HttpEndpointV1::Readyz, &response);
    response
}

async fn metrics(State(state): State<Arc<AppStateV1>>, headers: HeaderMap) -> Response {
    let response = if let Err(error) = authorize_user(&headers, &state.config) {
        error.into_response()
    } else if let Some(metrics) = &state.config.metrics {
        let ready = state.config.lifecycle.is_ready() && state.scheduler.is_accepting();
        for entry in state.registry.entries() {
            metrics.set_model_ready(entry.alias(), ready);
        }
        let memory = state
            .registry
            .entries()
            .iter()
            .map(|entry| (entry.alias(), entry.observability_snapshot()))
            .collect::<Vec<_>>();
        (
            StatusCode::OK,
            [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
            metrics.render_with_memory(&state.scheduler.snapshot(), &memory),
        )
            .into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    };
    record_http(&state, HttpEndpointV1::Metrics, &response);
    response
}

async fn props(State(state): State<Arc<AppStateV1>>, headers: HeaderMap) -> Response {
    let response = if let Err(error) = authorize_user(&headers, &state.config) {
        error.into_response()
    } else {
        let models = state
            .registry
            .entries()
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "alias": entry.alias(),
                    "lock_fingerprint": entry.lock_fingerprint(),
                    "runtime_memory": entry.observability_snapshot(),
                })
            })
            .collect::<Vec<_>>();
        axum::Json(serde_json::json!({
            "schema_version": "sllm-server-props-v1",
            "state": state.config.lifecycle.state(),
            "models": models,
            "scheduler": state.scheduler.snapshot(),
            "features": {
                "metrics": state.config.metrics.is_some(),
                "resumable_sse": state.config.replay.is_some(),
                "cors": !state.config.cors_origins.is_empty(),
                "authentication": !state.config.credentials.is_open(),
                "admin": state.config.credentials.has_admin_credentials(),
            }
        }))
        .into_response()
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn slots(State(state): State<Arc<AppStateV1>>, headers: HeaderMap) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        axum::Json(state.scheduler.snapshot()).into_response()
    };
    record_http(&state, HttpEndpointV1::Slots, &response);
    response
}

async fn cancel_slot(
    State(state): State<Arc<AppStateV1>>,
    Path(id): Path<u64>,
    headers: HeaderMap,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else if state.scheduler.cancel_slot(id) {
        axum::Json(serde_json::json!({"id": id, "state": "cancelled"})).into_response()
    } else {
        ApiErrorV1::slot_not_found().into_response()
    };
    record_http(&state, HttpEndpointV1::SlotCancel, &response);
    response
}

async fn reload_keys(State(state): State<Arc<AppStateV1>>, headers: HeaderMap) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match state.config.credentials.reload() {
            Ok(()) => axum::Json(serde_json::json!({"state": "reloaded"})).into_response(),
            Err(_) => ApiErrorV1::generation_failed("credential reload failed").into_response(),
        }
    };
    record_http(&state, HttpEndpointV1::KeysReload, &response);
    response
}

async fn list_models(State(state): State<Arc<AppStateV1>>, headers: HeaderMap) -> Response {
    let response = if let Err(error) = authorize_user(&headers, &state.config) {
        error.into_response()
    } else {
        let data = state
            .registry
            .entries()
            .iter()
            .map(|entry| ModelObjectV1 {
                id: entry.alias(),
                object: "model",
                created: entry.created(),
                owned_by: entry.owned_by(),
            })
            .collect();
        axum::Json(ModelListV1 {
            object: "list",
            data,
        })
        .into_response()
    };
    record_http(&state, HttpEndpointV1::Models, &response);
    response
}

async fn create_chat_completion(
    State(state): State<Arc<AppStateV1>>,
    request: Request<Body>,
) -> Response {
    if let Err(error) = authorize_user(request.headers(), &state.config) {
        let response = error.into_response();
        record_http(&state, HttpEndpointV1::ChatCompletions, &response);
        return response;
    }
    if let Err(error) = validate_content_type(request.headers()) {
        let response = error.into_response();
        record_http(&state, HttpEndpointV1::ChatCompletions, &response);
        return response;
    }
    if request
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|length| length > MAX_REQUEST_BODY_BYTES as u64)
    {
        let response = request_too_large();
        record_http(&state, HttpEndpointV1::ChatCompletions, &response);
        return response;
    }
    let body = match to_bytes(request.into_body(), MAX_REQUEST_BODY_BYTES).await {
        Ok(body) => body,
        Err(_) => {
            let response = request_too_large();
            record_http(&state, HttpEndpointV1::ChatCompletions, &response);
            return response;
        }
    };
    let request = match parse_chat_completion_request_for_profile(
        &body,
        state.config.compatibility_profile,
    ) {
        Ok(request) => request,
        Err(error) => {
            let response = error.into_response();
            record_http(&state, HttpEndpointV1::ChatCompletions, &response);
            return response;
        }
    };
    if request.resumable() && state.config.replay.is_none() {
        let response = ApiErrorV1::invalid_value(
            "sllm.resumable",
            "resumable streaming is not enabled on this server",
        )
        .into_response();
        record_http(&state, HttpEndpointV1::ChatCompletions, &response);
        return response;
    }
    let model = match state.registry.get(request.model()) {
        Some(model) => model,
        None => {
            let response = ApiErrorV1::model_not_found(request.model()).into_response();
            record_http(&state, HttpEndpointV1::ChatCompletions, &response);
            return response;
        }
    };
    let context = ResponseContextV1::new(model.alias(), request.reasoning());
    let stream_response = request.stream();
    let resumable = request.resumable();
    let reserved_replay = if stream_response && resumable {
        let replay = state
            .config
            .replay
            .clone()
            .expect("resumable feature availability was validated before admission");
        if replay.create(&context.id).is_err() {
            if let Some(metrics) = &state.config.metrics {
                metrics.record_rejected(&context.model, stream_response);
            }
            let response = ApiErrorV1::rate_limited().into_response();
            record_http(&state, HttpEndpointV1::ChatCompletions, &response);
            return response;
        }
        Some(replay)
    } else {
        None
    };
    let choice_count = request.choice_count();
    let mut receivers = Vec::with_capacity(choice_count as usize);
    let mut admission_error = None;
    for index in 0..choice_count {
        let choice_request = match request.for_choice(index) {
            Ok(request) => request,
            Err(error) => {
                admission_error = Some(error);
                break;
            }
        };
        match state.scheduler.submit(Arc::clone(&model), choice_request) {
            Ok(receiver) => receivers.push(IndexedGenerationReceiverV1 { index, receiver }),
            Err(error) => {
                admission_error = Some(error);
                break;
            }
        }
    }
    if let Some(error) = admission_error {
        // Dropping already-admitted receivers cancels their independent slots;
        // an n-choice request is never partially exposed to the client.
        drop(receivers);
        if let Some(replay) = &reserved_replay {
            replay.discard(&context.id);
        }
        if let Some(metrics) = &state.config.metrics {
            metrics.record_rejected(&context.model, stream_response);
        }
        let response = error.into_response();
        record_http(&state, HttpEndpointV1::ChatCompletions, &response);
        return response;
    }
    let request_metrics = state
        .config
        .metrics
        .as_ref()
        .and_then(|metrics| metrics.admit(&context.model, stream_response));
    let response = if stream_response && resumable {
        let replay =
            reserved_replay.expect("resumable replay was reserved before scheduler admission");
        spawn_resumable_producer(receivers, context.clone(), replay.clone(), request_metrics);
        replay_stream(replay, context.id.clone(), 0).into_response()
    } else if stream_response {
        stream_chat_completion(receivers, context, request_metrics).into_response()
    } else {
        non_stream_chat_completion(receivers, context, request_metrics).await
    };
    record_http(&state, HttpEndpointV1::ChatCompletions, &response);
    response
}

fn request_too_large() -> Response {
    ApiErrorV1::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "request body exceeds 100663296 bytes",
        "invalid_request_error",
        None,
        ErrorCodeV1::RequestTooLarge,
    )
    .into_response()
}

fn authorize_user(headers: &HeaderMap, config: &ServerConfigV1) -> Result<(), ApiErrorV1> {
    authorize(headers, config, false)
}

fn authorize_admin(headers: &HeaderMap, config: &ServerConfigV1) -> Result<(), ApiErrorV1> {
    authorize(headers, config, true)
}

fn authorize(headers: &HeaderMap, config: &ServerConfigV1, admin: bool) -> Result<(), ApiErrorV1> {
    let mut values = headers.get_all(AUTHORIZATION).iter();
    let first = values.next().and_then(|value| value.to_str().ok());
    let unique = values.next().is_none();
    let accepted = unique
        && if admin {
            config.credentials.authorize_admin(first)
        } else {
            config.credentials.authorize_user(first)
        };
    if accepted {
        Ok(())
    } else {
        Err(ApiErrorV1::new(
            StatusCode::UNAUTHORIZED,
            "invalid bearer credential",
            "invalid_request_error",
            None,
            ErrorCodeV1::InvalidApiKey,
        ))
    }
}

fn record_http(state: &AppStateV1, endpoint: HttpEndpointV1, response: &Response) {
    if let Some(metrics) = &state.config.metrics {
        metrics.record_http(endpoint, response.status().as_u16());
    }
}

fn validate_content_type(headers: &HeaderMap) -> Result<(), ApiErrorV1> {
    let accepted = headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("application/json"));
    if accepted {
        Ok(())
    } else {
        Err(ApiErrorV1::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
            "invalid_request_error",
            None,
            ErrorCodeV1::UnsupportedMediaType,
        ))
    }
}

struct IndexedGenerationReceiverV1 {
    index: u32,
    receiver: GenerationReceiverV1,
}

async fn non_stream_chat_completion(
    receivers: Vec<IndexedGenerationReceiverV1>,
    context: ResponseContextV1,
    mut metrics: Option<MetricsRequestHandleV1>,
) -> Response {
    let mut choices = Vec::with_capacity(receivers.len());
    let mut usage = None;
    for IndexedGenerationReceiverV1 {
        index,
        mut receiver,
    } in receivers
    {
        let mut splitter = ReasoningSplitterV1::new(context.reasoning);
        let mut content = String::new();
        let mut reasoning_content = String::new();
        let mut logprobs = None;
        let mut completed = false;
        while let Some(event) = receiver.recv().await {
            match event {
                SchedulerEventV1::Delta(delta) => {
                    if let Some(metrics) = &mut metrics {
                        metrics.observe_ttft_since_start();
                    }
                    append_split_parts(splitter.feed(&delta), &mut content, &mut reasoning_content);
                }
                SchedulerEventV1::Logprobs(values) => {
                    logprobs = Some(ChatLogprobsV1::from_backend(values));
                }
                SchedulerEventV1::Finished(completion) => {
                    if let Some(metrics) = &mut metrics {
                        metrics.observe_ttft_since_start();
                    }
                    append_split_parts(splitter.finish(), &mut content, &mut reasoning_content);
                    if let Err(error) = merge_choice_usage(&mut usage, completion.usage) {
                        if let Some(metrics) = &mut metrics {
                            metrics.finish(RequestOutcomeV1::Error);
                        }
                        return error.into_response();
                    }
                    choices.push(ChatCompletionChoiceV1 {
                        index,
                        message: AssistantMessageV1 {
                            role: "assistant",
                            content,
                            reasoning_content: context
                                .reasoning
                                .separate_reasoning()
                                .then_some(reasoning_content),
                        },
                        logprobs,
                        finish_reason: completion.finish_reason,
                    });
                    completed = true;
                    break;
                }
                SchedulerEventV1::Failed(error) => {
                    if let Some(metrics) = &mut metrics {
                        metrics.finish(RequestOutcomeV1::Error);
                    }
                    return error.into_response();
                }
            }
        }
        if !completed {
            if let Some(metrics) = &mut metrics {
                metrics.finish(RequestOutcomeV1::Error);
            }
            return ApiErrorV1::generation_failed("generation ended without a terminal event")
                .into_response();
        }
    }
    let usage = usage.expect("validated request always contains at least one choice");
    if let Some(metrics) = &mut metrics {
        metrics.record_tokens(usage.prompt_tokens, usage.completion_tokens);
        metrics.finish(RequestOutcomeV1::Success);
    }
    axum::Json(ChatCompletionResponseV1 {
        id: context.id,
        object: "chat.completion",
        created: context.created,
        model: context.model,
        choices,
        usage,
    })
    .into_response()
}

fn stream_chat_completion(
    receivers: Vec<IndexedGenerationReceiverV1>,
    context: ResponseContextV1,
    metrics: Option<MetricsRequestHandleV1>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let reasoning = context.reasoning;
    let mut receivers = VecDeque::from(receivers);
    let current = receivers
        .pop_front()
        .expect("validated request always contains at least one choice");
    let state = StreamStateV1 {
        current,
        receivers,
        context,
        role_pending: true,
        queued: VecDeque::new(),
        terminal: false,
        splitter: ReasoningSplitterV1::new(reasoning),
        metrics,
        usage: None,
        logprobs: None,
    };
    Sse::new(stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.queued.pop_front() {
                return Some((Ok(event), state));
            }
            if state.terminal {
                return None;
            }
            if state.role_pending {
                state.role_pending = false;
                let chunk = StreamChunkV1::role(&state.context, state.current.index);
                return Some((Ok(json_event(&chunk)), state));
            }
            match state.current.receiver.recv().await {
                Some(SchedulerEventV1::Delta(delta)) => {
                    if let Some(metrics) = &mut state.metrics {
                        metrics.observe_ttft_since_start();
                    }
                    for part in state.splitter.feed(&delta) {
                        let chunk =
                            StreamChunkV1::delta(&state.context, state.current.index, &part);
                        state.queued.push_back(json_event(&chunk));
                    }
                }
                Some(SchedulerEventV1::Logprobs(values)) => {
                    state.logprobs = Some(ChatLogprobsV1::from_backend(values));
                }
                Some(SchedulerEventV1::Finished(completion)) => {
                    if let Some(metrics) = &mut state.metrics {
                        metrics.observe_ttft_since_start();
                    }
                    for part in state.splitter.finish() {
                        let chunk =
                            StreamChunkV1::delta(&state.context, state.current.index, &part);
                        state.queued.push_back(json_event(&chunk));
                    }
                    if merge_choice_usage(&mut state.usage, completion.usage).is_err() {
                        if let Some(metrics) = &mut state.metrics {
                            metrics.finish(RequestOutcomeV1::Error);
                        }
                        state.terminal = true;
                        let error = ApiErrorV1::generation_failed(
                            "independent choice token accounting is inconsistent",
                        );
                        return Some((Ok(json_event(&error.envelope())), state));
                    }
                    let last_choice = state.receivers.is_empty();
                    let chunk = StreamChunkV1::finished(
                        &state.context,
                        state.current.index,
                        completion.finish_reason,
                        last_choice.then_some(
                            state
                                .usage
                                .expect("usage was merged before the final choice"),
                        ),
                        state.logprobs.as_ref(),
                    );
                    state.queued.push_back(json_event(&chunk));
                    if let Some(next) = state.receivers.pop_front() {
                        state.current = next;
                        state.splitter = ReasoningSplitterV1::new(state.context.reasoning);
                        state.role_pending = true;
                        state.logprobs = None;
                    } else {
                        let usage = state
                            .usage
                            .expect("usage was merged before the final choice");
                        if let Some(metrics) = &mut state.metrics {
                            metrics.record_tokens(usage.prompt_tokens, usage.completion_tokens);
                            metrics.finish(RequestOutcomeV1::Success);
                        }
                        state.queued.push_back(Event::default().data("[DONE]"));
                        state.terminal = true;
                    }
                }
                Some(SchedulerEventV1::Failed(error)) => {
                    if let Some(metrics) = &mut state.metrics {
                        metrics.finish(RequestOutcomeV1::Error);
                    }
                    state.terminal = true;
                    return Some((Ok(json_event(&error.envelope())), state));
                }
                None => {
                    if let Some(metrics) = &mut state.metrics {
                        metrics.finish(RequestOutcomeV1::Error);
                    }
                    state.terminal = true;
                    let error =
                        ApiErrorV1::generation_failed("generation ended without a terminal event");
                    return Some((Ok(json_event(&error.envelope())), state));
                }
            }
        }
    }))
}

async fn resume_chat_completion(
    State(state): State<Arc<AppStateV1>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let response = if let Err(error) = authorize_user(&headers, &state.config) {
        error.into_response()
    } else if let Some(replay) = state.config.replay.clone() {
        match parse_last_event_id(&headers) {
            Ok(cursor) => match replay.read_after(&id, cursor) {
                Ok(_) => replay_stream(replay, id, cursor).into_response(),
                Err(ReplayErrorV1::NotFound) => ApiErrorV1::replay_not_found().into_response(),
                Err(ReplayErrorV1::CursorOutOfRange) => {
                    ApiErrorV1::replay_out_of_range().into_response()
                }
                Err(
                    ReplayErrorV1::Capacity
                    | ReplayErrorV1::EventTooLarge
                    | ReplayErrorV1::IdentifierExhausted
                    | ReplayErrorV1::Terminal,
                ) => {
                    ApiErrorV1::generation_failed("resumable stream is unavailable").into_response()
                }
            },
            Err(error) => error.into_response(),
        }
    } else {
        ApiErrorV1::replay_not_found().into_response()
    };
    record_http(&state, HttpEndpointV1::ChatReplay, &response);
    response
}

fn parse_last_event_id(headers: &HeaderMap) -> Result<u64, ApiErrorV1> {
    let mut values = headers.get_all(&LAST_EVENT_ID).iter();
    let Some(first) = values.next() else {
        return Ok(0);
    };
    if values.next().is_some() {
        return Err(ApiErrorV1::invalid_value(
            "Last-Event-ID",
            "Last-Event-ID must occur at most once",
        ));
    }
    let value = first
        .to_str()
        .ok()
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| {
            ApiErrorV1::invalid_value(
                "Last-Event-ID",
                "Last-Event-ID must be an unsigned decimal integer",
            )
        })?;
    value.parse::<u64>().map_err(|_| {
        ApiErrorV1::invalid_value(
            "Last-Event-ID",
            "Last-Event-ID is outside the supported range",
        )
    })
}

fn spawn_resumable_producer(
    receivers: Vec<IndexedGenerationReceiverV1>,
    context: ResponseContextV1,
    replay: ResumableStoreV1,
    mut metrics: Option<MetricsRequestHandleV1>,
) {
    tokio::spawn(async move {
        let choice_count = receivers.len();
        let mut usage = None;
        for (
            position,
            IndexedGenerationReceiverV1 {
                index,
                mut receiver,
            },
        ) in receivers.into_iter().enumerate()
        {
            let role = StreamChunkV1::role(&context, index);
            if append_replay_json(&replay, &context.id, &role, false).is_err() {
                if let Some(metrics) = &mut metrics {
                    metrics.finish(RequestOutcomeV1::Error);
                }
                terminate_replay_with_error(&replay, &context.id);
                return;
            }
            let mut splitter = ReasoningSplitterV1::new(context.reasoning);
            let mut logprobs = None;
            let mut completed = false;
            while let Some(event) = receiver.recv().await {
                match event {
                    SchedulerEventV1::Delta(delta) => {
                        if let Some(metrics) = &mut metrics {
                            metrics.observe_ttft_since_start();
                        }
                        for part in splitter.feed(&delta) {
                            let chunk = StreamChunkV1::delta(&context, index, &part);
                            if append_replay_json(&replay, &context.id, &chunk, false).is_err() {
                                if let Some(metrics) = &mut metrics {
                                    metrics.finish(RequestOutcomeV1::Error);
                                }
                                terminate_replay_with_error(&replay, &context.id);
                                return;
                            }
                        }
                    }
                    SchedulerEventV1::Logprobs(values) => {
                        logprobs = Some(ChatLogprobsV1::from_backend(values));
                    }
                    SchedulerEventV1::Finished(completion) => {
                        if let Some(metrics) = &mut metrics {
                            metrics.observe_ttft_since_start();
                        }
                        for part in splitter.finish() {
                            let chunk = StreamChunkV1::delta(&context, index, &part);
                            if append_replay_json(&replay, &context.id, &chunk, false).is_err() {
                                if let Some(metrics) = &mut metrics {
                                    metrics.finish(RequestOutcomeV1::Error);
                                }
                                terminate_replay_with_error(&replay, &context.id);
                                return;
                            }
                        }
                        if merge_choice_usage(&mut usage, completion.usage).is_err() {
                            if let Some(metrics) = &mut metrics {
                                metrics.finish(RequestOutcomeV1::Error);
                            }
                            terminate_replay_with_error(&replay, &context.id);
                            return;
                        }
                        let last_choice = position + 1 == choice_count;
                        let chunk = StreamChunkV1::finished(
                            &context,
                            index,
                            completion.finish_reason,
                            last_choice.then_some(
                                usage.expect("usage was merged before the final choice"),
                            ),
                            logprobs.as_ref(),
                        );
                        if append_replay_json(&replay, &context.id, &chunk, false).is_err() {
                            if let Some(metrics) = &mut metrics {
                                metrics.finish(RequestOutcomeV1::Error);
                            }
                            terminate_replay_with_error(&replay, &context.id);
                            return;
                        }
                        completed = true;
                        break;
                    }
                    SchedulerEventV1::Failed(error) => {
                        if let Some(metrics) = &mut metrics {
                            metrics.finish(RequestOutcomeV1::Error);
                        }
                        let data = serde_json::to_string(&error.envelope()).unwrap_or_else(|_| {
                            r#"{"error":{"message":"response serialization failed","type":"server_error","param":null,"code":"generation_failed"}}"#.to_owned()
                        });
                        if replay.append(&context.id, data, true).is_err() {
                            replay.terminate(&context.id);
                        }
                        return;
                    }
                }
            }
            if !completed {
                if let Some(metrics) = &mut metrics {
                    metrics.finish(RequestOutcomeV1::Error);
                }
                terminate_replay_with_error(&replay, &context.id);
                return;
            }
        }
        if replay
            .append(&context.id, "[DONE]".to_owned(), true)
            .is_err()
        {
            if let Some(metrics) = &mut metrics {
                metrics.finish(RequestOutcomeV1::Error);
            }
            terminate_replay_with_error(&replay, &context.id);
            return;
        }
        let usage = usage.expect("validated request always contains at least one choice");
        if let Some(metrics) = &mut metrics {
            metrics.record_tokens(usage.prompt_tokens, usage.completion_tokens);
            metrics.finish(RequestOutcomeV1::Success);
        }
    });
}

fn terminate_replay_with_error(replay: &ResumableStoreV1, id: &str) {
    let error = ApiErrorV1::generation_failed("resumable replay limit exceeded");
    let data = serde_json::to_string(&error.envelope()).unwrap_or_else(|_| {
        r#"{"error":{"message":"resumable replay failed","type":"server_error","param":null,"code":"generation_failed"}}"#.to_owned()
    });
    if replay.append(id, data, true).is_err() {
        replay.terminate(id);
    }
}

fn append_replay_json(
    replay: &ResumableStoreV1,
    id: &str,
    value: &impl Serialize,
    terminal: bool,
) -> Result<u64, ReplayErrorV1> {
    let data = serde_json::to_string(value).map_err(|_| ReplayErrorV1::Terminal)?;
    replay.append(id, data, terminal)
}

struct ReplayStreamStateV1 {
    replay: ResumableStoreV1,
    id: String,
    cursor: u64,
    queued: VecDeque<crate::resume::ReplayEventV1>,
    terminal: bool,
}

fn replay_stream(
    replay: ResumableStoreV1,
    id: String,
    cursor: u64,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state = ReplayStreamStateV1 {
        replay,
        id,
        cursor,
        queued: VecDeque::new(),
        terminal: false,
    };
    Sse::new(stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.queued.pop_front() {
                state.cursor = event.id;
                return Some((
                    Ok(Event::default().id(event.id.to_string()).data(event.data)),
                    state,
                ));
            }
            if state.terminal {
                return None;
            }
            match state.replay.read_after(&state.id, state.cursor) {
                Ok(read) => {
                    state.terminal = read.terminal;
                    state.queued.extend(read.events);
                    if state.queued.is_empty() && !state.terminal {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }
                Err(_) => return None,
            }
        }
    }))
}

fn json_event(value: &impl Serialize) -> Event {
    Event::default().json_data(value).unwrap_or_else(|_| {
        Event::default().data(
            r#"{"error":{"message":"response serialization failed","type":"server_error","param":null,"code":"generation_failed"}}"#,
        )
    })
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Debug)]
struct SplitPartV1 {
    content: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(Debug)]
struct ReasoningSplitterV1 {
    separate: bool,
    in_reasoning: bool,
    opening_checked: bool,
    trim_content_prefix: bool,
    pending: String,
}

impl ReasoningSplitterV1 {
    fn new(options: ReasoningOptionsV1) -> Self {
        Self {
            separate: options.separate_reasoning(),
            in_reasoning: options.enabled(),
            opening_checked: !options.enabled(),
            trim_content_prefix: false,
            pending: String::new(),
        }
    }

    fn feed(&mut self, delta: &str) -> Vec<SplitPartV1> {
        if !self.separate {
            return vec![SplitPartV1 {
                content: Some(delta.to_owned()),
                reasoning_content: None,
            }];
        }
        if !self.in_reasoning {
            let content = self.trim_content(delta);
            return (!content.is_empty())
                .then_some(SplitPartV1 {
                    content: Some(content),
                    reasoning_content: None,
                })
                .into_iter()
                .collect();
        }

        self.pending.push_str(delta);
        if !self.opening_checked {
            if self.pending.len() < THINK_OPEN.len() && THINK_OPEN.starts_with(&self.pending) {
                return Vec::new();
            }
            if self.pending.starts_with(THINK_OPEN) {
                self.pending.drain(..THINK_OPEN.len());
                while matches!(self.pending.as_bytes().first(), Some(b'\r' | b'\n')) {
                    self.pending.remove(0);
                }
            }
            self.opening_checked = true;
        }

        if let Some(close) = self.pending.find(THINK_CLOSE) {
            let reasoning = self.pending[..close].to_owned();
            let remainder = self.pending[close + THINK_CLOSE.len()..].to_owned();
            self.pending.clear();
            self.in_reasoning = false;
            self.trim_content_prefix = true;
            let content = self.trim_content(&remainder);
            let mut parts = Vec::new();
            if !reasoning.is_empty() {
                parts.push(SplitPartV1 {
                    content: None,
                    reasoning_content: Some(reasoning),
                });
            }
            if !content.is_empty() {
                parts.push(SplitPartV1 {
                    content: Some(content),
                    reasoning_content: None,
                });
            }
            return parts;
        }

        let indices = self
            .pending
            .char_indices()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let keep = THINK_CLOSE.chars().count().saturating_sub(1);
        if indices.len() <= keep {
            return Vec::new();
        }
        let split = indices[indices.len() - keep];
        let reasoning = self.pending[..split].to_owned();
        self.pending.drain(..split);
        vec![SplitPartV1 {
            content: None,
            reasoning_content: Some(reasoning),
        }]
    }

    fn finish(&mut self) -> Vec<SplitPartV1> {
        if !self.separate || self.pending.is_empty() {
            return Vec::new();
        }
        let value = std::mem::take(&mut self.pending);
        if self.in_reasoning {
            vec![SplitPartV1 {
                content: None,
                reasoning_content: Some(value),
            }]
        } else {
            let content = self.trim_content(&value);
            (!content.is_empty())
                .then_some(SplitPartV1 {
                    content: Some(content),
                    reasoning_content: None,
                })
                .into_iter()
                .collect()
        }
    }

    fn trim_content(&mut self, value: &str) -> String {
        if !self.trim_content_prefix {
            return value.to_owned();
        }
        let trimmed = value.trim_start_matches(['\r', '\n']);
        if !trimmed.is_empty() {
            self.trim_content_prefix = false;
        }
        trimmed.to_owned()
    }
}

fn append_split_parts(
    parts: Vec<SplitPartV1>,
    content: &mut String,
    reasoning_content: &mut String,
) {
    for part in parts {
        if let Some(value) = part.content {
            content.push_str(&value);
        }
        if let Some(value) = part.reasoning_content {
            reasoning_content.push_str(&value);
        }
    }
}

fn merge_choice_usage(
    aggregate: &mut Option<TokenUsageV1>,
    choice: TokenUsageV1,
) -> Result<(), ApiErrorV1> {
    let Some(current) = aggregate.as_mut() else {
        *aggregate = Some(choice);
        return Ok(());
    };
    if current.prompt_tokens != choice.prompt_tokens {
        return Err(ApiErrorV1::generation_failed(
            "independent choices reported different prompt token counts",
        ));
    }
    current.completion_tokens = current
        .completion_tokens
        .checked_add(choice.completion_tokens)
        .ok_or_else(|| ApiErrorV1::generation_failed("choice token accounting overflowed"))?;
    current.total_tokens = current
        .prompt_tokens
        .checked_add(current.completion_tokens)
        .ok_or_else(|| ApiErrorV1::generation_failed("choice token accounting overflowed"))?;
    Ok(())
}

struct StreamStateV1 {
    current: IndexedGenerationReceiverV1,
    receivers: VecDeque<IndexedGenerationReceiverV1>,
    context: ResponseContextV1,
    role_pending: bool,
    queued: VecDeque<Event>,
    terminal: bool,
    splitter: ReasoningSplitterV1,
    metrics: Option<MetricsRequestHandleV1>,
    usage: Option<TokenUsageV1>,
    logprobs: Option<ChatLogprobsV1>,
}

#[derive(Clone, Debug)]
struct ResponseContextV1 {
    id: String,
    created: u64,
    model: String,
    reasoning: ReasoningOptionsV1,
}

impl ResponseContextV1 {
    fn new(model: &str, reasoning: ReasoningOptionsV1) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            id: format!("chatcmpl-sllm-{created:016x}{counter:016x}"),
            created,
            model: model.to_owned(),
            reasoning,
        }
    }
}

#[derive(Serialize)]
struct ModelListV1<'a> {
    object: &'static str,
    data: Vec<ModelObjectV1<'a>>,
}

#[derive(Serialize)]
struct ModelObjectV1<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    owned_by: &'a str,
}

#[derive(Serialize)]
struct ChatCompletionResponseV1 {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoiceV1>,
    usage: TokenUsageV1,
}

#[derive(Serialize)]
struct ChatCompletionChoiceV1 {
    index: u32,
    message: AssistantMessageV1,
    logprobs: Option<ChatLogprobsV1>,
    finish_reason: FinishReasonV1,
}

#[derive(Clone, Debug, Serialize)]
struct ChatLogprobsV1 {
    content: Vec<ChatTokenLogprobV1>,
}

impl ChatLogprobsV1 {
    fn from_backend(values: Vec<BackendTokenLogprobV1>) -> Self {
        Self {
            content: values
                .into_iter()
                .map(|value| ChatTokenLogprobV1 {
                    token: value.token,
                    bytes: value.bytes,
                    logprob: value.logprob,
                    top_logprobs: value
                        .top_logprobs
                        .into_iter()
                        .map(|top| ChatTopLogprobV1 {
                            token: top.token,
                            bytes: top.bytes,
                            logprob: top.logprob,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct ChatTokenLogprobV1 {
    token: String,
    bytes: Option<Vec<u8>>,
    logprob: f64,
    top_logprobs: Vec<ChatTopLogprobV1>,
}

#[derive(Clone, Debug, Serialize)]
struct ChatTopLogprobV1 {
    token: String,
    bytes: Option<Vec<u8>>,
    logprob: f64,
}

#[derive(Serialize)]
struct AssistantMessageV1 {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Serialize)]
struct StreamChunkV1<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: [StreamChoiceV1<'a>; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<TokenUsageV1>,
}

impl<'a> StreamChunkV1<'a> {
    fn role(context: &'a ResponseContextV1, index: u32) -> Self {
        Self::new(
            context,
            index,
            StreamDeltaV1 {
                role: Some("assistant"),
                content: Some(""),
                reasoning_content: None,
            },
            None,
            None,
            None,
        )
    }

    fn delta(context: &'a ResponseContextV1, index: u32, part: &'a SplitPartV1) -> Self {
        Self::new(
            context,
            index,
            StreamDeltaV1 {
                role: None,
                content: part.content.as_deref(),
                reasoning_content: part.reasoning_content.as_deref(),
            },
            None,
            None,
            None,
        )
    }

    fn finished(
        context: &'a ResponseContextV1,
        index: u32,
        finish_reason: FinishReasonV1,
        usage: Option<TokenUsageV1>,
        logprobs: Option<&'a ChatLogprobsV1>,
    ) -> Self {
        Self::new(
            context,
            index,
            StreamDeltaV1 {
                role: None,
                content: None,
                reasoning_content: None,
            },
            Some(finish_reason),
            usage,
            logprobs,
        )
    }

    fn new(
        context: &'a ResponseContextV1,
        index: u32,
        delta: StreamDeltaV1<'a>,
        finish_reason: Option<FinishReasonV1>,
        usage: Option<TokenUsageV1>,
        logprobs: Option<&'a ChatLogprobsV1>,
    ) -> Self {
        Self {
            id: &context.id,
            object: "chat.completion.chunk",
            created: context.created,
            model: &context.model,
            choices: [StreamChoiceV1 {
                index,
                delta,
                logprobs,
                finish_reason,
            }],
            usage,
        }
    }
}

#[derive(Serialize)]
struct StreamChoiceV1<'a> {
    index: u32,
    delta: StreamDeltaV1<'a>,
    logprobs: Option<&'a ChatLogprobsV1>,
    finish_reason: Option<FinishReasonV1>,
}

#[derive(Serialize)]
struct StreamDeltaV1<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
}
