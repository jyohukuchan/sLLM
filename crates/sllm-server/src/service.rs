use std::collections::{BTreeMap, VecDeque};
use std::convert::Infallible;
use std::net::SocketAddr;
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
use base64::Engine;
use futures_util::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use sllm_core::{CosineEmbeddingRerankV1, EmbeddingVectorV1};
use sllm_frontend::{
    DecodeModeV1, Qwen35ChatMessageV1, Qwen35RenderOptionsV1, ThinkingModeV1, TokenPieceV1,
    TokenizeOptionsV1,
};
use tower_http::cors::CorsLayer;

use crate::api::{
    ApiErrorV1, ChatCompatibilityProfileV1, ChatCompletionRequestV1, ErrorCodeV1, FinishReasonV1,
    GenerationRequestInputV1, MAX_REQUEST_BODY_BYTES, ReasoningOptionsV1, TokenUsageV1,
    parse_chat_completion_request_for_profile,
};
use crate::hugging_face::{HuggingFaceErrorV1, HuggingFaceHubV1};
use crate::lifecycle::ServerLifecycleV1;
use crate::metrics::{HttpEndpointV1, MetricsRequestHandleV1, RequestOutcomeV1, ServerMetricsV1};
use crate::model_library::{ModelLibraryDeviceV1, ModelLibraryV1};
use crate::model_lifecycle::{
    ModelLifecycleErrorV1, ModelLifecycleLeaseV1, ModelLifecycleRegistryV1,
};
use crate::phase42_api::{
    self as phase42, ApplyTemplateRequestV1, CompletionRequestV1, DetokenizeRequestV1,
    EmbeddingEncodingFormatV1, EmbeddingRequestV1, InfillRequestV1, InputTokensInputV1,
    InputTokensRequestV1, PromptV1, RerankRequestV1, TemplateMessageV1, TemplateRoleV1,
    TokenizeRequestV1,
};
use crate::resume::{ReplayErrorV1, ResumableStoreV1};
use crate::runtime::{
    BackendEmbeddingInputV1, BackendEmbeddingRequestV1, BackendTokenLogprobV1,
    GenerationReceiverV1, ModelRegistryEntryV1, ModelRegistryV1, SchedulerEventV1, SchedulerV1,
};
use crate::security::CredentialStoreV1;

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

const LAST_EVENT_ID: HeaderName = HeaderName::from_static("last-event-id");
const MAX_CORS_ORIGINS: usize = 32;

#[derive(Clone)]
pub struct ServerConfigV1 {
    pub(crate) credentials: CredentialStoreV1,
    compatibility_profile: ChatCompatibilityProfileV1,
    lifecycle: ServerLifecycleV1,
    pub(crate) metrics: Option<ServerMetricsV1>,
    pub(crate) replay: Option<ResumableStoreV1>,
    cors_origins: Vec<HeaderValue>,
    hardware: Option<ModelLibraryDeviceV1>,
    model_library: Option<ModelLibraryV1>,
    hugging_face: Option<HuggingFaceHubV1>,
    loopback_admin: bool,
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
            hardware: None,
            model_library: None,
            hugging_face: None,
            loopback_admin: false,
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

    pub fn with_hardware(mut self, hardware: ModelLibraryDeviceV1) -> Self {
        self.hardware = Some(hardware);
        self
    }

    pub fn with_model_library(mut self, model_library: ModelLibraryV1) -> Self {
        self.hugging_face = Some(HuggingFaceHubV1::new(model_library.clone()));
        self.model_library = Some(model_library);
        self
    }

    /// Allows credential-free administrative actions only for an explicitly
    /// verified loopback listener.
    pub fn with_loopback_admin(mut self, listen: SocketAddr) -> Result<Self, ApiErrorV1> {
        if !listen.ip().is_loopback() {
            return Err(ApiErrorV1::generation_failed(
                "credential-free admin requires a loopback listener",
            ));
        }
        self.loopback_admin = true;
        Ok(self)
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
pub(crate) struct AppStateV1 {
    pub(crate) registry: ModelRegistryV1,
    pub(crate) lifecycle: Option<Arc<ModelLifecycleRegistryV1>>,
    pub(crate) scheduler: SchedulerV1,
    pub(crate) config: ServerConfigV1,
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
        lifecycle: None,
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
        .route("/v1/completions", post(create_completion))
        .route("/v1/embeddings", post(create_embeddings))
        .route("/v1/rerank", post(create_rerank))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/v1/apply-template", post(apply_template))
        .route("/v1/input-tokens", post(input_tokens))
        .route("/v1/infill", post(create_infill))
        .route(
            "/v1/responses",
            post(crate::phase43_service::create_response),
        )
        .route(
            "/v1/responses/{id}/events",
            get(crate::phase43_service::resume_response),
        )
        .route(
            "/v1/messages",
            post(crate::phase43_service::create_anthropic_message),
        )
        .route(
            "/v1/messages/{id}/events",
            get(crate::phase43_service::resume_anthropic_message),
        )
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
                .allow_headers([
                    AUTHORIZATION,
                    CONTENT_TYPE,
                    LAST_EVENT_ID,
                    HeaderName::from_static("anthropic-version"),
                ]),
        )
    }
}

/// Builds the API router with the bounded dynamic model lifecycle registry.
/// The static registry remains available for compatibility and metadata, while
/// every model execution path resolves through `lifecycle`.
pub fn build_dynamic_router_v1(
    lifecycle: ModelLifecycleRegistryV1,
    scheduler: SchedulerV1,
    config: ServerConfigV1,
) -> Router {
    let lifecycle = Arc::new(lifecycle);
    let cors_origins = config.cors_origins.clone();
    let state = Arc::new(AppStateV1 {
        registry: ModelRegistryV1::empty_for_dynamic(),
        lifecycle: Some(Arc::clone(&lifecycle)),
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
        .route("/admin/models/{alias}/load", post(admin_model_load))
        .route("/admin/models/{alias}/preload", post(admin_model_load))
        .route("/admin/models/{alias}/unload", post(admin_model_unload))
        .route(
            "/admin/models/{alias}/clear-quarantine",
            post(admin_model_clear_quarantine),
        )
        .route("/admin/models/evict-idle", post(admin_model_evict_idle))
        .route("/admin/model-library", get(admin_model_library))
        .route(
            "/admin/model-library/browse",
            post(admin_model_library_browse),
        )
        .route(
            "/admin/model-library/select",
            post(admin_model_library_select),
        )
        .route(
            "/admin/model-library/rescan",
            post(admin_model_library_rescan),
        )
        .route("/admin/hugging-face/status", get(admin_hugging_face_status))
        .route(
            "/admin/hugging-face/search",
            post(admin_hugging_face_search),
        )
        .route("/admin/hugging-face/files", post(admin_hugging_face_files))
        .route(
            "/admin/hugging-face/downloads",
            post(admin_hugging_face_start_download),
        )
        .route(
            "/admin/hugging-face/downloads/{id}",
            get(admin_hugging_face_download_job),
        )
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(create_chat_completion))
        .route("/v1/completions", post(create_completion))
        .route("/v1/embeddings", post(create_embeddings))
        .route("/v1/rerank", post(create_rerank))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/v1/apply-template", post(apply_template))
        .route("/v1/input-tokens", post(input_tokens))
        .route("/v1/infill", post(create_infill))
        .route(
            "/v1/responses",
            post(crate::phase43_service::create_response),
        )
        .route(
            "/v1/responses/{id}/events",
            get(crate::phase43_service::resume_response),
        )
        .route(
            "/v1/messages",
            post(crate::phase43_service::create_anthropic_message),
        )
        .route(
            "/v1/messages/{id}/events",
            get(crate::phase43_service::resume_anthropic_message),
        )
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
                .allow_headers([
                    AUTHORIZATION,
                    CONTENT_TYPE,
                    LAST_EVENT_ID,
                    HeaderName::from_static("anthropic-version"),
                ]),
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
        if let Some(lifecycle) = state.lifecycle.as_ref() {
            for alias in lifecycle.configured_aliases() {
                metrics.set_model_ready(&alias, ready);
            }
        } else {
            for entry in state.registry.entries() {
                metrics.set_model_ready(entry.alias(), ready);
            }
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
        if let Some(lifecycle) = state.lifecycle.as_ref() {
            for alias in lifecycle.configured_aliases() {
                metrics.set_model_ready(&alias, ready);
            }
        } else {
            for entry in state.registry.entries() {
                metrics.set_model_ready(entry.alias(), ready);
            }
        }
        let memory = if let Some(lifecycle) = state.lifecycle.as_ref() {
            lifecycle.observability_snapshots()
        } else {
            state
                .registry
                .entries()
                .iter()
                .map(|entry| (entry.alias().to_owned(), entry.observability_snapshot()))
                .collect::<Vec<_>>()
        };
        let memory = memory
            .iter()
            .map(|(alias, snapshot)| (alias.as_str(), *snapshot))
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
        let models = if let Some(lifecycle) = state.lifecycle.as_ref() {
            lifecycle
                .snapshots()
                .into_iter()
                .map(|snapshot| {
                    serde_json::json!({
                        "alias": snapshot.alias,
                        "lifecycle": snapshot.state,
                        "active_leases": snapshot.active_leases,
                        "resident_bytes": snapshot.resident_bytes,
                        "last_used": snapshot.last_used,
                    })
                })
                .collect::<Vec<_>>()
        } else {
            state
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
                .collect::<Vec<_>>()
        };
        axum::Json(serde_json::json!({
            "schema_version": "sllm-server-props-v1",
            "state": state.config.lifecycle.state(),
            "models": models,
            "scheduler": state.scheduler.snapshot(),
            "hardware": state.config.hardware.as_ref().map(|hardware| serde_json::json!({
                "vendor": "AMD",
                "device_index": hardware.device_index,
                "name": hardware.name,
                "target": hardware.target,
                "memory_bytes": hardware.total_memory_bytes,
            })),
            "features": {
                "metrics": state.config.metrics.is_some(),
                "resumable_sse": state.config.replay.is_some(),
                "cors": !state.config.cors_origins.is_empty(),
                "authentication": !state.config.credentials.is_open(),
                "admin": state.config.credentials.has_admin_credentials() || state.config.loopback_admin,
                "model_library": state.config.model_library.is_some(),
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

async fn admin_model_load(
    State(state): State<Arc<AppStateV1>>,
    Path(alias): Path<String>,
    headers: HeaderMap,
    request: Request<Body>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match reject_admin_body(request).await {
            Err(response) => response,
            Ok(()) => match state.lifecycle.as_ref() {
                Some(lifecycle) => match tokio::task::spawn_blocking({
                    let lifecycle = Arc::clone(lifecycle);
                    let alias = alias.clone();
                    move || lifecycle.preload(&alias)
                })
                .await
                {
                    Ok(Ok(())) => axum::Json(serde_json::json!({"alias": alias, "state": "ready"}))
                        .into_response(),
                    Ok(Err(error)) => lifecycle_error_response(&alias, error),
                    Err(_) => ApiErrorV1::generation_failed("dynamic model load worker failed")
                        .into_response(),
                },
                None => ApiErrorV1::model_not_found(&alias).into_response(),
            },
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_model_unload(
    State(state): State<Arc<AppStateV1>>,
    Path(alias): Path<String>,
    headers: HeaderMap,
    request: Request<Body>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match reject_admin_body(request).await {
            Err(response) => response,
            Ok(()) => match state.lifecycle.as_ref() {
                Some(lifecycle) => match tokio::task::spawn_blocking({
                    let lifecycle = Arc::clone(lifecycle);
                    let alias = alias.clone();
                    move || lifecycle.unload(&alias)
                })
                .await
                {
                    Ok(Ok(())) => {
                        axum::Json(serde_json::json!({"alias": alias, "state": "unloaded"}))
                            .into_response()
                    }
                    Ok(Err(error)) => lifecycle_error_response(&alias, error),
                    Err(_) => ApiErrorV1::generation_failed("dynamic model unload worker failed")
                        .into_response(),
                },
                None => ApiErrorV1::model_not_found(&alias).into_response(),
            },
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_model_clear_quarantine(
    State(state): State<Arc<AppStateV1>>,
    Path(alias): Path<String>,
    headers: HeaderMap,
    request: Request<Body>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match reject_admin_body(request).await {
            Err(response) => response,
            Ok(()) => match state.lifecycle.as_ref() {
                Some(lifecycle) => match tokio::task::spawn_blocking({
                    let lifecycle = Arc::clone(lifecycle);
                    let alias = alias.clone();
                    move || lifecycle.clear_quarantine(&alias)
                })
                .await
                {
                    Ok(Ok(())) => {
                        axum::Json(serde_json::json!({"alias": alias, "state": "unloaded"}))
                            .into_response()
                    }
                    Ok(Err(error)) => lifecycle_error_response(&alias, error),
                    Err(_) => ApiErrorV1::generation_failed("dynamic model cleanup worker failed")
                        .into_response(),
                },
                None => ApiErrorV1::model_not_found(&alias).into_response(),
            },
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_model_evict_idle(
    State(state): State<Arc<AppStateV1>>,
    headers: HeaderMap,
    request: Request<Body>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match reject_admin_body(request).await {
            Err(response) => response,
            Ok(()) => match state.lifecycle.as_ref() {
                Some(lifecycle) => match tokio::task::spawn_blocking({
                    let lifecycle = Arc::clone(lifecycle);
                    move || lifecycle.evict_idle()
                })
                .await
                {
                    Ok(Ok(count)) => {
                        axum::Json(serde_json::json!({"evicted": count})).into_response()
                    }
                    Ok(Err(error)) => lifecycle_error_response("*", error),
                    Err(_) => ApiErrorV1::generation_failed("dynamic model eviction worker failed")
                        .into_response(),
                },
                None => ApiErrorV1::generation_failed("dynamic model lifecycle is not configured")
                    .into_response(),
            },
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelLibraryBrowseQueryV1 {
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelLibrarySelectRequestV1 {
    path: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HuggingFaceSearchRequestV1 {
    query: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HuggingFaceFilesRequestV1 {
    repo_id: String,
    revision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HuggingFaceDownloadRequestV1 {
    repo_id: String,
    revision: String,
    file_path: String,
    derived_lock_path: Option<String>,
}

async fn admin_model_library(State(state): State<Arc<AppStateV1>>, headers: HeaderMap) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match state.config.model_library.as_ref() {
            Some(library) => axum::Json(library.snapshot()).into_response(),
            None => model_library_unavailable(),
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_model_library_browse(
    State(state): State<Arc<AppStateV1>>,
    headers: HeaderMap,
    axum::Json(query): axum::Json<ModelLibraryBrowseQueryV1>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match state.config.model_library.as_ref() {
            Some(library) => {
                let library = library.clone();
                let path = query.path.map(std::path::PathBuf::from);
                match tokio::task::spawn_blocking(move || library.browse(path.as_deref())).await {
                    Ok(Ok(listing)) => axum::Json(listing).into_response(),
                    Ok(Err(error)) => ApiErrorV1::invalid_value("path", error).into_response(),
                    Err(_) => ApiErrorV1::generation_failed("model folder browser worker failed")
                        .into_response(),
                }
            }
            None => model_library_unavailable(),
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_model_library_select(
    State(state): State<Arc<AppStateV1>>,
    headers: HeaderMap,
    axum::Json(request): axum::Json<ModelLibrarySelectRequestV1>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else if request.path.len() > 4096 || request.path.contains('\0') {
        ApiErrorV1::invalid_value("path", "model folder path is invalid").into_response()
    } else {
        match state.config.model_library.as_ref() {
            Some(library) => {
                let library = library.clone();
                let path = std::path::PathBuf::from(request.path);
                match tokio::task::spawn_blocking(move || library.select(&path)).await {
                    Ok(Ok(snapshot)) => axum::Json(snapshot).into_response(),
                    Ok(Err(error)) => ApiErrorV1::invalid_value("path", error).into_response(),
                    Err(_) => ApiErrorV1::generation_failed("model folder selection worker failed")
                        .into_response(),
                }
            }
            None => model_library_unavailable(),
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_model_library_rescan(
    State(state): State<Arc<AppStateV1>>,
    headers: HeaderMap,
    request: Request<Body>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match reject_admin_body(request).await {
            Err(response) => response,
            Ok(()) => match state.config.model_library.as_ref() {
                Some(library) => {
                    let library = library.clone();
                    match tokio::task::spawn_blocking(move || library.rescan()).await {
                        Ok(Ok(snapshot)) => axum::Json(snapshot).into_response(),
                        Ok(Err(error)) => ApiErrorV1::generation_failed(error).into_response(),
                        Err(_) => {
                            ApiErrorV1::generation_failed("model folder rescan worker failed")
                                .into_response()
                        }
                    }
                }
                None => model_library_unavailable(),
            },
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_hugging_face_status(
    State(state): State<Arc<AppStateV1>>,
    headers: HeaderMap,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match state.config.hugging_face.as_ref() {
            Some(hub) => {
                let hub = hub.clone();
                match tokio::task::spawn_blocking(move || hub.status()).await {
                    Ok(status) => axum::Json(status).into_response(),
                    Err(_) => ApiErrorV1::generation_failed("Hugging Face status worker failed")
                        .into_response(),
                }
            }
            None => hugging_face_unavailable(),
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_hugging_face_search(
    State(state): State<Arc<AppStateV1>>,
    headers: HeaderMap,
    axum::Json(request): axum::Json<HuggingFaceSearchRequestV1>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match state.config.hugging_face.as_ref() {
            Some(hub) => {
                let hub = hub.clone();
                match tokio::task::spawn_blocking(move || hub.search(&request.query)).await {
                    Ok(Ok(results)) => axum::Json(results).into_response(),
                    Ok(Err(error)) => hugging_face_error(error),
                    Err(_) => ApiErrorV1::generation_failed("Hugging Face search worker failed")
                        .into_response(),
                }
            }
            None => hugging_face_unavailable(),
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_hugging_face_files(
    State(state): State<Arc<AppStateV1>>,
    headers: HeaderMap,
    axum::Json(request): axum::Json<HuggingFaceFilesRequestV1>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match state.config.hugging_face.as_ref() {
            Some(hub) => {
                let hub = hub.clone();
                match tokio::task::spawn_blocking(move || {
                    hub.files(&request.repo_id, &request.revision)
                })
                .await
                {
                    Ok(Ok(files)) => axum::Json(files).into_response(),
                    Ok(Err(error)) => hugging_face_error(error),
                    Err(_) => {
                        ApiErrorV1::generation_failed("Hugging Face repository worker failed")
                            .into_response()
                    }
                }
            }
            None => hugging_face_unavailable(),
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_hugging_face_start_download(
    State(state): State<Arc<AppStateV1>>,
    headers: HeaderMap,
    axum::Json(request): axum::Json<HuggingFaceDownloadRequestV1>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match state.config.hugging_face.as_ref() {
            Some(hub) => match hub.start_download(
                &request.repo_id,
                &request.revision,
                &request.file_path,
                request.derived_lock_path.as_deref(),
            ) {
                Ok(job) => (StatusCode::ACCEPTED, axum::Json(job)).into_response(),
                Err(error) => hugging_face_error(error),
            },
            None => hugging_face_unavailable(),
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

async fn admin_hugging_face_download_job(
    State(state): State<Arc<AppStateV1>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let response = if let Err(error) = authorize_admin(&headers, &state.config) {
        error.into_response()
    } else {
        match state.config.hugging_face.as_ref() {
            Some(hub) => match hub.download_job(&id) {
                Ok(job) => axum::Json(job).into_response(),
                Err(error) => hugging_face_error(error),
            },
            None => hugging_face_unavailable(),
        }
    };
    record_http(&state, HttpEndpointV1::Props, &response);
    response
}

fn hugging_face_error(error: HuggingFaceErrorV1) -> Response {
    match error.param {
        Some(param) => ApiErrorV1::invalid_value(param, error.message).into_response(),
        None => ApiErrorV1::generation_failed(error.message).into_response(),
    }
}

fn hugging_face_unavailable() -> Response {
    ApiErrorV1::generation_failed(
        "Hugging Face integration is available only on the loopback dynamic server",
    )
    .into_response()
}

fn model_library_unavailable() -> Response {
    ApiErrorV1::generation_failed("model library is available only on the loopback dynamic server")
        .into_response()
}

async fn reject_admin_body(request: Request<Body>) -> Result<(), Response> {
    match to_bytes(request.into_body(), 1).await {
        Ok(body) if body.is_empty() => Ok(()),
        _ => Err(ApiErrorV1::invalid_value(
            "body",
            "dynamic model actions do not accept a request body",
        )
        .into_response()),
    }
}

async fn list_models(State(state): State<Arc<AppStateV1>>, headers: HeaderMap) -> Response {
    let response = if let Err(error) = authorize_user(&headers, &state.config) {
        error.into_response()
    } else {
        let data = if let Some(lifecycle) = state.lifecycle.as_ref() {
            lifecycle
                .configured_aliases()
                .iter()
                .map(|alias| ModelObjectV1 {
                    id: alias.clone(),
                    object: "model",
                    created: 0,
                    owned_by: "sllm".to_owned(),
                })
                .collect()
        } else {
            state
                .registry
                .entries()
                .iter()
                .map(|entry| ModelObjectV1 {
                    id: entry.alias().to_owned(),
                    object: "model",
                    created: entry.created(),
                    owned_by: entry.owned_by().to_owned(),
                })
                .collect()
        };
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
    let (model, mut initial_lease) = match resolve_model(&state, request.model()) {
        Ok(value) => value,
        Err(response) => {
            record_http(&state, HttpEndpointV1::ChatCompletions, &response);
            return *response;
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
    let mut admission_error: Option<Response> = None;
    for index in 0..choice_count {
        let choice_request = match request.for_choice(index) {
            Ok(request) => request,
            Err(error) => {
                admission_error = Some(error.into_response());
                break;
            }
        };
        let (choice_model, lease) = if index == 0 {
            (Arc::clone(&model), initial_lease.take())
        } else {
            match state.lifecycle.as_ref() {
                Some(_) => match resolve_model(&state, request.model()) {
                    Ok((choice_model, lease)) => (choice_model, lease),
                    Err(error) => {
                        admission_error = Some(*error);
                        break;
                    }
                },
                None => (Arc::clone(&model), None),
            }
        };
        match state
            .scheduler
            .submit_with_lease(choice_model, choice_request, lease)
        {
            Ok(receiver) => receivers.push(IndexedGenerationReceiverV1 { index, receiver }),
            Err(error) => {
                admission_error = Some(error.into_response());
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
        let response = error;
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

async fn read_phase42_body(
    request: Request<Body>,
    config: &ServerConfigV1,
) -> Result<Vec<u8>, ApiErrorV1> {
    authorize_user(request.headers(), config)?;
    validate_content_type(request.headers())?;
    if request
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|length| length > MAX_REQUEST_BODY_BYTES as u64)
    {
        return Err(request_too_large_error());
    }
    to_bytes(request.into_body(), MAX_REQUEST_BODY_BYTES)
        .await
        .map(|body| body.to_vec())
        .map_err(|_| request_too_large_error())
}

fn request_too_large_error() -> ApiErrorV1 {
    ApiErrorV1::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "request body exceeds 100663296 bytes",
        "invalid_request_error",
        None,
        ErrorCodeV1::RequestTooLarge,
    )
}

fn phase42_error(error: phase42::ApiErrorV1) -> ApiErrorV1 {
    let code = match error.code().as_str() {
        "invalid_json" => ErrorCodeV1::InvalidJson,
        "invalid_value" => ErrorCodeV1::InvalidValue,
        "unsupported_parameter" => ErrorCodeV1::UnsupportedParameter,
        "request_too_large" => ErrorCodeV1::RequestTooLarge,
        _ => ErrorCodeV1::InvalidValue,
    };
    ApiErrorV1::new(
        error.status(),
        error.message().to_owned(),
        "invalid_request_error",
        error.param().map(str::to_owned),
        code,
    )
}

async fn create_completion(
    State(state): State<Arc<AppStateV1>>,
    request: Request<Body>,
) -> Response {
    let response = match read_phase42_body(request, &state.config).await {
        Err(error) => error.into_response(),
        Ok(body) => match phase42::parse_completion_request(&body) {
            Err(error) => phase42_error(error).into_response(),
            Ok(request) => handle_completion(&state, request).await,
        },
    };
    record_http(&state, HttpEndpointV1::Completions, &response);
    response
}

async fn handle_completion(state: &AppStateV1, request: CompletionRequestV1) -> Response {
    let (model, mut initial_lease) = match resolve_model(state, request.model()) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let inputs = match completion_inputs(request.prompt()) {
        Ok(inputs) => inputs,
        Err(error) => return error.into_response(),
    };
    let mut receivers = Vec::new();
    let mut index = 0_u32;
    for (prompt_index, input) in inputs.into_iter().enumerate() {
        let prompt_index =
            u32::try_from(prompt_index).expect("bounded completion prompt count fits u32");
        for choice in 0..request.n() {
            let expected_prompt_tokens = match &input {
                GenerationRequestInputV1::TokenIds(tokens) => {
                    Some(u64::try_from(tokens.len()).expect("bounded token input fits u64"))
                }
                _ => None,
            };
            let generated = match ChatCompletionRequestV1::from_completion(&request, input.clone())
                .and_then(|request| request.for_choice(choice))
            {
                Ok(request) => request,
                Err(error) => return error.into_response(),
            };
            let (choice_model, lease) = if index == 0 {
                (Arc::clone(&model), initial_lease.take())
            } else {
                match state.lifecycle.as_ref() {
                    Some(_) => match resolve_model(state, request.model()) {
                        Ok((choice_model, lease)) => (choice_model, lease),
                        Err(response) => return *response,
                    },
                    None => (Arc::clone(&model), None),
                }
            };
            match state
                .scheduler
                .submit_with_lease(choice_model, generated, lease)
            {
                Ok(receiver) => receivers.push(IndexedTextGenerationReceiverV1 {
                    index,
                    prompt_index,
                    expected_prompt_tokens,
                    receiver,
                }),
                Err(error) => {
                    drop(receivers);
                    return error.into_response();
                }
            }
            index = index.saturating_add(1);
        }
    }
    let context = TextResponseContextV1::new(model.alias());
    let metrics = state
        .config
        .metrics
        .as_ref()
        .and_then(|metrics| metrics.admit(model.alias(), request.stream()));
    if request.stream() {
        stream_text_completion(receivers, context, metrics).into_response()
    } else {
        non_stream_text_completion(receivers, context, metrics).await
    }
}

fn completion_inputs(prompt: &PromptV1) -> Result<Vec<GenerationRequestInputV1>, ApiErrorV1> {
    match prompt {
        PromptV1::Text(value) => Ok(vec![GenerationRequestInputV1::RawText(value.clone())]),
        PromptV1::Texts(values) => Ok(values
            .iter()
            .cloned()
            .map(GenerationRequestInputV1::RawText)
            .collect()),
        PromptV1::Tokens(values) => Ok(vec![GenerationRequestInputV1::TokenIds(values.clone())]),
        PromptV1::TokenSequences(values) => Ok(values
            .iter()
            .cloned()
            .map(GenerationRequestInputV1::TokenIds)
            .collect()),
    }
}

async fn create_embeddings(
    State(state): State<Arc<AppStateV1>>,
    request: Request<Body>,
) -> Response {
    let response = match read_phase42_body(request, &state.config).await {
        Err(error) => error.into_response(),
        Ok(body) => match phase42::parse_embedding_request(&body) {
            Err(error) => phase42_error(error).into_response(),
            Ok(request) => handle_embeddings(Arc::clone(&state), request).await,
        },
    };
    record_http(&state, HttpEndpointV1::Embeddings, &response);
    response
}

async fn handle_embeddings(state: Arc<AppStateV1>, request: EmbeddingRequestV1) -> Response {
    let (model, lease) = match resolve_model(&state, request.model()) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let Some(model_dimension) = model.embedding_dimension() else {
        return ApiErrorV1::new(
            StatusCode::BAD_REQUEST,
            "embeddings are not supported by this model lock",
            "invalid_request_error",
            Some("model".to_owned()),
            ErrorCodeV1::UnsupportedParameter,
        )
        .into_response();
    };
    let inputs = match embedding_inputs(request.input()) {
        Ok(inputs) => inputs,
        Err(error) => return error.into_response(),
    };
    let backend_request = match BackendEmbeddingRequestV1::new(inputs) {
        Ok(request) => request,
        Err(error) => return ApiErrorV1::invalid_value("input", error.to_string()).into_response(),
    };
    let expected_token_counts = match backend_request
        .inputs()
        .iter()
        .enumerate()
        .map(|(index, input)| {
            model.validate_embedding_input(input).map_err(|error| {
                ApiErrorV1::invalid_value(format!("input[{index}]"), error.to_string())
            })
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(counts) => counts,
        Err(error) => return error.into_response(),
    };
    let expected_dimensions = request.dimensions();
    let encoding = request.encoding_format();
    if expected_dimensions.is_some_and(|dimensions| dimensions != model_dimension) {
        return ApiErrorV1::invalid_value(
            "dimensions",
            format!("only the model hidden dimension {model_dimension} is supported"),
        )
        .into_response();
    }
    let receiver =
        match state
            .scheduler
            .submit_embedding_with_lease(model.clone(), backend_request, lease)
        {
            Ok(receiver) => receiver,
            Err(error) => return error.into_response(),
        };
    let batch = match receiver.recv().await {
        Ok(batch) => batch,
        Err(error) => return error.into_response(),
    };
    if batch.dimension() != model_dimension || batch.vectors().len() != expected_token_counts.len()
    {
        return ApiErrorV1::generation_failed(
            "embedding backend output dimension or row count differed from its model contract",
        )
        .into_response();
    }
    if batch
        .vectors()
        .iter()
        .zip(&expected_token_counts)
        .any(|(vector, expected)| vector.prompt_tokens() != *expected)
    {
        return ApiErrorV1::generation_failed(
            "embedding backend token usage differed from CPU admission",
        )
        .into_response();
    }
    let mut data = Vec::with_capacity(batch.vectors().len());
    for (index, vector) in batch.vectors().iter().enumerate() {
        let embedding = match encoding {
            EmbeddingEncodingFormatV1::Float => serde_json::Value::Array(
                vector
                    .values()
                    .iter()
                    .map(|value| serde_json::json!(value))
                    .collect(),
            ),
            EmbeddingEncodingFormatV1::Base64 => {
                let mut bytes = Vec::with_capacity(vector.values().len() * 4);
                for value in vector.values() {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                serde_json::json!(base64::engine::general_purpose::STANDARD.encode(bytes))
            }
        };
        data.push(serde_json::json!({
            "object": "embedding",
            "index": index,
            "embedding": embedding,
        }));
    }
    let total_tokens = match batch.total_prompt_tokens() {
        Ok(tokens) => tokens,
        Err(error) => return ApiErrorV1::generation_failed(error.to_string()).into_response(),
    };
    axum::Json(serde_json::json!({
        "object": "list",
        "data": data,
        "model": request.model(),
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }))
    .into_response()
}

fn embedding_inputs(prompt: &PromptV1) -> Result<Vec<BackendEmbeddingInputV1>, ApiErrorV1> {
    let values = match prompt {
        PromptV1::Text(value) => vec![BackendEmbeddingInputV1::Text(value.clone())],
        PromptV1::Texts(values) => values
            .iter()
            .cloned()
            .map(BackendEmbeddingInputV1::Text)
            .collect(),
        PromptV1::Tokens(values) => vec![BackendEmbeddingInputV1::TokenIds(values.clone())],
        PromptV1::TokenSequences(values) => values
            .iter()
            .cloned()
            .map(BackendEmbeddingInputV1::TokenIds)
            .collect(),
    };
    if values.len() > 256 {
        return Err(ApiErrorV1::invalid_value(
            "input",
            "too many embedding inputs",
        ));
    }
    Ok(values)
}

async fn create_rerank(State(state): State<Arc<AppStateV1>>, request: Request<Body>) -> Response {
    let response = match read_phase42_body(request, &state.config).await {
        Err(error) => error.into_response(),
        Ok(body) => match phase42::parse_rerank_request(&body) {
            Err(error) => phase42_error(error).into_response(),
            Ok(request) => handle_rerank(Arc::clone(&state), request).await,
        },
    };
    record_http(&state, HttpEndpointV1::Rerank, &response);
    response
}

async fn handle_rerank(state: Arc<AppStateV1>, request: RerankRequestV1) -> Response {
    let (model, lease) = match resolve_model(&state, request.model()) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    if model.embedding_dimension().is_none() {
        return ApiErrorV1::new(
            StatusCode::BAD_REQUEST,
            "rerank is not supported by this model lock",
            "invalid_request_error",
            Some("model".to_owned()),
            ErrorCodeV1::UnsupportedParameter,
        )
        .into_response();
    }
    let mut inputs = Vec::with_capacity(request.documents().len() + 1);
    inputs.push(BackendEmbeddingInputV1::Text(request.query().to_owned()));
    inputs.extend(
        request
            .documents()
            .iter()
            .cloned()
            .map(BackendEmbeddingInputV1::Text),
    );
    let request_input = match BackendEmbeddingRequestV1::new(inputs) {
        Ok(request) => request,
        Err(error) => return ApiErrorV1::invalid_value("input", error.to_string()).into_response(),
    };
    let expected_token_counts = match request_input
        .inputs()
        .iter()
        .enumerate()
        .map(|(index, input)| {
            model.validate_embedding_input(input).map_err(|error| {
                ApiErrorV1::invalid_value(format!("input[{index}]"), error.to_string())
            })
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(counts) => counts,
        Err(error) => return error.into_response(),
    };
    let receiver =
        match state
            .scheduler
            .submit_embedding_with_lease(model.clone(), request_input, lease)
        {
            Ok(receiver) => receiver,
            Err(error) => return error.into_response(),
        };
    let batch = match receiver.recv().await {
        Ok(batch) => batch,
        Err(error) => return error.into_response(),
    };
    if batch.dimension() != model.embedding_dimension().expect("checked above")
        || batch.vectors().len() != expected_token_counts.len()
        || batch
            .vectors()
            .iter()
            .zip(&expected_token_counts)
            .any(|(vector, expected)| vector.prompt_tokens() != *expected)
    {
        return ApiErrorV1::generation_failed(
            "rerank embedding output differed from its admitted model contract",
        )
        .into_response();
    }
    let vectors = match batch
        .vectors()
        .iter()
        .map(|vector| EmbeddingVectorV1::from_finite_f32(vector.values().to_vec()))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(vectors) => vectors,
        Err(error) => return ApiErrorV1::generation_failed(error.to_string()).into_response(),
    };
    let rerank = CosineEmbeddingRerankV1::new();
    let ranked = match rerank.rank(
        &vectors[0],
        &vectors[1..],
        request.top_n().map(|value| value as usize),
    ) {
        Ok(ranked) => ranked,
        Err(error) => return ApiErrorV1::generation_failed(error.to_string()).into_response(),
    };
    let total_tokens = match batch.total_prompt_tokens() {
        Ok(tokens) => tokens,
        Err(error) => return ApiErrorV1::generation_failed(error.to_string()).into_response(),
    };
    let results = ranked
        .into_iter()
        .map(|entry| {
            let mut result = serde_json::json!({
                "index": entry.index(),
                "relevance_score": entry.relevance_score(),
            });
            if request.return_documents() {
                result["document"] = serde_json::json!(request.documents()[entry.index()]);
            }
            result
        })
        .collect::<Vec<_>>();
    axum::Json(serde_json::json!({
        "id": format!("rerank-sllm-{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)),
        "object": "rerank",
        "profile": "cosine-embedding-v1",
        "model": request.model(),
        "results": results,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }))
    .into_response()
}

async fn tokenize(State(state): State<Arc<AppStateV1>>, request: Request<Body>) -> Response {
    let response = match read_phase42_body(request, &state.config).await {
        Err(error) => error.into_response(),
        Ok(body) => match phase42::parse_tokenize_request(&body) {
            Err(error) => phase42_error(error).into_response(),
            Ok(request) => handle_tokenize(&state, request),
        },
    };
    record_http(&state, HttpEndpointV1::Tokenize, &response);
    response
}

fn handle_tokenize(state: &AppStateV1, request: TokenizeRequestV1) -> Response {
    let (model, _lease) = match resolve_model(state, request.model()) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let options = if request.with_pieces() {
        TokenizeOptionsV1::with_pieces()
    } else {
        TokenizeOptionsV1::default()
    };
    let result = match model.tokenize_utility(request.text(), options) {
        Ok(result) => result,
        Err(error) => return utility_error("text", error.to_string()),
    };
    let pieces = result.pieces().map(|pieces| {
        pieces
            .iter()
            .map(|piece| match piece {
                TokenPieceV1::Utf8(value) => serde_json::json!({"kind": "utf8", "value": value}),
                TokenPieceV1::Bytes(value) => serde_json::json!({
                    "kind": "bytes",
                    "base64": base64::engine::general_purpose::STANDARD.encode(value),
                }),
            })
            .collect::<Vec<_>>()
    });
    axum::Json(serde_json::json!({
        "version": result.version(),
        "tokens": result.token_ids().as_slice(),
        "count": result.count(),
        "pieces": pieces,
        "model": request.model(),
        "model_lock_fingerprint": model.lock_fingerprint(),
    }))
    .into_response()
}

async fn detokenize(State(state): State<Arc<AppStateV1>>, request: Request<Body>) -> Response {
    let response = match read_phase42_body(request, &state.config).await {
        Err(error) => error.into_response(),
        Ok(body) => match phase42::parse_detokenize_request(&body) {
            Err(error) => phase42_error(error).into_response(),
            Ok(request) => handle_detokenize(&state, request),
        },
    };
    record_http(&state, HttpEndpointV1::Detokenize, &response);
    response
}

fn handle_detokenize(state: &AppStateV1, request: DetokenizeRequestV1) -> Response {
    let (model, _lease) = match resolve_model(state, request.model()) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let mode = if request.skip_special_tokens() {
        DecodeModeV1::SkipSpecialTokens
    } else {
        DecodeModeV1::PreserveSpecialTokens
    };
    let text = match model.detokenize_utility(request.tokens(), mode) {
        Ok(text) => text,
        Err(error) => return utility_error("tokens", error.to_string()),
    };
    axum::Json(serde_json::json!({
        "text": text,
        "tokens": request.tokens(),
        "count": request.tokens().len(),
        "model": request.model(),
        "model_lock_fingerprint": model.lock_fingerprint(),
    }))
    .into_response()
}

async fn apply_template(State(state): State<Arc<AppStateV1>>, request: Request<Body>) -> Response {
    let response = match read_phase42_body(request, &state.config).await {
        Err(error) => error.into_response(),
        Ok(body) => match phase42::parse_apply_template_request(&body) {
            Err(error) => phase42_error(error).into_response(),
            Ok(request) => handle_apply_template(&state, request),
        },
    };
    record_http(&state, HttpEndpointV1::ApplyTemplate, &response);
    response
}

fn handle_apply_template(state: &AppStateV1, request: ApplyTemplateRequestV1) -> Response {
    let (model, _lease) = match resolve_model(state, request.model()) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    if !model.reviewed_chat_template_available() {
        return ApiErrorV1::new(
            StatusCode::BAD_REQUEST,
            "reviewed chat template is not supported by this model lock",
            "invalid_request_error",
            Some("model".to_owned()),
            ErrorCodeV1::UnsupportedParameter,
        )
        .into_response();
    }
    let messages = match template_messages(request.messages()) {
        Ok(messages) => messages,
        Err(error) => return error.into_response(),
    };
    let options = render_options(request.add_generation_prompt(), request.thinking());
    let result = match model.apply_template_utility(&messages, options) {
        Ok(result) => result,
        Err(error) => return utility_error("model", error.to_string()),
    };
    axum::Json(serde_json::json!({
        "version": result.version(),
        "prompt": result.rendered(),
        "tokens": result.token_ids().as_slice(),
        "count": result.count(),
        "template": {
            "kind": result.identity().kind(),
            "version": result.identity().version(),
            "digest": result.identity().digest(),
            "size_bytes": result.identity().size_bytes(),
            "consistency_label": result.identity().consistency_label(),
        },
        "model": request.model(),
        "model_lock_fingerprint": model.lock_fingerprint(),
    }))
    .into_response()
}

async fn input_tokens(State(state): State<Arc<AppStateV1>>, request: Request<Body>) -> Response {
    let response = match read_phase42_body(request, &state.config).await {
        Err(error) => error.into_response(),
        Ok(body) => match phase42::parse_input_tokens_request(&body) {
            Err(error) => phase42_error(error).into_response(),
            Ok(request) => handle_input_tokens(&state, request),
        },
    };
    record_http(&state, HttpEndpointV1::InputTokens, &response);
    response
}

fn handle_input_tokens(state: &AppStateV1, request: InputTokensRequestV1) -> Response {
    let (model, _lease) = match resolve_model(state, request.model()) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let tokens = match request.input() {
        InputTokensInputV1::Text(text) => {
            match model.tokenize_utility(text, TokenizeOptionsV1::default()) {
                Ok(result) => result.token_ids().as_slice().to_vec(),
                Err(error) => return utility_error("text", error.to_string()),
            }
        }
        InputTokensInputV1::Messages(messages) => {
            if !model.reviewed_chat_template_available() {
                return ApiErrorV1::new(
                    StatusCode::BAD_REQUEST,
                    "reviewed chat template is not supported by this model lock",
                    "invalid_request_error",
                    Some("model".to_owned()),
                    ErrorCodeV1::UnsupportedParameter,
                )
                .into_response();
            }
            let messages = match template_messages(messages) {
                Ok(messages) => messages,
                Err(error) => return error.into_response(),
            };
            let result = match model.apply_template_utility(
                &messages,
                render_options(request.add_generation_prompt(), request.thinking()),
            ) {
                Ok(result) => result,
                Err(error) => return utility_error("model", error.to_string()),
            };
            result.token_ids().as_slice().to_vec()
        }
    };
    axum::Json(serde_json::json!({
        "count": tokens.len(),
        "tokens": tokens,
        "model": request.model(),
        "model_lock_fingerprint": model.lock_fingerprint(),
    }))
    .into_response()
}

async fn create_infill(State(state): State<Arc<AppStateV1>>, request: Request<Body>) -> Response {
    let response = match read_phase42_body(request, &state.config).await {
        Err(error) => error.into_response(),
        Ok(body) => match phase42::parse_infill_request(&body) {
            Err(error) => phase42_error(error).into_response(),
            Ok(request) => handle_infill(&state, request).await,
        },
    };
    record_http(&state, HttpEndpointV1::Infill, &response);
    response
}

async fn handle_infill(state: &AppStateV1, request: InfillRequestV1) -> Response {
    let (model, mut initial_lease) = match resolve_model(state, request.model()) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let Some(capability) = model.infill_capability() else {
        return ApiErrorV1::new(
            StatusCode::BAD_REQUEST,
            "infill is not supported by this model lock",
            "invalid_request_error",
            Some("model".to_owned()),
            ErrorCodeV1::UnsupportedParameter,
        )
        .into_response();
    };
    let tokenize = |text: &str, param: &str| {
        model.tokenize_infill_content(text).map_err(|error| {
            ApiErrorV1::new(
                StatusCode::BAD_REQUEST,
                error.to_string(),
                "invalid_request_error",
                Some(param.to_owned()),
                ErrorCodeV1::InvalidValue,
            )
        })
    };
    let prefix = match tokenize(request.prefix(), "prefix") {
        Ok(tokens) => tokens,
        Err(error) => return error.into_response(),
    };
    let suffix = match tokenize(request.suffix(), "suffix") {
        Ok(tokens) => tokens,
        Err(error) => return error.into_response(),
    };
    let prompt = match request.prompt() {
        Some(text) => match tokenize(text, "prompt") {
            Ok(tokens) => Some(tokens),
            Err(error) => return error.into_response(),
        },
        None => None,
    };
    let rendered = match capability
        .template()
        .render(&prefix, &suffix, prompt.as_deref())
    {
        Ok(tokens) => tokens,
        Err(error) => {
            return ApiErrorV1::invalid_value("prefix", error.to_string()).into_response();
        }
    };
    let required_context = match u64::try_from(rendered.len())
        .ok()
        .and_then(|tokens| tokens.checked_add(u64::from(request.max_tokens())))
    {
        Some(tokens) => tokens,
        None => {
            return ApiErrorV1::invalid_value("max_tokens", "infill context size overflowed")
                .into_response();
        }
    };
    if required_context > u64::from(capability.max_context_tokens()) {
        return ApiErrorV1::invalid_value(
            "max_tokens",
            format!(
                "rendered FIM input plus output requires {required_context} tokens, maximum is {}",
                capability.max_context_tokens()
            ),
        )
        .into_response();
    }
    let base = match ChatCompletionRequestV1::from_infill(
        &request,
        rendered.as_slice().to_vec(),
        capability.template().digest().to_owned(),
    ) {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    let mut receivers = Vec::new();
    for index in 0..request.n() {
        let choice = match base.for_choice(index) {
            Ok(request) => request,
            Err(error) => return error.into_response(),
        };
        let (choice_model, lease) = if index == 0 {
            (Arc::clone(&model), initial_lease.take())
        } else {
            match state.lifecycle.as_ref() {
                Some(_) => match resolve_model(state, request.model()) {
                    Ok((choice_model, lease)) => (choice_model, lease),
                    Err(response) => return *response,
                },
                None => (Arc::clone(&model), None),
            }
        };
        match state
            .scheduler
            .submit_with_lease(choice_model, choice, lease)
        {
            Ok(receiver) => receivers.push(IndexedTextGenerationReceiverV1 {
                index,
                prompt_index: 0,
                expected_prompt_tokens: Some(
                    u64::try_from(rendered.len()).expect("bounded FIM token count fits u64"),
                ),
                receiver,
            }),
            Err(error) => {
                drop(receivers);
                return error.into_response();
            }
        }
    }
    let context = TextResponseContextV1::new(model.alias());
    let metrics = state
        .config
        .metrics
        .as_ref()
        .and_then(|metrics| metrics.admit(model.alias(), request.stream()));
    if request.stream() {
        stream_text_completion(receivers, context, metrics).into_response()
    } else {
        non_stream_text_completion(receivers, context, metrics).await
    }
}

fn template_messages(
    messages: &[TemplateMessageV1],
) -> Result<Vec<Qwen35ChatMessageV1>, ApiErrorV1> {
    messages
        .iter()
        .map(|message| match message.role() {
            TemplateRoleV1::System => Ok(Qwen35ChatMessageV1::system(message.content())),
            TemplateRoleV1::User => Ok(Qwen35ChatMessageV1::user(message.content())),
            TemplateRoleV1::Assistant => {
                Ok(Qwen35ChatMessageV1::assistant(message.content(), None))
            }
        })
        .collect()
}

fn render_options(add_generation_prompt: bool, thinking: bool) -> Qwen35RenderOptionsV1 {
    Qwen35RenderOptionsV1 {
        add_generation_prompt,
        thinking: if thinking {
            ThinkingModeV1::Enabled
        } else {
            ThinkingModeV1::Disabled
        },
    }
}

fn utility_error(param: &str, message: String) -> Response {
    let code = if message.contains("not supported") || message.contains("unavailable") {
        ErrorCodeV1::UnsupportedParameter
    } else {
        ErrorCodeV1::InvalidValue
    };
    ApiErrorV1::new(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        Some(param.to_owned()),
        code,
    )
    .into_response()
}

pub(crate) fn resolve_model(
    state: &AppStateV1,
    alias: &str,
) -> Result<(Arc<ModelRegistryEntryV1>, Option<ModelLifecycleLeaseV1>), Box<Response>> {
    resolve_model_for_request(state, alias).map_err(|error| Box::new(error.into_response()))
}

pub(crate) fn resolve_model_for_request(
    state: &AppStateV1,
    alias: &str,
) -> Result<(Arc<ModelRegistryEntryV1>, Option<ModelLifecycleLeaseV1>), ApiErrorV1> {
    let Some(lifecycle) = state.lifecycle.as_ref() else {
        return state
            .registry
            .get(alias)
            .map(|model| (model, None))
            .ok_or_else(|| ApiErrorV1::model_not_found(alias));
    };
    match lifecycle.resolve(alias) {
        Ok(lease) => {
            let model = lease.owner();
            Ok((model, Some(lease)))
        }
        Err(error) => Err(lifecycle_api_error(alias, error)),
    }
}

pub(crate) fn lifecycle_error_response(alias: &str, error: ModelLifecycleErrorV1) -> Response {
    lifecycle_api_error(alias, error).into_response()
}

pub(crate) fn lifecycle_api_error(alias: &str, error: ModelLifecycleErrorV1) -> ApiErrorV1 {
    match error {
        ModelLifecycleErrorV1::AliasNotFound => ApiErrorV1::model_not_found(alias),
        ModelLifecycleErrorV1::ModelLoading
        | ModelLifecycleErrorV1::ModelDraining
        | ModelLifecycleErrorV1::Quarantined
        | ModelLifecycleErrorV1::LoadingTimeout
        | ModelLifecycleErrorV1::DrainTimeout
        | ModelLifecycleErrorV1::CapacityExceeded
        | ModelLifecycleErrorV1::QuotaExceeded
        | ModelLifecycleErrorV1::ShutdownFailed
        | ModelLifecycleErrorV1::StaleCompletion
        | ModelLifecycleErrorV1::LoaderFailed
        | ModelLifecycleErrorV1::QuarantineNeedsClear
        | ModelLifecycleErrorV1::AliasBusy => ApiErrorV1::new(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("model {alias} is temporarily unavailable"),
            "server_error",
            Some("model".to_owned()),
            ErrorCodeV1::GenerationFailed,
        ),
        _ => ApiErrorV1::generation_failed("dynamic model lifecycle operation failed"),
    }
}

#[derive(Clone, Debug)]
struct TextResponseContextV1 {
    id: String,
    created: u64,
    model: String,
}

impl TextResponseContextV1 {
    fn new(model: &str) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            id: format!("cmpl-sllm-{created:016x}{counter:016x}"),
            created,
            model: model.to_owned(),
        }
    }
}

#[derive(Serialize)]
struct TextCompletionResponseV1 {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<TextCompletionChoiceV1>,
    usage: TokenUsageV1,
}

#[derive(Serialize)]
struct TextCompletionChoiceV1 {
    text: String,
    index: u32,
    logprobs: Option<CompletionLogprobsV1>,
    finish_reason: FinishReasonV1,
}

async fn non_stream_text_completion(
    receivers: Vec<IndexedTextGenerationReceiverV1>,
    context: TextResponseContextV1,
    mut metrics: Option<MetricsRequestHandleV1>,
) -> Response {
    let mut choices = Vec::with_capacity(receivers.len());
    let mut usage = TextUsageAccumulatorV1::default();
    for IndexedTextGenerationReceiverV1 {
        index,
        prompt_index,
        expected_prompt_tokens,
        mut receiver,
    } in receivers
    {
        let mut text = String::new();
        let mut logprobs = None;
        let mut completed = false;
        while let Some(event) = receiver.recv().await {
            match event {
                SchedulerEventV1::Delta(delta) => {
                    text.push_str(&delta);
                    if let Some(metrics) = &mut metrics {
                        metrics.observe_ttft_since_start();
                    }
                }
                SchedulerEventV1::Logprobs(values) => {
                    logprobs = match CompletionLogprobsV1::from_backend(values) {
                        Ok(values) => Some(values),
                        Err(error) => return error.into_response(),
                    };
                }
                SchedulerEventV1::Finished(completion) => {
                    if usage
                        .merge(prompt_index, expected_prompt_tokens, completion.usage)
                        .is_err()
                    {
                        if let Some(metrics) = &mut metrics {
                            metrics.finish(RequestOutcomeV1::Error);
                        }
                        return ApiErrorV1::generation_failed(
                            "inconsistent completion token usage",
                        )
                        .into_response();
                    }
                    choices.push(TextCompletionChoiceV1 {
                        text,
                        index,
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
    let usage = match usage.finish() {
        Ok(usage) => usage,
        Err(error) => return error.into_response(),
    };
    if let Some(metrics) = &mut metrics {
        metrics.record_tokens(usage.prompt_tokens, usage.completion_tokens);
        metrics.finish(RequestOutcomeV1::Success);
    }
    axum::Json(TextCompletionResponseV1 {
        id: context.id,
        object: "text_completion",
        created: context.created,
        model: context.model,
        choices,
        usage,
    })
    .into_response()
}

struct TextStreamStateV1 {
    current: Option<IndexedTextGenerationReceiverV1>,
    receivers: VecDeque<IndexedTextGenerationReceiverV1>,
    context: TextResponseContextV1,
    metrics: Option<MetricsRequestHandleV1>,
    usage: TextUsageAccumulatorV1,
    logprobs: Option<CompletionLogprobsV1>,
    done: bool,
}

#[derive(Serialize)]
struct TextStreamChunkV1<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: [TextStreamChoiceV1<'a>; 1],
}

#[derive(Serialize)]
struct TextStreamChoiceV1<'a> {
    text: &'a str,
    index: u32,
    logprobs: Option<CompletionLogprobsV1>,
    finish_reason: Option<FinishReasonV1>,
}

fn stream_text_completion(
    receivers: Vec<IndexedTextGenerationReceiverV1>,
    context: TextResponseContextV1,
    metrics: Option<MetricsRequestHandleV1>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut receivers = VecDeque::from(receivers);
    let state = TextStreamStateV1 {
        current: receivers.pop_front(),
        receivers,
        context,
        metrics,
        usage: TextUsageAccumulatorV1::default(),
        logprobs: None,
        done: false,
    };
    Sse::new(stream::unfold(state, |mut state| async move {
        loop {
            if state.done {
                return None;
            }
            let Some(current) = state.current.as_mut() else {
                state.done = true;
                if let Some(metrics) = &mut state.metrics {
                    metrics.finish(RequestOutcomeV1::Success);
                }
                return Some((Ok(Event::default().data("[DONE]")), state));
            };
            let index = current.index;
            let prompt_index = current.prompt_index;
            let expected_prompt_tokens = current.expected_prompt_tokens;
            match current.receiver.recv().await {
                Some(SchedulerEventV1::Delta(delta)) => {
                    if let Some(metrics) = &mut state.metrics {
                        metrics.observe_ttft_since_start();
                    }
                    let chunk = TextStreamChunkV1 {
                        id: &state.context.id,
                        object: "text_completion",
                        created: state.context.created,
                        model: &state.context.model,
                        choices: [TextStreamChoiceV1 {
                            text: &delta,
                            index,
                            logprobs: None,
                            finish_reason: None,
                        }],
                    };
                    return Some((Ok(json_event(&chunk)), state));
                }
                Some(SchedulerEventV1::Logprobs(values)) => {
                    state.logprobs = match CompletionLogprobsV1::from_backend(values) {
                        Ok(values) => Some(values),
                        Err(error) => {
                            state.done = true;
                            if let Some(metrics) = &mut state.metrics {
                                metrics.finish(RequestOutcomeV1::Error);
                            }
                            return Some((Ok(json_event(&error.envelope())), state));
                        }
                    };
                    continue;
                }
                Some(SchedulerEventV1::Finished(completion)) => {
                    if state
                        .usage
                        .merge(prompt_index, expected_prompt_tokens, completion.usage)
                        .is_err()
                    {
                        state.done = true;
                        if let Some(metrics) = &mut state.metrics {
                            metrics.finish(RequestOutcomeV1::Error);
                        }
                        let error =
                            ApiErrorV1::generation_failed("inconsistent completion token usage");
                        return Some((Ok(json_event(&error.envelope())), state));
                    }
                    let chunk = TextStreamChunkV1 {
                        id: &state.context.id,
                        object: "text_completion",
                        created: state.context.created,
                        model: &state.context.model,
                        choices: [TextStreamChoiceV1 {
                            text: "",
                            index,
                            logprobs: state.logprobs.take(),
                            finish_reason: Some(completion.finish_reason),
                        }],
                    };
                    if state.receivers.is_empty() {
                        if let Some(metrics) = &mut state.metrics {
                            match state.usage.finish() {
                                Ok(usage) => metrics
                                    .record_tokens(usage.prompt_tokens, usage.completion_tokens),
                                Err(error) => {
                                    state.done = true;
                                    metrics.finish(RequestOutcomeV1::Error);
                                    return Some((Ok(json_event(&error.envelope())), state));
                                }
                            }
                        }
                        state.current = None;
                    } else {
                        state.current = state.receivers.pop_front();
                    }
                    return Some((Ok(json_event(&chunk)), state));
                }
                Some(SchedulerEventV1::Failed(error)) => {
                    state.done = true;
                    if let Some(metrics) = &mut state.metrics {
                        metrics.finish(RequestOutcomeV1::Error);
                    }
                    return Some((Ok(json_event(&error.envelope())), state));
                }
                None => {
                    state.done = true;
                    if let Some(metrics) = &mut state.metrics {
                        metrics.finish(RequestOutcomeV1::Error);
                    }
                    let error =
                        ApiErrorV1::generation_failed("generation ended without a terminal event");
                    return Some((Ok(json_event(&error.envelope())), state));
                }
            }
        }
    }))
}

struct IndexedTextGenerationReceiverV1 {
    index: u32,
    prompt_index: u32,
    expected_prompt_tokens: Option<u64>,
    receiver: GenerationReceiverV1,
}

/// Legacy Completions logprob envelope from the pinned OpenAI OpenAPI 2.3.0
/// schema. Chat Completions deliberately retains its distinct token-object
/// representation below.
#[derive(Clone, Debug, Serialize)]
struct CompletionLogprobsV1 {
    text_offset: Vec<u64>,
    token_logprobs: Vec<f64>,
    tokens: Vec<String>,
    top_logprobs: Vec<BTreeMap<String, f64>>,
}

impl CompletionLogprobsV1 {
    fn from_backend(values: Vec<BackendTokenLogprobV1>) -> Result<Self, ApiErrorV1> {
        let mut text_offset = Vec::with_capacity(values.len());
        let mut token_logprobs = Vec::with_capacity(values.len());
        let mut tokens = Vec::with_capacity(values.len());
        let mut top_logprobs = Vec::with_capacity(values.len());
        let mut offset = 0_u64;
        for value in values {
            if !value.logprob.is_finite()
                || value
                    .top_logprobs
                    .iter()
                    .any(|candidate| !candidate.logprob.is_finite())
            {
                return Err(ApiErrorV1::generation_failed(
                    "completion logprob output contained a non-finite value",
                ));
            }
            text_offset.push(offset);
            let token_chars = u64::try_from(value.token.chars().count()).map_err(|_| {
                ApiErrorV1::generation_failed("completion logprob text offset overflowed")
            })?;
            offset = offset.checked_add(token_chars).ok_or_else(|| {
                ApiErrorV1::generation_failed("completion logprob text offset overflowed")
            })?;
            token_logprobs.push(value.logprob);
            tokens.push(value.token);
            let mut candidates = BTreeMap::new();
            for candidate in value.top_logprobs {
                if candidates
                    .insert(candidate.token, candidate.logprob)
                    .is_some()
                {
                    return Err(ApiErrorV1::generation_failed(
                        "completion top_logprobs contained duplicate token text",
                    ));
                }
            }
            top_logprobs.push(candidates);
        }
        Ok(Self {
            text_offset,
            token_logprobs,
            tokens,
            top_logprobs,
        })
    }
}

#[derive(Default)]
struct TextUsageAccumulatorV1 {
    prompt_tokens: BTreeMap<u32, u64>,
    completion_tokens: u64,
}

impl TextUsageAccumulatorV1 {
    fn merge(
        &mut self,
        prompt_index: u32,
        expected_prompt_tokens: Option<u64>,
        usage: TokenUsageV1,
    ) -> Result<(), ApiErrorV1> {
        if expected_prompt_tokens.is_some_and(|expected| expected != usage.prompt_tokens) {
            return Err(ApiErrorV1::generation_failed(
                "completion prompt token usage differed from admitted input",
            ));
        }
        match self.prompt_tokens.get(&prompt_index) {
            Some(&tokens) if tokens != usage.prompt_tokens => {
                return Err(ApiErrorV1::generation_failed(
                    "choice prompt token accounting is inconsistent",
                ));
            }
            Some(_) => {}
            None => {
                self.prompt_tokens.insert(prompt_index, usage.prompt_tokens);
            }
        }
        self.completion_tokens = self
            .completion_tokens
            .checked_add(usage.completion_tokens)
            .ok_or_else(|| ApiErrorV1::generation_failed("completion token usage overflowed"))?;
        Ok(())
    }

    fn finish(&self) -> Result<TokenUsageV1, ApiErrorV1> {
        let prompt_tokens = self
            .prompt_tokens
            .values()
            .try_fold(0_u64, |total, value| {
                total
                    .checked_add(*value)
                    .ok_or_else(|| ApiErrorV1::generation_failed("prompt token usage overflowed"))
            })?;
        TokenUsageV1::new(prompt_tokens, self.completion_tokens)
    }
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
            (config.loopback_admin && config.credentials.is_open() && first.is_none())
                || config.credentials.authorize_admin(first)
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
struct ModelListV1 {
    object: &'static str,
    data: Vec<ModelObjectV1>,
}

#[derive(Serialize)]
struct ModelObjectV1 {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
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

#[cfg(test)]
mod tests {
    use super::{ServerConfigV1, authorize_admin};
    use axum::http::HeaderMap;

    #[test]
    fn credential_free_admin_can_only_be_enabled_for_loopback() {
        let loopback = ServerConfigV1::default()
            .with_loopback_admin("127.0.0.1:8080".parse().unwrap())
            .unwrap();
        assert!(authorize_admin(&HeaderMap::new(), &loopback).is_ok());

        let remote = ServerConfigV1::default().with_loopback_admin("0.0.0.0:8080".parse().unwrap());
        assert!(remote.is_err());
    }
}
