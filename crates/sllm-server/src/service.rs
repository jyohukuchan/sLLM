use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::stream::{self, Stream};
use serde::Serialize;

use crate::api::{
    ApiErrorV1, ErrorCodeV1, FinishReasonV1, MAX_REQUEST_BODY_BYTES, TokenUsageV1,
    parse_chat_completion_request,
};
use crate::runtime::{GenerationReceiverV1, ModelRegistryV1, SchedulerEventV1, SchedulerV1};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Default)]
pub struct ServerConfigV1 {
    bearer_token: Option<String>,
}

impl ServerConfigV1 {
    pub fn new(bearer_token: Option<String>) -> Result<Self, ApiErrorV1> {
        if bearer_token.as_ref().is_some_and(|token| {
            token.is_empty()
                || token
                    .bytes()
                    .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace())
        }) {
            return Err(ApiErrorV1::generation_failed(
                "bearer token configuration is invalid",
            ));
        }
        Ok(Self { bearer_token })
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
    let state = Arc::new(AppStateV1 {
        registry,
        scheduler,
        config,
    });
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(create_chat_completion))
        .with_state(state)
}

async fn list_models(State(state): State<Arc<AppStateV1>>, headers: HeaderMap) -> Response {
    if let Err(error) = authorize(&headers, &state.config) {
        return error.into_response();
    }
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
}

async fn create_chat_completion(
    State(state): State<Arc<AppStateV1>>,
    request: Request<Body>,
) -> Response {
    if let Err(error) = authorize(request.headers(), &state.config) {
        return error.into_response();
    }
    if let Err(error) = validate_content_type(request.headers()) {
        return error.into_response();
    }
    let body = match to_bytes(request.into_body(), MAX_REQUEST_BODY_BYTES).await {
        Ok(body) => body,
        Err(_) => {
            return ApiErrorV1::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body exceeds 1048576 bytes",
                "invalid_request_error",
                None,
                ErrorCodeV1::RequestTooLarge,
            )
            .into_response();
        }
    };
    let request = match parse_chat_completion_request(&body) {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    let model = match state.registry.get(request.model()) {
        Some(model) => model,
        None => return ApiErrorV1::model_not_found(request.model()).into_response(),
    };
    let context = ResponseContextV1::new(model.alias());
    let stream_response = request.stream();
    let receiver = match state.scheduler.submit(model, request) {
        Ok(receiver) => receiver,
        Err(error) => return error.into_response(),
    };
    if stream_response {
        stream_chat_completion(receiver, context).into_response()
    } else {
        non_stream_chat_completion(receiver, context).await
    }
}

fn authorize(headers: &HeaderMap, config: &ServerConfigV1) -> Result<(), ApiErrorV1> {
    let Some(expected) = config.bearer_token.as_deref() else {
        return Ok(());
    };
    let accepted = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value == format!("Bearer {expected}"));
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

async fn non_stream_chat_completion(
    mut receiver: GenerationReceiverV1,
    context: ResponseContextV1,
) -> Response {
    let mut text = String::new();
    while let Some(event) = receiver.recv().await {
        match event {
            SchedulerEventV1::Delta(delta) => text.push_str(&delta),
            SchedulerEventV1::Finished(completion) => {
                return axum::Json(ChatCompletionResponseV1 {
                    id: &context.id,
                    object: "chat.completion",
                    created: context.created,
                    model: &context.model,
                    choices: [ChatCompletionChoiceV1 {
                        index: 0,
                        message: AssistantMessageV1 {
                            role: "assistant",
                            content: &text,
                        },
                        logprobs: None,
                        finish_reason: completion.finish_reason,
                    }],
                    usage: completion.usage,
                })
                .into_response();
            }
            SchedulerEventV1::Failed(error) => return error.into_response(),
        }
    }
    ApiErrorV1::generation_failed("generation ended without a terminal event").into_response()
}

fn stream_chat_completion(
    receiver: GenerationReceiverV1,
    context: ResponseContextV1,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state = StreamStateV1 {
        receiver,
        context,
        role_pending: true,
        queued: VecDeque::new(),
        terminal: false,
    };
    Sse::new(stream::unfold(state, |mut state| async move {
        if let Some(event) = state.queued.pop_front() {
            return Some((Ok(event), state));
        }
        if state.terminal {
            return None;
        }
        if state.role_pending {
            state.role_pending = false;
            let chunk = StreamChunkV1::role(&state.context);
            return Some((Ok(json_event(&chunk)), state));
        }
        match state.receiver.recv().await {
            Some(SchedulerEventV1::Delta(delta)) => {
                let chunk = StreamChunkV1::content(&state.context, &delta);
                Some((Ok(json_event(&chunk)), state))
            }
            Some(SchedulerEventV1::Finished(completion)) => {
                let chunk = StreamChunkV1::finished(
                    &state.context,
                    completion.finish_reason,
                    completion.usage,
                );
                state.queued.push_back(Event::default().data("[DONE]"));
                state.terminal = true;
                Some((Ok(json_event(&chunk)), state))
            }
            Some(SchedulerEventV1::Failed(error)) => {
                state.terminal = true;
                Some((Ok(json_event(&error.envelope())), state))
            }
            None => {
                state.terminal = true;
                let error =
                    ApiErrorV1::generation_failed("generation ended without a terminal event");
                Some((Ok(json_event(&error.envelope())), state))
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

struct StreamStateV1 {
    receiver: GenerationReceiverV1,
    context: ResponseContextV1,
    role_pending: bool,
    queued: VecDeque<Event>,
    terminal: bool,
}

#[derive(Clone, Debug)]
struct ResponseContextV1 {
    id: String,
    created: u64,
    model: String,
}

impl ResponseContextV1 {
    fn new(model: &str) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            id: format!("chatcmpl-sllm-{created:016x}{counter:016x}"),
            created,
            model: model.to_owned(),
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
struct ChatCompletionResponseV1<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: [ChatCompletionChoiceV1<'a>; 1],
    usage: TokenUsageV1,
}

#[derive(Serialize)]
struct ChatCompletionChoiceV1<'a> {
    index: u32,
    message: AssistantMessageV1<'a>,
    logprobs: Option<()>,
    finish_reason: FinishReasonV1,
}

#[derive(Serialize)]
struct AssistantMessageV1<'a> {
    role: &'static str,
    content: &'a str,
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
    fn role(context: &'a ResponseContextV1) -> Self {
        Self::new(
            context,
            StreamDeltaV1 {
                role: Some("assistant"),
                content: Some(""),
            },
            None,
            None,
        )
    }

    fn content(context: &'a ResponseContextV1, content: &'a str) -> Self {
        Self::new(
            context,
            StreamDeltaV1 {
                role: None,
                content: Some(content),
            },
            None,
            None,
        )
    }

    fn finished(
        context: &'a ResponseContextV1,
        finish_reason: FinishReasonV1,
        usage: TokenUsageV1,
    ) -> Self {
        Self::new(
            context,
            StreamDeltaV1 {
                role: None,
                content: None,
            },
            Some(finish_reason),
            Some(usage),
        )
    }

    fn new(
        context: &'a ResponseContextV1,
        delta: StreamDeltaV1<'a>,
        finish_reason: Option<FinishReasonV1>,
        usage: Option<TokenUsageV1>,
    ) -> Self {
        Self {
            id: &context.id,
            object: "chat.completion.chunk",
            created: context.created,
            model: &context.model,
            choices: [StreamChoiceV1 {
                index: 0,
                delta,
                logprobs: None,
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
    logprobs: Option<()>,
    finish_reason: Option<FinishReasonV1>,
}

#[derive(Serialize)]
struct StreamDeltaV1<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}
