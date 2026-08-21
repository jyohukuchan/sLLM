//! HTTP/runtime integration for the Phase 43 protocol profiles.
//!
//! This module only lowers validated protocol data into the existing bounded
//! generation scheduler. Generated calls are serialized back to the client;
//! no call is executed or resolved by the server.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{Body, to_bytes};
use axum::extract::{Path, State};
use axum::http::header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, HeaderName};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sllm_core::CompiledGrammar;
use sllm_frontend::{
    CanonicalGenerationEnvelopeV1, Qwen35ChatMessageV1, ToolCallPolicyV1,
    ToolCallV1 as FrontendToolCallV1, ToolChoiceV1 as FrontendToolChoiceV1,
    ToolDefinitionV1 as FrontendToolDefinitionV1, ToolMessageRoleV1, ToolProtocolItemV1,
    ToolProtocolV1, ToolResultV1,
};

use crate::api::{
    ApiErrorV1, ChatCompletionRequestV1, ErrorCodeV1, FinishReasonV1, MAX_REQUEST_BODY_BYTES,
};
use crate::metrics::{HttpEndpointV1, MetricsRequestHandleV1, RequestOutcomeV1};
use crate::phase43_api::{
    AnthropicContentBlockV1, AnthropicMessagesRequestV1, AnthropicRoleV1, AnthropicSystemV1,
    Phase43ApiErrorV1, Phase43ErrorCodeV1, ResponsesInputItemV1, ResponsesInputV1,
    ResponsesMessageRoleV1, ResponsesRequestV1, ToolChoiceV1 as WireToolChoiceV1,
    parse_anthropic_request_v1, parse_responses_request_v1,
};
use crate::phase43_transport::{
    AnthropicStreamBuilderV1, Phase43CompletedOutputV1, Phase43FinishReasonV1, Phase43SseEventV1,
    Phase43ToolCallV1, Phase43UsageV1, ResponsesStreamBuilderV1, anthropic_non_stream_v1,
    responses_non_stream_v1,
};
use crate::resume::{ReplayErrorV1, ResumableStoreV1};
use crate::runtime::{GenerationReceiverV1, SchedulerEventV1};
use crate::service::AppStateV1;

const ANTHROPIC_VERSION_HEADER: HeaderName = HeaderName::from_static("anthropic-version");
const LAST_EVENT_ID: HeaderName = HeaderName::from_static("last-event-id");
const MAX_PROTOCOL_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESUMABLE_PROTOCOL_TOKENS: u32 = 40;

static PHASE43_REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProtocolProfileV1 {
    Responses,
    Anthropic,
}

#[derive(Clone, Debug)]
struct ProtocolContextV1 {
    profile: ProtocolProfileV1,
    id: String,
    item_id: String,
    reasoning_item_id: String,
    created_at: u64,
    model: String,
    reasoning: bool,
}

impl ProtocolContextV1 {
    fn new(profile: ProtocolProfileV1, model: &str, reasoning: bool) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let counter = PHASE43_REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let suffix = format!("{created_at:016x}{counter:016x}");
        let id = match profile {
            ProtocolProfileV1::Responses => format!("resp_{suffix}"),
            ProtocolProfileV1::Anthropic => format!("msg_{suffix}"),
        };
        Self {
            profile,
            id,
            item_id: format!("msg_{suffix}"),
            reasoning_item_id: format!("rs_{suffix}"),
            created_at,
            model: model.to_owned(),
            reasoning,
        }
    }

    fn tool_call(&self, index: usize, name: &str, arguments: &Value) -> Phase43ToolCallV1 {
        let suffix = self
            .id
            .split_once('_')
            .map_or(self.id.as_str(), |(_, suffix)| suffix);
        Phase43ToolCallV1::new(
            format!("fc_{suffix}_{index}"),
            format!("call_{suffix}_{index}"),
            name,
            arguments.to_string(),
        )
    }
}

#[derive(Clone)]
struct ToolDecodeContextV1 {
    protocol: ToolProtocolV1,
    choice: FrontendToolChoiceV1,
    policy: ToolCallPolicyV1,
    include_reasoning: bool,
}

struct PreparedProtocolV1 {
    request: ChatCompletionRequestV1,
    decoder: Option<ToolDecodeContextV1>,
    context: ProtocolContextV1,
    stream: bool,
    resumable: bool,
}

pub(crate) async fn create_response(
    State(state): State<Arc<AppStateV1>>,
    request: Request<Body>,
) -> Response {
    let endpoint = HttpEndpointV1::Responses;
    let response = match take_request_body(request, &state).await {
        Ok(body) => match parse_responses_request_v1(&body).map_err(convert_phase43_error) {
            Ok(request) => match prepare_responses(request, &state) {
                Ok(prepared) => execute_protocol(prepared, &state).await,
                Err(error) => protocol_error_response(ProtocolProfileV1::Responses, &error),
            },
            Err(error) => protocol_error_response(ProtocolProfileV1::Responses, &error),
        },
        Err(error) => protocol_error_response(ProtocolProfileV1::Responses, &error),
    };
    record_http(&state, endpoint, &response);
    response
}

pub(crate) async fn create_anthropic_message(
    State(state): State<Arc<AppStateV1>>,
    request: Request<Body>,
) -> Response {
    let endpoint = HttpEndpointV1::AnthropicMessages;
    let version = unique_header(request.headers(), &ANTHROPIC_VERSION_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let response = match take_request_body(request, &state).await {
        Ok(body) => match parse_anthropic_request_v1(&body, version.as_deref())
            .map_err(convert_phase43_error)
        {
            Ok(request) => match prepare_anthropic(request, &state) {
                Ok(prepared) => execute_protocol(prepared, &state).await,
                Err(error) => anthropic_error_response(&error),
            },
            Err(error) => anthropic_error_response(&error),
        },
        Err(error) => anthropic_error_response(&error),
    };
    record_http(&state, endpoint, &response);
    response
}

async fn take_request_body(
    request: Request<Body>,
    state: &AppStateV1,
) -> Result<Vec<u8>, ApiErrorV1> {
    authorize_user(request.headers(), state)?;
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

fn prepare_responses(
    request: ResponsesRequestV1,
    state: &AppStateV1,
) -> Result<PreparedProtocolV1, ApiErrorV1> {
    validate_resumable_budget(
        request.sllm().resumable(),
        request.max_output_tokens(),
        "max_output_tokens",
    )?;
    let model = state
        .registry
        .get(request.model())
        .ok_or_else(|| ApiErrorV1::model_not_found(request.model()))?;
    let reasoning = request.reasoning_effort().is_some();
    let (history, simple_messages, assistant_prefill, has_tool_history) =
        lower_responses_history(&request)?;
    let tools = lower_tools(request.tools())?;
    let uses_tool_protocol = !tools.is_empty() || has_tool_history;
    if uses_tool_protocol && !model.tool_protocol_v1_available() {
        return Err(ApiErrorV1::unsupported("tools"));
    }
    let (generation_request, decoder) = if uses_tool_protocol {
        if assistant_prefill.is_some() {
            return Err(ApiErrorV1::unsupported(
                "input.assistant_prefill_with_tools",
            ));
        }
        let protocol = ToolProtocolV1::new(tools).map_err(tool_error)?;
        protocol.validate_history(&history).map_err(tool_error)?;
        let choice = lower_tool_choice(request.tool_choice());
        let policy = if request.parallel_tool_calls() {
            ToolCallPolicyV1::parallel()
        } else {
            ToolCallPolicyV1::sequential()
        };
        let schema = protocol
            .generation_schema_with_reasoning(&choice, policy, reasoning)
            .map_err(tool_error)?;
        // Compile before scheduler/GPU admission. Production compiles the same
        // immutable value again at backend preparation as a defensive check.
        CompiledGrammar::from_json_schema(&schema)
            .map_err(|_| ApiErrorV1::invalid_value("tools", "tool schema is unsupported"))?;
        let prompt = protocol
            .render_qwen_tool_prompt(&history, &choice, policy)
            .map_err(tool_error)?;
        let generation_request = ChatCompletionRequestV1::from_protocol_text(
            request.model().to_owned(),
            prompt,
            None,
            request.max_output_tokens(),
            request.temperature().unwrap_or(1.0),
            request.top_p().unwrap_or(1.0),
            Vec::new(),
            request.stream(),
            request.sllm().resumable(),
            reasoning,
            Some(schema),
        )?;
        (
            generation_request,
            Some(ToolDecodeContextV1 {
                protocol,
                choice,
                policy,
                include_reasoning: reasoning,
            }),
        )
    } else {
        let generation_request = ChatCompletionRequestV1::from_protocol_messages(
            request.model().to_owned(),
            simple_messages,
            assistant_prefill,
            request.max_output_tokens(),
            request.temperature().unwrap_or(1.0),
            request.top_p().unwrap_or(1.0),
            Vec::new(),
            request.stream(),
            request.sllm().resumable(),
            reasoning,
        )?;
        (generation_request, None)
    };
    Ok(PreparedProtocolV1 {
        context: ProtocolContextV1::new(ProtocolProfileV1::Responses, request.model(), reasoning),
        request: generation_request,
        decoder,
        stream: request.stream(),
        resumable: request.sllm().resumable(),
    })
}

fn prepare_anthropic(
    request: AnthropicMessagesRequestV1,
    state: &AppStateV1,
) -> Result<PreparedProtocolV1, ApiErrorV1> {
    validate_resumable_budget(
        request.sllm().resumable(),
        request.max_tokens(),
        "max_tokens",
    )?;
    let model = state
        .registry
        .get(request.model())
        .ok_or_else(|| ApiErrorV1::model_not_found(request.model()))?;
    let (history, simple_messages, has_tool_history) = lower_anthropic_history(&request)?;
    let tools = lower_tools(request.tools())?;
    let uses_tool_protocol = !tools.is_empty() || has_tool_history;
    if uses_tool_protocol && !model.tool_protocol_v1_available() {
        return Err(ApiErrorV1::unsupported("tools"));
    }
    let (generation_request, decoder) = if uses_tool_protocol {
        let protocol = ToolProtocolV1::new(tools).map_err(tool_error)?;
        protocol.validate_history(&history).map_err(tool_error)?;
        let choice = lower_tool_choice(request.tool_choice());
        let policy = if request.tool_choice().allows_parallel() {
            ToolCallPolicyV1::parallel()
        } else {
            ToolCallPolicyV1::sequential()
        };
        let schema = protocol
            .generation_schema_with_reasoning(&choice, policy, false)
            .map_err(tool_error)?;
        CompiledGrammar::from_json_schema(&schema)
            .map_err(|_| ApiErrorV1::invalid_value("tools", "tool schema is unsupported"))?;
        let prompt = protocol
            .render_qwen_tool_prompt(&history, &choice, policy)
            .map_err(tool_error)?;
        let generation_request = ChatCompletionRequestV1::from_protocol_text(
            request.model().to_owned(),
            prompt,
            None,
            request.max_tokens(),
            1.0,
            1.0,
            request.stop_sequences().to_vec(),
            request.stream(),
            request.sllm().resumable(),
            false,
            Some(schema),
        )?;
        (
            generation_request,
            Some(ToolDecodeContextV1 {
                protocol,
                choice,
                policy,
                include_reasoning: false,
            }),
        )
    } else {
        let generation_request = ChatCompletionRequestV1::from_protocol_messages(
            request.model().to_owned(),
            simple_messages,
            None,
            request.max_tokens(),
            1.0,
            1.0,
            request.stop_sequences().to_vec(),
            request.stream(),
            request.sllm().resumable(),
            false,
        )?;
        (generation_request, None)
    };
    Ok(PreparedProtocolV1 {
        context: ProtocolContextV1::new(ProtocolProfileV1::Anthropic, request.model(), false),
        request: generation_request,
        decoder,
        stream: request.stream(),
        resumable: request.sllm().resumable(),
    })
}

fn lower_tools(
    tools: &[crate::phase43_api::ToolDefinitionV1],
) -> Result<Vec<FrontendToolDefinitionV1>, ApiErrorV1> {
    tools
        .iter()
        .map(|tool| {
            FrontendToolDefinitionV1::new(
                tool.name(),
                tool.description().map(str::to_owned),
                tool.parameters().clone(),
            )
            .map_err(tool_error)
        })
        .collect()
}

fn validate_resumable_budget(
    resumable: bool,
    max_tokens: u32,
    param: &'static str,
) -> Result<(), ApiErrorV1> {
    if resumable && max_tokens > MAX_RESUMABLE_PROTOCOL_TOKENS {
        return Err(ApiErrorV1::invalid_value(
            param,
            "resumable protocol generation is limited to 40 output tokens",
        ));
    }
    Ok(())
}

type LoweredHistoryV1 = (
    Vec<ToolProtocolItemV1>,
    Vec<Qwen35ChatMessageV1>,
    Option<String>,
    bool,
);

fn lower_responses_history(request: &ResponsesRequestV1) -> Result<LoweredHistoryV1, ApiErrorV1> {
    let mut history = Vec::new();
    let mut simple = Vec::<(ToolMessageRoleV1, String)>::new();
    let mut has_tool_history = false;
    if let Some(instructions) = request.instructions() {
        history.push(ToolProtocolItemV1::message(
            ToolMessageRoleV1::System,
            instructions,
        ));
        simple.push((ToolMessageRoleV1::System, instructions.to_owned()));
    }
    match request.input() {
        ResponsesInputV1::Text(text) => {
            history.push(ToolProtocolItemV1::message(ToolMessageRoleV1::User, text));
            simple.push((ToolMessageRoleV1::User, text.clone()));
        }
        ResponsesInputV1::Items(items) => {
            for item in items {
                match item {
                    ResponsesInputItemV1::Message { role, content } => {
                        let role = match role {
                            ResponsesMessageRoleV1::User => ToolMessageRoleV1::User,
                            ResponsesMessageRoleV1::Assistant => ToolMessageRoleV1::Assistant,
                            ResponsesMessageRoleV1::System | ResponsesMessageRoleV1::Developer => {
                                ToolMessageRoleV1::System
                            }
                        };
                        let content = content.iter().map(|part| part.text()).collect::<String>();
                        history.push(ToolProtocolItemV1::message(role, &content));
                        simple.push((role, content));
                    }
                    ResponsesInputItemV1::FunctionCall {
                        call_id,
                        name,
                        arguments,
                    } => {
                        let arguments: Value =
                            serde_json::from_str(arguments).map_err(|error| {
                                ApiErrorV1::invalid_value("input.arguments", error.to_string())
                            })?;
                        history.push(ToolProtocolItemV1::ToolCall(
                            FrontendToolCallV1::new(call_id, name, arguments)
                                .map_err(tool_error)?,
                        ));
                        has_tool_history = true;
                    }
                    ResponsesInputItemV1::FunctionCallOutput { call_id, output } => {
                        history.push(ToolProtocolItemV1::ToolResult(
                            ToolResultV1::new(call_id, Value::String(output.clone()), false)
                                .map_err(tool_error)?,
                        ));
                        has_tool_history = true;
                    }
                }
            }
        }
    }
    let assistant_prefill = simple
        .last()
        .is_some_and(|(role, _)| *role == ToolMessageRoleV1::Assistant)
        .then(|| simple.pop().expect("last entry exists").1);
    let messages = normalize_simple_messages(simple)?;
    Ok((history, messages, assistant_prefill, has_tool_history))
}

type LoweredAnthropicHistoryV1 = (Vec<ToolProtocolItemV1>, Vec<Qwen35ChatMessageV1>, bool);

fn lower_anthropic_history(
    request: &AnthropicMessagesRequestV1,
) -> Result<LoweredAnthropicHistoryV1, ApiErrorV1> {
    let mut history = Vec::new();
    let mut simple = Vec::<(ToolMessageRoleV1, String)>::new();
    let mut has_tool_history = false;
    if let Some(system) = request.system() {
        let system = match system {
            AnthropicSystemV1::Text(text) => text.clone(),
            AnthropicSystemV1::Blocks(blocks) => blocks.join(""),
        };
        history.push(ToolProtocolItemV1::message(
            ToolMessageRoleV1::System,
            &system,
        ));
        simple.push((ToolMessageRoleV1::System, system));
    }
    for message in request.messages() {
        let role = match message.role() {
            AnthropicRoleV1::User => ToolMessageRoleV1::User,
            AnthropicRoleV1::Assistant => ToolMessageRoleV1::Assistant,
        };
        let mut text = String::new();
        for block in message.content() {
            match block {
                AnthropicContentBlockV1::Text(value) => {
                    text.push_str(value);
                    history.push(ToolProtocolItemV1::message(role, value));
                }
                AnthropicContentBlockV1::ToolUse { id, name, input } => {
                    history.push(ToolProtocolItemV1::ToolCall(
                        FrontendToolCallV1::new(id, name, input.clone()).map_err(tool_error)?,
                    ));
                    has_tool_history = true;
                }
                AnthropicContentBlockV1::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    history.push(ToolProtocolItemV1::ToolResult(
                        ToolResultV1::new(tool_use_id, Value::String(content.clone()), *is_error)
                            .map_err(tool_error)?,
                    ));
                    has_tool_history = true;
                }
            }
        }
        if !text.is_empty() {
            simple.push((role, text));
        }
    }
    Ok((
        history,
        normalize_simple_messages(simple)?,
        has_tool_history,
    ))
}

fn normalize_simple_messages(
    messages: Vec<(ToolMessageRoleV1, String)>,
) -> Result<Vec<Qwen35ChatMessageV1>, ApiErrorV1> {
    let mut system = Vec::new();
    let mut ordinary = Vec::new();
    let mut user_seen = false;
    let mut ordinary_started = false;
    for (role, content) in messages {
        match role {
            ToolMessageRoleV1::System => {
                if ordinary_started {
                    return Err(ApiErrorV1::invalid_value(
                        "input",
                        "system and developer messages must precede ordinary messages",
                    ));
                }
                system.push(content);
            }
            ToolMessageRoleV1::User => {
                ordinary_started = true;
                user_seen = true;
                ordinary.push(Qwen35ChatMessageV1::user(content));
            }
            ToolMessageRoleV1::Assistant => {
                ordinary_started = true;
                ordinary.push(Qwen35ChatMessageV1::assistant(content, None));
            }
        }
    }
    if !user_seen {
        return Err(ApiErrorV1::invalid_value(
            "input",
            "at least one user message is required",
        ));
    }
    if !system.is_empty() {
        ordinary.insert(0, Qwen35ChatMessageV1::system(system.join("\n\n")));
    }
    Ok(ordinary)
}

fn lower_tool_choice(choice: &WireToolChoiceV1) -> FrontendToolChoiceV1 {
    match choice {
        WireToolChoiceV1::Auto { .. } => FrontendToolChoiceV1::Auto,
        WireToolChoiceV1::None => FrontendToolChoiceV1::None,
        WireToolChoiceV1::Required { .. } => FrontendToolChoiceV1::Required,
        WireToolChoiceV1::Specific { name, .. } => FrontendToolChoiceV1::named(name.clone()),
    }
}

async fn execute_protocol(prepared: PreparedProtocolV1, state: &AppStateV1) -> Response {
    if prepared.resumable && state.config.replay.is_none() {
        return ApiErrorV1::invalid_value(
            "sllm.resumable",
            "resumable streaming is not enabled on this server",
        )
        .into_response();
    }
    let Some(model) = state.registry.get(prepared.request.model()) else {
        return ApiErrorV1::model_not_found(prepared.request.model()).into_response();
    };
    let receiver = match state.scheduler.submit(model, prepared.request) {
        Ok(receiver) => receiver,
        Err(error) => return protocol_error_response(prepared.context.profile, &error),
    };
    let metrics = state
        .config
        .metrics
        .as_ref()
        .and_then(|metrics| metrics.admit(&prepared.context.model, prepared.stream));
    if prepared.stream && prepared.resumable {
        let replay = state
            .config
            .replay
            .clone()
            .expect("resumable availability checked before admission");
        if replay.create(&prepared.context.id).is_err() {
            drop(receiver);
            return protocol_error_response(prepared.context.profile, &ApiErrorV1::rate_limited());
        }
        spawn_protocol_replay_producer(
            receiver,
            prepared.context.clone(),
            prepared.decoder,
            replay.clone(),
            metrics,
        );
        named_replay_stream(replay, prepared.context.id, 0).into_response()
    } else if prepared.stream {
        protocol_stream(receiver, prepared.context, prepared.decoder, metrics).into_response()
    } else {
        match collect_protocol_output(receiver, &prepared.context, prepared.decoder, metrics).await
        {
            Ok(output) => match prepared.context.profile {
                ProtocolProfileV1::Responses => responses_non_stream_v1(&output)
                    .map(axum::Json)
                    .map(IntoResponse::into_response)
                    .unwrap_or_else(transport_error_response),
                ProtocolProfileV1::Anthropic => anthropic_non_stream_v1(&output)
                    .map(axum::Json)
                    .map(IntoResponse::into_response)
                    .unwrap_or_else(|error| {
                        anthropic_error_response(&ApiErrorV1::generation_failed(error.to_string()))
                    }),
            },
            Err(error) => protocol_error_response(prepared.context.profile, &error),
        }
    }
}

async fn collect_protocol_output(
    mut receiver: GenerationReceiverV1,
    context: &ProtocolContextV1,
    decoder: Option<ToolDecodeContextV1>,
    mut metrics: Option<MetricsRequestHandleV1>,
) -> Result<Phase43CompletedOutputV1, ApiErrorV1> {
    let mut raw = String::new();
    while let Some(event) = receiver.recv().await {
        match event {
            SchedulerEventV1::Delta(delta) => {
                let next = raw
                    .len()
                    .checked_add(delta.len())
                    .ok_or_else(|| ApiErrorV1::generation_failed("protocol output overflowed"))?;
                if next > MAX_PROTOCOL_OUTPUT_BYTES {
                    return Err(ApiErrorV1::generation_failed(
                        "protocol output exceeded 16 MiB",
                    ));
                }
                raw.push_str(&delta);
                if let Some(metrics) = &mut metrics {
                    metrics.observe_ttft_since_start();
                }
            }
            SchedulerEventV1::Logprobs(_) => {
                return Err(ApiErrorV1::generation_failed(
                    "protocol generation unexpectedly returned logprobs",
                ));
            }
            SchedulerEventV1::Finished(completion) => {
                let output = finish_protocol_output(context, decoder.as_ref(), raw, &completion)?;
                if let Some(metrics) = &mut metrics {
                    metrics.record_tokens(
                        completion.usage.prompt_tokens,
                        completion.usage.completion_tokens,
                    );
                    metrics.finish(RequestOutcomeV1::Success);
                }
                return Ok(output);
            }
            SchedulerEventV1::Failed(error) => {
                if let Some(metrics) = &mut metrics {
                    metrics.finish(RequestOutcomeV1::Error);
                }
                return Err(error);
            }
        }
    }
    if let Some(metrics) = &mut metrics {
        metrics.finish(RequestOutcomeV1::Error);
    }
    Err(ApiErrorV1::generation_failed(
        "generation ended without a terminal event",
    ))
}

fn finish_protocol_output(
    context: &ProtocolContextV1,
    decoder: Option<&ToolDecodeContextV1>,
    raw: String,
    completion: &crate::runtime::BackendCompletionV1,
) -> Result<Phase43CompletedOutputV1, ApiErrorV1> {
    let usage = Phase43UsageV1::new(
        completion.usage.prompt_tokens,
        completion.usage.completion_tokens,
    )
    .map_err(|error| ApiErrorV1::generation_failed(error.to_string()))?;
    let (text, reasoning, calls) = if let Some(decoder) = decoder {
        let envelope = decoder
            .protocol
            .decode_generation_envelope_with_reasoning(
                &raw,
                &decoder.choice,
                decoder.policy,
                decoder.include_reasoning,
            )
            .map_err(|_| ApiErrorV1::generation_failed("generated protocol output was invalid"))?;
        match envelope {
            CanonicalGenerationEnvelopeV1::Message { text, reasoning } => {
                (Some(text), reasoning, Vec::new())
            }
            CanonicalGenerationEnvelopeV1::ToolCalls { calls, reasoning } => {
                let calls = calls
                    .iter()
                    .enumerate()
                    .map(|(index, call)| context.tool_call(index, call.name(), call.arguments()))
                    .collect();
                (None, reasoning, calls)
            }
        }
    } else if context.reasoning {
        let (reasoning, text) = split_completed_reasoning(&raw)?;
        (Some(text), reasoning, Vec::new())
    } else {
        (Some(raw), None, Vec::new())
    };
    let finish_reason = if !calls.is_empty() {
        Phase43FinishReasonV1::ToolUse
    } else if completion.matched_stop.is_some() {
        Phase43FinishReasonV1::StopSequence
    } else {
        match completion.finish_reason {
            FinishReasonV1::Stop => Phase43FinishReasonV1::Stop,
            FinishReasonV1::Length => Phase43FinishReasonV1::Length,
        }
    };
    let mut output = Phase43CompletedOutputV1::new(
        context.id.clone(),
        context.item_id.clone(),
        finish_reason,
        usage,
    )
    .with_model(context.model.clone())
    .with_created_at(context.created_at)
    .with_tool_calls(calls);
    if let Some(text) = text {
        output = output.with_text(text);
    }
    if let Some(reasoning) = reasoning {
        output = output
            .with_reasoning_item_id(context.reasoning_item_id.clone())
            .with_reasoning(reasoning);
    }
    if let Some(sequence) = &completion.matched_stop {
        output = output.with_stop_sequence(sequence.clone());
    }
    Ok(output)
}

fn split_completed_reasoning(raw: &str) -> Result<(Option<String>, String), ApiErrorV1> {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let trimmed = raw.trim_start_matches(['\r', '\n', ' ']);
    if !trimmed.starts_with(OPEN) {
        return Ok((None, raw.to_owned()));
    }
    let remainder = &trimmed[OPEN.len()..];
    let Some(close) = remainder.find(CLOSE) else {
        return Err(ApiErrorV1::generation_failed(
            "reasoning output omitted the closing marker",
        ));
    };
    let reasoning = remainder[..close]
        .trim_start_matches(['\r', '\n'])
        .to_owned();
    let text = remainder[close + CLOSE.len()..]
        .trim_start_matches(['\r', '\n', ' '])
        .to_owned();
    Ok((Some(reasoning), text))
}

struct ProtocolStreamStateV1 {
    receiver: Option<GenerationReceiverV1>,
    context: ProtocolContextV1,
    decoder: Option<ToolDecodeContextV1>,
    raw: String,
    queued: VecDeque<Phase43SseEventV1>,
    metrics: Option<MetricsRequestHandleV1>,
    terminal: bool,
}

fn protocol_stream(
    receiver: GenerationReceiverV1,
    context: ProtocolContextV1,
    decoder: Option<ToolDecodeContextV1>,
    metrics: Option<MetricsRequestHandleV1>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state = ProtocolStreamStateV1 {
        receiver: Some(receiver),
        context,
        decoder,
        raw: String::new(),
        queued: VecDeque::new(),
        metrics,
        terminal: false,
    };
    Sse::new(stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.queued.pop_front() {
                return Some((Ok(named_event(event)), state));
            }
            if state.terminal {
                return None;
            }
            let receiver = state.receiver.as_mut()?;
            match receiver.recv().await {
                Some(SchedulerEventV1::Delta(delta)) => {
                    let next = state.raw.len().checked_add(delta.len());
                    if next.is_none_or(|next| next > MAX_PROTOCOL_OUTPUT_BYTES) {
                        queue_stream_error(
                            &mut state,
                            "generation_failed",
                            "protocol output exceeded 16 MiB",
                        );
                    } else {
                        state.raw.push_str(&delta);
                        if let Some(metrics) = &mut state.metrics {
                            metrics.observe_ttft_since_start();
                        }
                    }
                }
                Some(SchedulerEventV1::Logprobs(_)) => queue_stream_error(
                    &mut state,
                    "generation_failed",
                    "protocol generation unexpectedly returned logprobs",
                ),
                Some(SchedulerEventV1::Finished(completion)) => {
                    let raw = std::mem::take(&mut state.raw);
                    match finish_protocol_output(
                        &state.context,
                        state.decoder.as_ref(),
                        raw,
                        &completion,
                    )
                    .and_then(|output| stream_events(&state.context, &output))
                    {
                        Ok(events) => {
                            if let Some(metrics) = &mut state.metrics {
                                metrics.record_tokens(
                                    completion.usage.prompt_tokens,
                                    completion.usage.completion_tokens,
                                );
                                metrics.finish(RequestOutcomeV1::Success);
                            }
                            state.queued.extend(events);
                            state.receiver = None;
                            state.terminal = true;
                        }
                        Err(error) => {
                            queue_stream_error(
                                &mut state,
                                error.code().as_str(),
                                public_error_message(error.code()),
                            );
                        }
                    }
                }
                Some(SchedulerEventV1::Failed(error)) => {
                    queue_stream_error(
                        &mut state,
                        error.code().as_str(),
                        public_error_message(error.code()),
                    );
                }
                None => queue_stream_error(
                    &mut state,
                    "generation_failed",
                    "generation ended without a terminal event",
                ),
            }
        }
    }))
}

fn stream_events(
    context: &ProtocolContextV1,
    output: &Phase43CompletedOutputV1,
) -> Result<Vec<Phase43SseEventV1>, ApiErrorV1> {
    match context.profile {
        ProtocolProfileV1::Responses => ResponsesStreamBuilderV1::new(&context.id)
            .complete(output)
            .map_err(|error| ApiErrorV1::generation_failed(error.to_string())),
        ProtocolProfileV1::Anthropic => AnthropicStreamBuilderV1::new(&context.id)
            .complete(output)
            .map_err(|error| ApiErrorV1::generation_failed(error.to_string())),
    }
}

fn queue_stream_error(state: &mut ProtocolStreamStateV1, code: &str, message: &str) {
    if let Some(metrics) = &mut state.metrics {
        metrics.finish(RequestOutcomeV1::Error);
    }
    let event = match state.context.profile {
        ProtocolProfileV1::Responses => ResponsesStreamBuilderV1::new(&state.context.id)
            .error(code, message)
            .expect("fresh Responses builder is open"),
        ProtocolProfileV1::Anthropic => AnthropicStreamBuilderV1::new(&state.context.id)
            .error("api_error", message)
            .expect("fresh Anthropic builder is open"),
    };
    state.queued.push_back(event);
    state.receiver = None;
    state.terminal = true;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReplayFrameV1 {
    event: String,
    data: Value,
}

fn spawn_protocol_replay_producer(
    receiver: GenerationReceiverV1,
    context: ProtocolContextV1,
    decoder: Option<ToolDecodeContextV1>,
    replay: ResumableStoreV1,
    metrics: Option<MetricsRequestHandleV1>,
) {
    tokio::spawn(async move {
        match collect_protocol_output(receiver, &context, decoder, metrics).await {
            Ok(output) => match stream_events(&context, &output) {
                Ok(events) => {
                    let mut frames = Vec::with_capacity(events.len());
                    for event in events {
                        let frame = ReplayFrameV1 {
                            event: event.event,
                            data: event.data,
                        };
                        let data = match serde_json::to_string(&frame) {
                            Ok(data) => data,
                            Err(_) => {
                                terminate_protocol_replay(
                                    &replay,
                                    &context,
                                    "response serialization failed",
                                );
                                return;
                            }
                        };
                        frames.push(data);
                    }
                    let lengths = frames.iter().map(String::len).collect::<Vec<_>>();
                    if !replay.can_retain_batch(&lengths) {
                        terminate_protocol_replay(
                            &replay,
                            &context,
                            "resumable stream exceeded the configured replay window",
                        );
                        return;
                    }
                    let count = frames.len();
                    for (index, data) in frames.into_iter().enumerate() {
                        if replay
                            .append(&context.id, data, index + 1 == count)
                            .is_err()
                        {
                            terminate_protocol_replay(
                                &replay,
                                &context,
                                "resumable replay limit exceeded",
                            );
                            return;
                        }
                    }
                }
                Err(error) => {
                    terminate_protocol_replay(
                        &replay,
                        &context,
                        public_error_message(error.code()),
                    );
                }
            },
            Err(error) => {
                terminate_protocol_replay(&replay, &context, public_error_message(error.code()));
            }
        }
    });
}

fn terminate_protocol_replay(
    replay: &ResumableStoreV1,
    context: &ProtocolContextV1,
    message: &str,
) {
    let event = match context.profile {
        ProtocolProfileV1::Responses => ResponsesStreamBuilderV1::new(&context.id)
            .error("generation_failed", message)
            .expect("fresh builder"),
        ProtocolProfileV1::Anthropic => AnthropicStreamBuilderV1::new(&context.id)
            .error("api_error", message)
            .expect("fresh builder"),
    };
    let data = serde_json::to_string(&ReplayFrameV1 {
        event: event.event,
        data: event.data,
    })
    .unwrap_or_else(|_| {
        r#"{"event":"error","data":{"type":"error","message":"replay failed"}}"#.to_owned()
    });
    if replay.append(&context.id, data, true).is_err() {
        replay.terminate(&context.id);
    }
}

struct NamedReplayStateV1 {
    replay: ResumableStoreV1,
    id: String,
    cursor: u64,
    queued: VecDeque<crate::resume::ReplayEventV1>,
    terminal: bool,
}

fn named_replay_stream(
    replay: ResumableStoreV1,
    id: String,
    cursor: u64,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state = NamedReplayStateV1 {
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
                let frame =
                    serde_json::from_str::<ReplayFrameV1>(&event.data).unwrap_or(ReplayFrameV1 {
                        event: "error".to_owned(),
                        data: json!({"type":"error","message":"replay frame is invalid"}),
                    });
                return Some((
                    Ok(Event::default()
                        .id(event.id.to_string())
                        .event(frame.event)
                        .data(frame.data.to_string())),
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
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                }
                Err(_) => return None,
            }
        }
    }))
}

pub(crate) async fn resume_response(
    State(state): State<Arc<AppStateV1>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let response = resume_protocol(&state, id, &headers, ProtocolProfileV1::Responses);
    record_http(&state, HttpEndpointV1::ResponsesReplay, &response);
    response
}

pub(crate) async fn resume_anthropic_message(
    State(state): State<Arc<AppStateV1>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let response = resume_protocol(&state, id, &headers, ProtocolProfileV1::Anthropic);
    record_http(&state, HttpEndpointV1::AnthropicReplay, &response);
    response
}

fn resume_protocol(
    state: &AppStateV1,
    id: String,
    headers: &HeaderMap,
    profile: ProtocolProfileV1,
) -> Response {
    if let Err(error) = authorize_user(headers, state) {
        return protocol_error_response(profile, &error);
    }
    let expected_prefix = match profile {
        ProtocolProfileV1::Responses => "resp_",
        ProtocolProfileV1::Anthropic => "msg_",
    };
    if !id.starts_with(expected_prefix) {
        return protocol_error_response(profile, &ApiErrorV1::replay_not_found());
    }
    let Some(replay) = state.config.replay.clone() else {
        return protocol_error_response(profile, &ApiErrorV1::replay_not_found());
    };
    let cursor = match parse_last_event_id(headers) {
        Ok(cursor) => cursor,
        Err(error) => return protocol_error_response(profile, &error),
    };
    match replay.read_after(&id, cursor) {
        Ok(_) => named_replay_stream(replay, id, cursor).into_response(),
        Err(ReplayErrorV1::NotFound) => {
            protocol_error_response(profile, &ApiErrorV1::replay_not_found())
        }
        Err(ReplayErrorV1::CursorOutOfRange) => {
            protocol_error_response(profile, &ApiErrorV1::replay_out_of_range())
        }
        Err(_) => protocol_error_response(
            profile,
            &ApiErrorV1::generation_failed("resumable stream is unavailable"),
        ),
    }
}

fn named_event(event: Phase43SseEventV1) -> Event {
    Event::default()
        .event(event.event)
        .data(event.data.to_string())
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

fn convert_phase43_error(error: Phase43ApiErrorV1) -> ApiErrorV1 {
    let code = match error.code() {
        Phase43ErrorCodeV1::InvalidJson => ErrorCodeV1::InvalidJson,
        Phase43ErrorCodeV1::InvalidValue => ErrorCodeV1::InvalidValue,
        Phase43ErrorCodeV1::UnsupportedParameter => ErrorCodeV1::UnsupportedParameter,
        Phase43ErrorCodeV1::RequestTooLarge => ErrorCodeV1::RequestTooLarge,
    };
    ApiErrorV1::new(
        error.status(),
        public_error_message(code),
        "invalid_request_error",
        public_protocol_param(error.param()),
        code,
    )
}

fn tool_error(_: impl ToString) -> ApiErrorV1 {
    ApiErrorV1::invalid_value("tools", "tool protocol validation failed")
}

fn authorize_user(headers: &HeaderMap, state: &AppStateV1) -> Result<(), ApiErrorV1> {
    let mut values = headers.get_all(AUTHORIZATION).iter();
    let first = values.next().and_then(|value| value.to_str().ok());
    if values.next().is_none() && state.config.credentials.authorize_user(first) {
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
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"));
    if accepted {
        Ok(())
    } else {
        Err(ApiErrorV1::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "content-type must be application/json",
            "invalid_request_error",
            Some("Content-Type".to_owned()),
            ErrorCodeV1::UnsupportedMediaType,
        ))
    }
}

fn unique_header<'a>(
    headers: &'a HeaderMap,
    name: &HeaderName,
) -> Option<&'a axum::http::HeaderValue> {
    let mut values = headers.get_all(name).iter();
    let first = values.next()?;
    values.next().is_none().then_some(first)
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

fn protocol_error_response(profile: ProtocolProfileV1, error: &ApiErrorV1) -> Response {
    let error = redact_protocol_error(error);
    match profile {
        ProtocolProfileV1::Responses => error.into_response(),
        ProtocolProfileV1::Anthropic => anthropic_error_response(&error),
    }
}

fn anthropic_error_response(error: &ApiErrorV1) -> Response {
    let error = redact_protocol_error(error);
    let error_type = match error.code() {
        ErrorCodeV1::RateLimitExceeded => "rate_limit_error",
        ErrorCodeV1::GenerationFailed
        | ErrorCodeV1::RequestCancelled
        | ErrorCodeV1::ServerShutdown => "api_error",
        ErrorCodeV1::InvalidApiKey => "authentication_error",
        ErrorCodeV1::ModelNotFound => "not_found_error",
        _ => "invalid_request_error",
    };
    (
        error.status(),
        axum::Json(json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": public_error_message(error.code()),
            }
        })),
    )
        .into_response()
}

fn transport_error_response(_: crate::phase43_transport::Phase43TransportError) -> Response {
    redact_protocol_error(&ApiErrorV1::generation_failed(
        "protocol response serialization failed",
    ))
    .into_response()
}

fn redact_protocol_error(error: &ApiErrorV1) -> ApiErrorV1 {
    let error_type = match error.code() {
        ErrorCodeV1::RateLimitExceeded => "rate_limit_error",
        ErrorCodeV1::GenerationFailed
        | ErrorCodeV1::RequestCancelled
        | ErrorCodeV1::ServerShutdown => "server_error",
        _ => "invalid_request_error",
    };
    ApiErrorV1::new(
        error.status(),
        public_error_message(error.code()),
        error_type,
        public_protocol_param(error.param()),
        error.code(),
    )
}

fn public_error_message(code: ErrorCodeV1) -> &'static str {
    match code {
        ErrorCodeV1::InvalidJson => "request body is not valid JSON",
        ErrorCodeV1::InvalidValue => "request validation failed",
        ErrorCodeV1::UnsupportedParameter => "requested parameter is unsupported",
        ErrorCodeV1::InvalidApiKey => "authentication failed",
        ErrorCodeV1::ModelNotFound => "requested model is not served",
        ErrorCodeV1::RequestTooLarge => "request body is too large",
        ErrorCodeV1::RateLimitExceeded => "request capacity is exhausted",
        ErrorCodeV1::UnsupportedMediaType => "content type is unsupported",
        ErrorCodeV1::ReplayNotFound => "resumable stream was not found",
        ErrorCodeV1::ReplayOutOfRange => "event cursor is outside the replay window",
        ErrorCodeV1::SlotNotFound => "scheduler slot was not found",
        ErrorCodeV1::RequestCancelled => "generation request was cancelled",
        ErrorCodeV1::GenerationFailed => "generation failed",
        ErrorCodeV1::ServerShutdown => "generation service is shutting down",
    }
}

fn public_protocol_param(param: Option<&str>) -> Option<String> {
    let param = param?;
    let top_level = param
        .split(['.', '['])
        .next()
        .filter(|value| !value.is_empty())?;
    matches!(
        top_level,
        "Content-Type"
            | "Last-Event-ID"
            | "anthropic-version"
            | "input"
            | "instructions"
            | "max_output_tokens"
            | "max_tokens"
            | "messages"
            | "metadata"
            | "model"
            | "parallel_tool_calls"
            | "reasoning"
            | "sllm"
            | "stop_sequences"
            | "store"
            | "stream"
            | "system"
            | "temperature"
            | "tool_choice"
            | "tools"
            | "top_p"
    )
    .then(|| top_level.to_owned())
}

fn record_http(state: &AppStateV1, endpoint: HttpEndpointV1, response: &Response) {
    if let Some(metrics) = &state.config.metrics {
        metrics.record_http(endpoint, response.status().as_u16());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_split_is_closed_and_does_not_republish_markers() {
        assert_eq!(
            split_completed_reasoning("<think>\nwhy</think>\nanswer").unwrap(),
            (Some("why".to_owned()), "answer".to_owned())
        );
        assert!(split_completed_reasoning("<think>unfinished").is_err());
    }

    #[test]
    fn contexts_keep_profile_ids_disjoint() {
        let responses = ProtocolContextV1::new(ProtocolProfileV1::Responses, "m", false);
        let anthropic = ProtocolContextV1::new(ProtocolProfileV1::Anthropic, "m", false);
        assert!(responses.id.starts_with("resp_"));
        assert!(anthropic.id.starts_with("msg_"));
    }

    #[test]
    fn anthropic_version_constant_is_the_pinned_profile() {
        assert_eq!(crate::phase43_api::ANTHROPIC_API_VERSION_V1, "2023-06-01");
    }

    #[test]
    fn resumable_budget_covers_worst_case_token_piece_and_json_escaping() {
        assert!(validate_resumable_budget(true, 40, "max_tokens").is_ok());
        assert!(validate_resumable_budget(true, 41, "max_tokens").is_err());
        assert!(validate_resumable_budget(false, 4096, "max_tokens").is_ok());

        let model = "\0".repeat(crate::phase43_api::MAX_MODEL_ALIAS_BYTES);
        let context = ProtocolContextV1::new(ProtocolProfileV1::Responses, &model, false);
        let text = "\0".repeat(
            usize::try_from(MAX_RESUMABLE_PROTOCOL_TOKENS).unwrap()
                * sllm_core::MAX_TOKEN_PIECE_BYTES,
        );
        let output = Phase43CompletedOutputV1::new(
            context.id.clone(),
            context.item_id.clone(),
            Phase43FinishReasonV1::Stop,
            Phase43UsageV1::new(1, u64::from(MAX_RESUMABLE_PROTOCOL_TOKENS)).unwrap(),
        )
        .with_model(model)
        .with_created_at(context.created_at)
        .with_text(text);
        let frames = stream_events(&context, &output)
            .unwrap()
            .into_iter()
            .map(|event| {
                serde_json::to_string(&ReplayFrameV1 {
                    event: event.event,
                    data: event.data,
                })
                .unwrap()
            })
            .collect::<Vec<_>>();
        let replay = ResumableStoreV1::new(1, 64).unwrap();
        let lengths = frames.iter().map(String::len).collect::<Vec<_>>();
        assert!(
            replay.can_retain_batch(&lengths),
            "replay lengths {lengths:?}, total {}",
            lengths.iter().sum::<usize>()
        );
    }
}
