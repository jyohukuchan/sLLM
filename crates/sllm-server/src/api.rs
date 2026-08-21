use std::collections::BTreeSet;
use std::fmt;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::{DeserializeOwned, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sllm_core::SamplingParametersV1;
use sllm_frontend::{
    BoundedImageBytesV1, GenerationConfigV1, MAX_TOTAL_VISUAL_TOKENS_V1, ProcessedVisionInputV1,
    Qwen35ChatMessageV1, Qwen35VisionProcessorV1, ThinkingModeV1,
};

pub const MAX_REQUEST_BODY_BYTES: usize = 96 * 1024 * 1024;
pub const DEFAULT_MAX_COMPLETION_TOKENS: u32 = 256;
pub const MAX_COMPLETION_TOKENS: u32 = 4_096;
const MAX_MODEL_ALIAS_BYTES: usize = 256;
const MAX_MESSAGES: usize = 1_024;

const SUPPORTED_FIELDS: &[&str] = &[
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "stream",
    "n",
    "sllm",
];

const KNOWN_UNSUPPORTED_FIELDS: &[&str] = &[
    "audio",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "metadata",
    "modalities",
    "parallel_tool_calls",
    "prediction",
    "reasoning_effort",
    "response_format",
    "service_tier",
    "store",
    "stream_options",
    "tool_choice",
    "tools",
    "top_logprobs",
    "user",
    "web_search_options",
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ChatCompatibilityProfileV1 {
    #[default]
    Strict,
    OpenWebUi,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReasoningOptionsV1 {
    thinking: ThinkingModeV1,
    separate_reasoning: bool,
}

impl ReasoningOptionsV1 {
    pub const fn disabled() -> Self {
        Self {
            thinking: ThinkingModeV1::Disabled,
            separate_reasoning: false,
        }
    }

    pub const fn thinking(self) -> ThinkingModeV1 {
        self.thinking
    }

    pub const fn separate_reasoning(self) -> bool {
        self.separate_reasoning
    }

    pub const fn enabled(self) -> bool {
        matches!(self.thinking, ThinkingModeV1::Enabled)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCodeV1 {
    InvalidJson,
    InvalidValue,
    UnsupportedParameter,
    InvalidApiKey,
    ModelNotFound,
    RequestTooLarge,
    RateLimitExceeded,
    UnsupportedMediaType,
    ReplayNotFound,
    ReplayOutOfRange,
    SlotNotFound,
    RequestCancelled,
    GenerationFailed,
    ServerShutdown,
}

impl ErrorCodeV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidValue => "invalid_value",
            Self::UnsupportedParameter => "unsupported_parameter",
            Self::InvalidApiKey => "invalid_api_key",
            Self::ModelNotFound => "model_not_found",
            Self::RequestTooLarge => "request_too_large",
            Self::RateLimitExceeded => "rate_limit_exceeded",
            Self::UnsupportedMediaType => "unsupported_media_type",
            Self::ReplayNotFound => "replay_not_found",
            Self::ReplayOutOfRange => "replay_out_of_range",
            Self::SlotNotFound => "slot_not_found",
            Self::RequestCancelled => "request_cancelled",
            Self::GenerationFailed => "generation_failed",
            Self::ServerShutdown => "server_shutdown",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ApiErrorV1 {
    status: StatusCode,
    message: String,
    error_type: &'static str,
    param: Option<String>,
    code: ErrorCodeV1,
}

impl ApiErrorV1 {
    pub fn new(
        status: StatusCode,
        message: impl Into<String>,
        error_type: &'static str,
        param: Option<String>,
        code: ErrorCodeV1,
    ) -> Self {
        Self {
            status,
            message: message.into(),
            error_type,
            param,
            code,
        }
    }

    pub fn invalid_json(message: impl Into<String>, param: Option<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            param,
            ErrorCodeV1::InvalidJson,
        )
    }

    pub fn invalid_value(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            Some(param.into()),
            ErrorCodeV1::InvalidValue,
        )
    }

    pub fn unsupported(param: impl Into<String>) -> Self {
        let param = param.into();
        Self::new(
            StatusCode::BAD_REQUEST,
            format!("parameter {param} is not supported by profile v1"),
            "invalid_request_error",
            Some(param),
            ErrorCodeV1::UnsupportedParameter,
        )
    }

    pub fn model_not_found(model: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            format!("model {model} is not served"),
            "invalid_request_error",
            Some("model".to_owned()),
            ErrorCodeV1::ModelNotFound,
        )
    }

    pub fn rate_limited() -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "the bounded request queue is full",
            "rate_limit_error",
            None,
            ErrorCodeV1::RateLimitExceeded,
        )
    }

    pub fn generation_failed(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            message,
            "server_error",
            None,
            ErrorCodeV1::GenerationFailed,
        )
    }

    pub fn request_cancelled() -> Self {
        Self::new(
            StatusCode::CONFLICT,
            "the generation request was cancelled",
            "server_error",
            None,
            ErrorCodeV1::RequestCancelled,
        )
    }

    pub fn replay_not_found() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "the resumable stream does not exist",
            "invalid_request_error",
            None,
            ErrorCodeV1::ReplayNotFound,
        )
    }

    pub fn replay_out_of_range() -> Self {
        Self::new(
            StatusCode::RANGE_NOT_SATISFIABLE,
            "the requested event cursor is older than the replay window",
            "invalid_request_error",
            None,
            ErrorCodeV1::ReplayOutOfRange,
        )
    }

    pub fn slot_not_found() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "the scheduler slot does not exist",
            "invalid_request_error",
            None,
            ErrorCodeV1::SlotNotFound,
        )
    }

    pub fn server_shutdown() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "the generation scheduler is shutting down",
            "server_error",
            None,
            ErrorCodeV1::ServerShutdown,
        )
    }

    pub const fn status(&self) -> StatusCode {
        self.status
    }

    pub const fn code(&self) -> ErrorCodeV1 {
        self.code
    }

    pub fn param(&self) -> Option<&str> {
        self.param.as_deref()
    }

    pub(crate) fn envelope(&self) -> ErrorEnvelopeV1<'_> {
        ErrorEnvelopeV1 {
            error: ErrorBodyV1 {
                message: &self.message,
                error_type: self.error_type,
                param: self.param.as_deref(),
                code: self.code.as_str(),
            },
        }
    }
}

impl fmt::Display for ApiErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ApiErrorV1 {}

impl IntoResponse for ApiErrorV1 {
    fn into_response(self) -> Response {
        (self.status, Json(self.envelope())).into_response()
    }
}

#[derive(Serialize)]
pub(crate) struct ErrorEnvelopeV1<'a> {
    error: ErrorBodyV1<'a>,
}

#[derive(Serialize)]
struct ErrorBodyV1<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    error_type: &'a str,
    param: Option<&'a str>,
    code: &'a str,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ChatContentPartV1 {
    Text(String),
    Image(ProcessedVisionInputV1),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessageV1 {
    pub(crate) inner: Qwen35ChatMessageV1,
    pub(crate) parts: Vec<ChatContentPartV1>,
}

impl ChatMessageV1 {
    pub fn inner(&self) -> &Qwen35ChatMessageV1 {
        &self.inner
    }

    pub fn parts(&self) -> &[ChatContentPartV1] {
        &self.parts
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatCompletionRequestV1 {
    model: String,
    messages: Vec<ChatMessageV1>,
    generation: GenerationConfigV1,
    seed: Option<i64>,
    stream: bool,
    reasoning: ReasoningOptionsV1,
    resumable: bool,
}

impl ChatCompletionRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn messages(&self) -> &[ChatMessageV1] {
        &self.messages
    }

    pub const fn generation(&self) -> &GenerationConfigV1 {
        &self.generation
    }

    pub const fn seed(&self) -> Option<i64> {
        self.seed
    }

    pub const fn sampling_seed(&self) -> Option<u64> {
        match self.seed {
            Some(seed) => Some(u64::from_ne_bytes(seed.to_ne_bytes())),
            None => None,
        }
    }

    pub const fn stream(&self) -> bool {
        self.stream
    }

    pub const fn reasoning(&self) -> ReasoningOptionsV1 {
        self.reasoning
    }

    pub const fn resumable(&self) -> bool {
        self.resumable
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FinishReasonV1 {
    Stop,
    Length,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct TokenUsageV1 {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

impl TokenUsageV1 {
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Result<Self, ApiErrorV1> {
        let total_tokens = prompt_tokens
            .checked_add(completion_tokens)
            .ok_or_else(|| ApiErrorV1::generation_failed("token usage accounting overflowed"))?;
        Ok(Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireChatCompletionRequest {
    model: String,
    messages: Vec<WireMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_completion_tokens: Option<u32>,
    max_tokens: Option<u32>,
    stop: Option<WireStop>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<i64>,
    stream: Option<bool>,
    n: Option<u32>,
    sllm: Option<WireSllmOptions>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSllmOptions {
    thinking: Option<WireThinkingMode>,
    separate_reasoning: Option<bool>,
    resumable: Option<bool>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum WireThinkingMode {
    Enabled,
    Disabled,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    role: String,
    content: Value,
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WireStop {
    One(String),
    Many(Vec<String>),
}

fn parse_text_message_content(
    index: usize,
    content: Value,
) -> Result<(String, Vec<ChatContentPartV1>), ApiErrorV1> {
    let param = format!("messages[{index}].content");
    let text = content.as_str().ok_or_else(|| {
        if content.is_array() || content.is_object() {
            ApiErrorV1::unsupported(param.clone())
        } else {
            ApiErrorV1::invalid_value(param.clone(), "message content must be a string")
        }
    })?;
    Ok((
        text.to_owned(),
        vec![ChatContentPartV1::Text(text.to_owned())],
    ))
}

fn parse_user_message_content(
    index: usize,
    content: Value,
    total_images: &mut usize,
    total_visual_tokens: &mut u64,
    image_seen: &mut BTreeSet<String>,
) -> Result<(String, Vec<ChatContentPartV1>), ApiErrorV1> {
    if content.is_string() {
        return parse_text_message_content(index, content);
    }
    let values = content.as_array().ok_or_else(|| {
        ApiErrorV1::invalid_value(
            format!("messages[{index}].content"),
            "user content must be a string or a content-part array",
        )
    })?;
    if values.is_empty() {
        return Err(ApiErrorV1::invalid_value(
            format!("messages[{index}].content"),
            "content-part array must not be empty",
        ));
    }
    let processor = Qwen35VisionProcessorV1;
    let mut trusted = String::new();
    let mut parts = Vec::with_capacity(values.len());
    let mut text_seen = BTreeSet::new();
    for (part_index, value) in values.iter().enumerate() {
        let param = format!("messages[{index}].content[{part_index}]");
        let object = value
            .as_object()
            .ok_or_else(|| ApiErrorV1::invalid_value(&param, "content part must be an object"))?;
        let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
            ApiErrorV1::invalid_value(format!("{param}.type"), "content part type is required")
        })?;
        match kind {
            "text" => {
                if object.keys().any(|key| key != "type" && key != "text") {
                    return Err(ApiErrorV1::unsupported(param));
                }
                let text = object.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ApiErrorV1::invalid_value(
                        format!("{param}.text"),
                        "text content must be a string",
                    )
                })?;
                if text.is_empty() || !text_seen.insert(text.to_owned()) {
                    return Err(ApiErrorV1::invalid_value(
                        format!("{param}.text"),
                        "text content must be nonempty and unique within the message",
                    ));
                }
                trusted.push_str(text);
                parts.push(ChatContentPartV1::Text(text.to_owned()));
            }
            "image_url" => {
                if object.keys().any(|key| key != "type" && key != "image_url") {
                    return Err(ApiErrorV1::unsupported(param));
                }
                let image_url = object
                    .get("image_url")
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        ApiErrorV1::invalid_value(
                            format!("{param}.image_url"),
                            "image_url must be an object",
                        )
                    })?;
                if image_url.keys().any(|key| key != "url" && key != "detail") {
                    return Err(ApiErrorV1::unsupported(format!("{param}.image_url")));
                }
                if let Some(detail) = image_url.get("detail") {
                    if detail.as_str() != Some("auto") {
                        return Err(ApiErrorV1::unsupported(format!("{param}.image_url.detail")));
                    }
                }
                let url = image_url
                    .get("url")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ApiErrorV1::invalid_value(
                            format!("{param}.image_url.url"),
                            "image URL must be a string",
                        )
                    })?;
                if !url.starts_with("data:image/") {
                    return Err(ApiErrorV1::unsupported(format!("{param}.image_url.url")));
                }
                let encoded = BoundedImageBytesV1::from_data_url(url).map_err(|error| {
                    ApiErrorV1::invalid_value(format!("{param}.image_url.url"), error.to_string())
                })?;
                let digest = encoded.digest_hex();
                if !image_seen.insert(digest) {
                    return Err(ApiErrorV1::invalid_value(
                        format!("{param}.image_url.url"),
                        "duplicate image content is unsupported",
                    ));
                }
                *total_images = total_images.checked_add(1).ok_or_else(|| {
                    ApiErrorV1::invalid_value("messages", "image count overflowed")
                })?;
                if *total_images > 2 {
                    return Err(ApiErrorV1::invalid_value(
                        "messages",
                        "at most two images are supported",
                    ));
                }
                let decoded = encoded.decode_rgb().map_err(|error| {
                    ApiErrorV1::invalid_value(format!("{param}.image_url.url"), error.to_string())
                })?;
                let processed = processor.process(&decoded).map_err(|error| {
                    ApiErrorV1::invalid_value(format!("{param}.image_url.url"), error.to_string())
                })?;
                *total_visual_tokens = total_visual_tokens
                    .checked_add(processed.visual_tokens)
                    .ok_or_else(|| {
                        ApiErrorV1::invalid_value("messages", "visual token count overflowed")
                    })?;
                if *total_visual_tokens > MAX_TOTAL_VISUAL_TOKENS_V1 {
                    return Err(ApiErrorV1::invalid_value(
                        "messages",
                        "total visual token limit exceeded",
                    ));
                }
                trusted.push_str("<|vision_start|>");
                for _ in 0..processed.visual_tokens {
                    trusted.push_str("<|image_pad|>");
                }
                trusted.push_str("<|vision_end|>");
                parts.push(ChatContentPartV1::Image(processed));
            }
            _ => return Err(ApiErrorV1::unsupported(format!("{param}.type"))),
        }
    }
    Ok((trusted, parts))
}

#[cfg(test)]
pub(crate) fn parse_chat_completion_request(
    body: &[u8],
) -> Result<ChatCompletionRequestV1, ApiErrorV1> {
    parse_chat_completion_request_for_profile(body, ChatCompatibilityProfileV1::Strict)
}

pub(crate) fn parse_chat_completion_request_for_profile(
    body: &[u8],
    profile: ChatCompatibilityProfileV1,
) -> Result<ChatCompletionRequestV1, ApiErrorV1> {
    let strict = serde_json::from_slice::<StrictValue>(body)
        .map_err(|error| ApiErrorV1::invalid_json(error.to_string(), None))?;
    let object = strict
        .0
        .as_object()
        .ok_or_else(|| ApiErrorV1::invalid_json("request body must be a JSON object", None))?;
    let supported = SUPPORTED_FIELDS.iter().copied().collect::<BTreeSet<_>>();
    let known_unsupported = KNOWN_UNSUPPORTED_FIELDS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    for field in object.keys() {
        if supported.contains(field.as_str())
            || (profile == ChatCompatibilityProfileV1::OpenWebUi && field == "max_tokens")
        {
            continue;
        }
        let _is_known_standard_field = known_unsupported.contains(field.as_str());
        return Err(ApiErrorV1::unsupported(field.clone()));
    }

    let wire: WireChatCompletionRequest = deserialize_with_path(body)?;
    if wire.model.is_empty() || wire.model.len() > MAX_MODEL_ALIAS_BYTES {
        return Err(ApiErrorV1::invalid_value(
            "model",
            "model must be a nonempty alias of at most 256 bytes",
        ));
    }
    if wire.messages.is_empty() || wire.messages.len() > MAX_MESSAGES {
        return Err(ApiErrorV1::invalid_value(
            "messages",
            "messages must contain between 1 and 1024 entries",
        ));
    }
    if wire.n.unwrap_or(1) != 1 {
        return Err(ApiErrorV1::unsupported("n"));
    }

    let mut messages = Vec::with_capacity(wire.messages.len());
    let mut system_seen = false;
    let mut user_seen = false;
    let mut total_images = 0_usize;
    let mut total_visual_tokens = 0_u64;
    let mut image_seen = BTreeSet::new();
    for (index, message) in wire.messages.into_iter().enumerate() {
        let reasoning_content = message.reasoning_content;
        let (inner, parts) = match message.role.as_str() {
            "system" => {
                if reasoning_content.is_some() {
                    return Err(ApiErrorV1::unsupported(format!(
                        "messages[{index}].reasoning_content"
                    )));
                }
                if index != 0 || system_seen {
                    return Err(ApiErrorV1::invalid_value(
                        format!("messages[{index}].role"),
                        "system is allowed once and only as the first message",
                    ));
                }
                system_seen = true;
                let (content, parts) = parse_text_message_content(index, message.content)?;
                (Qwen35ChatMessageV1::system(content), parts)
            }
            "user" => {
                if reasoning_content.is_some() {
                    return Err(ApiErrorV1::unsupported(format!(
                        "messages[{index}].reasoning_content"
                    )));
                }
                user_seen = true;
                let (content, parts) = parse_user_message_content(
                    index,
                    message.content,
                    &mut total_images,
                    &mut total_visual_tokens,
                    &mut image_seen,
                )?;
                (Qwen35ChatMessageV1::user(content), parts)
            }
            "assistant" => {
                let (content, parts) = parse_text_message_content(index, message.content)?;
                (
                    Qwen35ChatMessageV1::assistant(content, reasoning_content),
                    parts,
                )
            }
            "developer" | "tool" | "function" => {
                return Err(ApiErrorV1::unsupported(format!("messages[{index}].role")));
            }
            _ => {
                return Err(ApiErrorV1::invalid_value(
                    format!("messages[{index}].role"),
                    "message role must be system, user, or assistant",
                ));
            }
        };
        messages.push(ChatMessageV1 { inner, parts });
    }
    if !user_seen {
        return Err(ApiErrorV1::invalid_value(
            "messages",
            "at least one user message is required",
        ));
    }

    let sampling = SamplingParametersV1::new(
        wire.temperature.unwrap_or(1.0),
        wire.top_p.unwrap_or(1.0),
        wire.presence_penalty.unwrap_or(0.0),
        wire.frequency_penalty.unwrap_or(0.0),
    )
    .map_err(|error| {
        let text = error.to_string();
        let param = if text.starts_with("temperature") {
            "temperature"
        } else if text.starts_with("top_p") {
            "top_p"
        } else if text.starts_with("presence_penalty") {
            "presence_penalty"
        } else {
            "frequency_penalty"
        };
        ApiErrorV1::invalid_value(param, text)
    })?;
    if wire.max_completion_tokens.is_some() && wire.max_tokens.is_some() {
        return Err(ApiErrorV1::invalid_value(
            "max_tokens",
            "max_tokens and max_completion_tokens cannot both be specified",
        ));
    }
    let (max_completion_tokens, max_tokens_param) =
        match (wire.max_completion_tokens, wire.max_tokens) {
            (Some(value), None) => (value, "max_completion_tokens"),
            (None, Some(value)) if profile == ChatCompatibilityProfileV1::OpenWebUi => {
                (value, "max_tokens")
            }
            (None, Some(_)) => return Err(ApiErrorV1::unsupported("max_tokens")),
            (None, None) => (DEFAULT_MAX_COMPLETION_TOKENS, "max_completion_tokens"),
            (Some(_), Some(_)) => unreachable!("handled above"),
        };
    if !(1..=MAX_COMPLETION_TOKENS).contains(&max_completion_tokens) {
        return Err(ApiErrorV1::invalid_value(
            max_tokens_param,
            format!("{max_tokens_param} must be in [1,4096]"),
        ));
    }
    let stop_strings = match wire.stop {
        None => Vec::new(),
        Some(WireStop::One(stop)) => vec![stop],
        Some(WireStop::Many(stops)) => {
            if stops.is_empty() {
                return Err(ApiErrorV1::invalid_value(
                    "stop",
                    "stop array must contain between 1 and 4 strings",
                ));
            }
            stops
        }
    };
    let generation = GenerationConfigV1::new(max_completion_tokens, sampling, stop_strings)
        .map_err(|error| ApiErrorV1::invalid_value("stop", error.to_string()))?;
    let (reasoning, resumable) = match wire.sllm {
        None => (ReasoningOptionsV1::disabled(), false),
        Some(options) => {
            let thinking = match options.thinking.unwrap_or(WireThinkingMode::Disabled) {
                WireThinkingMode::Enabled => ThinkingModeV1::Enabled,
                WireThinkingMode::Disabled => ThinkingModeV1::Disabled,
            };
            let separate_reasoning = options.separate_reasoning.unwrap_or(false);
            if separate_reasoning && !matches!(thinking, ThinkingModeV1::Enabled) {
                return Err(ApiErrorV1::invalid_value(
                    "sllm.separate_reasoning",
                    "separate_reasoning requires sllm.thinking=enabled",
                ));
            }
            (
                ReasoningOptionsV1 {
                    thinking,
                    separate_reasoning,
                },
                options.resumable.unwrap_or(false),
            )
        }
    };
    let stream = wire.stream.unwrap_or(false);
    if resumable && !stream {
        return Err(ApiErrorV1::invalid_value(
            "sllm.resumable",
            "resumable requires stream=true",
        ));
    }

    Ok(ChatCompletionRequestV1 {
        model: wire.model,
        messages,
        generation,
        seed: wire.seed,
        stream,
        reasoning,
        resumable,
    })
}

fn deserialize_with_path<T: DeserializeOwned>(body: &[u8]) -> Result<T, ApiErrorV1> {
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let value = serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
        let path = error.path().to_string();
        ApiErrorV1::invalid_json(
            error.inner().to_string(),
            (!path.is_empty()).then_some(path),
        )
    })?;
    deserializer
        .end()
        .map_err(|error| ApiErrorV1::invalid_json(error.to_string(), None))?;
    Ok(value)
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor).map(Self)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object members")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("JSON number is not finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictValue::deserialize(deserializer).map(|value| value.0)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictValue>()? {
            values.push(value.0);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON object member {key}"
                )));
            }
            let value = map.next_value::<StrictValue>()?;
            values.insert(key, value.0);
        }
        Ok(Value::Object(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid(extra: &str) -> Vec<u8> {
        format!(r#"{{"model":"qwen","messages":[{{"role":"user","content":"hi"}}]{extra}}}"#)
            .into_bytes()
    }

    #[test]
    fn supported_request_defaults_and_boundaries_are_typed() {
        let request = parse_chat_completion_request(&valid(
            r#", "temperature":0, "top_p":0, "presence_penalty":-2, "frequency_penalty":2, "max_completion_tokens":4096, "stop":["x","終"], "seed":-9223372036854775808, "stream":true, "n":1"#,
        ))
        .unwrap();
        assert_eq!(request.model(), "qwen");
        assert_eq!(request.messages().len(), 1);
        assert!(request.stream());
        assert_eq!(request.generation().max_new_tokens(), 4096);
        assert_eq!(request.seed(), Some(i64::MIN));
        assert_eq!(request.sampling_seed(), Some(1_u64 << 63));
        let maximum =
            parse_chat_completion_request(&valid(r#", "seed":9223372036854775807"#)).unwrap();
        assert_eq!(maximum.seed(), Some(i64::MAX));
        assert_eq!(maximum.sampling_seed(), Some(i64::MAX as u64));
        for value in ["-9223372036854775809", "9223372036854775808"] {
            assert_eq!(
                parse_chat_completion_request(&valid(&format!(r#", "seed":{value}"#)))
                    .unwrap_err()
                    .code(),
                ErrorCodeV1::InvalidJson
            );
        }
    }

    #[test]
    fn unknown_unsupported_and_duplicate_fields_fail_closed() {
        for (body, param, code) in [
            (
                valid(r#", "tools":null"#),
                "tools",
                ErrorCodeV1::UnsupportedParameter,
            ),
            (
                valid(r#", "response_format":{"type":"json_object"}"#),
                "response_format",
                ErrorCodeV1::UnsupportedParameter,
            ),
            (
                valid(r#", "logprobs":false"#),
                "logprobs",
                ErrorCodeV1::UnsupportedParameter,
            ),
            (
                valid(r#", "mystery":1"#),
                "mystery",
                ErrorCodeV1::UnsupportedParameter,
            ),
            (valid(r#", "n":2"#), "n", ErrorCodeV1::UnsupportedParameter),
            (
                br#"{"model":"qwen","messages":[{"role":"developer","content":"x"}]}"#.to_vec(),
                "messages[0].role",
                ErrorCodeV1::UnsupportedParameter,
            ),
            (
                br#"{"model":"qwen","messages":[{"role":"tool","content":"x"}]}"#.to_vec(),
                "messages[0].role",
                ErrorCodeV1::UnsupportedParameter,
            ),
            (
                br#"{"model":"qwen","messages":[{"role":"function","content":"x"}]}"#.to_vec(),
                "messages[0].role",
                ErrorCodeV1::UnsupportedParameter,
            ),
        ] {
            let error = parse_chat_completion_request(&body).unwrap_err();
            assert_eq!(error.param(), Some(param));
            assert_eq!(error.code(), code);
        }
        let duplicate = br#"{"model":"a","model":"b","messages":[{"role":"user","content":"x"}]}"#;
        assert_eq!(
            parse_chat_completion_request(duplicate).unwrap_err().code(),
            ErrorCodeV1::InvalidJson
        );
        let multipart = br#"{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]}"#;
        let request = parse_chat_completion_request(multipart).unwrap();
        assert_eq!(
            request.messages()[0].parts(),
            [ChatContentPartV1::Text("x".to_owned())]
        );
        let remote = br#"{"model":"qwen","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.test/a.png"}}]}]}"#;
        let error = parse_chat_completion_request(remote).unwrap_err();
        assert_eq!(error.code(), ErrorCodeV1::UnsupportedParameter);
        assert_eq!(error.param(), Some("messages[0].content[0].image_url.url"));
    }

    #[test]
    fn image_data_url_is_typed_and_duplicate_scope_is_the_whole_request() {
        let url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAAA1UlEQVR42u3BMQEAAADCoPVP7WULoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAGwEtAAGey8LtAAAAAElFTkSuQmCC";
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "qwen",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url, "detail": "auto"}},
                    {"type": "text", "text": "describe"}
                ]
            }]
        }))
        .unwrap();
        let request = parse_chat_completion_request(&body).unwrap();
        assert!(matches!(
            request.messages()[0].parts()[0],
            ChatContentPartV1::Image(_)
        ));

        let duplicate = serde_json::to_vec(&serde_json::json!({
            "model": "qwen",
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
            ]
        }))
        .unwrap();
        let error = parse_chat_completion_request(&duplicate).unwrap_err();
        assert_eq!(error.param(), Some("messages[1].content[0].image_url.url"));
        assert_eq!(error.code(), ErrorCodeV1::InvalidValue);
    }

    #[test]
    fn numeric_and_stop_boundaries_reject_both_sides() {
        for extra in [
            r#", "temperature":-0.01"#,
            r#", "temperature":2.01"#,
            r#", "top_p":-0.01"#,
            r#", "top_p":1.01"#,
            r#", "presence_penalty":-2.01"#,
            r#", "frequency_penalty":2.01"#,
            r#", "max_completion_tokens":0"#,
            r#", "max_completion_tokens":4097"#,
            r#", "stop":[]"#,
            r#", "stop":["a","b","c","d","e"]"#,
        ] {
            assert!(
                parse_chat_completion_request(&valid(extra)).is_err(),
                "{extra}"
            );
        }
    }

    #[test]
    fn type_and_message_validation_preserve_field_paths() {
        for (body, param, code) in [
            (
                br#"{"model":"qwen","messages":"not-an-array"}"#.as_slice(),
                "messages",
                ErrorCodeV1::InvalidJson,
            ),
            (
                br#"{"model":"qwen","messages":[{"role":3,"content":"x"}]}"#.as_slice(),
                "messages[0].role",
                ErrorCodeV1::InvalidJson,
            ),
            (
                br#"{"model":"qwen","messages":[{"role":"assistant","content":"x"}]}"#
                    .as_slice(),
                "messages",
                ErrorCodeV1::InvalidValue,
            ),
            (
                br#"{"model":"qwen","messages":[{"role":"user","content":"x"},{"role":"system","content":"late"}]}"#.as_slice(),
                "messages[1].role",
                ErrorCodeV1::InvalidValue,
            ),
        ] {
            let error = parse_chat_completion_request(body).unwrap_err();
            assert_eq!(error.param(), Some(param));
            assert_eq!(error.code(), code);
        }
    }

    #[test]
    fn model_and_message_caps_accept_the_limit_and_reject_one_over() {
        for (length, accepted) in [(256, true), (257, false)] {
            let body = serde_json::to_vec(&serde_json::json!({
                "model": "m".repeat(length),
                "messages": [{"role": "user", "content": "x"}],
            }))
            .unwrap();
            assert_eq!(parse_chat_completion_request(&body).is_ok(), accepted);
        }

        let mut messages = (0..1_024)
            .map(|_| serde_json::json!({"role": "user", "content": "x"}))
            .collect::<Vec<_>>();
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "qwen",
            "messages": messages,
        }))
        .unwrap();
        assert_eq!(
            parse_chat_completion_request(&body)
                .unwrap()
                .messages()
                .len(),
            1_024
        );

        messages.push(serde_json::json!({"role": "user", "content": "x"}));
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "qwen",
            "messages": messages,
        }))
        .unwrap();
        let error = parse_chat_completion_request(&body).unwrap_err();
        assert_eq!(error.param(), Some("messages"));
        assert_eq!(error.code(), ErrorCodeV1::InvalidValue);
    }

    #[test]
    fn reasoning_extension_is_typed_and_fail_closed() {
        let request = parse_chat_completion_request(&valid(
            r#", "sllm":{"thinking":"enabled","separate_reasoning":true}"#,
        ))
        .unwrap();
        assert!(request.reasoning().enabled());
        assert!(request.reasoning().separate_reasoning());

        let error = parse_chat_completion_request(&valid(
            r#", "sllm":{"thinking":"disabled","separate_reasoning":true}"#,
        ))
        .unwrap_err();
        assert_eq!(error.param(), Some("sllm.separate_reasoning"));
        assert_eq!(error.code(), ErrorCodeV1::InvalidValue);

        let request =
            parse_chat_completion_request(&valid(r#", "stream":true, "sllm":{"resumable":true}"#))
                .unwrap();
        assert!(request.stream());
        assert!(request.resumable());

        let error =
            parse_chat_completion_request(&valid(r#", "sllm":{"resumable":true}"#)).unwrap_err();
        assert_eq!(error.param(), Some("sllm.resumable"));
        assert_eq!(error.code(), ErrorCodeV1::InvalidValue);
    }

    #[test]
    fn openwebui_max_tokens_alias_is_separate_from_strict_profile() {
        let body = valid(r#", "max_tokens":37"#);
        let strict = parse_chat_completion_request(&body).unwrap_err();
        assert_eq!(strict.param(), Some("max_tokens"));
        assert_eq!(strict.code(), ErrorCodeV1::UnsupportedParameter);

        let compatible =
            parse_chat_completion_request_for_profile(&body, ChatCompatibilityProfileV1::OpenWebUi)
                .unwrap();
        assert_eq!(compatible.generation().max_new_tokens(), 37);

        for extra in [
            r#", "max_tokens":0"#,
            r#", "max_tokens":4097"#,
            r#", "max_tokens":17, "max_completion_tokens":17"#,
        ] {
            assert!(
                parse_chat_completion_request_for_profile(
                    &valid(extra),
                    ChatCompatibilityProfileV1::OpenWebUi,
                )
                .is_err(),
                "{extra}"
            );
        }
    }

    #[test]
    fn assistant_reasoning_history_round_trips_into_the_typed_renderer() {
        let body = br#"{"model":"qwen","messages":[{"role":"user","content":"Q1"},{"role":"assistant","content":"A1","reasoning_content":"R1"},{"role":"user","content":"Q2"}]}"#;
        let request = parse_chat_completion_request(body).unwrap();
        assert_eq!(
            request.messages()[1].inner(),
            &Qwen35ChatMessageV1::assistant("A1", Some("R1".to_owned()))
        );

        for role in ["system", "user"] {
            let body = serde_json::to_vec(&serde_json::json!({
                "model": "qwen",
                "messages": [{"role": role, "content": "x", "reasoning_content": "no"}],
            }))
            .unwrap();
            let error = parse_chat_completion_request(&body).unwrap_err();
            assert_eq!(error.param(), Some("messages[0].reasoning_content"));
            assert_eq!(error.code(), ErrorCodeV1::UnsupportedParameter);
        }
    }
}
