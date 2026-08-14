use std::collections::BTreeSet;
use std::fmt;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::{DeserializeOwned, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sllm_core::SamplingParametersV1;
use sllm_frontend::{GenerationConfigV1, Qwen35ChatMessageV1};

pub const MAX_REQUEST_BODY_BYTES: usize = 1_048_576;
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
    "stream",
    "n",
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
    "seed",
    "service_tier",
    "store",
    "stream_options",
    "tool_choice",
    "tools",
    "top_logprobs",
    "user",
    "web_search_options",
];

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessageV1 {
    pub(crate) inner: Qwen35ChatMessageV1,
}

impl ChatMessageV1 {
    pub fn inner(&self) -> &Qwen35ChatMessageV1 {
        &self.inner
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatCompletionRequestV1 {
    model: String,
    messages: Vec<ChatMessageV1>,
    generation: GenerationConfigV1,
    stream: bool,
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

    pub const fn stream(&self) -> bool {
        self.stream
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
    stop: Option<WireStop>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    stream: Option<bool>,
    n: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMessage {
    role: String,
    content: Value,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WireStop {
    One(String),
    Many(Vec<String>),
}

pub(crate) fn parse_chat_completion_request(
    body: &[u8],
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
        if supported.contains(field.as_str()) {
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
    for (index, message) in wire.messages.into_iter().enumerate() {
        let param = format!("messages[{index}].content");
        let content = message.content.as_str().ok_or_else(|| {
            if message.content.is_array() || message.content.is_object() {
                ApiErrorV1::unsupported(param.clone())
            } else {
                ApiErrorV1::invalid_value(param.clone(), "message content must be a string")
            }
        })?;
        let inner = match message.role.as_str() {
            "system" => {
                if index != 0 || system_seen {
                    return Err(ApiErrorV1::invalid_value(
                        format!("messages[{index}].role"),
                        "system is allowed once and only as the first message",
                    ));
                }
                system_seen = true;
                Qwen35ChatMessageV1::system(content)
            }
            "user" => {
                user_seen = true;
                Qwen35ChatMessageV1::user(content)
            }
            "assistant" => Qwen35ChatMessageV1::assistant(content, None),
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
        messages.push(ChatMessageV1 { inner });
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
    let max_completion_tokens = wire
        .max_completion_tokens
        .unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS);
    if !(1..=MAX_COMPLETION_TOKENS).contains(&max_completion_tokens) {
        return Err(ApiErrorV1::invalid_value(
            "max_completion_tokens",
            "max_completion_tokens must be in [1,4096]",
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

    Ok(ChatCompletionRequestV1 {
        model: wire.model,
        messages,
        generation,
        stream: wire.stream.unwrap_or(false),
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
            r#", "temperature":0, "top_p":0, "presence_penalty":-2, "frequency_penalty":2, "max_completion_tokens":4096, "stop":["x","終"], "stream":true, "n":1"#,
        ))
        .unwrap();
        assert_eq!(request.model(), "qwen");
        assert_eq!(request.messages().len(), 1);
        assert!(request.stream());
        assert_eq!(request.generation().max_new_tokens(), 4096);
    }

    #[test]
    fn unknown_unsupported_duplicate_and_multipart_fail_closed() {
        for (body, param, code) in [
            (valid(r#", "tools":null"#), "tools", ErrorCodeV1::UnsupportedParameter),
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
                valid(r#", "seed":0"#),
                "seed",
                ErrorCodeV1::UnsupportedParameter,
            ),
            (valid(r#", "mystery":1"#), "mystery", ErrorCodeV1::UnsupportedParameter),
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
                br#"{"model":"qwen","messages":[{"role":"function","content":"x"}]}"#
                    .to_vec(),
                "messages[0].role",
                ErrorCodeV1::UnsupportedParameter,
            ),
            (
                br#"{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]}"#.to_vec(),
                "messages[0].content",
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
}
