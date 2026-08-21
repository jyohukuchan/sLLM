//! Strict, transport-independent wire contracts for the Phase 42 inference
//! endpoints.  This module intentionally has no dependency on the runtime or
//! the existing Chat Completions implementation.  The server can later map
//! these validated DTOs into backend-neutral requests.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use axum::http::StatusCode;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};

pub const PHASE42_PROFILE_VERSION: &str = "sllm-inference-endpoints-v1";
pub const MAX_REQUEST_BODY_BYTES: usize = 96 * 1024 * 1024;
pub const MAX_MODEL_ALIAS_BYTES: usize = 256;
pub const MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_INPUT_ITEMS: usize = 2_048;
pub const MAX_RERANK_DOCUMENTS: usize = 256;
pub const MAX_TOKEN_COUNT: usize = 1_048_576;
pub const MAX_COMPLETION_TOKENS: u32 = 4_096;
pub const DEFAULT_COMPLETION_TOKENS: u32 = 256;
pub const MAX_STOP_SEQUENCES: usize = 4;
pub const MAX_STOP_BYTES: usize = 16 * 1024;
pub const MAX_LOGIT_BIAS_ENTRIES: usize = 4_096;
pub const MAX_DOCUMENT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_DIMENSIONS: u32 = 32_768;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCodeV1 {
    InvalidJson,
    InvalidValue,
    UnsupportedParameter,
    RequestTooLarge,
}

impl ErrorCodeV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidValue => "invalid_value",
            Self::UnsupportedParameter => "unsupported_parameter",
            Self::RequestTooLarge => "request_too_large",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ApiErrorV1 {
    status: StatusCode,
    message: String,
    param: Option<String>,
    code: ErrorCodeV1,
}

impl ApiErrorV1 {
    pub fn new(
        status: StatusCode,
        message: impl Into<String>,
        param: Option<String>,
        code: ErrorCodeV1,
    ) -> Self {
        Self {
            status,
            message: message.into(),
            param,
            code,
        }
    }

    pub fn invalid_json(message: impl Into<String>, param: Option<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            param,
            ErrorCodeV1::InvalidJson,
        )
    }

    pub fn invalid_value(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            Some(param.into()),
            ErrorCodeV1::InvalidValue,
        )
    }

    pub fn unsupported(param: impl Into<String>) -> Self {
        let param = param.into();
        Self::new(
            StatusCode::BAD_REQUEST,
            format!("parameter {param} is not supported by profile v1"),
            Some(param),
            ErrorCodeV1::UnsupportedParameter,
        )
    }

    pub fn request_too_large() -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("request body exceeds {MAX_REQUEST_BODY_BYTES} bytes"),
            None,
            ErrorCodeV1::RequestTooLarge,
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
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ApiErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ApiErrorV1 {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptV1 {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
    TokenSequences(Vec<Vec<u32>>),
}

impl PromptV1 {
    pub fn text(&self) -> Option<&str> {
        if let Self::Text(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn texts(&self) -> Option<&[String]> {
        if let Self::Texts(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn tokens(&self) -> Option<&[u32]> {
        if let Self::Tokens(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn token_sequences(&self) -> Option<&[Vec<u32>]> {
        if let Self::TokenSequences(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

pub type EmbeddingInputV1 = PromptV1;

#[derive(Clone, Debug, PartialEq)]
pub struct CompletionRequestV1 {
    model: String,
    prompt: PromptV1,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    stop: Vec<String>,
    presence_penalty: f32,
    frequency_penalty: f32,
    seed: Option<i64>,
    stream: bool,
    n: u32,
    logit_bias: BTreeMap<u32, f32>,
    logprobs: Option<u8>,
}

impl CompletionRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn prompt(&self) -> &PromptV1 {
        &self.prompt
    }
    pub const fn max_tokens(&self) -> u32 {
        self.max_tokens
    }
    pub const fn temperature(&self) -> f32 {
        self.temperature
    }
    pub const fn top_p(&self) -> f32 {
        self.top_p
    }
    pub fn stop(&self) -> &[String] {
        &self.stop
    }
    pub const fn presence_penalty(&self) -> f32 {
        self.presence_penalty
    }
    pub const fn frequency_penalty(&self) -> f32 {
        self.frequency_penalty
    }
    pub const fn seed(&self) -> Option<i64> {
        self.seed
    }
    pub const fn stream(&self) -> bool {
        self.stream
    }
    pub const fn n(&self) -> u32 {
        self.n
    }
    pub fn logit_bias(&self) -> &BTreeMap<u32, f32> {
        &self.logit_bias
    }
    pub const fn logprobs(&self) -> Option<u8> {
        self.logprobs
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingEncodingFormatV1 {
    Float,
    Base64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingRequestV1 {
    model: String,
    input: EmbeddingInputV1,
    encoding_format: EmbeddingEncodingFormatV1,
    dimensions: Option<u32>,
}

impl EmbeddingRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn input(&self) -> &EmbeddingInputV1 {
        &self.input
    }
    pub const fn encoding_format(&self) -> EmbeddingEncodingFormatV1 {
        self.encoding_format
    }
    pub const fn dimensions(&self) -> Option<u32> {
        self.dimensions
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RerankRequestV1 {
    model: String,
    query: String,
    documents: Vec<String>,
    top_n: Option<u32>,
    return_documents: bool,
}

impl RerankRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn query(&self) -> &str {
        &self.query
    }
    pub fn documents(&self) -> &[String] {
        &self.documents
    }
    pub const fn top_n(&self) -> Option<u32> {
        self.top_n
    }
    pub const fn return_documents(&self) -> bool {
        self.return_documents
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenizeRequestV1 {
    model: String,
    text: String,
    with_pieces: bool,
}

impl TokenizeRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    pub const fn with_pieces(&self) -> bool {
        self.with_pieces
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetokenizeRequestV1 {
    model: String,
    tokens: Vec<u32>,
    skip_special_tokens: bool,
}

impl DetokenizeRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub const fn skip_special_tokens(&self) -> bool {
        self.skip_special_tokens
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemplateRoleV1 {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemplateMessageV1 {
    role: TemplateRoleV1,
    content: String,
}

impl TemplateMessageV1 {
    pub const fn role(&self) -> TemplateRoleV1 {
        self.role
    }
    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ApplyTemplateRequestV1 {
    model: String,
    messages: Vec<TemplateMessageV1>,
    add_generation_prompt: bool,
    thinking: bool,
}

impl ApplyTemplateRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn messages(&self) -> &[TemplateMessageV1] {
        &self.messages
    }
    pub const fn add_generation_prompt(&self) -> bool {
        self.add_generation_prompt
    }
    pub const fn thinking(&self) -> bool {
        self.thinking
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InputTokensInputV1 {
    Text(String),
    Messages(Vec<TemplateMessageV1>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputTokensRequestV1 {
    model: String,
    input: InputTokensInputV1,
    add_generation_prompt: bool,
    thinking: bool,
}

impl InputTokensRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub const fn input(&self) -> &InputTokensInputV1 {
        &self.input
    }
    pub const fn add_generation_prompt(&self) -> bool {
        self.add_generation_prompt
    }
    pub const fn thinking(&self) -> bool {
        self.thinking
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InfillRequestV1 {
    model: String,
    prefix: String,
    suffix: String,
    prompt: Option<String>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    stop: Vec<String>,
    seed: Option<i64>,
    stream: bool,
    n: u32,
}

impl InfillRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
    pub fn suffix(&self) -> &str {
        &self.suffix
    }
    pub fn prompt(&self) -> Option<&str> {
        self.prompt.as_deref()
    }
    pub const fn max_tokens(&self) -> u32 {
        self.max_tokens
    }
    pub const fn temperature(&self) -> f32 {
        self.temperature
    }
    pub const fn top_p(&self) -> f32 {
        self.top_p
    }
    pub fn stop(&self) -> &[String] {
        &self.stop
    }
    pub const fn seed(&self) -> Option<i64> {
        self.seed
    }
    pub const fn stream(&self) -> bool {
        self.stream
    }
    pub const fn n(&self) -> u32 {
        self.n
    }
}

pub fn parse_completion_request(body: &[u8]) -> Result<CompletionRequestV1, ApiErrorV1> {
    let map = parse_object(
        body,
        &[
            "model",
            "prompt",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "stream",
            "n",
            "logit_bias",
            "logprobs",
        ],
        &[
            "best_of",
            "echo",
            "messages",
            "max_completion_tokens",
            "stream_options",
            "suffix",
            "tools",
            "tool_choice",
            "response_format",
            "user",
        ],
    )?;
    let model = model(&map)?;
    let prompt = parse_prompt(required(&map, "prompt")?, "prompt")?;
    let max_tokens = opt_u32(&map, "max_tokens")?.unwrap_or(DEFAULT_COMPLETION_TOKENS);
    if !(1..=MAX_COMPLETION_TOKENS).contains(&max_tokens) {
        return Err(invalid("max_tokens", "must be in [1,4096]"));
    }
    let temperature = opt_f32(&map, "temperature")?.unwrap_or(1.0);
    bounded_float("temperature", temperature, 0.0, 2.0)?;
    let top_p = opt_f32(&map, "top_p")?.unwrap_or(1.0);
    bounded_float("top_p", top_p, 0.0, 1.0)?;
    let presence_penalty = opt_f32(&map, "presence_penalty")?.unwrap_or(0.0);
    bounded_float("presence_penalty", presence_penalty, -2.0, 2.0)?;
    let frequency_penalty = opt_f32(&map, "frequency_penalty")?.unwrap_or(0.0);
    bounded_float("frequency_penalty", frequency_penalty, -2.0, 2.0)?;
    let stop = parse_stop(&map)?;
    let seed = opt_i64(&map, "seed")?;
    let stream = opt_bool(&map, "stream")?.unwrap_or(false);
    let n = opt_u32(&map, "n")?.unwrap_or(1);
    if !(1..=8).contains(&n) {
        return Err(invalid("n", "must be in [1,8]"));
    }
    let logit_bias = parse_logit_bias(&map)?;
    let logprobs = match map.get("logprobs") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let n = as_u64(value, "logprobs")?;
            if n > 5 {
                return Err(invalid("logprobs", "must be in [0,5]"));
            }
            Some(n as u8)
        }
    };
    Ok(CompletionRequestV1 {
        model,
        prompt,
        max_tokens,
        temperature,
        top_p,
        stop,
        presence_penalty,
        frequency_penalty,
        seed,
        stream,
        n,
        logit_bias,
        logprobs,
    })
}

pub fn parse_embedding_request(body: &[u8]) -> Result<EmbeddingRequestV1, ApiErrorV1> {
    let map = parse_object(
        body,
        &["model", "input", "encoding_format", "dimensions"],
        &["user", "truncate"],
    )?;
    let model = model(&map)?;
    let input = parse_prompt(required(&map, "input")?, "input")?;
    if match &input {
        PromptV1::Text(text) => text.is_empty(),
        PromptV1::Texts(texts) => texts.iter().any(String::is_empty),
        PromptV1::Tokens(_) | PromptV1::TokenSequences(_) => false,
    } {
        return Err(invalid("input", "embedding text inputs must not be empty"));
    }
    let encoding_format = match map.get("encoding_format") {
        None => EmbeddingEncodingFormatV1::Float,
        Some(Value::String(value)) if value == "float" => EmbeddingEncodingFormatV1::Float,
        Some(Value::String(value)) if value == "base64" => EmbeddingEncodingFormatV1::Base64,
        Some(_) => return Err(invalid("encoding_format", "must be float or base64")),
    };
    let dimensions = opt_u32(&map, "dimensions")?;
    if let Some(value) = dimensions {
        if !(1..=MAX_DIMENSIONS).contains(&value) {
            return Err(invalid("dimensions", "must be in [1,32768]"));
        }
    }
    Ok(EmbeddingRequestV1 {
        model,
        input,
        encoding_format,
        dimensions,
    })
}

pub fn parse_rerank_request(body: &[u8]) -> Result<RerankRequestV1, ApiErrorV1> {
    let map = parse_object(
        body,
        &["model", "query", "documents", "top_n", "return_documents"],
        &["max_tokens", "stream"],
    )?;
    let model = model(&map)?;
    let query = text(required(&map, "query")?, "query", true)?;
    let raw_docs = required(&map, "documents")?
        .as_array()
        .ok_or_else(|| invalid("documents", "must be an array of strings"))?;
    if raw_docs.is_empty() || raw_docs.len() > MAX_RERANK_DOCUMENTS {
        return Err(invalid("documents", "must contain 1..=256 entries"));
    }
    let mut documents = Vec::with_capacity(raw_docs.len());
    let mut document_bytes = 0usize;
    for (index, value) in raw_docs.iter().enumerate() {
        let document = text(value, &format!("documents[{index}]"), true)?;
        document_bytes = document_bytes
            .checked_add(document.len())
            .ok_or_else(|| invalid("documents", "document byte count overflowed"))?;
        if document_bytes > MAX_DOCUMENT_BYTES {
            return Err(invalid("documents", "document text exceeds 16 MiB"));
        }
        documents.push(document);
    }
    let top_n = opt_u32(&map, "top_n")?;
    if let Some(value) = top_n {
        if value == 0
            || usize::try_from(value)
                .ok()
                .is_none_or(|v| v > documents.len())
        {
            return Err(invalid("top_n", "must be in [1, number of documents]"));
        }
    }
    let return_documents = opt_bool(&map, "return_documents")?.unwrap_or(false);
    Ok(RerankRequestV1 {
        model,
        query,
        documents,
        top_n,
        return_documents,
    })
}

pub fn parse_tokenize_request(body: &[u8]) -> Result<TokenizeRequestV1, ApiErrorV1> {
    let map = parse_object(
        body,
        &["model", "text", "with_pieces"],
        &["add_special", "parse_special"],
    )?;
    let model = model(&map)?;
    let text = text(required(&map, "text")?, "text", false)?;
    let with_pieces = opt_bool(&map, "with_pieces")?.unwrap_or(false);
    Ok(TokenizeRequestV1 {
        model,
        text,
        with_pieces,
    })
}

pub fn parse_detokenize_request(body: &[u8]) -> Result<DetokenizeRequestV1, ApiErrorV1> {
    let map = parse_object(body, &["model", "tokens", "skip_special_tokens"], &[])?;
    let model = model(&map)?;
    let tokens = parse_tokens(required(&map, "tokens")?, "tokens", false)?;
    let skip_special_tokens = opt_bool(&map, "skip_special_tokens")?.unwrap_or(false);
    Ok(DetokenizeRequestV1 {
        model,
        tokens,
        skip_special_tokens,
    })
}

pub fn parse_apply_template_request(body: &[u8]) -> Result<ApplyTemplateRequestV1, ApiErrorV1> {
    let map = parse_object(
        body,
        &["model", "messages", "add_generation_prompt", "thinking"],
        &["chat_template", "custom_kwargs"],
    )?;
    let model = model(&map)?;
    let messages = parse_messages(required(&map, "messages")?)?;
    let add_generation_prompt = opt_bool(&map, "add_generation_prompt")?.unwrap_or(true);
    let thinking = opt_bool(&map, "thinking")?.unwrap_or(false);
    Ok(ApplyTemplateRequestV1 {
        model,
        messages,
        add_generation_prompt,
        thinking,
    })
}

pub fn parse_input_tokens_request(body: &[u8]) -> Result<InputTokensRequestV1, ApiErrorV1> {
    let map = parse_object(
        body,
        &[
            "model",
            "text",
            "messages",
            "add_generation_prompt",
            "thinking",
        ],
        &["chat_template", "custom_kwargs"],
    )?;
    let model = model(&map)?;
    let input = match (map.get("text"), map.get("messages")) {
        (Some(value), None) => InputTokensInputV1::Text(text(value, "text", false)?),
        (None, Some(value)) => InputTokensInputV1::Messages(parse_messages(value)?),
        _ => {
            return Err(invalid(
                "input",
                "exactly one of text or messages is required",
            ));
        }
    };
    let add_generation_prompt = opt_bool(&map, "add_generation_prompt")?.unwrap_or(true);
    let thinking = opt_bool(&map, "thinking")?.unwrap_or(false);
    Ok(InputTokensRequestV1 {
        model,
        input,
        add_generation_prompt,
        thinking,
    })
}

pub fn parse_infill_request(body: &[u8]) -> Result<InfillRequestV1, ApiErrorV1> {
    let map = parse_object(
        body,
        &[
            "model",
            "prefix",
            "suffix",
            "prompt",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "seed",
            "stream",
            "n",
        ],
        &[
            "input_prefix",
            "input_suffix",
            "input_extra",
            "messages",
            "tools",
        ],
    )?;
    let model = model(&map)?;
    let prefix = text(required(&map, "prefix")?, "prefix", false)?;
    let suffix = text(required(&map, "suffix")?, "suffix", false)?;
    let prompt = match map.get("prompt") {
        None | Some(Value::Null) => None,
        Some(v) => Some(text(v, "prompt", false)?),
    };
    let max_tokens = opt_u32(&map, "max_tokens")?.unwrap_or(DEFAULT_COMPLETION_TOKENS);
    if !(1..=MAX_COMPLETION_TOKENS).contains(&max_tokens) {
        return Err(invalid("max_tokens", "must be in [1,4096]"));
    }
    let temperature = opt_f32(&map, "temperature")?.unwrap_or(1.0);
    bounded_float("temperature", temperature, 0.0, 2.0)?;
    let top_p = opt_f32(&map, "top_p")?.unwrap_or(1.0);
    bounded_float("top_p", top_p, 0.0, 1.0)?;
    let stop = parse_stop(&map)?;
    let seed = opt_i64(&map, "seed")?;
    let stream = opt_bool(&map, "stream")?.unwrap_or(true);
    let n = opt_u32(&map, "n")?.unwrap_or(1);
    if !(1..=8).contains(&n) {
        return Err(invalid("n", "must be in [1,8]"));
    }
    Ok(InfillRequestV1 {
        model,
        prefix,
        suffix,
        prompt,
        max_tokens,
        temperature,
        top_p,
        stop,
        seed,
        stream,
        n,
    })
}

fn parse_object(
    body: &[u8],
    supported: &[&str],
    known_unsupported: &[&str],
) -> Result<Map<String, Value>, ApiErrorV1> {
    if body.len() > MAX_REQUEST_BODY_BYTES {
        return Err(ApiErrorV1::request_too_large());
    }
    let value = serde_json::from_slice::<StrictValue>(body)
        .map_err(|error| ApiErrorV1::invalid_json(error.to_string(), None))?
        .0;
    let object = value
        .as_object()
        .ok_or_else(|| ApiErrorV1::invalid_json("request body must be a JSON object", None))?;
    let supported = supported.iter().copied().collect::<BTreeSet<_>>();
    let unsupported = known_unsupported.iter().copied().collect::<BTreeSet<_>>();
    for field in object.keys() {
        if !supported.contains(field.as_str()) {
            return if unsupported.contains(field.as_str()) {
                Err(ApiErrorV1::unsupported(field.clone()))
            } else {
                Err(ApiErrorV1::invalid_value(
                    field.clone(),
                    format!("unknown request field {field}"),
                ))
            };
        }
    }
    Ok(object.clone())
}

fn required<'a>(map: &'a Map<String, Value>, param: &str) -> Result<&'a Value, ApiErrorV1> {
    map.get(param)
        .ok_or_else(|| invalid(param, "field is required"))
}

fn model(map: &Map<String, Value>) -> Result<String, ApiErrorV1> {
    let value = text(required(map, "model")?, "model", true)?;
    if value.len() > MAX_MODEL_ALIAS_BYTES {
        return Err(invalid("model", "must contain at most 256 UTF-8 bytes"));
    }
    Ok(value)
}

fn text(value: &Value, param: &str, nonempty: bool) -> Result<String, ApiErrorV1> {
    let value = value
        .as_str()
        .ok_or_else(|| invalid(param, "must be a string"))?;
    if nonempty && value.is_empty() {
        return Err(invalid(param, "must be nonempty"));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(invalid(param, "text exceeds 16 MiB"));
    }
    Ok(value.to_owned())
}

fn parse_prompt(value: &Value, param: &str) -> Result<PromptV1, ApiErrorV1> {
    match value {
        Value::String(_) => Ok(PromptV1::Text(text(value, param, false)?)),
        Value::Array(values) => {
            if values.is_empty() || values.len() > MAX_INPUT_ITEMS {
                return Err(invalid(param, "array must contain 1..=2048 entries"));
            }
            if values.iter().all(Value::is_string) {
                let mut strings = Vec::with_capacity(values.len());
                let mut bytes = 0usize;
                for (i, item) in values.iter().enumerate() {
                    let value = text(item, &format!("{param}[{i}]"), false)?;
                    bytes = bytes
                        .checked_add(value.len())
                        .ok_or_else(|| invalid(param, "text byte count overflowed"))?;
                    if bytes > MAX_TEXT_BYTES {
                        return Err(invalid(param, "text input exceeds 16 MiB"));
                    }
                    strings.push(value);
                }
                return Ok(PromptV1::Texts(strings));
            }
            if values.iter().all(Value::is_array) {
                let mut sequences = Vec::with_capacity(values.len());
                for (i, item) in values.iter().enumerate() {
                    sequences.push(parse_tokens(item, &format!("{param}[{i}]"), false)?);
                }
                return Ok(PromptV1::TokenSequences(sequences));
            }
            if values.iter().all(Value::is_u64) {
                return Ok(PromptV1::Tokens(parse_tokens(value, param, false)?));
            }
            Err(invalid(
                param,
                "must be a string, string array, token array, or token-array array",
            ))
        }
        _ => Err(invalid(
            param,
            "must be a string or one of four supported array shapes",
        )),
    }
}

fn parse_tokens(value: &Value, param: &str, allow_empty: bool) -> Result<Vec<u32>, ApiErrorV1> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid(param, "must be an array of unsigned 32-bit token IDs"))?;
    if !allow_empty && values.is_empty() {
        return Err(invalid(param, "must contain at least one token"));
    }
    if values.len() > MAX_TOKEN_COUNT {
        return Err(invalid(param, "token count exceeds 1048576"));
    }
    let mut tokens = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let raw = item.as_u64().ok_or_else(|| {
            invalid(
                format!("{param}[{index}]"),
                "must be an unsigned 32-bit token ID",
            )
        })?;
        if raw > u64::from(u32::MAX) {
            return Err(invalid(
                format!("{param}[{index}]"),
                "must be an unsigned 32-bit token ID",
            ));
        }
        tokens.push(raw as u32);
    }
    Ok(tokens)
}

fn parse_stop(map: &Map<String, Value>) -> Result<Vec<String>, ApiErrorV1> {
    let Some(value) = map.get("stop") else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let mut values = match value {
        Value::String(_) => vec![text(value, "stop", true)?],
        Value::Array(values) => {
            if values.is_empty() || values.len() > MAX_STOP_SEQUENCES {
                return Err(invalid("stop", "must contain 1..=4 strings"));
            }
            values
                .iter()
                .enumerate()
                .map(|(i, v)| text(v, &format!("stop[{i}]"), true))
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => {
            return Err(invalid(
                "stop",
                "must be a string, array of strings, or null",
            ));
        }
    };
    let mut seen = BTreeSet::new();
    let mut bytes = 0usize;
    for item in &values {
        bytes = bytes
            .checked_add(item.len())
            .ok_or_else(|| invalid("stop", "byte count overflowed"))?;
        if bytes > MAX_STOP_BYTES {
            return Err(invalid("stop", "stop strings exceed 16 KiB"));
        }
        if !seen.insert(item) {
            return Err(invalid("stop", "stop strings must be unique"));
        }
    }
    Ok(std::mem::take(&mut values))
}

fn parse_logit_bias(map: &Map<String, Value>) -> Result<BTreeMap<u32, f32>, ApiErrorV1> {
    let Some(value) = map.get("logit_bias") else {
        return Ok(BTreeMap::new());
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid("logit_bias", "must be an object"))?;
    if object.len() > MAX_LOGIT_BIAS_ENTRIES {
        return Err(invalid("logit_bias", "must contain at most 4096 entries"));
    }
    let mut result = BTreeMap::new();
    for (raw_id, value) in object {
        let id = raw_id
            .parse::<u64>()
            .ok()
            .filter(|id| *id <= u64::from(u32::MAX))
            .ok_or_else(|| {
                invalid(
                    format!("logit_bias.{raw_id}"),
                    "key must be an unsigned 32-bit token ID",
                )
            })?;
        let bias = as_f32(value, &format!("logit_bias.{raw_id}"))?;
        bounded_float(&format!("logit_bias.{raw_id}"), bias, -100.0, 100.0)?;
        result.insert(id as u32, bias);
    }
    Ok(result)
}

fn parse_messages(value: &Value) -> Result<Vec<TemplateMessageV1>, ApiErrorV1> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid("messages", "must be an array"))?;
    if values.is_empty() || values.len() > 1_024 {
        return Err(invalid("messages", "must contain 1..=1024 entries"));
    }
    let mut messages = Vec::with_capacity(values.len());
    for (index, item) in values.iter().enumerate() {
        let object = item
            .as_object()
            .ok_or_else(|| invalid(format!("messages[{index}]"), "must be an object"))?;
        for field in object.keys() {
            if field != "role" && field != "content" {
                return Err(invalid(
                    format!("messages[{index}].{field}"),
                    format!("unknown message field {field}"),
                ));
            }
        }
        let role = match text(
            object
                .get("role")
                .ok_or_else(|| invalid(format!("messages[{index}].role"), "field is required"))?,
            &format!("messages[{index}].role"),
            true,
        )?
        .as_str()
        {
            "system" => TemplateRoleV1::System,
            "user" => TemplateRoleV1::User,
            "assistant" => TemplateRoleV1::Assistant,
            "developer" | "tool" | "function" => {
                return Err(ApiErrorV1::unsupported(format!("messages[{index}].role")));
            }
            _ => {
                return Err(invalid(
                    format!("messages[{index}].role"),
                    "role must be system, user, or assistant",
                ));
            }
        };
        let content = text(
            object.get("content").ok_or_else(|| {
                invalid(format!("messages[{index}].content"), "field is required")
            })?,
            &format!("messages[{index}].content"),
            false,
        )?;
        messages.push(TemplateMessageV1 { role, content });
    }
    Ok(messages)
}

fn as_u64(value: &Value, param: &str) -> Result<u64, ApiErrorV1> {
    value
        .as_u64()
        .ok_or_else(|| invalid(param, "must be an unsigned integer"))
}

fn as_f32(value: &Value, param: &str) -> Result<f32, ApiErrorV1> {
    let number = value
        .as_f64()
        .ok_or_else(|| invalid(param, "must be a finite number"))?;
    if !number.is_finite() {
        return Err(invalid(param, "must be a finite number"));
    }
    let value = number as f32;
    if !value.is_finite() {
        return Err(invalid(param, "must be a finite number"));
    }
    Ok(value)
}

fn bounded_float(param: &str, value: f32, min: f32, max: f32) -> Result<(), ApiErrorV1> {
    if !value.is_finite() || !(min..=max).contains(&value) {
        return Err(invalid(
            param,
            format!("must be finite and in [{min},{max}]"),
        ));
    }
    Ok(())
}

fn opt_u32(map: &Map<String, Value>, param: &str) -> Result<Option<u32>, ApiErrorV1> {
    match map.get(param) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let value = as_u64(value, param)?;
            if value > u64::from(u32::MAX) {
                return Err(invalid(param, "must be an unsigned 32-bit integer"));
            }
            Ok(Some(value as u32))
        }
    }
}

fn opt_i64(map: &Map<String, Value>, param: &str) -> Result<Option<i64>, ApiErrorV1> {
    match map.get(param) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_i64()
            .ok_or_else(|| invalid(param, "must be a signed 64-bit integer"))
            .map(Some),
        Some(_) => Err(invalid(param, "must be a signed 64-bit integer")),
    }
}

fn opt_f32(map: &Map<String, Value>, param: &str) -> Result<Option<f32>, ApiErrorV1> {
    match map.get(param) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => as_f32(value, param).map(Some),
    }
}

fn opt_bool(map: &Map<String, Value>, param: &str) -> Result<Option<bool>, ApiErrorV1> {
    match map.get(param) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid(param, "must be a boolean"))
            .map(Some),
    }
}

fn invalid(param: impl Into<String>, message: impl Into<String>) -> ApiErrorV1 {
    ApiErrorV1::invalid_value(param, message)
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
