use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::{DeserializeOwned, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sllm_core::{CompiledGrammar, SamplingParametersV1};
use sllm_frontend::{
    BoundedImageBytesV1, GenerationConfigV1, MAX_TOTAL_VISUAL_TOKENS_V1, ProcessedVisionInputV1,
    Qwen35ChatMessageV1, Qwen35VisionProcessorV1, ThinkingModeV1,
};

use crate::phase42_api::{CompletionRequestV1, InfillRequestV1};

pub const MAX_REQUEST_BODY_BYTES: usize = 96 * 1024 * 1024;
pub const DEFAULT_MAX_COMPLETION_TOKENS: u32 = 256;
pub const MAX_COMPLETION_TOKENS: u32 = 4_096;
const MAX_MODEL_ALIAS_BYTES: usize = 256;
const MAX_MESSAGES: usize = 1_024;
const MAX_LOGIT_BIAS_ENTRIES: usize = 4_096;
const MAX_SAMPLER_TOP_K: u32 = 1_000_000;
const MAX_SAMPLER_HISTORY: u32 = 4_096;
const MAX_SAMPLER_SEQUENCE_BREAKERS: usize = 16;
const MAX_SAMPLER_SEQUENCE_BREAKER_BYTES: usize = 1_024;
const MAX_SCHEMA_NAME_BYTES: usize = 256;
const MAX_SCHEMA_DESCRIPTION_BYTES: usize = 4_096;
const MAX_SCHEMA_BYTES: usize = 64 * 1024;

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
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "sllm",
];

const KNOWN_UNSUPPORTED_FIELDS: &[&str] = &[
    "audio",
    "function_call",
    "functions",
    "max_tokens",
    "metadata",
    "modalities",
    "parallel_tool_calls",
    "prediction",
    "reasoning_effort",
    "service_tier",
    "store",
    "stream_options",
    "tool_choice",
    "tools",
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

/// A validated sparse logit-bias table.  The map is ordered by token ID so
/// every backend adapter observes a deterministic transport-independent view.
#[derive(Clone, Debug, PartialEq)]
pub struct LogitBiasV1 {
    entries: BTreeMap<u32, f32>,
}

impl LogitBiasV1 {
    pub fn entries(&self) -> &BTreeMap<u32, f32> {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrySamplingConfigV1 {
    multiplier: f32,
    base: f32,
    allowed_length: u32,
    penalty_last_n: u32,
    sequence_breakers: Vec<String>,
}

impl DrySamplingConfigV1 {
    pub const fn multiplier(&self) -> f32 {
        self.multiplier
    }

    pub const fn base(&self) -> f32 {
        self.base
    }

    pub const fn allowed_length(&self) -> u32 {
        self.allowed_length
    }

    pub const fn penalty_last_n(&self) -> u32 {
        self.penalty_last_n
    }

    pub fn sequence_breakers(&self) -> &[String] {
        &self.sequence_breakers
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XtcSamplingConfigV1 {
    probability: f32,
    threshold: f32,
    min_keep: u32,
}

impl XtcSamplingConfigV1 {
    pub const fn probability(self) -> f32 {
        self.probability
    }

    pub const fn threshold(self) -> f32 {
        self.threshold
    }

    pub const fn min_keep(self) -> u32 {
        self.min_keep
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirostatSamplingConfigV1 {
    version: u8,
    tau: f32,
    eta: f32,
}

impl MirostatSamplingConfigV1 {
    pub const fn version(self) -> u8 {
        self.version
    }

    pub const fn tau(self) -> f32 {
        self.tau
    }

    pub const fn eta(self) -> f32 {
        self.eta
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicTemperatureConfigV1 {
    range: f32,
    exponent: f32,
}

impl DynamicTemperatureConfigV1 {
    pub const fn range(self) -> f32 {
        self.range
    }

    pub const fn exponent(self) -> f32 {
        self.exponent
    }
}

/// Optional sLLM sampler controls.  All fields are disabled when absent;
/// `chain_version` fixes the backend-neutral stage order without accepting an
/// arbitrary client-defined sampler sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerExtensionConfigV1 {
    chain_version: u8,
    top_k: Option<u32>,
    min_p: Option<f32>,
    typical_p: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: u32,
    ignore_eos: bool,
    dry: Option<DrySamplingConfigV1>,
    xtc: Option<XtcSamplingConfigV1>,
    mirostat: Option<MirostatSamplingConfigV1>,
    dynamic_temperature: Option<DynamicTemperatureConfigV1>,
}

impl SamplerExtensionConfigV1 {
    pub const fn chain_version(&self) -> u8 {
        self.chain_version
    }

    pub const fn top_k(&self) -> Option<u32> {
        self.top_k
    }

    pub const fn min_p(&self) -> Option<f32> {
        self.min_p
    }

    pub const fn typical_p(&self) -> Option<f32> {
        self.typical_p
    }

    pub const fn repeat_penalty(&self) -> Option<f32> {
        self.repeat_penalty
    }

    pub const fn repeat_last_n(&self) -> u32 {
        self.repeat_last_n
    }

    pub const fn ignore_eos(&self) -> bool {
        self.ignore_eos
    }

    pub fn dry(&self) -> Option<&DrySamplingConfigV1> {
        self.dry.as_ref()
    }

    pub const fn xtc(&self) -> Option<XtcSamplingConfigV1> {
        self.xtc
    }

    pub const fn mirostat(&self) -> Option<MirostatSamplingConfigV1> {
        self.mirostat
    }

    pub const fn dynamic_temperature(&self) -> Option<DynamicTemperatureConfigV1> {
        self.dynamic_temperature
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct JsonSchemaFormatV1 {
    name: String,
    description: Option<String>,
    schema: Value,
    strict: Option<bool>,
}

impl JsonSchemaFormatV1 {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub const fn schema(&self) -> &Value {
        &self.schema
    }

    pub const fn strict(&self) -> Option<bool> {
        self.strict
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResponseFormatV1 {
    Text,
    JsonObject,
    JsonSchema(JsonSchemaFormatV1),
}

/// Validated logprob request controls.  `top_logprobs` is retained as an
/// option so omitted and explicitly-zero requests remain distinguishable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogprobOptionsV1 {
    enabled: bool,
    top_logprobs: Option<u8>,
}

impl LogprobOptionsV1 {
    pub const fn enabled(self) -> bool {
        self.enabled
    }

    pub const fn top_logprobs(self) -> Option<u8> {
        self.top_logprobs
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatCompletionRequestV1 {
    model: String,
    messages: Vec<ChatMessageV1>,
    input: GenerationRequestInputV1,
    generation: GenerationConfigV1,
    seed: Option<i64>,
    stream: bool,
    reasoning: ReasoningOptionsV1,
    resumable: bool,
    choice_count: u32,
    logit_bias: Option<LogitBiasV1>,
    logprobs: Option<LogprobOptionsV1>,
    response_format: Option<ResponseFormatV1>,
    sampler: Option<SamplerExtensionConfigV1>,
}

/// Backend-neutral input carried by the shared generation scheduler.
///
/// Chat remains the public default. Phase 42 raw completion and FIM requests
/// use explicit variants so neither wire contract is disguised as a chat
/// message and production backends can fail closed on unsupported FIM locks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GenerationRequestInputV1 {
    Chat,
    RawText(String),
    TokenIds(Vec<u32>),
    Infill {
        token_ids: Vec<u32>,
        template_digest: String,
    },
}

impl ChatCompletionRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn messages(&self) -> &[ChatMessageV1] {
        &self.messages
    }

    pub(crate) const fn input(&self) -> &GenerationRequestInputV1 {
        &self.input
    }

    /// Returns the capability-rendered FIM token sequence and bound template
    /// digest for backends that advertise an infill capability.
    pub fn prepared_infill(&self) -> Option<(&[u32], &str)> {
        match &self.input {
            GenerationRequestInputV1::Infill {
                token_ids,
                template_digest,
            } => Some((token_ids, template_digest)),
            _ => None,
        }
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

    pub const fn choice_count(&self) -> u32 {
        self.choice_count
    }

    pub const fn n(&self) -> u32 {
        self.choice_count
    }

    pub const fn logit_bias(&self) -> Option<&LogitBiasV1> {
        self.logit_bias.as_ref()
    }

    pub const fn logprobs(&self) -> Option<LogprobOptionsV1> {
        self.logprobs
    }

    pub const fn response_format(&self) -> Option<&ResponseFormatV1> {
        self.response_format.as_ref()
    }

    pub const fn sampler(&self) -> Option<&SamplerExtensionConfigV1> {
        self.sampler.as_ref()
    }

    /// Clone a validated request for one independent choice.  Choice zero is
    /// deliberately byte-for-byte compatible with the original seed; later
    /// choices receive a versioned deterministic derivation when a seed was
    /// provided.  Generation/KV ownership is created by the transport-neutral
    /// frontend, so this method only changes request-local choice metadata.
    pub fn for_choice(&self, index: u32) -> Result<Self, ApiErrorV1> {
        if index >= self.choice_count {
            return Err(ApiErrorV1::invalid_value(
                "n",
                format!("choice index {index} is outside n={}", self.choice_count),
            ));
        }
        let mut request = self.clone();
        request.choice_count = 1;
        if index != 0 {
            request.seed = derive_choice_seed_v1(self.seed, index);
        }
        Ok(request)
    }

    pub(crate) fn from_completion(
        request: &CompletionRequestV1,
        input: GenerationRequestInputV1,
    ) -> Result<Self, ApiErrorV1> {
        if matches!(
            input,
            GenerationRequestInputV1::Chat | GenerationRequestInputV1::Infill { .. }
        ) {
            return Err(ApiErrorV1::invalid_value(
                "prompt",
                "completion input must be raw text or token IDs",
            ));
        }
        let sampling = SamplingParametersV1::new(
            request.temperature(),
            request.top_p(),
            request.presence_penalty(),
            request.frequency_penalty(),
        )
        .map_err(|error| ApiErrorV1::invalid_value("sampling", error.to_string()))?;
        let generation =
            GenerationConfigV1::new(request.max_tokens(), sampling, request.stop().to_vec())
                .map_err(|error| ApiErrorV1::invalid_value("stop", error.to_string()))?;
        let logit_bias = (!request.logit_bias().is_empty()).then(|| LogitBiasV1 {
            entries: request.logit_bias().clone(),
        });
        let logprobs = request.logprobs().map(|top_logprobs| LogprobOptionsV1 {
            enabled: true,
            top_logprobs: Some(top_logprobs),
        });
        Ok(Self {
            model: request.model().to_owned(),
            messages: Vec::new(),
            input,
            generation,
            seed: request.seed(),
            stream: request.stream(),
            reasoning: ReasoningOptionsV1::disabled(),
            resumable: false,
            choice_count: request.n(),
            logit_bias,
            logprobs,
            response_format: None,
            sampler: None,
        })
    }

    pub(crate) fn from_infill(
        request: &InfillRequestV1,
        token_ids: Vec<u32>,
        template_digest: String,
    ) -> Result<Self, ApiErrorV1> {
        if token_ids.is_empty() || template_digest.is_empty() {
            return Err(ApiErrorV1::invalid_value(
                "prefix",
                "infill requires a rendered FIM token sequence and template identity",
            ));
        }
        let sampling = SamplingParametersV1::new(request.temperature(), request.top_p(), 0.0, 0.0)
            .map_err(|error| ApiErrorV1::invalid_value("sampling", error.to_string()))?;
        let generation =
            GenerationConfigV1::new(request.max_tokens(), sampling, request.stop().to_vec())
                .map_err(|error| ApiErrorV1::invalid_value("stop", error.to_string()))?;
        Ok(Self {
            model: request.model().to_owned(),
            messages: Vec::new(),
            input: GenerationRequestInputV1::Infill {
                token_ids,
                template_digest,
            },
            generation,
            seed: request.seed(),
            stream: request.stream(),
            reasoning: ReasoningOptionsV1::disabled(),
            resumable: false,
            choice_count: request.n(),
            logit_bias: None,
            logprobs: None,
            response_format: None,
            sampler: None,
        })
    }
}

/// Versioned, deterministic choice-seed derivation used by the API adapter.
/// An absent seed remains absent so the runtime's existing entropy source is
/// preserved for callers that did not request reproducibility.
pub fn derive_choice_seed_v1(seed: Option<i64>, index: u32) -> Option<i64> {
    let seed = seed?;
    if index == 0 {
        return Some(seed);
    }
    // Keep this bit-for-bit aligned with the frontend's public helper. The
    // signed wire seed is only a transport representation of the same u64
    // stream state.
    let mut state = u64::from_ne_bytes(seed.to_ne_bytes())
        ^ u64::from(index).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    Some(i64::from_ne_bytes((state ^ (state >> 31)).to_ne_bytes()))
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
    logit_bias: Option<BTreeMap<String, f32>>,
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
    response_format: Option<WireResponseFormat>,
    sllm: Option<WireSllmOptions>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSllmOptions {
    thinking: Option<WireThinkingMode>,
    separate_reasoning: Option<bool>,
    resumable: Option<bool>,
    sampling: Option<WireSamplerExtensionOptions>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSamplerExtensionOptions {
    chain_version: Option<u8>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    typical_p: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<u32>,
    ignore_eos: Option<bool>,
    dry: Option<WireDrySamplingOptions>,
    xtc: Option<WireXtcSamplingOptions>,
    mirostat: Option<WireMirostatSamplingOptions>,
    dynamic_temperature: Option<WireDynamicTemperatureOptions>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDrySamplingOptions {
    multiplier: Option<f32>,
    base: Option<f32>,
    allowed_length: Option<u32>,
    penalty_last_n: Option<u32>,
    sequence_breakers: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireXtcSamplingOptions {
    probability: Option<f32>,
    threshold: Option<f32>,
    min_keep: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireMirostatSamplingOptions {
    version: Option<u8>,
    tau: Option<f32>,
    eta: Option<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDynamicTemperatureOptions {
    range: Option<f32>,
    exponent: Option<f32>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum WireResponseFormat {
    #[serde(rename = "text")]
    Text(WireEmptyObject),
    #[serde(rename = "json_object")]
    JsonObject(WireEmptyObject),
    #[serde(rename = "json_schema")]
    JsonSchema(WireJsonSchemaFormat),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireEmptyObject {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireJsonSchemaFormat {
    json_schema: WireJsonSchema,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireJsonSchema {
    name: String,
    description: Option<String>,
    schema: Value,
    strict: Option<bool>,
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

fn parse_logit_bias(
    wire: Option<BTreeMap<String, f32>>,
) -> Result<Option<LogitBiasV1>, ApiErrorV1> {
    let Some(entries) = wire else {
        return Ok(None);
    };
    if entries.len() > MAX_LOGIT_BIAS_ENTRIES {
        return Err(ApiErrorV1::invalid_value(
            "logit_bias",
            format!("logit_bias must contain at most {MAX_LOGIT_BIAS_ENTRIES} entries"),
        ));
    }
    let mut validated = BTreeMap::new();
    for (raw_token_id, bias) in entries {
        let token_id = raw_token_id
            .parse::<u64>()
            .ok()
            .and_then(|value| (value <= u64::from(u32::MAX)).then_some(value as u32));
        let Some(token_id) = token_id else {
            return Err(ApiErrorV1::invalid_value(
                format!("logit_bias.{raw_token_id}"),
                "logit_bias keys must be unsigned 32-bit token IDs",
            ));
        };
        if !bias.is_finite() || !(-100.0..=100.0).contains(&bias) {
            return Err(ApiErrorV1::invalid_value(
                format!("logit_bias.{raw_token_id}"),
                "logit bias must be finite and in [-100,100]",
            ));
        }
        validated.insert(token_id, bias);
    }
    Ok(Some(LogitBiasV1 { entries: validated }))
}

fn parse_response_format(
    wire: Option<WireResponseFormat>,
) -> Result<Option<ResponseFormatV1>, ApiErrorV1> {
    let Some(wire) = wire else {
        return Ok(None);
    };
    let format = match wire {
        WireResponseFormat::Text(_) => ResponseFormatV1::Text,
        WireResponseFormat::JsonObject(_) => ResponseFormatV1::JsonObject,
        WireResponseFormat::JsonSchema(WireJsonSchemaFormat { json_schema }) => {
            if json_schema.name.is_empty() || json_schema.name.len() > MAX_SCHEMA_NAME_BYTES {
                return Err(ApiErrorV1::invalid_value(
                    "response_format.json_schema.name",
                    format!("schema name must contain 1..={MAX_SCHEMA_NAME_BYTES} bytes"),
                ));
            }
            if let Some(description) = &json_schema.description {
                if description.len() > MAX_SCHEMA_DESCRIPTION_BYTES {
                    return Err(ApiErrorV1::invalid_value(
                        "response_format.json_schema.description",
                        format!(
                            "schema description must contain at most {MAX_SCHEMA_DESCRIPTION_BYTES} bytes"
                        ),
                    ));
                }
            }
            if !json_schema.schema.is_object() {
                return Err(ApiErrorV1::invalid_value(
                    "response_format.json_schema.schema",
                    "schema must be a JSON object",
                ));
            }
            let schema_bytes = serde_json::to_vec(&json_schema.schema).map_err(|error| {
                ApiErrorV1::invalid_value(
                    "response_format.json_schema.schema",
                    format!("schema is not serializable: {error}"),
                )
            })?;
            if schema_bytes.len() > MAX_SCHEMA_BYTES {
                return Err(ApiErrorV1::invalid_value(
                    "response_format.json_schema.schema",
                    format!("schema must be at most {MAX_SCHEMA_BYTES} bytes"),
                ));
            }
            CompiledGrammar::from_json_schema(&json_schema.schema).map_err(|error| {
                ApiErrorV1::invalid_value(
                    "response_format.json_schema.schema",
                    format!("unsupported JSON Schema: {error}"),
                )
            })?;
            ResponseFormatV1::JsonSchema(JsonSchemaFormatV1 {
                name: json_schema.name,
                description: json_schema.description,
                schema: json_schema.schema,
                strict: json_schema.strict,
            })
        }
    };
    Ok(Some(format))
}

fn parse_sampler_extension(
    wire: Option<WireSamplerExtensionOptions>,
) -> Result<Option<SamplerExtensionConfigV1>, ApiErrorV1> {
    let Some(wire) = wire else {
        return Ok(None);
    };
    let chain_version = wire.chain_version.unwrap_or(1);
    if chain_version != 1 {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.chain_version",
            "only sampler chain version 1 is supported",
        ));
    }
    if let Some(top_k) = wire.top_k {
        if top_k > MAX_SAMPLER_TOP_K {
            return Err(ApiErrorV1::invalid_value(
                "sllm.sampling.top_k",
                format!("top_k must be in [0,{MAX_SAMPLER_TOP_K}]"),
            ));
        }
    }
    if let Some(min_p) = wire.min_p {
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err(ApiErrorV1::invalid_value(
                "sllm.sampling.min_p",
                "min_p must be finite and in [0,1]",
            ));
        }
    }
    if let Some(typical_p) = wire.typical_p {
        if !typical_p.is_finite() || !(0.0..=1.0).contains(&typical_p) || typical_p == 0.0 {
            return Err(ApiErrorV1::invalid_value(
                "sllm.sampling.typical_p",
                "typical_p must be finite and in (0,1]",
            ));
        }
    }
    if let Some(repeat_penalty) = wire.repeat_penalty {
        if !repeat_penalty.is_finite()
            || !(0.0..=100.0).contains(&repeat_penalty)
            || repeat_penalty == 0.0
        {
            return Err(ApiErrorV1::invalid_value(
                "sllm.sampling.repeat_penalty",
                "repeat_penalty must be finite and in (0,100]",
            ));
        }
    }
    let repeat_last_n = wire.repeat_last_n.unwrap_or(64);
    if repeat_last_n > MAX_SAMPLER_HISTORY {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.repeat_last_n",
            format!("repeat_last_n must be in [0,{MAX_SAMPLER_HISTORY}]"),
        ));
    }

    let dry = wire.dry.map(parse_dry_sampling).transpose()?;
    let xtc = wire.xtc.map(parse_xtc_sampling).transpose()?;
    let mirostat = wire.mirostat.map(parse_mirostat_sampling).transpose()?;
    let dynamic_temperature = wire
        .dynamic_temperature
        .map(parse_dynamic_temperature)
        .transpose()?;

    if mirostat.is_some()
        && (wire.top_k.is_some()
            || wire.min_p.is_some()
            || wire.typical_p.is_some()
            || xtc.is_some()
            || dynamic_temperature.is_some())
    {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.mirostat",
            "mirostat cannot be combined with top_k, min_p, typical_p, xtc, or dynamic_temperature",
        ));
    }

    Ok(Some(SamplerExtensionConfigV1 {
        chain_version,
        top_k: wire.top_k,
        min_p: wire.min_p,
        typical_p: wire.typical_p,
        repeat_penalty: wire.repeat_penalty,
        repeat_last_n,
        ignore_eos: wire.ignore_eos.unwrap_or(false),
        dry,
        xtc,
        mirostat,
        dynamic_temperature,
    }))
}

fn parse_dry_sampling(wire: WireDrySamplingOptions) -> Result<DrySamplingConfigV1, ApiErrorV1> {
    let multiplier = wire.multiplier.unwrap_or(0.0);
    if !multiplier.is_finite() || !(0.0..=100.0).contains(&multiplier) {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.dry.multiplier",
            "DRY multiplier must be finite and in [0,100]",
        ));
    }
    let base = wire.base.unwrap_or(1.75);
    if !base.is_finite() || !(1.0..=4.0).contains(&base) {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.dry.base",
            "DRY base must be finite and in [1,4]",
        ));
    }
    let allowed_length = wire.allowed_length.unwrap_or(2);
    if allowed_length > MAX_SAMPLER_HISTORY {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.dry.allowed_length",
            format!("DRY allowed_length must be in [0,{MAX_SAMPLER_HISTORY}]"),
        ));
    }
    let penalty_last_n = wire.penalty_last_n.unwrap_or(64);
    if penalty_last_n > MAX_SAMPLER_HISTORY {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.dry.penalty_last_n",
            format!("DRY penalty_last_n must be in [0,{MAX_SAMPLER_HISTORY}]"),
        ));
    }
    let sequence_breakers = wire.sequence_breakers.unwrap_or_else(|| {
        ["\n", ":", "\"", "*"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    });
    if sequence_breakers.len() > MAX_SAMPLER_SEQUENCE_BREAKERS {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.dry.sequence_breakers",
            format!(
                "DRY sequence_breakers must contain at most {MAX_SAMPLER_SEQUENCE_BREAKERS} entries"
            ),
        ));
    }
    let mut total_bytes = 0_usize;
    let mut seen = BTreeSet::new();
    for breaker in &sequence_breakers {
        if breaker.is_empty() || breaker.len() > 128 || !seen.insert(breaker) {
            return Err(ApiErrorV1::invalid_value(
                "sllm.sampling.dry.sequence_breakers",
                "DRY sequence breakers must be nonempty, unique, and at most 128 bytes",
            ));
        }
        total_bytes = total_bytes.checked_add(breaker.len()).ok_or_else(|| {
            ApiErrorV1::invalid_value(
                "sllm.sampling.dry.sequence_breakers",
                "DRY sequence breaker bytes overflowed",
            )
        })?;
    }
    if total_bytes > MAX_SAMPLER_SEQUENCE_BREAKER_BYTES {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.dry.sequence_breakers",
            format!(
                "DRY sequence breakers must contain at most {MAX_SAMPLER_SEQUENCE_BREAKER_BYTES} bytes"
            ),
        ));
    }
    Ok(DrySamplingConfigV1 {
        multiplier,
        base,
        allowed_length,
        penalty_last_n,
        sequence_breakers,
    })
}

fn parse_xtc_sampling(wire: WireXtcSamplingOptions) -> Result<XtcSamplingConfigV1, ApiErrorV1> {
    let probability = wire.probability.unwrap_or(0.0);
    let threshold = wire.threshold.unwrap_or(0.1);
    let min_keep = wire.min_keep.unwrap_or(1);
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.xtc.probability",
            "XTC probability must be finite and in [0,1]",
        ));
    }
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.xtc.threshold",
            "XTC threshold must be finite and in [0,1]",
        ));
    }
    if min_keep == 0 || min_keep > MAX_SAMPLER_HISTORY {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.xtc.min_keep",
            format!("XTC min_keep must be in [1,{MAX_SAMPLER_HISTORY}]"),
        ));
    }
    Ok(XtcSamplingConfigV1 {
        probability,
        threshold,
        min_keep,
    })
}

fn parse_mirostat_sampling(
    wire: WireMirostatSamplingOptions,
) -> Result<MirostatSamplingConfigV1, ApiErrorV1> {
    let version = wire.version.unwrap_or(2);
    if !matches!(version, 1 | 2) {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.mirostat.version",
            "Mirostat version must be 1 or 2",
        ));
    }
    let tau = wire.tau.unwrap_or(5.0);
    let eta = wire.eta.unwrap_or(0.1);
    if !tau.is_finite() || !(0.0..=100.0).contains(&tau) || tau == 0.0 {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.mirostat.tau",
            "Mirostat tau must be finite and in (0,100]",
        ));
    }
    if !eta.is_finite() || !(0.0..=1.0).contains(&eta) || eta == 0.0 {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.mirostat.eta",
            "Mirostat eta must be finite and in (0,1]",
        ));
    }
    Ok(MirostatSamplingConfigV1 { version, tau, eta })
}

fn parse_dynamic_temperature(
    wire: WireDynamicTemperatureOptions,
) -> Result<DynamicTemperatureConfigV1, ApiErrorV1> {
    let range = wire.range.unwrap_or(0.0);
    let exponent = wire.exponent.unwrap_or(1.0);
    if !range.is_finite() || !(0.0..=10.0).contains(&range) {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.dynamic_temperature.range",
            "dynamic temperature range must be finite and in [0,10]",
        ));
    }
    if !exponent.is_finite() || !(0.0..=10.0).contains(&exponent) || exponent == 0.0 {
        return Err(ApiErrorV1::invalid_value(
            "sllm.sampling.dynamic_temperature.exponent",
            "dynamic temperature exponent must be finite and in (0,10]",
        ));
    }
    Ok(DynamicTemperatureConfigV1 { range, exponent })
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
    let choice_count = wire.n.unwrap_or(1);
    if !(1..=8).contains(&choice_count) {
        return Err(ApiErrorV1::invalid_value(
            "n",
            "n must be an integer in [1,8]",
        ));
    }
    let logit_bias = parse_logit_bias(wire.logit_bias)?;
    if wire.top_logprobs.is_some() && wire.logprobs != Some(true) {
        return Err(ApiErrorV1::invalid_value(
            "top_logprobs",
            "top_logprobs requires logprobs=true",
        ));
    }
    let logprobs = match (wire.logprobs, wire.top_logprobs) {
        (None, None) => None,
        (enabled, top_logprobs) => Some(LogprobOptionsV1 {
            enabled: enabled.unwrap_or(false),
            top_logprobs,
        }),
    };
    if let Some(top_logprobs) = wire.top_logprobs {
        if top_logprobs > 20 {
            return Err(ApiErrorV1::invalid_value(
                "top_logprobs",
                "top_logprobs must be in [0,20]",
            ));
        }
    }
    let response_format = parse_response_format(wire.response_format)?;

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
    if matches!(response_format, Some(ResponseFormatV1::JsonObject))
        && !messages.iter().any(message_mentions_json)
    {
        return Err(ApiErrorV1::invalid_value(
            "messages",
            "response_format=json_object requires a message to mention JSON",
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
    let (reasoning, resumable, sampler) = match wire.sllm {
        None => (ReasoningOptionsV1::disabled(), false, None),
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
                parse_sampler_extension(options.sampling)?,
            )
        }
    };
    if sampler
        .as_ref()
        .is_some_and(|extension| extension.mirostat().is_some())
        && sampling.top_p() < 1.0
    {
        return Err(ApiErrorV1::invalid_value(
            "top_p",
            "top_p must be 1 when Mirostat is enabled",
        ));
    }
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
        input: GenerationRequestInputV1::Chat,
        generation,
        seed: wire.seed,
        stream,
        reasoning,
        resumable,
        choice_count,
        logit_bias,
        logprobs,
        response_format,
        sampler,
    })
}

fn message_mentions_json(message: &ChatMessageV1) -> bool {
    let content = match message.inner() {
        Qwen35ChatMessageV1::System { content }
        | Qwen35ChatMessageV1::User { content }
        | Qwen35ChatMessageV1::Assistant { content, .. } => content,
    };
    content
        .as_bytes()
        .windows(4)
        .any(|window| window.eq_ignore_ascii_case(b"json"))
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
    fn phase40_wire_fields_are_typed_without_changing_legacy_generation() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "return json"}],
            "temperature": 0.8,
            "seed": 41,
            "logit_bias": {"0": -100, "7": 2.5, "4294967295": 100},
            "logprobs": true,
            "top_logprobs": 20,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "description": "bounded answer",
                    "schema": {"type": "object", "properties": {}, "additionalProperties": false},
                    "strict": true
                }
            },
            "n": 8,
            "sllm": {
                "sampling": {
                    "chain_version": 1,
                    "top_k": 17,
                    "min_p": 0.05,
                    "typical_p": 0.9,
                    "repeat_penalty": 1.1,
                    "ignore_eos": true,
                    "dry": {"multiplier": 0.5, "base": 1.75, "allowed_length": 2, "penalty_last_n": 64},
                    "xtc": {"probability": 0.2, "threshold": 0.1, "min_keep": 1},
                    "dynamic_temperature": {"range": 0.25, "exponent": 1.2}
                }
            }
        }))
        .unwrap();
        let request = parse_chat_completion_request(&body).unwrap();
        assert_eq!(request.choice_count(), 8);
        assert_eq!(request.logit_bias().unwrap().entries().len(), 3);
        assert_eq!(request.logit_bias().unwrap().entries()[&0], -100.0);
        assert_eq!(
            request.logprobs(),
            Some(LogprobOptionsV1 {
                enabled: true,
                top_logprobs: Some(20)
            })
        );
        let ResponseFormatV1::JsonSchema(schema) = request.response_format().unwrap() else {
            panic!("expected json schema response format")
        };
        assert_eq!(schema.name(), "answer");
        assert_eq!(schema.strict(), Some(true));
        let sampler = request.sampler().unwrap();
        assert_eq!(sampler.chain_version(), 1);
        assert_eq!(sampler.top_k(), Some(17));
        assert_eq!(sampler.repeat_last_n(), 64);
        assert!(sampler.dry().is_some());
        assert!(sampler.xtc().is_some());
        assert!(sampler.dynamic_temperature().is_some());
        assert!(sampler.mirostat().is_none());
        assert_eq!(request.generation().sampling().temperature(), 0.8);

        let choice_zero = request.for_choice(0).unwrap();
        let choice_seven = request.for_choice(7).unwrap();
        assert_eq!(choice_zero.choice_count(), 1);
        assert_eq!(choice_zero.seed(), request.seed());
        assert_eq!(choice_seven.choice_count(), 1);
        assert_ne!(choice_seven.seed(), request.seed());
        assert!(request.for_choice(8).is_err());
    }

    #[test]
    fn response_format_variants_and_omitted_fields_are_accepted() {
        for (response_format, expected) in [
            (serde_json::json!({"type": "text"}), "text"),
            (serde_json::json!({"type": "json_object"}), "json_object"),
        ] {
            let body = serde_json::to_vec(&serde_json::json!({
                "model": "qwen",
                "messages": [{"role": "user", "content": "return JSON"}],
                "response_format": response_format,
            }))
            .unwrap();
            let request = parse_chat_completion_request(&body).unwrap();
            assert_eq!(
                match request.response_format().unwrap() {
                    ResponseFormatV1::Text => "text",
                    ResponseFormatV1::JsonObject => "json_object",
                    ResponseFormatV1::JsonSchema(_) => "json_schema",
                },
                expected
            );
            assert_eq!(request.choice_count(), 1);
            assert!(request.logit_bias().is_none());
            assert!(request.logprobs().is_none());
            assert!(request.sampler().is_none());
        }
    }

    #[test]
    fn structured_formats_fail_closed_before_generation() {
        let missing_json_instruction = serde_json::to_vec(&serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "return an object"}],
            "response_format": {"type": "json_object"}
        }))
        .unwrap();
        let error = parse_chat_completion_request(&missing_json_instruction).unwrap_err();
        assert_eq!(error.param(), Some("messages"));

        let unsupported_schema = serde_json::to_vec(&serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "return JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "string",
                        "pattern": "^[a-z]+$"
                    }
                }
            }
        }))
        .unwrap();
        let error = parse_chat_completion_request(&unsupported_schema).unwrap_err();
        assert_eq!(error.param(), Some("response_format.json_schema.schema"));
    }

    #[test]
    fn phase40_wire_limits_and_conflicts_fail_closed() {
        for (body, param, code) in [
            (valid(r#", "n":0"#), "n", ErrorCodeV1::InvalidValue),
            (valid(r#", "n":9"#), "n", ErrorCodeV1::InvalidValue),
            (
                valid(r#", "top_logprobs":1"#),
                "top_logprobs",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "logprobs":false, "top_logprobs":0"#),
                "top_logprobs",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "logprobs":true, "top_logprobs":21"#),
                "top_logprobs",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "logit_bias":{"-1":1}"#),
                "logit_bias.-1",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "logit_bias":{"1":101}"#),
                "logit_bias.1",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "sllm":{"sampling":{"typical_p":0}}"#),
                "sllm.sampling.typical_p",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "sllm":{"sampling":{"repeat_penalty":0}}"#),
                "sllm.sampling.repeat_penalty",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "sllm":{"sampling":{"repeat_last_n":4097}}"#),
                "sllm.sampling.repeat_last_n",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "sllm":{"sampling":{"xtc":{"min_keep":0}}}"#),
                "sllm.sampling.xtc.min_keep",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "response_format":{"type":"text","extra":true}"#),
                "response_format",
                ErrorCodeV1::InvalidJson,
            ),
            (
                valid(
                    r#", "response_format":{"type":"json_schema","json_schema":{"name":"x","schema":[]}}"#,
                ),
                "response_format.json_schema.schema",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "sllm":{"sampling":{"chain_version":2}}"#),
                "sllm.sampling.chain_version",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "sllm":{"sampling":{"mirostat":{"version":2},"top_k":4}}"#),
                "sllm.sampling.mirostat",
                ErrorCodeV1::InvalidValue,
            ),
            (
                valid(r#", "top_p":0.9, "sllm":{"sampling":{"mirostat":{"version":2}}}"#),
                "top_p",
                ErrorCodeV1::InvalidValue,
            ),
        ] {
            let error = parse_chat_completion_request(&body).unwrap_err();
            assert_eq!(error.param(), Some(param), "unexpected error for {param}");
            assert_eq!(error.code(), code);
        }

        let too_many_biases = (0..=MAX_LOGIT_BIAS_ENTRIES)
            .map(|token| (token.to_string(), 0.0_f32))
            .collect::<BTreeMap<_, _>>();
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "logit_bias": too_many_biases,
        }))
        .unwrap();
        let error = parse_chat_completion_request(&body).unwrap_err();
        assert_eq!(error.param(), Some("logit_bias"));
        assert_eq!(error.code(), ErrorCodeV1::InvalidValue);

        let too_large_schema = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "large",
                "schema": {"description": "x".repeat(MAX_SCHEMA_BYTES)},
            },
        });
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": too_large_schema,
        }))
        .unwrap();
        let error = parse_chat_completion_request(&body).unwrap_err();
        assert_eq!(error.param(), Some("response_format.json_schema.schema"));
        assert_eq!(error.code(), ErrorCodeV1::InvalidValue);
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
                valid(r#", "mystery":1"#),
                "mystery",
                ErrorCodeV1::UnsupportedParameter,
            ),
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
