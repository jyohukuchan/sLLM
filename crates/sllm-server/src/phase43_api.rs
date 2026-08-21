//! Strict wire contracts for the Phase 43 Responses and Anthropic profiles.
//!
//! This module intentionally stops at validation and transport-independent
//! lowering data.  It never executes a tool, resolves a URL/path/credential,
//! or starts model generation.  The HTTP adapters owned by the server map the
//! validated values into the common frontend/runtime request types.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use axum::http::StatusCode;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};

pub const PHASE43_RESPONSES_PROFILE_VERSION: &str = "openai-responses-v1";
pub const PHASE43_ANTHROPIC_PROFILE_VERSION: &str = "anthropic-messages-v1";
pub const ANTHROPIC_API_VERSION_V1: &str = "2023-06-01";
pub const MAX_REQUEST_BODY_BYTES: usize = 96 * 1024 * 1024;
pub const MAX_MODEL_ALIAS_BYTES: usize = 256;
pub const MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_INPUT_ITEMS: usize = 2_048;
pub const MAX_MESSAGES: usize = 1_024;
pub const MAX_CONTENT_BLOCKS: usize = 256;
pub const MAX_TOOL_DEFINITIONS: usize = 128;
pub const MAX_TOOL_NAME_BYTES: usize = 64;
pub const MAX_TOOL_DESCRIPTION_BYTES: usize = 16 * 1024;
pub const MAX_TOOL_SCHEMA_BYTES: usize = 1024 * 1024;
pub const MAX_TOOL_ID_BYTES: usize = 256;
pub const MAX_TOOL_ARGUMENT_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_TOOL_CALLS: usize = 16;
pub const MAX_COMPLETION_TOKENS: u32 = 4_096;
pub const DEFAULT_COMPLETION_TOKENS: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase43ErrorCodeV1 {
    InvalidJson,
    InvalidValue,
    UnsupportedParameter,
    RequestTooLarge,
}

impl Phase43ErrorCodeV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidValue => "invalid_value",
            Self::UnsupportedParameter => "unsupported_parameter",
            Self::RequestTooLarge => "request_too_large",
        }
    }
}

/// API error that remains independent from the legacy Chat API error type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Phase43ApiErrorV1 {
    status: StatusCode,
    message: String,
    param: Option<String>,
    code: Phase43ErrorCodeV1,
}

impl Phase43ApiErrorV1 {
    pub fn new(
        status: StatusCode,
        message: impl Into<String>,
        param: Option<String>,
        code: Phase43ErrorCodeV1,
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
            Phase43ErrorCodeV1::InvalidJson,
        )
    }

    pub fn invalid_value(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            Some(param.into()),
            Phase43ErrorCodeV1::InvalidValue,
        )
    }

    pub fn unsupported(param: impl Into<String>) -> Self {
        let param = param.into();
        Self::new(
            StatusCode::BAD_REQUEST,
            format!("parameter {param} is not supported by profile v1"),
            Some(param),
            Phase43ErrorCodeV1::UnsupportedParameter,
        )
    }

    pub fn request_too_large() -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("request body exceeds {MAX_REQUEST_BODY_BYTES} bytes"),
            None,
            Phase43ErrorCodeV1::RequestTooLarge,
        )
    }

    pub const fn status(&self) -> StatusCode {
        self.status
    }
    pub const fn code(&self) -> Phase43ErrorCodeV1 {
        self.code
    }
    pub fn param(&self) -> Option<&str> {
        self.param.as_deref()
    }
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Phase43ApiErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Phase43ApiErrorV1 {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponsesMessageRoleV1 {
    User,
    System,
    Developer,
    Assistant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponsesTextPartKindV1 {
    InputText,
    OutputText,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponsesTextPartV1 {
    kind: ResponsesTextPartKindV1,
    text: String,
}

impl ResponsesTextPartV1 {
    pub const fn kind(&self) -> ResponsesTextPartKindV1 {
        self.kind
    }
    pub fn text(&self) -> &str {
        &self.text
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResponsesInputItemV1 {
    Message {
        role: ResponsesMessageRoleV1,
        content: Vec<ResponsesTextPartV1>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResponsesInputV1 {
    Text(String),
    Items(Vec<ResponsesInputItemV1>),
}

impl ResponsesInputV1 {
    pub const fn as_text(&self) -> Option<&String> {
        if let Self::Text(value) = self {
            Some(value)
        } else {
            None
        }
    }
    pub fn items(&self) -> Option<&[ResponsesInputItemV1]> {
        if let Self::Items(value) = self {
            Some(value)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolDefinitionV1 {
    name: String,
    description: Option<String>,
    parameters: Value,
}

impl ToolDefinitionV1 {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
    pub const fn parameters(&self) -> &Value {
        &self.parameters
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolChoiceV1 {
    Auto {
        disable_parallel_tool_use: bool,
    },
    None,
    Required {
        disable_parallel_tool_use: bool,
    },
    Specific {
        name: String,
        disable_parallel_tool_use: bool,
    },
}

impl ToolChoiceV1 {
    pub const fn allows_parallel(&self) -> bool {
        !matches!(
            self,
            Self::Auto {
                disable_parallel_tool_use: true
            } | Self::Required {
                disable_parallel_tool_use: true
            } | Self::Specific {
                disable_parallel_tool_use: true,
                ..
            }
        )
    }
    pub fn specific_name(&self) -> Option<&str> {
        if let Self::Specific { name, .. } = self {
            Some(name)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SllmExtensionsV1 {
    resumable: bool,
}

impl SllmExtensionsV1 {
    pub const fn resumable(self) -> bool {
        self.resumable
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponsesReasoningEffortV1 {
    Low,
    Medium,
    High,
}

impl ResponsesReasoningEffortV1 {
    /// Phase 44's profile-defined lowering into the shared frontend budget.
    pub const fn max_reasoning_tokens(self) -> u32 {
        match self {
            Self::Low => 1_024,
            Self::Medium => 2_048,
            Self::High => 4_096,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponsesRequestV1 {
    model: String,
    input: ResponsesInputV1,
    instructions: Option<String>,
    max_output_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: bool,
    tools: Vec<ToolDefinitionV1>,
    tool_choice: ToolChoiceV1,
    parallel_tool_calls: bool,
    reasoning_effort: Option<ResponsesReasoningEffortV1>,
    metadata: BTreeMap<String, String>,
    store: bool,
    sllm: SllmExtensionsV1,
}

impl ResponsesRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub const fn input(&self) -> &ResponsesInputV1 {
        &self.input
    }
    pub fn instructions(&self) -> Option<&str> {
        self.instructions.as_deref()
    }
    pub const fn max_output_tokens(&self) -> u32 {
        self.max_output_tokens
    }
    pub const fn temperature(&self) -> Option<f32> {
        self.temperature
    }
    pub const fn top_p(&self) -> Option<f32> {
        self.top_p
    }
    pub const fn stream(&self) -> bool {
        self.stream
    }
    pub fn tools(&self) -> &[ToolDefinitionV1] {
        &self.tools
    }
    pub const fn tool_choice(&self) -> &ToolChoiceV1 {
        &self.tool_choice
    }
    pub const fn parallel_tool_calls(&self) -> bool {
        self.parallel_tool_calls
    }
    pub const fn reasoning_effort(&self) -> Option<ResponsesReasoningEffortV1> {
        self.reasoning_effort
    }
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }
    pub const fn store(&self) -> bool {
        self.store
    }
    pub const fn sllm(&self) -> SllmExtensionsV1 {
        self.sllm
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnthropicSystemV1 {
    Text(String),
    Blocks(Vec<String>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnthropicContentBlockV1 {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnthropicRoleV1 {
    User,
    Assistant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnthropicMessageV1 {
    role: AnthropicRoleV1,
    content: Vec<AnthropicContentBlockV1>,
}

impl AnthropicMessageV1 {
    pub const fn role(&self) -> AnthropicRoleV1 {
        self.role
    }
    pub fn content(&self) -> &[AnthropicContentBlockV1] {
        &self.content
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AnthropicMessagesRequestV1 {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessageV1>,
    system: Option<AnthropicSystemV1>,
    stream: bool,
    stop_sequences: Vec<String>,
    tools: Vec<ToolDefinitionV1>,
    tool_choice: ToolChoiceV1,
    sllm: SllmExtensionsV1,
}

impl AnthropicMessagesRequestV1 {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub const fn max_tokens(&self) -> u32 {
        self.max_tokens
    }
    pub fn messages(&self) -> &[AnthropicMessageV1] {
        &self.messages
    }
    pub const fn system(&self) -> Option<&AnthropicSystemV1> {
        self.system.as_ref()
    }
    pub const fn stream(&self) -> bool {
        self.stream
    }
    pub fn stop_sequences(&self) -> &[String] {
        &self.stop_sequences
    }
    pub fn tools(&self) -> &[ToolDefinitionV1] {
        &self.tools
    }
    pub const fn tool_choice(&self) -> &ToolChoiceV1 {
        &self.tool_choice
    }
    pub const fn sllm(&self) -> SllmExtensionsV1 {
        self.sllm
    }
}

/// Validate the required Anthropic compatibility header.  Header parsing is
/// kept outside the JSON parser so HTTP adapters can reject before admission.
pub fn validate_anthropic_version_header(version: Option<&str>) -> Result<(), Phase43ApiErrorV1> {
    match version {
        Some(value) if value == ANTHROPIC_API_VERSION_V1 => Ok(()),
        Some(_) => Err(Phase43ApiErrorV1::invalid_value(
            "anthropic-version",
            format!("must equal {ANTHROPIC_API_VERSION_V1}"),
        )),
        None => Err(Phase43ApiErrorV1::invalid_value(
            "anthropic-version",
            "header is required",
        )),
    }
}

pub fn parse_responses_request_v1(body: &[u8]) -> Result<ResponsesRequestV1, Phase43ApiErrorV1> {
    let map = parse_object(
        body,
        &[
            "model",
            "input",
            "instructions",
            "max_output_tokens",
            "temperature",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "metadata",
            "store",
            "sllm",
        ],
        &[
            "background",
            "conversation",
            "include",
            "previous_response_id",
            "truncation",
            "user",
            "service_tier",
            "prompt_cache_key",
            "prompt_cache_retention",
        ],
    )?;
    let model = model(&map)?;
    let input = parse_responses_input(required(&map, "input", "request")?)?;
    let instructions = optional_text(&map, "instructions")?;
    let max_output_tokens =
        opt_u32(&map, "max_output_tokens")?.unwrap_or(DEFAULT_COMPLETION_TOKENS);
    if !(1..=MAX_COMPLETION_TOKENS).contains(&max_output_tokens) {
        return Err(invalid("max_output_tokens", "must be in [1,4096]"));
    }
    let temperature = opt_f32(&map, "temperature")?;
    if let Some(value) = temperature {
        bounded_float("temperature", value, 0.0, 2.0)?;
    }
    let top_p = opt_f32(&map, "top_p")?;
    if let Some(value) = top_p {
        bounded_float("top_p", value, 0.0, 1.0)?;
    }
    let stream = opt_bool(&map, "stream")?.unwrap_or(false);
    let tools = parse_responses_tools(map.get("tools"))?;
    let tool_choice = parse_responses_tool_choice(map.get("tool_choice"))?;
    validate_tool_choice_has_tools(&tool_choice, tools.len(), "tool_choice")?;
    let parallel_tool_calls = opt_bool(&map, "parallel_tool_calls")?.unwrap_or(true);
    let parallel_tool_calls = parallel_tool_calls && tool_choice.allows_parallel();
    if let Some(name) = tool_choice.specific_name() {
        if !tools.iter().any(|tool| tool.name() == name) {
            return Err(invalid("tool_choice.name", "must refer to a declared tool"));
        }
    }
    let reasoning_effort = parse_responses_reasoning(map.get("reasoning"))?;
    let metadata = parse_metadata(map.get("metadata"))?;
    let store = match map.get("store") {
        None => false,
        Some(Value::Bool(false)) => false,
        Some(Value::Bool(true)) => return Err(Phase43ApiErrorV1::unsupported("store")),
        Some(_) => return Err(invalid("store", "must be false")),
    };
    let sllm = parse_sllm(map.get("sllm"))?;
    if sllm.resumable() && !stream {
        return Err(invalid("sllm.resumable", "requires stream=true"));
    }
    Ok(ResponsesRequestV1 {
        model,
        input,
        instructions,
        max_output_tokens,
        temperature,
        top_p,
        stream,
        tools,
        tool_choice,
        parallel_tool_calls,
        reasoning_effort,
        metadata,
        store,
        sllm,
    })
}

pub fn parse_anthropic_request_v1(
    body: &[u8],
    version_header: Option<&str>,
) -> Result<AnthropicMessagesRequestV1, Phase43ApiErrorV1> {
    validate_anthropic_version_header(version_header)?;
    let map = parse_object(
        body,
        &[
            "model",
            "max_tokens",
            "messages",
            "system",
            "stream",
            "stop_sequences",
            "tools",
            "tool_choice",
            "sllm",
        ],
        &[
            "metadata",
            "service_tier",
            "container",
            "mcp_servers",
            "betas",
            "thinking",
            "top_k",
            "top_p",
            "temperature",
            "cache_control",
        ],
    )?;
    let model = model(&map)?;
    let max_tokens =
        opt_u32(&map, "max_tokens")?.ok_or_else(|| invalid("max_tokens", "field is required"))?;
    if !(1..=MAX_COMPLETION_TOKENS).contains(&max_tokens) {
        return Err(invalid("max_tokens", "must be in [1,4096]"));
    }
    let messages = parse_anthropic_messages(required(&map, "messages", "request")?)?;
    let system = parse_anthropic_system(map.get("system"))?;
    let stream = opt_bool(&map, "stream")?.unwrap_or(false);
    let stop_sequences = parse_stop_sequences(map.get("stop_sequences"))?;
    let tools = parse_anthropic_tools(map.get("tools"))?;
    let tool_choice = parse_anthropic_tool_choice(map.get("tool_choice"))?;
    validate_tool_choice_has_tools(&tool_choice, tools.len(), "tool_choice")?;
    if let Some(name) = tool_choice.specific_name() {
        if !tools.iter().any(|tool| tool.name() == name) {
            return Err(invalid("tool_choice.name", "must refer to a declared tool"));
        }
    }
    let sllm = parse_sllm(map.get("sllm"))?;
    if sllm.resumable() && !stream {
        return Err(invalid("sllm.resumable", "requires stream=true"));
    }
    Ok(AnthropicMessagesRequestV1 {
        model,
        max_tokens,
        messages,
        system,
        stream,
        stop_sequences,
        tools,
        tool_choice,
        sllm,
    })
}

fn parse_responses_input(value: &Value) -> Result<ResponsesInputV1, Phase43ApiErrorV1> {
    if let Value::String(text) = value {
        return bounded_text(text, "input", false).map(ResponsesInputV1::Text);
    }
    let values = value
        .as_array()
        .ok_or_else(|| invalid("input", "must be a string or array of typed items"))?;
    if values.is_empty() || values.len() > MAX_INPUT_ITEMS {
        return Err(invalid("input", "must contain 1..=2048 items"));
    }
    let mut items = Vec::with_capacity(values.len());
    let mut calls = BTreeSet::new();
    let mut results = BTreeSet::new();
    let mut call_count = 0usize;
    let mut outstanding = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        let path = format!("input[{index}]");
        let object = object(value, &path)?;
        let item_type = required_text(object, "type", &path, true)?;
        let item = match item_type.as_str() {
            "message" => {
                if !outstanding.is_empty() {
                    return Err(invalid(
                        &path,
                        "function_call_output must resolve every preceding function_call before a message",
                    ));
                }
                parse_responses_message(object, &path)?
            }
            "function_call" => {
                check_fields(
                    object,
                    &["type", "call_id", "name", "arguments"],
                    &[],
                    &path,
                )?;
                call_count += 1;
                if call_count > MAX_TOOL_CALLS {
                    return Err(invalid("input", "tool call count exceeds 16"));
                }
                let call_id = bounded_id(
                    required_text(object, "call_id", &path, true)?,
                    &format!("{path}.call_id"),
                )?;
                if !calls.insert(call_id.clone()) {
                    return Err(invalid(format!("{path}.call_id"), "duplicate call ID"));
                }
                outstanding.insert(call_id.clone());
                if outstanding.len() > MAX_TOOL_CALLS {
                    return Err(invalid(
                        format!("{path}.call_id"),
                        "at most 16 function calls may be outstanding",
                    ));
                }
                let name = tool_name(
                    required_text(object, "name", &path, true)?,
                    &format!("{path}.name"),
                )?;
                let arguments = required_text(object, "arguments", &path, false)?;
                validate_arguments_json(&arguments, &format!("{path}.arguments"))?;
                ResponsesInputItemV1::FunctionCall {
                    call_id,
                    name,
                    arguments,
                }
            }
            "function_call_output" => {
                check_fields(object, &["type", "call_id", "output"], &[], &path)?;
                let call_id = bounded_id(
                    required_text(object, "call_id", &path, true)?,
                    &format!("{path}.call_id"),
                )?;
                if !calls.contains(&call_id) {
                    return Err(invalid(
                        format!("{path}.call_id"),
                        "must refer to a preceding function_call",
                    ));
                }
                if !results.insert(call_id.clone()) {
                    return Err(invalid(
                        format!("{path}.call_id"),
                        "duplicate function_call_output",
                    ));
                }
                if !outstanding.remove(&call_id) {
                    return Err(invalid(
                        format!("{path}.call_id"),
                        "function call was already resolved",
                    ));
                }
                let output = parse_tool_output(
                    required(object, "output", &path)?,
                    &format!("{path}.output"),
                )?;
                ResponsesInputItemV1::FunctionCallOutput { call_id, output }
            }
            "input_image" | "input_file" | "computer_call_output" | "local_shell_call" => {
                return Err(Phase43ApiErrorV1::unsupported(format!("{path}.type")));
            }
            _ => {
                return Err(invalid(
                    format!("{path}.type"),
                    "unsupported input item type",
                ));
            }
        };
        items.push(item);
    }
    if !outstanding.is_empty() {
        return Err(invalid(
            "input",
            "every function_call must have a following function_call_output",
        ));
    }
    Ok(ResponsesInputV1::Items(items))
}

fn parse_responses_message(
    map: &Map<String, Value>,
    path: &str,
) -> Result<ResponsesInputItemV1, Phase43ApiErrorV1> {
    check_fields(map, &["type", "role", "content"], &[], path)?;
    let role = match required_text(map, "role", path, true)?.as_str() {
        "user" => ResponsesMessageRoleV1::User,
        "system" => ResponsesMessageRoleV1::System,
        "developer" => ResponsesMessageRoleV1::Developer,
        "assistant" => ResponsesMessageRoleV1::Assistant,
        _ => {
            return Err(invalid(
                format!("{path}.role"),
                "must be user, system, developer, or assistant",
            ));
        }
    };
    let content_value = required(map, "content", path)?;
    let content = if let Value::String(text) = content_value {
        let kind = if matches!(role, ResponsesMessageRoleV1::Assistant) {
            ResponsesTextPartKindV1::OutputText
        } else {
            ResponsesTextPartKindV1::InputText
        };
        vec![ResponsesTextPartV1 {
            kind,
            text: bounded_text(text, &format!("{path}.content"), false)?,
        }]
    } else {
        let values = content_value
            .as_array()
            .ok_or_else(|| invalid(format!("{path}.content"), "must be a string or array"))?;
        if values.is_empty() || values.len() > MAX_CONTENT_BLOCKS {
            return Err(invalid(
                format!("{path}.content"),
                "must contain 1..=256 items",
            ));
        }
        let mut result = Vec::with_capacity(values.len());
        for (index, value) in values.iter().enumerate() {
            let item_path = format!("{path}.content[{index}]");
            let part = object(value, &item_path)?;
            let part_type = required_text(part, "type", &item_path, true)?;
            let expected = if matches!(role, ResponsesMessageRoleV1::Assistant) {
                "output_text"
            } else {
                "input_text"
            };
            if part_type != expected {
                return Err(Phase43ApiErrorV1::unsupported(format!("{item_path}.type")));
            }
            check_fields(part, &["type", "text"], &[], &item_path)?;
            result.push(ResponsesTextPartV1 {
                kind: if expected == "output_text" {
                    ResponsesTextPartKindV1::OutputText
                } else {
                    ResponsesTextPartKindV1::InputText
                },
                text: required_text(part, "text", &item_path, false)?,
            });
        }
        result
    };
    Ok(ResponsesInputItemV1::Message { role, content })
}

fn parse_responses_tools(
    value: Option<&Value>,
) -> Result<Vec<ToolDefinitionV1>, Phase43ApiErrorV1> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let values = value
        .as_array()
        .ok_or_else(|| invalid("tools", "must be an array"))?;
    if values.is_empty() || values.len() > MAX_TOOL_DEFINITIONS {
        return Err(invalid("tools", "must contain between 1 and 128 entries"));
    }
    let mut result = Vec::with_capacity(values.len());
    let mut names = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        let path = format!("tools[{index}]");
        let object = object(value, &path)?;
        check_fields(
            object,
            &["type", "name", "description", "parameters", "strict"],
            &[],
            &path,
        )?;
        if required_text(object, "type", &path, true)? != "function" {
            return Err(Phase43ApiErrorV1::unsupported(format!("{path}.type")));
        }
        let tool = parse_tool_common(object, "parameters", &path)?;
        if !names.insert(tool.name.clone()) {
            return Err(invalid(format!("{path}.name"), "duplicate tool name"));
        }
        if let Some(strict) = object.get("strict") {
            if !strict.is_boolean() {
                return Err(invalid(format!("{path}.strict"), "must be a boolean"));
            }
        }
        result.push(tool);
    }
    Ok(result)
}

fn parse_anthropic_tools(
    value: Option<&Value>,
) -> Result<Vec<ToolDefinitionV1>, Phase43ApiErrorV1> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let values = value
        .as_array()
        .ok_or_else(|| invalid("tools", "must be an array"))?;
    if values.is_empty() || values.len() > MAX_TOOL_DEFINITIONS {
        return Err(invalid("tools", "must contain between 1 and 128 entries"));
    }
    let mut result = Vec::with_capacity(values.len());
    let mut names = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        let path = format!("tools[{index}]");
        let object = object(value, &path)?;
        check_fields(object, &["name", "description", "input_schema"], &[], &path)?;
        let tool = parse_tool_common(object, "input_schema", &path)?;
        if !names.insert(tool.name.clone()) {
            return Err(invalid(format!("{path}.name"), "duplicate tool name"));
        }
        result.push(tool);
    }
    Ok(result)
}

fn parse_tool_common(
    object: &Map<String, Value>,
    schema_field: &str,
    path: &str,
) -> Result<ToolDefinitionV1, Phase43ApiErrorV1> {
    let name = tool_name(
        required_text(object, "name", path, true)?,
        &format!("{path}.name"),
    )?;
    let description = match object.get("description") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let description = value
                .as_str()
                .ok_or_else(|| invalid(format!("{path}.description"), "must be a string"))?;
            if description.len() > MAX_TOOL_DESCRIPTION_BYTES {
                return Err(invalid(
                    format!("{path}.description"),
                    "description exceeds 16 KiB",
                ));
            }
            Some(description.to_owned())
        }
    };
    let schema = required(object, schema_field, path)?;
    if !schema.is_object() {
        return Err(invalid(
            format!("{path}.{schema_field}"),
            "must be an object",
        ));
    }
    let schema_bytes = serde_json::to_vec(schema).map_err(|error| {
        Phase43ApiErrorV1::invalid_json(error.to_string(), Some(format!("{path}.{schema_field}")))
    })?;
    if schema_bytes.len() > MAX_TOOL_SCHEMA_BYTES {
        return Err(invalid(
            format!("{path}.{schema_field}"),
            "schema exceeds 1 MiB",
        ));
    }
    Ok(ToolDefinitionV1 {
        name,
        description,
        parameters: schema.clone(),
    })
}

fn parse_responses_tool_choice(value: Option<&Value>) -> Result<ToolChoiceV1, Phase43ApiErrorV1> {
    let Some(value) = value else {
        return Ok(ToolChoiceV1::Auto {
            disable_parallel_tool_use: false,
        });
    };
    match value {
        Value::String(choice) => match choice.as_str() {
            "auto" => Ok(ToolChoiceV1::Auto {
                disable_parallel_tool_use: false,
            }),
            "none" => Ok(ToolChoiceV1::None),
            "required" => Ok(ToolChoiceV1::Required {
                disable_parallel_tool_use: false,
            }),
            _ => Err(invalid("tool_choice", "must be auto, none, or required")),
        },
        Value::Object(object) => {
            check_fields(object, &["type", "name"], &[], "tool_choice")?;
            if required_text(object, "type", "tool_choice", true)? != "function" {
                return Err(invalid("tool_choice.type", "must be function"));
            }
            let name = tool_name(
                required_text(object, "name", "tool_choice", true)?,
                "tool_choice.name",
            )?;
            Ok(ToolChoiceV1::Specific {
                name,
                disable_parallel_tool_use: false,
            })
        }
        _ => Err(invalid(
            "tool_choice",
            "must be a string or function object",
        )),
    }
}

fn parse_anthropic_tool_choice(value: Option<&Value>) -> Result<ToolChoiceV1, Phase43ApiErrorV1> {
    let Some(value) = value else {
        return Ok(ToolChoiceV1::Auto {
            disable_parallel_tool_use: false,
        });
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid("tool_choice", "must be an object"))?;
    let kind = required_text(object, "type", "tool_choice", true)?;
    let disable = match object.get("disable_parallel_tool_use") {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid("tool_choice.disable_parallel_tool_use", "must be a boolean"))?,
    };
    match kind.as_str() {
        "auto" => {
            check_fields(
                object,
                &["type", "disable_parallel_tool_use"],
                &[],
                "tool_choice",
            )?;
            Ok(ToolChoiceV1::Auto {
                disable_parallel_tool_use: disable,
            })
        }
        "none" => {
            check_fields(object, &["type"], &[], "tool_choice")?;
            if disable {
                return Err(invalid(
                    "tool_choice.disable_parallel_tool_use",
                    "is not valid for none",
                ));
            }
            Ok(ToolChoiceV1::None)
        }
        "any" => {
            check_fields(
                object,
                &["type", "disable_parallel_tool_use"],
                &[],
                "tool_choice",
            )?;
            Ok(ToolChoiceV1::Required {
                disable_parallel_tool_use: disable,
            })
        }
        "tool" => {
            check_fields(
                object,
                &["type", "name", "disable_parallel_tool_use"],
                &[],
                "tool_choice",
            )?;
            let name = tool_name(
                required_text(object, "name", "tool_choice", true)?,
                "tool_choice.name",
            )?;
            Ok(ToolChoiceV1::Specific {
                name,
                disable_parallel_tool_use: disable,
            })
        }
        _ => Err(invalid(
            "tool_choice.type",
            "must be auto, none, any, or tool",
        )),
    }
}

fn validate_tool_choice_has_tools(
    choice: &ToolChoiceV1,
    count: usize,
    param: &str,
) -> Result<(), Phase43ApiErrorV1> {
    if count == 0 && !matches!(choice, ToolChoiceV1::Auto { .. } | ToolChoiceV1::None) {
        return Err(invalid(param, "requires at least one declared tool"));
    }
    Ok(())
}

fn parse_responses_reasoning(
    value: Option<&Value>,
) -> Result<Option<ResponsesReasoningEffortV1>, Phase43ApiErrorV1> {
    let Some(value) = value else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid("reasoning", "must be an object"))?;
    check_fields(object, &["effort"], &[], "reasoning")?;
    let effort = required_text(object, "effort", "reasoning", true)?;
    match effort.as_str() {
        "low" => Ok(Some(ResponsesReasoningEffortV1::Low)),
        "medium" => Ok(Some(ResponsesReasoningEffortV1::Medium)),
        "high" => Ok(Some(ResponsesReasoningEffortV1::High)),
        _ => Err(invalid("reasoning.effort", "must be low, medium, or high")),
    }
}

fn parse_metadata(value: Option<&Value>) -> Result<BTreeMap<String, String>, Phase43ApiErrorV1> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid("metadata", "must be an object"))?;
    if object.len() > 16 {
        return Err(invalid("metadata", "must contain at most 16 entries"));
    }
    let mut result = BTreeMap::new();
    for (key, value) in object {
        if key.is_empty() || key.len() > 64 {
            return Err(invalid(
                format!("metadata.{key}"),
                "key must contain 1..=64 bytes",
            ));
        }
        let value = value
            .as_str()
            .ok_or_else(|| invalid(format!("metadata.{key}"), "value must be a string"))?;
        if value.len() > 512 {
            return Err(invalid(
                format!("metadata.{key}"),
                "value exceeds 512 bytes",
            ));
        }
        result.insert(key.clone(), value.to_owned());
    }
    Ok(result)
}

fn parse_sllm(value: Option<&Value>) -> Result<SllmExtensionsV1, Phase43ApiErrorV1> {
    let Some(value) = value else {
        return Ok(SllmExtensionsV1::default());
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid("sllm", "must be an object"))?;
    check_fields(object, &["resumable"], &[], "sllm")?;
    Ok(SllmExtensionsV1 {
        resumable: opt_bool_object(object, "resumable")?.unwrap_or(false),
    })
}

fn parse_anthropic_system(
    value: Option<&Value>,
) -> Result<Option<AnthropicSystemV1>, Phase43ApiErrorV1> {
    let Some(value) = value else {
        return Ok(None);
    };
    if let Value::String(text) = value {
        return Ok(Some(AnthropicSystemV1::Text(bounded_text(
            text, "system", false,
        )?)));
    }
    let values = value
        .as_array()
        .ok_or_else(|| invalid("system", "must be a string or text block array"))?;
    if values.is_empty() || values.len() > MAX_CONTENT_BLOCKS {
        return Err(invalid("system", "must contain 1..=256 blocks"));
    }
    let mut blocks = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let path = format!("system[{index}]");
        let object = object(value, &path)?;
        check_fields(object, &["type", "text"], &[], &path)?;
        if required_text(object, "type", &path, true)? != "text" {
            return Err(invalid(format!("{path}.type"), "must be text"));
        }
        blocks.push(required_text(object, "text", &path, false)?);
    }
    Ok(Some(AnthropicSystemV1::Blocks(blocks)))
}

fn parse_anthropic_messages(value: &Value) -> Result<Vec<AnthropicMessageV1>, Phase43ApiErrorV1> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid("messages", "must be an array"))?;
    if values.is_empty() || values.len() > MAX_MESSAGES {
        return Err(invalid("messages", "must contain 1..=1024 messages"));
    }
    let mut result = Vec::with_capacity(values.len());
    let mut outstanding = BTreeSet::new();
    let mut seen_ids = BTreeSet::new();
    let mut tool_use_count = 0usize;
    for (index, value) in values.iter().enumerate() {
        let path = format!("messages[{index}]");
        let object = object(value, &path)?;
        check_fields(object, &["role", "content"], &[], &path)?;
        let role = match required_text(object, "role", &path, true)?.as_str() {
            "user" => AnthropicRoleV1::User,
            "assistant" => AnthropicRoleV1::Assistant,
            _ => return Err(invalid(format!("{path}.role"), "must be user or assistant")),
        };
        let content = parse_anthropic_content(required(object, "content", &path)?, role, &path)?;
        let has_result = content
            .iter()
            .any(|item| matches!(item, AnthropicContentBlockV1::ToolResult { .. }));
        let has_use = content
            .iter()
            .any(|item| matches!(item, AnthropicContentBlockV1::ToolUse { .. }));
        if !outstanding.is_empty()
            && !(matches!(role, AnthropicRoleV1::User)
                && matches!(
                    content.first(),
                    Some(AnthropicContentBlockV1::ToolResult { .. })
                ))
        {
            return Err(invalid(
                "messages",
                "tool_result must immediately follow the assistant tool_use message",
            ));
        }
        if has_result {
            if !matches!(role, AnthropicRoleV1::User) {
                return Err(invalid(
                    format!("{path}.content"),
                    "tool_result requires a user message",
                ));
            }
            if !matches!(
                content.first(),
                Some(AnthropicContentBlockV1::ToolResult { .. })
            ) {
                return Err(invalid(
                    format!("{path}.content"),
                    "tool_result blocks must be first",
                ));
            }
            for item in &content {
                if let AnthropicContentBlockV1::ToolResult { tool_use_id, .. } = item {
                    if !outstanding.remove(tool_use_id) {
                        return Err(invalid(
                            format!("{path}.content"),
                            "tool_result must match one preceding tool_use exactly once",
                        ));
                    }
                }
            }
            if content
                .iter()
                .skip_while(|item| matches!(item, AnthropicContentBlockV1::ToolResult { .. }))
                .any(|item| matches!(item, AnthropicContentBlockV1::ToolResult { .. }))
            {
                return Err(invalid(
                    format!("{path}.content"),
                    "tool_result blocks must be contiguous",
                ));
            }
        }
        if has_use {
            if !matches!(role, AnthropicRoleV1::Assistant) {
                return Err(invalid(
                    format!("{path}.content"),
                    "tool_use requires an assistant message",
                ));
            }
            for item in &content {
                if let AnthropicContentBlockV1::ToolUse { id, .. } = item {
                    tool_use_count += 1;
                    if tool_use_count > MAX_TOOL_CALLS {
                        return Err(invalid("messages", "tool call count exceeds 16"));
                    }
                    if !seen_ids.insert(id.clone()) {
                        return Err(invalid(format!("{path}.content"), "duplicate tool_use id"));
                    }
                    outstanding.insert(id.clone());
                    if outstanding.len() > MAX_TOOL_CALLS {
                        return Err(invalid(
                            format!("{path}.content"),
                            "at most 16 tool_use blocks may be outstanding",
                        ));
                    }
                }
            }
        }
        result.push(AnthropicMessageV1 { role, content });
    }
    // A terminal assistant text message is an assistant prefill in this
    // profile.  Tool-use messages are allowed only when their results follow.
    if let Some(last) = result.last() {
        if matches!(last.role, AnthropicRoleV1::Assistant)
            && last
                .content
                .iter()
                .any(|item| matches!(item, AnthropicContentBlockV1::Text(_)))
        {
            return Err(Phase43ApiErrorV1::unsupported("messages"));
        }
    }
    if !outstanding.is_empty() {
        return Err(invalid(
            "messages",
            "every tool_use must have one following tool_result",
        ));
    }
    Ok(result)
}

fn parse_anthropic_content(
    value: &Value,
    role: AnthropicRoleV1,
    path: &str,
) -> Result<Vec<AnthropicContentBlockV1>, Phase43ApiErrorV1> {
    if let Value::String(text) = value {
        return Ok(vec![AnthropicContentBlockV1::Text(bounded_text(
            text,
            &format!("{path}.content"),
            false,
        )?)]);
    }
    let values = value
        .as_array()
        .ok_or_else(|| invalid(format!("{path}.content"), "must be a string or block array"))?;
    if values.is_empty() || values.len() > MAX_CONTENT_BLOCKS {
        return Err(invalid(
            format!("{path}.content"),
            "must contain 1..=256 blocks",
        ));
    }
    let mut blocks = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let item_path = format!("{path}.content[{index}]");
        let object = object(value, &item_path)?;
        let kind = required_text(object, "type", &item_path, true)?;
        match kind.as_str() {
            "text" => {
                check_fields(object, &["type", "text"], &[], &item_path)?;
                blocks.push(AnthropicContentBlockV1::Text(required_text(
                    object, "text", &item_path, false,
                )?));
            }
            "tool_use" => {
                if !matches!(role, AnthropicRoleV1::Assistant) {
                    return Err(Phase43ApiErrorV1::unsupported(format!("{item_path}.type")));
                }
                check_fields(object, &["type", "id", "name", "input"], &[], &item_path)?;
                let id = bounded_id(
                    required_text(object, "id", &item_path, true)?,
                    &format!("{item_path}.id"),
                )?;
                let name = tool_name(
                    required_text(object, "name", &item_path, true)?,
                    &format!("{item_path}.name"),
                )?;
                let input = required(object, "input", &item_path)?;
                if !input.is_object() {
                    return Err(invalid(format!("{item_path}.input"), "must be an object"));
                }
                let bytes = serde_json::to_vec(input).map_err(|error| {
                    Phase43ApiErrorV1::invalid_json(
                        error.to_string(),
                        Some(format!("{item_path}.input")),
                    )
                })?;
                if bytes.len() > MAX_TOOL_ARGUMENT_BYTES {
                    return Err(invalid(
                        format!("{item_path}.input"),
                        "arguments exceed 16 MiB",
                    ));
                }
                blocks.push(AnthropicContentBlockV1::ToolUse {
                    id,
                    name,
                    input: input.clone(),
                });
            }
            "tool_result" => {
                if !matches!(role, AnthropicRoleV1::User) {
                    return Err(Phase43ApiErrorV1::unsupported(format!("{item_path}.type")));
                }
                check_fields(
                    object,
                    &["type", "tool_use_id", "content", "is_error"],
                    &[],
                    &item_path,
                )?;
                let tool_use_id = bounded_id(
                    required_text(object, "tool_use_id", &item_path, true)?,
                    &format!("{item_path}.tool_use_id"),
                )?;
                let content = parse_tool_output(
                    required(object, "content", &item_path)?,
                    &format!("{item_path}.content"),
                )?;
                let is_error = match object.get("is_error") {
                    None => false,
                    Some(value) => value.as_bool().ok_or_else(|| {
                        invalid(format!("{item_path}.is_error"), "must be a boolean")
                    })?,
                };
                blocks.push(AnthropicContentBlockV1::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                });
            }
            "image" | "document" | "thinking" | "redacted_thinking" => {
                return Err(Phase43ApiErrorV1::unsupported(format!("{item_path}.type")));
            }
            _ => {
                return Err(invalid(
                    format!("{item_path}.type"),
                    "unsupported content block type",
                ));
            }
        }
    }
    Ok(blocks)
}

fn parse_stop_sequences(value: Option<&Value>) -> Result<Vec<String>, Phase43ApiErrorV1> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let values = value
        .as_array()
        .ok_or_else(|| invalid("stop_sequences", "must be an array of strings"))?;
    if values.is_empty() || values.len() > 4 {
        return Err(invalid(
            "stop_sequences",
            "must contain between 1 and 4 strings",
        ));
    }
    let mut result = Vec::with_capacity(values.len());
    let mut seen = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        let text = bounded_text(
            value
                .as_str()
                .ok_or_else(|| invalid(format!("stop_sequences[{index}]"), "must be a string"))?,
            &format!("stop_sequences[{index}]"),
            true,
        )?;
        if !seen.insert(text.clone()) {
            return Err(invalid("stop_sequences", "values must be unique"));
        }
        result.push(text);
    }
    Ok(result)
}

fn parse_tool_output(value: &Value, path: &str) -> Result<String, Phase43ApiErrorV1> {
    if let Some(text) = value.as_str() {
        return bounded_text(text, path, false);
    }
    let values = value
        .as_array()
        .ok_or_else(|| invalid(path, "must be a string or text block array"))?;
    if values.is_empty() || values.len() > MAX_CONTENT_BLOCKS {
        return Err(invalid(path, "must contain 1..=256 blocks"));
    }
    let mut result = String::new();
    for (index, value) in values.iter().enumerate() {
        let item_path = format!("{path}[{index}]");
        let object = object(value, &item_path)?;
        check_fields(object, &["type", "text"], &[], &item_path)?;
        if required_text(object, "type", &item_path, true)? != "text" {
            return Err(Phase43ApiErrorV1::unsupported(format!("{item_path}.type")));
        }
        let text = required_text(object, "text", &item_path, false)?;
        result.push_str(&text);
        if result.len() > MAX_TOOL_ARGUMENT_BYTES {
            return Err(invalid(path, "result exceeds 16 MiB"));
        }
    }
    Ok(result)
}

fn validate_arguments_json(arguments: &str, path: &str) -> Result<(), Phase43ApiErrorV1> {
    let value = serde_json::from_str::<StrictValue>(arguments)
        .map_err(|error| Phase43ApiErrorV1::invalid_json(error.to_string(), Some(path.to_owned())))?
        .0;
    if !value.is_object() {
        return Err(invalid(path, "arguments must be a JSON object"));
    }
    Ok(())
}

fn parse_object(
    body: &[u8],
    supported: &[&str],
    unsupported: &[&str],
) -> Result<Map<String, Value>, Phase43ApiErrorV1> {
    if body.len() > MAX_REQUEST_BODY_BYTES {
        return Err(Phase43ApiErrorV1::request_too_large());
    }
    let value = serde_json::from_slice::<StrictValue>(body)
        .map_err(|error| Phase43ApiErrorV1::invalid_json(error.to_string(), None))?
        .0;
    let object = value.as_object().ok_or_else(|| {
        Phase43ApiErrorV1::invalid_json("request body must be a JSON object", None)
    })?;
    let supported = supported.iter().copied().collect::<BTreeSet<_>>();
    let unsupported = unsupported.iter().copied().collect::<BTreeSet<_>>();
    for field in object.keys() {
        if !supported.contains(field.as_str()) {
            if unsupported.contains(field.as_str()) {
                return Err(Phase43ApiErrorV1::unsupported(field.clone()));
            }
            return Err(invalid(
                field.clone(),
                format!("unknown request field {field}"),
            ));
        }
    }
    Ok(object.clone())
}

fn object<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>, Phase43ApiErrorV1> {
    value
        .as_object()
        .ok_or_else(|| invalid(path, "must be an object"))
}
fn required<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<&'a Value, Phase43ApiErrorV1> {
    object
        .get(field)
        .ok_or_else(|| invalid(format!("{path}.{field}"), "field is required"))
}
fn required_text(
    object: &Map<String, Value>,
    field: &str,
    path: &str,
    nonempty: bool,
) -> Result<String, Phase43ApiErrorV1> {
    let value = required(object, field, path)?;
    let text = value
        .as_str()
        .ok_or_else(|| invalid(format!("{path}.{field}"), "must be a string"))?;
    bounded_text(text, &format!("{path}.{field}"), nonempty)
}
fn optional_text(
    map: &Map<String, Value>,
    field: &str,
) -> Result<Option<String>, Phase43ApiErrorV1> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => Ok(Some(bounded_text(
            value
                .as_str()
                .ok_or_else(|| invalid(field, "must be a string"))?,
            field,
            false,
        )?)),
    }
}
fn model(map: &Map<String, Value>) -> Result<String, Phase43ApiErrorV1> {
    let value = map
        .get("model")
        .ok_or_else(|| invalid("model", "field is required"))?;
    let model = value
        .as_str()
        .ok_or_else(|| invalid("model", "must be a string"))?;
    if model.is_empty() || model.len() > MAX_MODEL_ALIAS_BYTES {
        return Err(invalid("model", "must contain 1..=256 UTF-8 bytes"));
    }
    Ok(model.to_owned())
}
fn bounded_text(value: &str, param: &str, nonempty: bool) -> Result<String, Phase43ApiErrorV1> {
    if nonempty && value.is_empty() {
        return Err(invalid(param, "must be nonempty"));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(invalid(param, "text exceeds 16 MiB"));
    }
    Ok(value.to_owned())
}
fn bounded_id(value: String, param: &str) -> Result<String, Phase43ApiErrorV1> {
    if value.is_empty() || value.len() > MAX_TOOL_ID_BYTES {
        return Err(invalid(param, "ID must contain 1..=256 UTF-8 bytes"));
    }
    Ok(value)
}
fn tool_name(value: String, param: &str) -> Result<String, Phase43ApiErrorV1> {
    if value.is_empty()
        || value.len() > MAX_TOOL_NAME_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return Err(invalid(param, "name must match [A-Za-z0-9_-]{1,64}"));
    }
    Ok(value)
}
fn check_fields(
    object: &Map<String, Value>,
    supported: &[&str],
    unsupported: &[&str],
    path: &str,
) -> Result<(), Phase43ApiErrorV1> {
    let supported = supported.iter().copied().collect::<BTreeSet<_>>();
    let unsupported = unsupported.iter().copied().collect::<BTreeSet<_>>();
    for field in object.keys() {
        if !supported.contains(field.as_str()) {
            if unsupported.contains(field.as_str()) {
                return Err(Phase43ApiErrorV1::unsupported(format!("{path}.{field}")));
            }
            return Err(invalid(
                format!("{path}.{field}"),
                format!("unknown field {field}"),
            ));
        }
    }
    Ok(())
}
fn opt_u32(map: &Map<String, Value>, param: &str) -> Result<Option<u32>, Phase43ApiErrorV1> {
    match map.get(param) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let value = value
                .as_u64()
                .ok_or_else(|| invalid(param, "must be an unsigned integer"))?;
            if value > u64::from(u32::MAX) {
                return Err(invalid(param, "must be an unsigned 32-bit integer"));
            }
            Ok(Some(value as u32))
        }
    }
}
fn opt_f32(map: &Map<String, Value>, param: &str) -> Result<Option<f32>, Phase43ApiErrorV1> {
    match map.get(param) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => as_f32(value, param).map(Some),
    }
}
fn opt_bool(map: &Map<String, Value>, param: &str) -> Result<Option<bool>, Phase43ApiErrorV1> {
    match map.get(param) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid(param, "must be a boolean"))
            .map(Some),
    }
}
fn opt_bool_object(
    map: &Map<String, Value>,
    param: &str,
) -> Result<Option<bool>, Phase43ApiErrorV1> {
    opt_bool(map, param)
}
fn as_f32(value: &Value, param: &str) -> Result<f32, Phase43ApiErrorV1> {
    let value = value
        .as_f64()
        .ok_or_else(|| invalid(param, "must be a finite number"))?;
    if !value.is_finite() || !(value as f32).is_finite() {
        return Err(invalid(param, "must be a finite number"));
    }
    Ok(value as f32)
}
fn bounded_float(param: &str, value: f32, min: f32, max: f32) -> Result<(), Phase43ApiErrorV1> {
    if !value.is_finite() || !(min..=max).contains(&value) {
        return Err(invalid(
            param,
            format!("must be finite and in [{min},{max}]"),
        ));
    }
    Ok(())
}
fn invalid(param: impl Into<String>, message: impl Into<String>) -> Phase43ApiErrorV1 {
    Phase43ApiErrorV1::invalid_value(param, message)
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
