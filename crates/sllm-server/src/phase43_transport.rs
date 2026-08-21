//! Pure Phase 43 response and stream serializers.
//!
//! This module deliberately does not know about axum, the scheduler, or a model
//! backend.  A caller supplies one completed, transport-neutral output and the
//! serializers produce the wire envelope (or the ordered named events) for one
//! compatibility profile.  In particular, this module never executes a tool.

use std::{collections::BTreeSet, fmt};

use serde::Serialize;
use serde_json::{Map, Value, json};

const MAX_STREAM_DELTA_BYTES: usize = 16 * 1024;

/// The stop reason understood by the transport-neutral completion result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase43FinishReasonV1 {
    Stop,
    Length,
    ToolUse,
    StopSequence,
}

/// Token accounting used by both compatibility profiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Phase43UsageV1 {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

impl Phase43UsageV1 {
    pub fn new(input_tokens: u64, output_tokens: u64) -> Result<Self, Phase43TransportError> {
        let total_tokens = input_tokens
            .checked_add(output_tokens)
            .ok_or(Phase43TransportError::InvalidOutput("token usage overflow"))?;
        Ok(Self {
            input_tokens,
            output_tokens,
            total_tokens,
        })
    }
}

/// One model-produced function call.  `item_id` and `call_id` are supplied by
/// the caller and are copied byte-for-byte into every applicable event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Phase43ToolCallV1 {
    pub item_id: String,
    pub call_id: String,
    pub name: String,
    /// A JSON object encoded as a string.  The serializer compacts and checks
    /// it before putting it on the wire.
    pub arguments: String,
}

impl Phase43ToolCallV1 {
    pub fn new(
        item_id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            item_id: item_id.into(),
            call_id: call_id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

/// The completed output consumed by either serializer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Phase43CompletedOutputV1 {
    /// Request-local response/message ID.  It is not a server-side store key.
    pub id: String,
    /// Model alias selected by the caller.  Both wire profiles expose it.
    pub model: String,
    /// Responses creation timestamp in Unix seconds, supplied by the caller.
    pub created_at: u64,
    /// ID for the primary text/message output item.
    pub item_id: String,
    /// Optional caller-provided ID for a reasoning output item.
    pub reasoning_item_id: Option<String>,
    pub text: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<Phase43ToolCallV1>,
    pub finish_reason: Phase43FinishReasonV1,
    pub stop_sequence: Option<String>,
    pub usage: Phase43UsageV1,
}

impl Phase43CompletedOutputV1 {
    pub fn new(
        id: impl Into<String>,
        item_id: impl Into<String>,
        finish_reason: Phase43FinishReasonV1,
        usage: Phase43UsageV1,
    ) -> Self {
        Self {
            id: id.into(),
            model: String::new(),
            created_at: 0,
            item_id: item_id.into(),
            reasoning_item_id: None,
            text: None,
            reasoning: None,
            tool_calls: Vec::new(),
            finish_reason,
            stop_sequence: None,
            usage,
        }
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub const fn with_created_at(mut self, created_at: u64) -> Self {
        self.created_at = created_at;
        self
    }

    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    pub fn with_reasoning_item_id(mut self, item_id: impl Into<String>) -> Self {
        self.reasoning_item_id = Some(item_id.into());
        self
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<Phase43ToolCallV1>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    pub fn with_stop_sequence(mut self, stop_sequence: impl Into<String>) -> Self {
        self.stop_sequence = Some(stop_sequence.into());
        self
    }
}

/// A named SSE event.  The service layer can pass `data` through
/// `serde_json::to_string` and then call `axum::response::sse::Event::event` /
/// `data`; no profile-specific framing is hidden in this type.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Phase43SseEventV1 {
    pub event: String,
    pub data: Value,
}

impl Phase43SseEventV1 {
    fn new(event: &'static str, data: Value) -> Self {
        Self {
            event: event.to_owned(),
            data,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Phase43TransportError {
    InvalidOutput(&'static str),
    InvalidToolArguments(String),
    StreamClosed,
}

impl fmt::Display for Phase43TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutput(message) => f.write_str(message),
            Self::InvalidToolArguments(message) => write!(f, "invalid tool arguments: {message}"),
            Self::StreamClosed => f.write_str("stream is already closed"),
        }
    }
}

impl std::error::Error for Phase43TransportError {}

/// Serialize a completed output as an OpenAI Responses object.
pub fn responses_non_stream_v1(
    output: &Phase43CompletedOutputV1,
) -> Result<Value, Phase43TransportError> {
    validate_output(output)?;
    let items = responses_items(output)?;
    let status = if output.finish_reason == Phase43FinishReasonV1::Length {
        "incomplete"
    } else {
        "completed"
    };
    let mut response = json!({
        "id": output.id,
        "object": "response",
        "created_at": output.created_at,
        "model": output.model,
        "status": status,
        "output": items,
        "output_text": output.text.as_deref().unwrap_or(""),
        "error": Value::Null,
        "incomplete_details": Value::Null,
        "usage": output.usage,
    });
    if output.finish_reason == Phase43FinishReasonV1::Length {
        response["incomplete_details"] = json!({"reason": "max_output_tokens"});
    }
    Ok(response)
}

/// Serialize a completed output as an Anthropic Messages object.
pub fn anthropic_non_stream_v1(
    output: &Phase43CompletedOutputV1,
) -> Result<Value, Phase43TransportError> {
    validate_anthropic_output(output)?;
    let content = anthropic_content(output)?;
    Ok(json!({
        "id": output.id,
        "type": "message",
        "role": "assistant",
        "model": output.model,
        "content": content,
        "stop_reason": anthropic_stop_reason(output.finish_reason),
        "stop_sequence": output.stop_sequence,
        "usage": {
            "input_tokens": output.usage.input_tokens,
            "output_tokens": output.usage.output_tokens,
        },
    }))
}

/// Closed-state builder for the Responses named SSE stream.
#[derive(Clone, Debug)]
pub struct ResponsesStreamBuilderV1 {
    id: String,
    closed: bool,
}

impl ResponsesStreamBuilderV1 {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            closed: false,
        }
    }

    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn complete(
        &mut self,
        output: &Phase43CompletedOutputV1,
    ) -> Result<Vec<Phase43SseEventV1>, Phase43TransportError> {
        self.ensure_open()?;
        if output.id != self.id {
            return Err(Phase43TransportError::InvalidOutput(
                "stream ID does not match completed output ID",
            ));
        }
        let mut events = Vec::new();
        let response = responses_non_stream_v1(output)?;
        events.push(Phase43SseEventV1::new(
            "response.created",
            json!({"type":"response.created", "response": response_with_status(&response, "in_progress")}),
        ));
        events.push(Phase43SseEventV1::new(
            "response.in_progress",
            json!({"type":"response.in_progress", "response": response_with_status(&response, "in_progress")}),
        ));

        let mut output_index = 0_u64;
        if let Some(reasoning) = output.reasoning.as_deref() {
            let item_id = reasoning_item_id(output)?;
            append_responses_reasoning_events(&mut events, item_id, output_index, reasoning);
            output_index += 1;
        }
        if let Some(text) = output.text.as_deref() {
            append_responses_text_events(&mut events, &output.item_id, output_index, text);
            output_index += 1;
        }
        for call in &output.tool_calls {
            append_responses_tool_events(&mut events, call, output_index)?;
            output_index += 1;
        }
        let mut completed = response;
        completed["id"] = Value::String(self.id.clone());
        events.push(Phase43SseEventV1::new(
            "response.completed",
            json!({"type":"response.completed", "response": completed}),
        ));
        self.closed = true;
        Ok(events)
    }

    pub fn error(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<Phase43SseEventV1, Phase43TransportError> {
        self.ensure_open()?;
        self.closed = true;
        Ok(Phase43SseEventV1::new(
            "error",
            json!({"type":"error", "code":code.into(), "message":message.into()}),
        ))
    }

    fn ensure_open(&self) -> Result<(), Phase43TransportError> {
        (!self.closed)
            .then_some(())
            .ok_or(Phase43TransportError::StreamClosed)
    }
}

/// Closed-state builder for the Anthropic named SSE stream.
#[derive(Clone, Debug)]
pub struct AnthropicStreamBuilderV1 {
    id: String,
    closed: bool,
}

impl AnthropicStreamBuilderV1 {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            closed: false,
        }
    }

    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn complete(
        &mut self,
        output: &Phase43CompletedOutputV1,
    ) -> Result<Vec<Phase43SseEventV1>, Phase43TransportError> {
        self.ensure_open()?;
        if output.id != self.id {
            return Err(Phase43TransportError::InvalidOutput(
                "stream ID does not match completed output ID",
            ));
        }
        validate_anthropic_output(output)?;
        let mut events = Vec::new();
        events.push(Phase43SseEventV1::new(
            "message_start",
            json!({"type":"message_start", "message": {
                "id": self.id,
                "type":"message",
                "role":"assistant",
                "model":output.model,
                "content":[],
                "stop_reason":Value::Null,
                "stop_sequence":Value::Null,
                "usage":{"input_tokens":output.usage.input_tokens,"output_tokens":0}
            }}),
        ));

        let mut index = 0_u64;
        if let Some(reasoning) = output.reasoning.as_deref() {
            append_anthropic_block_events(
                &mut events,
                index,
                json!({"type":"thinking","thinking":""}),
                utf8_chunks(reasoning, MAX_STREAM_DELTA_BYTES)
                    .into_iter()
                    .map(|chunk| json!({"type":"thinking_delta","thinking":chunk}))
                    .collect(),
            );
            index += 1;
        }
        if let Some(text) = output.text.as_deref() {
            append_anthropic_block_events(
                &mut events,
                index,
                json!({"type":"text","text":""}),
                utf8_chunks(text, MAX_STREAM_DELTA_BYTES)
                    .into_iter()
                    .map(|chunk| json!({"type":"text_delta","text":chunk}))
                    .collect(),
            );
            index += 1;
        }
        for call in &output.tool_calls {
            let arguments = canonical_arguments(&call.arguments)?;
            append_anthropic_block_events(
                &mut events,
                index,
                json!({"type":"tool_use","id":call.call_id,"name":call.name,"input":{}}),
                utf8_chunks(&arguments, MAX_STREAM_DELTA_BYTES)
                    .into_iter()
                    .map(|chunk| json!({"type":"input_json_delta","partial_json":chunk}))
                    .collect(),
            );
            index += 1;
        }
        events.push(Phase43SseEventV1::new(
            "message_delta",
            json!({"type":"message_delta", "delta": {
                "stop_reason": anthropic_stop_reason(output.finish_reason),
                "stop_sequence": output.stop_sequence
            }, "usage":{"output_tokens":output.usage.output_tokens}}),
        ));
        events.push(Phase43SseEventV1::new(
            "message_stop",
            json!({"type":"message_stop"}),
        ));
        self.closed = true;
        Ok(events)
    }

    pub fn error(
        &mut self,
        error_type: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<Phase43SseEventV1, Phase43TransportError> {
        self.ensure_open()?;
        self.closed = true;
        Ok(Phase43SseEventV1::new(
            "error",
            json!({"type":"error", "error":{"type":error_type.into(), "message":message.into()}}),
        ))
    }

    fn ensure_open(&self) -> Result<(), Phase43TransportError> {
        (!self.closed)
            .then_some(())
            .ok_or(Phase43TransportError::StreamClosed)
    }
}

fn canonical_arguments(arguments: &str) -> Result<String, Phase43TransportError> {
    let value: Value = serde_json::from_str(arguments)
        .map_err(|error| Phase43TransportError::InvalidToolArguments(error.to_string()))?;
    if !value.is_object() {
        return Err(Phase43TransportError::InvalidToolArguments(
            "arguments must be a JSON object".to_owned(),
        ));
    }
    Ok(canonicalize_json(value).to_string())
}

/// serde_json is compiled with `preserve_order` in this workspace.  Rebuild
/// objects in lexical key order so argument strings are canonical regardless
/// of how the model ordered properties.
fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<(String, Value)> = object.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = Map::new();
            for (key, value) in entries {
                sorted.insert(key, canonicalize_json(value));
            }
            Value::Object(sorted)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_json).collect()),
        value => value,
    }
}

fn responses_items(output: &Phase43CompletedOutputV1) -> Result<Vec<Value>, Phase43TransportError> {
    let mut items = Vec::new();
    if let Some(reasoning) = output.reasoning.as_deref() {
        let id = reasoning_item_id(output)?;
        items.push(json!({"id":id,"type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":reasoning}]}));
    }
    if let Some(text) = output.text.as_deref() {
        items.push(json!({"id":output.item_id,"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":text,"annotations":[]}]}));
    }
    for call in &output.tool_calls {
        items.push(json!({
            "id":call.item_id,
            "type":"function_call",
            "status":"completed",
            "call_id":call.call_id,
            "name":call.name,
            "arguments":canonical_arguments(&call.arguments)?
        }));
    }
    Ok(items)
}

fn anthropic_content(
    output: &Phase43CompletedOutputV1,
) -> Result<Vec<Value>, Phase43TransportError> {
    let mut content = Vec::new();
    if let Some(reasoning) = output.reasoning.as_deref() {
        content.push(json!({"type":"thinking","thinking":reasoning}));
    }
    if let Some(text) = output.text.as_deref() {
        content.push(json!({"type":"text","text":text}));
    }
    for call in &output.tool_calls {
        let arguments: Value = serde_json::from_str(&canonical_arguments(&call.arguments)?)
            .map_err(|_| {
                Phase43TransportError::InvalidOutput("canonical arguments were not JSON")
            })?;
        content
            .push(json!({"type":"tool_use","id":call.call_id,"name":call.name,"input":arguments}));
    }
    Ok(content)
}

fn reasoning_item_id(output: &Phase43CompletedOutputV1) -> Result<&str, Phase43TransportError> {
    output
        .reasoning_item_id
        .as_deref()
        .or_else(|| output.text.is_none().then_some(output.item_id.as_str()))
        .ok_or(Phase43TransportError::InvalidOutput(
            "reasoning output needs a distinct reasoning_item_id when text is also present",
        ))
}

fn validate_output(output: &Phase43CompletedOutputV1) -> Result<(), Phase43TransportError> {
    if output.id.is_empty() || output.item_id.is_empty() || output.model.is_empty() {
        return Err(Phase43TransportError::InvalidOutput(
            "response/item IDs and model must not be empty",
        ));
    }
    if output.reasoning.is_some() {
        let _ = reasoning_item_id(output)?;
    }
    match (output.finish_reason, output.stop_sequence.as_deref()) {
        (Phase43FinishReasonV1::StopSequence, Some(sequence)) if !sequence.is_empty() => {}
        (Phase43FinishReasonV1::StopSequence, _) => {
            return Err(Phase43TransportError::InvalidOutput(
                "stop_sequence finish requires a nonempty sequence",
            ));
        }
        (_, Some(_)) => {
            return Err(Phase43TransportError::InvalidOutput(
                "stop_sequence is only valid for a stop-sequence finish",
            ));
        }
        _ => {}
    }
    let mut item_ids = BTreeSet::new();
    if output.text.is_some() {
        item_ids.insert(output.item_id.as_str());
    }
    if output.reasoning.is_some() {
        let id = reasoning_item_id(output)?;
        if !item_ids.insert(id) {
            return Err(Phase43TransportError::InvalidOutput(
                "output item IDs must be unique",
            ));
        }
    }
    let mut call_ids = BTreeSet::new();
    for call in &output.tool_calls {
        if call.item_id.is_empty() || call.call_id.is_empty() || call.name.is_empty() {
            return Err(Phase43TransportError::InvalidOutput(
                "tool item, call, and name IDs must not be empty",
            ));
        }
        if !item_ids.insert(call.item_id.as_str()) || !call_ids.insert(call.call_id.as_str()) {
            return Err(Phase43TransportError::InvalidOutput(
                "tool item and call IDs must be unique",
            ));
        }
        let _ = canonical_arguments(&call.arguments)?;
    }
    Ok(())
}

fn validate_anthropic_output(
    output: &Phase43CompletedOutputV1,
) -> Result<(), Phase43TransportError> {
    validate_output(output)?;
    if output.reasoning.is_some() {
        return Err(Phase43TransportError::InvalidOutput(
            "Anthropic profile v1 does not expose thinking blocks",
        ));
    }
    Ok(())
}

fn anthropic_stop_reason(reason: Phase43FinishReasonV1) -> &'static str {
    match reason {
        Phase43FinishReasonV1::Stop => "end_turn",
        Phase43FinishReasonV1::Length => "max_tokens",
        Phase43FinishReasonV1::ToolUse => "tool_use",
        Phase43FinishReasonV1::StopSequence => "stop_sequence",
    }
}

fn response_with_status(response: &Value, status: &'static str) -> Value {
    let mut response = response.clone();
    response["status"] = Value::String(status.to_owned());
    response["output"] = json!([]);
    response["output_text"] = Value::String(String::new());
    response["incomplete_details"] = Value::Null;
    response["usage"] = Value::Null;
    response
}

fn append_responses_reasoning_events(
    events: &mut Vec<Phase43SseEventV1>,
    item_id: &str,
    output_index: u64,
    text: &str,
) {
    events.push(Phase43SseEventV1::new(
        "response.output_item.added",
        json!({"type":"response.output_item.added","output_index":output_index,"item":{"id":item_id,"type":"reasoning","status":"in_progress","summary":[]}}),
    ));
    events.push(Phase43SseEventV1::new(
        "response.content_part.added",
        json!({"type":"response.content_part.added","item_id":item_id,"output_index":output_index,"content_index":0,"part":{"type":"summary_text","text":""}}),
    ));
    for chunk in utf8_chunks(text, MAX_STREAM_DELTA_BYTES) {
        events.push(Phase43SseEventV1::new(
            "response.reasoning_summary_text.delta",
            json!({"type":"response.reasoning_summary_text.delta","item_id":item_id,"output_index":output_index,"summary_index":0,"delta":chunk}),
        ));
    }
    events.push(Phase43SseEventV1::new(
        "response.reasoning_summary_text.done",
        json!({"type":"response.reasoning_summary_text.done","item_id":item_id,"output_index":output_index,"summary_index":0,"text":text}),
    ));
    events.push(Phase43SseEventV1::new(
        "response.content_part.done",
        json!({"type":"response.content_part.done","item_id":item_id,"output_index":output_index,"content_index":0,"part":{"type":"summary_text","text":text}}),
    ));
    events.push(Phase43SseEventV1::new(
        "response.output_item.done",
        json!({"type":"response.output_item.done","output_index":output_index,"item":{"id":item_id,"type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":text}]}}),
    ));
}

fn append_responses_text_events(
    events: &mut Vec<Phase43SseEventV1>,
    item_id: &str,
    output_index: u64,
    text: &str,
) {
    events.push(Phase43SseEventV1::new(
        "response.output_item.added",
        json!({"type":"response.output_item.added","output_index":output_index,"item":{"id":item_id,"type":"message","role":"assistant","status":"in_progress","content":[]}}),
    ));
    events.push(Phase43SseEventV1::new(
        "response.content_part.added",
        json!({"type":"response.content_part.added","item_id":item_id,"output_index":output_index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}),
    ));
    for chunk in utf8_chunks(text, MAX_STREAM_DELTA_BYTES) {
        events.push(Phase43SseEventV1::new(
            "response.output_text.delta",
            json!({"type":"response.output_text.delta","item_id":item_id,"output_index":output_index,"content_index":0,"delta":chunk,"logprobs":[]}),
        ));
    }
    events.push(Phase43SseEventV1::new(
        "response.output_text.done",
        json!({"type":"response.output_text.done","item_id":item_id,"output_index":output_index,"content_index":0,"text":text}),
    ));
    events.push(Phase43SseEventV1::new(
        "response.content_part.done",
        json!({"type":"response.content_part.done","item_id":item_id,"output_index":output_index,"content_index":0,"part":{"type":"output_text","text":text,"annotations":[]}}),
    ));
    events.push(Phase43SseEventV1::new(
        "response.output_item.done",
        json!({"type":"response.output_item.done","output_index":output_index,"item":{"id":item_id,"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":text,"annotations":[]}]}}),
    ));
}

fn append_responses_tool_events(
    events: &mut Vec<Phase43SseEventV1>,
    call: &Phase43ToolCallV1,
    output_index: u64,
) -> Result<(), Phase43TransportError> {
    let arguments = canonical_arguments(&call.arguments)?;
    events.push(Phase43SseEventV1::new(
        "response.output_item.added",
        json!({"type":"response.output_item.added","output_index":output_index,"item":{"id":call.item_id,"type":"function_call","status":"in_progress","call_id":call.call_id,"name":call.name,"arguments":""}}),
    ));
    for chunk in utf8_chunks(&arguments, MAX_STREAM_DELTA_BYTES) {
        events.push(Phase43SseEventV1::new(
            "response.function_call_arguments.delta",
            json!({"type":"response.function_call_arguments.delta","item_id":call.item_id,"output_index":output_index,"delta":chunk}),
        ));
    }
    events.push(Phase43SseEventV1::new(
        "response.function_call_arguments.done",
        json!({"type":"response.function_call_arguments.done","item_id":call.item_id,"output_index":output_index,"arguments":arguments}),
    ));
    events.push(Phase43SseEventV1::new(
        "response.output_item.done",
        json!({"type":"response.output_item.done","output_index":output_index,"item":{"id":call.item_id,"type":"function_call","status":"completed","call_id":call.call_id,"name":call.name,"arguments":arguments}}),
    ));
    Ok(())
}

fn append_anthropic_block_events(
    events: &mut Vec<Phase43SseEventV1>,
    index: u64,
    block: Value,
    deltas: Vec<Value>,
) {
    events.push(Phase43SseEventV1::new(
        "content_block_start",
        json!({"type":"content_block_start","index":index,"content_block":block}),
    ));
    for delta in deltas {
        events.push(Phase43SseEventV1::new(
            "content_block_delta",
            json!({"type":"content_block_delta","index":index,"delta":delta}),
        ));
    }
    events.push(Phase43SseEventV1::new(
        "content_block_stop",
        json!({"type":"content_block_stop","index":index}),
    ));
}

fn utf8_chunks(value: &str, max_bytes: usize) -> Vec<&str> {
    if value.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < value.len() {
        let mut end = value.len().min(start + max_bytes);
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        chunks.push(&value[start..end]);
        start = end;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output() -> Phase43CompletedOutputV1 {
        Phase43CompletedOutputV1::new(
            "resp_1",
            "msg_1",
            Phase43FinishReasonV1::Stop,
            Phase43UsageV1::new(7, 3).expect("usage"),
        )
        .with_model("model")
        .with_created_at(1)
        .with_text("hello")
    }

    #[test]
    fn responses_non_stream_maps_text_usage_and_stop() {
        let response = responses_non_stream_v1(&output()).expect("response");
        assert_eq!(response["id"], "resp_1");
        assert_eq!(response["model"], "model");
        assert_eq!(response["created_at"], 1);
        assert_eq!(response["error"], Value::Null);
        assert_eq!(response["incomplete_details"], Value::Null);
        assert_eq!(response["status"], "completed");
        assert_eq!(response["output_text"], "hello");
        assert_eq!(response["usage"]["input_tokens"], 7);
        assert_eq!(response["output"][0]["id"], "msg_1");
    }

    #[test]
    fn anthropic_non_stream_maps_text_and_stop_reason() {
        let response = anthropic_non_stream_v1(&output()).expect("response");
        assert_eq!(response["id"], "resp_1");
        assert_eq!(response["model"], "model");
        assert_eq!(response["stop_reason"], "end_turn");
        assert_eq!(response["usage"]["output_tokens"], 3);
        assert_eq!(response["content"][0]["type"], "text");
    }

    #[test]
    fn responses_stream_has_ordered_text_events_and_no_done_sentinel() {
        let mut builder = ResponsesStreamBuilderV1::new("resp_1");
        let events = builder.complete(&output()).expect("events");
        let names: Vec<&str> = events.iter().map(|event| event.event.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(events[0].data["response"]["model"], "model");
        assert_eq!(events[0].data["response"]["created_at"], 1);
        assert!(names.iter().all(|name| *name != "[DONE]"));
        assert!(builder.is_closed());
    }

    #[test]
    fn stream_deltas_are_utf8_safe_and_bounded() {
        let text = "界".repeat(MAX_STREAM_DELTA_BYTES);
        let large = Phase43CompletedOutputV1::new(
            "resp_large",
            "msg_large",
            Phase43FinishReasonV1::Stop,
            Phase43UsageV1::new(1, 1).unwrap(),
        )
        .with_model("model")
        .with_text(&text);
        let events = ResponsesStreamBuilderV1::new("resp_large")
            .complete(&large)
            .unwrap();
        let deltas = events
            .iter()
            .filter(|event| event.event == "response.output_text.delta")
            .map(|event| event.data["delta"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(deltas.len() > 1);
        assert!(
            deltas
                .iter()
                .all(|delta| delta.len() <= MAX_STREAM_DELTA_BYTES)
        );
        assert_eq!(deltas.concat(), text);
    }

    #[test]
    fn anthropic_stream_has_block_indices_and_terminal_order() {
        let output = output().with_tool_calls(vec![Phase43ToolCallV1::new(
            "tool_item",
            "tool_call",
            "lookup",
            r#"{"q":"x"}"#,
        )]);
        let mut builder = AnthropicStreamBuilderV1::new("resp_1");
        let events = builder.complete(&output).expect("events");
        let names: Vec<&str> = events.iter().map(|event| event.event.as_str()).collect();
        assert_eq!(names.first(), Some(&"message_start"));
        assert_eq!(names.last(), Some(&"message_stop"));
        assert_eq!(names[names.len() - 2], "message_delta");
        assert_eq!(events[0].data["message"]["model"], "model");
        let starts: Vec<u64> = events
            .iter()
            .filter(|event| event.event == "content_block_start")
            .map(|event| event.data["index"].as_u64().expect("index"))
            .collect();
        assert_eq!(starts, [0, 1]);
    }

    #[test]
    fn single_and_parallel_tool_calls_are_stable_and_canonical() {
        let calls = vec![
            Phase43ToolCallV1::new("item_a", "call_a", "one", r#"{ "z": 1, "a": true }"#),
            Phase43ToolCallV1::new("item_b", "call_b", "two", r#"{"value":[1,2]}"#),
        ];
        let output = Phase43CompletedOutputV1::new(
            "resp_tools",
            "unused",
            Phase43FinishReasonV1::ToolUse,
            Phase43UsageV1::new(2, 4).expect("usage"),
        )
        .with_model("model")
        .with_tool_calls(calls);
        let response = responses_non_stream_v1(&output).expect("response");
        assert_eq!(response["output"][0]["id"], "item_a");
        assert_eq!(response["output"][0]["call_id"], "call_a");
        assert_eq!(response["output"][0]["arguments"], r#"{"a":true,"z":1}"#);
        assert_eq!(response["output"][1]["id"], "item_b");
        let mut builder = AnthropicStreamBuilderV1::new("resp_tools");
        let events = builder.complete(&output).expect("events");
        let indices: Vec<u64> = events
            .iter()
            .filter(|event| event.event == "content_block_start")
            .map(|event| event.data["index"].as_u64().expect("index"))
            .collect();
        assert_eq!(indices, [0, 1]);
    }

    #[test]
    fn terminal_error_closes_stream_and_forbids_success_or_duplicate_error() {
        let mut responses = ResponsesStreamBuilderV1::new("resp_error");
        let event = responses.error("server_error", "broken").expect("error");
        assert_eq!(event.event, "error");
        assert!(responses.complete(&output()).is_err());
        assert!(responses.error("server_error", "again").is_err());

        let mut anthropic = AnthropicStreamBuilderV1::new("msg_error");
        assert_eq!(
            anthropic.error("api_error", "broken").expect("error").event,
            "error"
        );
        assert!(anthropic.complete(&output()).is_err());
        assert!(anthropic.error("api_error", "again").is_err());
    }

    #[test]
    fn invalid_tool_arguments_fail_before_stream_terminal() {
        let output = Phase43CompletedOutputV1::new(
            "resp_bad",
            "unused",
            Phase43FinishReasonV1::ToolUse,
            Phase43UsageV1::new(1, 1).expect("usage"),
        )
        .with_model("model")
        .with_tool_calls(vec![Phase43ToolCallV1::new(
            "item", "call", "tool", "not json",
        )]);
        assert!(responses_non_stream_v1(&output).is_err());
        let mut builder = ResponsesStreamBuilderV1::new("resp_bad");
        assert!(builder.complete(&output).is_err());
        assert!(!builder.is_closed());
        assert!(builder.error("invalid_value", "tool arguments").is_ok());
        assert!(builder.is_closed());
    }
}
