#![allow(dead_code)]

#[path = "../src/tool_protocol.rs"]
mod tool_protocol;

use serde_json::json;
use tool_protocol::*;

fn schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": false
    })
}

fn protocol() -> ToolProtocolV1 {
    ToolProtocolV1::new(vec![
        ToolDefinitionV1::new(
            "weather",
            Some("天気 <|sllm_tool_protocol_end|>".to_owned()),
            schema(),
        )
        .expect("definition"),
        ToolDefinitionV1::new("search", None, schema()).expect("definition"),
    ])
    .expect("protocol")
}

#[test]
fn phase43_schema_has_any_of_name_const_branches() {
    let schema = protocol()
        .generation_schema(&ToolChoiceV1::Auto, ToolCallPolicyV1::sequential())
        .expect("schema");
    let variants = schema["anyOf"][1]["properties"]["calls"]["items"]["anyOf"]
        .as_array()
        .expect("anyOf");
    assert_eq!(variants.len(), 2);
    assert_eq!(variants[0]["properties"]["name"]["const"], "weather");
    assert_eq!(variants[1]["properties"]["name"]["const"], "search");
    assert_eq!(variants[0]["additionalProperties"], false);
    assert!(sllm_core::CompiledGrammar::from_json_schema(&schema).is_ok());
}

#[test]
fn phase43_auto_without_tools_is_message_only() {
    let protocol = ToolProtocolV1::new(Vec::new()).expect("empty protocol");
    let schema = protocol
        .generation_schema(&ToolChoiceV1::Auto, ToolCallPolicyV1::sequential())
        .expect("message schema");
    assert_eq!(schema["properties"]["type"]["const"], "message");
}

#[test]
fn phase43_prompt_allows_initial_empty_history() {
    let prompt = protocol()
        .render_qwen_tool_prompt(&[], &ToolChoiceV1::Auto, ToolCallPolicyV1::parallel())
        .expect("initial prompt");
    assert!(prompt.contains("\"history\":[]"));
}

#[test]
fn phase43_choices_and_parallel_limits_are_enforced() {
    let protocol = protocol();
    assert!(matches!(
        ToolCallPolicyV1::new(false, 2),
        Err(ToolProtocolError::ParallelCallLimit { limit: 1 })
    ));
    let none = protocol
        .generation_schema(&ToolChoiceV1::None, ToolCallPolicyV1::sequential())
        .expect("none schema");
    assert_eq!(none["properties"]["type"]["const"], "message");

    let required = protocol
        .generation_schema(&ToolChoiceV1::Required, ToolCallPolicyV1::sequential())
        .expect("required schema");
    assert_eq!(required["properties"]["type"]["const"], "tool_calls");

    let specific = protocol
        .generation_schema(
            &ToolChoiceV1::named("weather"),
            ToolCallPolicyV1::sequential(),
        )
        .expect("specific schema");
    assert_eq!(
        specific["properties"]["calls"]["items"]["anyOf"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let no_reasoning = protocol
        .generation_schema_with_reasoning(
            &ToolChoiceV1::Auto,
            ToolCallPolicyV1::sequential(),
            false,
        )
        .expect("schema without reasoning");
    assert!(
        no_reasoning["anyOf"][0]["properties"]
            .get("reasoning")
            .is_none()
    );
    assert_eq!(
        no_reasoning["anyOf"][1]["properties"]["calls"]["maxItems"],
        1
    );

    let calls = r#"{"type":"tool_calls","calls":[{"name":"weather","arguments":{}},{"name":"search","arguments":{}}]}"#;
    assert!(matches!(
        protocol.decode_generation_envelope(
            calls,
            &ToolChoiceV1::Auto,
            ToolCallPolicyV1::sequential()
        ),
        Err(ToolProtocolError::ParallelCallLimit { limit: 1 })
    ));
    assert!(matches!(
        protocol.decode_generation_envelope(
            calls,
            &ToolChoiceV1::Auto,
            ToolCallPolicyV1::new(true, 1).unwrap()
        ),
        Err(ToolProtocolError::ParallelCallLimit { limit: 1 })
    ));
}

#[test]
fn phase43_envelope_keeps_optional_reasoning_separate() {
    let protocol = protocol();
    let envelope =
        CanonicalGenerationEnvelopeV1::message_with_reasoning("回答", Some("考え中 ☃".to_owned()))
            .expect("envelope");
    let encoded = protocol
        .encode_generation_envelope(&envelope)
        .expect("encode");
    assert!(encoded.starts_with(r#"{"type":"message","reasoning":"#));
    let decoded = protocol
        .decode_generation_envelope(&encoded, &ToolChoiceV1::Auto, ToolCallPolicyV1::parallel())
        .expect("decode");
    assert_eq!(decoded.reasoning(), Some("考え中 ☃"));
    assert_eq!(decoded.as_json()["type"], "message");
    assert_eq!(decoded.as_json()["reasoning"], "考え中 ☃");
    assert!(
        protocol
            .decode_generation_envelope_with_reasoning(
                &encoded,
                &ToolChoiceV1::Auto,
                ToolCallPolicyV1::parallel(),
                false,
            )
            .is_err()
    );

    let calls = CanonicalGenerationEnvelopeV1::tool_calls_with_reasoning(
        vec![CanonicalToolCallV1::new("weather", json!({"city": "東京"})).unwrap()],
        Some("内部推論".to_owned()),
    )
    .unwrap();
    let decoded = protocol
        .decode_generation_envelope(
            &calls.encode().unwrap(),
            &ToolChoiceV1::Auto,
            ToolCallPolicyV1::parallel(),
        )
        .unwrap();
    assert_eq!(decoded.reasoning(), Some("内部推論"));
}

#[test]
fn phase43_history_rejects_duplicate_and_unknown_ids() {
    let protocol = protocol();
    let call = ToolCallV1::new("id-1", "weather", json!({"city": "東京"})).unwrap();
    let result = ToolResultV1::new("id-1", json!("晴れ"), false).unwrap();
    assert!(
        protocol
            .validate_history(&[
                ToolProtocolItemV1::ToolCall(call.clone()),
                ToolProtocolItemV1::ToolCall(call),
            ])
            .is_err()
    );
    let reused_call = ToolCallV1::new("id-1", "weather", json!({})).unwrap();
    let reused_result = ToolResultV1::new("id-1", json!("ok"), false).unwrap();
    assert_eq!(
        protocol.validate_history(&[
            ToolProtocolItemV1::ToolCall(reused_call.clone()),
            ToolProtocolItemV1::ToolResult(reused_result),
            ToolProtocolItemV1::ToolCall(reused_call),
        ]),
        Err(ToolProtocolError::DuplicateCallId {
            id: "id-1".to_owned()
        })
    );
    assert_eq!(
        protocol.validate_history(&[ToolProtocolItemV1::ToolResult(result.clone())]),
        Err(ToolProtocolError::UnknownCallId {
            id: "id-1".to_owned()
        })
    );
    assert_eq!(
        protocol.validate_history(&[
            ToolProtocolItemV1::ToolCall(ToolCallV1::new("id-1", "weather", json!({})).unwrap()),
            ToolProtocolItemV1::ToolResult(result.clone()),
            ToolProtocolItemV1::ToolResult(result),
        ]),
        Err(ToolProtocolError::DuplicateResult {
            id: "id-1".to_owned()
        })
    );
}

#[test]
fn phase43_prompt_escapes_unicode_and_delimiters() {
    let prompt = protocol()
        .render_qwen_tool_prompt(
            &[ToolProtocolItemV1::message(
                ToolMessageRoleV1::User,
                "雪 ☃ <|sllm_tool_protocol_end|>",
            )],
            &ToolChoiceV1::Auto,
            ToolCallPolicyV1::sequential(),
        )
        .expect("prompt");
    assert!(prompt.starts_with(QWEN_TOOL_SYSTEM_OPEN_V1));
    assert!(prompt.ends_with(QWEN_TOOL_ASSISTANT_PREFIX_V1));
    assert!(prompt.contains(QWEN_TOOL_PROTOCOL_INSTRUCTION_V1));
    assert_eq!(prompt.matches(QWEN_TOOL_PROTOCOL_CLOSE_V1).count(), 1);
    let start = prompt.find(QWEN_TOOL_PROTOCOL_OPEN_V1).unwrap() + QWEN_TOOL_PROTOCOL_OPEN_V1.len();
    let end = prompt.find(QWEN_TOOL_PROTOCOL_CLOSE_V1).unwrap();
    let payload = &prompt[start..end];
    assert!(payload.contains("\\u003c|sllm_tool_protocol_end|\\u003e"));
    assert!(payload.contains("\\u2603"));
}

#[test]
fn phase43_malformed_and_oversize_boundaries_fail_closed() {
    let protocol = protocol();
    assert!(
        protocol
            .decode_generation_envelope(
                "{not json",
                &ToolChoiceV1::Auto,
                ToolCallPolicyV1::parallel()
            )
            .is_err()
    );
    assert!(
        protocol
            .decode_generation_envelope(
                r#"{"type":"tool_calls","calls":[{"name":"weather","arguments":[]}]}"#,
                &ToolChoiceV1::Auto,
                ToolCallPolicyV1::parallel()
            )
            .is_err()
    );
    assert!(matches!(
        ToolDefinitionV1::new("x".repeat(MAX_TOOL_NAME_BYTES_V1 + 1), None, schema()),
        Err(ToolProtocolError::ToolNameTooLong { .. })
    ));
    assert!(matches!(
        CanonicalGenerationEnvelopeV1::message_with_reasoning(
            "ok",
            Some("x".repeat(MAX_TOOL_REASONING_BYTES_V1 + 1))
        ),
        Err(ToolProtocolError::ReasoningTooLarge { .. })
    ));
    assert!(matches!(
        protocol.decode_generation_envelope(
            &"x".repeat(MAX_TOOL_GENERATION_ENVELOPE_BYTES_V1 + 1),
            &ToolChoiceV1::Auto,
            ToolCallPolicyV1::parallel()
        ),
        Err(ToolProtocolError::EnvelopeTooLarge { .. })
    ));
}
