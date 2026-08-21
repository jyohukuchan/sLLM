#![allow(dead_code)]

#[path = "../src/phase43_api.rs"]
mod phase43_api;

use phase43_api::*;

fn json(value: serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap()
}

fn basic_responses() -> serde_json::Value {
    serde_json::json!({
        "model": "m",
        "input": "hello",
        "store": false,
    })
}

#[test]
fn responses_accepts_string_and_ordered_typed_history() {
    let body = serde_json::json!({
        "model":"m",
        "input":[
            {"type":"message","role":"user","content":[{"type":"input_text","text":"call"}]},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":""}]},
            {"type":"function_call","call_id":"call-1","name":"lookup","arguments":"{\"q\":\"x\"}"},
            {"type":"function_call_output","call_id":"call-1","output":"result"}
        ],
        "tools":[{"type":"function","name":"lookup","description":"read-only","parameters":{"type":"object"}}],
        "tool_choice":{"type":"function","name":"lookup"},
        "parallel_tool_calls":false,
        "reasoning":{"effort":"low"},
        "stream":true,
        "sllm":{"resumable":true}
    });
    let request = parse_responses_request_v1(&json(body)).unwrap();
    assert_eq!(request.model(), "m");
    assert!(matches!(request.input(), ResponsesInputV1::Items(items) if items.len() == 4));
    assert_eq!(request.tools().len(), 1);
    assert_eq!(request.tool_choice().specific_name(), Some("lookup"));
    assert!(!request.parallel_tool_calls());
    assert!(request.sllm().resumable());
    assert_eq!(
        request
            .reasoning_effort()
            .expect("reasoning effort")
            .max_reasoning_tokens(),
        1_024
    );
    for (effort, expected) in [("low", 1_024), ("medium", 2_048), ("high", 4_096)] {
        let mut value = basic_responses();
        value["reasoning"] = serde_json::json!({"effort": effort});
        let request = parse_responses_request_v1(&json(value)).unwrap();
        assert_eq!(
            request
                .reasoning_effort()
                .expect("reasoning effort")
                .max_reasoning_tokens(),
            expected
        );
    }

    let request = parse_responses_request_v1(&json(basic_responses())).unwrap();
    assert!(matches!(request.input(), ResponsesInputV1::Text(text) if text == "hello"));
}

#[test]
fn responses_rejects_unknown_duplicate_unsupported_and_bad_history() {
    let cases = [
        (
            br#"{"model":"m","input":"x","mystery":1}"#.to_vec(),
            "mystery",
            Phase43ErrorCodeV1::InvalidValue,
        ),
        (
            br#"{"model":"m","input":"x","model":"n"}"#.to_vec(),
            "",
            Phase43ErrorCodeV1::InvalidJson,
        ),
        (
            json(serde_json::json!({"model":"m","input":"x","store":true})),
            "store",
            Phase43ErrorCodeV1::UnsupportedParameter,
        ),
        (
            json(
                serde_json::json!({"model":"m","input":[{"type":"input_image","image_url":"https://example.invalid"}]}),
            ),
            "input[0].type",
            Phase43ErrorCodeV1::UnsupportedParameter,
        ),
        (
            json(
                serde_json::json!({"model":"m","input":[{"type":"function_call_output","call_id":"missing","output":"x"}]}),
            ),
            "input[0].call_id",
            Phase43ErrorCodeV1::InvalidValue,
        ),
        (
            json(serde_json::json!({"model":"m","input":"x","sllm":{"resumable":true}})),
            "sllm.resumable",
            Phase43ErrorCodeV1::InvalidValue,
        ),
    ];
    for (body, param, code) in cases {
        let error = parse_responses_request_v1(&body).unwrap_err();
        assert_eq!(error.code(), code, "{param}");
        if !param.is_empty() {
            assert_eq!(error.param(), Some(param));
        }
    }
}

#[test]
fn responses_validate_tool_schema_choice_and_limits() {
    let empty_tools = serde_json::json!({"model":"m","input":"x","tools":[]});
    assert_eq!(
        parse_responses_request_v1(&json(empty_tools))
            .unwrap_err()
            .param(),
        Some("tools")
    );

    let duplicate_tools = serde_json::json!({
        "model":"m", "input":"x",
        "tools":[
            {"type":"function","name":"a","parameters":{"type":"object"}},
            {"type":"function","name":"a","parameters":{"type":"object"}}
        ]
    });
    assert_eq!(
        parse_responses_request_v1(&json(duplicate_tools))
            .unwrap_err()
            .param(),
        Some("tools[1].name")
    );

    let bad_name = serde_json::json!({"model":"m","input":"x","tools":[{"type":"function","name":"bad.name","parameters":{"type":"object"}}]});
    assert_eq!(
        parse_responses_request_v1(&json(bad_name))
            .unwrap_err()
            .param(),
        Some("tools[0].name")
    );

    let wrong_schema = serde_json::json!({"model":"m","input":"x","tools":[{"type":"function","name":"a","parameters":[]} ]});
    assert_eq!(
        parse_responses_request_v1(&json(wrong_schema))
            .unwrap_err()
            .param(),
        Some("tools[0].parameters")
    );

    let unknown_choice = serde_json::json!({"model":"m","input":"x","tools":[{"type":"function","name":"a","parameters":{"type":"object"}}],"tool_choice":{"type":"function","name":"missing"}});
    assert_eq!(
        parse_responses_request_v1(&json(unknown_choice))
            .unwrap_err()
            .param(),
        Some("tool_choice.name")
    );

    let huge = serde_json::json!({"model":"m","input":"x","tools":[{"type":"function","name":"a","parameters":{"description":"x".repeat(MAX_TOOL_SCHEMA_BYTES)}}]});
    assert_eq!(
        parse_responses_request_v1(&json(huge)).unwrap_err().param(),
        Some("tools[0].parameters")
    );

    let too_many_calls = serde_json::json!({
        "model":"m",
        "input": (0..=MAX_TOOL_CALLS).map(|index| serde_json::json!({
            "type":"function_call", "call_id": format!("c{index}"), "name":"a", "arguments":"{}"
        })).collect::<Vec<_>>()
    });
    assert_eq!(
        parse_responses_request_v1(&json(too_many_calls))
            .unwrap_err()
            .param(),
        Some("input")
    );
}

#[test]
fn anthropic_header_messages_and_tool_result_order_are_strict() {
    let body = serde_json::json!({
        "model":"m", "max_tokens":128,
        "system":[{"type":"text","text":"system"}],
        "messages":[
            {"role":"user","content":"find"},
            {"role":"assistant","content":[{"type":"tool_use","id":"u1","name":"lookup","input":{"q":"x"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"u1","content":[{"type":"text","text":"ok"}]}]}
        ],
        "tools":[{"name":"lookup","description":"read-only","input_schema":{"type":"object"}}],
        "tool_choice":{"type":"any","disable_parallel_tool_use":true},
        "stream":true,
        "sllm":{"resumable":true}
    });
    let request = parse_anthropic_request_v1(&json(body), Some(ANTHROPIC_API_VERSION_V1)).unwrap();
    assert_eq!(request.messages().len(), 3);
    assert_eq!(request.max_tokens(), 128);
    assert!(!request.tool_choice().allows_parallel());
    assert!(request.sllm().resumable());
    assert!(validate_anthropic_version_header(Some(ANTHROPIC_API_VERSION_V1)).is_ok());
}

#[test]
fn anthropic_rejects_header_prefill_bad_order_unknown_nested_and_unsupported() {
    let basic =
        serde_json::json!({"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}]});
    for header in [None, Some("2024-01-01")] {
        assert_eq!(
            parse_anthropic_request_v1(&json(basic.clone()), header)
                .unwrap_err()
                .param(),
            Some("anthropic-version")
        );
    }
    let prefill = serde_json::json!({"model":"m","max_tokens":1,"messages":[{"role":"assistant","content":"prefill"}]});
    assert_eq!(
        parse_anthropic_request_v1(&json(prefill), Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .code(),
        Phase43ErrorCodeV1::UnsupportedParameter
    );

    let wrong_order = serde_json::json!({"model":"m","max_tokens":1,"messages":[
        {"role":"assistant","content":[{"type":"tool_use","id":"u1","name":"a","input":{}}]},
        {"role":"user","content":"intervening"},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"u1","content":"x"}]}
    ]});
    assert_eq!(
        parse_anthropic_request_v1(&json(wrong_order), Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .param(),
        Some("messages")
    );

    let unknown = serde_json::json!({"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"x","extra":true}]}]});
    assert_eq!(
        parse_anthropic_request_v1(&json(unknown), Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .param(),
        Some("messages[0].content[0].extra")
    );

    let unsupported = serde_json::json!({"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"image","source":{}}]}]});
    assert_eq!(
        parse_anthropic_request_v1(&json(unsupported), Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .code(),
        Phase43ErrorCodeV1::UnsupportedParameter
    );

    let duplicate = br#"{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],"sllm":{"resumable":false,"resumable":true}}"#;
    assert_eq!(
        parse_anthropic_request_v1(duplicate, Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .code(),
        Phase43ErrorCodeV1::InvalidJson
    );
}

#[test]
fn anthropic_boundaries_fail_closed() {
    let empty_tools = serde_json::json!({"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],"tools":[]});
    assert_eq!(
        parse_anthropic_request_v1(&json(empty_tools), Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .param(),
        Some("tools")
    );
    let empty_stops = serde_json::json!({"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],"stop_sequences":[]});
    assert_eq!(
        parse_anthropic_request_v1(&json(empty_stops), Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .param(),
        Some("stop_sequences")
    );

    let oversized = vec![b' '; MAX_REQUEST_BODY_BYTES + 1];
    assert_eq!(
        parse_anthropic_request_v1(&oversized, Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .code(),
        Phase43ErrorCodeV1::RequestTooLarge
    );
    let zero =
        serde_json::json!({"model":"m","max_tokens":0,"messages":[{"role":"user","content":"x"}]});
    assert_eq!(
        parse_anthropic_request_v1(&json(zero), Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .param(),
        Some("max_tokens")
    );
    let too_many = serde_json::json!({"model":"m","max_tokens":1,"messages":(0..=MAX_MESSAGES).map(|_| serde_json::json!({"role":"user","content":"x"})).collect::<Vec<_>>()});
    assert_eq!(
        parse_anthropic_request_v1(&json(too_many), Some(ANTHROPIC_API_VERSION_V1))
            .unwrap_err()
            .param(),
        Some("messages")
    );
}
