#![allow(dead_code)]

#[path = "../src/phase42_api.rs"]
mod phase42_api;

use phase42_api::*;

fn completion(value: serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(&value).unwrap()
}

#[test]
fn completion_accepts_all_prompt_shapes_and_defaults() {
    let cases = [
        serde_json::json!({"model":"m","prompt":"hello"}),
        serde_json::json!({"model":"m","prompt":["hello","world"]}),
        serde_json::json!({"model":"m","prompt":[1,2,3]}),
        serde_json::json!({"model":"m","prompt":[[1,2],[3]]}),
    ];
    for body in cases {
        let request = parse_completion_request(&completion(body)).unwrap();
        assert_eq!(request.max_tokens(), DEFAULT_COMPLETION_TOKENS);
        assert_eq!(request.n(), 1);
    }
}

#[test]
fn completion_rejects_unknown_unsupported_wrong_type_and_nonfinite() {
    for (body, param, code) in [
        (
            serde_json::json!({"model":"m","prompt":"x","mystery":1}),
            "mystery",
            ErrorCodeV1::InvalidValue,
        ),
        (
            serde_json::json!({"model":"m","prompt":"x","messages":[]}),
            "messages",
            ErrorCodeV1::UnsupportedParameter,
        ),
        (
            serde_json::json!({"model":"m","prompt":"x","suffix":"tail"}),
            "suffix",
            ErrorCodeV1::UnsupportedParameter,
        ),
        (
            serde_json::json!({"model":"m","prompt":1}),
            "prompt",
            ErrorCodeV1::InvalidValue,
        ),
        (
            serde_json::json!({"model":"m","prompt":"x","temperature":3}),
            "temperature",
            ErrorCodeV1::InvalidValue,
        ),
        (
            serde_json::json!({"model":"m","prompt":"x","logprobs":true}),
            "logprobs",
            ErrorCodeV1::InvalidValue,
        ),
    ] {
        let error = parse_completion_request(&completion(body)).unwrap_err();
        assert_eq!(error.param(), Some(param));
        assert_eq!(error.code(), code);
    }
    let duplicate = br#"{"model":"a","model":"b","prompt":"x"}"#;
    assert_eq!(
        parse_completion_request(duplicate).unwrap_err().code(),
        ErrorCodeV1::InvalidJson
    );
}

#[test]
fn completion_validates_stop_bias_and_limits() {
    let request = parse_completion_request(&completion(serde_json::json!({
        "model":"m", "prompt":"x", "stop":["a","b"], "logit_bias":{"1":-100,"2":100},
        "max_tokens":4096, "temperature":0, "top_p":0, "n":8, "logprobs":5,
    })))
    .unwrap();
    assert_eq!(request.stop(), &["a", "b"]);
    assert_eq!(request.logit_bias().len(), 2);
    for value in [
        serde_json::json!({"model":"m","prompt":"x","max_tokens":0}),
        serde_json::json!({"model":"m","prompt":"x","max_tokens":4097}),
        serde_json::json!({"model":"m","prompt":"x","stop":["x","x"]}),
        serde_json::json!({"model":"m","prompt":"x","n":9}),
    ] {
        assert_eq!(
            parse_completion_request(&completion(value))
                .unwrap_err()
                .code(),
            ErrorCodeV1::InvalidValue
        );
    }
}

#[test]
fn embeddings_accept_four_shapes_and_encoding() {
    for input in [
        serde_json::json!("hello"),
        serde_json::json!(["hello", "world"]),
        serde_json::json!([1, 2, 3]),
        serde_json::json!([[1, 2], [3]]),
    ] {
        let request =
            parse_embedding_request(&completion(serde_json::json!({"model":"m","input":input})))
                .unwrap();
        assert_eq!(request.encoding_format(), EmbeddingEncodingFormatV1::Float);
    }
    let request = parse_embedding_request(&completion(serde_json::json!({
        "model":"m","input":"x","encoding_format":"base64","dimensions":128,
    })))
    .unwrap();
    assert_eq!(request.encoding_format(), EmbeddingEncodingFormatV1::Base64);
    assert_eq!(request.dimensions(), Some(128));
}

#[test]
fn embeddings_reject_mixed_and_bad_dimensions() {
    for input in [serde_json::json!(""), serde_json::json!(["a", ""])] {
        let error = parse_embedding_request(&completion(serde_json::json!({
            "model":"m", "input": input
        })))
        .unwrap_err();
        assert_eq!(error.param(), Some("input"));
        assert_eq!(error.code(), ErrorCodeV1::InvalidValue);
    }
    for value in [
        serde_json::json!({"model":"m","input":["x",1]}),
        serde_json::json!({"model":"m","input":[]}),
        serde_json::json!({"model":"m","input":"x","encoding_format":"binary"}),
        serde_json::json!({"model":"m","input":"x","dimensions":0}),
        serde_json::json!({"model":"m","input":"x","dimensions":32769}),
    ] {
        assert_eq!(
            parse_embedding_request(&completion(value))
                .unwrap_err()
                .code(),
            ErrorCodeV1::InvalidValue
        );
    }
}

#[test]
fn rerank_and_token_utilities_are_strict() {
    let rerank = parse_rerank_request(&completion(serde_json::json!({
        "model":"ranker","query":"q","documents":["a","b"],"top_n":1,"return_documents":true,
    })))
    .unwrap();
    assert_eq!(rerank.documents().len(), 2);
    assert_eq!(rerank.top_n(), Some(1));
    assert!(rerank.return_documents());
    assert_eq!(
        parse_rerank_request(&completion(serde_json::json!({
            "model":"m","query":"q","documents":[],
        })))
        .unwrap_err()
        .param(),
        Some("documents")
    );

    let tokenize =
        parse_tokenize_request(&completion(serde_json::json!({"model":"m","text":""}))).unwrap();
    assert_eq!(tokenize.text(), "");
    let detokenize =
        parse_detokenize_request(&completion(serde_json::json!({"model":"m","tokens":[1]})))
            .unwrap();
    assert_eq!(detokenize.tokens(), &[1]);
    assert!(!detokenize.skip_special_tokens());
    assert!(
        parse_tokenize_request(&completion(serde_json::json!({
            "model":"m","text":"x","add_special":true
        })))
        .is_err()
    );
}

#[test]
fn template_input_tokens_and_infill_share_strict_messages() {
    let body = completion(serde_json::json!({
        "model":"m","messages":[{"role":"user","content":"hello"}],
    }));
    let template = parse_apply_template_request(&body).unwrap();
    let count = parse_input_tokens_request(&body).unwrap();
    assert_eq!(template.messages().len(), 1);
    assert!(matches!(count.input(), InputTokensInputV1::Messages(messages) if messages.len() == 1));
    let raw = parse_input_tokens_request(&completion(serde_json::json!({
        "model":"m","text":"raw"
    })))
    .unwrap();
    assert!(matches!(raw.input(), InputTokensInputV1::Text(text) if text == "raw"));

    let infill = parse_infill_request(&completion(serde_json::json!({
        "model":"m","prefix":"a","suffix":"b","prompt":"c","stream":true,"max_tokens":4,
    })))
    .unwrap();
    assert_eq!(infill.prefix(), "a");
    assert!(infill.stream());
    assert_eq!(infill.prompt(), Some("c"));
    assert_eq!(
        parse_infill_request(&completion(serde_json::json!({
            "model":"m","prefix":"a","suffix":"b","input_extra":[],
        })))
        .unwrap_err()
        .code(),
        ErrorCodeV1::UnsupportedParameter
    );
}

#[test]
fn body_and_text_boundaries_are_fail_closed() {
    let body = serde_json::to_vec(&serde_json::json!({"model":"m","prompt":"x"})).unwrap();
    assert!(parse_completion_request(&body).is_ok());
    let oversized = vec![b' '; MAX_REQUEST_BODY_BYTES + 1];
    assert_eq!(
        parse_completion_request(&oversized).unwrap_err().code(),
        ErrorCodeV1::RequestTooLarge
    );
    let long_model =
        completion(serde_json::json!({"model":"m".repeat(MAX_MODEL_ALIAS_BYTES + 1),"prompt":"x"}));
    assert_eq!(
        parse_completion_request(&long_model).unwrap_err().param(),
        Some("model")
    );
}
