use sllm_core::{CompiledGrammar, GrammarError, TokenTrie, Utf8State};

#[test]
fn bounded_json_object_accepts_nested_scalars() {
    let grammar = CompiledGrammar::json_object().expect("JSON grammar");
    let mut state = grammar.initial_state();
    for (index, byte) in br#"{"a":[true,null,-12.5]}"#.iter().enumerate() {
        let active_before = state.active_state_count();
        state.accept(&[*byte]).unwrap_or_else(|error| {
            panic!("valid JSON prefix at {index} (active {active_before}): {error}")
        });
    }
    assert!(state.is_finished());
}

#[test]
fn bounded_repetition_and_reset_are_exact() {
    let grammar = CompiledGrammar::compile("root ::= \"a\"{2,4}").expect("grammar");
    let mut state = grammar.initial_state();
    state.accept(b"aa").expect("minimum");
    assert!(state.is_finished());
    state.accept(b"aa").expect("maximum");
    assert!(matches!(
        state.accept(b"a"),
        Err(GrammarError::TokenRejected)
    ));
    state.reset();
    assert!(!state.is_finished());
}

#[test]
fn token_trie_masks_only_byte_prefixes_and_reports_all_masked() {
    let grammar = CompiledGrammar::compile("root ::= \"ok\"").expect("grammar");
    let state = grammar.initial_state();
    let trie = TokenTrie::new([b"o".as_slice(), b"ok", b"x"]).expect("trie");
    assert_eq!(
        state.valid_token_mask_with_trie(&trie).expect("mask"),
        [true, true, false]
    );
    let impossible = TokenTrie::new([b"x".as_slice()]).expect("trie");
    assert!(matches!(
        state.valid_token_mask_with_trie(&impossible),
        Err(GrammarError::AllTokensMasked)
    ));
    let sparse = TokenTrie::new_optional([Some(b"o".as_slice()), None, Some(b"ok".as_slice())])
        .expect("sparse trie");
    assert_eq!(
        state
            .valid_token_mask_with_trie(&sparse)
            .expect("sparse mask"),
        [true, false, true]
    );
}

#[test]
fn partial_utf8_token_piece_can_cross_token_boundary() {
    let grammar = CompiledGrammar::compile("root ::= \"€\"").expect("grammar");
    let mut state = grammar.initial_state();
    state.accept(&[0xe2]).expect("first UTF-8 byte");
    assert_eq!(
        state.utf8_state(),
        &Utf8State::Partial {
            expected: 3,
            seen: 1
        }
    );
    state.accept(&[0x82, 0xac]).expect("remaining UTF-8 bytes");
    assert!(state.is_finished());
}

#[test]
fn json_schema_subset_preserves_property_order_and_rejects_keywords() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "first": {"type": "string"},
            "count": {"type": "integer"}
        },
        "required": ["first"],
        "additionalProperties": false
    });
    let grammar = CompiledGrammar::from_json_schema(&schema).expect("schema");
    let mut state = grammar.initial_state();
    for (index, byte) in br#"{"first":"x","count":1}"#.iter().enumerate() {
        let active_before = state.active_state_count();
        state.accept(&[*byte]).unwrap_or_else(|error| {
            panic!("valid JSON prefix at {index} (active {active_before}): {error}")
        });
    }
    assert!(state.is_finished());

    let unsupported = serde_json::json!({"type": "string", "pattern": "[a-z]+"});
    assert!(matches!(
        CompiledGrammar::from_json_schema(&unsupported),
        Err(GrammarError::UnsupportedSchemaKeyword(keyword)) if keyword == "pattern"
    ));

    let nested = serde_json::json!({
        "type": "object",
        "properties": {"inner": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"], "additionalProperties": false}},
        "required": ["inner"],
        "additionalProperties": false
    });
    let nested_grammar = CompiledGrammar::from_json_schema(&nested).expect("nested schema");
    let mut nested_state = nested_grammar.initial_state();
    for (index, byte) in br#"{"inner":{"id":1}}"#.iter().enumerate() {
        nested_state
            .accept(&[*byte])
            .unwrap_or_else(|error| panic!("nested JSON prefix at {index}: {error}"));
    }
    assert!(nested_state.is_finished());
}
