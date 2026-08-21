use sllm_frontend::{TokenByteTableV1, TokenPieceClassV1};

#[test]
fn fixture_table_is_id_ordered_and_classifies_special_rows() {
    let table = TokenByteTableV1::from_tokenizer_json(
        include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json"),
        16,
    )
    .expect("fixture token-byte table constructs");

    assert_eq!(table.len(), 16);
    assert_eq!(
        table.entry(0).expect("row 0").class(),
        TokenPieceClassV1::Ordinary
    );
    assert_eq!(table.entry(0).expect("row 0").bytes(), Some(&b"<unk>"[..]));
    assert_eq!(
        table.entry(8).expect("EOS row").class(),
        TokenPieceClassV1::Special
    );
    assert_eq!(table.entry(8).expect("EOS row").bytes(), None);
    assert_eq!(
        table.entry(15).expect("reserved row").class(),
        TokenPieceClassV1::Reserved
    );
    assert_eq!(table.entry(15).expect("reserved row").piece(), None);
    assert!(!table.entry(8).expect("EOS row").is_grammar_eligible());
    assert!(table.entry(1).expect("ordinary row").is_grammar_eligible());
}

#[test]
fn table_rejects_a_token_piece_over_the_bounded_length() {
    let piece = "x".repeat(129);
    let tokenizer = format!(
        r#"{{
          "version": "1.0",
          "added_tokens": [],
          "model": {{
            "type": "WordLevel",
            "vocab": {{"{piece}": 0}},
            "unk_token": "{piece}"
          }}
        }}"#
    );
    let error = TokenByteTableV1::from_tokenizer_json(tokenizer.as_bytes(), 1)
        .expect_err("overlong pieces must fail closed");
    assert!(matches!(
        error,
        sllm_frontend::TokenizerError::TokenBytePieceTooLong { id: 0, len: 129 }
    ));
}
