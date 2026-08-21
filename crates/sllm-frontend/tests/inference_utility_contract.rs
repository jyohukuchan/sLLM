use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};
use sllm_core::{fingerprint_for_json, parse_model_lock, verify_model_cache};
use sllm_frontend::{
    DecodeModeV1, InputTokenCountInputV1, Qwen35ChatMessageV1, Qwen35RenderOptionsV1, TokenPieceV1,
    TokenizeOptionsV1, TokenizerFrontendV1, TokenizerUtilityErrorV1, TokenizerUtilityServiceV1,
};

#[test]
fn fim_template_is_versioned_digest_bound_and_ordered() {
    let template = sllm_frontend::FimTemplateV1::new(100, 101, 102).unwrap();
    assert_eq!(template.version(), 1);
    assert!(template.digest().starts_with("sha256:"));
    assert_eq!(template.digest().len(), 71);
    let rendered = template.render(&[1, 2], &[3], Some(&[9])).unwrap();
    assert_eq!(rendered.as_slice(), &[100, 9, 1, 2, 101, 3, 102]);
    assert!(sllm_frontend::FimTemplateV1::new(100, 100, 102).is_err());
    assert!(template.render(&[], &[], None).is_err());
}

#[test]
fn backend_verified_template_adapter_rejects_floating_or_empty_results() {
    let identity = sllm_frontend::TemplateIdentityV1::from_verified_parts(
        "synthetic-template-v1",
        1,
        "fixture",
        format!("sha256:{}", "0".repeat(64)),
        1,
    )
    .unwrap();
    assert!(
        sllm_frontend::ApplyTemplateResultV1::from_verified_parts(
            "hello".to_owned(),
            sllm_frontend::TokenIdsV1::from_slice(&[1]),
            identity,
        )
        .is_ok()
    );
    assert!(
        sllm_frontend::TemplateIdentityV1::from_verified_parts(
            "synthetic-template-v1",
            1,
            "fixture",
            "latest",
            1,
        )
        .is_err()
    );
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Fixture {
    tokenizer: TokenizerFrontendV1,
    _directory: TestDirectory,
}

fn repository_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn copy_cache(directory: &Path) {
    let base = repository_path("ci/fixtures/model-lock-v1/cache");
    for entry in fs::read_dir(base).expect("read fixture cache") {
        let entry = entry.expect("read fixture cache entry");
        fs::copy(entry.path(), directory.join(entry.file_name())).expect("copy fixture cache file");
    }
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

fn set_file_metadata(lock: &mut Value, path: &str, bytes: &[u8]) {
    let files = lock["model"]["files"]
        .as_array_mut()
        .expect("lock files array");
    let file = files
        .iter_mut()
        .find(|file| file["path"] == path)
        .expect("fixture file entry");
    file["size_bytes"] = json!(bytes.len());
    file["sha256"] = json!(sha256(bytes));
}

fn fixture() -> Fixture {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let directory = TestDirectory(std::env::temp_dir().join(format!(
        "sllm-inference-utility-{}-{sequence}",
        std::process::id()
    )));
    fs::create_dir(&directory.0).expect("create fixture directory");
    copy_cache(&directory.0);

    let tokenizer = include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json");
    let config = br#"{
  "architectures": ["FixtureModel"],
  "model_type": "fixture",
  "eos_token_id": 8,
  "text_config": {"model_type": "fixture_text", "eos_token_id": 8}
}
"#;
    let tokenizer_config = br#"{
  "eos_token": "<|im_end|>",
  "added_tokens_decoder": {
    "8": {"content": "<|endoftext|>", "special": true},
    "9": {"content": "<|im_end|>", "special": true}
  }
}
"#;
    fs::write(directory.0.join("tokenizer.json"), tokenizer).expect("write tokenizer");
    fs::write(directory.0.join("config.json"), config).expect("write config");
    fs::write(directory.0.join("tokenizer_config.json"), tokenizer_config)
        .expect("write tokenizer config");

    let mut lock: Value = serde_json::from_slice(
        &fs::read(repository_path("ci/fixtures/model-lock-v1/lock.json")).expect("read lock"),
    )
    .expect("parse lock");
    set_file_metadata(&mut lock, "tokenizer.json", tokenizer);
    set_file_metadata(&mut lock, "config.json", config);
    set_file_metadata(&mut lock, "tokenizer_config.json", tokenizer_config);
    let contract = &mut lock["model"]["tokenizer_contract"];
    contract["files"] = json!([
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json"
    ]);
    contract["vocab_size"] = json!(12);
    contract["eos_token_id"] = json!(8);
    contract["special_token_ids"] = json!({"bos": 10, "eos": 8});
    contract["stop_identity"] = json!({
        "config_eos": {"token": "<|endoftext|>", "token_id": 8, "source_file": "config.json"},
        "tokenizer_eos": {"token": "<|im_end|>", "token_id": 9, "source_files": ["tokenizer_config.json", "tokenizer.json"]}
    });
    contract["generation_stop_policy"]["stop_token_ids"] = json!([9, 8]);
    let fingerprint = fingerprint_for_json(&serde_json::to_vec(&lock).expect("serialize lock"))
        .expect("fingerprint lock");
    lock["fingerprint"] = json!(fingerprint);
    let lock_bytes = serde_json::to_vec(&lock).expect("serialize fingerprinted lock");
    let lock = parse_model_lock(&lock_bytes).expect("parse assembled lock");
    let cache = verify_model_cache(&lock, &directory.0).expect("verify assembled cache");
    let tokenizer = TokenizerFrontendV1::from_verified_cache(&lock, &cache)
        .expect("construct verified tokenizer");
    Fixture {
        tokenizer,
        _directory: directory,
    }
}

fn service(fixture: &Fixture) -> TokenizerUtilityServiceV1<'_> {
    TokenizerUtilityServiceV1::new(&fixture.tokenizer, None)
}

#[test]
fn token_count_boundaries_and_unicode_use_model_default_special_policy() {
    let fixture = fixture();
    let utility = service(&fixture);
    assert_eq!(utility.input_token_count_raw("").unwrap(), 0);
    assert_eq!(utility.input_token_count_raw("hello").unwrap(), 1);
    assert_eq!(utility.input_token_count_raw("hello world é").unwrap(), 3);
    assert_eq!(
        utility.input_token_count_raw(
            "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello",
        )
        .unwrap(),
        17
    );
    assert!(utility.tokenize_default("").unwrap().token_ids().is_empty());
    assert_eq!(utility.tokenize_default("é 世界,").unwrap().count(), 3);
}

#[test]
fn optional_pieces_are_decoder_aware_and_detokenize_preserves_special_policy() {
    let fixture = fixture();
    let utility = service(&fixture);
    let result = utility
        .tokenize("hello world", TokenizeOptionsV1::with_pieces())
        .expect("tokenize with pieces");
    assert_eq!(result.count(), 2);
    let pieces = result.pieces().expect("pieces requested");
    assert_eq!(
        pieces,
        [
            TokenPieceV1::Utf8("hello".into()),
            TokenPieceV1::Utf8("world".into())
        ]
    );
    assert_eq!(
        utility
            .detokenize_ids(&[1, 8, 2], DecodeModeV1::SkipSpecialTokens)
            .unwrap(),
        "hello world"
    );
    assert_eq!(
        utility
            .detokenize_ids(&[1, 8, 2], DecodeModeV1::PreserveSpecialTokens)
            .unwrap(),
        "hello <|endoftext|> world"
    );
}

#[test]
fn raw_non_utf8_piece_variant_is_lossless() {
    let piece = TokenPieceV1::Bytes(vec![0xff, 0x00, 0x80]);
    assert_eq!(piece.as_utf8(), None);
    assert_eq!(piece.as_bytes(), [0xff, 0x00, 0x80]);
}

#[test]
fn unknown_token_ids_fail_closed_and_template_absence_is_explicit() {
    let fixture = fixture();
    let utility = service(&fixture);
    assert!(matches!(
        utility.detokenize_ids(&[17], DecodeModeV1::PreserveSpecialTokens),
        Err(TokenizerUtilityErrorV1::Detokenize(
            sllm_frontend::TokenizerError::UnknownTokenId { id: 17 }
        ))
    ));
    let messages = [Qwen35ChatMessageV1::user("hello")];
    assert_eq!(
        utility
            .input_token_count(InputTokenCountInputV1::RawText("hello"))
            .unwrap(),
        1
    );
    assert!(matches!(
        utility.apply_template(&messages, Qwen35RenderOptionsV1::default()),
        Err(TokenizerUtilityErrorV1::TemplateUnavailable)
    ));
    assert!(matches!(
        utility.input_token_count_messages(&messages, Qwen35RenderOptionsV1::default()),
        Err(TokenizerUtilityErrorV1::TemplateUnavailable)
    ));
}
