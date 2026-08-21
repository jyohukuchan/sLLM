use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sllm_core::{fingerprint_for_json, parse_model_lock, verify_model_cache};
use sllm_frontend::{
    GenericTemplateErrorV1, GenericTemplateInputKindV1, GenericTemplateInputV1,
    GenericTemplateMessagesInputV1, GenericTemplateProviderV1, TokenizerFrontendV1,
    TokenizerUtilityErrorV1, TokenizerUtilityServiceV1,
};

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

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn copy_cache(directory: &Path) {
    let base = repository_path("ci/fixtures/model-lock-v1/cache");
    for entry in fs::read_dir(base).expect("read fixture cache") {
        let entry = entry.expect("read fixture cache entry");
        fs::copy(entry.path(), directory.join(entry.file_name())).expect("copy fixture cache file");
    }
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
        "sllm-phase44-template-adapter-{}-{sequence}",
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
    let lock = parse_model_lock(&serde_json::to_vec(&lock).expect("serialize fingerprinted lock"))
        .expect("parse assembled lock");
    let cache = verify_model_cache(&lock, &directory.0).expect("verify assembled cache");
    let tokenizer = TokenizerFrontendV1::from_verified_cache(&lock, &cache)
        .expect("construct verified tokenizer");
    Fixture {
        tokenizer,
        _directory: directory,
    }
}

fn provider(source: &str) -> GenericTemplateProviderV1 {
    let digest = format!("sha256:{:x}", Sha256::digest(source.as_bytes()));
    GenericTemplateProviderV1::new(source, &digest).expect("valid template")
}

#[test]
fn generic_adapter_tokenizes_typed_messages_and_preserves_full_identity() {
    let fixture = fixture();
    let utility = TokenizerUtilityServiceV1::new(&fixture.tokenizer, None);
    let template = "{{ special_tokens.bos }}{{ messages[0].role }}:{{ messages[0].content }} {{ kwargs.suffix }}{% if add_generation_prompt %}{{ special_tokens.assistant }}{% endif %}";
    let mut kwargs = serde_json::Map::new();
    kwargs.insert("suffix".to_owned(), json!("世界"));
    let mut special_tokens = serde_json::Map::new();
    special_tokens.insert("bos".to_owned(), json!("<bos>"));
    special_tokens.insert("assistant".to_owned(), json!("<assistant>"));
    let messages = GenericTemplateMessagesInputV1::from_parts(
        vec![json!({"role": "user", "content": "hello"})],
        kwargs,
        special_tokens,
        true,
        false,
        Some("medium".to_owned()),
    )
    .unwrap();
    let result = utility
        .apply_generic_template(
            &provider(template),
            GenericTemplateInputV1::messages(messages),
        )
        .unwrap();
    assert_eq!(result.rendered(), "<bos>user:hello 世界<assistant>");
    assert_eq!(result.identity().kind(), "generic-jinja-v1");
    assert_eq!(result.identity().digest(), provider(template).digest());
    assert!(result.generic_identity().is_some());
    assert!(result.kwargs_digest().is_some());
    assert_eq!(
        result.count(),
        utility
            .input_token_count_generic(
                &provider(template),
                GenericTemplateInputV1::json(json!({
                    "messages": [{"role": "user", "content": "hello"}],
                    "kwargs": {"suffix": "世界"},
                    "special_tokens": {"bos": "<bos>", "assistant": "<assistant>"},
                    "add_generation_prompt": true,
                    "enable_thinking": false,
                    "reasoning_effort": "medium"
                }))
                .unwrap()
            )
            .unwrap()
    );
}

#[test]
fn generic_json_context_is_explicit_and_invalid_message_data_rejects_before_tokenize() {
    let fixture = fixture();
    let utility = TokenizerUtilityServiceV1::new(&fixture.tokenizer, None);
    let renderer = provider("{{ messages|length }}");
    let bad = GenericTemplateInputV1::json(json!({"messages": [{"content": "missing role"}]}));
    assert!(matches!(
        bad,
        Err(TokenizerUtilityErrorV1::GenericTemplate(
            GenericTemplateErrorV1::InvalidContext
        ))
    ));
    let bad = GenericTemplateMessagesInputV1::from_parts(
        vec![json!({"role": 42, "content": "bad"})],
        Default::default(),
        Default::default(),
        true,
        false,
        None,
    );
    assert!(matches!(
        bad,
        Err(TokenizerUtilityErrorV1::GenericTemplate(
            GenericTemplateErrorV1::InvalidContext
        ))
    ));
    let too_many = (0..1025)
        .map(|_| json!({"role": "user", "content": "x"}))
        .collect();
    let bad = GenericTemplateMessagesInputV1::new(too_many);
    assert!(matches!(
        bad,
        Err(TokenizerUtilityErrorV1::GenericTemplate(
            GenericTemplateErrorV1::TooManyMessages { .. }
        ))
    ));
    let _ = (utility, renderer);
}

#[test]
fn raw_and_gemma_inputs_fail_before_tokenization() {
    let fixture = fixture();
    let utility = TokenizerUtilityServiceV1::new(&fixture.tokenizer, None);
    let renderer = provider("raw");
    for (input, kind) in [
        (
            GenericTemplateInputV1::raw_text("hello"),
            GenericTemplateInputKindV1::RawText,
        ),
        (
            GenericTemplateInputV1::gemma_raw_text("hello"),
            GenericTemplateInputKindV1::GemmaRawText,
        ),
    ] {
        assert!(matches!(
            utility.apply_generic_template(&renderer, input),
            Err(TokenizerUtilityErrorV1::UnsupportedGenericTemplateInput { kind: actual }) if actual == kind
        ));
    }
}

#[test]
fn typed_special_tokens_and_reasoning_effort_are_bounded() {
    let bad_tokens = GenericTemplateMessagesInputV1::from_parts(
        vec![json!({"role": "user", "content": "x"})],
        Default::default(),
        [("bos".to_owned(), json!(42))].into_iter().collect(),
        true,
        true,
        None,
    );
    assert!(matches!(
        bad_tokens,
        Err(TokenizerUtilityErrorV1::GenericTemplate(
            GenericTemplateErrorV1::InvalidContext
        ))
    ));
    let bad_effort = GenericTemplateMessagesInputV1::from_parts(
        vec![json!({"role": "user", "content": "x"})],
        Default::default(),
        Default::default(),
        true,
        true,
        Some("x".repeat(33)),
    );
    assert!(matches!(
        bad_effort,
        Err(TokenizerUtilityErrorV1::GenericTemplate(
            GenericTemplateErrorV1::InvalidContext
        ))
    ));
}
