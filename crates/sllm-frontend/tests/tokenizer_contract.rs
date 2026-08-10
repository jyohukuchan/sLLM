use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use sllm_core::{
    ModelLock, VerifiedCache, fingerprint_for_json, parse_model_lock, verify_model_cache,
};
use sllm_frontend::{
    DecodeModeV1, EosIdentityV1, TokenIdContextV1, TokenIdsV1, TokenizerError, TokenizerFrontendV1,
};
use tokenizers::Tokenizer;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Fixture {
    cache: VerifiedCache,
    lock: ModelLock,
    directory: TestDirectory,
}

fn repository_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn test_directory(label: &str) -> TestDirectory {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "sllm-tokenizer-contract-{label}-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir(&path).expect("create tokenizer test directory");
    TestDirectory(path)
}

fn sha256_hex(bytes: &[u8]) -> String {
    // This is a host-test helper only. Production code never shells out and
    // never hashes an unverified model payload here.
    let mut child = Command::new("sha256sum")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("sha256sum is available on the supported Linux host");
    child
        .stdin
        .take()
        .expect("sha256sum stdin is available")
        .write_all(bytes)
        .expect("write bytes to sha256sum");
    let output = child.wait_with_output().expect("wait for sha256sum");
    assert!(output.status.success(), "sha256sum failed");
    String::from_utf8(output.stdout)
        .expect("sha256sum output is UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum output has a digest")
        .to_owned()
}

fn replace_once(source: String, old: &str, new: &str) -> String {
    assert_eq!(source.matches(old).count(), 1, "replacement must be unique");
    source.replacen(old, new, 1)
}

fn lock_bytes(
    tokenizer_bytes: &[u8],
    config_bytes: &[u8],
    tokenizer_config_bytes: &[u8],
) -> Vec<u8> {
    let source = String::from_utf8(
        fs::read(repository_path("ci/fixtures/model-lock-v1/lock.json")).expect("base lock exists"),
    )
    .expect("base lock is UTF-8");
    let old_file = r#"        "path": "tokenizer.json",
        "size_bytes": 111,
        "sha256": "47902732b5a9eefbecb3c12ecefaa07167453e7f52602e3bf80c183de16e0448","#;
    let new_file = format!(
        "        \"path\": \"tokenizer.json\",\n        \"size_bytes\": {},\n        \"sha256\": \"{}\",",
        tokenizer_bytes.len(),
        sha256_hex(tokenizer_bytes)
    );
    let source = replace_once(source, old_file, &new_file);
    let old_config_file = r#"        "path": "config.json",
        "size_bytes": 168,
        "sha256": "ca329e82e70943de4efe770ea30d487a9d3bbb42b459abed3a6c9d8d8ea8166c","#;
    let new_config_file = format!(
        "        \"path\": \"config.json\",\n        \"size_bytes\": {},\n        \"sha256\": \"{}\",",
        config_bytes.len(),
        sha256_hex(config_bytes)
    );
    let source = replace_once(source, old_config_file, &new_config_file);
    let old_tokenizer_config_file = r#"        "path": "tokenizer_config.json",
        "size_bytes": 141,
        "sha256": "9142864de0fa95cef90d7eeff12419dd705f6e8db77b4e281347fd98ee73a137","#;
    let new_tokenizer_config_file = format!(
        "        \"path\": \"tokenizer_config.json\",\n        \"size_bytes\": {},\n        \"sha256\": \"{}\",",
        tokenizer_config_bytes.len(),
        sha256_hex(tokenizer_config_bytes)
    );
    let source = replace_once(
        source,
        old_tokenizer_config_file,
        &new_tokenizer_config_file,
    );

    let old_contract = r#"    "tokenizer_contract": {
      "files": ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"],
      "chat_template_path": "chat_template.jinja",
      "vocab_size": 1,
      "eos_token_id": 0,
      "special_token_ids": {"eos": 0},
      "stop_identity": {
        "config_eos": {"token": "<|endoftext|>", "token_id": 0, "source_file": "config.json"},
        "tokenizer_eos": {"token": "<|endoftext|>", "token_id": 0, "source_files": ["tokenizer_config.json", "tokenizer.json"]}
      },
      "generation_stop_policy": {
        "version": 1,
        "stop_token_ids": [0],
        "evaluation": "newly_generated_after_argmax",
        "prompt_evaluation": "never_stop",
        "stop_token": {"visible_output": false, "subsequent_decode_input": false},
        "budget_boundary": "stop_token_wins",
        "max_new_tokens_zero": "max_new_tokens_before_decode",
        "reason_version": 1
      }
    },"#;
    let new_contract = r#"    "tokenizer_contract": {
      "files": ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"],
      "chat_template_path": "chat_template.jinja",
      "vocab_size": 8,
      "eos_token_id": 8,
      "special_token_ids": {"bos": 10, "eos": 8},
      "stop_identity": {
        "config_eos": {"token": "<|endoftext|>", "token_id": 8, "source_file": "config.json"},
        "tokenizer_eos": {"token": "<|im_end|>", "token_id": 9, "source_files": ["tokenizer_config.json", "tokenizer.json"]}
      },
      "generation_stop_policy": {
        "version": 1,
        "stop_token_ids": [9, 8],
        "evaluation": "newly_generated_after_argmax",
        "prompt_evaluation": "never_stop",
        "stop_token": {"visible_output": false, "subsequent_decode_input": false},
        "budget_boundary": "stop_token_wins",
        "max_new_tokens_zero": "max_new_tokens_before_decode",
        "reason_version": 1
      }
    },"#;
    let source = replace_once(source, old_contract, new_contract);
    let fingerprint = fingerprint_for_json(source.as_bytes()).expect("recompute lock fingerprint");
    let source = replace_once(
        source,
        "  \"fingerprint\": \"sha256:7201b4dddf49fb09e4d871778c5dd75eaec29d3d0ab3911ae7eb7ea62548a490\",",
        &format!("  \"fingerprint\": \"{fingerprint}\","),
    );
    source.into_bytes()
}

fn fixture_from_bytes(label: &str, tokenizer_bytes: &[u8]) -> Fixture {
    let directory = test_directory(label);
    let base_cache = repository_path("ci/fixtures/model-lock-v1/cache");
    for entry in fs::read_dir(base_cache).expect("read base cache") {
        let entry = entry.expect("read base cache entry");
        fs::copy(entry.path(), directory.0.join(entry.file_name())).expect("copy base cache file");
    }
    let mut config_bytes = fs::read(directory.0.join("config.json")).expect("read config asset");
    let config_text = String::from_utf8(config_bytes).expect("config is UTF-8");
    assert_eq!(config_text.matches("\"eos_token_id\": 0").count(), 2);
    config_bytes = config_text
        .replace("\"eos_token_id\": 0", "\"eos_token_id\": 8")
        .into_bytes();
    let tokenizer_config_bytes = br#"{
  "eos_token": "<|im_end|>",
  "added_tokens_decoder": {
    "8": {
      "content": "<|endoftext|>",
      "special": true
    },
    "9": {
      "content": "<|im_end|>",
      "special": true
    }
  }
}
"#
    .to_vec();
    fs::write(directory.0.join("config.json"), &config_bytes).expect("write config asset");
    fs::write(
        directory.0.join("tokenizer_config.json"),
        &tokenizer_config_bytes,
    )
    .expect("write tokenizer config asset");
    fs::write(directory.0.join("tokenizer.json"), tokenizer_bytes).expect("write tokenizer asset");
    let lock = parse_model_lock(&lock_bytes(
        tokenizer_bytes,
        &config_bytes,
        &tokenizer_config_bytes,
    ))
    .expect("assembled lock parses");
    let cache = verify_model_cache(&lock, &directory.0).expect("assembled cache verifies");
    Fixture {
        cache,
        lock,
        directory,
    }
}

fn valid_fixture(label: &str) -> Fixture {
    fixture_from_bytes(
        label,
        include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json"),
    )
}

fn frontend(fixture: &Fixture) -> TokenizerFrontendV1 {
    TokenizerFrontendV1::from_verified_cache(&fixture.lock, &fixture.cache)
        .expect("fixture tokenizer frontend constructs")
}

fn json_string(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len() + 2);
    encoded.push('"');
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            '\t' => encoded.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write;

                write!(encoded, "\\u{:04x}", character as u32).expect("write JSON escape");
            }
            character => encoded.push(character),
        }
    }
    encoded.push('"');
    encoded
}

fn json_ids(ids: &[u32]) -> String {
    ids.iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn eos_manifest(identity: &sllm_frontend::EosIdentitySnapshotV1) -> String {
    format!(
        "{{\"token\": {}, \"token_id\": {}, \"observed_content\": {}}}",
        json_string(identity.token()),
        identity.token_id(),
        json_string(identity.observed_content()),
    )
}

fn tokenizer_manifest(tokenizer: &TokenizerFrontendV1) -> String {
    let snapshot = tokenizer.snapshot();
    let special_roles = snapshot
        .special_roles()
        .iter()
        .map(|role| {
            format!(
                "    {{\"role\": {}, \"token_id\": {}, \"content\": {}}}",
                json_string(role.role()),
                role.token_id(),
                json_string(role.content()),
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let hello_world = tokenizer
        .encode("hello world")
        .expect("manifest hello encode");
    let unicode = tokenizer
        .encode("é 世界,")
        .expect("manifest Unicode encode");
    let empty = tokenizer.encode("").expect("manifest empty encode");
    let decode_ids = TokenIdsV1::from_slice(&[1, 8, 2]);
    let skip_special_tokens = tokenizer
        .decode(&decode_ids, DecodeModeV1::SkipSpecialTokens)
        .expect("manifest skipped decode");
    let preserve_special_tokens = tokenizer
        .decode(&decode_ids, DecodeModeV1::PreserveSpecialTokens)
        .expect("manifest preserved decode");

    format!(
        "{{\n  \"fixture_version\": \"tokenizer-v1\",\n  \"consistency_label\": {},\n  \"vocab_size\": {},\n  \"special_roles\": [\n{}\n  ],\n  \"config_eos\": {},\n  \"tokenizer_eos\": {},\n  \"stop_token_ids\": [{}],\n  \"encode\": {{\n    \"hello world\": [{}],\n    \"é 世界,\": [{}],\n    \"\": [{}]\n  }},\n  \"decode\": {{\n    \"ids\": [{}],\n    \"skip_special_tokens\": {},\n    \"preserve_special_tokens\": {}\n  }}\n}}\n",
        json_string(snapshot.fingerprint()),
        snapshot.vocab_size(),
        special_roles,
        eos_manifest(snapshot.config_eos()),
        eos_manifest(snapshot.tokenizer_eos()),
        json_ids(snapshot.stop_token_ids()),
        json_ids(hello_world.as_slice()),
        json_ids(unicode.as_slice()),
        json_ids(empty.as_slice()),
        json_ids(decode_ids.as_slice()),
        json_string(&skip_special_tokens),
        json_string(&preserve_special_tokens),
    )
}

fn fixture_lock_variant<F>(label: &str, mutate: F) -> (Fixture, TokenizerError)
where
    F: FnOnce(&mut ModelLock),
{
    let mut fixture = valid_fixture(label);
    mutate(&mut fixture.lock);
    let error = TokenizerFrontendV1::from_verified_cache(&fixture.lock, &fixture.cache)
        .expect_err("variant must be rejected");
    (fixture, error)
}

#[test]
fn expected_manifest_is_the_authoritative_public_contract() {
    let fixture = valid_fixture("manifest");
    let tokenizer = frontend(&fixture);
    let manifest = tokenizer_manifest(&tokenizer);
    assert_eq!(
        manifest,
        include_str!("../../../ci/fixtures/tokenizer-v1/expected.json")
    );
    assert_eq!(
        manifest.as_bytes(),
        include_bytes!("../../../ci/fixtures/tokenizer-v1/expected.json")
    );
}

#[test]
fn fixture_post_processor_adds_bos_only_on_requested_raw_path() {
    let raw = Tokenizer::from_bytes(include_bytes!(
        "../../../ci/fixtures/tokenizer-v1/tokenizer.json"
    ))
    .expect("fixture tokenizer loads");
    assert_eq!(raw.encode("hello", true).unwrap().get_ids(), &[10, 1]);

    let fixture = valid_fixture("frontend-no-specials");
    assert_eq!(frontend(&fixture).encode("hello").unwrap().as_slice(), &[1]);
}

#[test]
fn empty_and_unicode_inputs_are_supported() {
    let fixture = valid_fixture("empty-unicode");
    let tokenizer = frontend(&fixture);
    assert!(
        tokenizer
            .encode("")
            .expect("empty encode succeeds")
            .is_empty()
    );
    assert_eq!(
        tokenizer
            .encode("é 世界,")
            .expect("unicode encode succeeds")
            .as_slice(),
        &[3, 4, 7]
    );
}

#[test]
fn decode_skip_and_preserve_special_tokens_are_distinct() {
    let fixture = valid_fixture("decode-special");
    let tokenizer = frontend(&fixture);
    let ids = TokenIdsV1::from_slice(&[1, 8, 2]);
    assert_eq!(
        tokenizer
            .decode(&ids, DecodeModeV1::SkipSpecialTokens)
            .unwrap(),
        "hello world"
    );
    assert_eq!(
        tokenizer
            .decode(&ids, DecodeModeV1::PreserveSpecialTokens)
            .unwrap(),
        "hello <|endoftext|> world"
    );
}

#[test]
fn fingerprint_mismatch_is_rejected_without_path_details() {
    let (fixture, error) = fixture_lock_variant("fingerprint", |lock| {
        lock.fingerprint = format!("sha256:{}", "0".repeat(64));
    });
    assert!(matches!(
        error,
        TokenizerError::LockFingerprintMismatch { .. }
    ));
    assert!(!error.to_string().contains("tokenizer.json"));
    drop(fixture);
}

#[test]
fn u64_token_id_overflow_is_checked_before_tokenizer_lookup() {
    let (fixture, error) = fixture_lock_variant("u64-overflow", |lock| {
        lock.model
            .tokenizer_contract
            .special_token_ids
            .insert("overflow".to_owned(), u64::from(u32::MAX) + 1);
    });
    assert!(matches!(
        error,
        TokenizerError::TokenIdOverflow {
            context: TokenIdContextV1::SpecialRole,
            value: 4_294_967_296
        }
    ));
    drop(fixture);
}

#[test]
fn vocab_size_mismatch_is_rejected() {
    let (fixture, error) = fixture_lock_variant("vocab-mismatch", |lock| {
        lock.model.tokenizer_contract.vocab_size = 7;
    });
    assert!(matches!(
        error,
        TokenizerError::VocabSizeMismatch {
            lock: 7,
            tokenizer: 8
        }
    ));
    drop(fixture);
}

#[test]
fn typed_special_roles_require_known_ids_and_added_entries() {
    let (fixture, unknown) = fixture_lock_variant("special-unknown", |lock| {
        lock.model
            .tokenizer_contract
            .special_token_ids
            .insert("unknown".to_owned(), 12);
    });
    assert!(matches!(
        unknown,
        TokenizerError::SpecialTokenIdMissing { id: 12, .. }
    ));
    drop(fixture);

    let (fixture, missing_added) = fixture_lock_variant("special-missing-added", |lock| {
        lock.model
            .tokenizer_contract
            .special_token_ids
            .insert("base".to_owned(), 1);
    });
    assert!(matches!(
        missing_added,
        TokenizerError::SpecialTokenDecoderMissing { id: 1, .. }
    ));
    drop(fixture);
}

#[test]
fn typed_special_roles_reject_duplicate_ids() {
    let (fixture, error) = fixture_lock_variant("special-duplicate-id", |lock| {
        lock.model
            .tokenizer_contract
            .special_token_ids
            .insert("bos".to_owned(), 8);
    });
    assert!(matches!(
        error,
        TokenizerError::DuplicateSpecialId { id: 8, .. }
    ));
    drop(fixture);
}

#[test]
fn eos_forward_identity_and_special_checks_are_fail_closed() {
    let (fixture, forward) = fixture_lock_variant("eos-forward", |lock| {
        lock.model.tokenizer_contract.stop_identity.config_eos.token =
            "<|not-in-vocab|>".to_owned();
    });
    assert!(matches!(
        forward,
        TokenizerError::EosTokenToIdMismatch {
            identity: EosIdentityV1::Config,
            id: 8
        }
    ));
    drop(fixture);

    let mut tokenizer_bytes = String::from_utf8(
        include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json").to_vec(),
    )
    .unwrap();
    tokenizer_bytes = replace_once(
        tokenizer_bytes,
        "      \"id\": 9,\n      \"content\": \"<|im_end|>\",\n      \"single_word\": false,\n      \"lstrip\": false,\n      \"rstrip\": false,\n      \"normalized\": false,\n      \"special\": true",
        "      \"id\": 9,\n      \"content\": \"<|im_end|>\",\n      \"single_word\": false,\n      \"lstrip\": false,\n      \"rstrip\": false,\n      \"normalized\": false,\n      \"special\": false",
    );
    let fixture = fixture_from_bytes("eos-special", tokenizer_bytes.as_bytes());
    let error = TokenizerFrontendV1::from_verified_cache(&fixture.lock, &fixture.cache)
        .expect_err("non-special EOS must fail");
    assert!(matches!(
        error,
        TokenizerError::EosAddedTokenNotMarkedSpecial {
            identity: EosIdentityV1::Tokenizer,
            id: 9
        }
    ));
}

#[test]
fn invalid_stop_policy_and_contract_eos_id_are_rejected() {
    let (fixture, policy_error) = fixture_lock_variant("invalid-stop-policy", |lock| {
        lock.model.tokenizer_contract.generation_stop_policy.version = 2;
    });
    assert_eq!(policy_error, TokenizerError::InvalidGenerationStopPolicy);
    drop(fixture);

    let (fixture, contract_error) = fixture_lock_variant("contract-eos-mismatch", |lock| {
        lock.model.tokenizer_contract.eos_token_id = 9;
    });
    assert!(matches!(
        contract_error,
        TokenizerError::EosContractMismatch {
            contract_id: 9,
            config_id: 8
        }
    ));
    drop(fixture);
}

#[test]
fn eos_reverse_identity_check_rejects_inconsistent_id_mapping() {
    let mut tokenizer_bytes = String::from_utf8(
        include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json").to_vec(),
    )
    .unwrap();
    tokenizer_bytes = replace_once(
        tokenizer_bytes,
        "    }\n  ],\n  \"normalizer\": null",
        r#"    },
    {
      "id": 12,
      "content": "model-side",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null"#,
    );
    tokenizer_bytes = replace_once(
        tokenizer_bytes,
        "      \"42\": 6,\n      \",\": 7",
        "      \"42\": 6,\n      \"model-side\": 8",
    );
    let fixture = fixture_from_bytes("eos-reverse", tokenizer_bytes.as_bytes());
    let error = TokenizerFrontendV1::from_verified_cache(&fixture.lock, &fixture.cache)
        .expect_err("inconsistent reverse EOS mapping must fail");
    assert!(matches!(
        error,
        TokenizerError::EosIdToTokenMismatch {
            identity: EosIdentityV1::Config,
            id: 8
        }
    ));
}

#[test]
fn eos_stop_ids_are_ordered_and_exact_identity_deduplication_is_allowed() {
    let mut fixture = valid_fixture("eos-dedup");
    fixture
        .lock
        .model
        .tokenizer_contract
        .stop_identity
        .tokenizer_eos
        .token = "<|endoftext|>".to_owned();
    fixture
        .lock
        .model
        .tokenizer_contract
        .stop_identity
        .tokenizer_eos
        .token_id = 8;
    fixture.lock.model.tokenizer_contract.eos_token_id = 8;
    fixture
        .lock
        .model
        .tokenizer_contract
        .generation_stop_policy
        .stop_token_ids = vec![8];
    let tokenizer = TokenizerFrontendV1::from_verified_cache(&fixture.lock, &fixture.cache)
        .expect("identical EOS identities deduplicate");
    assert_eq!(tokenizer.encode("hello").unwrap().as_slice(), &[1]);

    let (fixture, error) = fixture_lock_variant("eos-stop-order", |lock| {
        lock.model
            .tokenizer_contract
            .generation_stop_policy
            .stop_token_ids = vec![8, 9];
    });
    assert!(matches!(error, TokenizerError::StopPolicyMismatch { .. }));
    drop(fixture);
}

#[test]
fn invalid_tokenizer_version_and_component_are_rejected_after_core_verification() {
    let mut version = String::from_utf8(
        include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json").to_vec(),
    )
    .unwrap();
    version = replace_once(version, "\"version\": \"1.0\"", "\"version\": \"2.0\"");
    let fixture = fixture_from_bytes("invalid-version", version.as_bytes());
    let error = TokenizerFrontendV1::from_verified_cache(&fixture.lock, &fixture.cache)
        .expect_err("unknown tokenizer version must fail");
    assert_eq!(error, TokenizerError::InvalidTokenizer);

    let mut component = String::from_utf8(
        include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json").to_vec(),
    )
    .unwrap();
    component = replace_once(
        component,
        "{\"type\": \"Whitespace\"}",
        "{\"type\": \"NoSuchComponent\"}",
    );
    let fixture = fixture_from_bytes("invalid-component", component.as_bytes());
    let error = TokenizerFrontendV1::from_verified_cache(&fixture.lock, &fixture.cache)
        .expect_err("unknown tokenizer component must fail");
    assert_eq!(error, TokenizerError::InvalidTokenizer);
}

#[test]
fn unknown_decode_ids_fail_before_decoder() {
    let fixture = valid_fixture("unknown-decode");
    let tokenizer = frontend(&fixture);
    let error = tokenizer
        .decode(
            &TokenIdsV1::from_slice(&[1, 12, 2]),
            DecodeModeV1::PreserveSpecialTokens,
        )
        .expect_err("unknown ID must fail");
    assert_eq!(error, TokenizerError::UnknownTokenId { id: 12 });
}

#[test]
fn verified_cache_rejects_changed_tokenizer_bytes_before_frontend() {
    let fixture = valid_fixture("core-rejection");
    let mut changed = include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json").to_vec();
    changed.extend_from_slice(b"\n");
    fs::write(fixture.directory.0.join("tokenizer.json"), changed).expect("change cache asset");
    assert!(verify_model_cache(&fixture.lock, &fixture.directory.0).is_err());
}

#[test]
fn contract_fixture_uses_non_power_of_two_and_boundary_ids() {
    let fixture = valid_fixture("boundaries");
    let tokenizer = frontend(&fixture);
    for id in [0, 1, 7, 8, 9, 10, 11] {
        let decoded = tokenizer
            .decode(
                &TokenIdsV1::from_slice(&[id]),
                DecodeModeV1::PreserveSpecialTokens,
            )
            .expect("fixture ID decodes");
        assert!(!decoded.is_empty());
    }
    assert_eq!(fixture.lock.model.tokenizer_contract.vocab_size, 8);
    assert_eq!(
        fixture.lock.model.tokenizer_contract.special_token_ids,
        BTreeMap::from([(String::from("bos"), 10), (String::from("eos"), 8)])
    );
}
