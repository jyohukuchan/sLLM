use std::collections::{HashMap, HashSet};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;
use sha2::{Digest, Sha256};
use sllm_core::{
    FrontendAssetKind, ModelError, TensorDType, fingerprint_for_json, parse_model_lock,
    read_model_lock, verify_model_cache,
};

fn repository_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

#[test]
fn fixture_lock_and_read_only_tensor_descriptor_validate() {
    let lock_path = repository_path("ci/fixtures/model-lock-v1/lock.json");
    let cache_path = repository_path("ci/fixtures/model-lock-v1/cache");
    let lock = read_model_lock(lock_path).expect("tiny lock parses");
    let cache = verify_model_cache(&lock, cache_path).expect("tiny cache validates");
    let tensor = cache
        .tensor("fixture.tensor")
        .expect("fixture tensor is indexed");
    assert_eq!(tensor.dtype, TensorDType::Bf16);
    assert_eq!(tensor.shape, [2]);
    assert_eq!(tensor.absolute_byte_range, [107, 111]);
    assert_eq!(tensor.byte_size, 4);
    assert_eq!(cache.files.len(), 6);
}

#[test]
fn fingerprint_is_only_schema_version_and_model() {
    let path = repository_path("ci/fixtures/model-lock-v1/lock.json");
    let bytes = std::fs::read(path).expect("fixture lock exists");
    let lock = parse_model_lock(&bytes).expect("fixture lock parses");
    assert_eq!(
        fingerprint_for_json(&bytes).expect("fingerprint computes"),
        lock.fingerprint()
    );
}

#[test]
fn generation_stop_policy_accessor_is_typed_and_preserves_order() {
    let tiny = read_model_lock(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("tiny lock parses");
    let policy = tiny.generation_stop_policy();
    assert_eq!(policy.version, 1);
    assert_eq!(policy.stop_token_ids, [0]);
    assert!(!policy.stop_token.visible_output);
    assert!(!policy.stop_token.subsequent_decode_input);
    assert_eq!(policy.reason_version, 1);
    assert_eq!(
        serde_json::to_value(policy).expect("policy serializes"),
        serde_json::json!({
            "version": 1,
            "stop_token_ids": [0],
            "evaluation": "newly_generated_after_argmax",
            "prompt_evaluation": "never_stop",
            "stop_token": {
                "visible_output": false,
                "subsequent_decode_input": false
            },
            "budget_boundary": "stop_token_wins",
            "max_new_tokens_zero": "max_new_tokens_before_decode",
            "reason_version": 1
        })
    );

    let qwen = read_model_lock(repository_path("docs/models/locks/qwen3.5-4b-bf16.json"))
        .expect("Qwen lock parses");
    assert_eq!(
        qwen.generation_stop_policy().stop_token_ids,
        [248046, 248044]
    );
}

#[test]
fn qwen_text_config_retains_typed_full_attention_contract() {
    let qwen = read_model_lock(repository_path("docs/models/locks/qwen3.5-4b-bf16.json"))
        .expect("Qwen lock parses");
    let text = &qwen.model().architecture.text_config;
    assert!(!text.attention_bias);
    assert_eq!(text.attention_dropout, "0");
    assert!(text.attn_output_gate);
    assert_eq!(text.max_position_embeddings, 262144);
    assert!(text.use_cache);

    assert_eq!(text.rope_parameters.rope_type, sllm_core::RopeType::Default);
    assert_eq!(text.rope_parameters.rope_theta, 10_000_000);
    assert_eq!(text.rope_parameters.partial_rotary_factor, "0.25");
    assert!(text.rope_parameters.mrope_interleaved);
    assert_eq!(text.rope_parameters.mrope_section, [11, 11, 10]);
}

#[test]
fn reviewed_qwen35_family_locks_preserve_shape_and_output_contracts() {
    let cases = [
        ("qwen3.5-2b-bf16.json", 2_048, 24, 8, 2, 6_144, true, 632),
        ("qwen3.5-4b-bf16.json", 2_560, 32, 16, 4, 9_216, true, 738),
        ("qwen3.5-9b-bf16.json", 4_096, 32, 16, 4, 12_288, false, 775),
    ];
    for (file, hidden, layers, heads, kv_heads, intermediate, tied, tensors) in cases {
        let lock = read_model_lock(repository_path(&format!("docs/models/locks/{file}")))
            .unwrap_or_else(|error| panic!("{file} must parse: {error}"));
        let text = &lock.model().architecture.text_config;
        assert_eq!(text.hidden_size, hidden, "{file}");
        assert_eq!(text.num_hidden_layers, layers, "{file}");
        assert_eq!(text.num_attention_heads, heads, "{file}");
        assert_eq!(text.num_key_value_heads, kv_heads, "{file}");
        assert_eq!(text.intermediate_size, intermediate, "{file}");
        assert_eq!(text.tie_word_embeddings, tied, "{file}");
        assert_eq!(
            lock.model().tensor_contract.indexed_tensor_count,
            tensors,
            "{file}"
        );
        assert_eq!(text.layer_types.len(), layers as usize, "{file}");
    }
}

#[test]
#[cfg(feature = "reviewed-qwen35-external-cache")]
fn reviewed_qwen35_external_caches_build_load_plans_and_graphs() {
    let cases = [
        ("qwen3.5-2b-bf16.json", "SLLM_QWEN35_2B_CACHE", 320, true),
        ("qwen3.5-9b-bf16.json", "SLLM_QWEN35_9B_CACHE", 427, false),
    ];
    for (file, variable, required_count, tied) in cases {
        let cache_path = std::env::var_os(variable)
            .unwrap_or_else(|| panic!("{variable} must name the exact external cache"));
        let lock = read_model_lock(repository_path(&format!("docs/models/locks/{file}")))
            .unwrap_or_else(|error| panic!("{file} must parse: {error}"));
        let cache = verify_model_cache(&lock, cache_path)
            .unwrap_or_else(|error| panic!("{file} cache must verify: {error}"));
        let plan = sllm_core::build_verified_weight_load_plan(&lock, &cache)
            .unwrap_or_else(|error| panic!("{file} plan must build: {error}"));
        let required = plan
            .entries
            .iter()
            .filter(|entry| entry.classification == sllm_core::WeightClassification::Required)
            .count();
        assert_eq!(required, required_count, "{file}");
        assert_eq!(plan.tied_embeddings, tied, "{file}");
        let lm_head = plan
            .entries
            .iter()
            .find(|entry| entry.tensor_name == "lm_head.weight");
        if tied {
            assert!(lm_head.is_none(), "{file} must alias embedding output");
        } else {
            let lm_head = lm_head.expect("untied model must load lm_head.weight");
            let consumer = lm_head.consumer.expect("lm_head consumer must be typed");
            assert_eq!(consumer.layer, None, "{file}");
            assert_eq!(
                consumer.role,
                sllm_core::WeightConsumer::OutputProjection,
                "{file}"
            );
            assert_eq!(
                lm_head.classification,
                sllm_core::WeightClassification::Required,
                "{file}"
            );
        }
        for token_count in [1, 3, 17, 255, 256, 257] {
            let graph = sllm_core::build_qwen35_graph(&lock, &plan, token_count, 257)
                .unwrap_or_else(|error| {
                    panic!("{file} graph for {token_count} tokens must build: {error}")
                });
            assert_eq!(
                graph.layer_types().len(),
                lock.model().architecture.text_config.num_hidden_layers as usize
            );
            assert_eq!(graph.weight_bindings().len(), required_count, "{file}");
            let output = graph
                .weight_bindings()
                .iter()
                .find(|binding| {
                    binding.consumer().role
                        == if tied {
                            sllm_core::WeightConsumer::EmbeddingAndTiedOutput
                        } else {
                            sllm_core::WeightConsumer::OutputProjection
                        }
                })
                .expect("output projection binding must be explicit");
            assert_eq!(
                output.tensor_name(),
                if tied {
                    "model.language_model.embed_tokens.weight"
                } else {
                    "lm_head.weight"
                }
            );
        }
    }
}

#[test]
fn generation_stop_policy_shape_mutations_fail_closed() {
    let baseline = fs::read_to_string(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("tiny lock exists");
    let mutations = [
        ("missing-version", "        \"version\": 1,\n", ""),
        (
            "unknown-field",
            "        \"reason_version\": 1\n",
            "        \"reason_version\": 1,\n        \"unknown\": 0\n",
        ),
        (
            "unknown-version",
            "        \"version\": 1,",
            "        \"version\": 2,",
        ),
        (
            "unknown-reason-version",
            "        \"reason_version\": 1\n",
            "        \"reason_version\": 2\n",
        ),
        (
            "missing-ids",
            "        \"stop_token_ids\": [0],",
            "        \"stop_token_ids\": [],",
        ),
        (
            "duplicate-ids",
            "        \"stop_token_ids\": [0],",
            "        \"stop_token_ids\": [0, 0],",
        ),
        (
            "negative-id",
            "        \"stop_token_ids\": [0],",
            "        \"stop_token_ids\": [-1],",
        ),
        (
            "overflow-id",
            "        \"stop_token_ids\": [0],",
            "        \"stop_token_ids\": [4294967296],",
        ),
        (
            "boolean-id",
            "        \"stop_token_ids\": [0],",
            "        \"stop_token_ids\": [true],",
        ),
        (
            "float-id",
            "        \"stop_token_ids\": [0],",
            "        \"stop_token_ids\": [0.0],",
        ),
        (
            "string-id",
            "        \"stop_token_ids\": [0],",
            "        \"stop_token_ids\": [\"0\"],",
        ),
        (
            "enum-string",
            "        \"evaluation\": \"newly_generated_after_argmax\",",
            "        \"evaluation\": \"argmax\",",
        ),
        (
            "boolean-handling",
            "\"visible_output\": false",
            "\"visible_output\": true",
        ),
    ];
    for (label, from, to) in mutations {
        let changed = baseline.replacen(from, to, 1);
        assert_ne!(changed, baseline, "mutation must change fixture: {label}");
        assert!(
            parse_model_lock(changed.as_bytes()).is_err(),
            "must reject {label}"
        );
    }
}

#[test]
fn duplicate_unknown_and_floating_lock_input_fail_closed() {
    let duplicate = br#"{"schema_version":"model-lock-v1","schema_version":"model-lock-v1"}"#;
    assert!(matches!(
        parse_model_lock(duplicate),
        Err(ModelError::Json(_))
    ));
    let unknown = br#"{"schema_version":"model-lock-v1","unknown":1}"#;
    assert!(parse_model_lock(unknown).is_err());
    let floating = br#"{"schema_version":1.5}"#;
    assert!(parse_model_lock(floating).is_err());
}

#[test]
fn lock_controls_calendar_and_parser_bounds_fail_closed() {
    let bytes = std::fs::read(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("fixture lock exists");
    let text = String::from_utf8(bytes).expect("fixture lock is UTF-8");
    for replacement in [
        "fixture\\nrevision",
        "fixture\\u0085revision",
        "fixture\\u007frevision",
    ] {
        let mutated = text.replacen("fixture-v1", replacement, 1);
        assert!(parse_model_lock(mutated.as_bytes()).is_err());
    }
    let invalid_day = text.replacen("2026-08-04T00:00:00Z", "2026-02-30T00:00:00Z", 1);
    assert!(parse_model_lock(invalid_day.as_bytes()).is_err());
    let offset_timestamp = text.replacen("2026-08-04T00:00:00Z", "2026-08-04T00:00:00+00:00", 1);
    assert!(parse_model_lock(offset_timestamp.as_bytes()).is_err());
    let nested = format!("{}null{}", "[".repeat(65), "]".repeat(65));
    assert!(parse_model_lock(nested.as_bytes()).is_err());
    let oversized = format!("{{\"s\":\"{}\"}}", "x".repeat(1024 * 1024 + 1));
    assert!(parse_model_lock(oversized.as_bytes()).is_err());
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn test_directory(label: &str) -> TestDirectory {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "sllm-model-contract-{label}-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir(&path).expect("create test directory");
    TestDirectory(path)
}

fn fixture_copy(label: &str) -> TestDirectory {
    let temporary = test_directory(label);
    for entry in fs::read_dir(repository_path("ci/fixtures/model-lock-v1/cache"))
        .expect("read fixture cache")
    {
        let entry = entry.expect("read fixture entry");
        fs::copy(entry.path(), temporary.0.join(entry.file_name())).expect("copy fixture file");
    }
    temporary
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn lock_with_asset(relative: &str, bytes: &[u8]) -> sllm_core::ModelLock {
    let source = fs::read(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("fixture lock exists");
    let mut document: Value = serde_json::from_slice(&source).expect("fixture lock is JSON");
    let files = document["model"]["files"]
        .as_array_mut()
        .expect("fixture lock has files");
    let entry = files
        .iter_mut()
        .find(|file| file["path"].as_str() == Some(relative))
        .expect("asset is present in the fixture lock");
    entry["size_bytes"] = Value::from(u64::try_from(bytes.len()).expect("test asset fits u64"));
    entry["sha256"] = Value::String(sha256_hex(bytes));

    let fingerprint_input = serde_json::to_vec_pretty(&document).expect("serialize lock input");
    document["fingerprint"] = Value::String(
        fingerprint_for_json(&fingerprint_input).expect("recompute fixture fingerprint"),
    );
    let updated = serde_json::to_vec_pretty(&document).expect("serialize updated lock");
    parse_model_lock(&updated).expect("updated fixture lock parses")
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RegularFileIdentity {
    device: u64,
    inode: u64,
}

fn regular_file_identity(path: &std::path::Path) -> RegularFileIdentity {
    let metadata = fs::metadata(path).expect("fixture path metadata is available");
    assert!(metadata.is_file(), "fixture path must be a regular file");
    RegularFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

fn fixture_file_identities(directory: &std::path::Path) -> HashSet<RegularFileIdentity> {
    fs::read_dir(directory)
        .expect("read fixture directory")
        .map(|entry| {
            let entry = entry.expect("read fixture entry");
            regular_file_identity(&entry.path())
        })
        .collect()
}

fn open_regular_file_identities() -> HashSet<RegularFileIdentity> {
    let mut identities = HashSet::new();
    for entry in fs::read_dir("/proc/self/fd").expect("Linux exposes process descriptors") {
        let entry = entry.unwrap_or_else(|error| panic!("read /proc/self/fd entry: {error}"));
        let path = entry.path();
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => panic!("metadata for /proc/self/fd entry {path:?}: {error}"),
        };
        if metadata.is_file() {
            identities.insert(RegularFileIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
            });
        }
    }
    identities
}

fn open_regular_file_identity_counts() -> HashMap<RegularFileIdentity, usize> {
    let mut counts = HashMap::new();
    for entry in fs::read_dir("/proc/self/fd").expect("Linux exposes process descriptors") {
        let entry = entry.unwrap_or_else(|error| panic!("read /proc/self/fd entry: {error}"));
        let path = entry.path();
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => panic!("metadata for /proc/self/fd entry {path:?}: {error}"),
        };
        if metadata.is_file() {
            let identity = RegularFileIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
            };
            *counts.entry(identity).or_insert(0) += 1;
        }
    }
    counts
}

fn assert_target_fd_count(targets: &HashSet<RegularFileIdentity>, expected: usize, context: &str) {
    let counts = open_regular_file_identity_counts();
    for target in targets {
        assert_eq!(
            counts.get(target).copied().unwrap_or(0),
            expected,
            "{context}: unexpected descriptor count for {target:?}",
        );
    }
}

fn assert_no_open_target_identities(targets: &HashSet<RegularFileIdentity>, context: &str) {
    let open = open_regular_file_identities();
    let remaining: Vec<_> = targets.intersection(&open).copied().collect();
    assert!(
        remaining.is_empty(),
        "{context}: target regular-file descriptors remain open: {remaining:?}"
    );
}

fn same_size_replacement(bytes: &[u8]) -> Vec<u8> {
    let mut replacement = bytes.to_vec();
    let index = replacement
        .iter()
        .position(|byte| *byte == b'\n')
        .expect("fixture contains whitespace");
    replacement[index] = b' ';
    replacement
}

#[test]
fn lock_reader_binds_one_regular_descriptor_and_rejects_links_and_oversize_inputs() {
    let source = repository_path("ci/fixtures/model-lock-v1/lock.json");
    let temporary = fixture_copy("lock-reader");
    let lock_path = temporary.0.join("lock.json");
    fs::copy(&source, &lock_path).expect("copy fixture lock");
    let lock_identity = regular_file_identity(&lock_path);
    for _ in 0..32 {
        read_model_lock(&lock_path).expect("regular copied lock parses through its bound FD");
    }

    let link = temporary.0.join("lock-link.json");
    std::os::unix::fs::symlink(&lock_path, &link).expect("create lock symlink");
    assert!(
        read_model_lock(&link).is_err(),
        "lock symlinks must not be followed"
    );

    let oversized = temporary.0.join("oversized-lock.json");
    fs::copy(&source, &oversized).expect("copy oversized fixture lock");
    fs::File::options()
        .write(true)
        .open(&oversized)
        .expect("open oversized fixture lock")
        .set_len(1024 * 1024 + 1)
        .expect("extend sparse oversized fixture lock");
    let oversized_identity = regular_file_identity(&oversized);
    assert!(
        read_model_lock(&oversized).is_err(),
        "lock size must be bounded before allocation"
    );
    assert_no_open_target_identities(
        &HashSet::from([lock_identity, oversized_identity]),
        "repeated bound lock reads must close their descriptors",
    );
    drop(temporary);
}

#[test]
fn hashed_fd_binding_rejects_path_and_same_inode_races_and_recovers_files() {
    let lock = read_model_lock(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("tiny lock parses");

    {
        let temporary = fixture_copy("thread-recovery");
        let target_identities = fixture_file_identities(&temporary.0);
        {
            let cache = verify_model_cache(&lock, &temporary.0).expect("cache validates");
            std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        assert_eq!(
                            cache
                                .read_tensor_range("fixture.tensor", 0, 4)
                                .expect("positional range read"),
                            vec![0, 1, 2, 3]
                        );
                    })
                    .join()
                    .expect("range-read thread succeeds");
            });
        }
        assert_no_open_target_identities(
            &target_identities,
            "verified files are closed on drop without descriptor growth",
        );
        drop(temporary);
    }

    let temporary = fixture_copy("path-replacement");
    let cache = verify_model_cache(&lock, &temporary.0).expect("cache validates");
    let original = fs::read(temporary.0.join("config.json")).expect("read config");
    let replacement = same_size_replacement(&original);
    let replacement_path = temporary.0.join("replacement");
    fs::write(&replacement_path, replacement).expect("write replacement");
    fs::rename(&replacement_path, temporary.0.join("config.json")).expect("replace config path");
    assert!(
        cache.read_tensor_range("fixture.tensor", 0, 4).is_err(),
        "a verified FD must not silently follow a replaced cache path"
    );
    drop(cache);

    let temporary = fixture_copy("same-inode");
    let cache = verify_model_cache(&lock, &temporary.0).expect("cache validates");
    let config = temporary.0.join("config.json");
    let bytes = fs::read(&config).expect("read config");
    let mutation = same_size_replacement(&bytes);
    fs::write(&config, mutation).expect("mutate the same inode without changing its size");
    assert!(
        cache.read_tensor_range("fixture.tensor", 0, 4).is_err(),
        "same-inode same-size mutation must invalidate the FD binding"
    );
    drop(cache);

    for label in ["truncate", "extend"] {
        let temporary = fixture_copy(label);
        let cache = verify_model_cache(&lock, &temporary.0).expect("cache validates");
        let config = temporary.0.join("config.json");
        if label == "truncate" {
            fs::File::options()
                .write(true)
                .open(&config)
                .expect("open config")
                .set_len(1)
                .expect("truncate config");
        } else {
            let mut bytes = fs::read(&config).expect("read config");
            bytes.push(b' ');
            fs::write(&config, bytes).expect("extend config");
        }
        assert!(cache.read_tensor_range("fixture.tensor", 0, 4).is_err());
    }

    let temporary = fixture_copy("hardlink");
    let source_directory = test_directory("hardlink-source");
    let external = source_directory.0.join("external-config");
    fs::copy(temporary.0.join("config.json"), &external).expect("copy external config");
    fs::remove_file(temporary.0.join("config.json")).expect("remove fixture config");
    fs::hard_link(&external, temporary.0.join("config.json")).expect("create hardlink");
    assert!(verify_model_cache(&lock, &temporary.0).is_err());

    let temporary = fixture_copy("symlink");
    fs::remove_file(temporary.0.join("config.json")).expect("remove fixture config");
    std::os::unix::fs::symlink(
        repository_path("ci/fixtures/model-lock-v1/cache/config.json"),
        temporary.0.join("config.json"),
    )
    .expect("create symlink");
    assert!(verify_model_cache(&lock, &temporary.0).is_err());
}

fn assert_error_contains(result: Result<Vec<u8>, ModelError>, needle: &str) {
    let error = result.expect_err("operation must fail");
    assert!(
        error.to_string().contains(needle),
        "error {error:?} does not contain stable semantic substring {needle:?}"
    );
}

#[test]
fn fixed_frontend_assets_read_exact_bytes_from_temporary_cache() {
    let lock = read_model_lock(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("tiny lock parses");
    let temporary = fixture_copy("frontend-exact");
    let cache = verify_model_cache(&lock, &temporary.0).expect("tiny cache validates");
    let assets = [
        (FrontendAssetKind::ConfigJson, "config.json"),
        (FrontendAssetKind::TokenizerJson, "tokenizer.json"),
        (
            FrontendAssetKind::TokenizerConfigJson,
            "tokenizer_config.json",
        ),
        (FrontendAssetKind::ChatTemplateJinja, "chat_template.jinja"),
    ];
    for (kind, relative) in assets {
        assert_eq!(
            cache
                .read_frontend_asset(kind)
                .expect("fixed frontend asset read succeeds"),
            fs::read(temporary.0.join(relative)).expect("temporary asset exists"),
            "asset bytes must be exact for {relative}",
        );
    }
}

#[test]
fn chat_template_cap_is_inclusive_and_checked_before_read_allocation() {
    const CAP: usize = 64 * 1024;
    for (label, length, succeeds) in [
        ("template-cap-minus-one", CAP - 1, true),
        ("template-cap", CAP, true),
        ("template-cap-plus-one", CAP + 1, false),
    ] {
        let temporary = fixture_copy(label);
        let bytes = vec![b'x'; length];
        fs::write(temporary.0.join("chat_template.jinja"), &bytes)
            .expect("write generated template");
        let lock = lock_with_asset("chat_template.jinja", &bytes);
        let cache = verify_model_cache(&lock, &temporary.0).expect("generated cache validates");
        let result = cache.read_frontend_asset(FrontendAssetKind::ChatTemplateJinja);
        if succeeds {
            assert_eq!(result.expect("cap-bound template read succeeds"), bytes);
        } else {
            assert_error_contains(result, "frontend asset chat_template.jinja");
            assert_error_contains(
                cache.read_frontend_asset(FrontendAssetKind::ChatTemplateJinja),
                "bounded read limit",
            );
        }
    }
}

#[test]
fn frontend_asset_read_rechecks_root_and_path_bindings() {
    let lock = read_model_lock(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("tiny lock parses");

    let temporary = fixture_copy("frontend-path-replacement");
    let cache = verify_model_cache(&lock, &temporary.0).expect("tiny cache validates");
    let replacement = temporary.0.join("replacement");
    let original = fs::read(temporary.0.join("config.json")).expect("read config");
    fs::write(&replacement, same_size_replacement(&original)).expect("write replacement");
    fs::rename(&replacement, temporary.0.join("config.json")).expect("replace config path");
    assert_error_contains(
        cache.read_frontend_asset(FrontendAssetKind::ConfigJson),
        "changed during frontend asset read",
    );
    drop(cache);

    let temporary = fixture_copy("frontend-symlink-replacement");
    let cache = verify_model_cache(&lock, &temporary.0).expect("tiny cache validates");
    let config = temporary.0.join("config.json");
    let external = temporary.0.join("external-config");
    fs::copy(&config, &external).expect("copy external config");
    fs::remove_file(&config).expect("remove config");
    std::os::unix::fs::symlink(&external, &config).expect("replace config with symlink");
    assert_error_contains(
        cache.read_frontend_asset(FrontendAssetKind::ConfigJson),
        "changed during frontend asset read",
    );
    drop(cache);

    let temporary = fixture_copy("frontend-hardlink-replacement");
    let cache = verify_model_cache(&lock, &temporary.0).expect("tiny cache validates");
    let config = temporary.0.join("config.json");
    let external = temporary.0.join("external-config");
    fs::copy(&config, &external).expect("copy hardlink source");
    fs::remove_file(&config).expect("remove config");
    fs::hard_link(&external, &config).expect("replace config with hardlink");
    assert_error_contains(
        cache.read_frontend_asset(FrontendAssetKind::ConfigJson),
        "changed during frontend asset read",
    );
    drop(cache);

    let temporary = fixture_copy("frontend-root-replacement");
    let cache = verify_model_cache(&lock, &temporary.0).expect("tiny cache validates");
    let moved = temporary.0.with_extension("moved");
    fs::rename(&temporary.0, &moved).expect("move cache root");
    std::os::unix::fs::symlink(&moved, &temporary.0).expect("replace cache root with symlink");
    assert_error_contains(
        cache.read_frontend_asset(FrontendAssetKind::ConfigJson),
        "cache root changed during frontend asset read",
    );
    fs::remove_file(&temporary.0).expect("remove root symlink");
    fs::rename(&moved, &temporary.0).expect("restore cache root");
    drop(cache);
}

#[test]
fn frontend_asset_read_rejects_same_inode_mutation_truncation_and_extension() {
    let lock = read_model_lock(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("tiny lock parses");

    let temporary = fixture_copy("frontend-same-inode");
    let cache = verify_model_cache(&lock, &temporary.0).expect("tiny cache validates");
    let config = temporary.0.join("config.json");
    let original = fs::read(&config).expect("read config");
    fs::write(&config, same_size_replacement(&original)).expect("mutate config in place");
    assert_error_contains(
        cache.read_frontend_asset(FrontendAssetKind::ConfigJson),
        "verified file changed during frontend asset read",
    );
    drop(cache);

    for (label, truncate) in [("frontend-truncate", true), ("frontend-extend", false)] {
        let temporary = fixture_copy(label);
        let cache = verify_model_cache(&lock, &temporary.0).expect("tiny cache validates");
        let config = temporary.0.join("config.json");
        if truncate {
            fs::File::options()
                .write(true)
                .open(&config)
                .expect("open config")
                .set_len(1)
                .expect("truncate config");
        } else {
            let mut bytes = fs::read(&config).expect("read config");
            bytes.push(b' ');
            fs::write(&config, bytes).expect("extend config");
        }
        assert_error_contains(
            cache.read_frontend_asset(FrontendAssetKind::ConfigJson),
            "verified file changed during frontend asset read",
        );
    }
}

#[test]
fn concurrent_positional_whole_file_reads_are_repeatable() {
    let lock = read_model_lock(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("tiny lock parses");
    let temporary = fixture_copy("frontend-concurrent");
    let cache = verify_model_cache(&lock, &temporary.0).expect("tiny cache validates");
    let assets = [
        (
            FrontendAssetKind::ConfigJson,
            fs::read(temporary.0.join("config.json")).expect("read config"),
        ),
        (
            FrontendAssetKind::TokenizerJson,
            fs::read(temporary.0.join("tokenizer.json")).expect("read tokenizer"),
        ),
        (
            FrontendAssetKind::TokenizerConfigJson,
            fs::read(temporary.0.join("tokenizer_config.json")).expect("read tokenizer config"),
        ),
        (
            FrontendAssetKind::ChatTemplateJinja,
            fs::read(temporary.0.join("chat_template.jinja")).expect("read template"),
        ),
    ];
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..32 {
                    for (kind, expected) in &assets {
                        assert_eq!(
                            cache
                                .read_frontend_asset(*kind)
                                .expect("concurrent positional read succeeds"),
                            *expected,
                        );
                    }
                }
            });
        }
    });
}

#[test]
fn repeated_frontend_reads_and_errors_do_not_grow_file_descriptors() {
    const TEMPLATE_CAP: usize = 64 * 1024;
    let lock = read_model_lock(repository_path("ci/fixtures/model-lock-v1/lock.json"))
        .expect("tiny lock parses");

    let temporary = fixture_copy("frontend-fd-success");
    let targets = fixture_file_identities(&temporary.0);
    assert_target_fd_count(&targets, 0, "before successful frontend reads");
    {
        let cache = verify_model_cache(&lock, &temporary.0).expect("tiny cache validates");
        assert_target_fd_count(&targets, 1, "after cache verification");
        for _ in 0..32 {
            for kind in [
                FrontendAssetKind::ConfigJson,
                FrontendAssetKind::TokenizerJson,
                FrontendAssetKind::TokenizerConfigJson,
                FrontendAssetKind::ChatTemplateJinja,
            ] {
                cache
                    .read_frontend_asset(kind)
                    .expect("repeated frontend read succeeds");
            }
        }
        assert_target_fd_count(&targets, 1, "after repeated successful reads");
    }
    assert_target_fd_count(&targets, 0, "after successful cache drop");
    drop(temporary);

    let bytes = vec![b'x'; TEMPLATE_CAP + 1];
    let temporary = fixture_copy("frontend-fd-error");
    fs::write(temporary.0.join("chat_template.jinja"), &bytes)
        .expect("write generated oversized template");
    let lock = lock_with_asset("chat_template.jinja", &bytes);
    let targets = fixture_file_identities(&temporary.0);
    let cache = verify_model_cache(&lock, &temporary.0).expect("oversized cache validates");
    assert_target_fd_count(&targets, 1, "after error cache verification");
    for _ in 0..64 {
        assert_error_contains(
            cache.read_frontend_asset(FrontendAssetKind::ChatTemplateJinja),
            "bounded read limit",
        );
    }
    assert_target_fd_count(&targets, 1, "after repeated frontend errors");
    drop(cache);
    assert_target_fd_count(&targets, 0, "after error cache drop");
}
