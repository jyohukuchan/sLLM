use std::collections::HashSet;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use sllm_core::{
    ModelError, TensorDType, fingerprint_for_json, parse_model_lock, read_model_lock,
    verify_model_cache,
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

fn fixture_copy(label: &str) -> TestDirectory {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "sllm-model-contract-{label}-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir(&path).expect("create test directory");
    for entry in fs::read_dir(repository_path("ci/fixtures/model-lock-v1/cache"))
        .expect("read fixture cache")
    {
        let entry = entry.expect("read fixture entry");
        fs::copy(entry.path(), path.join(entry.file_name())).expect("copy fixture file");
    }
    TestDirectory(path)
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
    let external = temporary.0.join("external-config");
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
