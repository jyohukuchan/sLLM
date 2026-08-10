use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};
use sllm_core::{
    ModelLock, VerifiedCache, fingerprint_for_json, parse_model_lock, verify_model_cache,
};
use sllm_frontend::{
    ChatRenderError, QWEN35_CHAT_TEMPLATE_FILENAME, QWEN35_CHAT_TEMPLATE_SHA256,
    QWEN35_CHAT_TEMPLATE_SIZE_BYTES, Qwen35ChatTemplateV1,
};

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
        "sllm-chat-contract-{label}-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir(&path).expect("create chat test directory");
    TestDirectory(path)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn replace_once(source: String, old: &str, new: &str) -> String {
    assert_eq!(source.matches(old).count(), 1, "replacement must be unique");
    source.replacen(old, new, 1)
}

fn fixture_with_spoofed_metadata(template_bytes: &[u8]) -> Fixture {
    assert_eq!(
        template_bytes.len(),
        QWEN35_CHAT_TEMPLATE_SIZE_BYTES as usize
    );
    let directory = test_directory("spoofed");
    let base_cache = repository_path("ci/fixtures/model-lock-v1/cache");
    for entry in fs::read_dir(base_cache).expect("read base cache") {
        let entry = entry.expect("read base cache entry");
        fs::copy(entry.path(), directory.0.join(entry.file_name())).expect("copy cache file");
    }
    fs::write(
        directory.0.join(QWEN35_CHAT_TEMPLATE_FILENAME),
        template_bytes,
    )
    .expect("write synthetic bounded template carrier");

    let source = String::from_utf8(
        fs::read(repository_path("ci/fixtures/model-lock-v1/lock.json")).expect("base lock exists"),
    )
    .expect("base lock is UTF-8");
    let source = replace_once(
        source,
        r#"        "path": "chat_template.jinja",
        "size_bytes": 44,
        "sha256": "00458c8b559de6bbd4c15a4d6ca59b56015d25f95ca5ff29e7f5eae1d8dee31f","#,
        &format!(
            "        \"path\": \"chat_template.jinja\",\n        \"size_bytes\": {},\n        \"sha256\": \"{}\",",
            template_bytes.len(),
            sha256_hex(template_bytes)
        ),
    );
    let fingerprint =
        fingerprint_for_json(source.as_bytes()).expect("recompute fixture fingerprint");
    let source = replace_once(
        source,
        "  \"fingerprint\": \"sha256:7201b4dddf49fb09e4d871778c5dd75eaec29d3d0ab3911ae7eb7ea62548a490\",",
        &format!("  \"fingerprint\": \"{fingerprint}\","),
    );
    let mut lock = parse_model_lock(source.as_bytes()).expect("synthetic lock parses");
    let mut cache = verify_model_cache(&lock, &directory.0).expect("synthetic cache verifies");

    lock.model.repo_id = "Qwen/Qwen3.5-4B".to_owned();
    lock.model.resolved_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".to_owned();

    // Deliberately mutate the public labels after core has verified different
    // bytes. The frontend constructor must bind itself to the bytes it reads,
    // so these labels can never turn this fixture into the fixed template.
    lock.model
        .files
        .iter_mut()
        .find(|file| file.path == QWEN35_CHAT_TEMPLATE_FILENAME)
        .expect("chat lock entry")
        .sha256 = QWEN35_CHAT_TEMPLATE_SHA256.to_owned();
    cache
        .files
        .iter_mut()
        .find(|file| file.path == QWEN35_CHAT_TEMPLATE_FILENAME)
        .expect("chat cache entry")
        .sha256 = QWEN35_CHAT_TEMPLATE_SHA256.to_owned();

    Fixture {
        cache,
        lock,
        directory,
    }
}

fn same_size_utf8_spoof() -> Fixture {
    fixture_with_spoofed_metadata(&vec![b' '; QWEN35_CHAT_TEMPLATE_SIZE_BYTES as usize])
}

#[test]
fn fixed_constructor_rejects_same_size_utf8_bytes_with_spoofed_metadata() {
    let fixture = same_size_utf8_spoof();
    assert_eq!(
        Qwen35ChatTemplateV1::from_verified_cache(&fixture.lock, &fixture.cache),
        Err(ChatRenderError::UnsupportedTemplateIdentity)
    );
}

#[test]
fn fixed_constructor_rejects_wrong_repo_id() {
    let mut fixture = same_size_utf8_spoof();
    fixture.lock.model.repo_id = "Qwen/Qwen3.5-4B-Base".to_owned();
    assert_eq!(
        Qwen35ChatTemplateV1::from_verified_cache(&fixture.lock, &fixture.cache),
        Err(ChatRenderError::UnsupportedTemplateIdentity)
    );
}

#[test]
fn fixed_constructor_rejects_wrong_resolved_revision() {
    let mut fixture = same_size_utf8_spoof();
    fixture.lock.model.resolved_revision = "0123456789abcdef0123456789abcdef01234567".to_owned();
    assert_eq!(
        Qwen35ChatTemplateV1::from_verified_cache(&fixture.lock, &fixture.cache),
        Err(ChatRenderError::UnsupportedTemplateIdentity)
    );
}

#[test]
fn fixed_constructor_identity_metadata_and_bounded_read_fail_closed() {
    let mut fixture = same_size_utf8_spoof();
    fixture.lock.model.tokenizer_contract.chat_template_path = "other.jinja".to_owned();
    assert_eq!(
        Qwen35ChatTemplateV1::from_verified_cache(&fixture.lock, &fixture.cache),
        Err(ChatRenderError::UnsupportedTemplateIdentity)
    );

    let mut fixture = same_size_utf8_spoof();
    fixture
        .lock
        .model
        .files
        .iter_mut()
        .find(|file| file.path == QWEN35_CHAT_TEMPLATE_FILENAME)
        .expect("chat lock entry")
        .size_bytes -= 1;
    assert_eq!(
        Qwen35ChatTemplateV1::from_verified_cache(&fixture.lock, &fixture.cache),
        Err(ChatRenderError::UnsupportedTemplateIdentity)
    );

    let mut fixture = same_size_utf8_spoof();
    fixture
        .cache
        .files
        .iter_mut()
        .find(|file| file.path == QWEN35_CHAT_TEMPLATE_FILENAME)
        .expect("chat cache entry")
        .sha256 = "0".repeat(64);
    assert_eq!(
        Qwen35ChatTemplateV1::from_verified_cache(&fixture.lock, &fixture.cache),
        Err(ChatRenderError::UnsupportedTemplateIdentity)
    );

    let mut fixture = same_size_utf8_spoof();
    fixture.cache.lock_fingerprint = "sha256:changed".to_owned();
    assert_eq!(
        Qwen35ChatTemplateV1::from_verified_cache(&fixture.lock, &fixture.cache),
        Err(ChatRenderError::LockCacheFingerprintMismatch)
    );

    let fixture = same_size_utf8_spoof();
    fs::rename(
        fixture.directory.0.join(QWEN35_CHAT_TEMPLATE_FILENAME),
        fixture.directory.0.join("moved-template"),
    )
    .expect("move verified path");
    let error = Qwen35ChatTemplateV1::from_verified_cache(&fixture.lock, &fixture.cache)
        .expect_err("moved verified asset must fail");
    assert_eq!(error, ChatRenderError::TemplateAssetUnavailable);
    assert!(
        !error
            .to_string()
            .contains(fixture.directory.0.to_str().expect("UTF-8 temp path"))
    );
}

#[test]
fn raw_digest_rejection_precedes_utf8_validation() {
    let mut bytes = vec![b' '; QWEN35_CHAT_TEMPLATE_SIZE_BYTES as usize];
    bytes[17] = 0xff;
    let fixture = fixture_with_spoofed_metadata(&bytes);
    assert_eq!(
        Qwen35ChatTemplateV1::from_verified_cache(&fixture.lock, &fixture.cache),
        Err(ChatRenderError::UnsupportedTemplateIdentity)
    );
}
