//! Optional model-side Stage 9 vocabulary artifact for Qwen3.8 MTP.
//!
//! The generator keeps the sorted token IDs and its metadata outside the
//! repository. A model directory may opt into the reduced draft head by
//! placing both files under ".sllm/"; a missing pair preserves the historical
//! full-vocabulary MTP path, while any partial or invalid pair fails closed.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::{
    QWEN35_VOCAB_SIZE, QWEN38_MTP_DRAFT_VOCAB_SIZE, decode_qwen38_mtp_draft_vocabulary_le,
};

pub const QWEN38_MTP_DRAFT_VOCAB_RELATIVE_PATH: &str = ".sllm/mtp-draft-vocab-98304.u32";
pub const QWEN38_MTP_DRAFT_VOCAB_METADATA_RELATIVE_PATH: &str =
    ".sllm/mtp-draft-vocab-98304.metadata.json";
/// SHA-256 of the reviewed Qwen3.8 tokenizer.json in the fixed model artifact.
pub const QWEN38_TOKENIZER_SHA256: &str =
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";
pub const QWEN38_MTP_DRAFT_VOCAB_SCHEMA: &str = "qwen38-draft-vocab-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen38MtpDraftVocabularyArtifact {
    ids: Vec<u32>,
    vocab_sha256: String,
    manifest_sha256: String,
    tokenizer_sha256: String,
    special_token_ids: Vec<u32>,
}

impl Qwen38MtpDraftVocabularyArtifact {
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn vocab_sha256(&self) -> &str {
        &self.vocab_sha256
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    pub fn special_token_ids(&self) -> &[u32] {
        &self.special_token_ids
    }
}

#[derive(Debug)]
pub enum Qwen38MtpDraftVocabularyError {
    Io {
        path: PathBuf,
        message: String,
    },
    InvalidMetadata(String),
    InvalidVocabulary(String),
    DigestMismatch {
        label: &'static str,
        expected: String,
        actual: String,
    },
}

impl fmt::Display for Qwen38MtpDraftVocabularyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(formatter, "{}: {message}", path.display()),
            Self::InvalidMetadata(message) => write!(
                formatter,
                "invalid Qwen3.8 draft vocabulary metadata: {message}"
            ),
            Self::InvalidVocabulary(message) => {
                write!(formatter, "invalid Qwen3.8 draft vocabulary: {message}")
            }
            Self::DigestMismatch {
                label,
                expected,
                actual,
            } => write!(
                formatter,
                "Qwen3.8 draft vocabulary {label} digest mismatch: expected={expected} actual={actual}"
            ),
        }
    }
}

impl std::error::Error for Qwen38MtpDraftVocabularyError {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    schema_version: String,
    #[serde(rename = "N")]
    n: usize,
    vocab_count: usize,
    vocab_sha256: String,
    manifest_sha256: String,
    tokenizer_sha256: String,
    special_token_ids: Vec<u32>,
    // The generator records reproducibility statistics here. The consumer
    // does not interpret them, but accepts the current schema's field.
    #[serde(rename = "domains")]
    _domains: serde_json::Value,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && value == value.to_ascii_lowercase()
}

fn io_error(path: &Path, error: impl fmt::Display) -> Qwen38MtpDraftVocabularyError {
    Qwen38MtpDraftVocabularyError::Io {
        path: path.to_owned(),
        message: error.to_string(),
    }
}

fn regular_file(path: &Path) -> Result<Option<bool>, Qwen38MtpDraftVocabularyError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata.file_type().is_file())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io_error(path, error)),
    }
}

fn sha256_file(path: &Path) -> Result<String, Qwen38MtpDraftVocabularyError> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Load the optional model-side Qwen3.8 MTP draft vocabulary.
pub fn load_qwen38_mtp_draft_vocabulary(
    model_root: impl AsRef<Path>,
) -> Result<Option<Qwen38MtpDraftVocabularyArtifact>, Qwen38MtpDraftVocabularyError> {
    let root = model_root.as_ref();
    let vocab_path = root.join(QWEN38_MTP_DRAFT_VOCAB_RELATIVE_PATH);
    let metadata_path = root.join(QWEN38_MTP_DRAFT_VOCAB_METADATA_RELATIVE_PATH);
    let vocab_present = regular_file(&vocab_path)?;
    let metadata_present = regular_file(&metadata_path)?;
    if vocab_present.is_none() && metadata_present.is_none() {
        return Ok(None);
    }
    if vocab_present != Some(true) || metadata_present != Some(true) {
        return Err(Qwen38MtpDraftVocabularyError::InvalidMetadata(
            "vocabulary and metadata files must be present together as regular files".to_owned(),
        ));
    }

    let tokenizer_path = root.join("tokenizer.json");
    let actual_tokenizer_sha256 = sha256_file(&tokenizer_path)?;
    if actual_tokenizer_sha256 != QWEN38_TOKENIZER_SHA256 {
        return Err(Qwen38MtpDraftVocabularyError::DigestMismatch {
            label: "tokenizer",
            expected: QWEN38_TOKENIZER_SHA256.to_owned(),
            actual: actual_tokenizer_sha256,
        });
    }
    let metadata_bytes =
        fs::read(&metadata_path).map_err(|error| io_error(&metadata_path, error))?;
    let metadata: Metadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|error| Qwen38MtpDraftVocabularyError::InvalidMetadata(error.to_string()))?;
    if metadata.schema_version != QWEN38_MTP_DRAFT_VOCAB_SCHEMA {
        return Err(Qwen38MtpDraftVocabularyError::InvalidMetadata(format!(
            "schema_version must be {}",
            QWEN38_MTP_DRAFT_VOCAB_SCHEMA
        )));
    }
    if metadata.n != QWEN38_MTP_DRAFT_VOCAB_SIZE || metadata.vocab_count != metadata.n {
        return Err(Qwen38MtpDraftVocabularyError::InvalidMetadata(format!(
            "N and vocab_count must both be {}",
            QWEN38_MTP_DRAFT_VOCAB_SIZE
        )));
    }
    for (label, digest) in [
        ("vocabulary", metadata.vocab_sha256.as_str()),
        ("manifest", metadata.manifest_sha256.as_str()),
        ("tokenizer metadata", metadata.tokenizer_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            return Err(Qwen38MtpDraftVocabularyError::InvalidMetadata(format!(
                "{label} digest must be lowercase SHA-256 hex"
            )));
        }
    }
    if metadata.tokenizer_sha256 != QWEN38_TOKENIZER_SHA256 {
        return Err(Qwen38MtpDraftVocabularyError::DigestMismatch {
            label: "tokenizer metadata",
            expected: QWEN38_TOKENIZER_SHA256.to_owned(),
            actual: metadata.tokenizer_sha256,
        });
    }
    let bytes = fs::read(&vocab_path).map_err(|error| io_error(&vocab_path, error))?;
    let actual_vocab_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_vocab_sha256 != metadata.vocab_sha256 {
        return Err(Qwen38MtpDraftVocabularyError::DigestMismatch {
            label: "payload",
            expected: metadata.vocab_sha256,
            actual: actual_vocab_sha256,
        });
    }
    let ids = decode_qwen38_mtp_draft_vocabulary_le(&bytes)
        .map_err(|error| Qwen38MtpDraftVocabularyError::InvalidVocabulary(error.to_string()))?;
    if ids.len() != metadata.n {
        return Err(Qwen38MtpDraftVocabularyError::InvalidVocabulary(format!(
            "payload contains {} IDs but metadata declares {}",
            ids.len(),
            metadata.n
        )));
    }
    if metadata
        .special_token_ids
        .windows(2)
        .any(|window| window[0] >= window[1])
        || metadata.special_token_ids.iter().any(|&id| {
            usize::try_from(id).map_or(true, |id| id >= QWEN35_VOCAB_SIZE)
                || ids.binary_search(&id).is_err()
        })
    {
        return Err(Qwen38MtpDraftVocabularyError::InvalidMetadata(
            "special_token_ids must be in the model vocabulary and selected payload".to_owned(),
        ));
    }
    Ok(Some(Qwen38MtpDraftVocabularyArtifact {
        ids,
        vocab_sha256: metadata.vocab_sha256,
        manifest_sha256: metadata.manifest_sha256,
        tokenizer_sha256: metadata.tokenizer_sha256,
        special_token_ids: metadata.special_token_ids,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("sllm-stage9-loader-{nonce}"))
    }

    #[test]
    fn both_artifacts_absent_preserve_full_vocabulary() {
        let root = temp_root();
        assert_eq!(load_qwen38_mtp_draft_vocabulary(&root).unwrap(), None);
    }

    #[test]
    fn partial_artifact_fails_closed() {
        let root = temp_root();
        fs::create_dir_all(root.join(".sllm")).expect("mkdir");
        fs::write(root.join(QWEN38_MTP_DRAFT_VOCAB_RELATIVE_PATH), []).expect("payload");
        let error = load_qwen38_mtp_draft_vocabulary(&root).unwrap_err();
        assert!(error.to_string().contains("present together"));
    }

    #[test]
    fn malformed_metadata_fails_closed_before_payload_use() {
        let root = temp_root();
        fs::create_dir_all(root.join(".sllm")).expect("mkdir");
        fs::write(root.join(QWEN38_MTP_DRAFT_VOCAB_RELATIVE_PATH), []).expect("payload");
        fs::write(
            root.join(QWEN38_MTP_DRAFT_VOCAB_METADATA_RELATIVE_PATH),
            b"{}",
        )
        .expect("metadata");
        let error = load_qwen38_mtp_draft_vocabulary(&root).unwrap_err();
        assert!(
            error.to_string().contains("tokenizer.json")
                || error.to_string().contains("missing field")
        );
    }
}
