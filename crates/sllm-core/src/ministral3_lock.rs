//! Immutable model-lock binding for the reviewed official Ministral 3 GGUF.
//!
//! The detailed config, safetensors headers, and GGUF catalog have dedicated
//! typed validators. This lock binds those independently validated contracts
//! to one production artifact and frontend identity without treating the
//! official Mistral file as an sLLM-derived conversion.

use crate::{
    MINISTRAL3_OFFICIAL_GGUF_FILE_BYTES, MINISTRAL3_OFFICIAL_GGUF_FILE_NAME,
    MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256, MINISTRAL3_OFFICIAL_GGUF_REPOSITORY,
    MINISTRAL3_OFFICIAL_GGUF_REVISION, ModelError, fingerprint_for_json,
};
use serde::Deserialize;
use serde_json::Value;

pub const MINISTRAL3_MODEL_LOCK_SCHEMA: &str = "ministral3-official-gguf-model-lock-v1";
pub const MINISTRAL3_MODEL_LOCK_FINGERPRINT: &str =
    "sha256:8a8701bb8e7838bbc87575bea3339a1884d83a0bcd4cc226f6c83e4c3f70759a";
pub const MINISTRAL3_MODEL_ALIAS: &str = "ministral3-3b-instruct-2512-bf16";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Ministral3ModelLock {
    pub schema_version: String,
    /// The exact model body is authenticated by the restricted-JCS
    /// fingerprint. Detailed consumers must still use the dedicated typed
    /// config/header/GGUF validators instead of interpreting arbitrary keys.
    model: Value,
    fingerprint: String,
    aliases: Vec<String>,
    generated_at: String,
}

impl Ministral3ModelLock {
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn aliases(&self) -> &[String] {
        &self.aliases
    }

    pub const fn repository(&self) -> &'static str {
        MINISTRAL3_OFFICIAL_GGUF_REPOSITORY
    }

    pub const fn revision(&self) -> &'static str {
        MINISTRAL3_OFFICIAL_GGUF_REVISION
    }

    pub const fn file_name(&self) -> &'static str {
        MINISTRAL3_OFFICIAL_GGUF_FILE_NAME
    }

    pub const fn file_size(&self) -> u64 {
        MINISTRAL3_OFFICIAL_GGUF_FILE_BYTES
    }

    pub const fn file_sha256(&self) -> &'static str {
        MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256
    }

    pub const fn supports_chat_messages(&self) -> bool {
        true
    }

    pub const fn supports_vision(&self) -> bool {
        false
    }
}

pub fn parse_ministral3_model_lock(bytes: &[u8]) -> Result<Ministral3ModelLock, ModelError> {
    // This performs the bounded duplicate-key/control-character/integer checks
    // before serde sees the document.
    let computed = fingerprint_for_json(bytes)?;
    let lock: Ministral3ModelLock =
        serde_json::from_slice(bytes).map_err(|error| ModelError::Schema(error.to_string()))?;
    if lock.schema_version != MINISTRAL3_MODEL_LOCK_SCHEMA {
        return Err(ModelError::Invalid(
            "unsupported Ministral 3 model-lock schema".to_owned(),
        ));
    }
    if lock.fingerprint != computed {
        return Err(ModelError::FingerprintMismatch {
            expected: lock.fingerprint,
            actual: computed,
        });
    }
    if lock.fingerprint != MINISTRAL3_MODEL_LOCK_FINGERPRINT
        || lock.aliases != [MINISTRAL3_MODEL_ALIAS.to_owned()]
        || lock.generated_at != "2026-08-31T00:00:00Z"
        || !lock.model.is_object()
    {
        return Err(ModelError::Invalid(
            "Ministral 3 lock is not the reviewed immutable identity".to_owned(),
        ));
    }
    Ok(lock)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK: &[u8] = include_bytes!(
        "../../../docs/models/locks/ministral3-3b-instruct-2512-official-bf16-gguf.json"
    );

    #[test]
    fn tracked_lock_is_the_reviewed_production_identity() {
        let lock = parse_ministral3_model_lock(LOCK).expect("tracked lock");
        assert_eq!(lock.fingerprint(), MINISTRAL3_MODEL_LOCK_FINGERPRINT);
        assert_eq!(lock.aliases(), [MINISTRAL3_MODEL_ALIAS]);
        assert_eq!(lock.repository(), MINISTRAL3_OFFICIAL_GGUF_REPOSITORY);
        assert_eq!(lock.revision(), MINISTRAL3_OFFICIAL_GGUF_REVISION);
        assert_eq!(lock.file_size(), 6_866_745_504);
        assert!(lock.supports_chat_messages());
        assert!(!lock.supports_vision());
    }

    #[test]
    fn fingerprint_schema_alias_and_duplicate_drift_fail_closed() {
        let text = std::str::from_utf8(LOCK).expect("utf8 lock");
        let fingerprint = text.replacen(
            MINISTRAL3_MODEL_LOCK_FINGERPRINT,
            &format!("sha256:{}", "0".repeat(64)),
            1,
        );
        assert!(parse_ministral3_model_lock(fingerprint.as_bytes()).is_err());
        let schema = text.replacen(MINISTRAL3_MODEL_LOCK_SCHEMA, "ministral3-lock-v2", 1);
        assert!(parse_ministral3_model_lock(schema.as_bytes()).is_err());
        let alias = text.replacen(MINISTRAL3_MODEL_ALIAS, "ministral3-drift", 1);
        assert!(parse_ministral3_model_lock(alias.as_bytes()).is_err());
        let duplicate = text.replacen(
            "\"schema_version\":",
            "\"schema_version\":\"duplicate\",\"schema_version\":",
            1,
        );
        assert!(parse_ministral3_model_lock(duplicate.as_bytes()).is_err());
    }
}
