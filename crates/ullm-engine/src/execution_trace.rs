// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Bounded production execution-trace artifact primitives.
//!
//! The worker wire protocol intentionally remains unchanged.  The production
//! trace is a local sidecar: this module owns the common engine-side limits,
//! privacy admission, canonical digesting, and immutable publication boundary;
//! the AQ4-specific producer and the independent validator live in `tools/`.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

/// Trace schema identifier accepted by the staged P1 producer.
pub const PRODUCTION_EXECUTION_TRACE_SCHEMA: &str = "ullm.production_execution_trace.v1";
/// Maximum trace artifact size from the evidence contract.
pub const MAX_PRODUCTION_EXECUTION_TRACE_BYTES: usize = 4 * 1024 * 1024;
/// Maximum nested JSON depth from the evidence contract.
pub const MAX_PRODUCTION_EXECUTION_TRACE_DEPTH: usize = 24;
/// Maximum JSON node count from the evidence contract.
pub const MAX_PRODUCTION_EXECUTION_TRACE_NODES: usize = 32_768;

/// Errors raised before a sidecar becomes visible to an evidence run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductionTraceError {
    /// The value contains prompt, token, request, or credential material.
    ForbiddenField(String),
    /// The value exceeds a bounded artifact rule.
    Bound(String),
    /// The value is not a supported JSON shape.
    Invalid(String),
    /// The destination already exists or cannot be published safely.
    Publish(String),
}

impl std::fmt::Display for ProductionTraceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForbiddenField(path) => write!(formatter, "forbidden trace field: {path}"),
            Self::Bound(message) | Self::Invalid(message) | Self::Publish(message) => {
                formatter.write_str(message)
            }
        }
    }
}

impl std::error::Error for ProductionTraceError {}

/// A validated JSON sidecar ready for immutable publication.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductionTraceArtifact {
    value: Value,
    bytes: Vec<u8>,
}

impl ProductionTraceArtifact {
    /// Builds a bounded artifact from a JSON value and rejects private content.
    pub fn new(value: Value) -> Result<Self, ProductionTraceError> {
        validate_tree(&value, "root", 0, &mut 0)?;
        reject_forbidden_fields(&value, "root")?;
        let bytes = serde_json::to_vec(&value)
            .map_err(|error| ProductionTraceError::Invalid(error.to_string()))?;
        if bytes.len() > MAX_PRODUCTION_EXECUTION_TRACE_BYTES {
            return Err(ProductionTraceError::Bound(format!(
                "trace exceeds {} bytes",
                MAX_PRODUCTION_EXECUTION_TRACE_BYTES
            )));
        }
        Ok(Self { value, bytes })
    }

    /// Returns the retained JSON value.
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Returns the exact bytes that are published and hashed.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the SHA-256 of the exact artifact bytes.
    pub fn sha256(&self) -> [u8; 32] {
        Sha256::digest(&self.bytes).into()
    }

    /// Publishes once through a sibling temporary file and refuses overwrite.
    pub fn publish_once(&self, destination: &Path) -> Result<(), ProductionTraceError> {
        if destination.exists() || destination.is_symlink() {
            return Err(ProductionTraceError::Publish(format!(
                "trace destination already exists: {}",
                destination.display()
            )));
        }
        let parent = destination.parent().ok_or_else(|| {
            ProductionTraceError::Publish("trace destination has no parent".into())
        })?;
        fs::create_dir_all(parent)
            .map_err(|error| ProductionTraceError::Publish(error.to_string()))?;
        let metadata = fs::metadata(parent)
            .map_err(|error| ProductionTraceError::Publish(error.to_string()))?;
        if metadata.permissions().mode() & 0o002 != 0 {
            return Err(ProductionTraceError::Publish(
                "trace destination directory is world-writable".into(),
            ));
        }
        let temporary = parent.join(format!(
            ".{}.incomplete",
            destination
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| ProductionTraceError::Publish(error.to_string()))?;
        file.write_all(&self.bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| ProductionTraceError::Publish(error.to_string()))?;
        fs::rename(&temporary, destination)
            .map_err(|error| ProductionTraceError::Publish(error.to_string()))
    }
}

/// Computes the digest of a canonical JSON representation with recursively sorted objects.
pub fn canonical_json_sha256(value: &Value) -> Result<[u8; 32], ProductionTraceError> {
    let canonical = canonicalize(value)?;
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| ProductionTraceError::Invalid(error.to_string()))?;
    Ok(Sha256::digest(bytes).into())
}

fn canonicalize(value: &Value) -> Result<Value, ProductionTraceError> {
    match value {
        Value::Object(object) => {
            let mut sorted = Map::new();
            let mut keys = object.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for key in keys {
                sorted.insert(key.clone(), canonicalize(&object[key])?);
            }
            Ok(Value::Object(sorted))
        }
        Value::Array(values) => values
            .iter()
            .map(canonicalize)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        other => Ok(other.clone()),
    }
}

fn validate_tree(
    value: &Value,
    path: &str,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), ProductionTraceError> {
    if depth > MAX_PRODUCTION_EXECUTION_TRACE_DEPTH {
        return Err(ProductionTraceError::Bound(format!(
            "JSON depth exceeds {MAX_PRODUCTION_EXECUTION_TRACE_DEPTH} at {path}"
        )));
    }
    *nodes = nodes
        .checked_add(1)
        .ok_or_else(|| ProductionTraceError::Bound("JSON node count overflows".into()))?;
    if *nodes > MAX_PRODUCTION_EXECUTION_TRACE_NODES {
        return Err(ProductionTraceError::Bound(
            "JSON node count exceeds trace bound".into(),
        ));
    }
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if key.len() > 16_384 || key.bytes().any(|byte| byte < 0x20) {
                    return Err(ProductionTraceError::Invalid(format!(
                        "invalid object key at {path}"
                    )));
                }
                validate_tree(child, &format!("{path}.{key}"), depth + 1, nodes)?;
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                validate_tree(child, &format!("{path}[{index}]"), depth + 1, nodes)?;
            }
        }
        Value::String(text) => {
            if text.len() > 16_384 || text.chars().any(|character| character.is_control()) {
                return Err(ProductionTraceError::Invalid(format!(
                    "invalid string at {path}"
                )));
            }
        }
        Value::Number(number) if number.as_f64().is_none() => {
            return Err(ProductionTraceError::Invalid(format!(
                "invalid number at {path}"
            )));
        }
        _ => {}
    }
    Ok(())
}

fn reject_forbidden_fields(value: &Value, path: &str) -> Result<(), ProductionTraceError> {
    const FORBIDDEN: &[&str] = &[
        "prompt_text",
        "prompt_or_token_content",
        "response_text",
        "response_body",
        "token_ids",
        "generated_text",
        "request_id",
        "api_key",
        "authorization",
        "account_id",
    ];
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                if FORBIDDEN
                    .iter()
                    .any(|forbidden| key.eq_ignore_ascii_case(forbidden))
                {
                    return Err(ProductionTraceError::ForbiddenField(format!(
                        "{path}.{key}"
                    )));
                }
                reject_forbidden_fields(child, &format!("{path}.{key}"))?;
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                reject_forbidden_fields(child, &format!("{path}[{index}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_private_trace_fields_but_allows_aggregate_counts() {
        let value = serde_json::json!({"request_summary": {"prompt_token_count": 128}, "request_id": "private"});
        assert!(matches!(
            ProductionTraceArtifact::new(value),
            Err(ProductionTraceError::ForbiddenField(_))
        ));
    }

    #[test]
    fn canonical_digest_is_independent_of_object_insertion_order() {
        let first = serde_json::json!({"b": 2, "a": 1});
        let second = serde_json::json!({"a": 1, "b": 2});
        assert_eq!(
            canonical_json_sha256(&first).unwrap(),
            canonical_json_sha256(&second).unwrap()
        );
    }

    #[test]
    fn publish_refuses_overwrite() {
        let root = std::env::temp_dir().join(format!("ullm-trace-test-{}", std::process::id()));
        let path = root.join("trace.json");
        let _ = fs::remove_dir_all(&root);
        let artifact = ProductionTraceArtifact::new(serde_json::json!({"ok": true})).unwrap();
        artifact.publish_once(&path).unwrap();
        assert!(artifact.publish_once(&path).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
