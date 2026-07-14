// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Versioned, AQ4-only worker wire for production-server prefill evidence.
//!
//! This protocol is intentionally separate from `ullm.worker.v1` and `.v2`.
//! Opting into it changes the complete stdin/stdout contract of the AQ4 worker;
//! ordinary generate commands and events keep their byte-for-byte schemas.

use crate::inference_api::SamplingParams;
use crate::sq8_worker_protocol::{Sq8WorkerProfile, validate_worker_request_id};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const AQ4_BENCHMARK_WORKER_SCHEMA_VERSION: &str = "ullm.aq4_p2.worker_benchmark.v1";
pub const AQ4_BENCHMARK_TERMINAL_EVIDENCE_SCHEMA_VERSION: &str =
    "ullm.aq4_p2.worker_terminal_evidence.v1";
pub const AQ4_BENCHMARK_INPUT_HASH_ALGORITHM: &str = "sha256-u64le-token-ids-v1";
pub const AQ4_BENCHMARK_PREFILL_M_GRID: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];
const MAX_CASE_ID_BYTES: usize = 256;
const MAX_RUN_INDEX: u32 = 4095;
const CASE_BINDING_KEYS: &[&str] = &[
    "baseline_mode",
    "cached_prefix_tokens",
    "case_id",
    "case_sha256",
    "context_tokens",
    "control",
    "control_id",
    "decode_request_count",
    "decode_start_tokens",
    "device",
    "fixture_id",
    "format_id",
    "generated_tokens",
    "implementation_id",
    "mode",
    "path_oracle_case_id",
    "path_oracle_result_sha256",
    "phase",
    "prefill_requested_m",
    "prompt_tokens",
    "request_count",
    "resolved_m",
    "sampling",
    "scope",
    "stage_id",
    "stage_order",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aq4BenchmarkRunKind {
    Warmup,
    Measured,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aq4BenchmarkPrefillCommand {
    pub request_id: String,
    pub case_id: String,
    pub case_sha256: String,
    pub case_binding: serde_json::Value,
    pub run_kind: Aq4BenchmarkRunKind,
    pub run_index: u32,
    pub requested_m: usize,
    pub resolved_m: usize,
    pub generated_tokens: usize,
    pub fixture_sha256: String,
    pub input_sha256: String,
    pub prompt_token_ids: Vec<usize>,
}

impl Aq4BenchmarkPrefillCommand {
    pub fn into_inference_request(
        &self,
        profile: &Sq8WorkerProfile,
        seed: i64,
    ) -> crate::inference_api::InferenceRequest {
        crate::inference_api::InferenceRequest::new_with_eos(
            self.request_id.clone(),
            self.prompt_token_ids.clone(),
            0,
            profile.eos_token_ids.clone(),
            SamplingParams::greedy_with_top_k(seed, profile.top_k),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Aq4BenchmarkWorkerCommand {
    Prefill(Aq4BenchmarkPrefillCommand),
    Cancel { request_id: String },
    Shutdown,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, tag = "type")]
enum RawCommand {
    #[serde(rename = "benchmark_prefill")]
    Prefill {
        schema_version: String,
        request_id: String,
        case_id: String,
        case_sha256: String,
        case_binding: serde_json::Value,
        run_kind: Aq4BenchmarkRunKind,
        run_index: u32,
        requested_m: u64,
        resolved_m: u64,
        generated_tokens: u64,
        fixture_sha256: String,
        input_sha256: String,
        prompt_token_ids: Vec<u64>,
    },
    #[serde(rename = "cancel")]
    Cancel {
        schema_version: String,
        request_id: String,
    },
    #[serde(rename = "shutdown")]
    Shutdown { schema_version: String },
}

pub fn decode_aq4_benchmark_worker_command(
    payload: &[u8],
    profile: &Sq8WorkerProfile,
) -> Result<Aq4BenchmarkWorkerCommand, String> {
    if payload.len() > crate::worker_protocol::WORKER_MAX_RECORD_BYTES {
        return Err("AQ4 benchmark command exceeds the worker record bound".into());
    }
    let mut deserializer = serde_json::Deserializer::from_slice(payload);
    let raw = RawCommand::deserialize(&mut deserializer)
        .and_then(|value| deserializer.end().map(|()| value))
        .map_err(|_| "AQ4 benchmark command does not match the exact schema".to_string())?;
    match raw {
        RawCommand::Prefill {
            schema_version,
            request_id,
            case_id,
            case_sha256,
            case_binding,
            run_kind,
            run_index,
            requested_m,
            resolved_m,
            generated_tokens,
            fixture_sha256,
            input_sha256,
            prompt_token_ids,
        } => {
            require_schema(&schema_version)?;
            validate_worker_request_id(&request_id).map_err(|_| {
                "AQ4 benchmark request_id violates the worker identifier syntax".to_string()
            })?;
            validate_case_id(&case_id)?;
            validate_sha256(&case_sha256, "case_sha256")?;
            validate_sha256(&fixture_sha256, "fixture_sha256")?;
            validate_sha256(&input_sha256, "input_sha256")?;
            if run_index > MAX_RUN_INDEX {
                return Err("AQ4 benchmark run_index exceeds its bound".into());
            }
            let requested_m = usize::try_from(requested_m)
                .map_err(|_| "AQ4 benchmark requested_m does not fit usize".to_string())?;
            let resolved_m = usize::try_from(resolved_m)
                .map_err(|_| "AQ4 benchmark resolved_m does not fit usize".to_string())?;
            if !AQ4_BENCHMARK_PREFILL_M_GRID.contains(&requested_m)
                || !AQ4_BENCHMARK_PREFILL_M_GRID.contains(&resolved_m)
                || (resolved_m != requested_m && resolved_m != 1)
            {
                return Err("AQ4 benchmark requested/resolved M contract differs".into());
            }
            let generated_tokens = usize::try_from(generated_tokens)
                .map_err(|_| "AQ4 benchmark generated_tokens does not fit usize".to_string())?;
            if generated_tokens != 0 {
                return Err("AQ4 benchmark command must be prefill-only".into());
            }
            if prompt_token_ids.is_empty() || prompt_token_ids.len() > profile.context_length {
                return Err("AQ4 benchmark prompt token count is out of range".into());
            }
            let prompt_token_ids = prompt_token_ids
                .into_iter()
                .enumerate()
                .map(|(index, token_id)| {
                    let token_id = usize::try_from(token_id).map_err(|_| {
                        format!("AQ4 benchmark prompt_token_ids[{index}] does not fit usize")
                    })?;
                    if token_id >= profile.vocab_size {
                        return Err(format!(
                            "AQ4 benchmark prompt_token_ids[{index}] exceeds the vocabulary"
                        ));
                    }
                    Ok(token_id)
                })
                .collect::<Result<Vec<_>, String>>()?;
            if aq4_benchmark_input_sha256(&prompt_token_ids) != input_sha256 {
                return Err("AQ4 benchmark input_sha256 does not bind prompt_token_ids".into());
            }
            validate_case_binding(
                &case_binding,
                &case_id,
                &case_sha256,
                requested_m,
                resolved_m,
                generated_tokens,
                prompt_token_ids.len(),
            )?;
            Ok(Aq4BenchmarkWorkerCommand::Prefill(
                Aq4BenchmarkPrefillCommand {
                    request_id,
                    case_id,
                    case_sha256,
                    case_binding,
                    run_kind,
                    run_index,
                    requested_m,
                    resolved_m,
                    generated_tokens,
                    fixture_sha256,
                    input_sha256,
                    prompt_token_ids,
                },
            ))
        }
        RawCommand::Cancel {
            schema_version,
            request_id,
        } => {
            require_schema(&schema_version)?;
            validate_worker_request_id(&request_id).map_err(|_| {
                "AQ4 benchmark cancel request_id violates the worker identifier syntax".to_string()
            })?;
            Ok(Aq4BenchmarkWorkerCommand::Cancel { request_id })
        }
        RawCommand::Shutdown { schema_version } => {
            require_schema(&schema_version)?;
            Ok(Aq4BenchmarkWorkerCommand::Shutdown)
        }
    }
}

pub fn aq4_benchmark_input_sha256(prompt_token_ids: &[usize]) -> String {
    let mut digest = Sha256::new();
    for &token_id in prompt_token_ids {
        digest.update(u64::try_from(token_id).unwrap_or(u64::MAX).to_le_bytes());
    }
    encode_sha256(digest.finalize().as_slice())
}

pub fn aq4_benchmark_case_sha256(case_binding: &serde_json::Value) -> Result<String, String> {
    let mut value = case_binding.clone();
    let object = value
        .as_object_mut()
        .ok_or_else(|| "AQ4 benchmark case_binding must be an object".to_string())?;
    if !object.contains_key("case_sha256") {
        return Err("AQ4 benchmark case_binding lacks case_sha256".into());
    }
    object.insert("case_sha256".into(), serde_json::Value::Null);
    sha256_json(&value)
}

fn validate_case_binding(
    value: &serde_json::Value,
    case_id: &str,
    case_sha256: &str,
    requested_m: usize,
    resolved_m: usize,
    generated_tokens: usize,
    prompt_tokens: usize,
) -> Result<(), String> {
    let case = value
        .as_object()
        .ok_or_else(|| "AQ4 benchmark case_binding must be an object".to_string())?;
    if case.len() != CASE_BINDING_KEYS.len()
        || CASE_BINDING_KEYS.iter().any(|key| !case.contains_key(*key))
    {
        return Err("AQ4 benchmark case_binding does not contain the canonical full case".into());
    }
    if case.get("case_id").and_then(serde_json::Value::as_str) != Some(case_id)
        || case.get("fixture_id").and_then(serde_json::Value::as_str) != Some(case_id)
        || case.get("case_sha256").and_then(serde_json::Value::as_str) != Some(case_sha256)
        || aq4_benchmark_case_sha256(value)? != case_sha256
    {
        return Err("AQ4 benchmark case ID/SHA binding differs".into());
    }
    let expected_numbers = [
        ("prefill_requested_m", requested_m),
        ("resolved_m", resolved_m),
        ("generated_tokens", generated_tokens),
        ("prompt_tokens", prompt_tokens),
        ("context_tokens", prompt_tokens),
        ("request_count", 1),
        ("decode_request_count", 0),
        ("decode_start_tokens", 0),
        ("cached_prefix_tokens", 0),
    ];
    for (key, expected) in expected_numbers {
        if case.get(key).and_then(serde_json::Value::as_u64) != u64::try_from(expected).ok() {
            return Err(format!("AQ4 benchmark case_binding.{key} differs"));
        }
    }
    if case.get("scope").and_then(serde_json::Value::as_str) != Some("full_model")
        || case.get("phase").and_then(serde_json::Value::as_str) != Some("cold_prefill")
        || case.get("stage_id").and_then(serde_json::Value::as_str) != Some("representative")
        || case.get("control_id").and_then(serde_json::Value::as_str) != Some("aq4_0_target")
        || case.get("format_id").and_then(serde_json::Value::as_str) != Some("AQ4_0")
        || case
            .get("implementation_id")
            .and_then(serde_json::Value::as_str)
            != Some("qwen35_aq4_rdna4_v1")
    {
        return Err("AQ4 benchmark case_binding production identity differs".into());
    }
    Ok(())
}

pub fn sha256_json(value: &serde_json::Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("failed to encode AQ4 benchmark audit: {error}"))?;
    Ok(encode_sha256(Sha256::digest(bytes).as_slice()))
}

fn encode_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn require_schema(value: &str) -> Result<(), String> {
    if value == AQ4_BENCHMARK_WORKER_SCHEMA_VERSION {
        Ok(())
    } else {
        Err("AQ4 benchmark schema_version differs".into())
    }
}

fn validate_case_id(value: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    if bytes.is_empty()
        || bytes.len() > MAX_CASE_ID_BYTES
        || !bytes[0].is_ascii_alphanumeric()
        || bytes[1..].iter().any(|byte| {
            !byte.is_ascii_alphanumeric() && !matches!(*byte, b'.' | b'_' | b':' | b'-')
        })
    {
        return Err("AQ4 benchmark case_id violates the bounded identifier syntax".into());
    }
    Ok(())
}

pub(crate) fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || value
            .as_bytes()
            .iter()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(byte))
    {
        return Err(format!("AQ4 benchmark {label} is not lowercase SHA-256"));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Aq4BenchmarkCapability {
    pub command_schema: &'static str,
    pub terminal_evidence_schema: &'static str,
    pub input_hash_algorithm: &'static str,
    pub configurable_m: bool,
    pub prefill_only: bool,
    pub m_grid: &'static [usize],
}

impl Default for Aq4BenchmarkCapability {
    fn default() -> Self {
        Self {
            command_schema: AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
            terminal_evidence_schema: AQ4_BENCHMARK_TERMINAL_EVIDENCE_SCHEMA_VERSION,
            input_hash_algorithm: AQ4_BENCHMARK_INPUT_HASH_ALGORITHM,
            configurable_m: true,
            prefill_only: true,
            m_grid: AQ4_BENCHMARK_PREFILL_M_GRID,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Aq4BenchmarkTerminalStatus {
    Ok,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Aq4BenchmarkReuse {
    Allowed,
    Forbidden,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Aq4BenchmarkResetEvidence {
    pub attempted: u64,
    pub complete: u64,
    pub failed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Aq4BenchmarkFallbackEvidence {
    pub used: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Aq4BenchmarkEvidenceLinks {
    pub fixture_sha256: String,
    pub input_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_audit_sha256: Option<String>,
    pub resource_observation_key: String,
    pub resource_samples_embedded: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Aq4BenchmarkExecutionEvidence {
    pub status: Aq4BenchmarkTerminalStatus,
    pub reuse: Aq4BenchmarkReuse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_code: Option<String>,
    pub requested_m: usize,
    pub resolved_m: usize,
    pub actual_m: Option<usize>,
    pub actual_token_batch_width: Option<usize>,
    pub actual_request_batch_width: Option<usize>,
    pub fallback: Aq4BenchmarkFallbackEvidence,
    pub lifecycle: serde_json::Value,
    pub reset: Aq4BenchmarkResetEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sanitized_audit: Option<serde_json::Value>,
    pub links: Aq4BenchmarkEvidenceLinks,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields, tag = "type")]
pub enum Aq4BenchmarkWorkerEvent<'a> {
    #[serde(rename = "ready")]
    Ready {
        schema_version: &'static str,
        model: &'a str,
        model_revision: &'a str,
        artifact_content_sha256: &'a str,
        package_manifest_sha256: &'a str,
        device: &'a str,
        execution_profile: &'a str,
        context_length: usize,
        capability: Aq4BenchmarkCapability,
    },
    #[serde(rename = "started")]
    Started {
        schema_version: &'static str,
        request_id: &'a str,
        case_id: &'a str,
        run_kind: Aq4BenchmarkRunKind,
        run_index: u32,
        prompt_tokens: usize,
    },
    #[serde(rename = "progress")]
    Progress {
        schema_version: &'static str,
        request_id: &'a str,
        phase: &'static str,
        processed_prompt_tokens: usize,
    },
    #[serde(rename = "terminal_evidence")]
    TerminalEvidence {
        schema_version: &'static str,
        evidence_schema_version: &'static str,
        request_id: &'a str,
        case_id: &'a str,
        case_sha256: &'a str,
        run_kind: Aq4BenchmarkRunKind,
        run_index: u32,
        generated_tokens: usize,
        evidence: &'a Aq4BenchmarkExecutionEvidence,
    },
    #[serde(rename = "released")]
    Released {
        schema_version: &'static str,
        request_id: &'a str,
        status: Aq4BenchmarkTerminalStatus,
        reuse: Aq4BenchmarkReuse,
        reset_complete: bool,
    },
    #[serde(rename = "error")]
    Error {
        schema_version: &'static str,
        request_id: Option<&'a str>,
        code: &'static str,
        recoverable: bool,
        message: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> Sq8WorkerProfile {
        let mut value = Sq8WorkerProfile::sq8_defaults();
        value.context_length = 16;
        value.vocab_size = 32;
        value
    }

    fn case_binding() -> serde_json::Value {
        let mut case = serde_json::json!({
            "baseline_mode": "all_m1",
            "cached_prefix_tokens": 0,
            "case_id": "case-1",
            "case_sha256": null,
            "context_tokens": 3,
            "control": {"control_id": "aq4_0_target", "role": "target", "format_id": "AQ4_0", "implementation_id": "qwen35_aq4_rdna4_v1", "promotion_eligible": true},
            "control_id": "aq4_0_target",
            "decode_request_count": 0,
            "decode_start_tokens": 0,
            "device": {"device_id": "r9700-rdna4", "runtime_device_index": 1, "backend": "hip", "name": "AMD Radeon Graphics", "architecture": "gfx1201"},
            "fixture_id": "case-1",
            "format_id": "AQ4_0",
            "generated_tokens": 0,
            "implementation_id": "qwen35_aq4_rdna4_v1",
            "mode": "all_m1",
            "path_oracle_case_id": null,
            "path_oracle_result_sha256": null,
            "phase": "cold_prefill",
            "prefill_requested_m": 64,
            "prompt_tokens": 3,
            "request_count": 1,
            "resolved_m": 1,
            "sampling": {"mode": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
            "scope": "full_model",
            "stage_id": "representative",
            "stage_order": 1,
        });
        let digest = aq4_benchmark_case_sha256(&case).unwrap();
        case["case_sha256"] = digest.into();
        case
    }

    fn command(extra: &str) -> Vec<u8> {
        let input = aq4_benchmark_input_sha256(&[1, 2, 3]);
        let binding = case_binding();
        let case_sha256 = binding["case_sha256"].as_str().unwrap();
        let mut encoded = serde_json::to_string(&serde_json::json!({
            "schema_version": AQ4_BENCHMARK_WORKER_SCHEMA_VERSION,
            "type": "benchmark_prefill",
            "request_id": "req-1",
            "case_id": "case-1",
            "case_sha256": case_sha256,
            "case_binding": binding,
            "run_kind": "measured",
            "run_index": 2,
            "requested_m": 64,
            "resolved_m": 1,
            "generated_tokens": 0,
            "fixture_sha256": "b".repeat(64),
            "input_sha256": input,
            "prompt_token_ids": [1, 2, 3],
        }))
        .unwrap();
        encoded.pop();
        encoded.push_str(extra);
        encoded.push('}');
        encoded.into_bytes()
    }

    #[test]
    fn exact_prefill_command_binds_input_and_all_m1_resolution() {
        let decoded = decode_aq4_benchmark_worker_command(&command(""), &profile()).unwrap();
        let Aq4BenchmarkWorkerCommand::Prefill(decoded) = decoded else {
            panic!("expected prefill command");
        };
        assert_eq!(decoded.requested_m, 64);
        assert_eq!(decoded.resolved_m, 1);
        assert_eq!(decoded.generated_tokens, 0);
        assert_eq!(decoded.prompt_token_ids, [1, 2, 3]);
    }

    #[test]
    fn exact_prefill_command_rejects_unknown_duplicate_and_hash_drift() {
        assert!(
            decode_aq4_benchmark_worker_command(&command(",\"unknown\":1"), &profile()).is_err()
        );
        assert!(
            decode_aq4_benchmark_worker_command(&command(",\"case_id\":\"swap\""), &profile())
                .is_err()
        );
        let mut value: serde_json::Value = serde_json::from_slice(&command("")).unwrap();
        value["prompt_token_ids"] = serde_json::json!([1, 2, 4]);
        assert!(
            decode_aq4_benchmark_worker_command(&serde_json::to_vec(&value).unwrap(), &profile())
                .is_err()
        );
        let mut swapped: serde_json::Value = serde_json::from_slice(&command("")).unwrap();
        swapped["case_id"] = "case-swapped".into();
        assert!(
            decode_aq4_benchmark_worker_command(&serde_json::to_vec(&swapped).unwrap(), &profile())
                .is_err()
        );
    }

    #[test]
    fn ordinary_worker_generate_is_not_a_benchmark_command() {
        let payload = br#"{"schema_version":"ullm.worker.v1","type":"generate","request_id":"req-1","prompt_token_ids":[1],"max_new_tokens":1,"sampling":{"temperature":0.0,"top_p":1.0,"top_k":1,"seed":0},"eos_token_ids":[2]}"#;
        assert!(decode_aq4_benchmark_worker_command(payload, &profile()).is_err());
    }
}
