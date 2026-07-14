// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Offline Qwen3.5 AQ4 P2 full-vector calibration capture.
//!
//! One clean process captures one strictly bound case. Calibration device-to-host transfers are
//! diagnostic only and are never performance-timing evidence.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{CString, OsString};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

use ullm_engine::aq4_worker_backend::QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::inference_api::{CancellationToken, InferenceRequest, SamplingParams};
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4CalibrationObserver, Qwen35Aq4ModelLoadConfig,
};
use ullm_engine::qwen35_aq4_session::{
    Qwen35Aq4CalibrationReplay, Qwen35Aq4InferenceSession, Qwen35Aq4SessionConfig,
    Qwen35Aq4SessionStatus,
};
use ullm_engine::served_model::{ServedModel, load_served_model};
use ullm_engine::worker_driver::{InferenceSession, SessionAdvance};

const TARGET_SCHEMA: &str = "ullm.qwen35_aq4_target_calibration.v1";
const SOURCE_SCHEMA: &str = "ullm.qwen35_aq4_source_calibration.v1";
const SOURCE_CASES_SCHEMA: &str = "ullm.qwen35_aq4_source_calibration_cases.v1";
const HIDDEN_SIZE: usize = 4096;
const VOCAB_SIZE: usize = 248_320;
const TOP_K: usize = 10;
const F32_BYTES: usize = 4;
const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;
const MAX_ROW_LINE_BYTES: usize = 64 * 1024;
const MAX_ROWS: usize = 16_384;
const MAX_CASES: usize = 8192;
const MAX_STEPS: usize = 128;
const MAX_PACKAGE_FILES: usize = 65_536;
const MAX_PACKAGE_DEPTH: usize = 32;
const HASH_CHUNK_BYTES: usize = 1024 * 1024;
const DEFAULT_CHUNK_ELEMENTS: usize = 65_536;
const DEFAULT_LOAD_CHUNK_BYTES: usize = 1024 * 1024;
const DEFAULT_LM_HEAD_CHUNK_ROWS: usize = 8192;
const DIRECT_TOP1_ENV: &str = "ULLM_ENABLE_AQ4_LM_HEAD_DIRECT_TOP1";
const PREFLIGHT_FIELDS: &[&str] = &[
    "weights_bytes",
    "persistent_state_bytes",
    "kv_cache_bytes",
    "workspace_bytes",
    "temporary_bytes",
    "vram_headroom_bytes",
    "gpu_process_snapshot",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    served_model_manifest: PathBuf,
    fixture: PathBuf,
    case_binding: PathBuf,
    identity_binding: PathBuf,
    preflight: PathBuf,
    source_root: PathBuf,
    output: PathBuf,
    case_id: String,
    policy_id: String,
    oracle_kind: String,
    requested_m: usize,
    device_index: u32,
    chunk_elements: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureFile {
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureCase {
    case_id: String,
    prompt_token_ids: Vec<usize>,
    step_count: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2CaseBinding {
    case_id: String,
    fixture_id: String,
    case_sha256: String,
    stage_id: String,
    stage_order: u64,
    scope: String,
    phase: String,
    mode: String,
    baseline_mode: String,
    prompt_tokens: usize,
    cached_prefix_tokens: usize,
    context_tokens: usize,
    decode_start_tokens: usize,
    prefill_requested_m: usize,
    resolved_m: usize,
    request_count: usize,
    decode_request_count: usize,
    generated_tokens: usize,
    device: P2CaseDevice,
    control_id: String,
    control: P2CaseControl,
    sampling: P2CaseSampling,
    format_id: String,
    implementation_id: String,
    path_oracle_case_id: Option<String>,
    path_oracle_result_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2CaseDevice {
    device_id: String,
    backend: String,
    name: String,
    architecture: String,
    runtime_device_index: i32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2CaseControl {
    control_id: String,
    role: String,
    format_id: String,
    implementation_id: String,
    promotion_eligible: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct P2CaseSampling {
    mode: String,
    temperature: f64,
    top_p: f64,
    top_k: usize,
    seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCasesFile {
    schema_version: String,
    cases: Vec<SourceCase>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCase {
    case_id: String,
    prompt_token_ids: Vec<usize>,
    step_count: usize,
    #[serde(default)]
    semantic_input_id: Option<String>,
    #[serde(default)]
    observation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct VectorRef {
    offset_bytes: u64,
    bytes: u64,
    elements: usize,
    dtype: String,
    endianness: String,
    sha256: String,
    nonfinite_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TopEntry {
    token_id: usize,
    logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct VectorRow {
    case_id: String,
    step: usize,
    semantic_input_id: String,
    observation: String,
    input_token_ids_sha256: String,
    hidden: VectorRef,
    logits: VectorRef,
    greedy_token_id: usize,
    topk: Vec<TopEntry>,
    finite: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ExecutionRow {
    case_id: String,
    step: usize,
    source_sequence_sha256: String,
    source_row_sha256: String,
    predicted_token_id: usize,
    committed_token_id: usize,
    diverged: bool,
    generation_epoch: u64,
    observation_complete: bool,
    publication_committed: bool,
    lifecycle: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Link {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone)]
struct RuntimeDevice {
    requested_index: u32,
    device_id: i32,
    backend: String,
    name: String,
    architecture: String,
}

#[derive(Debug, Clone)]
struct PackageTree {
    sha256: String,
    file_count: usize,
    bytes: u64,
}

struct CommonBinding {
    model: ServedModel,
    fixture: FixtureCase,
    case: P2CaseBinding,
    case_link: Link,
    identity_link: Link,
    preflight_link: Link,
    preflight: Value,
    package_dir: PathBuf,
    package_tree: PackageTree,
    device: RuntimeDevice,
    capture_binary: Link,
}

struct SourceArtifact {
    manifest: Value,
    manifest_link: Link,
    tokenizer_sha256: String,
    rows: Vec<VectorRow>,
    row_sha256s: Vec<String>,
    replay_tokens: Vec<usize>,
    replay_sha256: String,
}

struct StrictJson(Value);

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("strict JSON without duplicate keys or non-finite numbers")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictJson(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictJson(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictJson(Value::Number(value.into())))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .map(StrictJson)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictJson(Value::String(value.to_string())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictJson(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJson(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJson(Value::Null))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(StrictJson(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(StrictJson(Value::Array(values)))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut values = serde_json::Map::new();
        while let Some((key, StrictJson(value))) = map.next_entry::<String, StrictJson>()? {
            if values.insert(key.clone(), value).is_some() {
                return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
            }
        }
        Ok(StrictJson(Value::Object(values)))
    }
}

fn parse_strict_json(bytes: &[u8], label: &str) -> Result<Value, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictJson::deserialize(&mut deserializer)
        .map_err(|error| format!("{label} JSON rejected: {error}"))?;
    deserializer
        .end()
        .map_err(|error| format!("{label} has trailing JSON: {error}"))?;
    Ok(value.0)
}

fn exact_fields(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!("{label} fields differ"));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Option<Args>, String> {
    let mut values = args.into_iter();
    let mut map = BTreeMap::<String, OsString>::new();
    while let Some(raw) = values.next() {
        let name = raw
            .to_str()
            .ok_or_else(|| "argument name is not UTF-8".to_string())?;
        if matches!(name, "-h" | "--help") {
            return Ok(None);
        }
        if !name.starts_with("--") {
            return Err(format!("unexpected positional argument {name:?}"));
        }
        let value = values
            .next()
            .ok_or_else(|| format!("{name} requires a value"))?;
        if map.insert(name.to_string(), value).is_some() {
            return Err(format!("duplicate argument {name}"));
        }
    }
    let mut take = |name: &str| {
        map.remove(name)
            .ok_or_else(|| format!("{name} is required"))
    };
    let path = |value: OsString| PathBuf::from(value);
    let text = |value: OsString, label: &str| {
        value
            .into_string()
            .map_err(|_| format!("{label} is not UTF-8"))
    };
    let served_model_manifest = path(take("--served-model-manifest")?);
    let fixture = path(take("--fixture")?);
    let case_binding = path(take("--case")?);
    let identity_binding = path(take("--identity")?);
    let preflight = path(take("--preflight")?);
    let source_root = path(take("--source")?);
    let output = path(take("--output")?);
    let case_id = text(take("--case-id")?, "case id")?;
    let policy_id = text(take("--policy-id")?, "policy id")?;
    let oracle_kind = text(take("--oracle-kind")?, "oracle kind")?;
    let requested_m = text(take("--m")?, "M")?
        .parse::<usize>()
        .map_err(|error| format!("invalid M: {error}"))?;
    let device_index = map
        .remove("--device-index")
        .map(|value| text(value, "device index"))
        .transpose()?
        .unwrap_or_else(|| "0".to_string())
        .parse::<u32>()
        .map_err(|error| format!("invalid device index: {error}"))?;
    let chunk_elements = map
        .remove("--chunk-elements")
        .map(|value| text(value, "chunk elements"))
        .transpose()?
        .unwrap_or_else(|| DEFAULT_CHUNK_ELEMENTS.to_string())
        .parse::<usize>()
        .map_err(|error| format!("invalid chunk elements: {error}"))?;
    if !map.is_empty() {
        return Err(format!(
            "unknown arguments: {}",
            map.keys().cloned().collect::<Vec<_>>().join(", ")
        ));
    }
    if case_id.is_empty()
        || case_id.len() > 128
        || policy_id.is_empty()
        || policy_id.len() > 128
        || !matches!(requested_m, 1 | 8 | 16 | 32 | 64 | 128)
        || chunk_elements == 0
        || chunk_elements > 1_048_576
    {
        return Err("case/policy/M/chunk arguments exceed the fixed contract".to_string());
    }
    let expected_kind = if requested_m == 1 {
        "aq4_target"
    } else {
        "aq4_optimized"
    };
    if oracle_kind != expected_kind {
        return Err(format!(
            "oracle kind {oracle_kind:?} differs from M={requested_m} root kind {expected_kind:?}"
        ));
    }
    Ok(Some(Args {
        served_model_manifest,
        fixture,
        case_binding,
        identity_binding,
        preflight,
        source_root,
        output,
        case_id,
        policy_id,
        oracle_kind,
        requested_m,
        device_index,
        chunk_elements,
    }))
}

fn print_help() {
    eprintln!(
        "Usage: ullm-aq4-p2-calibration --served-model-manifest PATH --fixture PATH --case PATH --identity PATH --preflight PATH --source DIR --output DIR --case-id ID --policy-id ID --oracle-kind aq4_target|aq4_optimized --m 1|8|16|32|64|128 [--device-index N] [--chunk-elements N]"
    );
}

fn reject_symlink_components(path: &Path) -> Result<(), String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|error| format!("current directory lookup failed: {error}"))?
            .join(path)
    };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        current.push(component.as_os_str());
        if let Ok(metadata) = fs::symlink_metadata(&current)
            && metadata.file_type().is_symlink()
        {
            return Err(format!(
                "path component is a symlink: {}",
                current.display()
            ));
        }
    }
    Ok(())
}

fn read_regular(path: &Path, label: &str, max_bytes: usize) -> Result<Vec<u8>, String> {
    reject_symlink_components(path)?;
    let before =
        fs::symlink_metadata(path).map_err(|error| format!("{label} metadata failed: {error}"))?;
    if !before.file_type().is_file() || before.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    if before.len() > max_bytes as u64 {
        return Err(format!("{label} exceeds {max_bytes} bytes"));
    }
    let mut file = File::open(path).map_err(|error| format!("{label} open failed: {error}"))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("{label} opened metadata failed: {error}"))?;
    if metadata_identity(&before) != metadata_identity(&opened) {
        return Err(format!("{label} identity changed while opening"));
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("{label} read failed: {error}"))?;
    let after = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} post-read metadata failed: {error}"))?;
    if metadata_identity(&before) != metadata_identity(&after) {
        return Err(format!("{label} changed while reading"));
    }
    Ok(bytes)
}

fn metadata_identity(metadata: &fs::Metadata) -> (u64, u64, u64, i64, i64, u32) {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.mode(),
    )
}

fn sha256_file(path: &Path, label: &str) -> Result<String, String> {
    reject_symlink_components(path)?;
    let before =
        fs::symlink_metadata(path).map_err(|error| format!("{label} metadata failed: {error}"))?;
    if !before.file_type().is_file() || before.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let mut file = File::open(path).map_err(|error| format!("{label} open failed: {error}"))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; HASH_CHUNK_BYTES];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("{label} read failed: {error}"))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    let after = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} post-hash metadata failed: {error}"))?;
    if metadata_identity(&before) != metadata_identity(&after) {
        return Err(format!("{label} changed while hashing"));
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn safe_relative(root: &Path, raw: &str, label: &str) -> Result<PathBuf, String> {
    let relative = Path::new(raw);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("{label} is not a safe relative path"));
    }
    let path = root.join(relative);
    reject_symlink_components(&path)?;
    Ok(path)
}

fn canonical_token_hash(tokens: &[usize]) -> Result<String, String> {
    let mut bytes = serde_json::to_vec(tokens)
        .map_err(|error| format!("token hash serialization failed: {error}"))?;
    bytes.push(b'\n');
    Ok(sha256_bytes(&bytes))
}

fn canonical_value_hash(value: &Value, self_field: Option<&str>) -> Result<String, String> {
    let mut value = value.clone();
    if let Some(field) = self_field {
        value
            .as_object_mut()
            .ok_or_else(|| "self-hashed JSON root must be an object".to_string())?
            .insert(field.to_string(), Value::Null);
    }
    let bytes = serde_json::to_vec(&value)
        .map_err(|error| format!("canonical JSON serialization failed: {error}"))?;
    Ok(sha256_bytes(&bytes))
}

fn flag_enabled(value: Option<OsString>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.to_string_lossy().trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn is_oom(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    error.contains("out of memory")
        || error.contains("memory allocation")
        || error.contains("hiperroroutofmemory")
        || error.contains("std::bad_alloc")
}

fn complete_capture_status(nonfinite_rows: usize) -> &'static str {
    if nonfinite_rows == 0 {
        "available"
    } else {
        "blocked"
    }
}

fn blocked_reason_code(error: &str) -> &'static str {
    if is_oom(error) {
        "runtime_out_of_memory"
    } else if error.contains("identity")
        || error.contains("binding")
        || error.contains("source")
        || error.contains("hash")
    {
        "identity_or_source_rejected"
    } else if error.contains("direct-top1") {
        "direct_top1_full_logits_unavailable"
    } else {
        "capture_failed_or_incomplete"
    }
}

fn load_json_link(path: &Path, label: &str) -> Result<(Value, Link), String> {
    let bytes = read_regular(path, label, MAX_JSON_BYTES)?;
    let value = parse_strict_json(&bytes, label)?;
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("{label} canonicalization failed: {error}"))?;
    Ok((
        value,
        Link {
            path: canonical.to_string_lossy().into_owned(),
            sha256: sha256_bytes(&bytes),
        },
    ))
}

fn validate_preflight(value: &Value) -> Result<(), String> {
    exact_fields(value, PREFLIGHT_FIELDS, "P2 preflight")?;
    let object = value.as_object().expect("exact_fields checked object");
    for field in PREFLIGHT_FIELDS
        .iter()
        .copied()
        .filter(|field| *field != "gpu_process_snapshot")
    {
        if object.get(field).and_then(Value::as_u64).is_none() {
            return Err(format!(
                "P2 preflight {field} must be a non-negative integer"
            ));
        }
    }
    for process in object["gpu_process_snapshot"]
        .as_array()
        .ok_or_else(|| "P2 preflight gpu_process_snapshot must be an array".to_string())?
    {
        exact_fields(
            process,
            &["pid", "process_name", "vram_bytes"],
            "P2 process snapshot",
        )?;
        if process.get("pid").and_then(Value::as_u64).is_none()
            || process
                .get("process_name")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            || process.get("vram_bytes").and_then(Value::as_u64).is_none()
        {
            return Err("P2 process snapshot values are invalid".to_string());
        }
    }
    Ok(())
}

fn load_fixture(path: &Path, case_id: &str) -> Result<FixtureCase, String> {
    let value = parse_strict_json(&read_regular(path, "fixture", 1024 * 1024)?, "fixture")?;
    let fixture: FixtureFile = serde_json::from_value(value)
        .map_err(|error| format!("fixture exact schema rejected: {error}"))?;
    if fixture.cases.is_empty() || fixture.cases.len() > 128 {
        return Err("fixture must contain 1..=128 cases".to_string());
    }
    let mut ids = BTreeSet::new();
    for case in &fixture.cases {
        if !ids.insert(case.case_id.as_str()) {
            return Err("fixture contains duplicate case_id".to_string());
        }
    }
    let case = fixture
        .cases
        .into_iter()
        .find(|case| case.case_id == case_id)
        .ok_or_else(|| "requested fixture case is absent".to_string())?;
    if case.case_id.is_empty()
        || case.case_id.len() > 128
        || !case
            .case_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        || case.prompt_token_ids.is_empty()
        || case.prompt_token_ids.len() > 4096
        || case.step_count == 0
        || case.step_count > MAX_STEPS
    {
        return Err("fixture case exceeds bounded calibration limits".to_string());
    }
    Ok(case)
}

fn validate_model(model: &ServedModel) -> Result<(), String> {
    if model.format.format_id != "AQ4_0"
        || model.worker.identity.device != "gfx1201"
        || model.worker.identity.execution_profile != "rdna4_aq4_resident"
        || model.format.implementation_id != "qwen35_aq4_rdna4_v1"
    {
        return Err("served model is not the AQ4 resident production identity".to_string());
    }
    if model.generation.sampling.temperature
        || model.generation.sampling.top_p
        || model.generation.sampling.top_k != 1
    {
        return Err("served model greedy sampling contract differs".to_string());
    }
    if model.product.artifact.is_some()
        || model.product.package.manifest_path.is_empty()
        || model.product.package.manifest_sha256.is_empty()
    {
        return Err("served model does not have a complete package-only product contract".into());
    }
    let mut actual_environment = model.worker.required_environment.iter().collect::<Vec<_>>();
    actual_environment.sort_unstable();
    let mut expected_environment = QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV.to_vec();
    expected_environment.sort_unstable();
    if actual_environment.len() != expected_environment.len()
        || actual_environment
            .iter()
            .zip(expected_environment)
            .any(|(actual, expected)| actual.as_str() != expected)
    {
        return Err("served model required-environment contract is incomplete".to_string());
    }
    Ok(())
}

fn validate_required_environment(model: &ServedModel) -> Result<(), String> {
    for name in &model.worker.required_environment {
        if env::var_os(name).as_deref() != Some(std::ffi::OsStr::new("1")) {
            return Err(format!("required environment {name} must equal 1"));
        }
    }
    Ok(())
}

fn observe_device(index: u32) -> Result<RuntimeDevice, String> {
    let device = ullm_runtime_sys::device_info(index)
        .map_err(|error| format!("runtime device query failed: {error}"))?;
    if device.device_id < 0
        || device.backend.is_empty()
        || device.name.is_empty()
        || device.gcn_arch_name.is_empty()
    {
        return Err("runtime device identity is incomplete".to_string());
    }
    Ok(RuntimeDevice {
        requested_index: index,
        device_id: device.device_id,
        backend: device.backend,
        name: device.name,
        architecture: device.gcn_arch_name,
    })
}

fn package_tree_identity(root: &Path) -> Result<PackageTree, String> {
    reject_symlink_components(root)?;
    let canonical = root
        .canonicalize()
        .map_err(|error| format!("package root canonicalization failed: {error}"))?;
    let metadata = fs::symlink_metadata(&canonical)
        .map_err(|error| format!("package root metadata failed: {error}"))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("package root must be a non-symlink directory".to_string());
    }
    let mut pending = vec![(canonical.clone(), 0usize)];
    let mut files = Vec::new();
    while let Some((directory, depth)) = pending.pop() {
        if depth > MAX_PACKAGE_DEPTH {
            return Err("package tree exceeds the depth limit".to_string());
        }
        for entry in fs::read_dir(&directory)
            .map_err(|error| format!("package directory read failed: {error}"))?
        {
            let entry = entry.map_err(|error| format!("package entry read failed: {error}"))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("package entry metadata failed: {error}"))?;
            if metadata.file_type().is_symlink() {
                return Err("package tree contains a symlink".to_string());
            }
            if metadata.is_dir() {
                pending.push((path, depth + 1));
            } else if metadata.is_file() {
                files.push(path);
                if files.len() > MAX_PACKAGE_FILES {
                    return Err("package tree exceeds the file-count limit".to_string());
                }
            } else {
                return Err("package tree contains a non-regular entry".to_string());
            }
        }
    }
    if files.is_empty() {
        return Err("package tree is empty".to_string());
    }
    files.sort_by(|left, right| {
        left.strip_prefix(&canonical)
            .expect("package path has prefix")
            .as_os_str()
            .as_bytes()
            .cmp(
                right
                    .strip_prefix(&canonical)
                    .expect("package path has prefix")
                    .as_os_str()
                    .as_bytes(),
            )
    });
    let mut aggregate = Sha256::new();
    let mut bytes = 0_u64;
    for file in &files {
        let relative = file
            .strip_prefix(&canonical)
            .expect("package path has prefix")
            .as_os_str()
            .as_bytes();
        let digest = sha256_file(file, "package file")?;
        let size = fs::metadata(file)
            .map_err(|error| format!("package file metadata failed: {error}"))?
            .len();
        bytes = bytes
            .checked_add(size)
            .ok_or_else(|| "package byte count overflows".to_string())?;
        aggregate.update((relative.len() as u64).to_le_bytes());
        aggregate.update(relative);
        aggregate.update(size.to_le_bytes());
        let digest_bytes = (0..32)
            .map(|index| u8::from_str_radix(&digest[index * 2..index * 2 + 2], 16).unwrap())
            .collect::<Vec<_>>();
        aggregate.update(digest_bytes);
    }
    Ok(PackageTree {
        sha256: format!("{:x}", aggregate.finalize()),
        file_count: files.len(),
        bytes,
    })
}

fn validate_case(
    args: &Args,
    case: &P2CaseBinding,
    fixture: &FixtureCase,
    model: &ServedModel,
) -> Result<(), String> {
    let expected_mode = if args.requested_m == 1 {
        "all_m1"
    } else {
        "cold_batched"
    };
    if case.case_id != args.case_id
        || case.fixture_id != args.case_id
        || case.case_sha256.is_empty()
        || case.stage_id.is_empty()
        || case.stage_order == 0
        || case.scope != "full_model"
        || case.phase != "cold_prefill"
        || case.mode != expected_mode
        || case.baseline_mode != expected_mode
        || case.prompt_tokens != fixture.prompt_token_ids.len()
        || case.cached_prefix_tokens != 0
        || case.context_tokens != fixture.prompt_token_ids.len()
        || case.decode_start_tokens != fixture.prompt_token_ids.len()
        || case.prefill_requested_m != args.requested_m
        || case.resolved_m != args.requested_m
        || case.request_count != 1
        || case.decode_request_count != 0
        || case.generated_tokens != fixture.step_count
        || case.format_id != "AQ4_0"
        || case.format_id != model.format.format_id
        || case.implementation_id != model.format.implementation_id
        || case.control_id != "aq4_0_target"
        || case.control.control_id != case.control_id
        || case.control.role != "target"
        || case.control.format_id != case.format_id
        || case.control.implementation_id != case.implementation_id
        || !case.control.promotion_eligible
        || case.sampling.mode != "greedy"
        || case.sampling.temperature != 0.0
        || case.sampling.top_p != 1.0
        || case.sampling.top_k != 1
        || case.sampling.seed != 0
        || (args.requested_m == 1
            && (case.path_oracle_case_id.is_some() || case.path_oracle_result_sha256.is_some()))
        || (args.requested_m != 1
            && (case
                .path_oracle_case_id
                .as_deref()
                .is_none_or(str::is_empty)
                || case.path_oracle_result_sha256.is_some()))
    {
        return Err("P2 case workload/control fields do not exactly match".to_string());
    }
    Ok(())
}

fn validate_device_binding(
    case: &P2CaseDevice,
    device: &RuntimeDevice,
    model: &ServedModel,
) -> Result<(), String> {
    let architecture = match device.architecture.as_str() {
        "gfx1201" => "RDNA4",
        "gfx1030" | "gfx1031" => "RDNA2",
        _ => return Err("runtime architecture is unsupported by the P2 binding".to_string()),
    };
    if case.runtime_device_index != device.device_id
        || case.backend != device.backend
        || case.name != device.name
        || case.architecture != architecture
        || case.device_id != "r9700-rdna4"
        || model.worker.identity.device != device.architecture
    {
        return Err("P2 device binding differs from runtime/served identity".to_string());
    }
    Ok(())
}

fn validate_identity_binding(
    value: &Value,
    case_link: &Link,
    model: &ServedModel,
    package_dir: &Path,
    package: &PackageTree,
) -> Result<(), String> {
    if value.get("schema_version").and_then(Value::as_str)
        != Some("ullm.aq4_production_p2_identity.v2")
        || value.get("status").and_then(Value::as_str) != Some("bound")
    {
        return Err("P2 identity is not a bound v2 identity".to_string());
    }
    let declared = value
        .get("identity_sha256")
        .and_then(Value::as_str)
        .filter(|value| valid_sha256(value))
        .ok_or_else(|| "P2 identity SHA-256 is invalid".to_string())?;
    if canonical_value_hash(value, Some("identity_sha256"))? != declared {
        return Err("P2 identity self-hash differs".to_string());
    }
    let model_identity = value
        .get("model_identity")
        .and_then(Value::as_object)
        .ok_or_else(|| "P2 model identity is missing".to_string())?;
    if model_identity.get("id").and_then(Value::as_str) != Some(model.public.id.as_str())
        || model_identity.get("revision").and_then(Value::as_str)
            != Some(model.public.revision.as_str())
        || model_identity.get("format_id").and_then(Value::as_str)
            != Some(model.format.format_id.as_str())
        || model_identity
            .get("implementation_id")
            .and_then(Value::as_str)
            != Some(model.format.implementation_id.as_str())
    {
        return Err("P2 model identity differs from served model".to_string());
    }
    let hashes = value
        .get("hash_binding")
        .and_then(Value::as_object)
        .ok_or_else(|| "P2 hash binding is missing".to_string())?;
    if value
        .get("expanded_manifest_sha256")
        .and_then(Value::as_str)
        != hashes
            .get("bound_case_manifest_sha256")
            .and_then(Value::as_str)
        || hashes
            .get("served_model_manifest_sha256")
            .and_then(Value::as_str)
            != Some(model.manifest_sha256.as_str())
        || hashes.get("worker_binary_sha256").and_then(Value::as_str)
            != Some(model.worker.binary_sha256.as_str())
        || hashes
            .get("package_manifest_sha256")
            .and_then(Value::as_str)
            != Some(model.product.package.manifest_sha256.as_str())
        || hashes.get("package_content_sha256").and_then(Value::as_str)
            != Some(package.sha256.as_str())
    {
        return Err("P2 identity hash binding differs".to_string());
    }
    let artifacts = value
        .get("artifacts")
        .and_then(Value::as_object)
        .ok_or_else(|| "P2 artifact paths are missing".to_string())?;
    let paths = [
        (
            artifacts
                .get("served_model_manifest")
                .and_then(Value::as_str),
            model.manifest_path.as_path(),
        ),
        (
            artifacts.get("worker").and_then(Value::as_str),
            model.worker.binary.as_path(),
        ),
        (
            artifacts.get("package_root").and_then(Value::as_str),
            package_dir,
        ),
    ];
    for (declared, actual) in paths {
        let declared = declared
            .ok_or_else(|| "P2 artifact path is missing".to_string())?
            .to_string();
        if Path::new(&declared)
            .canonicalize()
            .map_err(|error| format!("P2 artifact path failed: {error}"))?
            != actual
                .canonicalize()
                .map_err(|error| format!("actual artifact path failed: {error}"))?
        {
            return Err("P2 artifact path differs".to_string());
        }
    }
    if value.get("package_file_count").and_then(Value::as_u64) != Some(package.file_count as u64)
        || !valid_sha256(&case_link.sha256)
    {
        return Err("P2 package/case identity is incomplete".to_string());
    }
    Ok(())
}

fn load_common_binding(args: &Args) -> Result<CommonBinding, String> {
    if flag_enabled(env::var_os(DIRECT_TOP1_ENV)) {
        return Err("direct-top1 is incompatible with calibration full logits".to_string());
    }
    reject_symlink_components(&args.output)?;
    if fs::symlink_metadata(&args.output).is_ok() {
        return Err("output root already exists; overwrite is forbidden".to_string());
    }
    let (case_value, case_link) = load_json_link(&args.case_binding, "P2 case")?;
    let declared_case_sha = case_value
        .get("case_sha256")
        .and_then(Value::as_str)
        .filter(|value| valid_sha256(value))
        .ok_or_else(|| "P2 case SHA-256 is invalid".to_string())?
        .to_string();
    if canonical_value_hash(&case_value, Some("case_sha256"))? != declared_case_sha {
        return Err("P2 case self-hash differs".to_string());
    }
    let case: P2CaseBinding = serde_json::from_value(case_value)
        .map_err(|error| format!("P2 case exact schema rejected: {error}"))?;
    let (identity, identity_link) = load_json_link(&args.identity_binding, "P2 identity")?;
    let (preflight, preflight_link) = load_json_link(&args.preflight, "P2 preflight")?;
    validate_preflight(&preflight)?;
    let fixture = load_fixture(&args.fixture, &args.case_id)?;
    let model = load_served_model(&args.served_model_manifest)
        .map_err(|error| format!("served model rejected: {error}"))?;
    validate_model(&model)?;
    validate_required_environment(&model)?;
    let profile = model.profile_snapshot();
    if fixture.prompt_token_ids.len() + fixture.step_count > profile.context_length
        || fixture.step_count > profile.max_new_tokens
        || fixture
            .prompt_token_ids
            .iter()
            .any(|token| *token >= profile.vocab_size)
    {
        return Err("fixture request exceeds served model bounds".to_string());
    }
    validate_case(args, &case, &fixture, &model)?;
    if case.case_sha256 != declared_case_sha {
        return Err("typed P2 case SHA-256 differs".to_string());
    }
    let package_dir = model
        .product
        .root
        .join(&model.product.package.manifest_path)
        .parent()
        .ok_or_else(|| "served package manifest has no parent".to_string())?
        .to_path_buf();
    let package_tree = package_tree_identity(&package_dir)?;
    validate_identity_binding(&identity, &case_link, &model, &package_dir, &package_tree)?;
    let device = observe_device(args.device_index)?;
    if device.requested_index != args.device_index {
        return Err("requested device index differs".to_string());
    }
    validate_device_binding(&case.device, &device, &model)?;
    let executable = env::current_exe()
        .map_err(|error| format!("capture binary lookup failed: {error}"))?
        .canonicalize()
        .map_err(|error| format!("capture binary canonicalization failed: {error}"))?;
    let capture_binary = Link {
        path: executable.to_string_lossy().into_owned(),
        sha256: sha256_file(&executable, "capture binary")?,
    };
    if capture_binary.sha256 == model.worker.binary_sha256 {
        return Err("capture binary and served worker roles must be distinct".to_string());
    }
    Ok(CommonBinding {
        model,
        fixture,
        case,
        case_link,
        identity_link,
        preflight_link,
        preflight,
        package_dir,
        package_tree,
        device,
        capture_binary,
    })
}

fn canonical_source_aggregate(value: &Value) -> Result<String, String> {
    let mut bytes = serde_json::to_vec(value)
        .map_err(|error| format!("source aggregate serialization failed: {error}"))?;
    bytes.push(b'\n');
    Ok(sha256_bytes(&bytes))
}

fn validate_source_file_identities(value: &Value, label: &str) -> Result<(), String> {
    let files = value
        .as_array()
        .filter(|files| !files.is_empty())
        .ok_or_else(|| format!("{label} must be a nonempty array"))?;
    let mut previous = None::<&str>;
    for file in files {
        exact_fields(file, &["bytes", "file", "sha256"], label)?;
        let name = file
            .get("file")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| format!("{label} file name is invalid"))?;
        if previous.is_some_and(|value| value >= name)
            || file
                .get("bytes")
                .and_then(Value::as_u64)
                .is_none_or(|v| v == 0)
            || file
                .get("sha256")
                .and_then(Value::as_str)
                .is_none_or(|value| !valid_sha256(value))
        {
            return Err(format!("{label} file identity is invalid or unsorted"));
        }
        previous = Some(name);
    }
    Ok(())
}

fn validate_source_identity(manifest: &Value, common: &CommonBinding) -> Result<String, String> {
    let identity = manifest
        .get("identity")
        .ok_or_else(|| "source identity is missing".to_string())?;
    exact_fields(
        identity,
        &[
            "artifact",
            "model_id",
            "model_revision",
            "source_checkpoint",
            "tokenizer",
            "hidden_size",
            "vocab_size",
        ],
        "source identity",
    )?;
    if identity.get("model_id").and_then(Value::as_str)
        != Some(common.model.public.upstream_id.as_str())
        || identity.get("model_revision").and_then(Value::as_str)
            != Some(common.model.public.revision.as_str())
        || identity.get("hidden_size").and_then(Value::as_u64) != Some(HIDDEN_SIZE as u64)
        || identity.get("vocab_size").and_then(Value::as_u64) != Some(VOCAB_SIZE as u64)
    {
        return Err("source model/vector identity differs from served model".to_string());
    }
    let artifact = &identity["artifact"];
    exact_fields(
        artifact,
        &["package_manifest_sha256", "artifact_manifest_sha256"],
        "source artifact identity",
    )?;
    if !artifact["package_manifest_sha256"].is_null()
        || !artifact["artifact_manifest_sha256"].is_null()
    {
        return Err("independent source must not claim AQ4 artifact identity".to_string());
    }
    let source = &identity["source_checkpoint"];
    exact_fields(
        source,
        &["aggregate_sha256", "dtype", "files", "root"],
        "source checkpoint identity",
    )?;
    let tokenizer = &identity["tokenizer"];
    exact_fields(
        tokenizer,
        &["aggregate_sha256", "files", "root"],
        "source tokenizer identity",
    )?;
    for (value, label) in [
        (source, "source checkpoint identity"),
        (tokenizer, "source tokenizer identity"),
    ] {
        let root = value
            .get("root")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{label} root is missing"))?;
        if !Path::new(root).is_absolute() {
            return Err(format!("{label} root must be absolute"));
        }
        validate_source_file_identities(&value["files"], label)?;
        let aggregate = value
            .get("aggregate_sha256")
            .and_then(Value::as_str)
            .filter(|value| valid_sha256(value))
            .ok_or_else(|| format!("{label} aggregate is invalid"))?;
        if canonical_source_aggregate(&value["files"])? != aggregate {
            return Err(format!("{label} aggregate differs"));
        }
    }
    let dtype = source
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| "source checkpoint dtype is missing".to_string())?;
    if !matches!(dtype, "bfloat16" | "bf16" | "float32" | "f32") {
        return Err("source checkpoint dtype is not BF16/F32".to_string());
    }
    Ok(tokenizer["aggregate_sha256"]
        .as_str()
        .expect("validated tokenizer aggregate")
        .to_string())
}

fn validate_vector_ref(
    vector: &VectorRef,
    offset: u64,
    elements: usize,
    label: &str,
) -> Result<(), String> {
    let bytes = elements
        .checked_mul(F32_BYTES)
        .ok_or_else(|| format!("{label} byte count overflows"))? as u64;
    if vector.offset_bytes != offset
        || vector.bytes != bytes
        || vector.elements != elements
        || vector.dtype != "f32"
        || vector.endianness != "little"
        || !valid_sha256(&vector.sha256)
    {
        return Err(format!("{label} vector contract differs"));
    }
    Ok(())
}

fn validate_source_row_identity(
    row: &VectorRow,
    source_case: &SourceCase,
    step: usize,
    previous_greedy: Option<usize>,
) -> Result<(), String> {
    if row.case_id != source_case.case_id
        || row.step != step
        || row.semantic_input_id
            != source_case
                .semantic_input_id
                .as_deref()
                .unwrap_or(&source_case.case_id)
        || row.observation != source_case.observation.as_deref().unwrap_or("first_token")
    {
        return Err("source row case/step semantics are swapped".to_string());
    }
    let input_tokens = if step == 0 {
        source_case.prompt_token_ids.as_slice()
    } else {
        std::slice::from_ref(
            previous_greedy
                .as_ref()
                .ok_or_else(|| "source previous greedy is missing".to_string())?,
        )
    };
    if row.input_token_ids_sha256 != canonical_token_hash(input_tokens)? {
        return Err("source row input-token hash differs".to_string());
    }
    Ok(())
}

fn update_topk(topk: &mut Vec<TopEntry>, token_id: usize, logit: f32) {
    topk.push(TopEntry { token_id, logit });
    topk.sort_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
    topk.truncate(TOP_K);
}

fn scan_vector_region(
    file: &mut File,
    offset: u64,
    elements: usize,
    chunk_elements: usize,
    rank: bool,
) -> Result<(String, u64, Vec<TopEntry>), String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("vector seek failed: {error}"))?;
    let mut digest = Sha256::new();
    let mut nonfinite = 0_u64;
    let mut topk = Vec::with_capacity(TOP_K + 1);
    let mut visited = 0usize;
    let mut buffer = vec![0_u8; chunk_elements.min(elements).max(1) * F32_BYTES];
    while visited < elements {
        let count = (elements - visited).min(chunk_elements);
        let bytes = count * F32_BYTES;
        file.read_exact(&mut buffer[..bytes])
            .map_err(|error| format!("vector row is short: {error}"))?;
        digest.update(&buffer[..bytes]);
        for (index, encoded) in buffer[..bytes].chunks_exact(F32_BYTES).enumerate() {
            let value = f32::from_le_bytes(encoded.try_into().expect("f32 chunk length"));
            if !value.is_finite() {
                nonfinite = nonfinite
                    .checked_add(1)
                    .ok_or_else(|| "nonfinite counter overflows".to_string())?;
            }
            if rank {
                update_topk(&mut topk, visited + index, value);
            }
        }
        visited += count;
    }
    Ok((format!("{:x}", digest.finalize()), nonfinite, topk))
}

fn validate_source_sums(root: &Path) -> Result<(), String> {
    let sums_path = root.join("SHA256SUMS");
    let bytes = read_regular(&sums_path, "source SHA256SUMS", 8 * 1024 * 1024)?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("source SHA256SUMS is not UTF-8: {error}"))?;
    let mut sums = BTreeMap::new();
    for (line_number, line) in text.lines().enumerate() {
        let (digest, name) = line
            .split_once("  ")
            .ok_or_else(|| format!("source SHA256SUMS line {} is invalid", line_number + 1))?;
        if !valid_sha256(digest)
            || name.is_empty()
            || sums.insert(name.to_string(), digest.to_string()).is_some()
        {
            return Err(format!(
                "source SHA256SUMS line {} is invalid",
                line_number + 1
            ));
        }
    }
    let expected = BTreeSet::from([
        "manifest.json".to_string(),
        "rows.jsonl".to_string(),
        "vectors/hidden.f32le".to_string(),
        "vectors/logits.f32le".to_string(),
    ]);
    if sums.keys().cloned().collect::<BTreeSet<_>>() != expected {
        return Err("source SHA256SUMS file set differs".to_string());
    }
    for (name, expected) in sums {
        let path = safe_relative(root, &name, "source SHA256SUMS entry")?;
        if sha256_file(&path, "source SHA256SUMS entry")? != expected {
            return Err(format!("source SHA256SUMS digest differs for {name}"));
        }
    }
    Ok(())
}

fn load_source_artifact(args: &Args, common: &CommonBinding) -> Result<SourceArtifact, String> {
    reject_symlink_components(&args.source_root)?;
    let root_metadata = fs::symlink_metadata(&args.source_root)
        .map_err(|error| format!("source root metadata failed: {error}"))?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err("source root must be a non-symlink directory".to_string());
    }
    let root = args
        .source_root
        .canonicalize()
        .map_err(|error| format!("source root canonicalization failed: {error}"))?;
    let output_absolute = if args.output.is_absolute() {
        args.output.clone()
    } else {
        env::current_dir()
            .map_err(|error| format!("current directory lookup failed: {error}"))?
            .join(&args.output)
    };
    if output_absolute == root
        || output_absolute.starts_with(&root)
        || root.starts_with(&output_absolute)
    {
        return Err("source and output roots must be distinct".to_string());
    }
    validate_source_sums(&root)?;
    let manifest_path = root.join("manifest.json");
    let (manifest, manifest_link) = load_json_link(&manifest_path, "source manifest")?;
    exact_fields(
        &manifest,
        &[
            "schema_version",
            "oracle_kind",
            "status",
            "evidence_class",
            "usable_as_source_evidence",
            "promotion_eligible",
            "created_utc",
            "identity",
            "parent_sampled_oracle",
            "vector_contract",
            "limits",
            "cases",
            "files",
            "runtime",
            "legacy_cross_check",
        ],
        "source manifest",
    )?;
    if manifest["schema_version"] != SOURCE_SCHEMA
        || manifest["oracle_kind"] != "independent_source_full"
        || manifest["status"] != "available"
        || manifest["evidence_class"] != "production"
        || manifest["usable_as_source_evidence"] != true
        || manifest["promotion_eligible"] != false
    {
        return Err("source manifest is not available independent source evidence".to_string());
    }
    let tokenizer_sha256 = validate_source_identity(&manifest, common)?;
    exact_fields(
        &manifest["vector_contract"],
        &[
            "hidden_shape",
            "logits_shape",
            "dtype",
            "endianness",
            "layout",
            "chunk_elements",
            "row_bytes",
            "semantic_hidden",
            "semantic_logits",
        ],
        "source vector contract",
    )?;
    if manifest["vector_contract"]["hidden_shape"] != json!([HIDDEN_SIZE])
        || manifest["vector_contract"]["logits_shape"] != json!([VOCAB_SIZE])
        || manifest["vector_contract"]["dtype"] != "f32"
        || manifest["vector_contract"]["endianness"] != "little"
        || manifest["vector_contract"]["layout"] != "flat"
        || manifest["vector_contract"]["semantic_hidden"] != "final_rmsnorm_hidden_used_by_lm_head"
        || manifest["vector_contract"]["semantic_logits"] != "raw_pre_softmax_lm_head_logits"
        || manifest["vector_contract"]["row_bytes"].as_u64()
            != Some(((HIDDEN_SIZE + VOCAB_SIZE) * F32_BYTES) as u64)
    {
        return Err("source vector shape/layout differs".to_string());
    }
    let source_chunk = manifest["vector_contract"]["chunk_elements"]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| (1..=1_048_576).contains(value))
        .ok_or_else(|| "source chunk_elements is invalid".to_string())?;
    exact_fields(
        &manifest["limits"],
        &["max_case_file_bytes", "max_cases", "max_rows", "max_steps"],
        "source limits",
    )?;
    if manifest["limits"]
        != json!({
            "max_case_file_bytes": 4 * 1024 * 1024,
            "max_cases": MAX_CASES,
            "max_rows": MAX_ROWS,
            "max_steps": MAX_STEPS,
        })
    {
        return Err("source limits differ from v1 contract".to_string());
    }
    exact_fields(
        &manifest["files"],
        &["rows", "hidden", "logits"],
        "source files",
    )?;
    let rows_path = safe_relative(
        &root,
        manifest["files"]["rows"]
            .as_str()
            .ok_or_else(|| "source rows path is invalid".to_string())?,
        "source rows",
    )?;
    let hidden_path = safe_relative(
        &root,
        manifest["files"]["hidden"]
            .as_str()
            .ok_or_else(|| "source hidden path is invalid".to_string())?,
        "source hidden",
    )?;
    let logits_path = safe_relative(
        &root,
        manifest["files"]["logits"]
            .as_str()
            .ok_or_else(|| "source logits path is invalid".to_string())?,
        "source logits",
    )?;
    if manifest["files"]
        != json!({
            "rows": "rows.jsonl",
            "hidden": "vectors/hidden.f32le",
            "logits": "vectors/logits.f32le",
        })
    {
        return Err("source file names differ from v1 contract".to_string());
    }
    exact_fields(
        &manifest["cases"],
        &["path", "sha256", "case_count", "row_count"],
        "source cases",
    )?;
    let cases_path = PathBuf::from(
        manifest["cases"]["path"]
            .as_str()
            .ok_or_else(|| "source cases path is invalid".to_string())?,
    );
    let cases_bytes = read_regular(&cases_path, "source cases", 4 * 1024 * 1024)?;
    let cases_sha = sha256_bytes(&cases_bytes);
    if manifest["cases"]["sha256"].as_str() != Some(cases_sha.as_str()) {
        return Err("source cases hash differs".to_string());
    }
    let cases: SourceCasesFile =
        serde_json::from_value(parse_strict_json(&cases_bytes, "source cases")?)
            .map_err(|error| format!("source cases exact schema rejected: {error}"))?;
    if cases.schema_version != SOURCE_CASES_SCHEMA
        || cases.cases.is_empty()
        || cases.cases.len() > MAX_CASES
        || manifest["cases"]["case_count"].as_u64() != Some(cases.cases.len() as u64)
    {
        return Err("source cases identity/count differs".to_string());
    }
    let row_count = cases.cases.iter().try_fold(0usize, |sum, case| {
        if case.case_id.is_empty()
            || case.prompt_token_ids.is_empty()
            || case.prompt_token_ids.len() > 4096
            || case.step_count == 0
            || case.step_count > MAX_STEPS
            || case
                .prompt_token_ids
                .iter()
                .any(|token| *token >= VOCAB_SIZE)
        {
            return Err("source case exceeds bounded contract".to_string());
        }
        sum.checked_add(case.step_count)
            .ok_or_else(|| "source row count overflows".to_string())
    })?;
    if row_count > MAX_ROWS || manifest["cases"]["row_count"].as_u64() != Some(row_count as u64) {
        return Err("source row count differs or exceeds bound".to_string());
    }
    let selected = cases
        .cases
        .iter()
        .find(|case| case.case_id == args.case_id)
        .ok_or_else(|| "source cases do not contain requested case".to_string())?;
    if selected.prompt_token_ids != common.fixture.prompt_token_ids
        || selected.step_count != common.fixture.step_count
    {
        return Err("source case prompt/steps differ from bound fixture".to_string());
    }
    let hidden_expected = row_count as u64 * HIDDEN_SIZE as u64 * F32_BYTES as u64;
    let logits_expected = row_count as u64 * VOCAB_SIZE as u64 * F32_BYTES as u64;
    if fs::metadata(&hidden_path)
        .map_err(|error| format!("source hidden metadata failed: {error}"))?
        .len()
        != hidden_expected
        || fs::metadata(&logits_path)
            .map_err(|error| format!("source logits metadata failed: {error}"))?
            .len()
            != logits_expected
    {
        return Err("source sidecar sizes differ from row count".to_string());
    }
    let rows_file =
        File::open(&rows_path).map_err(|error| format!("source rows open failed: {error}"))?;
    let mut reader = BufReader::new(rows_file);
    let mut hidden =
        File::open(&hidden_path).map_err(|error| format!("source hidden open failed: {error}"))?;
    let mut logits =
        File::open(&logits_path).map_err(|error| format!("source logits open failed: {error}"))?;
    let source_metadata_before = [
        fs::symlink_metadata(&rows_path)
            .map_err(|error| format!("source rows metadata failed: {error}"))?,
        fs::symlink_metadata(&hidden_path)
            .map_err(|error| format!("source hidden metadata failed: {error}"))?,
        fs::symlink_metadata(&logits_path)
            .map_err(|error| format!("source logits metadata failed: {error}"))?,
    ];
    let mut line = Vec::new();
    let mut hidden_offset = 0_u64;
    let mut logits_offset = 0_u64;
    let mut selected_rows = Vec::with_capacity(selected.step_count);
    let mut selected_row_sha256s = Vec::with_capacity(selected.step_count);
    let mut seen_cases = BTreeSet::new();
    for source_case in &cases.cases {
        if !seen_cases.insert(source_case.case_id.as_str()) {
            return Err("source cases contain duplicate case_id".to_string());
        }
        let mut previous_greedy = None;
        for step in 0..source_case.step_count {
            line.clear();
            let read = reader
                .read_until(b'\n', &mut line)
                .map_err(|error| format!("source rows read failed: {error}"))?;
            if read == 0 || line.len() > MAX_ROW_LINE_BYTES {
                return Err("source rows are short or a row exceeds its bound".to_string());
            }
            let row_value = parse_strict_json(&line, "source row")?;
            let row_sha256 = sha256_bytes(&line);
            let row: VectorRow = serde_json::from_value(row_value)
                .map_err(|error| format!("source row exact schema rejected: {error}"))?;
            validate_source_row_identity(&row, source_case, step, previous_greedy)?;
            validate_vector_ref(&row.hidden, hidden_offset, HIDDEN_SIZE, "source hidden")?;
            validate_vector_ref(&row.logits, logits_offset, VOCAB_SIZE, "source logits")?;
            let (hidden_sha, hidden_nonfinite, _) =
                scan_vector_region(&mut hidden, hidden_offset, HIDDEN_SIZE, source_chunk, false)?;
            let (logits_sha, logits_nonfinite, topk) =
                scan_vector_region(&mut logits, logits_offset, VOCAB_SIZE, source_chunk, true)?;
            if hidden_sha != row.hidden.sha256
                || hidden_nonfinite != row.hidden.nonfinite_count
                || logits_sha != row.logits.sha256
                || logits_nonfinite != row.logits.nonfinite_count
                || topk != row.topk
                || row.topk.len() != TOP_K
                || row.greedy_token_id != row.topk[0].token_id
                || row.finite != (hidden_nonfinite == 0 && logits_nonfinite == 0)
                || !row.finite
            {
                return Err("source row hash/top-k/finite contract differs".to_string());
            }
            hidden_offset += row.hidden.bytes;
            logits_offset += row.logits.bytes;
            previous_greedy = Some(row.greedy_token_id);
            if source_case.case_id == args.case_id {
                selected_rows.push(row);
                selected_row_sha256s.push(row_sha256);
            }
        }
    }
    line.clear();
    if reader
        .read_until(b'\n', &mut line)
        .map_err(|error| format!("source rows tail read failed: {error}"))?
        != 0
    {
        return Err("source rows contain extra records".to_string());
    }
    if selected_rows.len() != selected.step_count {
        return Err("selected source row coverage differs".to_string());
    }
    for (path, before) in [
        (&rows_path, &source_metadata_before[0]),
        (&hidden_path, &source_metadata_before[1]),
        (&logits_path, &source_metadata_before[2]),
    ] {
        let after = fs::symlink_metadata(path)
            .map_err(|error| format!("source post-scan metadata failed: {error}"))?;
        if metadata_identity(before) != metadata_identity(&after) {
            return Err("source artifact changed during row validation".to_string());
        }
    }
    let replay_tokens = selected_rows
        .iter()
        .map(|row| row.greedy_token_id)
        .collect::<Vec<_>>();
    let replay_sha256 =
        Qwen35Aq4CalibrationReplay::source_sequence_sha256_for_tokens(&replay_tokens)?;
    Ok(SourceArtifact {
        manifest,
        manifest_link,
        tokenizer_sha256,
        rows: selected_rows,
        row_sha256s: selected_row_sha256s,
        replay_tokens,
        replay_sha256,
    })
}

struct CaptureObserver<'a> {
    hidden: &'a mut File,
    logits: &'a mut File,
    hidden_offset: u64,
    logits_offset: u64,
    hidden_elements: usize,
    logit_elements: usize,
    hidden_digest: Sha256,
    logits_digest: Sha256,
    hidden_nonfinite: u64,
    logits_nonfinite: u64,
    topk: Vec<TopEntry>,
    began: bool,
    finished: bool,
}

impl<'a> CaptureObserver<'a> {
    fn new(hidden: &'a mut File, logits: &'a mut File) -> Result<Self, String> {
        let hidden_offset = hidden
            .stream_position()
            .map_err(|error| format!("target hidden position failed: {error}"))?;
        let logits_offset = logits
            .stream_position()
            .map_err(|error| format!("target logits position failed: {error}"))?;
        Ok(Self {
            hidden,
            logits,
            hidden_offset,
            logits_offset,
            hidden_elements: 0,
            logit_elements: 0,
            hidden_digest: Sha256::new(),
            logits_digest: Sha256::new(),
            hidden_nonfinite: 0,
            logits_nonfinite: 0,
            topk: Vec::with_capacity(TOP_K + 1),
            began: false,
            finished: false,
        })
    }

    fn write_values(
        file: &mut File,
        digest: &mut Sha256,
        nonfinite: &mut u64,
        values: &[f32],
    ) -> Result<(), String> {
        let mut bytes = Vec::with_capacity(values.len() * F32_BYTES);
        for value in values {
            if !value.is_finite() {
                *nonfinite = nonfinite
                    .checked_add(1)
                    .ok_or_else(|| "target nonfinite counter overflows".to_string())?;
            }
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&bytes)
            .map_err(|error| format!("target sidecar write failed: {error}"))?;
        digest.update(&bytes);
        Ok(())
    }

    fn finish_row(self, source: &VectorRow) -> Result<VectorRow, String> {
        if !self.began
            || !self.finished
            || self.hidden_elements != HIDDEN_SIZE
            || self.logit_elements != VOCAB_SIZE
            || self.topk.len() != TOP_K
        {
            return Err("target observer did not complete the exact row".to_string());
        }
        let hidden_nonfinite = self.hidden_nonfinite;
        let logits_nonfinite = self.logits_nonfinite;
        let hidden = VectorRef {
            offset_bytes: self.hidden_offset,
            bytes: (HIDDEN_SIZE * F32_BYTES) as u64,
            elements: HIDDEN_SIZE,
            dtype: "f32".into(),
            endianness: "little".into(),
            sha256: format!("{:x}", self.hidden_digest.finalize()),
            nonfinite_count: hidden_nonfinite,
        };
        let logits = VectorRef {
            offset_bytes: self.logits_offset,
            bytes: (VOCAB_SIZE * F32_BYTES) as u64,
            elements: VOCAB_SIZE,
            dtype: "f32".into(),
            endianness: "little".into(),
            sha256: format!("{:x}", self.logits_digest.finalize()),
            nonfinite_count: logits_nonfinite,
        };
        Ok(VectorRow {
            case_id: source.case_id.clone(),
            step: source.step,
            semantic_input_id: source.semantic_input_id.clone(),
            observation: source.observation.clone(),
            input_token_ids_sha256: source.input_token_ids_sha256.clone(),
            hidden,
            logits,
            greedy_token_id: self.topk[0].token_id,
            topk: self.topk,
            finite: hidden_nonfinite == 0 && logits_nonfinite == 0,
        })
    }
}

impl Qwen35Aq4CalibrationObserver for CaptureObserver<'_> {
    fn begin(&mut self, hidden_elements: usize, logit_elements: usize) -> Result<(), String> {
        if self.began || hidden_elements != HIDDEN_SIZE || logit_elements != VOCAB_SIZE {
            return Err(format!(
                "target observer shape differs: hidden={hidden_elements} logits={logit_elements}"
            ));
        }
        self.began = true;
        Ok(())
    }

    fn observe_hidden_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        if !self.began || self.finished || start != self.hidden_elements || values.is_empty() {
            return Err("target hidden chunks are empty, overlapping, or out of order".to_string());
        }
        let next = start
            .checked_add(values.len())
            .ok_or_else(|| "target hidden element count overflows".to_string())?;
        if next > HIDDEN_SIZE {
            return Err("target hidden chunk exceeds shape".to_string());
        }
        Self::write_values(
            self.hidden,
            &mut self.hidden_digest,
            &mut self.hidden_nonfinite,
            values,
        )?;
        self.hidden_elements = next;
        Ok(())
    }

    fn observe_logit_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        if !self.began || self.finished || start != self.logit_elements || values.is_empty() {
            return Err("target logit chunks are empty, overlapping, or out of order".to_string());
        }
        let next = start
            .checked_add(values.len())
            .ok_or_else(|| "target logit element count overflows".to_string())?;
        if next > VOCAB_SIZE {
            return Err("target logit chunk exceeds shape".to_string());
        }
        Self::write_values(
            self.logits,
            &mut self.logits_digest,
            &mut self.logits_nonfinite,
            values,
        )?;
        for (offset, value) in values.iter().copied().enumerate() {
            update_topk(&mut self.topk, start + offset, value);
        }
        self.logit_elements = next;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        if !self.began
            || self.finished
            || self.hidden_elements != HIDDEN_SIZE
            || self.logit_elements != VOCAB_SIZE
        {
            return Err("target observer finished an incomplete row".to_string());
        }
        self.finished = true;
        Ok(())
    }
}

fn write_json_line<T: Serialize>(file: &mut File, value: &T, label: &str) -> Result<(), String> {
    serde_json::to_writer(&mut *file, value)
        .map_err(|error| format!("{label} serialization failed: {error}"))?;
    file.write_all(b"\n")
        .map_err(|error| format!("{label} newline write failed: {error}"))
}

fn validate_terminal_audit(audit: &Value, case: &P2CaseBinding) -> Result<(), String> {
    if audit.get("requested_m").and_then(Value::as_u64) != Some(case.prefill_requested_m as u64)
        || audit.get("resolved_m").and_then(Value::as_u64) != Some(case.resolved_m as u64)
        || audit
            .get("actual_token_batch_width")
            .and_then(Value::as_u64)
            != Some(case.resolved_m as u64)
        || audit
            .get("actual_request_batch_width")
            .and_then(Value::as_u64)
            != Some(case.request_count as u64)
    {
        return Err("terminal requested/resolved/actual M differs from case binding".to_string());
    }
    let lifecycle = audit
        .get("lifecycle")
        .and_then(Value::as_object)
        .ok_or_else(|| "terminal lifecycle is missing".to_string())?;
    let count = |object: &serde_json::Map<String, Value>, name: &str| {
        object
            .get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("terminal lifecycle {name} is missing"))
    };
    let prepare = count(lifecycle, "prepare")?;
    let commit = count(lifecycle, "commit")?;
    let discard = count(lifecycle, "discard")?;
    if prepare == 0
        || prepare != commit + discard
        || count(lifecycle, "error")? != 0
        || count(lifecycle, "cancel")? != 0
    {
        return Err("terminal lifecycle counts do not reconcile".to_string());
    }
    for phase in ["prefill", "publication"] {
        let phase = lifecycle
            .get(phase)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("terminal lifecycle {phase} is missing"))?;
        if count(phase, "prepare")? != count(phase, "commit")? + count(phase, "discard")? {
            return Err("terminal phase lifecycle does not reconcile".to_string());
        }
    }
    let reset = lifecycle
        .get("reset")
        .and_then(Value::as_object)
        .ok_or_else(|| "terminal reset lifecycle is missing".to_string())?;
    if count(reset, "attempted")? != 1
        || count(reset, "complete")? != 1
        || count(reset, "failed")? != 0
    {
        return Err("terminal reset did not complete exactly once".to_string());
    }
    let operation = audit
        .get("operation_audit")
        .and_then(Value::as_object)
        .ok_or_else(|| "terminal operation audit is missing".to_string())?;
    if operation.get("coverage_complete").and_then(Value::as_bool) != Some(true) {
        return Err("terminal operation audit coverage is incomplete".to_string());
    }
    Ok(())
}

fn create_temporary_root(output: &Path) -> Result<PathBuf, String> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    reject_symlink_components(parent)?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("output parent creation failed: {error}"))?;
    let name = output
        .file_name()
        .ok_or_else(|| "output root has no file name".to_string())?
        .to_string_lossy();
    let temporary = parent.join(format!(".{name}.{}.incomplete", std::process::id()));
    fs::create_dir(&temporary)
        .map_err(|error| format!("temporary root creation failed: {error}"))?;
    Ok(temporary)
}

#[cfg(target_os = "linux")]
fn rename_noreplace(source: &Path, destination: &Path) -> Result<(), String> {
    const AT_FDCWD: i32 = -100;
    const RENAME_NOREPLACE: u32 = 1;
    unsafe extern "C" {
        fn renameat2(
            olddirfd: i32,
            oldpath: *const std::ffi::c_char,
            newdirfd: i32,
            newpath: *const std::ffi::c_char,
            flags: u32,
        ) -> i32;
    }
    let source = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| "temporary root path contains NUL".to_string())?;
    let destination = CString::new(destination.as_os_str().as_bytes())
        .map_err(|_| "output root path contains NUL".to_string())?;
    // SAFETY: both C strings remain alive for the call and point to NUL-terminated path bytes.
    let result = unsafe {
        renameat2(
            AT_FDCWD,
            source.as_ptr(),
            AT_FDCWD,
            destination.as_ptr(),
            RENAME_NOREPLACE,
        )
    };
    if result != 0 {
        return Err(format!(
            "atomic non-overwrite publication failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn rename_noreplace(_source: &Path, _destination: &Path) -> Result<(), String> {
    Err("atomic non-overwrite directory publication requires Linux renameat2".to_string())
}

fn write_manifest_and_sums(root: &Path, manifest: &Value) -> Result<(), String> {
    let manifest_path = root.join("manifest.json");
    let mut manifest_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&manifest_path)
        .map_err(|error| format!("target manifest creation failed: {error}"))?;
    serde_json::to_writer_pretty(&mut manifest_file, manifest)
        .map_err(|error| format!("target manifest serialization failed: {error}"))?;
    manifest_file
        .write_all(b"\n")
        .map_err(|error| format!("target manifest newline failed: {error}"))?;
    manifest_file
        .sync_all()
        .map_err(|error| format!("target manifest fsync failed: {error}"))?;
    drop(manifest_file);
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)
            .map_err(|error| format!("target artifact enumeration failed: {error}"))?
        {
            let path = entry
                .map_err(|error| format!("target artifact entry failed: {error}"))?
                .path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("target artifact metadata failed: {error}"))?;
            if metadata.file_type().is_symlink() {
                return Err("target artifact contains a symlink".to_string());
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() && path.file_name().is_none_or(|name| name != "SHA256SUMS")
            {
                files.push(path);
            }
        }
    }
    files.sort_by_key(|path| {
        path.strip_prefix(root)
            .expect("artifact path has root")
            .as_os_str()
            .as_bytes()
            .to_vec()
    });
    let sums_path = root.join("SHA256SUMS");
    let mut sums = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&sums_path)
        .map_err(|error| format!("target SHA256SUMS creation failed: {error}"))?;
    for path in files {
        let relative = path
            .strip_prefix(root)
            .expect("artifact path has root")
            .to_string_lossy();
        writeln!(
            sums,
            "{}  {}",
            sha256_file(&path, "target artifact file")?,
            relative
        )
        .map_err(|error| format!("target SHA256SUMS write failed: {error}"))?;
    }
    sums.sync_all()
        .map_err(|error| format!("target SHA256SUMS fsync failed: {error}"))?;
    File::open(root)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("target root fsync failed: {error}"))
}

fn publish_root(temporary: &Path, output: &Path) -> Result<(), String> {
    if fs::symlink_metadata(output).is_ok() {
        return Err("output root appeared before publication".to_string());
    }
    rename_noreplace(temporary, output)?;
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("output parent fsync failed: {error}"))
}

fn execution_row(
    source: &SourceArtifact,
    step: usize,
    predicted_token_id: usize,
    committed_token_id: usize,
    generation_epoch: u64,
) -> Result<ExecutionRow, String> {
    if step >= source.rows.len()
        || source.replay_tokens.get(step).copied() != Some(committed_token_id)
        || generation_epoch == 0
    {
        return Err("execution row source step/epoch binding differs".to_string());
    }
    Ok(ExecutionRow {
        case_id: source.rows[step].case_id.clone(),
        step,
        source_sequence_sha256: source.replay_sha256.clone(),
        source_row_sha256: source.row_sha256s[step].clone(),
        predicted_token_id,
        committed_token_id,
        diverged: predicted_token_id != committed_token_id,
        generation_epoch,
        observation_complete: true,
        publication_committed: true,
        lifecycle: "prepared_observed_source_committed",
    })
}

fn capture_into_temporary(
    args: &Args,
    common: &CommonBinding,
    source: &SourceArtifact,
    temporary: &Path,
) -> Result<bool, String> {
    let vectors = temporary.join("vectors");
    fs::create_dir(&vectors)
        .map_err(|error| format!("target vectors directory creation failed: {error}"))?;
    let mut hidden = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(vectors.join("hidden.f32le"))
        .map_err(|error| format!("target hidden creation failed: {error}"))?;
    let mut logits = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(vectors.join("logits.f32le"))
        .map_err(|error| format!("target logits creation failed: {error}"))?;
    let mut rows = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary.join("rows.jsonl"))
        .map_err(|error| format!("target rows creation failed: {error}"))?;
    let mut execution_rows = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary.join("execution-rows.jsonl"))
        .map_err(|error| format!("target execution rows creation failed: {error}"))?;
    let profile = common.model.profile_snapshot();
    if profile.vocab_size != VOCAB_SIZE {
        return Err("served profile calibration shapes differ".to_string());
    }
    let model_config = Qwen35Aq4ModelLoadConfig {
        package_dir: common.package_dir.clone(),
        device_index: args.device_index,
        expected_architecture: Some(profile.device.clone()),
        chunk_bytes: DEFAULT_LOAD_CHUNK_BYTES,
        context_length: profile.context_length,
        kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
        layer_indices: None,
        lm_head_mode: PackageLmHeadMode::GpuResidentF32,
        lm_head_chunk_rows: DEFAULT_LM_HEAD_CHUNK_ROWS,
    };
    let session_config =
        Qwen35Aq4SessionConfig::greedy(common.fixture.step_count, profile.eos_token_ids.clone())
            .with_prefill_chunk_tokens(args.requested_m)?;
    let mut session = Qwen35Aq4InferenceSession::load(model_config, session_config)?;
    let request = InferenceRequest::new_with_eos(
        format!("aq4-p2-calibration-{}", args.case_id),
        common.fixture.prompt_token_ids.clone(),
        common.fixture.step_count,
        profile.eos_token_ids,
        SamplingParams::greedy_with_top_k(0, 1),
    );
    let replay = Qwen35Aq4CalibrationReplay::new(
        source.replay_sha256.clone(),
        source.replay_tokens.clone(),
    )?;
    session.start_calibration_request(request, CancellationToken::new(), replay)?;
    let mut step = 0usize;
    let mut nonfinite_rows = 0usize;
    let drive_result = (|| -> Result<Value, String> {
        loop {
            match session.prepare_advance()? {
                SessionAdvance::PromptProgress { .. } => {}
                SessionAdvance::Token { prepared, .. } => {
                    let source_row = source
                        .rows
                        .get(step)
                        .ok_or_else(|| "target emitted an extra prepared row".to_string())?;
                    let generation_epoch = session
                        .model()
                        .last_generation_state_epoch()
                        .ok_or_else(|| "prepared target row has no generation epoch".to_string())?;
                    let mut observer = CaptureObserver::new(&mut hidden, &mut logits)?;
                    session.observe_prepared_calibration(&prepared, &mut observer)?;
                    let target_row = observer.finish_row(source_row)?;
                    if target_row.greedy_token_id != prepared.token_id {
                        return Err("observer top-1 differs from prepared token".to_string());
                    }
                    let committed = source.replay_tokens[step];
                    let predicted = prepared.token_id;
                    session.publish_calibration_prepared(prepared, |_| Ok(()))?;
                    let execution =
                        execution_row(source, step, predicted, committed, generation_epoch)?;
                    nonfinite_rows += usize::from(!target_row.finite);
                    write_json_line(&mut rows, &target_row, "target vector row")?;
                    write_json_line(&mut execution_rows, &execution, "target execution row")?;
                    step = step
                        .checked_add(1)
                        .ok_or_else(|| "target row counter overflows".to_string())?;
                    if session.status() == Qwen35Aq4SessionStatus::Terminal {
                        session.finish_and_reset()?;
                        break;
                    }
                }
                SessionAdvance::CancellationObserved => {
                    return Err("calibration unexpectedly observed cancellation".to_string());
                }
            }
        }
        if step != source.rows.len() || step != common.fixture.step_count {
            return Err("target emitted a short row sequence".to_string());
        }
        let terminal_audit = session
            .last_terminal_request_execution_audit()
            .ok_or_else(|| "calibration terminal audit is missing".to_string())?;
        let terminal_audit = serde_json::to_value(terminal_audit)
            .map_err(|error| format!("terminal audit serialization failed: {error}"))?;
        validate_terminal_audit(&terminal_audit, &common.case)?;
        if session.status() != Qwen35Aq4SessionStatus::Ready {
            return Err("calibration did not return to Ready after reset".to_string());
        }
        Ok(terminal_audit)
    })();
    let terminal_audit = match drive_result {
        Ok(audit) => audit,
        Err(error) => {
            let reset = if session.status() == Qwen35Aq4SessionStatus::Ready {
                Ok(())
            } else {
                session.abort_and_reset().map(|_| ())
            };
            let shutdown = session.shutdown();
            return match (reset, shutdown) {
                (Ok(_), Ok(())) => Err(error),
                (reset, shutdown) => Err(format!(
                    "{error}; failure cleanup reset={:?} shutdown={:?}",
                    reset.err(),
                    shutdown.err()
                )),
            };
        }
    };
    session.shutdown()?;
    for file in [&mut hidden, &mut logits, &mut rows, &mut execution_rows] {
        file.sync_all()
            .map_err(|error| format!("target sidecar fsync failed: {error}"))?;
    }
    drop(hidden);
    drop(logits);
    drop(rows);
    drop(execution_rows);
    let rows_link = Link {
        path: "rows.jsonl".into(),
        sha256: sha256_file(&temporary.join("rows.jsonl"), "target rows")?,
    };
    let execution_rows_link = Link {
        path: "execution-rows.jsonl".into(),
        sha256: sha256_file(
            &temporary.join("execution-rows.jsonl"),
            "target execution rows",
        )?,
    };
    let status = complete_capture_status(nonfinite_rows);
    let identity = json!({
        "model_id": common.model.public.upstream_id,
        "model_revision": common.model.public.revision,
        "format_id": common.model.format.format_id,
        "implementation_id": common.model.format.implementation_id,
        "tokenizer": {"aggregate_sha256": source.tokenizer_sha256},
        "package_content_sha256": common.package_tree.sha256,
        "package_manifest_sha256": common.model.product.package.manifest_sha256,
        "served_model_manifest_sha256": common.model.manifest_sha256,
        "worker_binary_sha256": common.model.worker.binary_sha256,
        "capture_binary_sha256": common.capture_binary.sha256,
    });
    let manifest = json!({
        "schema_version": TARGET_SCHEMA,
        "oracle_kind": args.oracle_kind,
        "status": status,
        "immutable_status": status != "available",
        "capture_complete": true,
        "evidence_class": if status == "available" { "diagnostic_calibration" } else { "blocked" },
        "promotion_eligible": false,
        "identity": identity,
        "binding": {
            "case_id": args.case_id,
            "case_sha256": common.case.case_sha256,
            "requested_m": args.requested_m,
            "resolved_m": terminal_audit["resolved_m"],
            "actual_token_batch_width": terminal_audit["actual_token_batch_width"],
            "actual_request_batch_width": terminal_audit["actual_request_batch_width"],
            "package_root": common.package_dir,
            "package_file_count": common.package_tree.file_count,
            "package_bytes": common.package_tree.bytes,
            "worker_binary": common.model.worker.binary,
            "capture_binary": common.capture_binary.path,
            "device": {
                "requested_index": common.device.requested_index,
                "device_id": common.device.device_id,
                "backend": common.device.backend,
                "name": common.device.name,
                "architecture": common.device.architecture,
            },
            "source": {
                "root": args.source_root,
                "schema_version": source.manifest["schema_version"],
                "oracle_kind": source.manifest["oracle_kind"],
                "manifest": source.manifest_link,
                "source_sequence_sha256": source.replay_sha256,
            },
            "policy": {
                "policy_id": args.policy_id,
                "sampling": "greedy_top1",
                "teacher_forcing": true,
                "committed_sequence": "independent_source_greedy",
            },
            "links": {
                "case": common.case_link,
                "identity": common.identity_link,
                "preflight": common.preflight_link,
            },
            "preflight": common.preflight,
        },
        "vector_contract": {
            "hidden_shape": [HIDDEN_SIZE],
            "logits_shape": [VOCAB_SIZE],
            "dtype": "f32",
            "endianness": "little",
            "layout": "flat",
            "chunk_elements": args.chunk_elements,
            "row_bytes": (HIDDEN_SIZE + VOCAB_SIZE) * F32_BYTES,
            "semantic_hidden": "post_final_rmsnorm_hidden_used_by_lm_head",
            "semantic_logits": "raw_pre_softmax_lm_head_logits",
        },
        "files": {
            "rows": "rows.jsonl",
            "hidden": "vectors/hidden.f32le",
            "logits": "vectors/logits.f32le",
        },
        "execution_rows": {
            "file": execution_rows_link.path,
            "sha256": execution_rows_link.sha256,
            "record_count": step,
            "lockstep_with_vector_rows": true,
        },
        "rows": {
            "file": rows_link.path,
            "sha256": rows_link.sha256,
            "record_count": step,
            "nonfinite_rows": nonfinite_rows,
        },
        "lifecycle": terminal_audit["lifecycle"],
        "operation_audit": terminal_audit["operation_audit"],
        "performance": {
            "timing_eligible": false,
            "reason": "calibration_observer_device_to_host_transfer",
            "raw_v2_schema_emitted": false,
        },
    });
    write_manifest_and_sums(temporary, &manifest)?;
    Ok(status == "available")
}

fn run(args: &Args) -> Result<bool, String> {
    let common = load_common_binding(args)?;
    let source = load_source_artifact(args, &common)?;
    let temporary = create_temporary_root(&args.output)?;
    let result = capture_into_temporary(args, &common, &source, &temporary);
    match result {
        Ok(available) => {
            if let Err(error) = publish_root(&temporary, &args.output) {
                let _ = fs::remove_dir_all(&temporary);
                return Err(error);
            }
            Ok(available)
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            Err(error)
        }
    }
}

fn publish_blocked(args: &Args, error: &str) -> Result<(), String> {
    if fs::symlink_metadata(&args.output).is_ok() {
        return Err("refusing to overwrite an existing output with blocked evidence".to_string());
    }
    let temporary = create_temporary_root(&args.output)?;
    let reason_code = blocked_reason_code(error);
    let manifest = json!({
        "schema_version": TARGET_SCHEMA,
        "oracle_kind": args.oracle_kind,
        "status": if is_oom(error) { "oom" } else { "blocked" },
        "immutable_status": true,
        "capture_complete": false,
        "evidence_class": "blocked",
        "promotion_eligible": false,
        "case_id": args.case_id,
        "requested_m": args.requested_m,
        "policy_id": args.policy_id,
        "failure": {
            "stage": "calibration_capture",
            "reason_code": reason_code,
            "oom": is_oom(error),
        },
        "performance": {
            "timing_eligible": false,
            "raw_v2_schema_emitted": false,
        },
    });
    let result = write_manifest_and_sums(&temporary, &manifest)
        .and_then(|_| publish_root(&temporary, &args.output));
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}

fn main() -> ExitCode {
    let args = match parse_args(env::args_os().skip(1)) {
        Ok(Some(args)) => args,
        Ok(None) => {
            print_help();
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("Qwen3.5 AQ4 P2 calibration arguments rejected: {error}");
            print_help();
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("Qwen3.5 AQ4 P2 calibration completed as immutable blocked evidence");
            ExitCode::from(1)
        }
        Err(error) => {
            if let Err(publish_error) = publish_blocked(&args, &error) {
                eprintln!(
                    "Qwen3.5 AQ4 P2 calibration failed: {error}; blocked artifact publication failed: {publish_error}"
                );
            } else {
                eprintln!("Qwen3.5 AQ4 P2 calibration failed: {error}");
            }
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!(
            "ullm-aq4-p2-calibration-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn vector_ref(elements: usize) -> VectorRef {
        VectorRef {
            offset_bytes: 0,
            bytes: (elements * F32_BYTES) as u64,
            elements,
            dtype: "f32".into(),
            endianness: "little".into(),
            sha256: "a".repeat(64),
            nonfinite_count: 0,
        }
    }

    fn source_row(case_id: &str, step: usize, input_hash: String, greedy: usize) -> VectorRow {
        let topk = (0..TOP_K)
            .map(|index| TopEntry {
                token_id: greedy + index,
                logit: (TOP_K - index) as f32,
            })
            .collect();
        VectorRow {
            case_id: case_id.into(),
            step,
            semantic_input_id: case_id.into(),
            observation: "first_token".into(),
            input_token_ids_sha256: input_hash,
            hidden: vector_ref(HIDDEN_SIZE),
            logits: vector_ref(VOCAB_SIZE),
            greedy_token_id: greedy,
            topk,
            finite: true,
        }
    }

    fn source_artifact(tokens: &[usize]) -> SourceArtifact {
        let rows = tokens
            .iter()
            .copied()
            .enumerate()
            .map(|(step, token)| {
                source_row(
                    "case-0",
                    step,
                    canonical_token_hash(if step == 0 {
                        &[11, 12]
                    } else {
                        &tokens[step - 1..step]
                    })
                    .unwrap(),
                    token,
                )
            })
            .collect::<Vec<_>>();
        SourceArtifact {
            manifest: json!({"schema_version": SOURCE_SCHEMA, "oracle_kind": "independent_source_full"}),
            manifest_link: Link {
                path: "/source/manifest.json".into(),
                sha256: "b".repeat(64),
            },
            tokenizer_sha256: "c".repeat(64),
            row_sha256s: (0..tokens.len()).map(|_| "d".repeat(64)).collect(),
            rows,
            replay_tokens: tokens.to_vec(),
            replay_sha256: Qwen35Aq4CalibrationReplay::source_sequence_sha256_for_tokens(tokens)
                .unwrap(),
        }
    }

    fn p2_case() -> P2CaseBinding {
        P2CaseBinding {
            case_id: "case-0".into(),
            fixture_id: "case-0".into(),
            case_sha256: "e".repeat(64),
            stage_id: "stage".into(),
            stage_order: 1,
            scope: "full_model".into(),
            phase: "cold_prefill".into(),
            mode: "cold_batched".into(),
            baseline_mode: "cold_batched".into(),
            prompt_tokens: 8,
            cached_prefix_tokens: 0,
            context_tokens: 8,
            decode_start_tokens: 8,
            prefill_requested_m: 8,
            resolved_m: 8,
            request_count: 1,
            decode_request_count: 0,
            generated_tokens: 2,
            device: P2CaseDevice {
                device_id: "0".into(),
                backend: "HIP".into(),
                name: "mock".into(),
                architecture: "RDNA4".into(),
                runtime_device_index: 0,
            },
            control_id: "aq4_0_target".into(),
            control: P2CaseControl {
                control_id: "aq4_0_target".into(),
                role: "target".into(),
                format_id: "AQ4_0".into(),
                implementation_id: "qwen35_aq4_rdna4_v1".into(),
                promotion_eligible: true,
            },
            sampling: P2CaseSampling {
                mode: "greedy".into(),
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
                seed: 0,
            },
            format_id: "AQ4_0".into(),
            implementation_id: "qwen35_aq4_rdna4_v1".into(),
            path_oracle_case_id: Some("all-m1".into()),
            path_oracle_result_sha256: None,
        }
    }

    fn terminal_audit(cancel: u64, reset_complete: u64) -> Value {
        json!({
            "requested_m": 8,
            "resolved_m": 8,
            "actual_token_batch_width": 8,
            "actual_request_batch_width": 1,
            "lifecycle": {
                "prepare": 3,
                "commit": 3,
                "discard": 0,
                "error": 0,
                "cancel": cancel,
                "prefill": {"prepare": 1, "commit": 1, "discard": 0},
                "publication": {"prepare": 2, "commit": 2, "discard": 0},
                "reset": {"attempted": 1, "complete": reset_complete, "failed": 1 - reset_complete},
            },
            "operation_audit": {"coverage_complete": true},
        })
    }

    fn blocked_args(output: PathBuf) -> Args {
        Args {
            served_model_manifest: "served.json".into(),
            fixture: "fixture.json".into(),
            case_binding: "case.json".into(),
            identity_binding: "identity.json".into(),
            preflight: "preflight.json".into(),
            source_root: "source".into(),
            output,
            case_id: "case-0".into(),
            policy_id: "policy-0".into(),
            oracle_kind: "aq4_target".into(),
            requested_m: 1,
            device_index: 0,
            chunk_elements: DEFAULT_CHUNK_ELEMENTS,
        }
    }

    #[test]
    fn source_first_step_and_forced_multistep_hashes_are_strict() {
        let case = SourceCase {
            case_id: "case-0".into(),
            prompt_token_ids: vec![11, 12],
            step_count: 2,
            semantic_input_id: None,
            observation: None,
        };
        let first = source_row(
            "case-0",
            0,
            canonical_token_hash(&case.prompt_token_ids).unwrap(),
            7,
        );
        validate_source_row_identity(&first, &case, 0, None).unwrap();
        let second = source_row("case-0", 1, canonical_token_hash(&[7]).unwrap(), 8);
        validate_source_row_identity(&second, &case, 1, Some(7)).unwrap();
        assert!(validate_source_row_identity(&second, &case, 1, Some(9)).is_err());
    }

    #[test]
    fn source_hash_and_step_swap_are_rejected() {
        let case = SourceCase {
            case_id: "case-0".into(),
            prompt_token_ids: vec![11, 12],
            step_count: 2,
            semantic_input_id: None,
            observation: None,
        };
        let mut row = source_row("case-0", 1, canonical_token_hash(&[7]).unwrap(), 8);
        assert!(validate_source_row_identity(&row, &case, 0, None).is_err());
        row.step = 0;
        row.input_token_ids_sha256 = "f".repeat(64);
        assert!(validate_source_row_identity(&row, &case, 0, None).is_err());
    }

    #[test]
    fn execution_rows_record_equal_and_divergent_teacher_forcing() {
        let source = source_artifact(&[7, 8]);
        let first = execution_row(&source, 0, 7, 7, 1).unwrap();
        assert!(!first.diverged);
        let second = execution_row(&source, 1, 9, 8, 2).unwrap();
        assert!(second.diverged);
        assert_eq!(second.predicted_token_id, 9);
        assert_eq!(second.committed_token_id, 8);
        assert!(execution_row(&source, 1, 9, 7, 2).is_err());
        assert!(execution_row(&source, 2, 9, 8, 3).is_err());
    }

    #[test]
    fn direct_top1_values_are_rejected_by_policy() {
        assert!(flag_enabled(Some(OsString::from("1"))));
        assert!(flag_enabled(Some(OsString::from(" true "))));
        assert!(!flag_enabled(Some(OsString::from("0"))));
        assert!(!flag_enabled(None));
    }

    #[test]
    fn oom_identity_and_nonfinite_statuses_are_fail_closed() {
        assert_eq!(complete_capture_status(0), "available");
        assert_eq!(complete_capture_status(1), "blocked");
        assert_eq!(
            blocked_reason_code("HIP out of memory"),
            "runtime_out_of_memory"
        );
        assert_eq!(
            blocked_reason_code("source identity differs"),
            "identity_or_source_rejected"
        );
        assert_eq!(
            blocked_reason_code("direct-top1 is incompatible"),
            "direct_top1_full_logits_unavailable"
        );
    }

    #[test]
    fn blocked_oom_artifact_is_atomic_and_immutable() {
        let parent = temp_path("blocked");
        fs::create_dir(&parent).unwrap();
        let output = parent.join("artifact");
        let args = blocked_args(output.clone());
        publish_blocked(&args, "HIP out of memory").unwrap();
        let manifest = parse_strict_json(
            &read_regular(
                &output.join("manifest.json"),
                "blocked manifest",
                MAX_JSON_BYTES,
            )
            .unwrap(),
            "blocked manifest",
        )
        .unwrap();
        assert_eq!(manifest["status"], "oom");
        assert_eq!(manifest["capture_complete"], false);
        assert_eq!(manifest["immutable_status"], true);
        assert_eq!(manifest["performance"]["timing_eligible"], false);
        assert!(output.join("SHA256SUMS").is_file());
        assert!(publish_blocked(&args, "second failure").is_err());
        fs::remove_dir_all(parent).unwrap();
    }

    #[test]
    fn terminal_audit_rejects_cancel_and_incomplete_reset() {
        let case = p2_case();
        validate_terminal_audit(&terminal_audit(0, 1), &case).unwrap();
        assert!(validate_terminal_audit(&terminal_audit(1, 1), &case).is_err());
        assert!(validate_terminal_audit(&terminal_audit(0, 0), &case).is_err());
    }

    #[test]
    fn observer_rejects_short_extra_and_out_of_order_chunks() {
        let root = temp_path("observer-negative");
        fs::create_dir(&root).unwrap();
        let mut hidden = File::create(root.join("hidden")).unwrap();
        let mut logits = File::create(root.join("logits")).unwrap();
        let mut observer = CaptureObserver::new(&mut hidden, &mut logits).unwrap();
        observer.begin(HIDDEN_SIZE, VOCAB_SIZE).unwrap();
        assert!(observer.observe_hidden_chunk(1, &[0.0]).is_err());
        observer.observe_hidden_chunk(0, &[0.0]).unwrap();
        assert!(observer.observe_hidden_chunk(0, &[0.0]).is_err());
        assert!(observer.finish().is_err());
        drop(observer);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn observer_writes_f32le_hash_and_stable_top10() {
        let root = temp_path("observer-positive");
        fs::create_dir(&root).unwrap();
        let hidden_path = root.join("hidden");
        let logits_path = root.join("logits");
        let mut hidden = File::create(&hidden_path).unwrap();
        let mut logits = File::create(&logits_path).unwrap();
        let mut observer = CaptureObserver::new(&mut hidden, &mut logits).unwrap();
        observer.begin(HIDDEN_SIZE, VOCAB_SIZE).unwrap();
        observer
            .observe_hidden_chunk(0, &vec![0.25; HIDDEN_SIZE])
            .unwrap();
        let mut values = vec![0.0; VOCAB_SIZE];
        for (index, value) in values.iter_mut().take(TOP_K).enumerate() {
            *value = (TOP_K - index) as f32;
        }
        observer.observe_logit_chunk(0, &values).unwrap();
        observer.finish().unwrap();
        let source = source_row("case-0", 0, "a".repeat(64), 0);
        let row = observer.finish_row(&source).unwrap();
        drop(hidden);
        drop(logits);
        assert_eq!(row.greedy_token_id, 0);
        assert_eq!(row.topk[TOP_K - 1].token_id, TOP_K - 1);
        assert_eq!(
            row.hidden.sha256,
            sha256_file(&hidden_path, "hidden").unwrap()
        );
        assert_eq!(
            row.logits.sha256,
            sha256_file(&logits_path, "logits").unwrap()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn strict_json_rejects_duplicate_keys_and_nonfinite() {
        assert!(parse_strict_json(br#"{"a":1,"a":2}"#, "duplicate").is_err());
        assert!(parse_strict_json(br#"{"a":NaN}"#, "nonfinite").is_err());
    }

    #[test]
    fn atomic_publication_never_overwrites_existing_root() {
        let parent = temp_path("atomic");
        fs::create_dir(&parent).unwrap();
        let first = parent.join("first");
        let second = parent.join("second");
        let output = parent.join("output");
        fs::create_dir(&first).unwrap();
        fs::create_dir(&second).unwrap();
        rename_noreplace(&first, &output).unwrap();
        assert!(rename_noreplace(&second, &output).is_err());
        assert!(output.is_dir());
        assert!(second.is_dir());
        fs::remove_dir_all(parent).unwrap();
    }
}
