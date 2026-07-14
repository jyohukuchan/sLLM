//! Dedicated 24-row AQ4 fidelity producer.
//!
//! This binary is diagnostic-only and is never reached by the production worker.  It loads the
//! active package once, then runs the hash-bound full-context/step-zero fixtures with the row's
//! requested prefill width.  Final hidden and full-logit rows are streamed directly to F32LE
//! sidecars, so no sequence-by-vocabulary matrix is retained in host memory.

use serde::de::{self, Deserialize as DeDeserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

use ullm_engine::aq4_worker_backend::QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::execution_batch::ExecutionPhase;
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4CalibrationObserver, Qwen35Aq4ModelLoadConfig,
    Qwen35Aq4ModelRuntime,
};
use ullm_engine::qwen35_aq4_session::{QWEN35_AQ4_ROPE_BASE, QWEN35_AQ4_ROTARY_DIM};
use ullm_engine::served_model::load_served_model;

const SCHEMA: &str = "ullm.qwen35_aq4_target_calibration.v1";
const SOURCE_SCHEMA: &str = "ullm.qwen35_aq4_source_calibration.v1";
const SPLIT_SCHEMA: &str = "ullm.aq4_p2_fidelity_split.v1";
const HIDDEN_SIZE: usize = 4096;
const VOCAB_SIZE: usize = 248_320;
const TOP_K: usize = 10;
const F32_BYTES: usize = 4;
const MAX_ROWS: usize = 24;
const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;
const MAX_ROW_BYTES: usize = 64 * 1024;
const MAX_CHUNK_ELEMENTS: usize = 1_048_576;
const MAX_VECTOR_BYTES: u64 = (MAX_ROWS * (HIDDEN_SIZE + VOCAB_SIZE) * F32_BYTES) as u64;
const POLICY_SCHEMA: &str = "ullm.aq4_p2_fidelity_policy.v1";
const POLICY_METRICS: &[(&str, &str, &str, &str, &str)] = &[
    ("token_agreement_rate", "promotion", "higher", "wilson_lower_one_sided", "wilson_lower_one_sided(successes=sum(exact 1.0 rows), n=24, confidence_level=0.95); no mean-minus-margin"),
    ("topk_overlap_rate_k10", "promotion", "higher", "mean", "higher: max(absolute_floor, mean-max(absolute_margin, relative_margin*abs(mean))); lower: min(absolute_ceiling, mean+max(absolute_margin, relative_margin*abs(mean)))"),
    ("logits_cosine", "promotion", "higher", "mean", "higher: max(absolute_floor, mean-max(absolute_margin, relative_margin*abs(mean))); lower: min(absolute_ceiling, mean+max(absolute_margin, relative_margin*abs(mean)))"),
    ("logits_relative_l2", "promotion", "lower", "mean", "higher: max(absolute_floor, mean-max(absolute_margin, relative_margin*abs(mean))); lower: min(absolute_ceiling, mean+max(absolute_margin, relative_margin*abs(mean)))"),
    ("hidden_cosine", "promotion", "higher", "mean", "higher: max(absolute_floor, mean-max(absolute_margin, relative_margin*abs(mean))); lower: min(absolute_ceiling, mean+max(absolute_margin, relative_margin*abs(mean)))"),
    ("hidden_relative_l2", "promotion", "lower", "mean", "higher: max(absolute_floor, mean-max(absolute_margin, relative_margin*abs(mean))); lower: min(absolute_ceiling, mean+max(absolute_margin, relative_margin*abs(mean)))"),
    ("hidden_max_abs", "diagnostic_only", "diagnostic", "max", "diagnostic_max only; no promotion bound and no absolute ceiling"),
    ("bf16_top1_retained_in_aq4_top10_rate", "promotion", "higher", "wilson_lower_one_sided", "wilson_lower_one_sided(successes=sum(exact 1.0 rows), n=24, confidence_level=0.95); no mean-minus-margin"),
];

#[derive(Debug)]
struct Args {
    served_model_manifest: PathBuf,
    split_root: PathBuf,
    source_root: PathBuf,
    cases: PathBuf,
    output: PathBuf,
    device_index: u32,
    chunk_elements: usize,
    expected_split_manifest_sha256: String,
    expected_policy_sha256: String,
    expected_calibration_cases_sha256: String,
    expected_served_model_manifest_sha256: String,
    expected_package_manifest_sha256: String,
    expected_worker_binary_sha256: String,
    expected_guard_sha256: String,
    expected_device_architecture: String,
    expected_quantized_artifact_revision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitRow {
    case_id: String,
    case_sha256: String,
    fixture_sha256: String,
    fixture_path: String,
    prompt_token_ids_sha256: String,
    context_token_ids_sha256: String,
    prompt_tokens: usize,
    cached_prefix_tokens: usize,
    context_tokens: usize,
    generated_tokens: usize,
    baseline_mode: String,
    prefill_requested_m: usize,
    resolved_m: usize,
    step: usize,
    row_count: usize,
    subset: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceCases {
    schema_version: String,
    cases: Vec<SourceCase>,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Clone, Serialize)]
struct VectorRef {
    offset_bytes: u64,
    bytes: u64,
    elements: usize,
    dtype: &'static str,
    endianness: &'static str,
    sha256: String,
    nonfinite_count: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TopEntry {
    token_id: usize,
    logit: f32,
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug)]
struct SourceProvenance {
    model_revision: String,
    source_checkpoint: Value,
    tokenizer: Value,
}

struct StrictJson(Value);

impl<'de> DeDeserialize<'de> for StrictJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        struct StrictVisitor;
        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = Value;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { formatter.write_str("a JSON value without duplicate object keys") }
            fn visit_unit<E>(self) -> Result<Value, E> where E: de::Error { Ok(Value::Null) }
            fn visit_bool<E>(self, value: bool) -> Result<Value, E> where E: de::Error { Ok(Value::Bool(value)) }
            fn visit_i64<E>(self, value: i64) -> Result<Value, E> where E: de::Error { Ok(Value::Number(value.into())) }
            fn visit_u64<E>(self, value: u64) -> Result<Value, E> where E: de::Error { Ok(Value::Number(value.into())) }
            fn visit_f64<E>(self, value: f64) -> Result<Value, E> where E: de::Error { serde_json::Number::from_f64(value).map(Value::Number).ok_or_else(|| E::custom("non-finite JSON number")) }
            fn visit_str<E>(self, value: &str) -> Result<Value, E> where E: de::Error { Ok(Value::String(value.to_string())) }
            fn visit_string<E>(self, value: String) -> Result<Value, E> where E: de::Error { Ok(Value::String(value)) }
            fn visit_seq<A>(self, mut access: A) -> Result<Value, A::Error> where A: SeqAccess<'de> { let mut values = Vec::new(); while let Some(value) = access.next_element::<StrictJson>()? { values.push(value.0); } Ok(Value::Array(values)) }
            fn visit_map<A>(self, mut access: A) -> Result<Value, A::Error> where A: MapAccess<'de> { let mut object = serde_json::Map::new(); while let Some(key) = access.next_key::<String>()? { if object.contains_key(&key) { return Err(de::Error::custom(format!("duplicate JSON key: {key}"))); } let value = access.next_value::<StrictJson>()?; object.insert(key, value.0); } Ok(Value::Object(object)) }
        }
        deserializer.deserialize_any(StrictVisitor).map(StrictJson)
    }
}

struct CaptureObserver<'a> {
    hidden: &'a mut File,
    logits: &'a mut File,
    hidden_offset: u64,
    logits_offset: u64,
    expected_hidden: usize,
    expected_logits: usize,
    hidden_seen: usize,
    logits_seen: usize,
    hidden_hash: Sha256,
    logits_hash: Sha256,
    hidden_nonfinite: u64,
    logits_nonfinite: u64,
    topk: Vec<TopEntry>,
}

impl<'a> CaptureObserver<'a> {
    fn new(hidden: &'a mut File, logits: &'a mut File, chunk_elements: usize) -> Result<Self, String> {
        if chunk_elements == 0 || chunk_elements > MAX_CHUNK_ELEMENTS {
            return Err("chunk_elements is outside the bounded contract".into());
        }
        let hidden_offset = hidden.stream_position().map_err(|e| format!("hidden offset: {e}"))?;
        let logits_offset = logits.stream_position().map_err(|e| format!("logits offset: {e}"))?;
        Ok(Self { hidden, logits, hidden_offset, logits_offset, expected_hidden: HIDDEN_SIZE, expected_logits: VOCAB_SIZE, hidden_seen: 0, logits_seen: 0, hidden_hash: Sha256::new(), logits_hash: Sha256::new(), hidden_nonfinite: 0, logits_nonfinite: 0, topk: Vec::with_capacity(TOP_K), })
    }

    fn write_chunk(file: &mut File, digest: &mut Sha256, values: &[f32], label: &str) -> Result<(), String> {
        let mut bytes = Vec::with_capacity(values.len() * F32_BYTES);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&bytes).map_err(|e| format!("{label} write: {e}"))?;
        digest.update(bytes);
        Ok(())
    }

    fn update_topk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        for (offset, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err("non-finite logit observed".into());
            }
            self.topk.push(TopEntry { token_id: start + offset, logit: value });
            self.topk.sort_by(|left, right| right.logit.total_cmp(&left.logit).then_with(|| left.token_id.cmp(&right.token_id)));
            self.topk.truncate(TOP_K);
        }
        Ok(())
    }

    fn finish_row(self, case: &SplitRow, semantic: String, observation: String) -> Result<VectorRow, String> {
        if self.hidden_seen != self.expected_hidden || self.logits_seen != self.expected_logits || self.topk.len() != TOP_K {
            return Err("calibration observer row shape is incomplete".into());
        }
        let greedy = self.topk.first().ok_or("calibration observer top-k is empty")?.token_id;
        Ok(VectorRow { case_id: case.case_id.clone(), step: 0, semantic_input_id: semantic, observation, input_token_ids_sha256: case.context_token_ids_sha256.clone(), hidden: VectorRef { offset_bytes: self.hidden_offset, bytes: (HIDDEN_SIZE * F32_BYTES) as u64, elements: HIDDEN_SIZE, dtype: "f32", endianness: "little", sha256: format!("{:x}", self.hidden_hash.finalize()), nonfinite_count: self.hidden_nonfinite }, logits: VectorRef { offset_bytes: self.logits_offset, bytes: (VOCAB_SIZE * F32_BYTES) as u64, elements: VOCAB_SIZE, dtype: "f32", endianness: "little", sha256: format!("{:x}", self.logits_hash.finalize()), nonfinite_count: self.logits_nonfinite }, greedy_token_id: greedy, topk: self.topk, finite: self.hidden_nonfinite == 0 && self.logits_nonfinite == 0 })
    }
}

impl Qwen35Aq4CalibrationObserver for CaptureObserver<'_> {
    fn begin(&mut self, hidden_elements: usize, logit_elements: usize) -> Result<(), String> {
        if hidden_elements != self.expected_hidden || logit_elements != self.expected_logits { return Err("runtime calibration shape differs".into()); }
        Ok(())
    }
    fn observe_hidden_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        if start != self.hidden_seen || self.hidden_seen + values.len() > self.expected_hidden { return Err("hidden chunks are not contiguous".into()); }
        if values.iter().any(|value| !value.is_finite()) { return Err("non-finite hidden value observed".into()); }
        Self::write_chunk(self.hidden, &mut self.hidden_hash, values, "hidden")?;
        self.hidden_seen += values.len();
        Ok(())
    }
    fn observe_logit_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        if start != self.logits_seen || self.logits_seen + values.len() > self.expected_logits { return Err("logit chunks are not contiguous".into()); }
        if values.iter().any(|value| !value.is_finite()) { return Err("non-finite logit value observed".into()); }
        Self::write_chunk(self.logits, &mut self.logits_hash, values, "logits")?;
        self.update_topk(start, values)?;
        self.logits_seen += values.len();
        Ok(())
    }
    fn finish(&mut self) -> Result<(), String> { if self.hidden_seen != self.expected_hidden || self.logits_seen != self.expected_logits { return Err("runtime calibration observer ended short".into()); } Ok(()) }
}

fn sha_file(path: &Path, label: &str) -> Result<String, String> {
    let metadata = fs::symlink_metadata(path).map_err(|e| format!("{label} metadata: {e}"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.nlink() != 1 { return Err(format!("{label} must be a single-link regular file")); }
    let mut file = File::open(path).map_err(|e| format!("{label} open: {e}"))?;
    let mut digest = Sha256::new();
    let mut buf = [0_u8; 1024 * 1024];
    loop { let read = file.read(&mut buf).map_err(|e| format!("{label} read: {e}"))?; if read == 0 { break; } digest.update(&buf[..read]); }
    Ok(format!("{:x}", digest.finalize()))
}

fn read_json(path: &Path, label: &str) -> Result<Value, String> {
    let metadata = fs::symlink_metadata(path).map_err(|e| format!("{label} metadata: {e}"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > MAX_JSON_BYTES as u64 { return Err(format!("{label} is not a bounded regular file")); }
    let bytes = fs::read(path).map_err(|e| format!("{label} read: {e}"))?;
    parse_strict_json(&bytes, label)
}

fn parse_strict_json(bytes: &[u8], label: &str) -> Result<Value, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictJson::deserialize(&mut deserializer).map_err(|e| format!("{label} JSON: {e}"))?.0;
    deserializer.end().map_err(|e| format!("{label} trailing JSON: {e}"))?;
    Ok(value)
}

fn safe_source_name(name: &str, label: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(name);
    if path.is_absolute() || path.components().any(|component| matches!(component, Component::CurDir | Component::ParentDir | Component::RootDir | Component::Prefix(_))) {
        return Err(format!("{label} is not a safe relative path"));
    }
    Ok(path)
}

fn source_file_identity(root: &Path, name: &str, label: &str) -> Result<Value, String> {
    let relative = safe_source_name(name, label)?;
    let path = root.join(&relative);
    let metadata = fs::symlink_metadata(&path).map_err(|e| format!("{label} metadata: {e}"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.nlink() != 1 {
        return Err(format!("{label} must be a single-link regular file"));
    }
    Ok(json!({"bytes": metadata.len(), "file": name, "sha256": sha_file(&path, label)?}))
}

fn aggregate_source_files(root: &Path, names: &[String], label: &str) -> Result<(Vec<Value>, String), String> {
    let mut ordered = names.to_vec();
    ordered.sort();
    ordered.dedup();
    if ordered.is_empty() {
        return Err(format!("{label} file set is empty"));
    }
    let entries = ordered.iter().map(|name| source_file_identity(root, name, &format!("{label} {name}"))).collect::<Result<Vec<_>, _>>()?;
    let canonical = serde_json::to_vec(&entries).map_err(|e| format!("{label} canonical identity: {e}"))?;
    let mut bytes = canonical;
    bytes.push(b'\n');
    Ok((entries, format!("{:x}", Sha256::digest(bytes))))
}

fn validate_policy(policy: &Value) -> Result<(), String> {
    if policy.get("schema_version").and_then(Value::as_str) != Some(POLICY_SCHEMA) || policy.get("status").and_then(Value::as_str) != Some("formula_frozen_unbound") || policy.get("promotion_eligible").and_then(Value::as_bool) != Some(false) { return Err("fidelity policy schema/status differs".into()); }
    for field in ["attempt2_threshold_source_forbidden", "observed_attempt2_values_forbidden", "calibration_subset_only_for_active_bf16_envelope", "holdout_evaluation_allowed_once"] { if policy.get(field).and_then(Value::as_bool) != Some(true) { return Err(format!("fidelity policy safety flag differs: {field}")); } }
    let metrics = policy.get("metrics").and_then(Value::as_object).ok_or("fidelity policy metrics are missing")?;
    if metrics.len() != POLICY_METRICS.len() { return Err("fidelity policy metric set differs".into()); }
    for (name, role, direction, aggregation, formula) in POLICY_METRICS {
        let item = metrics.get(*name).and_then(Value::as_object).ok_or_else(|| format!("fidelity policy metric missing: {name}"))?;
        if item.get("role").and_then(Value::as_str) != Some(*role) || item.get("direction").and_then(Value::as_str) != Some(*direction) || item.get("aggregation").and_then(Value::as_str) != Some(*aggregation) || item.get("formula").and_then(Value::as_str) != Some(*formula) || item.get("sample_minimum").and_then(Value::as_u64) != Some(MAX_ROWS as u64) { return Err(format!("fidelity policy metric contract differs: {name}")); }
    }
    if policy.get("quality_task").and_then(|value| value.get("kind")).and_then(Value::as_str) != Some("binary_retention_rate") || policy.get("quality_task").and_then(|value| value.get("score")).and_then(Value::as_str) != Some("bf16_top1_retained_in_aq4_top10") { return Err("fidelity policy quality task differs".into()); }
    if policy.get("relative_l2_rejection").and_then(|value| value.get("ceiling")).and_then(Value::as_f64) != Some(1.0) || policy.get("relative_l2_rejection").and_then(|value| value.get("action")).and_then(Value::as_str) != Some("reject any observed relative-L2 > 1 before aggregation") { return Err("fidelity policy relative-L2 rejection differs".into()); }
    Ok(())
}

fn source_provenance(package_manifest: &Value) -> Result<SourceProvenance, String> {
    let raw_root = package_manifest.get("source_model_dir").and_then(Value::as_str).ok_or("package source_model_dir is missing")?;
    let root = PathBuf::from(raw_root);
    let root_meta = fs::symlink_metadata(&root).map_err(|e| format!("package source model metadata: {e}"))?;
    if !root_meta.is_dir() || root_meta.file_type().is_symlink() {
        return Err("package source model root must be a real directory".into());
    }
    let config = read_json(&root.join("config.json"), "source config")?;
    if config.get("model_type").and_then(Value::as_str) != Some("qwen3_5") {
        return Err("package source config is not Qwen3.5".into());
    }
    let dtype = config.get("text_config").and_then(|value| value.get("dtype")).and_then(Value::as_str).ok_or("source config dtype is missing")?;
    let index = read_json(&root.join("model.safetensors.index.json"), "source weight index")?;
    let weight_map = index.get("weight_map").and_then(Value::as_object).ok_or("source weight index has no weight_map")?;
    let mut names = vec!["config.json".to_string(), "model.safetensors.index.json".to_string()];
    names.extend(weight_map.values().map(|value| value.as_str().ok_or("source weight index shard is invalid").map(str::to_string)).collect::<Result<Vec<_>, _>>()?);
    let (checkpoint_files, checkpoint_sha) = aggregate_source_files(&root, &names, "source checkpoint")?;
    let tokenizer_names = ["chat_template.jinja", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"].into_iter().map(str::to_string).collect::<Vec<_>>();
    let (tokenizer_files, tokenizer_sha) = aggregate_source_files(&root, &tokenizer_names, "source tokenizer")?;
    let metadata_dir = root.join(".cache/huggingface/download");
    let mut revisions = BTreeSet::new();
    for entry in fs::read_dir(&metadata_dir).map_err(|e| format!("source revision metadata: {e}"))? {
        let path = entry.map_err(|e| format!("source revision metadata entry: {e}"))?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("metadata") { continue; }
        let metadata = fs::read_to_string(&path).map_err(|e| format!("source revision metadata read: {e}"))?;
        if let Some(first) = metadata.lines().next().filter(|line| !line.is_empty()) { revisions.insert(first.to_string()); }
    }
    if revisions.len() != 1 { return Err("source model revision metadata is missing or ambiguous".into()); }
    let model_revision = revisions.into_iter().next().ok_or("source model revision is missing")?;
    Ok(SourceProvenance {
        model_revision,
        source_checkpoint: json!({"aggregate_sha256": checkpoint_sha, "dtype": dtype, "files": checkpoint_files, "root": root.canonicalize().map_err(|e| format!("source root: {e}"))?.to_string_lossy()}),
        tokenizer: json!({"aggregate_sha256": tokenizer_sha, "files": tokenizer_files, "root": root.canonicalize().map_err(|e| format!("source root: {e}"))?.to_string_lossy()}),
    })
}

fn valid_sha(value: &str) -> bool { value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) }

fn canonical_token_hash(tokens: &[usize], newline: bool) -> Result<String, String> {
    let mut bytes = serde_json::to_vec(tokens).map_err(|e| format!("token hash: {e}"))?;
    if newline { bytes.push(b'\n'); }
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn load_args() -> Result<Args, String> {
    let mut values = env::args_os().skip(1);
    let mut map = BTreeMap::<String, String>::new();
    while let Some(raw) = values.next() {
        let name = raw.to_string_lossy().into_owned();
        if name == "--help" { return Err("help".into()); }
        let value = values.next().ok_or_else(|| format!("{name} requires a value"))?.to_string_lossy().into_owned();
        if map.insert(name.clone(), value).is_some() { return Err(format!("duplicate argument {name}")); }
    }
    let take = |name: &str, map: &mut BTreeMap<String, String>| map.remove(name).ok_or_else(|| format!("{name} is required"));
    let args = Args {
        served_model_manifest: PathBuf::from(take("--served-model-manifest", &mut map)?),
        split_root: PathBuf::from(take("--split-root", &mut map)?),
        source_root: PathBuf::from(take("--source", &mut map)?),
        cases: PathBuf::from(take("--cases", &mut map)?),
        output: PathBuf::from(take("--output", &mut map)?),
        device_index: map.remove("--device-index").unwrap_or_else(|| "0".into()).parse().map_err(|_| "device-index is invalid")?,
        chunk_elements: map.remove("--chunk-elements").unwrap_or_else(|| "65536".into()).parse().map_err(|_| "chunk-elements is invalid")?,
        expected_split_manifest_sha256: take("--expected-split-manifest-sha256", &mut map)?,
        expected_policy_sha256: take("--expected-policy-sha256", &mut map)?,
        expected_calibration_cases_sha256: take("--expected-calibration-cases-sha256", &mut map)?,
        expected_served_model_manifest_sha256: take("--expected-served-model-manifest-sha256", &mut map)?,
        expected_package_manifest_sha256: take("--expected-package-manifest-sha256", &mut map)?,
        expected_worker_binary_sha256: take("--expected-worker-binary-sha256", &mut map)?,
        expected_guard_sha256: take("--expected-guard-sha256", &mut map)?,
        expected_device_architecture: take("--expected-device-architecture", &mut map)?,
        expected_quantized_artifact_revision: take("--expected-quantized-artifact-revision", &mut map)?,
    };
    for (label, value) in [("expected split manifest SHA", &args.expected_split_manifest_sha256), ("expected policy SHA", &args.expected_policy_sha256), ("expected calibration cases SHA", &args.expected_calibration_cases_sha256), ("expected served manifest SHA", &args.expected_served_model_manifest_sha256), ("expected package manifest SHA", &args.expected_package_manifest_sha256), ("expected worker SHA", &args.expected_worker_binary_sha256), ("expected guard SHA", &args.expected_guard_sha256)] {
        if !valid_sha(value) { return Err(format!("{label} is invalid")); }
    }
    if args.expected_device_architecture.is_empty() || args.expected_quantized_artifact_revision.is_empty() { return Err("expected device/revision binding is empty".into()); }
    if !map.is_empty() || args.chunk_elements == 0 || args.chunk_elements > MAX_CHUNK_ELEMENTS { return Err("unknown arguments or bounded option violation".into()); }
    Ok(args)
}

fn load_split(root: &Path, expected_split_sha: &str, expected_policy_sha: &str, expected_cases_sha: &str) -> Result<(Value, Vec<SplitRow>, String, String, String), String> {
    let manifest_path = root.join("split-manifest.json");
    let policy_path = root.join("policy.json");
    let cases_path = root.join("calibration-cases.jsonl");
    let manifest = read_json(&manifest_path, "split manifest")?;
    let policy = read_json(&policy_path, "fidelity policy")?;
    if manifest.get("schema_version").and_then(Value::as_str) != Some(SPLIT_SCHEMA) || manifest.get("status").and_then(Value::as_str) != Some("ready_for_calibration") { return Err("split manifest schema/status differs".into()); }
    validate_policy(&policy)?;
    let split_sha = sha_file(&manifest_path, "split manifest")?;
    let policy_sha = sha_file(&policy_path, "policy")?;
    let cases_sha = sha_file(&cases_path, "calibration cases")?;
    if split_sha != expected_split_sha || policy_sha != expected_policy_sha || cases_sha != expected_cases_sha {
        return Err("split/policy/calibration SHA does not match the pinned execution contract".into());
    }
    let sums = fs::read_to_string(root.join("SHA256SUMS")).map_err(|e| format!("split SHA256SUMS: {e}"))?;
    let mut declared = BTreeMap::new();
    for line in sums.lines() {
        let (digest, name) = line.split_once("  ").ok_or("split SHA256SUMS line is invalid")?;
        if !valid_sha(digest) || declared.insert(name.to_string(), digest.to_string()).is_some() { return Err("split SHA256SUMS contains an invalid or duplicate entry".into()); }
    }
    let expected_names = ["calibration-cases.jsonl", "holdout-cases.jsonl", "policy.json", "split-manifest.json"];
    if declared.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_names.into_iter().collect::<BTreeSet<_>>() { return Err("split SHA256SUMS file set differs".into()); }
    for name in expected_names { let expected = declared.get(name).ok_or("split SHA256SUMS entry is missing")?; if sha_file(&root.join(name), name)? != *expected { return Err(format!("split SHA256SUMS digest differs for {name}")); } }
    if manifest.get("calibration_sha256").and_then(Value::as_str) != Some(cases_sha.as_str()) || manifest.get("policy_sha256").and_then(Value::as_str) != Some(policy_sha.as_str()) { return Err("split manifest file binding differs".into()); }
    let file = File::open(&cases_path).map_err(|e| format!("calibration cases open: {e}"))?;
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("calibration row {line_no}: {e}"))?;
        if line.is_empty() || line.len() > MAX_ROW_BYTES { return Err("calibration row exceeds bounded size".into()); }
        let row: SplitRow = serde_json::from_value(parse_strict_json(line.as_bytes(), &format!("calibration row {line_no}"))?).map_err(|e| format!("calibration row {line_no}: {e}"))?;
        if !seen.insert(row.case_id.clone()) { return Err("calibration rows contain duplicate case_id".into()); }
        if row.subset != "calibration" || row.step != 0 || row.row_count != 1 || row.cached_prefix_tokens != 0 || row.generated_tokens != 0 || row.prompt_tokens != row.context_tokens || !valid_sha(&row.case_sha256) || !valid_sha(&row.fixture_sha256) || !valid_sha(&row.prompt_token_ids_sha256) || !valid_sha(&row.context_token_ids_sha256) { return Err(format!("calibration row {} violates identity/step contract", row.case_id)); }
        rows.push(row);
    }
    if rows.len() != MAX_ROWS { return Err(format!("calibration cases must contain exactly {MAX_ROWS} rows")); }
    Ok((manifest, rows, split_sha, policy_sha, cases_sha))
}

fn fixture_tokens(split_root: &Path, row: &SplitRow) -> Result<Vec<usize>, String> {
    let path = { let raw = PathBuf::from(&row.fixture_path); if raw.is_absolute() { raw } else { split_root.join(raw) } };
    if sha_file(&path, "fixture")? != row.fixture_sha256 { return Err(format!("fixture hash differs: {}", row.case_id)); }
    let value = read_json(&path, "fixture")?;
    if let Some(ids) = value.get("prompt_token_ids").and_then(Value::as_array) { return ids.iter().map(|v| v.as_u64().ok_or_else(|| "fixture token is invalid".into()).map(|v| v as usize)).collect(); }
    let cases = value.get("cases").and_then(Value::as_array).ok_or("fixture has no prompt token IDs")?;
    let item = cases.iter().find(|v| v.get("case_id").and_then(Value::as_str) == Some(row.case_id.as_str())).ok_or("fixture case is missing")?;
    item.get("prompt_token_ids").and_then(Value::as_array).ok_or("fixture prompt token IDs are missing")?.iter().map(|v| v.as_u64().ok_or_else(|| "fixture token is invalid".into()).map(|v| v as usize)).collect()
}

fn package_tree_hash(root: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).map_err(|e| format!("package directory: {e}"))? {
            let entry = entry.map_err(|e| format!("package entry: {e}"))?;
            let path = entry.path(); let meta = fs::symlink_metadata(&path).map_err(|e| format!("package metadata: {e}"))?;
            if meta.file_type().is_symlink() { return Err("package tree contains symlink".into()); }
            if meta.is_dir() { pending.push(path); } else if meta.is_file() { files.push(path); } else { return Err("package tree contains non-regular entry".into()); }
        }
    }
    files.sort_by_key(|path| path.strip_prefix(root).map(|relative| relative.to_string_lossy().into_owned()).unwrap_or_default());
    let mut digest = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(root).map_err(|_| "package path escapes root")?.to_string_lossy();
        let file_sha = sha_file(&path, "package file")?;
        let mut raw_sha = [0_u8; 32];
        for (index, pair) in file_sha.as_bytes().chunks_exact(2).enumerate() {
            raw_sha[index] = (pair[0] as char).to_digit(16).ok_or("package file hash is invalid")? as u8 * 16 + (pair[1] as char).to_digit(16).ok_or("package file hash is invalid")? as u8;
        }
        digest.update(relative.as_bytes());
        digest.update([0]);
        digest.update(raw_sha);
        digest.update(b"\n");
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(target_os = "linux")]
fn rename_noreplace(source: &Path, destination: &Path) -> Result<(), String> {
    const AT_FDCWD: i32 = -100;
    const RENAME_NOREPLACE: u32 = 1;
    unsafe extern "C" {
        fn renameat2(olddirfd: i32, oldpath: *const std::ffi::c_char, newdirfd: i32, newpath: *const std::ffi::c_char, flags: u32) -> i32;
    }
    let source = CString::new(source.as_os_str().as_bytes()).map_err(|_| "temporary root path contains NUL".to_string())?;
    let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| "output root path contains NUL".to_string())?;
    // SAFETY: both C strings remain alive for the call and point to NUL-terminated path bytes.
    let result = unsafe { renameat2(AT_FDCWD, source.as_ptr(), AT_FDCWD, destination.as_ptr(), RENAME_NOREPLACE) };
    if result != 0 { return Err(format!("atomic non-overwrite publication failed: {}", std::io::Error::last_os_error())); }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn rename_noreplace(_source: &Path, _destination: &Path) -> Result<(), String> {
    Err("atomic non-overwrite directory publication requires Linux renameat2".into())
}

fn run(args: Args) -> Result<(), String> {
    if fs::symlink_metadata(&args.output).is_ok() { return Err("output root already exists; overwrite is forbidden".into()); }
    let (split_manifest, split_rows, split_sha, policy_sha, cases_sha) = load_split(&args.split_root, &args.expected_split_manifest_sha256, &args.expected_policy_sha256, &args.expected_calibration_cases_sha256)?;
    let source_manifest_path = args.source_root.join("manifest.json");
    let source_manifest = read_json(&source_manifest_path, "source manifest")?;
    if source_manifest.get("schema_version").and_then(Value::as_str) != Some(SOURCE_SCHEMA) { return Err("source manifest schema differs".into()); }
    let source_identity = source_manifest.get("identity").cloned().ok_or("source identity is missing")?;
    let source_manifest_sha = sha_file(&source_manifest_path, "source manifest")?;
    let cases_bytes = fs::read(&args.cases).map_err(|e| format!("source cases read: {e}"))?;
    let source_cases: SourceCases = serde_json::from_value(parse_strict_json(&cases_bytes, "source cases")?).map_err(|e| format!("source cases JSON: {e}"))?;
    if source_cases.schema_version != "ullm.qwen35_aq4_source_calibration_cases.v1" || source_cases.cases.len() != MAX_ROWS || sha_file(&args.cases, "source cases")? != source_manifest.get("cases").and_then(|v| v.get("sha256")).and_then(Value::as_str).unwrap_or("") { return Err("source cases identity differs".into()); }
    let source_by_id = source_cases.cases.iter().map(|case| (case.case_id.as_str(), case)).collect::<BTreeMap<_, _>>();
    let model = load_served_model(&args.served_model_manifest).map_err(|e| format!("served model: {e}"))?;
    if model.format.format_id != "AQ4_0" || model.format.implementation_id != "qwen35_aq4_rdna4_v1" || model.worker.identity.device != "gfx1201" || model.worker.identity.execution_profile != "rdna4_aq4_resident" { return Err("served model is not the AQ4 RDNA4 active identity".into()); }
    let actual_guards = model.worker.required_environment.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let expected_guards = QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV.iter().copied().collect::<BTreeSet<_>>();
    if actual_guards != expected_guards { return Err("served required-environment guard set differs".into()); }
    for name in &model.worker.required_environment { if env::var(name).ok().as_deref() != Some("1") { return Err(format!("required guard {name} is not set to 1")); } }
    let manifest_sha = sha_file(&args.served_model_manifest, "served model manifest")?;
    let worker_sha = sha_file(&model.worker.binary, "served worker")?;
    if manifest_sha != model.manifest_sha256 || manifest_sha != args.expected_served_model_manifest_sha256 || worker_sha != model.worker.binary_sha256 || worker_sha != args.expected_worker_binary_sha256 { return Err("active served identity hash differs from the pinned execution contract".into()); }
    if model.public.revision != args.expected_quantized_artifact_revision { return Err("quantized artifact revision differs from the pinned execution contract".into()); }
    let package_manifest = model.product.root.join(&model.product.package.manifest_path);
    let package_manifest_sha = sha_file(&package_manifest, "package manifest")?;
    if package_manifest_sha != model.product.package.manifest_sha256 || package_manifest_sha != args.expected_package_manifest_sha256 { return Err("package manifest hash differs from the pinned execution contract".into()); }
    let package_manifest_value = read_json(&package_manifest, "package manifest")?;
    let provenance = source_provenance(&package_manifest_value)?;
    if provenance.model_revision == model.public.revision { return Err("upstream and quantized artifact revisions must remain distinct".into()); }
    let source_model_revision = source_identity.get("model_revision").and_then(Value::as_str).ok_or("source model revision is missing")?;
    if source_model_revision != provenance.model_revision || source_identity.get("source_checkpoint").and_then(|value| value.get("aggregate_sha256")).and_then(Value::as_str) != provenance.source_checkpoint.get("aggregate_sha256").and_then(Value::as_str) || source_identity.get("tokenizer").and_then(|value| value.get("aggregate_sha256")).and_then(Value::as_str) != provenance.tokenizer.get("aggregate_sha256").and_then(Value::as_str) { return Err("source identity does not match package-bound upstream provenance".into()); }
    let package_root = package_manifest.parent().ok_or("package manifest has no parent")?;
    let package_content_sha = package_tree_hash(package_root)?;
    let device = ullm_runtime_sys::device_info(args.device_index).map_err(|e| format!("device query: {e}"))?;
    if device.gcn_arch_name != args.expected_device_architecture || device.gcn_arch_name != model.worker.identity.device { return Err("runtime device differs from the pinned active identity".into()); }
    let capture_binary = env::current_exe().map_err(|e| format!("capture binary: {e}"))?;
    let capture_sha = sha_file(&capture_binary, "capture binary")?;
    let mut guard_digest = Sha256::new();
    guard_digest.update(b"ullm-aq4-p2-resident-guards-v1\0");
    for name in QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV.iter().copied().collect::<BTreeSet<_>>() { guard_digest.update(format!("{name}=1\n").as_bytes()); }
    let guard_sha = format!("{:x}", guard_digest.finalize());
    if guard_sha != args.expected_guard_sha256 { return Err("required guard SHA differs from the pinned execution contract".into()); }
    if source_identity.get("model_id").and_then(Value::as_str) != Some(model.public.upstream_id.as_str()) { return Err("source/active model ID differs".into()); }
    fs::create_dir_all(args.output.parent().ok_or("output parent missing")?).map_err(|e| format!("output parent: {e}"))?;
    let parent = json!({"path": source_manifest_path.canonicalize().map_err(|e| format!("source path: {e}"))?.to_string_lossy(), "manifest_sha256": source_manifest_sha, "schema_version": SOURCE_SCHEMA});
    let temporary = args.output.with_file_name(format!(".{}.incomplete-{}", args.output.file_name().ok_or("output has no name")?.to_string_lossy(), std::process::id()));
    if fs::symlink_metadata(&temporary).is_ok() { return Err("incomplete output already exists".into()); }
    fs::create_dir(&temporary).map_err(|e| format!("output temporary root: {e}"))?;
    fs::create_dir(temporary.join("vectors")).map_err(|e| format!("output vectors directory: {e}"))?;
    let mut hidden = OpenOptions::new().write(true).create_new(true).open(temporary.join("vectors/hidden.f32le")).map_err(|e| format!("hidden output: {e}"))?;
    let mut logits = OpenOptions::new().write(true).create_new(true).open(temporary.join("vectors/logits.f32le")).map_err(|e| format!("logits output: {e}"))?;
    let mut rows_file = OpenOptions::new().write(true).create_new(true).open(temporary.join("rows.jsonl")).map_err(|e| format!("rows output: {e}"))?;
    let profile = model.profile_snapshot();
    let model_config = Qwen35Aq4ModelLoadConfig { package_dir: package_root.to_path_buf(), device_index: args.device_index, expected_architecture: Some(profile.device.clone()), chunk_bytes: 1024 * 1024, context_length: profile.context_length, kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE, layer_indices: None, lm_head_mode: PackageLmHeadMode::GpuResidentF32, lm_head_chunk_rows: 8192 };
    let mut runtime = Qwen35Aq4ModelRuntime::load(model_config)?;
    if !runtime.calibration_full_logits_top1_available() { return Err("active LM head has no full-logit calibration path".into()); }
    let mut output_rows = 0usize;
    let mut finite_rows = 0usize;
    for row in &split_rows {
        let source_case = source_by_id.get(row.case_id.as_str()).ok_or_else(|| format!("source case missing: {}", row.case_id))?;
        if source_case.step_count != 1 { return Err("fidelity source cases must be one step".into()); }
        let prompt = fixture_tokens(&args.split_root, row)?;
        if prompt.len() != row.prompt_tokens || canonical_token_hash(&prompt, false)? != row.prompt_token_ids_sha256 || canonical_token_hash(&prompt, true)? != row.context_token_ids_sha256 || source_case.prompt_token_ids != prompt { return Err(format!("fixture/case token identity differs: {}", row.case_id)); }
        let requested = row.prefill_requested_m;
        if !matches!(requested, 1 | 8 | 16 | 32 | 64 | 128) { return Err(format!("unsupported prefill M for {}", row.case_id)); }
        // `all_m1` is the same-artifact M=1 baseline even when its case label
        // retains the requested-M value for the paired comparison.  The
        // cold-batched rows execute their requested width.
        let effective_m = if row.baseline_mode == "all_m1" {
            if row.resolved_m != 1 { return Err(format!("all_m1 resolved M differs for {}", row.case_id)); }
            1
        } else if row.baseline_mode == "cold_batched" {
            if row.resolved_m != requested { return Err(format!("cold_batched resolved M differs for {}", row.case_id)); }
            requested
        } else {
            return Err(format!("unknown baseline mode for {}", row.case_id));
        };
        let mut offset = 0usize;
        while offset < prompt.len() { let width = effective_m.min(prompt.len() - offset); let label = format!("aq4-fidelity-{}-{offset}", row.case_id); if width == 1 { runtime.dispatch_token_for_phase(prompt[offset], QWEN35_AQ4_ROTARY_DIM, QWEN35_AQ4_ROPE_BASE, offset, offset, ExecutionPhase::ColdPrefill, false, &label)?; } else { runtime.dispatch_prefill_chunk_for_phase(&prompt[offset..offset + width], QWEN35_AQ4_ROTARY_DIM, QWEN35_AQ4_ROPE_BASE, offset, ExecutionPhase::ColdPrefill, false, &label)?; } offset += width; }
        runtime.synchronize()?;
        let top = runtime.top_logits_from_last_layer(TOP_K, &format!("aq4-fidelity-top-{}", row.case_id))?;
        let epoch = runtime.last_generation_state_epoch().ok_or("active generation epoch is missing")?;
        let mut observer = CaptureObserver::new(&mut hidden, &mut logits, args.chunk_elements)?;
        runtime.visit_last_generation_state(epoch, &mut observer)?;
        let vector_row = observer.finish_row(row, source_case.semantic_input_id.clone().unwrap_or_else(|| row.case_id.clone()), source_case.observation.clone().unwrap_or_else(|| "fidelity_full_context_step0".into()))?;
        if !vector_row.finite { return Err(format!("non-finite active vector row: {}", row.case_id)); }
        finite_rows += 1;
        if top.first().map(|item| item.token_id) != Some(vector_row.greedy_token_id) { return Err(format!("active top-1 differs from full-logit row: {}", row.case_id)); }
        serde_json::to_writer(&mut rows_file, &vector_row).map_err(|e| format!("row JSON: {e}"))?; rows_file.write_all(b"\n").map_err(|e| format!("row newline: {e}"))?; output_rows += 1;
        runtime.reset_all_request_state_synchronized()?;
    }
    if output_rows != MAX_ROWS { return Err("active fidelity output row count differs".into()); }
    hidden.sync_all().map_err(|e| format!("hidden sync: {e}"))?; logits.sync_all().map_err(|e| format!("logits sync: {e}"))?; rows_file.sync_all().map_err(|e| format!("rows sync: {e}"))?;
    if fs::metadata(temporary.join("vectors/hidden.f32le")).map_err(|e| format!("hidden metadata: {e}"))?.len() != (MAX_ROWS * HIDDEN_SIZE * F32_BYTES) as u64 || fs::metadata(temporary.join("vectors/logits.f32le")).map_err(|e| format!("logits metadata: {e}"))?.len() != (MAX_ROWS * VOCAB_SIZE * F32_BYTES) as u64 { return Err("vector sidecar size differs from the 24-row bound".into()); }
    drop(hidden); drop(logits); drop(rows_file);
    let identity = json!({"artifact": {"package_manifest_sha256": Value::Null, "artifact_manifest_sha256": Value::Null}, "model_id": model.public.upstream_id, "model_revision": provenance.model_revision, "source_checkpoint": provenance.source_checkpoint, "tokenizer": provenance.tokenizer, "hidden_size": HIDDEN_SIZE, "vocab_size": VOCAB_SIZE, "package_content_sha256": package_content_sha, "package_manifest_sha256": package_manifest_sha, "worker_binary_sha256": worker_sha});
    let nonfinite_rows = output_rows.checked_sub(finite_rows).ok_or("finite row count exceeds output rows")?;
    let run = json!({"row_count": output_rows, "nonfinite_rows": nonfinite_rows, "elapsed_seconds": 0.0});
    let manifest = json!({"schema_version": SCHEMA, "oracle_kind": "aq4_target", "status": "available", "evidence_class": "production", "usable_as_source_evidence": false, "promotion_eligible": false, "created_utc": source_manifest.get("created_utc").cloned().unwrap_or_else(|| Value::String("2026-01-01T00:00:00Z".into())), "identity": identity, "parent_sampled_oracle": parent, "vector_contract": {"hidden_shape": [HIDDEN_SIZE], "logits_shape": [VOCAB_SIZE], "dtype": "f32", "endianness": "little", "layout": "flat", "chunk_elements": args.chunk_elements, "row_bytes": (HIDDEN_SIZE + VOCAB_SIZE) * F32_BYTES, "semantic_hidden": "post_final_rmsnorm_hidden_used_by_lm_head", "semantic_logits": "raw_pre_softmax_lm_head_logits"}, "limits": {"max_case_file_bytes": MAX_JSON_BYTES, "max_cases": MAX_ROWS, "max_rows": MAX_ROWS, "max_steps": 1}, "cases": {"path": args.cases.canonicalize().map_err(|e| format!("cases path: {e}"))?.to_string_lossy(), "sha256": sha_file(&args.cases, "source cases")?, "case_count": MAX_ROWS, "row_count": MAX_ROWS}, "files": {"rows": "rows.jsonl", "hidden": "vectors/hidden.f32le", "logits": "vectors/logits.f32le"}, "runtime": {"runtime": {"name": "ullm-aq4-fidelity-capture", "build_sha256": capture_sha, "one_model_load": true, "split_manifest_sha256": split_sha, "policy_sha256": policy_sha, "calibration_cases_sha256": cases_sha, "served_model_manifest_sha256": manifest_sha, "package_manifest_sha256": package_manifest_sha, "worker_binary_sha256": worker_sha, "guard_sha256": guard_sha, "upstream_model_revision": provenance.model_revision, "quantized_artifact_revision": model.public.revision, "source_checkpoint_aggregate_sha256": provenance.source_checkpoint.get("aggregate_sha256").and_then(Value::as_str).unwrap_or(""), "tokenizer_aggregate_sha256": provenance.tokenizer.get("aggregate_sha256").and_then(Value::as_str).unwrap_or(""), "device": {"requested_index": args.device_index, "device_id": device.device_id, "backend": device.backend, "name": device.name, "architecture": device.gcn_arch_name}}, "transformers": Value::Null, "torch": Value::Null, "safetensors": Value::Null, "python": Value::Null, "device": "gpu", "dtype": "f32", "low_cpu_mem_usage": false, "torch_num_threads": 1, "torch_num_interop_threads": 1, "model_loads": 1, "inference_mode": true, "full_vocab_ranking": true, "max_resident_logit_rows": 1, "memory_preflight": {"checkpoint_bytes": 0, "mem_total_bytes": Value::Null, "mem_available_bytes": Value::Null, "required_headroom_bytes": 0, "headroom_factor": 1.0, "status": "streaming"}, "disk_preflight": {"expected_vector_bytes": MAX_VECTOR_BYTES, "required_free_bytes": MAX_VECTOR_BYTES, "free_bytes": MAX_VECTOR_BYTES, "status": "bounded_streaming"}, "run": run}, "legacy_cross_check": {"status": "not_applicable", "legacy_manifest_sha256": source_manifest_sha, "legacy_payload_sha256": source_manifest.get("payload").and_then(|v| v.get("sha256")).and_then(Value::as_str).unwrap_or(""), "row_count": MAX_ROWS, "hidden_sample_max_abs_diff": 0.0, "logit_sample_max_abs_diff": 0.0}});
    fs::write(temporary.join("manifest.json"), serde_json::to_vec_pretty(&manifest).map_err(|e| format!("manifest JSON: {e}"))?).map_err(|e| format!("manifest write: {e}"))?;
    let files = ["manifest.json", "rows.jsonl", "vectors/hidden.f32le", "vectors/logits.f32le"];
    let mut sums = String::new(); for file in files { sums.push_str(&format!("{}  {file}\n", sha_file(&temporary.join(file), file)?)); }
    fs::write(temporary.join("SHA256SUMS"), sums).map_err(|e| format!("SHA256SUMS: {e}"))?;
    rename_noreplace(&temporary, &args.output)?;
    let _ = split_manifest;
    Ok(())
}

fn main() -> ExitCode {
    let args = match load_args() { Ok(args) => args, Err(error) if error == "help" => { eprintln!("usage: ullm-aq4-fidelity-capture --served-model-manifest PATH --split-root DIR --source DIR --cases FILE --output DIR [--device-index N] [--chunk-elements N]"); return ExitCode::SUCCESS; }, Err(error) => { eprintln!("AQ4 fidelity capture arguments rejected: {error}"); return ExitCode::from(2); } };
    match run(args) { Ok(()) => ExitCode::SUCCESS, Err(error) => { eprintln!("AQ4 fidelity capture failed: {error}"); ExitCode::from(1) } }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(label: &str) -> PathBuf {
        env::temp_dir().join(format!("ullm-aq4-fidelity-{label}-{}", std::process::id()))
    }

    #[test]
    fn observer_streams_bounded_rows_and_orders_top10() {
        let root = temp("observer");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut hidden = File::create(root.join("hidden")).unwrap();
        let mut logits = File::create(root.join("logits")).unwrap();
        let mut observer = CaptureObserver::new(&mut hidden, &mut logits, 8).unwrap();
        observer.begin(HIDDEN_SIZE, VOCAB_SIZE).unwrap();
        observer.observe_hidden_chunk(0, &[1.0; HIDDEN_SIZE]).unwrap();
        observer.observe_logit_chunk(0, &[0.0; VOCAB_SIZE]).unwrap();
        let row = SplitRow {
            case_id: "case".into(), case_sha256: "a".repeat(64), fixture_sha256: "b".repeat(64), fixture_path: "fixture".into(),
            prompt_token_ids_sha256: "c".repeat(64), context_token_ids_sha256: "d".repeat(64), prompt_tokens: 1,
            cached_prefix_tokens: 0, context_tokens: 1, generated_tokens: 0, baseline_mode: "all_m1".into(),
            prefill_requested_m: 1, resolved_m: 1, step: 0, row_count: 1, subset: "calibration".into(),
        };
        let result = observer.finish_row(&row, "case".into(), "test".into()).unwrap();
        assert_eq!(result.hidden.elements, HIDDEN_SIZE);
        assert_eq!(result.logits.elements, VOCAB_SIZE);
        assert_eq!(result.topk.len(), TOP_K);
        assert_eq!(result.greedy_token_id, 0);
        drop(hidden);
        drop(logits);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn observer_rejects_nonfinite_hidden_and_logits() {
        let root = temp("nonfinite");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut hidden = File::create(root.join("hidden")).unwrap();
        let mut logits = File::create(root.join("logits")).unwrap();
        let mut observer = CaptureObserver::new(&mut hidden, &mut logits, 8).unwrap();
        observer.begin(HIDDEN_SIZE, VOCAB_SIZE).unwrap();
        assert!(observer.observe_hidden_chunk(0, &[f32::NAN]).is_err());
        assert!(observer.observe_logit_chunk(0, &[f32::INFINITY]).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn token_hash_separates_prompt_and_context_bindings() {
        let prompt = canonical_token_hash(&[11, 12, 13], false).unwrap();
        let context = canonical_token_hash(&[11, 12, 13], true).unwrap();
        assert_ne!(prompt, context);
        assert!(valid_sha(&prompt));
        assert!(valid_sha(&context));
    }

    #[test]
    fn strict_json_rejects_duplicate_keys() {
        assert!(parse_strict_json(br#"{"a":1,"a":2}"#, "duplicate").is_err());
    }

    #[test]
    fn package_tree_hash_matches_canonical_cross_language_contract() {
        let root = temp("package-hash");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("dir")).unwrap();
        fs::write(root.join("a.txt"), b"alpha").unwrap();
        fs::write(root.join("dir/b.bin"), [0_u8, 1]).unwrap();
        assert_eq!(package_tree_hash(&root).unwrap(), "0440739e282bc7be23704973be9428815c4e05924b3e66dfd5216e6c3e46913f");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_provenance_rejects_ambiguous_revision_metadata() {
        let root = temp("provenance");
        let _ = fs::remove_dir_all(&root);
        for relative in [".cache/huggingface/download", ""] { fs::create_dir_all(root.join(relative)).unwrap(); }
        fs::write(root.join("config.json"), br#"{"model_type":"qwen3_5","text_config":{"dtype":"bfloat16"}}"#).unwrap();
        fs::write(root.join("model.safetensors.index.json"), br#"{"weight_map":{"weight":"model.safetensors-00001-of-00001.safetensors"}}"#).unwrap();
        fs::write(root.join("model.safetensors-00001-of-00001.safetensors"), b"weights").unwrap();
        for name in ["chat_template.jinja", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"] { fs::write(root.join(name), name.as_bytes()).unwrap(); }
        fs::write(root.join(".cache/huggingface/download/a.metadata"), b"revision-a\n").unwrap();
        fs::write(root.join(".cache/huggingface/download/b.metadata"), b"revision-b\n").unwrap();
        let manifest = json!({"source_model_dir": root});
        assert!(source_provenance(&manifest).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn noreplace_publication_rejects_existing_and_dangling_symlink() {
        let root = temp("noreplace");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source");
        fs::create_dir(&source).unwrap();
        let destination = root.join("destination");
        fs::create_dir(&destination).unwrap();
        assert!(rename_noreplace(&source, &destination).is_err());
        let dangling = root.join("dangling");
        std::os::unix::fs::symlink(root.join("missing"), &dangling).unwrap();
        let source2 = root.join("source2");
        fs::create_dir(&source2).unwrap();
        assert!(rename_noreplace(&source2, &dangling).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
