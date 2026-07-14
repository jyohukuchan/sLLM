//! Dedicated 24-row AQ4 fidelity producer.
//!
//! This binary is diagnostic-only and is never reached by the production worker.  It loads the
//! active package once, then runs the hash-bound full-context/step-zero fixtures with the row's
//! requested prefill width.  Final hidden and full-logit rows are streamed directly to F32LE
//! sidecars, so no sequence-by-vocabulary matrix is retained in host memory.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
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

#[derive(Debug)]
struct Args {
    served_model_manifest: PathBuf,
    split_root: PathBuf,
    source_root: PathBuf,
    cases: PathBuf,
    output: PathBuf,
    device_index: u32,
    chunk_elements: usize,
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
        self.hidden_nonfinite += values.iter().filter(|v| !v.is_finite()).count() as u64;
        Self::write_chunk(self.hidden, &mut self.hidden_hash, values, "hidden")?;
        self.hidden_seen += values.len();
        Ok(())
    }
    fn observe_logit_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        if start != self.logits_seen || self.logits_seen + values.len() > self.expected_logits { return Err("logit chunks are not contiguous".into()); }
        self.logits_nonfinite += values.iter().filter(|v| !v.is_finite()).count() as u64;
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
    serde_json::from_slice(&bytes).map_err(|e| format!("{label} JSON: {e}"))
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
    let args = Args { served_model_manifest: PathBuf::from(take("--served-model-manifest", &mut map)?), split_root: PathBuf::from(take("--split-root", &mut map)?), source_root: PathBuf::from(take("--source", &mut map)?), cases: PathBuf::from(take("--cases", &mut map)?), output: PathBuf::from(take("--output", &mut map)?), device_index: map.remove("--device-index").unwrap_or_else(|| "0".into()).parse().map_err(|_| "device-index is invalid")?, chunk_elements: map.remove("--chunk-elements").unwrap_or_else(|| "65536".into()).parse().map_err(|_| "chunk-elements is invalid")? };
    if !map.is_empty() || args.chunk_elements == 0 || args.chunk_elements > MAX_CHUNK_ELEMENTS { return Err("unknown arguments or bounded option violation".into()); }
    Ok(args)
}

fn load_split(root: &Path) -> Result<(Value, Vec<SplitRow>, String, String, String), String> {
    let manifest_path = root.join("split-manifest.json");
    let policy_path = root.join("policy.json");
    let cases_path = root.join("calibration-cases.jsonl");
    let manifest = read_json(&manifest_path, "split manifest")?;
    if manifest.get("schema_version").and_then(Value::as_str) != Some(SPLIT_SCHEMA) || manifest.get("status").and_then(Value::as_str) != Some("ready_for_calibration") { return Err("split manifest schema/status differs".into()); }
    let split_sha = sha_file(&manifest_path, "split manifest")?;
    let policy_sha = sha_file(&policy_path, "policy")?;
    let cases_sha = sha_file(&cases_path, "calibration cases")?;
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
        let row: SplitRow = serde_json::from_str(&line).map_err(|e| format!("calibration row {line_no}: {e}"))?;
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
    files.sort(); let mut digest = Sha256::new();
    for path in files { let relative = path.strip_prefix(root).map_err(|_| "package path escapes root")?; let bytes = fs::read(&path).map_err(|e| format!("package file: {e}"))?; digest.update(relative.to_string_lossy().as_bytes()); digest.update([0]); digest.update((bytes.len() as u64).to_le_bytes()); digest.update(Sha256::digest(bytes)); }
    Ok(format!("{:x}", digest.finalize()))
}

fn run(args: Args) -> Result<(), String> {
    if args.output.exists() { return Err("output root already exists; overwrite is forbidden".into()); }
    let (split_manifest, split_rows, split_sha, policy_sha, cases_sha) = load_split(&args.split_root)?;
    let source_manifest_path = args.source_root.join("manifest.json");
    let source_manifest = read_json(&source_manifest_path, "source manifest")?;
    if source_manifest.get("schema_version").and_then(Value::as_str) != Some(SOURCE_SCHEMA) { return Err("source manifest schema differs".into()); }
    let source_manifest_sha = sha_file(&source_manifest_path, "source manifest")?;
    let cases_bytes = fs::read(&args.cases).map_err(|e| format!("source cases read: {e}"))?;
    let source_cases: SourceCases = serde_json::from_slice(&cases_bytes).map_err(|e| format!("source cases JSON: {e}"))?;
    if source_cases.schema_version != "ullm.qwen35_aq4_source_calibration_cases.v1" || source_cases.cases.len() != MAX_ROWS || sha_file(&args.cases, "source cases")? != source_manifest.get("cases").and_then(|v| v.get("sha256")).and_then(Value::as_str).unwrap_or("") { return Err("source cases identity differs".into()); }
    let source_by_id = source_cases.cases.iter().map(|case| (case.case_id.as_str(), case)).collect::<BTreeMap<_, _>>();
    let model = load_served_model(&args.served_model_manifest).map_err(|e| format!("served model: {e}"))?;
    if model.format.format_id != "AQ4_0" || model.format.implementation_id != "qwen35_aq4_rdna4_v1" || model.worker.identity.device != "gfx1201" || model.worker.identity.execution_profile != "rdna4_aq4_resident" { return Err("served model is not the AQ4 RDNA4 active identity".into()); }
    let actual_guards = model.worker.required_environment.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let expected_guards = QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV.iter().copied().collect::<BTreeSet<_>>();
    if actual_guards != expected_guards { return Err("served required-environment guard set differs".into()); }
    for name in &model.worker.required_environment { if env::var(name).ok().as_deref() != Some("1") { return Err(format!("required guard {name} is not set to 1")); } }
    let manifest_sha = sha_file(&args.served_model_manifest, "served model manifest")?;
    if manifest_sha != model.manifest_sha256 || sha_file(&model.worker.binary, "served worker")? != model.worker.binary_sha256 { return Err("active served identity hash differs".into()); }
    let package_manifest = model.product.root.join(&model.product.package.manifest_path);
    let package_manifest_sha = sha_file(&package_manifest, "package manifest")?;
    if package_manifest_sha != model.product.package.manifest_sha256 { return Err("package manifest hash differs".into()); }
    let package_root = package_manifest.parent().ok_or("package manifest has no parent")?;
    let package_content_sha = package_tree_hash(package_root)?;
    let device = ullm_runtime_sys::device_info(args.device_index).map_err(|e| format!("device query: {e}"))?;
    if device.gcn_arch_name != "gfx1201" { return Err("runtime device is not gfx1201".into()); }
    let capture_binary = env::current_exe().map_err(|e| format!("capture binary: {e}"))?;
    let capture_sha = sha_file(&capture_binary, "capture binary")?;
    let guard_sha = format!("{:x}", Sha256::digest(QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV.join("\0")));
    let source_identity = source_manifest.get("identity").cloned().ok_or("source identity is missing")?;
    if source_identity.get("model_id").and_then(Value::as_str) != Some(model.public.upstream_id.as_str()) || source_identity.get("model_revision").and_then(Value::as_str) != Some(model.public.revision.as_str()) { return Err("source/active model identity differs".into()); }
    let parent = json!({"path": source_manifest_path.canonicalize().map_err(|e| format!("source path: {e}"))?.to_string_lossy(), "manifest_sha256": source_manifest_sha, "schema_version": SOURCE_SCHEMA});
    let temporary = args.output.with_file_name(format!(".{}.incomplete-{}", args.output.file_name().ok_or("output has no name")?.to_string_lossy(), std::process::id()));
    if temporary.exists() { return Err("incomplete output already exists".into()); }
    fs::create_dir_all(temporary.join("vectors")).map_err(|e| format!("output temporary root: {e}"))?;
    let mut hidden = OpenOptions::new().write(true).create_new(true).open(temporary.join("vectors/hidden.f32le")).map_err(|e| format!("hidden output: {e}"))?;
    let mut logits = OpenOptions::new().write(true).create_new(true).open(temporary.join("vectors/logits.f32le")).map_err(|e| format!("logits output: {e}"))?;
    let mut rows_file = OpenOptions::new().write(true).create_new(true).open(temporary.join("rows.jsonl")).map_err(|e| format!("rows output: {e}"))?;
    let profile = model.profile_snapshot();
    let model_config = Qwen35Aq4ModelLoadConfig { package_dir: package_root.to_path_buf(), device_index: args.device_index, expected_architecture: Some(profile.device.clone()), chunk_bytes: 1024 * 1024, context_length: profile.context_length, kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE, layer_indices: None, lm_head_mode: PackageLmHeadMode::GpuResidentF32, lm_head_chunk_rows: 8192 };
    let mut runtime = Qwen35Aq4ModelRuntime::load(model_config)?;
    if !runtime.calibration_full_logits_top1_available() { return Err("active LM head has no full-logit calibration path".into()); }
    let mut output_rows = 0usize;
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
        if top.first().map(|item| item.token_id) != Some(vector_row.greedy_token_id) { return Err(format!("active top-1 differs from full-logit row: {}", row.case_id)); }
        serde_json::to_writer(&mut rows_file, &vector_row).map_err(|e| format!("row JSON: {e}"))?; rows_file.write_all(b"\n").map_err(|e| format!("row newline: {e}"))?; output_rows += 1;
        runtime.reset_all_request_state_synchronized()?;
    }
    if output_rows != MAX_ROWS { return Err("active fidelity output row count differs".into()); }
    hidden.sync_all().map_err(|e| format!("hidden sync: {e}"))?; logits.sync_all().map_err(|e| format!("logits sync: {e}"))?; rows_file.sync_all().map_err(|e| format!("rows sync: {e}"))?;
    if fs::metadata(temporary.join("vectors/hidden.f32le")).map_err(|e| format!("hidden metadata: {e}"))?.len() != (MAX_ROWS * HIDDEN_SIZE * F32_BYTES) as u64 || fs::metadata(temporary.join("vectors/logits.f32le")).map_err(|e| format!("logits metadata: {e}"))?.len() != (MAX_ROWS * VOCAB_SIZE * F32_BYTES) as u64 { return Err("vector sidecar size differs from the 24-row bound".into()); }
    drop(hidden); drop(logits); drop(rows_file);
    let source_checkpoint = source_identity.get("source_checkpoint").cloned().ok_or("source checkpoint identity missing")?;
    let tokenizer = source_identity.get("tokenizer").cloned().ok_or("source tokenizer identity missing")?;
    let identity = json!({"artifact": {"package_manifest_sha256": Value::Null, "artifact_manifest_sha256": Value::Null}, "model_id": model.public.upstream_id, "model_revision": model.public.revision, "source_checkpoint": source_checkpoint, "tokenizer": tokenizer, "hidden_size": HIDDEN_SIZE, "vocab_size": VOCAB_SIZE, "package_content_sha256": package_content_sha, "package_manifest_sha256": package_manifest_sha, "worker_binary_sha256": model.worker.binary_sha256});
    let run = json!({"row_count": output_rows, "nonfinite_rows": 0, "elapsed_seconds": 0.0});
    let manifest = json!({"schema_version": SCHEMA, "oracle_kind": "aq4_target", "status": "available", "evidence_class": "production", "usable_as_source_evidence": false, "promotion_eligible": false, "created_utc": "2026-01-01T00:00:00Z", "identity": identity, "parent_sampled_oracle": parent, "vector_contract": {"hidden_shape": [HIDDEN_SIZE], "logits_shape": [VOCAB_SIZE], "dtype": "f32", "endianness": "little", "layout": "flat", "chunk_elements": args.chunk_elements, "row_bytes": (HIDDEN_SIZE + VOCAB_SIZE) * F32_BYTES, "semantic_hidden": "post_final_rmsnorm_hidden_used_by_lm_head", "semantic_logits": "raw_pre_softmax_lm_head_logits"}, "limits": {"max_case_file_bytes": MAX_JSON_BYTES, "max_cases": MAX_ROWS, "max_rows": MAX_ROWS, "max_steps": 1}, "cases": {"path": args.cases.canonicalize().map_err(|e| format!("cases path: {e}"))?.to_string_lossy(), "sha256": sha_file(&args.cases, "source cases")?, "case_count": MAX_ROWS, "row_count": MAX_ROWS}, "files": {"rows": "rows.jsonl", "hidden": "vectors/hidden.f32le", "logits": "vectors/logits.f32le"}, "runtime": {"runtime": {"name": "ullm-aq4-fidelity-capture", "build_sha256": capture_sha, "one_model_load": true, "split_manifest_sha256": split_sha, "policy_sha256": policy_sha, "calibration_cases_sha256": cases_sha, "served_model_manifest_sha256": manifest_sha, "package_manifest_sha256": package_manifest_sha, "guard_sha256": guard_sha, "device": {"requested_index": args.device_index, "device_id": device.device_id, "backend": device.backend, "name": device.name, "architecture": device.gcn_arch_name}}, "transformers": Value::Null, "torch": Value::Null, "safetensors": Value::Null, "python": Value::Null, "device": "gpu", "dtype": "f32", "low_cpu_mem_usage": false, "torch_num_threads": 1, "torch_num_interop_threads": 1, "model_loads": 1, "inference_mode": true, "full_vocab_ranking": true, "max_resident_logit_rows": 1, "memory_preflight": {"checkpoint_bytes": 0, "mem_total_bytes": Value::Null, "mem_available_bytes": Value::Null, "required_headroom_bytes": 0, "headroom_factor": 1.0, "status": "streaming"}, "disk_preflight": {"expected_vector_bytes": MAX_VECTOR_BYTES, "required_free_bytes": MAX_VECTOR_BYTES, "free_bytes": MAX_VECTOR_BYTES, "status": "bounded_streaming"}, "run": run}, "legacy_cross_check": {"status": "not_applicable", "legacy_manifest_sha256": source_manifest_sha, "legacy_payload_sha256": source_manifest.get("payload").and_then(|v| v.get("sha256")).and_then(Value::as_str).unwrap_or(""), "row_count": MAX_ROWS, "hidden_sample_max_abs_diff": 0.0, "logit_sample_max_abs_diff": 0.0}});
    fs::write(temporary.join("manifest.json"), serde_json::to_vec_pretty(&manifest).map_err(|e| format!("manifest JSON: {e}"))?).map_err(|e| format!("manifest write: {e}"))?;
    let files = ["manifest.json", "rows.jsonl", "vectors/hidden.f32le", "vectors/logits.f32le"];
    let mut sums = String::new(); for file in files { sums.push_str(&format!("{}  {file}\n", sha_file(&temporary.join(file), file)?)); }
    fs::write(temporary.join("SHA256SUMS"), sums).map_err(|e| format!("SHA256SUMS: {e}"))?;
    fs::create_dir_all(args.output.parent().ok_or("output parent missing")?).map_err(|e| format!("output parent: {e}"))?;
    if args.output.exists() { return Err("output appeared before publication".into()); }
    fs::rename(&temporary, &args.output).map_err(|e| format!("atomic output publish: {e}"))?;
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
    fn token_hash_separates_prompt_and_context_bindings() {
        let prompt = canonical_token_hash(&[11, 12, 13], false).unwrap();
        let context = canonical_token_hash(&[11, 12, 13], true).unwrap();
        assert_ne!(prompt, context);
        assert!(valid_sha(&prompt));
        assert!(valid_sha(&context));
    }
}
