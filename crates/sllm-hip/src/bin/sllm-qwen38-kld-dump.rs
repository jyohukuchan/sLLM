//! Dump teacher-forced, full-vocabulary Qwen3.8-27B logits for KLD studies.
//!
//! This is an evidence adapter rather than a generation path.  A resident
//! Unsloth Qwen3.8 NVFP4 model is loaded once, while each manifest case gets a
//! fresh request.  The first token is executed as a one-row prefill and each
//! subsequent token is executed as a one-row decode, so row `i` is the model
//! distribution conditioned on `token_ids[..=i]`.  The output is deliberately
//! raw little-endian FP32, one row per requested position.

use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, KvCacheEncoding, QWEN35_VOCAB_SIZE, QwenResidentModel,
    UNSLOTH_QWEN38_NVFP4_MODEL_SHA256, UNSLOTH_QWEN38_NVFP4_REPOSITORY,
    UNSLOTH_QWEN38_NVFP4_REVISION, build_qwen35_unsloth_qwen38_nvfp4_graph,
    build_qwen38_nvfp4_weight_load_plan, read_model_lock, verify_unsloth_qwen38_nvfp4,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(600);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);
const SCHEMA_VERSION: &str = "sllm-qwen38-kld-logit-dump-v1";
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_CASES: usize = 256;
const MAX_CASE_TOKENS: usize = 262_144;
const SERIAL_EXECUTION_MODE: &str = "prefill_1_then_serial_decode_with_last_logits";
const BLOCK_EXECUTION_MODE: &str = "prefill_1_then_decode_block_with_mtp_state_and_logits_resolve";
const SERIAL_LOGIT_STORAGE_PRECISION: &str = "FP32 last_logits API";
const BLOCK_LOGIT_STORAGE_PRECISION: &str =
    "initial FP32 last_logits; subsequent BF16 widened to FP32";
const SEMANTIC_LOCK: &str = "docs/models/locks/qwen3.5-27b-bf16.json";
const SEMANTIC_LOCK_SCOPE: &str = "Qwen3.5 reviewed graph contract only; Unsloth Qwen3.8 NVFP4 artifact identity is verified separately";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    cases: Vec<ManifestCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestCase {
    id: String,
    token_ids: Vec<i32>,
    /// If omitted, all positions are emitted.  Otherwise positions are
    /// zero-based input-token rows and must be unique and in range.
    #[serde(default)]
    positions: Option<Vec<usize>>,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    id: String,
    token_ids: Vec<i32>,
    token_ids_sha256: String,
    all_positions: bool,
    positions: Vec<usize>,
    rows: usize,
    vocab_size: usize,
    logits_dtype: &'static str,
    logits_file: String,
    logits_file_sha256: String,
    logits_file_bytes: u64,
    submission_count: u64,
    kernel_dispatch_count: u64,
    all_dispatches_hip: bool,
    fallback_used: bool,
    actual_chunk_sizes: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct OutputManifestCase {
    id: String,
    logits_file: String,
    shape: [usize; 2],
    positions: Vec<usize>,
    input_token_ids_sha256: String,
    nonfinite_count: u64,
    actual_chunk_sizes: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct OutputManifest {
    engine: &'static str,
    model: &'static str,
    model_fingerprint: String,
    kv: &'static str,
    chunk_size: usize,
    execution_mode: &'static str,
    logit_storage_precision: &'static str,
    vocab_size: usize,
    position_semantics: &'static str,
    cases: Vec<OutputManifestCase>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    model_repository: &'static str,
    model_revision: &'static str,
    model_fingerprint: String,
    recipe_digest: String,
    semantic_lock: &'static str,
    semantic_lock_scope: &'static str,
    target: String,
    device_index: u32,
    kv_cache_encoding: &'static str,
    chunk_size: usize,
    execution_mode: &'static str,
    logit_storage_precision: &'static str,
    manifest: String,
    manifest_sha256: String,
    output_dir: String,
    output_manifest: String,
    cases: Vec<CaseReport>,
}

struct Args {
    model_root: PathBuf,
    manifest: PathBuf,
    output_dir: PathBuf,
    target: String,
    device_index: u32,
    kv: KvCacheEncoding,
    chunk_size: usize,
}

fn usage() -> &'static str {
    "usage: sllm-qwen38-kld-dump --model-root DIR --manifest FILE --output-dir DIR --target gfx1030|gfx1201 --device-index N --kv fp16|kv-mxfp8-e4|kv-mxfp8-e5 [--chunk-size N]"
}

fn parse_args(arguments: &[String]) -> Result<Args, String> {
    let mut model_root = None;
    let mut manifest = None;
    let mut output_dir = None;
    let mut target = None;
    let mut device_index = None;
    let mut kv = None;
    let mut chunk_size = 1_usize;
    let mut index = 0;
    while index < arguments.len() {
        let value = arguments[index].as_str();
        let next = |index: &mut usize, name: &str| -> Result<String, String> {
            *index += 1;
            arguments
                .get(*index)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match value {
            "--model-root" => model_root = Some(PathBuf::from(next(&mut index, value)?)),
            "--manifest" => manifest = Some(PathBuf::from(next(&mut index, value)?)),
            "--output-dir" => output_dir = Some(PathBuf::from(next(&mut index, value)?)),
            "--target" => target = Some(next(&mut index, value)?),
            "--device-index" => {
                device_index = Some(
                    next(&mut index, value)?
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                )
            }
            "--kv" => {
                let name = next(&mut index, value)?;
                kv = Some(parse_kv(&name)?);
            }
            "--chunk-size" => {
                chunk_size = next(&mut index, value)?
                    .parse::<usize>()
                    .map_err(|_| "--chunk-size must be a positive usize".to_owned())?;
                if chunk_size == 0 {
                    return Err("--chunk-size must be positive".to_owned());
                }
                if chunk_size > MAX_CASE_TOKENS {
                    return Err(format!("--chunk-size must be <= {MAX_CASE_TOKENS}"));
                }
            }
            "--help" | "-h" => return Err(usage().to_owned()),
            other => return Err(format!("unknown argument {other}; {}", usage())),
        }
        index += 1;
    }
    let model_root = model_root.ok_or_else(|| format!("--model-root is required; {}", usage()))?;
    let manifest = manifest.ok_or_else(|| format!("--manifest is required; {}", usage()))?;
    let output_dir = output_dir.ok_or_else(|| format!("--output-dir is required; {}", usage()))?;
    let target = target.ok_or_else(|| format!("--target is required; {}", usage()))?;
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("--target must be gfx1030 or gfx1201".to_owned());
    }
    let device_index =
        device_index.ok_or_else(|| format!("--device-index is required; {}", usage()))?;
    if device_index != 0 {
        return Err("Qwen3.8 direct artifact requires logical device index 0".to_owned());
    }
    let kv = kv.ok_or_else(|| format!("--kv is required; {}", usage()))?;
    Ok(Args {
        model_root,
        manifest,
        output_dir,
        target,
        device_index,
        kv,
        chunk_size,
    })
}

fn parse_kv(name: &str) -> Result<KvCacheEncoding, String> {
    match name {
        "fp16" => Ok(KvCacheEncoding::Fp16),
        "kv-mxfp8-e4" | "mxfp8-e4" => Ok(KvCacheEncoding::Mxfp8E4),
        "kv-mxfp8-e5" | "mxfp8-e5" => Ok(KvCacheEncoding::Mxfp8E5),
        _ => Err(format!(
            "unsupported --kv {name}; expected fp16, kv-mxfp8-e4, or kv-mxfp8-e5"
        )),
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn token_ids_sha256(tokens: &[i32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn validate_manifest(bytes: &[u8]) -> Result<Manifest, String> {
    if bytes.is_empty() || bytes.len() > MAX_MANIFEST_BYTES {
        return Err(format!(
            "manifest bytes must be in 1..={MAX_MANIFEST_BYTES}"
        ));
    }
    let manifest: Manifest = serde_json::from_slice(bytes)
        .map_err(|error| format!("manifest JSON is invalid: {error}"))?;
    if manifest.schema_version != "qwen38-kld-manifest-v1" {
        return Err(format!(
            "unsupported manifest schema {}; expected qwen38-kld-manifest-v1",
            manifest.schema_version
        ));
    }
    if manifest.cases.is_empty() || manifest.cases.len() > MAX_CASES {
        return Err(format!("manifest cases must be in 1..={MAX_CASES}"));
    }
    let mut ids = BTreeSet::new();
    for case in &manifest.cases {
        if case.id.is_empty()
            || case.id == "."
            || case.id == ".."
            || case.id.contains('/')
            || case.id.contains('\\')
            || !ids.insert(case.id.clone())
        {
            return Err("manifest case ids must be non-empty, path-safe, and unique".to_owned());
        }
        if case.token_ids.is_empty() || case.token_ids.len() > MAX_CASE_TOKENS {
            return Err(format!(
                "case {} token_ids must be in 1..={MAX_CASE_TOKENS}",
                case.id
            ));
        }
        if case.token_ids.iter().any(|token| {
            *token < 0 || usize::try_from(*token).map_or(true, |token| token >= QWEN35_VOCAB_SIZE)
        }) {
            return Err(format!(
                "case {} has a token outside the model vocabulary",
                case.id
            ));
        }
        if let Some(positions) = &case.positions {
            let mut seen = BTreeSet::new();
            for position in positions {
                if *position >= case.token_ids.len() || !seen.insert(*position) {
                    return Err(format!(
                        "case {} positions must be unique and in 0..{}",
                        case.id,
                        case.token_ids.len()
                    ));
                }
            }
            if positions.is_empty() {
                return Err(format!("case {} positions cannot be empty", case.id));
            }
        }
    }
    Ok(manifest)
}

fn selected_positions(case: &ManifestCase) -> Vec<usize> {
    let mut positions = case
        .positions
        .clone()
        .unwrap_or_else(|| (0..case.token_ids.len()).collect());
    // Rows are emitted in teacher-forced traversal order.  Canonicalizing
    // here makes the raw row order unambiguous even when a caller listed
    // selected positions in a different order.
    positions.sort_unstable();
    positions
}

fn validate_audit(audit: &sllm_core::QwenExecutionAudit, target: &str) -> Result<(), String> {
    if audit.selected_backend() != "hip"
        || audit.target() != target
        || audit.fallback_used()
        || !audit.all_dispatches_hip()
        || audit.kernel_dispatch_count() == 0
    {
        return Err(format!(
            "execution was not exact HIP/no-fallback: {audit:?}"
        ));
    }
    Ok(())
}

fn write_fp32_row(writer: &mut BufWriter<File>, row: &[f32]) -> Result<(), String> {
    if row.len() != QWEN35_VOCAB_SIZE || row.iter().any(|value| !value.is_finite()) {
        return Err("logit row is non-finite or has the wrong vocabulary width".to_owned());
    }
    for value in row {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| format!("write FP32 logits: {error}"))?;
    }
    Ok(())
}

fn write_bf16_row(writer: &mut BufWriter<File>, row: &[u16]) -> Result<(), String> {
    if row.len() != QWEN35_VOCAB_SIZE {
        return Err("BF16 logit row has the wrong vocabulary width".to_owned());
    }
    for bits in row {
        let value = f32::from_bits(u32::from(*bits) << 16);
        if !value.is_finite() {
            return Err("BF16 logit row contains a non-finite value".to_owned());
        }
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| format!("write widened BF16 logits: {error}"))?;
    }
    Ok(())
}

fn write_case_logits(
    request: &mut sllm_core::QwenExecutionRequest,
    case: &ManifestCase,
    output_path: &Path,
    target: &str,
    chunk_size: usize,
) -> Result<CaseReport, String> {
    let positions = selected_positions(case);
    let wanted: BTreeSet<usize> = positions.iter().copied().collect();
    let file = File::create(output_path)
        .map_err(|error| format!("create {}: {error}", output_path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut actual_chunk_sizes = vec![1_usize];
    let first = request
        .prefill_with_last_logits(&[case.token_ids[0]])
        .map_err(|error| format!("case {} prefill row 0: {error}", case.id))?;
    if wanted.contains(&0) {
        let logits = first.last_logits().ok_or_else(|| {
            format!(
                "case {} row 0 did not publish full-vocabulary logits",
                case.id
            )
        })?;
        write_fp32_row(&mut writer, logits)
            .map_err(|error| format!("case {} row 0: {error}", case.id))?;
    }
    if chunk_size == 1 {
        for (position, token) in case.token_ids.iter().copied().enumerate().skip(1) {
            actual_chunk_sizes.push(1);
            let output = request
                .decode_with_last_logits(token)
                .map_err(|error| format!("case {} decode row {position}: {error}", case.id))?;
            if wanted.contains(&position) {
                let logits = output.last_logits().ok_or_else(|| {
                    format!(
                        "case {} row {position} omitted full-vocabulary logits",
                        case.id
                    )
                })?;
                write_fp32_row(&mut writer, logits)
                    .map_err(|error| format!("case {} row {position}: {error}", case.id))?;
            }
        }
    } else {
        for (chunk_index, chunk) in case.token_ids[1..].chunks(chunk_size).enumerate() {
            actual_chunk_sizes.push(chunk.len());
            let start_position = 1 + chunk_index * chunk_size;
            let output = request
                .decode_block_with_mtp_state_and_logits(chunk)
                .map_err(|error| {
                    format!(
                        "case {} decode block at row {start_position}: {error}",
                        case.id
                    )
                })?;
            let logits = output.logits_bf16().ok_or_else(|| {
                format!(
                    "case {} decode block at row {start_position} omitted BF16 logits",
                    case.id
                )
            })?;
            let expected_words = chunk
                .len()
                .checked_mul(QWEN35_VOCAB_SIZE)
                .ok_or_else(|| "decode block logit word count overflowed".to_owned())?;
            if logits.len() != expected_words {
                return Err(format!(
                    "case {} decode block at row {start_position} has {} words, expected {expected_words}",
                    case.id,
                    logits.len()
                ));
            }
            for local in 0..chunk.len() {
                let position = start_position + local;
                if wanted.contains(&position) {
                    let begin = local * QWEN35_VOCAB_SIZE;
                    write_bf16_row(&mut writer, &logits[begin..begin + QWEN35_VOCAB_SIZE])
                        .map_err(|error| format!("case {} row {position}: {error}", case.id))?;
                }
            }
            request.resolve_decode_block(chunk.len()).map_err(|error| {
                format!(
                    "case {} resolve decode block at row {start_position}: {error}",
                    case.id
                )
            })?;
        }
    }
    writer
        .flush()
        .map_err(|error| format!("flush {}: {error}", output_path.display()))?;
    let audit = request
        .audit_snapshot()
        .map_err(|error| format!("case {} audit: {error}", case.id))?;
    validate_audit(&audit, target).map_err(|error| format!("case {} audit: {error}", case.id))?;
    let bytes = output_path
        .metadata()
        .map_err(|error| format!("stat {}: {error}", output_path.display()))?
        .len();
    let expected = (wanted.len() as u64)
        .checked_mul(QWEN35_VOCAB_SIZE as u64)
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| "logit output byte count overflowed".to_owned())?;
    if bytes != expected {
        return Err(format!(
            "case {} wrote {} bytes, expected {}",
            case.id, bytes, expected
        ));
    }
    let file_name = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "logit output filename is not valid UTF-8".to_owned())?;
    Ok(CaseReport {
        id: case.id.clone(),
        token_ids: case.token_ids.clone(),
        token_ids_sha256: token_ids_sha256(&case.token_ids),
        all_positions: case.positions.is_none(),
        positions,
        rows: bytes as usize / (QWEN35_VOCAB_SIZE * 4),
        vocab_size: QWEN35_VOCAB_SIZE,
        logits_dtype: "FP32 little-endian row-major",
        logits_file: file_name.to_owned(),
        logits_file_sha256: sha256_file(output_path)?,
        logits_file_bytes: bytes,
        submission_count: audit.submission_count(),
        kernel_dispatch_count: audit.kernel_dispatch_count(),
        all_dispatches_hip: audit.all_dispatches_hip(),
        fallback_used: audit.fallback_used(),
        actual_chunk_sizes,
    })
}

fn run(args: Args) -> Result<Report, String> {
    let manifest_bytes = fs::read(&args.manifest)
        .map_err(|error| format!("read manifest {}: {error}", args.manifest.display()))?;
    let manifest = validate_manifest(&manifest_bytes)?;
    fs::create_dir_all(&args.output_dir)
        .map_err(|error| format!("create output directory: {error}"))?;
    let artifact =
        Arc::new(verify_unsloth_qwen38_nvfp4(&args.model_root).map_err(|error| error.to_string())?);
    let lock_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(SEMANTIC_LOCK);
    let lock = read_model_lock(&lock_path).map_err(|error| error.to_string())?;
    let plan =
        build_qwen38_nvfp4_weight_load_plan(&lock, &artifact).map_err(|error| error.to_string())?;
    let max_tokens = manifest
        .cases
        .iter()
        .map(|case| case.token_ids.len())
        .max()
        .ok_or_else(|| "manifest has no cases".to_owned())?;
    let state_capacity = u64::try_from(max_tokens)
        .map_err(|_| "manifest token count does not fit u64".to_owned())?
        .checked_add(1)
        .ok_or_else(|| "state capacity overflowed u64".to_owned())?
        .max(args.chunk_size as u64);
    let graph = build_qwen35_unsloth_qwen38_nvfp4_graph(
        &lock,
        &plan,
        &artifact,
        args.chunk_size as u64,
        state_capacity,
        args.kv,
    )
    .map_err(|error| error.to_string())?;
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(args.device_index, args.target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let operation = (|| -> Result<Vec<CaseReport>, String> {
        let resident = QwenResidentModel::new_unsloth_qwen38_nvfp4(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Arc::clone(&artifact),
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("model provisioning failed: {error}"))?;
        let mut reports = Vec::with_capacity(manifest.cases.len());
        for case in &manifest.cases {
            let graph = build_qwen35_unsloth_qwen38_nvfp4_graph(
                &lock,
                &plan,
                &artifact,
                args.chunk_size as u64,
                state_capacity,
                args.kv,
            )
            .map_err(|error| format!("case graph {}: {error}", case.id))?;
            let mut request = resident
                .new_request(graph)
                .map_err(|error| format!("request {} creation failed: {error}", case.id))?;
            let output_path = args.output_dir.join(format!("{}.f32", case.id));
            eprintln!(
                "[sllm-qwen38-kld-dump] case {}/{} begin id={} tokens={} positions={} chunk_size={}",
                reports.len() + 1,
                manifest.cases.len(),
                case.id,
                case.token_ids.len(),
                selected_positions(case).len(),
                args.chunk_size
            );
            reports.push(write_case_logits(
                &mut request,
                case,
                &output_path,
                &args.target,
                args.chunk_size,
            )?);
            eprintln!(
                "[sllm-qwen38-kld-dump] case {}/{} complete id={}",
                reports.len(),
                manifest.cases.len(),
                case.id
            );
        }
        Ok(reports)
    })();
    let shutdown = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("session shutdown failed: {error}"))?;
    if shutdown.retryable_cleanup != 0 || shutdown.durable_quarantine != 0 {
        return Err(format!(
            "session cleanup was nonzero: retryable={} durable={}",
            shutdown.retryable_cleanup, shutdown.durable_quarantine
        ));
    }
    let cases = operation?;
    let output_manifest = OutputManifest {
        engine: "sLLM",
        model: "Qwen3.8-27B",
        model_fingerprint: format!("sha256:{UNSLOTH_QWEN38_NVFP4_MODEL_SHA256}"),
        kv: args.kv.canonical_name(),
        chunk_size: args.chunk_size,
        execution_mode: if args.chunk_size == 1 {
            SERIAL_EXECUTION_MODE
        } else {
            BLOCK_EXECUTION_MODE
        },
        logit_storage_precision: if args.chunk_size == 1 {
            SERIAL_LOGIT_STORAGE_PRECISION
        } else {
            BLOCK_LOGIT_STORAGE_PRECISION
        },
        vocab_size: QWEN35_VOCAB_SIZE,
        position_semantics: "logits after token_ids[:position+1], predicting next token",
        cases: cases
            .iter()
            .map(|case| OutputManifestCase {
                id: case.id.clone(),
                logits_file: case.logits_file.clone(),
                shape: [case.rows, case.vocab_size],
                positions: case.positions.clone(),
                input_token_ids_sha256: case.token_ids_sha256.clone(),
                nonfinite_count: 0,
                actual_chunk_sizes: case.actual_chunk_sizes.clone(),
            })
            .collect(),
    };
    let output_manifest_path = args.output_dir.join("manifest.json");
    let output_manifest_bytes = serde_json::to_vec_pretty(&output_manifest)
        .map_err(|error| format!("serialize output manifest: {error}"))?;
    fs::write(&output_manifest_path, &output_manifest_bytes)
        .map_err(|error| format!("write {}: {error}", output_manifest_path.display()))?;
    Ok(Report {
        schema_version: SCHEMA_VERSION,
        state: "PASS",
        model_repository: UNSLOTH_QWEN38_NVFP4_REPOSITORY,
        model_revision: UNSLOTH_QWEN38_NVFP4_REVISION,
        model_fingerprint: format!("sha256:{UNSLOTH_QWEN38_NVFP4_MODEL_SHA256}"),
        recipe_digest: artifact.recipe_digest().to_owned(),
        semantic_lock: SEMANTIC_LOCK,
        semantic_lock_scope: SEMANTIC_LOCK_SCOPE,
        target: args.target,
        device_index: args.device_index,
        kv_cache_encoding: args.kv.canonical_name(),
        chunk_size: args.chunk_size,
        execution_mode: if args.chunk_size == 1 {
            SERIAL_EXECUTION_MODE
        } else {
            BLOCK_EXECUTION_MODE
        },
        logit_storage_precision: if args.chunk_size == 1 {
            SERIAL_LOGIT_STORAGE_PRECISION
        } else {
            BLOCK_LOGIT_STORAGE_PRECISION
        },
        manifest: args.manifest.display().to_string(),
        manifest_sha256: sha256_bytes(&manifest_bytes),
        output_dir: args.output_dir.display().to_string(),
        output_manifest: output_manifest_path.display().to_string(),
        cases,
    })
}

fn main() -> ExitCode {
    match parse_args(&env::args().skip(1).collect::<Vec<_>>()).and_then(run) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("Qwen3.8 KLD logit dump failed: {error}");
            ExitCode::FAILURE
        }
    }
}
