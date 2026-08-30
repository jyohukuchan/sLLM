//! One-shot FP16 versus standard OCP MXFP8 E4 KV quality probe for Qwen3.5-4B.
//!
//! The two resident models are deliberately provisioned sequentially. The
//! FP16 resident is dropped and the session allocation snapshot must return
//! to zero before the MXFP8 resident can be created.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSession, ExecutionSessionRequest, KvCacheEncoding, KvCacheSelection,
    KvCacheSelectionRequest, QWEN35_4B_FINGERPRINT, QWEN35_4B_REPO_ID, QWEN35_4B_REVISION,
    QwenExecutionAudit, QwenResidentModel, build_qwen35_graph_with_kv_cache_encoding,
    build_qwen35_graph_with_kv_cache_selection, build_verified_weight_load_plan, read_model_lock,
    resolve_kv_cache_selection,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(180);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const DATASET_SHA256: &str = "a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d";
const VOCAB_SIZE: usize = 248_320;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    schema_version: String,
    dataset_id: String,
    license: String,
    provenance: String,
    seed: u64,
    token_generator: String,
    sample_order: String,
    cases: Vec<DatasetCase>,
    coverage: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetCase {
    id: String,
    length: usize,
    start: u64,
    step: u64,
    expected_next: i32,
    band: String,
    block_tail: bool,
}

#[derive(Debug)]
struct PreparedCase {
    id: String,
    tokens: Vec<i32>,
    expected_next: i32,
}

#[derive(Debug)]
struct LogitPair {
    prefill: Vec<f32>,
    decode: Vec<f32>,
}

#[derive(Debug)]
struct EncodingRun {
    rows: Vec<LogitPair>,
    model_resident_bytes: u64,
    request_state_peak_bytes: u64,
    workspace_peak_bytes: u64,
    total_peak_bytes: u64,
    dispatches: u64,
}

#[derive(Debug, Serialize)]
struct RowComparison {
    case_id: String,
    phase: &'static str,
    position: usize,
    fp16_top1: usize,
    fp8_top1: usize,
    top1_match: bool,
    kld_fp16_to_fp8: f64,
    max_abs_logit_error: f32,
}

#[derive(Debug, Serialize)]
struct PerplexityComparison {
    token_count: usize,
    fp16_loss_sum: f64,
    fp8_loss_sum: f64,
    fp16_perplexity: f64,
    fp8_perplexity: f64,
    relative_delta: f64,
}

#[derive(Debug, Serialize)]
struct MemoryComparison {
    fp16_model_resident_bytes: u64,
    fp8_model_resident_bytes: u64,
    fp16_request_state_peak_bytes: u64,
    fp8_request_state_peak_bytes: u64,
    request_state_reduction_bytes: i128,
    request_state_reduction_fraction: f64,
    fp16_workspace_peak_bytes: u64,
    fp8_workspace_peak_bytes: u64,
    fp16_total_peak_bytes: u64,
    fp8_total_peak_bytes: u64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    model_repo_id: &'static str,
    model_revision: &'static str,
    model_fingerprint: &'static str,
    dataset_sha256: String,
    executable_sha256: String,
    target: String,
    device_index: u32,
    reference_kv_encoding: &'static str,
    candidate_kv_encoding: &'static str,
    candidate_scale_granularity: &'static str,
    repeats: u32,
    case_count: usize,
    row_count: usize,
    sequential_residents: bool,
    fp16_released_before_fp8: bool,
    fp16_dispatches: u64,
    fp8_dispatches: u64,
    top1_matches: usize,
    top1_agreement: f64,
    kld_mean: f64,
    kld_p50: f64,
    kld_p90: f64,
    kld_p99: f64,
    kld_max: f64,
    max_abs_logit_error: f32,
    perplexity: PerplexityComparison,
    memory: MemoryComparison,
    rows: Vec<RowComparison>,
    final_cleanup_empty: bool,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn load_dataset(path: &Path) -> Result<(String, Vec<PreparedCase>), String> {
    let bytes = fs::read(path).map_err(|error| format!("read dataset: {error}"))?;
    let digest = sha256(&bytes);
    if digest != DATASET_SHA256 {
        return Err(format!(
            "dataset digest differs: expected {DATASET_SHA256}, got {digest}"
        ));
    }
    let dataset: Dataset =
        serde_json::from_slice(&bytes).map_err(|error| format!("parse dataset: {error}"))?;
    if dataset.schema_version != "sllm-phase46-kv-quality-dataset-v1"
        || dataset.dataset_id != "phase46-kv-quality-baseline-v1"
        || dataset.license != "CC0-1.0"
        || dataset.provenance.is_empty()
        || dataset.token_generator != "token[i] = 1 + ((start + i * step + seed) mod 200000)"
        || dataset.sample_order != "listed"
        || dataset.seed != 1729
        || dataset.cases.len() != 10
        || !dataset.coverage.is_object()
    {
        return Err("dataset identity or generator contract differs".to_owned());
    }
    let mut prepared = Vec::with_capacity(dataset.cases.len());
    for case in dataset.cases {
        if case.id.is_empty()
            || case.length == 0
            || case.length > 513
            || case.band.is_empty()
            || !(0..VOCAB_SIZE as i32).contains(&case.expected_next)
        {
            return Err(format!("dataset case {} is invalid", case.id));
        }
        let mut tokens = Vec::with_capacity(case.length);
        for index in 0..case.length {
            let value = case
                .start
                .checked_add(
                    (index as u64)
                        .checked_mul(case.step)
                        .ok_or("token product overflow")?,
                )
                .and_then(|value| value.checked_add(dataset.seed))
                .ok_or("token generator overflow")?
                % 200_000
                + 1;
            tokens.push(i32::try_from(value).map_err(|_| "token does not fit i32")?);
        }
        let _ = case.block_tail;
        prepared.push(PreparedCase {
            id: case.id,
            tokens,
            expected_next: case.expected_next,
        });
    }
    Ok((format!("sha256:{digest}"), prepared))
}

fn validate_audit(audit: &QwenExecutionAudit, target: &str) -> Result<(), String> {
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

fn validate_logits(values: &[f32], label: &str) -> Result<(), String> {
    if values.len() != VOCAB_SIZE || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} logits are non-finite or truncated"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_encoding(
    session: &Arc<ExecutionSession>,
    lock: &sllm_core::ModelLock,
    cache: &Arc<sllm_core::VerifiedCache>,
    plan: &sllm_core::WeightLoadPlan,
    cases: &[PreparedCase],
    encoding: KvCacheEncoding,
    selection: Option<KvCacheSelection>,
    target: &str,
) -> Result<EncodingRun, String> {
    if session.memory_snapshot().current_bytes() != 0 {
        return Err("session was not empty before resident creation".to_owned());
    }
    let maximum = cases
        .iter()
        .map(|case| case.tokens.len())
        .max()
        .unwrap_or(0);
    let build_graph = |token_count, state_capacity| match selection {
        Some(selection) => build_qwen35_graph_with_kv_cache_selection(
            lock,
            plan,
            token_count,
            state_capacity,
            selection,
        ),
        None => build_qwen35_graph_with_kv_cache_encoding(
            lock,
            plan,
            token_count,
            state_capacity,
            encoding,
        ),
    };
    let seed_graph = build_graph(maximum as u64, maximum as u64 + 1)
        .map_err(|error| format!("build seed graph: {error}"))?;
    let resident = QwenResidentModel::new(
        Arc::clone(session),
        seed_graph,
        plan.clone(),
        Arc::clone(cache),
        COMPLETION_TIMEOUT,
    )
    .map_err(|error| format!("create resident model: {error}"))?;
    let ready = session.memory_snapshot();
    if ready.poisoned()
        || ready.model_resident().current_bytes() == 0
        || ready.request_state().current_bytes() != 0
        || ready.workspace().current_bytes() != 0
    {
        return Err(format!("resident baseline is invalid: {ready:?}"));
    }

    let mut rows = Vec::with_capacity(cases.len());
    let mut request_state_peak_bytes = 0_u64;
    let mut workspace_peak_bytes = 0_u64;
    let mut total_peak_bytes = ready.current_bytes();
    let mut dispatches = 0_u64;
    for case in cases {
        let graph = build_graph(case.tokens.len() as u64, case.tokens.len() as u64 + 1)
            .map_err(|error| format!("build case graph {}: {error}", case.id))?;
        let mut request = resident
            .new_request(graph)
            .map_err(|error| format!("create request {}: {error}", case.id))?;
        let prefill = request
            .prefill_with_last_logits(&case.tokens)
            .map_err(|error| format!("prefill {}: {error}", case.id))?;
        let prefill = prefill
            .last_logits()
            .ok_or_else(|| format!("prefill {} omitted full logits", case.id))?
            .to_vec();
        validate_logits(&prefill, "prefill")?;
        let decode = request
            .decode_with_last_logits(case.expected_next)
            .map_err(|error| format!("decode {}: {error}", case.id))?;
        let decode = decode
            .last_logits()
            .ok_or_else(|| format!("decode {} omitted full logits", case.id))?
            .to_vec();
        validate_logits(&decode, "decode")?;
        let audit = request
            .audit_snapshot()
            .map_err(|error| format!("audit {}: {error}", case.id))?;
        validate_audit(&audit, target)?;
        dispatches = dispatches
            .checked_add(audit.kernel_dispatch_count())
            .ok_or("dispatch count overflow")?;
        let active = session.memory_snapshot();
        request_state_peak_bytes =
            request_state_peak_bytes.max(active.request_state().current_bytes());
        workspace_peak_bytes = workspace_peak_bytes.max(active.workspace().current_bytes());
        total_peak_bytes = total_peak_bytes.max(active.current_bytes());
        rows.push(LogitPair { prefill, decode });
        drop(request);
        let restored = session.memory_snapshot();
        if restored.poisoned()
            || restored.model_resident().current_bytes() != ready.model_resident().current_bytes()
            || restored.request_state().current_bytes() != 0
            || restored.workspace().current_bytes() != 0
            || restored.current_bytes() != ready.current_bytes()
        {
            return Err(format!(
                "request cleanup did not restore resident baseline: {restored:?}"
            ));
        }
    }
    let model_resident_bytes = ready.model_resident().current_bytes();
    drop(resident);
    let released = session.memory_snapshot();
    if released.poisoned() || released.current_bytes() != 0 {
        return Err(format!("resident release was incomplete: {released:?}"));
    }
    Ok(EncodingRun {
        rows,
        model_resident_bytes,
        request_state_peak_bytes,
        workspace_peak_bytes,
        total_peak_bytes,
        dispatches,
    })
}

fn top1(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map_or(0, |(index, _)| index)
}

fn logsumexp(values: &[f32]) -> f64 {
    let maximum = f64::from(values.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    maximum
        + values
            .iter()
            .map(|value| (f64::from(*value) - maximum).exp())
            .sum::<f64>()
            .ln()
}

fn nll(values: &[f32], target: i32) -> f64 {
    logsumexp(values) - f64::from(values[target as usize])
}

fn kld(reference: &[f32], candidate: &[f32]) -> f64 {
    let reference_lse = logsumexp(reference);
    let candidate_lse = logsumexp(candidate);
    reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| {
            let log_p = f64::from(*reference) - reference_lse;
            let log_q = f64::from(*candidate) - candidate_lse;
            log_p.exp() * (log_p - log_q)
        })
        .sum::<f64>()
        .max(0.0)
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values[index.min(values.len() - 1)]
}

fn compare_runs(
    cases: &[PreparedCase],
    fp16: EncodingRun,
    fp8: EncodingRun,
) -> Result<(Vec<RowComparison>, PerplexityComparison, MemoryComparison), String> {
    if fp16.rows.len() != cases.len() || fp8.rows.len() != cases.len() {
        return Err("encoding run row count differs from dataset".to_owned());
    }
    let mut rows = Vec::with_capacity(cases.len() * 2);
    let mut fp16_loss_sum = 0.0_f64;
    let mut fp8_loss_sum = 0.0_f64;
    for ((case, reference), candidate) in cases.iter().zip(&fp16.rows).zip(&fp8.rows) {
        fp16_loss_sum += nll(&reference.prefill, case.expected_next);
        fp8_loss_sum += nll(&candidate.prefill, case.expected_next);
        for (phase, position, left, right) in [
            (
                "prefill",
                case.tokens.len().saturating_sub(1),
                &reference.prefill,
                &candidate.prefill,
            ),
            (
                "decode",
                case.tokens.len(),
                &reference.decode,
                &candidate.decode,
            ),
        ] {
            let fp16_top1 = top1(left);
            let fp8_top1 = top1(right);
            rows.push(RowComparison {
                case_id: case.id.clone(),
                phase,
                position,
                fp16_top1,
                fp8_top1,
                top1_match: fp16_top1 == fp8_top1,
                kld_fp16_to_fp8: kld(left, right),
                max_abs_logit_error: left
                    .iter()
                    .zip(right)
                    .map(|(left, right)| (*left - *right).abs())
                    .fold(0.0_f32, f32::max),
            });
        }
    }
    let token_count = cases.len();
    let fp16_perplexity = (fp16_loss_sum / token_count as f64).exp();
    let fp8_perplexity = (fp8_loss_sum / token_count as f64).exp();
    for (label, value) in [
        ("fp16 loss", fp16_loss_sum),
        ("fp8 loss", fp8_loss_sum),
        ("fp16 perplexity", fp16_perplexity),
        ("fp8 perplexity", fp8_perplexity),
    ] {
        if !value.is_finite() {
            return Err(format!("{label} is non-finite"));
        }
    }
    let perplexity = PerplexityComparison {
        token_count,
        fp16_loss_sum,
        fp8_loss_sum,
        fp16_perplexity,
        fp8_perplexity,
        relative_delta: (fp8_perplexity - fp16_perplexity) / fp16_perplexity,
    };
    let reduction =
        i128::from(fp16.request_state_peak_bytes) - i128::from(fp8.request_state_peak_bytes);
    let reduction_fraction = if fp16.request_state_peak_bytes == 0 {
        0.0
    } else {
        reduction as f64 / fp16.request_state_peak_bytes as f64
    };
    let memory = MemoryComparison {
        fp16_model_resident_bytes: fp16.model_resident_bytes,
        fp8_model_resident_bytes: fp8.model_resident_bytes,
        fp16_request_state_peak_bytes: fp16.request_state_peak_bytes,
        fp8_request_state_peak_bytes: fp8.request_state_peak_bytes,
        request_state_reduction_bytes: reduction,
        request_state_reduction_fraction: reduction_fraction,
        fp16_workspace_peak_bytes: fp16.workspace_peak_bytes,
        fp8_workspace_peak_bytes: fp8.workspace_peak_bytes,
        fp16_total_peak_bytes: fp16.total_peak_bytes,
        fp8_total_peak_bytes: fp8.total_peak_bytes,
    };
    Ok((rows, perplexity, memory))
}

fn valid_target(target: &str) -> bool {
    target.len() > 3
        && target.starts_with("gfx")
        && target[3..].bytes().all(|byte| byte.is_ascii_digit())
}

fn run(arguments: &[String]) -> Result<(Report, PathBuf), String> {
    if arguments.len() != 6 {
        return Err("usage: LOCK CACHE DATASET_JSON DEVICE_INDEX TARGET OUTPUT_JSON".to_owned());
    }
    let lock_path = PathBuf::from(&arguments[0]);
    let cache_path = PathBuf::from(&arguments[1]);
    let dataset_path = PathBuf::from(&arguments[2]);
    let device_index = arguments[3]
        .parse::<u32>()
        .map_err(|_| "device index must be u32".to_owned())?;
    let target = arguments[4].clone();
    if !valid_target(&target) {
        return Err("target must be an exact gfx target without feature suffixes".to_owned());
    }
    let output_path = PathBuf::from(&arguments[5]);
    if output_path.exists() {
        return Err("output already exists".to_owned());
    }
    let (dataset_sha256, cases) = load_dataset(&dataset_path)?;
    let lock = read_model_lock(&lock_path).map_err(|error| format!("read model lock: {error}"))?;
    if lock.model.repo_id != QWEN35_4B_REPO_ID
        || lock.model.resolved_revision != QWEN35_4B_REVISION
        || lock.fingerprint() != QWEN35_4B_FINGERPRINT
    {
        return Err("probe requires the reviewed Qwen3.5-4B lock".to_owned());
    }
    let cache = Arc::new(
        lock.verify_cache(cache_path)
            .map_err(|error| format!("verify cache: {error}"))?,
    );
    let plan = build_verified_weight_load_plan(&lock, &cache)
        .map_err(|error| format!("build weight plan: {error}"))?;
    let backend = HipBackend::connect().map_err(|error| format!("connect HIP: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| format!("session request: {error}"))?,
        )
        .map_err(|error| format!("open HIP session: {error}"))?;

    let execution = (|| {
        let default_selection = resolve_kv_cache_selection(KvCacheSelectionRequest::new(
            None,
            &target,
            lock.fingerprint(),
            true,
            true,
            true,
            256,
        ))
        .map_err(|error| format!("resolve default KV selection: {error}"))?;
        if default_selection.resolved() != KvCacheEncoding::Mxfp8E4 {
            return Err("default KV selection did not resolve to standard MXFP8 E4".to_owned());
        }
        let fp16 = execute_encoding(
            &session,
            &lock,
            &cache,
            &plan,
            &cases,
            KvCacheEncoding::Fp16,
            None,
            &target,
        )?;
        if session.memory_snapshot().current_bytes() != 0 {
            return Err("FP16 resident remained before MXFP8 provisioning".to_owned());
        }
        let fp8 = execute_encoding(
            &session,
            &lock,
            &cache,
            &plan,
            &cases,
            default_selection.resolved(),
            Some(default_selection),
            &target,
        )?;
        if session.memory_snapshot().current_bytes() != 0 {
            return Err("MXFP8 resident remained after measurement".to_owned());
        }
        let fp16_dispatches = fp16.dispatches;
        let fp8_dispatches = fp8.dispatches;
        let (rows, perplexity, memory) = compare_runs(&cases, fp16, fp8)?;
        let mut klds = rows
            .iter()
            .map(|row| row.kld_fp16_to_fp8)
            .collect::<Vec<_>>();
        klds.sort_by(f64::total_cmp);
        if klds.is_empty() || klds.iter().any(|value| !value.is_finite()) {
            return Err("KLD output is empty or non-finite".to_owned());
        }
        let top1_matches = rows.iter().filter(|row| row.top1_match).count();
        let executable = env::current_exe().map_err(|error| format!("locate binary: {error}"))?;
        let executable_sha256 = format!(
            "sha256:{}",
            sha256(&fs::read(executable).map_err(|error| format!("read binary: {error}"))?)
        );
        Ok(Report {
            schema_version: "sllm-qwen35-kv-quality-probe-v1",
            state: "PASS",
            model_repo_id: QWEN35_4B_REPO_ID,
            model_revision: QWEN35_4B_REVISION,
            model_fingerprint: QWEN35_4B_FINGERPRINT,
            dataset_sha256,
            executable_sha256,
            target,
            device_index,
            reference_kv_encoding: "fp16",
            candidate_kv_encoding: "kv-mxfp8-e4",
            candidate_scale_granularity: "per-block32-e8m0",
            repeats: 1,
            case_count: cases.len(),
            row_count: rows.len(),
            sequential_residents: true,
            fp16_released_before_fp8: true,
            fp16_dispatches,
            fp8_dispatches,
            top1_matches,
            top1_agreement: top1_matches as f64 / rows.len() as f64,
            kld_mean: klds.iter().sum::<f64>() / klds.len() as f64,
            kld_p50: percentile(&klds, 0.50),
            kld_p90: percentile(&klds, 0.90),
            kld_p99: percentile(&klds, 0.99),
            kld_max: *klds.last().expect("non-empty KLD list"),
            max_abs_logit_error: rows
                .iter()
                .map(|row| row.max_abs_logit_error)
                .fold(0.0_f32, f32::max),
            perplexity,
            memory,
            rows,
            final_cleanup_empty: true,
        })
    })();
    let report = match execution {
        Ok(report) => report,
        Err(error) => {
            let _ = session.shutdown(SHUTDOWN_TIMEOUT);
            return Err(error);
        }
    };
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("shutdown session: {error}"))?;
    if cleanup.retryable_cleanup != 0
        || cleanup.durable_quarantine != 0
        || session.memory_snapshot().current_bytes() != 0
    {
        return Err(format!("final cleanup was nonzero: {cleanup:?}"));
    }
    Ok((report, output_path))
}

fn publish(report: &Report, output: &Path) -> Result<String, String> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| format!("create output parent: {error}"))?;
    let name = output
        .file_name()
        .ok_or_else(|| "output must name a file".to_owned())?
        .to_string_lossy();
    let partial = parent.join(format!(".{name}.partial.{}", std::process::id()));
    let bytes = serde_json::to_vec_pretty(report).map_err(|error| format!("serialize: {error}"))?;
    let digest = sha256(&bytes);
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .map_err(|error| format!("create partial: {error}"))?;
        file.write_all(&bytes)
            .map_err(|error| format!("write partial: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("sync partial: {error}"))?;
        fs::hard_link(&partial, output).map_err(|error| format!("publish no-replace: {error}"))?;
        fs::remove_file(&partial).map_err(|error| format!("remove partial link: {error}"))?;
        FileSync::sync(parent)?;
        Ok::<(), String>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
        let _ = fs::remove_file(output);
    }
    result?;
    Ok(format!("sha256:{digest}"))
}

struct FileSync;

impl FileSync {
    fn sync(path: &Path) -> Result<(), String> {
        OpenOptions::new()
            .read(true)
            .open(path)
            .and_then(|file| file.sync_all())
            .map_err(|error| format!("sync output directory: {error}"))
    }
}

fn main() -> ExitCode {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok((report, output)) => match publish(&report, &output) {
            Ok(digest) => {
                println!("{} {digest}", output.display());
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("KV quality probe publication failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("KV quality probe failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_metrics_are_zero_for_identical_rows() {
        let values = [1.0_f32, 2.0, -3.0];
        assert_eq!(top1(&values), 1);
        assert_eq!(kld(&values, &values), 0.0);
        assert_eq!(nll(&values, 1), logsumexp(&values) - 2.0);
    }
}
