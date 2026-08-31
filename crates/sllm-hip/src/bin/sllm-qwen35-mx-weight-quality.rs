//! BF16 versus OCP MX weight/activation quality comparison for Qwen3.5-4B.
//!
//! The reviewed BF16 and candidate GGUF artifacts are provisioned strictly
//! sequentially. Both use explicit FP16 KV so only the model weight and
//! dynamic activation formats differ. The fixed Phase 46 dataset contributes
//! one prefill and one teacher-forced decode logit row per case.

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
    Backend, ExecutionSession, ExecutionSessionRequest, KvCacheEncoding, QWEN35_4B_FINGERPRINT,
    QWEN35_4B_REPO_ID, QWEN35_4B_REVISION, QwenComponentSelection, QwenExecutionAudit,
    QwenResidentModel, VerifiedGgufWeightSource, WeightLoadPlan,
    build_qwen35_gguf_mx_weight_activation_graph, build_qwen35_graph_with_kv_cache_encoding,
    build_verified_gguf_qwen_weight_load_plan, read_derived_gguf_lock, read_model_lock,
    verify_derived_gguf,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(300);
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
struct ArtifactRun {
    rows: Vec<LogitPair>,
    weight_encoding: String,
    artifact_sha256: String,
    artifact_size_bytes: u64,
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
    reference_top1: usize,
    candidate_top1: usize,
    top1_match: bool,
    kld_reference_to_candidate: f64,
    max_abs_logit_error: f32,
}

#[derive(Debug, Serialize)]
struct PerplexityComparison {
    token_count: usize,
    reference_loss_sum: f64,
    candidate_loss_sum: f64,
    reference_perplexity: f64,
    candidate_perplexity: f64,
    relative_delta: f64,
}

#[derive(Debug, Serialize)]
struct MemoryComparison {
    reference_model_resident_bytes: u64,
    candidate_model_resident_bytes: u64,
    model_resident_reduction_bytes: i128,
    model_resident_reduction_fraction: f64,
    reference_request_state_peak_bytes: u64,
    candidate_request_state_peak_bytes: u64,
    reference_workspace_peak_bytes: u64,
    candidate_workspace_peak_bytes: u64,
    reference_total_peak_bytes: u64,
    candidate_total_peak_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ArtifactIdentity {
    weight_encoding: String,
    sha256: String,
    size_bytes: u64,
    dispatches: u64,
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
    kv_encoding: &'static str,
    accumulation: &'static str,
    output_dtype: &'static str,
    quality_gate_applied: bool,
    case_count: usize,
    row_count: usize,
    sequential_residents: bool,
    reference_released_before_candidate: bool,
    reference: ArtifactIdentity,
    candidate: ArtifactIdentity,
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
    retryable_cleanup: usize,
    durable_quarantine: usize,
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

fn build_graph(
    lock: &sllm_core::ModelLock,
    plan: &WeightLoadPlan,
    source: &VerifiedGgufWeightSource,
    token_count: u64,
    state_capacity: u64,
) -> Result<sllm_core::QwenGraph, String> {
    if source.has_mx_weight_activation_recipe() {
        build_qwen35_gguf_mx_weight_activation_graph(
            lock,
            plan,
            source,
            token_count,
            state_capacity,
            KvCacheEncoding::Fp16,
        )
    } else if source.has_quantized_linear_recipe() {
        return Err("quality runner accepts only BF16 or OCP MX GGUF artifacts".to_owned());
    } else {
        build_qwen35_graph_with_kv_cache_encoding(
            lock,
            plan,
            token_count,
            state_capacity,
            KvCacheEncoding::Fp16,
        )
    }
    .map_err(|error| format!("build graph: {error}"))
}

#[allow(clippy::too_many_arguments)]
fn execute_artifact(
    session: &Arc<ExecutionSession>,
    lock: &sllm_core::ModelLock,
    gguf_path: &Path,
    derived_lock_path: &Path,
    cases: &[PreparedCase],
    target: &str,
    require_mx: bool,
) -> Result<ArtifactRun, String> {
    if session.memory_snapshot().current_bytes() != 0 {
        return Err("session was not empty before resident creation".to_owned());
    }
    let derived_lock = read_derived_gguf_lock(derived_lock_path)
        .map_err(|error| format!("read derived lock: {error}"))?;
    let artifact_sha256 = derived_lock.output.sha256.clone();
    let artifact_size_bytes = derived_lock.output.size_bytes;
    let verified = verify_derived_gguf(derived_lock, gguf_path)
        .map_err(|error| format!("verify GGUF: {error}"))?;
    let (source, plan) = build_verified_gguf_qwen_weight_load_plan(
        lock,
        verified,
        QwenComponentSelection::TEXT_ONLY,
    )
    .map_err(|error| format!("build GGUF weight plan: {error}"))?;
    let has_mx = source.has_mx_weight_activation_recipe();
    if has_mx != require_mx {
        return Err(if require_mx {
            "candidate GGUF is not an OCP MX weight/activation artifact".to_owned()
        } else {
            "reference GGUF is not the BF16 artifact".to_owned()
        });
    }
    let weight_encoding = source
        .mx_weight_activation_encoding_name()
        .unwrap_or("bf16")
        .to_owned();
    let source = Arc::new(source);
    let maximum = cases
        .iter()
        .map(|case| case.tokens.len())
        .max()
        .unwrap_or(0);
    let seed_graph = build_graph(
        lock,
        &plan,
        source.as_ref(),
        maximum as u64,
        maximum as u64 + 1,
    )?;
    let resident = QwenResidentModel::new_gguf(
        Arc::clone(session),
        seed_graph,
        plan.clone(),
        Arc::clone(&source),
        COMPLETION_TIMEOUT,
    )
    .map_err(|error| format!("create {weight_encoding} resident model: {error}"))?;
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
        let graph = build_graph(
            lock,
            &plan,
            source.as_ref(),
            case.tokens.len() as u64,
            case.tokens.len() as u64 + 1,
        )
        .map_err(|error| format!("{} case {}: {error}", weight_encoding, case.id))?;
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
    drop(source);
    let released = session.memory_snapshot();
    if released.poisoned() || released.current_bytes() != 0 {
        return Err(format!("resident release was incomplete: {released:?}"));
    }
    Ok(ArtifactRun {
        rows,
        weight_encoding,
        artifact_sha256,
        artifact_size_bytes,
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
    reference: ArtifactRun,
    candidate: ArtifactRun,
) -> Result<
    (
        Vec<RowComparison>,
        PerplexityComparison,
        MemoryComparison,
        ArtifactIdentity,
        ArtifactIdentity,
    ),
    String,
> {
    if reference.rows.len() != cases.len() || candidate.rows.len() != cases.len() {
        return Err("artifact run row count differs from dataset".to_owned());
    }
    let mut rows = Vec::with_capacity(cases.len() * 2);
    let mut reference_loss_sum = 0.0_f64;
    let mut candidate_loss_sum = 0.0_f64;
    for ((case, reference_row), candidate_row) in
        cases.iter().zip(&reference.rows).zip(&candidate.rows)
    {
        reference_loss_sum += nll(&reference_row.prefill, case.expected_next);
        candidate_loss_sum += nll(&candidate_row.prefill, case.expected_next);
        for (phase, position, left, right) in [
            (
                "prefill",
                case.tokens.len().saturating_sub(1),
                &reference_row.prefill,
                &candidate_row.prefill,
            ),
            (
                "decode",
                case.tokens.len(),
                &reference_row.decode,
                &candidate_row.decode,
            ),
        ] {
            let reference_top1 = top1(left);
            let candidate_top1 = top1(right);
            rows.push(RowComparison {
                case_id: case.id.clone(),
                phase,
                position,
                reference_top1,
                candidate_top1,
                top1_match: reference_top1 == candidate_top1,
                kld_reference_to_candidate: kld(left, right),
                max_abs_logit_error: left
                    .iter()
                    .zip(right)
                    .map(|(left, right)| (*left - *right).abs())
                    .fold(0.0_f32, f32::max),
            });
        }
    }
    let token_count = cases.len();
    let reference_perplexity = (reference_loss_sum / token_count as f64).exp();
    let candidate_perplexity = (candidate_loss_sum / token_count as f64).exp();
    for (label, value) in [
        ("reference loss", reference_loss_sum),
        ("candidate loss", candidate_loss_sum),
        ("reference perplexity", reference_perplexity),
        ("candidate perplexity", candidate_perplexity),
    ] {
        if !value.is_finite() {
            return Err(format!("{label} is non-finite"));
        }
    }
    let perplexity = PerplexityComparison {
        token_count,
        reference_loss_sum,
        candidate_loss_sum,
        reference_perplexity,
        candidate_perplexity,
        relative_delta: (candidate_perplexity - reference_perplexity) / reference_perplexity,
    };
    let model_reduction =
        i128::from(reference.model_resident_bytes) - i128::from(candidate.model_resident_bytes);
    let memory = MemoryComparison {
        reference_model_resident_bytes: reference.model_resident_bytes,
        candidate_model_resident_bytes: candidate.model_resident_bytes,
        model_resident_reduction_bytes: model_reduction,
        model_resident_reduction_fraction: model_reduction as f64
            / reference.model_resident_bytes as f64,
        reference_request_state_peak_bytes: reference.request_state_peak_bytes,
        candidate_request_state_peak_bytes: candidate.request_state_peak_bytes,
        reference_workspace_peak_bytes: reference.workspace_peak_bytes,
        candidate_workspace_peak_bytes: candidate.workspace_peak_bytes,
        reference_total_peak_bytes: reference.total_peak_bytes,
        candidate_total_peak_bytes: candidate.total_peak_bytes,
    };
    let reference_identity = ArtifactIdentity {
        weight_encoding: reference.weight_encoding,
        sha256: reference.artifact_sha256,
        size_bytes: reference.artifact_size_bytes,
        dispatches: reference.dispatches,
    };
    let candidate_identity = ArtifactIdentity {
        weight_encoding: candidate.weight_encoding,
        sha256: candidate.artifact_sha256,
        size_bytes: candidate.artifact_size_bytes,
        dispatches: candidate.dispatches,
    };
    Ok((
        rows,
        perplexity,
        memory,
        reference_identity,
        candidate_identity,
    ))
}

fn valid_target(target: &str) -> bool {
    matches!(target, "gfx1030" | "gfx1201")
}

fn run(arguments: &[String]) -> Result<(Report, PathBuf), String> {
    if arguments.len() != 9 {
        return Err(
            "usage: MODEL_LOCK DATASET_JSON DEVICE_INDEX TARGET REFERENCE_GGUF REFERENCE_DERIVED_LOCK CANDIDATE_GGUF CANDIDATE_DERIVED_LOCK OUTPUT_JSON"
                .to_owned(),
        );
    }
    let model_lock_path = PathBuf::from(&arguments[0]);
    let dataset_path = PathBuf::from(&arguments[1]);
    let device_index = arguments[2]
        .parse::<u32>()
        .map_err(|_| "device index must be u32".to_owned())?;
    let target = arguments[3].clone();
    if !valid_target(&target) {
        return Err("target must be exactly gfx1030 or gfx1201".to_owned());
    }
    let reference_gguf = PathBuf::from(&arguments[4]);
    let reference_lock = PathBuf::from(&arguments[5]);
    let candidate_gguf = PathBuf::from(&arguments[6]);
    let candidate_lock = PathBuf::from(&arguments[7]);
    let output_path = PathBuf::from(&arguments[8]);
    if output_path.exists() {
        return Err("output already exists".to_owned());
    }
    let (dataset_sha256, cases) = load_dataset(&dataset_path)?;
    let lock =
        read_model_lock(&model_lock_path).map_err(|error| format!("read model lock: {error}"))?;
    if lock.model.repo_id != QWEN35_4B_REPO_ID
        || lock.model.resolved_revision != QWEN35_4B_REVISION
        || lock.fingerprint() != QWEN35_4B_FINGERPRINT
    {
        return Err("quality runner requires the reviewed Qwen3.5-4B lock".to_owned());
    }
    let backend = HipBackend::connect().map_err(|error| format!("connect HIP: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| format!("session request: {error}"))?,
        )
        .map_err(|error| format!("open HIP session: {error}"))?;

    let execution = (|| {
        let reference = execute_artifact(
            &session,
            &lock,
            &reference_gguf,
            &reference_lock,
            &cases,
            &target,
            false,
        )?;
        if session.memory_snapshot().current_bytes() != 0 {
            return Err("reference resident remained before candidate provisioning".to_owned());
        }
        let candidate = execute_artifact(
            &session,
            &lock,
            &candidate_gguf,
            &candidate_lock,
            &cases,
            &target,
            true,
        )?;
        if session.memory_snapshot().current_bytes() != 0 {
            return Err("candidate resident remained after measurement".to_owned());
        }
        let (rows, perplexity, memory, reference, candidate) =
            compare_runs(&cases, reference, candidate)?;
        let mut klds = rows
            .iter()
            .map(|row| row.kld_reference_to_candidate)
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
            schema_version: "sllm-qwen35-mx-weight-quality-v1",
            state: "PASS",
            model_repo_id: QWEN35_4B_REPO_ID,
            model_revision: QWEN35_4B_REVISION,
            model_fingerprint: QWEN35_4B_FINGERPRINT,
            dataset_sha256,
            executable_sha256,
            target,
            device_index,
            kv_encoding: "fp16",
            accumulation: "fp32",
            output_dtype: "bf16",
            quality_gate_applied: false,
            case_count: cases.len(),
            row_count: rows.len(),
            sequential_residents: true,
            reference_released_before_candidate: true,
            reference,
            candidate,
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
            retryable_cleanup: 0,
            durable_quarantine: 0,
            final_cleanup_empty: true,
        })
    })();
    let mut report = match execution {
        Ok(report) => report,
        Err(error) => {
            let _ = session.shutdown(SHUTDOWN_TIMEOUT);
            return Err(error);
        }
    };
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("shutdown session: {error}"))?;
    let empty = cleanup.retryable_cleanup == 0
        && cleanup.durable_quarantine == 0
        && session.memory_snapshot().current_bytes() == 0;
    if !empty {
        return Err(format!("final cleanup was nonzero: {cleanup:?}"));
    }
    report.retryable_cleanup = cleanup.retryable_cleanup;
    report.durable_quarantine = cleanup.durable_quarantine;
    report.final_cleanup_empty = empty;
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
        OpenOptions::new()
            .read(true)
            .open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("sync output directory: {error}"))?;
        Ok::<(), String>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
        let _ = fs::remove_file(output);
    }
    result?;
    Ok(format!("sha256:{digest}"))
}

fn main() -> ExitCode {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok((report, output)) => match publish(&report, &output) {
            Ok(digest) => {
                println!("{} {digest}", output.display());
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("MX weight quality publication failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("MX weight quality failed: {error}");
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
