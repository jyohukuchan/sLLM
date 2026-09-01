//! BF16 versus OCP MX weight/activation quality comparison for Qwen3.5-4B.
//!
//! The reviewed BF16 and candidate GGUF artifacts are provisioned strictly
//! sequentially. Both use explicit FP16 KV so only the model weight and
//! dynamic activation formats differ. The fixed Phase 46 dataset contributes
//! one prefill and one teacher-forced decode logit row per case.
//!
//! `--provider-compare` instead provisions the same MXFP8 GGUF in isolated,
//! strictly sequential legacy-row8 and gfx1201-WMMA worker processes. Each
//! worker repeats the complete dataset and must release every resident and the
//! session with zero cleanup before the next provider can start.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
const PROVIDER_PROCESS_TIMEOUT: Duration = Duration::from_secs(600);
const DATASET_SHA256: &str = "a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d";
const VOCAB_SIZE: usize = 248_320;
const PROVIDER_COMPARE_MODE: &str = "--provider-compare";
const PROVIDER_WORKER_MODE: &str = "--provider-worker";
const MXFP8_WMMA_ENV: &str = "SLLM_MXFP8_PREFILL_FORCE_WMMA_GFX1201";
const MXFP8_ROW8_ENV: &str = "SLLM_MXFP8_PREFILL_FORCE_ROW8";
const MXFP8_WMMA_KERNEL_ID: u32 = 31;
const MXFP8_WMMA_KERNEL_SYMBOL: &str = "matmul.mxfp8.w8a8.e4m3.block32.prefill.wmma128x64x32.v2";
// Worker-only identity marker. The C++ selector deliberately ignores this
// variable; removing every force variable makes the normal target/M/N/K
// production selector authoritative before the candidate graph is prepared.
const MXFP8_SCOPED_DEFAULT_ENV: &str = "SLLM_MXFP8_PREFILL_SCOPED_DEFAULT_GFX1201";
const MX_WA_BASELINE_ENV: &str = "SLLM_MX_WA_PREFILL_FORCE_BASELINE";
const MX_WA_MMQ_COLUMNS_ENV: &str = "SLLM_MX_WA_PREFILL_FORCE_MMQ_COLUMNS";
const MXFP8_TILED16_ENV: &str = "SLLM_MXFP8_PREFILL_FORCE_TILED16";
const PROVIDER_ENVIRONMENTS: [&str; 6] = [
    MXFP8_WMMA_ENV,
    MXFP8_ROW8_ENV,
    MXFP8_SCOPED_DEFAULT_ENV,
    MX_WA_BASELINE_ENV,
    MX_WA_MMQ_COLUMNS_ENV,
    MXFP8_TILED16_ENV,
];

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

#[derive(Debug, Deserialize, Serialize)]
struct LogitPair {
    prefill: Vec<f32>,
    decode: Vec<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ArtifactRun {
    rows: Vec<LogitPair>,
    weight_encoding: String,
    artifact_sha256: String,
    artifact_size_bytes: u64,
    model_resident_bytes: u64,
    request_state_peak_bytes: u64,
    workspace_peak_bytes: u64,
    total_peak_bytes: u64,
    dispatch: DispatchAuditSummary,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct DispatchAuditSummary {
    selected_backend: String,
    target: String,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
    all_dispatches_hip: bool,
    segment_count: u64,
    boundary_count: u64,
    mxfp8_wmma_dispatch_count: u64,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProviderMode {
    LegacyRow8,
    WmmaCandidate,
}

impl ProviderMode {
    const fn argument(self) -> &'static str {
        match self {
            Self::LegacyRow8 => "legacy-row8",
            Self::WmmaCandidate => "gfx1201-wmma-scoped-default",
        }
    }

    const fn label(self) -> &'static str {
        self.argument()
    }

    const fn selector_environment(self) -> &'static str {
        match self {
            Self::LegacyRow8 => MXFP8_ROW8_ENV,
            Self::WmmaCandidate => MXFP8_SCOPED_DEFAULT_ENV,
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "legacy-row8" => Ok(Self::LegacyRow8),
            "gfx1201-wmma-scoped-default" => Ok(Self::WmmaCandidate),
            _ => Err(format!("unknown MXFP8 provider worker mode: {value}")),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct RepeatEvidence {
    primary_logit_digest: String,
    repeat_logit_digest: String,
    primary_token_digest: String,
    repeat_token_digest: String,
    bitwise_identical: bool,
    primary_dispatch: DispatchAuditSummary,
    repeat_dispatch: DispatchAuditSummary,
    dispatch_identity_repeated: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct WorkerCleanup {
    retryable_cleanup: usize,
    durable_quarantine: usize,
    final_cleanup_empty: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProviderWorkerOutput {
    provider_label: String,
    selector_environment: String,
    selector_value: String,
    artifact: ArtifactRun,
    repeat: RepeatEvidence,
    cleanup: WorkerCleanup,
}

#[derive(Debug, Serialize)]
struct ProviderIdentity {
    label: String,
    selector_environment: String,
    selector_value: String,
    identity_observability: &'static str,
    weight_encoding: String,
    sha256: String,
    size_bytes: u64,
    dispatch: DispatchAuditSummary,
    repeat: RepeatEvidence,
}

#[derive(Debug, Serialize)]
struct FirstLogitDivergence {
    case_id: String,
    phase: &'static str,
    position: usize,
    logit_index: usize,
    reference_value: f32,
    candidate_value: f32,
}

#[derive(Debug, Serialize)]
struct FirstTokenDivergence {
    case_id: String,
    phase: &'static str,
    position: usize,
    reference_top1: usize,
    candidate_top1: usize,
}

#[derive(Debug, Serialize)]
struct ProviderCleanupComparison {
    reference: WorkerCleanup,
    candidate: WorkerCleanup,
    all_workers_empty: bool,
}

#[derive(Debug, Serialize)]
struct ProviderReport {
    schema_version: &'static str,
    state: &'static str,
    comparison_mode: &'static str,
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
    same_artifact: bool,
    sequential_provider_processes: bool,
    reference_released_before_candidate: bool,
    provider_environment_isolated: bool,
    reference: ProviderIdentity,
    candidate: ProviderIdentity,
    first_logit_divergence: Option<FirstLogitDivergence>,
    first_token_divergence: Option<FirstTokenDivergence>,
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
    cleanup: ProviderCleanupComparison,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OutputReport {
    Artifact(Box<Report>),
    Provider(Box<ProviderReport>),
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

fn update_logit_digest(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

fn logit_digest(rows: &[LogitPair]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sllm-qwen35-mx-provider-logits-v1\0");
    hasher.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        hasher.update(b"prefill\0");
        update_logit_digest(&mut hasher, &row.prefill);
        hasher.update(b"decode\0");
        update_logit_digest(&mut hasher, &row.decode);
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn token_digest(rows: &[LogitPair]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sllm-qwen35-mx-provider-top1-v1\0");
    hasher.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        hasher.update((top1(&row.prefill) as u64).to_le_bytes());
        hasher.update((top1(&row.decode) as u64).to_le_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn rows_bitwise_identical(left: &[LogitPair], right: &[LogitPair]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.prefill.len() == right.prefill.len()
                && left.decode.len() == right.decode.len()
                && left
                    .prefill
                    .iter()
                    .zip(&right.prefill)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
                && left
                    .decode
                    .iter()
                    .zip(&right.decode)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
        })
}

fn first_logit_divergence(
    cases: &[PreparedCase],
    reference: &[LogitPair],
    candidate: &[LogitPair],
) -> Option<FirstLogitDivergence> {
    cases
        .iter()
        .zip(reference)
        .zip(candidate)
        .find_map(|((case, reference), candidate)| {
            [
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
            ]
            .into_iter()
            .find_map(|(phase, position, reference, candidate)| {
                reference
                    .iter()
                    .zip(candidate)
                    .enumerate()
                    .find(|(_, (reference, candidate))| reference.to_bits() != candidate.to_bits())
                    .map(
                        |(logit_index, (reference_value, candidate_value))| FirstLogitDivergence {
                            case_id: case.id.clone(),
                            phase,
                            position,
                            logit_index,
                            reference_value: *reference_value,
                            candidate_value: *candidate_value,
                        },
                    )
            })
        })
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
    let mut dispatch = DispatchAuditSummary {
        selected_backend: "hip".to_owned(),
        target: target.to_owned(),
        submission_count: 0,
        kernel_dispatch_count: 0,
        fallback_used: false,
        all_dispatches_hip: true,
        segment_count: 0,
        boundary_count: 0,
        mxfp8_wmma_dispatch_count: 0,
    };
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
        dispatch.submission_count = dispatch
            .submission_count
            .checked_add(audit.submission_count())
            .ok_or("submission count overflow")?;
        dispatch.kernel_dispatch_count = dispatch
            .kernel_dispatch_count
            .checked_add(audit.kernel_dispatch_count())
            .ok_or("dispatch count overflow")?;
        dispatch.fallback_used |= audit.fallback_used();
        dispatch.all_dispatches_hip &= audit.all_dispatches_hip();
        dispatch.segment_count = dispatch
            .segment_count
            .checked_add(audit.segment_count())
            .ok_or("segment count overflow")?;
        dispatch.boundary_count = dispatch
            .boundary_count
            .checked_add(audit.boundary_count())
            .ok_or("boundary count overflow")?;
        dispatch.mxfp8_wmma_dispatch_count = dispatch
            .mxfp8_wmma_dispatch_count
            .checked_add(
                audit.kernel_dispatch_count_for(MXFP8_WMMA_KERNEL_ID, MXFP8_WMMA_KERNEL_SYMBOL),
            )
            .ok_or("WMMA dispatch count overflow")?;
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
        dispatch,
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
        dispatches: reference.dispatch.kernel_dispatch_count,
    };
    let candidate_identity = ArtifactIdentity {
        weight_encoding: candidate.weight_encoding,
        sha256: candidate.artifact_sha256,
        size_bytes: candidate.artifact_size_bytes,
        dispatches: candidate.dispatch.kernel_dispatch_count,
    };
    Ok((
        rows,
        perplexity,
        memory,
        reference_identity,
        candidate_identity,
    ))
}

fn validate_reviewed_lock(path: &Path) -> Result<sllm_core::ModelLock, String> {
    let lock = read_model_lock(path).map_err(|error| format!("read model lock: {error}"))?;
    if lock.model.repo_id != QWEN35_4B_REPO_ID
        || lock.model.resolved_revision != QWEN35_4B_REVISION
        || lock.fingerprint() != QWEN35_4B_FINGERPRINT
    {
        return Err("quality runner requires the reviewed Qwen3.5-4B lock".to_owned());
    }
    Ok(lock)
}

fn validate_provider_worker_environment(provider: ProviderMode) -> Result<(), String> {
    for name in PROVIDER_ENVIRONMENTS {
        let value = env::var_os(name);
        if name == provider.selector_environment() {
            if value.as_deref() != Some(std::ffi::OsStr::new("1")) {
                return Err(format!(
                    "provider worker requires {name}=1 before HIP initialization"
                ));
            }
        } else if value.is_some() {
            return Err(format!(
                "competing provider environment {name} must be absent in provider worker"
            ));
        }
    }
    Ok(())
}

fn validate_provider_dispatch(
    provider: ProviderMode,
    dispatch: &DispatchAuditSummary,
) -> Result<(), String> {
    let valid = match provider {
        ProviderMode::LegacyRow8 => dispatch.mxfp8_wmma_dispatch_count == 0,
        ProviderMode::WmmaCandidate => dispatch.mxfp8_wmma_dispatch_count > 0,
    };
    if !valid {
        return Err(format!(
            "{} provider observed {} dispatches of kernel {} ({})",
            provider.label(),
            dispatch.mxfp8_wmma_dispatch_count,
            MXFP8_WMMA_KERNEL_ID,
            MXFP8_WMMA_KERNEL_SYMBOL
        ));
    }
    Ok(())
}

fn run_provider_worker(arguments: &[String]) -> Result<(ProviderWorkerOutput, PathBuf), String> {
    if arguments.len() != 8 {
        return Err(
            "internal usage: --provider-worker MODEL_LOCK DATASET_JSON DEVICE_INDEX TARGET MX_GGUF MX_DERIVED_LOCK PROVIDER OUTPUT_JSON"
                .to_owned(),
        );
    }
    let model_lock_path = PathBuf::from(&arguments[0]);
    let dataset_path = PathBuf::from(&arguments[1]);
    let device_index = arguments[2]
        .parse::<u32>()
        .map_err(|_| "device index must be u32".to_owned())?;
    let target = arguments[3].clone();
    if target != "gfx1201" {
        return Err("provider comparison requires exact gfx1201".to_owned());
    }
    let gguf_path = PathBuf::from(&arguments[4]);
    let derived_lock_path = PathBuf::from(&arguments[5]);
    let provider = ProviderMode::parse(&arguments[6])?;
    let output_path = PathBuf::from(&arguments[7]);
    if output_path.exists() {
        return Err("provider worker output already exists".to_owned());
    }
    validate_provider_worker_environment(provider)?;
    let (_, cases) = load_dataset(&dataset_path)?;
    let lock = validate_reviewed_lock(&model_lock_path)?;
    let backend = HipBackend::connect().map_err(|error| format!("connect HIP: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| format!("session request: {error}"))?,
        )
        .map_err(|error| format!("open HIP session: {error}"))?;

    let execution = (|| {
        let primary = execute_artifact(
            &session,
            &lock,
            &gguf_path,
            &derived_lock_path,
            &cases,
            &target,
            true,
        )?;
        if session.memory_snapshot().current_bytes() != 0 {
            return Err("primary provider resident remained before repeat".to_owned());
        }
        let repeated = execute_artifact(
            &session,
            &lock,
            &gguf_path,
            &derived_lock_path,
            &cases,
            &target,
            true,
        )?;
        if session.memory_snapshot().current_bytes() != 0 {
            return Err("repeat provider resident remained after measurement".to_owned());
        }
        if primary.weight_encoding != repeated.weight_encoding
            || primary.artifact_sha256 != repeated.artifact_sha256
            || primary.artifact_size_bytes != repeated.artifact_size_bytes
        {
            return Err("provider repeat did not use the same MXFP8 artifact".to_owned());
        }
        validate_provider_dispatch(provider, &primary.dispatch)?;
        validate_provider_dispatch(provider, &repeated.dispatch)?;
        let primary_logit_digest = logit_digest(&primary.rows);
        let repeat_logit_digest = logit_digest(&repeated.rows);
        let primary_token_digest = token_digest(&primary.rows);
        let repeat_token_digest = token_digest(&repeated.rows);
        let bitwise_identical = rows_bitwise_identical(&primary.rows, &repeated.rows);
        if !bitwise_identical
            || primary_logit_digest != repeat_logit_digest
            || primary_token_digest != repeat_token_digest
        {
            return Err(format!(
                "{} provider repeat was not bitwise deterministic: logits {primary_logit_digest} versus {repeat_logit_digest}, tokens {primary_token_digest} versus {repeat_token_digest}",
                provider.label()
            ));
        }
        let dispatch_identity_repeated = primary.dispatch == repeated.dispatch;
        if !dispatch_identity_repeated {
            return Err(format!(
                "{} provider aggregate dispatch identity changed between repeats",
                provider.label()
            ));
        }
        let repeat = RepeatEvidence {
            primary_logit_digest,
            repeat_logit_digest,
            primary_token_digest,
            repeat_token_digest,
            bitwise_identical,
            primary_dispatch: primary.dispatch.clone(),
            repeat_dispatch: repeated.dispatch,
            dispatch_identity_repeated,
        };
        Ok(ProviderWorkerOutput {
            provider_label: provider.label().to_owned(),
            selector_environment: provider.selector_environment().to_owned(),
            selector_value: "1".to_owned(),
            artifact: primary,
            repeat,
            cleanup: WorkerCleanup {
                retryable_cleanup: 0,
                durable_quarantine: 0,
                final_cleanup_empty: false,
            },
        })
    })();
    let mut output = match execution {
        Ok(output) => output,
        Err(error) => {
            let _ = session.shutdown(SHUTDOWN_TIMEOUT);
            return Err(error);
        }
    };
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("shutdown provider worker session: {error}"))?;
    let empty = cleanup.retryable_cleanup == 0
        && cleanup.durable_quarantine == 0
        && session.memory_snapshot().current_bytes() == 0;
    if !empty {
        return Err(format!("provider worker cleanup was nonzero: {cleanup:?}"));
    }
    output.cleanup = WorkerCleanup {
        retryable_cleanup: cleanup.retryable_cleanup,
        durable_quarantine: cleanup.durable_quarantine,
        final_cleanup_empty: empty,
    };
    Ok((output, output_path))
}

struct TemporaryWorkerOutput(PathBuf);

impl Drop for TemporaryWorkerOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn worker_output_path(provider: ProviderMode) -> Result<PathBuf, String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("read system time: {error}"))?
        .as_nanos();
    let path = env::temp_dir().join(format!(
        "sllm-qwen35-mx-provider-{}-{}-{timestamp}.json",
        std::process::id(),
        provider.argument()
    ));
    if path.exists() {
        return Err(format!(
            "provider worker temporary output already exists: {}",
            path.display()
        ));
    }
    Ok(path)
}

#[allow(clippy::too_many_arguments)]
fn spawn_provider_worker(
    executable: &Path,
    model_lock_path: &Path,
    dataset_path: &Path,
    device_index: u32,
    target: &str,
    gguf_path: &Path,
    derived_lock_path: &Path,
    provider: ProviderMode,
) -> Result<ProviderWorkerOutput, String> {
    let temporary = TemporaryWorkerOutput(worker_output_path(provider)?);
    let mut command = Command::new(executable);
    command
        .arg(PROVIDER_WORKER_MODE)
        .arg(model_lock_path)
        .arg(dataset_path)
        .arg(device_index.to_string())
        .arg(target)
        .arg(gguf_path)
        .arg(derived_lock_path)
        .arg(provider.argument())
        .arg(&temporary.0);
    for name in PROVIDER_ENVIRONMENTS {
        command.env_remove(name);
    }
    command.env(provider.selector_environment(), "1");
    let mut child = command
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("start {} provider worker: {error}", provider.label()))?;
    let deadline = Instant::now()
        .checked_add(PROVIDER_PROCESS_TIMEOUT)
        .ok_or("provider worker deadline overflow")?;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("wait {} provider worker: {error}", provider.label()))?
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "{} provider worker exceeded {} seconds and was terminated",
                provider.label(),
                PROVIDER_PROCESS_TIMEOUT.as_secs()
            ));
        }
        thread::sleep(Duration::from_millis(50));
    };
    if !status.success() {
        return Err(format!(
            "{} provider worker failed with {}",
            provider.label(),
            status
        ));
    }
    let bytes = fs::read(&temporary.0)
        .map_err(|error| format!("read {} provider worker output: {error}", provider.label()))?;
    let output: ProviderWorkerOutput = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {} provider worker output: {error}", provider.label()))?;
    if output.provider_label != provider.label()
        || output.selector_environment != provider.selector_environment()
        || output.selector_value != "1"
        || !output.cleanup.final_cleanup_empty
        || output.cleanup.retryable_cleanup != 0
        || output.cleanup.durable_quarantine != 0
    {
        return Err(format!(
            "{} provider worker reported an invalid identity or cleanup contract",
            provider.label()
        ));
    }
    Ok(output)
}

fn run_provider_comparison(arguments: &[String]) -> Result<(ProviderReport, PathBuf), String> {
    if arguments.len() != 7 {
        return Err(
            "usage: --provider-compare MODEL_LOCK DATASET_JSON DEVICE_INDEX gfx1201 MX_GGUF MX_DERIVED_LOCK OUTPUT_JSON"
                .to_owned(),
        );
    }
    let model_lock_path = PathBuf::from(&arguments[0]);
    let dataset_path = PathBuf::from(&arguments[1]);
    let device_index = arguments[2]
        .parse::<u32>()
        .map_err(|_| "device index must be u32".to_owned())?;
    let target = arguments[3].clone();
    if target != "gfx1201" {
        return Err("provider comparison requires exact gfx1201".to_owned());
    }
    let gguf_path = PathBuf::from(&arguments[4]);
    let derived_lock_path = PathBuf::from(&arguments[5]);
    let output_path = PathBuf::from(&arguments[6]);
    if output_path.exists() {
        return Err("output already exists".to_owned());
    }
    let (dataset_sha256, cases) = load_dataset(&dataset_path)?;
    let _ = validate_reviewed_lock(&model_lock_path)?;
    let executable = env::current_exe().map_err(|error| format!("locate binary: {error}"))?;
    let executable_sha256 = format!(
        "sha256:{}",
        sha256(&fs::read(&executable).map_err(|error| format!("read binary: {error}"))?)
    );

    // Each provider owns a fresh child process. The baseline worker exits only
    // after two resident releases and zero session cleanup; only then is the
    // candidate worker started. Child-only environments avoid mutating the
    // parent process environment while HIP may own internal threads.
    let reference_worker = spawn_provider_worker(
        &executable,
        &model_lock_path,
        &dataset_path,
        device_index,
        &target,
        &gguf_path,
        &derived_lock_path,
        ProviderMode::LegacyRow8,
    )?;
    let candidate_worker = spawn_provider_worker(
        &executable,
        &model_lock_path,
        &dataset_path,
        device_index,
        &target,
        &gguf_path,
        &derived_lock_path,
        ProviderMode::WmmaCandidate,
    )?;
    let same_artifact = reference_worker.artifact.weight_encoding
        == candidate_worker.artifact.weight_encoding
        && reference_worker.artifact.artifact_sha256 == candidate_worker.artifact.artifact_sha256
        && reference_worker.artifact.artifact_size_bytes
            == candidate_worker.artifact.artifact_size_bytes;
    if !same_artifact {
        return Err("provider workers did not execute the same MXFP8 artifact".to_owned());
    }
    let first_logit_divergence = first_logit_divergence(
        &cases,
        &reference_worker.artifact.rows,
        &candidate_worker.artifact.rows,
    );

    let ProviderWorkerOutput {
        provider_label: reference_label,
        selector_environment: reference_selector,
        selector_value: reference_selector_value,
        artifact: reference_artifact,
        repeat: reference_repeat,
        cleanup: reference_cleanup,
    } = reference_worker;
    let ProviderWorkerOutput {
        provider_label: candidate_label,
        selector_environment: candidate_selector,
        selector_value: candidate_selector_value,
        artifact: candidate_artifact,
        repeat: candidate_repeat,
        cleanup: candidate_cleanup,
    } = candidate_worker;
    let reference_identity = ProviderIdentity {
        label: reference_label,
        selector_environment: reference_selector,
        selector_value: reference_selector_value,
        identity_observability: "isolated prepare-time selector plus exact Qwen audit count for kernel ID 31 and matmul.mxfp8.w8a8.e4m3.block32.prefill.wmma128x64x32.v2",
        weight_encoding: reference_artifact.weight_encoding.clone(),
        sha256: reference_artifact.artifact_sha256.clone(),
        size_bytes: reference_artifact.artifact_size_bytes,
        dispatch: reference_artifact.dispatch.clone(),
        repeat: reference_repeat,
    };
    let candidate_identity = ProviderIdentity {
        label: candidate_label,
        selector_environment: candidate_selector,
        selector_value: candidate_selector_value,
        identity_observability: "isolated prepare-time selector plus exact Qwen audit count for kernel ID 31 and matmul.mxfp8.w8a8.e4m3.block32.prefill.wmma128x64x32.v2",
        weight_encoding: candidate_artifact.weight_encoding.clone(),
        sha256: candidate_artifact.artifact_sha256.clone(),
        size_bytes: candidate_artifact.artifact_size_bytes,
        dispatch: candidate_artifact.dispatch.clone(),
        repeat: candidate_repeat,
    };
    let (rows, perplexity, memory, _, _) =
        compare_runs(&cases, reference_artifact, candidate_artifact)?;
    let mut klds = rows
        .iter()
        .map(|row| row.kld_reference_to_candidate)
        .collect::<Vec<_>>();
    klds.sort_by(f64::total_cmp);
    if klds.is_empty() || klds.iter().any(|value| !value.is_finite()) {
        return Err("KLD output is empty or non-finite".to_owned());
    }
    let top1_matches = rows.iter().filter(|row| row.top1_match).count();
    let first_token_divergence =
        rows.iter()
            .find(|row| !row.top1_match)
            .map(|row| FirstTokenDivergence {
                case_id: row.case_id.clone(),
                phase: row.phase,
                position: row.position,
                reference_top1: row.reference_top1,
                candidate_top1: row.candidate_top1,
            });
    let all_workers_empty = reference_cleanup.final_cleanup_empty
        && candidate_cleanup.final_cleanup_empty
        && reference_cleanup.retryable_cleanup == 0
        && reference_cleanup.durable_quarantine == 0
        && candidate_cleanup.retryable_cleanup == 0
        && candidate_cleanup.durable_quarantine == 0;
    if !all_workers_empty {
        return Err("provider comparison worker cleanup was nonzero".to_owned());
    }
    Ok((
        ProviderReport {
            schema_version: "sllm-qwen35-mx-provider-quality-v1",
            state: "PASS",
            comparison_mode: "same-mxfp8-artifact-provider-comparison",
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
            same_artifact,
            sequential_provider_processes: true,
            reference_released_before_candidate: true,
            provider_environment_isolated: true,
            reference: reference_identity,
            candidate: candidate_identity,
            first_logit_divergence,
            first_token_divergence,
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
            cleanup: ProviderCleanupComparison {
                reference: reference_cleanup,
                candidate: candidate_cleanup,
                all_workers_empty,
            },
        },
        output_path,
    ))
}

fn valid_target(target: &str) -> bool {
    matches!(target, "gfx1030" | "gfx1201")
}

fn run_artifact_comparison(arguments: &[String]) -> Result<(Report, PathBuf), String> {
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
    let lock = validate_reviewed_lock(&model_lock_path)?;
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

fn run(arguments: &[String]) -> Result<(OutputReport, PathBuf), String> {
    if arguments.first().map(String::as_str) == Some(PROVIDER_COMPARE_MODE) {
        let (report, output) = run_provider_comparison(&arguments[1..])?;
        Ok((OutputReport::Provider(Box::new(report)), output))
    } else {
        let (report, output) = run_artifact_comparison(arguments)?;
        Ok((OutputReport::Artifact(Box::new(report)), output))
    }
}

fn publish<T: Serialize>(report: &T, output: &Path) -> Result<String, String> {
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
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().map(String::as_str) == Some(PROVIDER_WORKER_MODE) {
        return match run_provider_worker(&arguments[1..]) {
            Ok((report, output)) => match publish(&report, &output) {
                Ok(digest) => {
                    println!("{} {digest}", output.display());
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("MX provider worker publication failed: {error}");
                    ExitCode::from(2)
                }
            },
            Err(error) => {
                eprintln!("MX provider worker failed: {error}");
                ExitCode::FAILURE
            }
        };
    }
    match run(&arguments) {
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

    #[test]
    fn provider_modes_pin_the_reviewed_selector_environments() {
        assert_eq!(
            ProviderMode::parse("legacy-row8").unwrap(),
            ProviderMode::LegacyRow8
        );
        assert_eq!(
            ProviderMode::LegacyRow8.selector_environment(),
            "SLLM_MXFP8_PREFILL_FORCE_ROW8"
        );
        assert_eq!(
            ProviderMode::parse("gfx1201-wmma-scoped-default").unwrap(),
            ProviderMode::WmmaCandidate
        );
        assert_eq!(
            ProviderMode::WmmaCandidate.selector_environment(),
            "SLLM_MXFP8_PREFILL_SCOPED_DEFAULT_GFX1201"
        );
        assert!(ProviderMode::parse("default").is_err());
    }

    #[test]
    fn provider_repeat_digests_and_first_divergence_use_logit_bits() {
        let reference = vec![LogitPair {
            prefill: vec![1.0, 2.0, 3.0],
            decode: vec![4.0, 5.0, 6.0],
        }];
        let identical = vec![LogitPair {
            prefill: vec![1.0, 2.0, 3.0],
            decode: vec![4.0, 5.0, 6.0],
        }];
        assert!(rows_bitwise_identical(&reference, &identical));
        assert_eq!(logit_digest(&reference), logit_digest(&identical));
        assert_eq!(token_digest(&reference), token_digest(&identical));

        let candidate = vec![LogitPair {
            prefill: vec![1.0, 2.5, 3.0],
            decode: vec![4.0, 7.0, 6.0],
        }];
        let cases = vec![PreparedCase {
            id: "case".to_owned(),
            tokens: vec![1, 2, 3],
            expected_next: 0,
        }];
        assert!(!rows_bitwise_identical(&reference, &candidate));
        assert_ne!(logit_digest(&reference), logit_digest(&candidate));
        let divergence = first_logit_divergence(&cases, &reference, &candidate).unwrap();
        assert_eq!(divergence.case_id, "case");
        assert_eq!(divergence.phase, "prefill");
        assert_eq!(divergence.position, 2);
        assert_eq!(divergence.logit_index, 1);
        assert_eq!(divergence.reference_value, 2.0);
        assert_eq!(divergence.candidate_value, 2.5);
    }
}
