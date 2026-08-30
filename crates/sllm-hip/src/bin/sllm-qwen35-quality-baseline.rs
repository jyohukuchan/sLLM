//! Phase 46 exact-HIP FP16-only Qwen3.5-4B quality baseline.
//!
//! The input dataset is an offline, project-authored generator specification.
//! No prompt text, model candidate, quantization sidecar, or CPU fallback is
//! accepted by this runner.  A short baseline pass is paired with the first
//! measured repeat so logit comparison never requires retaining a full
//! vocabulary matrix in host memory.

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
    Backend, ExecutionSession, ExecutionSessionRequest, QWEN35_4B_FINGERPRINT, QWEN35_4B_REPO_ID,
    QWEN35_4B_REVISION, QwenExecutionAudit, QwenExecutionRequest, QwenResidentModel, TensorDType,
    build_qwen35_graph, build_verified_weight_load_plan, read_model_lock,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(180);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const DATASET_SCHEMA: &str = "sllm-phase46-kv-quality-dataset-v1";
const DATASET_ID: &str = "phase46-kv-quality-baseline-v1";
const DATASET_LICENSE: &str = "CC0-1.0";
const DATASET_PROVENANCE: &str = "Project-authored deterministic token-ID boundary fixtures; contains no copied prompt or response text.";
const TOKEN_GENERATOR: &str = "token[i] = 1 + ((start + i * step + seed) mod 200000)";
const DATASET_SHA256: &str =
    "sha256:a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d";
const MAX_DATASET_BYTES: usize = 1 << 20;
const MAX_CASES: usize = 32;
const MAX_CASE_LENGTH: usize = 513;
const MAX_REPEATS: u32 = 16;
const VOCAB_SIZE: usize = sllm_core::QWEN35_VOCAB_SIZE;

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
    coverage: Coverage,
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Coverage {
    kv_planes: Vec<String>,
    layers: Vec<u32>,
    kv_heads: Vec<u32>,
    position_bands: Vec<String>,
    boundaries: Vec<usize>,
}

#[derive(Clone)]
struct PreparedCase {
    id: String,
    tokens: Vec<i32>,
    expected_next: i32,
    band: String,
    block_tail: bool,
    token_digest: String,
}

#[derive(Default)]
struct RunAccumulator {
    loss_sum: f64,
    token_count: usize,
    top1_correct: usize,
    dispatches: u64,
    fallback_used: bool,
    all_dispatches_hip: bool,
}

#[derive(Serialize)]
struct RepeatReport {
    repeat: u32,
    token_count: usize,
    loss_sum: f64,
    mean_nll: f64,
    perplexity: f64,
    top1_correct: usize,
    top1_score: f64,
    task_score: f64,
    hip_dispatches: u64,
}

#[derive(Serialize, Default)]
struct BaselineComparison {
    sample_count: usize,
    top1_matches: usize,
    top1_agreement: f64,
    kld_baseline_to_first: f64,
    kld_mean: f64,
    kld_max: f64,
    max_logit_delta: f32,
    first_divergence_position: Option<usize>,
}

#[derive(Serialize)]
struct CaseReport {
    id: String,
    length: usize,
    band: String,
    block_tail: bool,
    token_digest: String,
    expected_next: i32,
    baseline: RepeatReport,
    repeats: Vec<RepeatReport>,
    baseline_vs_first: BaselineComparison,
}

#[derive(Serialize)]
struct LongContextCoverage {
    sample_count: usize,
    position_min: usize,
    position_max: usize,
    early: usize,
    middle: usize,
    tail: usize,
    coverage_ratio: f64,
    key_samples: usize,
    value_samples: usize,
    kv_planes: Vec<String>,
    layers: Vec<u32>,
    kv_heads: Vec<u32>,
    boundaries: Vec<usize>,
    block_tail_samples: usize,
}

#[derive(Serialize)]
struct CleanupReport {
    model_ready_current_bytes: u64,
    pre_shutdown_current_bytes: u64,
    final_current_bytes: u64,
    final_request_state_bytes: u64,
    final_workspace_bytes: u64,
    retryable_cleanup: usize,
    durable_quarantine: usize,
    empty: bool,
}

#[derive(Serialize)]
struct Report {
    #[serde(rename = "$schema")]
    schema: &'static str,
    schema_version: &'static str,
    state: &'static str,
    model_repo_id: &'static str,
    model_revision: &'static str,
    model_fingerprint: &'static str,
    weight_dtype: &'static str,
    kv_encoding: &'static str,
    target: String,
    device_index: u32,
    executable_sha256: String,
    dataset_id: &'static str,
    dataset_sha256: &'static str,
    dataset_seed: u64,
    dataset_case_count: usize,
    repeats: u32,
    baseline_pass: bool,
    baseline_vs_first: BaselineComparison,
    long_context: LongContextCoverage,
    cases: Vec<CaseReport>,
    cleanup: CleanupReport,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn token_digest(tokens: &[i32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn generated_token(case: &DatasetCase, seed: u64, index: usize) -> Result<i32, String> {
    let index = u64::try_from(index).map_err(|_| "token index exceeds u64".to_owned())?;
    let product = index
        .checked_mul(case.step)
        .ok_or_else(|| format!("{} token generator overflowed", case.id))?;
    let value = case
        .start
        .checked_add(product)
        .and_then(|value| value.checked_add(seed))
        .ok_or_else(|| format!("{} token generator overflowed", case.id))?
        % 200_000;
    i32::try_from(value + 1).map_err(|_| format!("{} token does not fit i32", case.id))
}

fn parse_dataset(path: &Path) -> Result<(Dataset, Vec<PreparedCase>), String> {
    let bytes = fs::read(path).map_err(|error| format!("read dataset: {error}"))?;
    if bytes.is_empty() || bytes.len() > MAX_DATASET_BYTES {
        return Err(format!("dataset bytes are outside 1..={MAX_DATASET_BYTES}"));
    }
    let digest = sha256_bytes(&bytes);
    if digest != DATASET_SHA256 {
        return Err(format!(
            "dataset digest mismatch: expected {DATASET_SHA256}, got {digest}"
        ));
    }
    let dataset: Dataset =
        serde_json::from_slice(&bytes).map_err(|error| format!("parse dataset: {error}"))?;
    if dataset.schema_version != DATASET_SCHEMA
        || dataset.dataset_id != DATASET_ID
        || dataset.license != DATASET_LICENSE
        || dataset.provenance != DATASET_PROVENANCE
        || dataset.token_generator != TOKEN_GENERATOR
        || dataset.sample_order != "listed"
        || dataset.seed != 1729
    {
        return Err(
            "dataset identity or generator specification is not the reviewed fixture".to_owned(),
        );
    }
    if dataset.cases.len() != dataset.coverage.boundaries.len()
        || dataset.cases.is_empty()
        || dataset.cases.len() > MAX_CASES
    {
        return Err("dataset case count is outside the bounded range".to_owned());
    }
    if dataset.coverage.kv_planes != ["K", "V"]
        || dataset.coverage.layers != [0, 13, 27]
        || dataset.coverage.kv_heads != [0, 1, 3]
        || dataset.coverage.position_bands != ["early", "middle", "tail"]
        || dataset.coverage.boundaries != [1, 15, 16, 17, 255, 256, 257, 511, 512, 513]
    {
        return Err("dataset coverage metadata is not the reviewed fixture".to_owned());
    }

    let mut prepared = Vec::with_capacity(dataset.cases.len());
    let mut ids = std::collections::BTreeSet::new();
    for (case_index, case) in dataset.cases.iter().enumerate() {
        if !ids.insert(case.id.clone())
            || case.length == 0
            || case.length > MAX_CASE_LENGTH
            || case.step == 0
            || case.length != dataset.coverage.boundaries[case_index]
            || !matches!(case.band.as_str(), "early" | "middle" | "tail")
        {
            return Err(format!("invalid bounded dataset case {}", case.id));
        }
        let mut tokens = Vec::with_capacity(case.length);
        for index in 0..case.length {
            tokens.push(generated_token(case, dataset.seed, index)?);
        }
        let expected = generated_token(case, dataset.seed, case.length)?;
        if case.expected_next != expected || case.expected_next <= 0 {
            return Err(format!(
                "{} expected_next disagrees with generator",
                case.id
            ));
        }
        let token_digest = token_digest(&tokens);
        prepared.push(PreparedCase {
            id: case.id.clone(),
            tokens,
            expected_next: case.expected_next,
            band: case.band.clone(),
            block_tail: case.block_tail,
            token_digest,
        });
    }
    Ok((dataset, prepared))
}

fn top1(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}

fn nll(values: &[f32], target: i32) -> Result<f64, String> {
    let target = usize::try_from(target).map_err(|_| "target token is negative".to_owned())?;
    if target >= values.len() {
        return Err("target token is outside the vocabulary".to_owned());
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err("logits contain no finite maximum".to_owned());
    }
    let sum = values
        .iter()
        .map(|value| f64::from(*value - max).exp())
        .sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("logit softmax sum is non-finite".to_owned());
    }
    let value = f64::from(max) + sum.ln() - f64::from(values[target]);
    if !value.is_finite() || value < 0.0 {
        return Err("next-token loss is non-finite or negative".to_owned());
    }
    Ok(value)
}

fn kld(reference: &[f32], candidate: &[f32]) -> Result<f64, String> {
    if reference.len() != VOCAB_SIZE || candidate.len() != VOCAB_SIZE {
        return Err("logit row does not have the reviewed vocabulary width".to_owned());
    }
    let reference_max = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let candidate_max = candidate.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !reference_max.is_finite() || !candidate_max.is_finite() {
        return Err("KLD input contains no finite maximum".to_owned());
    }
    let reference_exp = reference
        .iter()
        .map(|value| f64::from(*value - reference_max).exp())
        .collect::<Vec<_>>();
    let candidate_exp = candidate
        .iter()
        .map(|value| f64::from(*value - candidate_max).exp())
        .collect::<Vec<_>>();
    let reference_sum = reference_exp.iter().sum::<f64>();
    let candidate_sum = candidate_exp.iter().sum::<f64>();
    if !reference_sum.is_finite()
        || reference_sum <= 0.0
        || !candidate_sum.is_finite()
        || candidate_sum <= 0.0
    {
        return Err("KLD softmax sums are non-finite".to_owned());
    }
    let value = reference_exp
        .iter()
        .zip(candidate_exp)
        .map(|(reference, candidate)| {
            let p = reference / reference_sum;
            let q = candidate / candidate_sum;
            if p == 0.0 { 0.0 } else { p * (p / q).ln() }
        })
        .sum::<f64>();
    if !value.is_finite() || value < 0.0 {
        return Err("KLD is non-finite or negative".to_owned());
    }
    Ok(value)
}

fn validate_audit(audit: &QwenExecutionAudit, target: &str) -> Result<(), String> {
    if audit.selected_backend() != "hip"
        || audit.target() != target
        || audit.fallback_used()
        || !audit.all_dispatches_hip()
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
    {
        return Err(format!(
            "execution was not exact HIP/no-fallback: {audit:?}"
        ));
    }
    Ok(())
}

fn add_row(accumulator: &mut RunAccumulator, logits: &[f32], target: i32) -> Result<usize, String> {
    if logits.len() != VOCAB_SIZE || logits.iter().any(|value| !value.is_finite()) {
        return Err("execution produced a non-finite or truncated logit row".to_owned());
    }
    let loss = nll(logits, target)?;
    accumulator.loss_sum += loss;
    accumulator.token_count = accumulator
        .token_count
        .checked_add(1)
        .ok_or_else(|| "loss token count overflowed".to_owned())?;
    let winner = top1(logits);
    if winner == usize::try_from(target).map_err(|_| "target token is negative".to_owned())? {
        accumulator.top1_correct += 1;
    }
    Ok(winner)
}

fn finish_metrics(accumulator: &RunAccumulator, repeat: u32) -> Result<RepeatReport, String> {
    if accumulator.token_count == 0
        || !accumulator.loss_sum.is_finite()
        || accumulator.loss_sum < 0.0
        || !accumulator.all_dispatches_hip
        || accumulator.fallback_used
    {
        return Err("repeat produced zero/non-finite metrics or fallback".to_owned());
    }
    let mean_nll = accumulator.loss_sum / accumulator.token_count as f64;
    let perplexity = mean_nll.exp();
    let top1_score = accumulator.top1_correct as f64 / accumulator.token_count as f64;
    if !mean_nll.is_finite() || !perplexity.is_finite() || !top1_score.is_finite() {
        return Err("repeat quality metric is non-finite".to_owned());
    }
    Ok(RepeatReport {
        repeat,
        token_count: accumulator.token_count,
        loss_sum: accumulator.loss_sum,
        mean_nll,
        perplexity,
        top1_correct: accumulator.top1_correct,
        top1_score,
        task_score: top1_score,
        hip_dispatches: accumulator.dispatches,
    })
}

fn process_request(
    request: &mut QwenExecutionRequest,
    case: &PreparedCase,
    target: &str,
) -> Result<RunAccumulator, String> {
    let mut accumulator = RunAccumulator {
        all_dispatches_hip: true,
        ..RunAccumulator::default()
    };
    let output = request
        .prefill_with_last_logits(&case.tokens)
        .map_err(|error| format!("prefill {}: {error}", case.id))?;
    let logits = output
        .last_logits()
        .ok_or_else(|| format!("{} did not publish full logits", case.id))?;
    add_row(&mut accumulator, logits, case.expected_next)?;
    let decode_output = request
        .decode_with_last_logits(case.expected_next)
        .map_err(|error| format!("decode continuation {}: {error}", case.id))?;
    let decode_logits = decode_output
        .last_logits()
        .ok_or_else(|| format!("{} decode did not publish full logits", case.id))?;
    if decode_logits.len() != VOCAB_SIZE || decode_logits.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{} decode produced a non-finite or truncated logit row",
            case.id
        ));
    }
    let audit = request
        .audit_snapshot()
        .map_err(|error| format!("audit {}: {error}", case.id))?;
    validate_audit(&audit, target)?;
    accumulator.dispatches = audit.kernel_dispatch_count();
    accumulator.fallback_used = audit.fallback_used();
    accumulator.all_dispatches_hip = audit.all_dispatches_hip();
    Ok(accumulator)
}

fn compare_requests(
    baseline: &mut QwenExecutionRequest,
    first: &mut QwenExecutionRequest,
    case: &PreparedCase,
    target: &str,
) -> Result<(RunAccumulator, RunAccumulator, BaselineComparison), String> {
    let mut baseline_metrics = RunAccumulator {
        all_dispatches_hip: true,
        ..RunAccumulator::default()
    };
    let mut first_metrics = RunAccumulator {
        all_dispatches_hip: true,
        ..RunAccumulator::default()
    };
    let mut comparison = BaselineComparison::default();
    let mut kld_sum = 0.0_f64;
    let baseline_output = baseline
        .prefill_with_last_logits(&case.tokens)
        .map_err(|error| format!("baseline prefill {}: {error}", case.id))?;
    let first_output = first
        .prefill_with_last_logits(&case.tokens)
        .map_err(|error| format!("first prefill {}: {error}", case.id))?;
    let baseline_logits = baseline_output
        .last_logits()
        .ok_or_else(|| format!("baseline {} did not publish full logits", case.id))?;
    let first_logits = first_output
        .last_logits()
        .ok_or_else(|| format!("first {} did not publish full logits", case.id))?;
    let baseline_top1 = add_row(&mut baseline_metrics, baseline_logits, case.expected_next)?;
    let first_top1 = add_row(&mut first_metrics, first_logits, case.expected_next)?;
    let row_kld = kld(baseline_logits, first_logits)?;
    let row_delta = baseline_logits
        .iter()
        .zip(first_logits)
        .map(|(left, right)| (*left - *right).abs())
        .fold(0.0_f32, f32::max);
    comparison.sample_count = 1;
    comparison.top1_matches = usize::from(baseline_top1 == first_top1);
    comparison.first_divergence_position =
        (baseline_top1 != first_top1).then_some(case.tokens.len().saturating_sub(1));
    kld_sum += row_kld;
    comparison.kld_max = row_kld;
    comparison.max_logit_delta = row_delta;
    let baseline_decode = baseline
        .decode_with_last_logits(case.expected_next)
        .map_err(|error| format!("baseline decode continuation {}: {error}", case.id))?;
    let first_decode = first
        .decode_with_last_logits(case.expected_next)
        .map_err(|error| format!("first decode continuation {}: {error}", case.id))?;
    let baseline_decode_logits = baseline_decode
        .last_logits()
        .ok_or_else(|| format!("baseline {} decode did not publish full logits", case.id))?;
    let first_decode_logits = first_decode
        .last_logits()
        .ok_or_else(|| format!("first {} decode did not publish full logits", case.id))?;
    if baseline_decode_logits.len() != VOCAB_SIZE
        || first_decode_logits.len() != VOCAB_SIZE
        || baseline_decode_logits
            .iter()
            .any(|value| !value.is_finite())
        || first_decode_logits.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "{} decode comparison produced a non-finite or truncated logit row",
            case.id
        ));
    }
    let baseline_decode_top1 = top1(baseline_decode_logits);
    let first_decode_top1 = top1(first_decode_logits);
    let decode_kld = kld(baseline_decode_logits, first_decode_logits)?;
    let decode_delta = baseline_decode_logits
        .iter()
        .zip(first_decode_logits)
        .map(|(left, right)| (*left - *right).abs())
        .fold(0.0_f32, f32::max);
    comparison.sample_count += 1;
    comparison.top1_matches += usize::from(baseline_decode_top1 == first_decode_top1);
    if comparison.first_divergence_position.is_none() && baseline_decode_top1 != first_decode_top1 {
        comparison.first_divergence_position = Some(case.tokens.len());
    }
    kld_sum += decode_kld;
    comparison.kld_max = comparison.kld_max.max(decode_kld);
    comparison.max_logit_delta = comparison.max_logit_delta.max(decode_delta);
    let baseline_audit = baseline
        .audit_snapshot()
        .map_err(|error| format!("baseline audit {}: {error}", case.id))?;
    let first_audit = first
        .audit_snapshot()
        .map_err(|error| format!("first audit {}: {error}", case.id))?;
    validate_audit(&baseline_audit, target)?;
    validate_audit(&first_audit, target)?;
    baseline_metrics.dispatches = baseline_audit.kernel_dispatch_count();
    baseline_metrics.fallback_used = baseline_audit.fallback_used();
    baseline_metrics.all_dispatches_hip = baseline_audit.all_dispatches_hip();
    first_metrics.dispatches = first_audit.kernel_dispatch_count();
    first_metrics.fallback_used = first_audit.fallback_used();
    first_metrics.all_dispatches_hip = first_audit.all_dispatches_hip();
    if comparison.sample_count == 0 {
        return Err("baseline comparison selected zero rows".to_owned());
    }
    comparison.kld_mean = kld_sum / comparison.sample_count as f64;
    comparison.kld_baseline_to_first = comparison.kld_mean;
    comparison.top1_agreement = comparison.top1_matches as f64 / comparison.sample_count as f64;
    if !comparison.kld_mean.is_finite()
        || !comparison.kld_max.is_finite()
        || !comparison.top1_agreement.is_finite()
        || !comparison.max_logit_delta.is_finite()
    {
        return Err("baseline comparison metric is non-finite".to_owned());
    }
    Ok((baseline_metrics, first_metrics, comparison))
}

fn ensure_session_baseline(
    session: &ExecutionSession,
    baseline: sllm_core::AllocationSnapshot,
) -> Result<(), String> {
    let current = session.memory_snapshot();
    if current.poisoned()
        || current.model_resident().current_bytes() != baseline.model_resident().current_bytes()
        || current.request_state().current_bytes() != 0
        || current.workspace().current_bytes() != 0
        || current.current_bytes() != baseline.current_bytes()
    {
        return Err(format!(
            "request cleanup changed allocation baseline: {current:?}"
        ));
    }
    Ok(())
}

fn coverage(dataset: &Dataset, cases: &[PreparedCase]) -> Result<LongContextCoverage, String> {
    let early = cases.iter().filter(|case| case.band == "early").count();
    let middle = cases.iter().filter(|case| case.band == "middle").count();
    let tail = cases.iter().filter(|case| case.band == "tail").count();
    let block_tail_samples = cases.iter().filter(|case| case.block_tail).count();
    if early == 0 || middle == 0 || tail == 0 || block_tail_samples == 0 {
        return Err(
            "long-context coverage misses an early/middle/tail or block-tail sample".to_owned(),
        );
    }
    let position_min = cases
        .iter()
        .map(|case| case.tokens.len())
        .min()
        .unwrap_or(0);
    let position_max = cases
        .iter()
        .map(|case| case.tokens.len())
        .max()
        .unwrap_or(0);
    let boundary_count = dataset.coverage.boundaries.len();
    let coverage_ratio = cases
        .iter()
        .filter(|case| dataset.coverage.boundaries.contains(&case.tokens.len()))
        .count() as f64
        / boundary_count as f64;
    if position_min == 0
        || position_max == 0
        || !coverage_ratio.is_finite()
        || coverage_ratio <= 0.0
    {
        return Err("long-context coverage is empty or non-finite".to_owned());
    }
    Ok(LongContextCoverage {
        sample_count: cases.len(),
        position_min,
        position_max,
        early,
        middle,
        tail,
        coverage_ratio,
        key_samples: cases.len(),
        value_samples: cases.len(),
        kv_planes: dataset.coverage.kv_planes.clone(),
        layers: dataset.coverage.layers.clone(),
        kv_heads: dataset.coverage.kv_heads.clone(),
        boundaries: dataset.coverage.boundaries.clone(),
        block_tail_samples,
    })
}

fn aggregate_comparisons(cases: &[CaseReport]) -> Result<BaselineComparison, String> {
    let mut aggregate = BaselineComparison::default();
    let mut kld_sum = 0.0_f64;
    for case in cases {
        let comparison = &case.baseline_vs_first;
        aggregate.sample_count = aggregate
            .sample_count
            .checked_add(comparison.sample_count)
            .ok_or_else(|| "baseline comparison count overflowed".to_owned())?;
        aggregate.top1_matches = aggregate
            .top1_matches
            .checked_add(comparison.top1_matches)
            .ok_or_else(|| "baseline comparison top1 count overflowed".to_owned())?;
        kld_sum += comparison.kld_mean * comparison.sample_count as f64;
        aggregate.kld_max = aggregate.kld_max.max(comparison.kld_max);
        aggregate.max_logit_delta = aggregate.max_logit_delta.max(comparison.max_logit_delta);
        if let Some(position) = comparison.first_divergence_position {
            aggregate.first_divergence_position = Some(
                aggregate
                    .first_divergence_position
                    .map_or(position, |current| current.min(position)),
            );
        }
    }
    if aggregate.sample_count == 0 {
        return Err("baseline comparison selected zero rows".to_owned());
    }
    aggregate.kld_mean = kld_sum / aggregate.sample_count as f64;
    aggregate.kld_baseline_to_first = aggregate.kld_mean;
    aggregate.top1_agreement = aggregate.top1_matches as f64 / aggregate.sample_count as f64;
    if !aggregate.kld_mean.is_finite()
        || !aggregate.kld_max.is_finite()
        || !aggregate.top1_agreement.is_finite()
        || !aggregate.max_logit_delta.is_finite()
    {
        return Err("aggregate baseline comparison metric is non-finite".to_owned());
    }
    Ok(aggregate)
}

fn run(arguments: &[String]) -> Result<(Report, PathBuf), String> {
    if arguments.len() != 7 {
        return Err(
            "usage: LOCK CACHE DATASET_JSON DEVICE_INDEX TARGET REPEATS OUTPUT_JSON".to_owned(),
        );
    }
    let lock_path = PathBuf::from(&arguments[0]);
    let cache_path = PathBuf::from(&arguments[1]);
    let dataset_path = PathBuf::from(&arguments[2]);
    let device_index = arguments[3]
        .parse::<u32>()
        .map_err(|_| "device index must be u32".to_owned())?;
    let target = arguments[4].clone();
    if !valid_hip_gfx_target(&target) {
        return Err("target must be a non-empty ASCII HIP gfx target".to_owned());
    }
    let repeats = arguments[5]
        .parse::<u32>()
        .map_err(|_| "repeats must be u32".to_owned())?;
    if !(3..=MAX_REPEATS).contains(&repeats) {
        return Err(format!("repeats must be in 3..={MAX_REPEATS}"));
    }
    let output_path = PathBuf::from(&arguments[6]);
    if output_path.as_os_str().is_empty() || output_path.exists() {
        return Err("output path must be non-empty and must not already exist".to_owned());
    }
    let executable_sha256 = sha256_bytes(
        &fs::read(env::current_exe().map_err(|error| format!("locate executable: {error}"))?)
            .map_err(|error| format!("read executable: {error}"))?,
    );
    let (dataset, cases) = parse_dataset(&dataset_path)?;
    let long_context = coverage(&dataset, &cases)?;
    let lock = read_model_lock(&lock_path).map_err(|error| format!("read model lock: {error}"))?;
    if lock.model.repo_id != QWEN35_4B_REPO_ID
        || lock.model.resolved_revision != QWEN35_4B_REVISION
        || lock.fingerprint() != QWEN35_4B_FINGERPRINT
        || !lock.model.architecture.text_config.tie_word_embeddings
    {
        return Err("quality baseline requires the reviewed Qwen3.5-4B lock".to_owned());
    }
    let cache = Arc::new(
        lock.verify_cache(&cache_path)
            .map_err(|error| format!("verify cache: {error}"))?,
    );
    let plan = build_verified_weight_load_plan(&lock, &cache)
        .map_err(|error| format!("build verified weight plan: {error}"))?;
    if plan.entries.iter().any(|entry| {
        entry.classification == sllm_core::WeightClassification::Required
            && !matches!(entry.dtype, TensorDType::Bf16 | TensorDType::F32)
    }) {
        return Err(
            "quality baseline requires BF16 weights with only reviewed F32 auxiliary tensors"
                .to_owned(),
        );
    }
    let max_length = cases
        .iter()
        .map(|case| case.tokens.len())
        .max()
        .unwrap_or(0);
    let graph = build_qwen35_graph(
        &lock,
        &plan,
        u64::try_from(max_length).map_err(|_| "dataset length exceeds u64")?,
        u64::try_from(max_length + 1).map_err(|_| "dataset capacity exceeds u64")?,
    )
    .map_err(|error| format!("build Qwen graph: {error}"))?;
    let backend = HipBackend::connect().map_err(|error| format!("connect HIP backend: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| format!("build session request: {error}"))?,
        )
        .map_err(|error| format!("open HIP session: {error}"))?;
    let execution = (|| -> Result<(Vec<CaseReport>, sllm_core::AllocationSnapshot), String> {
        let resident = QwenResidentModel::new(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Arc::clone(&cache),
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("provision exact BF16 resident model: {error}"))?;
        let baseline_snapshot = session.memory_snapshot();
        if baseline_snapshot.poisoned()
            || baseline_snapshot.model_resident().current_bytes() == 0
            || baseline_snapshot.request_state().current_bytes() != 0
            || baseline_snapshot.workspace().current_bytes() != 0
            || baseline_snapshot.current_bytes()
                != baseline_snapshot.model_resident().current_bytes()
        {
            return Err(format!(
                "invalid model-ready allocation baseline: {baseline_snapshot:?}"
            ));
        }
        let mut reports = Vec::with_capacity(cases.len());
        for case in &cases {
            let case_graph = build_qwen35_graph(
                &lock,
                &plan,
                case.tokens.len() as u64,
                case.tokens.len() as u64 + 1,
            )
            .map_err(|error| format!("build case graph {}: {error}", case.id))?;
            let mut baseline_request = resident
                .new_request(case_graph.clone())
                .map_err(|error| format!("create baseline request {}: {error}", case.id))?;
            let mut first_request = resident
                .new_request(case_graph.clone())
                .map_err(|error| format!("create first request {}: {error}", case.id))?;
            let (baseline_metrics, first_metrics, comparison) =
                compare_requests(&mut baseline_request, &mut first_request, case, &target)?;
            let baseline = finish_metrics(&baseline_metrics, 0)?;
            let first = finish_metrics(&first_metrics, 0)?;
            drop(baseline_request);
            drop(first_request);
            ensure_session_baseline(&session, baseline_snapshot)?;
            let mut repeats_report = Vec::with_capacity(repeats as usize);
            repeats_report.push(first);
            for repeat in 1..repeats {
                let mut request = resident
                    .new_request(case_graph.clone())
                    .map_err(|error| format!("create repeat request {}: {error}", case.id))?;
                let metrics = process_request(&mut request, case, &target)?;
                let report = finish_metrics(&metrics, repeat)?;
                drop(request);
                ensure_session_baseline(&session, baseline_snapshot)?;
                repeats_report.push(report);
            }
            reports.push(CaseReport {
                id: case.id.clone(),
                length: case.tokens.len(),
                band: case.band.clone(),
                block_tail: case.block_tail,
                token_digest: case.token_digest.clone(),
                expected_next: case.expected_next,
                baseline,
                repeats: repeats_report,
                baseline_vs_first: comparison,
            });
        }
        drop(resident);
        Ok((reports, baseline_snapshot))
    })();
    let (reports, baseline_snapshot) = match execution {
        Ok(value) => value,
        Err(error) => {
            let _ = session.shutdown(SHUTDOWN_TIMEOUT);
            return Err(error);
        }
    };
    let pre_shutdown = session.memory_snapshot();
    if pre_shutdown.current_bytes() != 0 || pre_shutdown.poisoned() {
        let _ = session.shutdown(SHUTDOWN_TIMEOUT);
        return Err(format!(
            "resident model cleanup was not empty: {pre_shutdown:?}"
        ));
    }
    let shutdown = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("shutdown HIP session: {error}"))?;
    let final_snapshot = session.memory_snapshot();
    let cleanup_empty = shutdown.retryable_cleanup == 0
        && shutdown.durable_quarantine == 0
        && final_snapshot.current_bytes() == 0
        && final_snapshot.request_state().current_bytes() == 0
        && final_snapshot.workspace().current_bytes() == 0
        && !final_snapshot.poisoned();
    if !cleanup_empty {
        return Err(format!(
            "quality baseline cleanup was not empty: shutdown={shutdown:?}, final={final_snapshot:?}"
        ));
    }
    if baseline_snapshot.model_resident().current_bytes() == 0 {
        return Err("model-ready baseline unexpectedly had zero resident bytes".to_owned());
    }
    let baseline_vs_first = aggregate_comparisons(&reports)?;
    Ok((
        Report {
            schema: "https://sllm.dev/schema/phase46-qwen35-quality-baseline-v1.schema.json",
            schema_version: "sllm-phase46-qwen35-quality-baseline-v1",
            state: "PASS",
            model_repo_id: QWEN35_4B_REPO_ID,
            model_revision: QWEN35_4B_REVISION,
            model_fingerprint: QWEN35_4B_FINGERPRINT,
            weight_dtype: "BF16",
            kv_encoding: "fp16",
            target,
            device_index,
            executable_sha256,
            dataset_id: DATASET_ID,
            dataset_sha256: DATASET_SHA256,
            dataset_seed: dataset.seed,
            dataset_case_count: cases.len(),
            repeats,
            baseline_pass: true,
            baseline_vs_first,
            long_context,
            cases: reports,
            cleanup: CleanupReport {
                model_ready_current_bytes: baseline_snapshot.model_resident().current_bytes(),
                pre_shutdown_current_bytes: pre_shutdown.current_bytes(),
                final_current_bytes: final_snapshot.current_bytes(),
                final_request_state_bytes: final_snapshot.request_state().current_bytes(),
                final_workspace_bytes: final_snapshot.workspace().current_bytes(),
                retryable_cleanup: shutdown.retryable_cleanup,
                durable_quarantine: shutdown.durable_quarantine,
                empty: cleanup_empty,
            },
        },
        output_path,
    ))
}

fn valid_hip_gfx_target(target: &str) -> bool {
    if target.as_bytes().contains(&0) || !target.is_ascii() {
        return false;
    }
    let mut parts = target.split(':');
    let Some(base) = parts.next() else {
        return false;
    };
    if base.len() <= 3
        || !base.starts_with("gfx")
        || !base[3..].bytes().all(|byte| byte.is_ascii_digit())
    {
        return false;
    }
    let mut sramecc = false;
    let mut xnack = false;
    for feature in parts {
        match feature {
            "sramecc+" | "sramecc-" if !sramecc => sramecc = true,
            "xnack+" | "xnack-" if !xnack => xnack = true,
            _ => return false,
        }
    }
    true
}

fn publish_report(report: &Report, output_path: &Path) -> Result<String, String> {
    let parent = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| format!("create output directory: {error}"))?;
    let file_name = output_path
        .file_name()
        .ok_or_else(|| "output path must name a file".to_owned())?
        .to_string_lossy();
    let partial = parent.join(format!(".{file_name}.partial.{}", std::process::id()));
    let json = serde_json::to_vec(report)
        .map_err(|error| format!("serialize quality baseline: {error}"))?;
    let digest = sha256_bytes(&json);
    let result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .map_err(|error| format!("create partial report: {error}"))?;
        file.write_all(&json)
            .map_err(|error| format!("write partial report: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("sync partial report: {error}"))?;
        fs::hard_link(&partial, output_path)
            .map_err(|error| format!("publish report atomically without replacement: {error}"))?;
        fs::remove_file(&partial)
            .map_err(|error| format!("remove report staging link: {error}"))?;
        if let Err(error) = OpenOptions::new()
            .read(true)
            .open(parent)
            .and_then(|directory| directory.sync_all())
        {
            let _ = fs::remove_file(output_path);
            let _ = OpenOptions::new()
                .read(true)
                .open(parent)
                .and_then(|directory| directory.sync_all());
            return Err(format!("sync output directory: {error}"));
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial);
    }
    result?;
    Ok(digest)
}

fn main() -> ExitCode {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok((report, output_path)) => match publish_report(&report, &output_path) {
            Ok(digest) => {
                println!("{} {}", output_path.display(), digest);
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("quality baseline serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("Qwen3.5 quality baseline failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::valid_hip_gfx_target;

    #[test]
    fn exact_gfx_target_accepts_reviewed_feature_suffixes() {
        assert!(valid_hip_gfx_target("gfx1030"));
        assert!(valid_hip_gfx_target("gfx942:sramecc+:xnack-"));
        assert!(!valid_hip_gfx_target("gfx"));
        assert!(!valid_hip_gfx_target("gfx942:unknown+"));
        assert!(!valid_hip_gfx_target("gfx942:xnack-:xnack+"));
    }
}
