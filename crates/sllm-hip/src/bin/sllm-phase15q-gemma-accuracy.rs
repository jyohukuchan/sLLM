//! Phase 15Q matched Gemma 4 BF16/S0/U0/O0 full-model attribution.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, Gemma4ModelLock, Gemma4ResidentModel, VerifiedCache,
    VerifiedNvfp4Sidecar, WeightLoadPlan, build_verified_gemma4_weight_load_plan,
    parse_gemma4_model_lock, verify_gemma4_nvfp4_sidecar,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(180);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const KLD_BUDGET: f64 = 0.05;
const GENERATION_CASES: usize = 3;
const GENERATION_TOKENS: usize = 8;

struct Config {
    lock: PathBuf,
    cache: PathBuf,
    prompts: PathBuf,
    device_index: u32,
    target: String,
    allow_partial: bool,
    variants: Vec<VariantConfig>,
}

struct VariantConfig {
    id: String,
    manifest: PathBuf,
    artifact: PathBuf,
}

#[derive(Deserialize)]
struct PromptManifest {
    schema_version: String,
    tokenizer_sha256: String,
    tuning_set: bool,
    cases: Vec<PromptCase>,
}

#[derive(Deserialize)]
struct PromptCase {
    id: String,
    token_ids: Vec<i32>,
    comparison_positions: Vec<usize>,
}

struct VariantOutput {
    positions: Vec<PositionOutput>,
    generations: Vec<GenerationOutput>,
    execution: ExecutionReport,
}

struct PositionOutput {
    case_id: String,
    position: usize,
    logits: Vec<f32>,
}

struct GenerationOutput {
    case_id: String,
    tokens: Vec<usize>,
}

#[derive(Serialize)]
struct ExecutionReport {
    load_elapsed_ms: u128,
    request_elapsed_ms: u128,
    model_resident_bytes: u64,
    session_high_water_bytes: u64,
    submission_count: u64,
    kernel_dispatch_count: u64,
    segment_count: u64,
    boundary_count: u64,
    fallback_used: bool,
    cleanup_current_bytes: u64,
}

#[derive(Serialize)]
struct PositionReport {
    case_id: String,
    position: usize,
    reference_top1: usize,
    candidate_top1: usize,
    top1_match: bool,
    top10_overlap: usize,
    kld_bf16_to_candidate: f64,
    max_abs_logit_error: f32,
}

#[derive(Serialize)]
struct GenerationReport {
    case_id: String,
    reference_tokens: Vec<usize>,
    candidate_tokens: Vec<usize>,
    first_divergence: Option<usize>,
}

#[derive(Serialize)]
struct MetricSummary {
    count: usize,
    median_kld: f64,
    p90_kld: f64,
    max_kld: f64,
    worst_case_id: String,
    worst_position: usize,
    top1_matches: usize,
    top1_match_rate: f64,
    min_top10_overlap: usize,
    nonfinite_count: usize,
    kld_budget: f64,
    budget_pass: bool,
}

#[derive(Serialize)]
struct VariantReport {
    id: String,
    manifest_fingerprint: String,
    artifact_sha256: String,
    tensor_count: usize,
    execution: ExecutionReport,
    summary: MetricSummary,
    positions: Vec<PositionReport>,
    generations: Vec<GenerationReport>,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    model: &'static str,
    resolved_revision: String,
    lock_fingerprint: String,
    prompt_manifest_sha256: String,
    prompt_count: usize,
    comparison_position_count: usize,
    target: String,
    device_index: u32,
    provider: &'static str,
    arithmetic: &'static str,
    reference_execution: ExecutionReport,
    variants: Vec<VariantReport>,
    fallback_used: bool,
}

fn parse_config() -> Result<Config, String> {
    let mut lock = None;
    let mut cache = None;
    let mut prompts = None;
    let mut device_index = None;
    let mut target = None;
    let mut allow_partial = false;
    let mut variants = Vec::new();
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--allow-partial" if !allow_partial => {
                allow_partial = true;
                index += 1;
            }
            "--variant" => {
                let values = arguments
                    .get(index + 1..index + 4)
                    .ok_or_else(|| "--variant requires ID MANIFEST ARTIFACT".to_owned())?;
                variants.push(VariantConfig {
                    id: values[0].clone(),
                    manifest: PathBuf::from(&values[1]),
                    artifact: PathBuf::from(&values[2]),
                });
                index += 4;
            }
            argument => {
                let value = arguments
                    .get(index + 1)
                    .ok_or_else(|| format!("{argument} requires a value"))?
                    .clone();
                match argument {
                    "--lock" if lock.is_none() => lock = Some(PathBuf::from(value)),
                    "--cache" if cache.is_none() => cache = Some(PathBuf::from(value)),
                    "--prompts" if prompts.is_none() => prompts = Some(PathBuf::from(value)),
                    "--device-index" if device_index.is_none() => {
                        device_index = Some(
                            value
                                .parse::<u32>()
                                .map_err(|_| "--device-index must be u32".to_owned())?,
                        );
                    }
                    "--target"
                        if target.is_none() && matches!(value.as_str(), "gfx1030" | "gfx1201") =>
                    {
                        target = Some(value);
                    }
                    "--target" => {
                        return Err("--target must be exactly gfx1030 or gfx1201".to_owned());
                    }
                    _ => return Err(format!("duplicate or unknown argument: {argument}")),
                }
                index += 2;
            }
        }
    }
    if variants.is_empty() {
        return Err("at least one --variant is required".to_owned());
    }
    let mut ids = variants
        .iter()
        .map(|variant| &variant.id)
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() != variants.len() {
        return Err("variant IDs must be unique".to_owned());
    }
    Ok(Config {
        lock: lock.ok_or_else(|| "missing --lock".to_owned())?,
        cache: cache.ok_or_else(|| "missing --cache".to_owned())?,
        prompts: prompts.ok_or_else(|| "missing --prompts".to_owned())?,
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
        allow_partial,
        variants,
    })
}

fn validate_prompts(manifest: &PromptManifest) -> Result<(), String> {
    if manifest.schema_version != "phase15q-prompt-manifest-v1"
        || manifest.tuning_set
        || manifest.cases.len() < 32
        || manifest.tokenizer_sha256
            != "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    {
        return Err("prompt manifest identity or evaluation-set contract differs".to_owned());
    }
    for case in &manifest.cases {
        if case.id.is_empty()
            || case.comparison_positions.len() < 3
            || case
                .comparison_positions
                .windows(2)
                .any(|pair| pair[1] != pair[0] + 1)
            || case
                .comparison_positions
                .last()
                .is_none_or(|position| *position >= case.token_ids.len())
        {
            return Err(format!(
                "prompt case {} has invalid comparison positions",
                case.id
            ));
        }
    }
    Ok(())
}

fn execute_variant(
    lock: &Gemma4ModelLock,
    cache: &VerifiedCache,
    plan: &WeightLoadPlan,
    prompts: &PromptManifest,
    sidecar: Option<Arc<VerifiedNvfp4Sidecar>>,
    device_index: u32,
    target: &str,
) -> Result<VariantOutput, String> {
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let request = ExecutionSessionRequest::new(device_index, target.to_owned())
        .map_err(|error| format!("invalid execution request: {error}"))?;
    let session = Arc::new(
        backend
            .open_execution_session(request)
            .map_err(|error| format!("cannot open HIP execution session: {error}"))?,
    );
    let result: Result<(Vec<PositionOutput>, Vec<GenerationOutput>, ExecutionReport), String> =
        (|| {
            let load_started = Instant::now();
            let resident = match sidecar {
                Some(sidecar) => Gemma4ResidentModel::new_nvfp4(
                    Arc::clone(&session),
                    lock.clone(),
                    plan.clone(),
                    cache,
                    sidecar,
                    COMPLETION_TIMEOUT,
                ),
                None => Gemma4ResidentModel::new(
                    Arc::clone(&session),
                    lock.clone(),
                    plan.clone(),
                    cache,
                    COMPLETION_TIMEOUT,
                ),
            }
            .map_err(|error| format!("resident load failed: {error}"))?;
            let load_elapsed_ms = load_started.elapsed().as_millis();
            let model_resident_bytes = resident.memory_snapshot().model_resident().current_bytes();
            let request_started = Instant::now();
            let mut positions = Vec::new();
            let mut submission_count = 0_u64;
            let mut kernel_dispatch_count = 0_u64;
            let mut segment_count = 0_u64;
            let mut boundary_count = 0_u64;
            let mut fallback_used = false;
            for case in &prompts.cases {
                let first = case.comparison_positions[0];
                let capacity = u64::try_from(case.token_ids.len())
                    .map_err(|_| "prompt length does not fit u64".to_owned())?;
                let mut owner = resident
                    .new_request((first + 1) as u64, capacity)
                    .map_err(|error| format!("{} provisioning failed: {error}", case.id))?;
                let output = owner
                    .prefill_with_last_logits(&case.token_ids[..=first])
                    .map_err(|error| format!("{} prefill failed: {error}", case.id))?;
                positions.push(position_output(case, first, &output)?);
                for position in case.comparison_positions.iter().skip(1).copied() {
                    let output = owner
                        .decode_with_last_logits(case.token_ids[position])
                        .map_err(|error| {
                            format!("{} decode {position} failed: {error}", case.id)
                        })?;
                    positions.push(position_output(case, position, &output)?);
                }
                let audit = owner
                    .audit_snapshot()
                    .map_err(|error| format!("{} audit failed: {error}", case.id))?;
                submission_count += audit.submission_count();
                kernel_dispatch_count += audit.kernel_dispatch_count();
                segment_count += audit.segment_count();
                boundary_count += audit.boundary_count();
                fallback_used |= audit.fallback_used();
            }
            let mut generations = Vec::new();
            for case in prompts.cases.iter().take(GENERATION_CASES) {
                let capacity = case
                    .token_ids
                    .len()
                    .checked_add(GENERATION_TOKENS)
                    .and_then(|count| u64::try_from(count).ok())
                    .ok_or_else(|| "generation capacity overflowed".to_owned())?;
                let mut owner = resident
                    .new_request(case.token_ids.len() as u64, capacity)
                    .map_err(|error| {
                        format!("{} generation provisioning failed: {error}", case.id)
                    })?;
                let output = owner
                    .prefill_with_last_logits(&case.token_ids)
                    .map_err(|error| format!("{} generation prefill failed: {error}", case.id))?;
                let mut current = top1(require_logits(&output, &case.id)?);
                let mut tokens = vec![current];
                for _ in 1..GENERATION_TOKENS {
                    let token = i32::try_from(current)
                        .map_err(|_| "generated token does not fit i32".to_owned())?;
                    let output = owner.decode_with_last_logits(token).map_err(|error| {
                        format!("{} generation decode failed: {error}", case.id)
                    })?;
                    current = top1(require_logits(&output, &case.id)?);
                    tokens.push(current);
                }
                let audit = owner
                    .audit_snapshot()
                    .map_err(|error| format!("{} generation audit failed: {error}", case.id))?;
                submission_count += audit.submission_count();
                kernel_dispatch_count += audit.kernel_dispatch_count();
                segment_count += audit.segment_count();
                boundary_count += audit.boundary_count();
                fallback_used |= audit.fallback_used();
                generations.push(GenerationOutput {
                    case_id: case.id.clone(),
                    tokens,
                });
            }
            let request_elapsed_ms = request_started.elapsed().as_millis();
            let session_high_water_bytes = resident.memory_snapshot().high_water_bytes();
            drop(resident);
            Ok((
                positions,
                generations,
                ExecutionReport {
                    load_elapsed_ms,
                    request_elapsed_ms,
                    model_resident_bytes,
                    session_high_water_bytes,
                    submission_count,
                    kernel_dispatch_count,
                    segment_count,
                    boundary_count,
                    fallback_used,
                    cleanup_current_bytes: 0,
                },
            ))
        })();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("session cleanup failed: {error}"))?;
    let cleanup_current_bytes = session.memory_snapshot().current_bytes();
    if cleanup_current_bytes != 0
        || cleanup.retryable_cleanup != 0
        || cleanup.durable_quarantine != 0
    {
        return Err("execution cleanup retained runtime resources".to_owned());
    }
    let (positions, generations, mut execution) = result?;
    execution.cleanup_current_bytes = cleanup_current_bytes;
    if execution.fallback_used {
        return Err("execution audit reported a fallback".to_owned());
    }
    Ok(VariantOutput {
        positions,
        generations,
        execution,
    })
}

fn position_output(
    case: &PromptCase,
    position: usize,
    output: &sllm_core::Gemma4ExecutionOutput,
) -> Result<PositionOutput, String> {
    let logits = require_logits(output, &case.id)?;
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{} position {position} produced non-finite logits",
            case.id
        ));
    }
    Ok(PositionOutput {
        case_id: case.id.clone(),
        position,
        logits: logits.to_vec(),
    })
}

fn require_logits<'a>(
    output: &'a sllm_core::Gemma4ExecutionOutput,
    case_id: &str,
) -> Result<&'a [f32], String> {
    output
        .last_logits()
        .ok_or_else(|| format!("{case_id} did not publish logits"))
}

fn top1(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map_or(0, |(index, _)| index)
}

fn top_k(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(count.min(indices.len()));
    indices
}

fn kld(reference: &[f32], candidate: &[f32]) -> Result<f64, String> {
    if reference.len() != candidate.len() || reference.is_empty() {
        return Err("logit vectors differ in length".to_owned());
    }
    let reference_max = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let candidate_max = candidate.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let reference_sum = reference
        .iter()
        .map(|value| f64::from(*value - reference_max).exp())
        .sum::<f64>();
    let candidate_sum = candidate
        .iter()
        .map(|value| f64::from(*value - candidate_max).exp())
        .sum::<f64>();
    let reference_log_sum = reference_sum.ln();
    let candidate_log_sum = candidate_sum.ln();
    let divergence = reference
        .iter()
        .zip(candidate)
        .map(|(reference_value, candidate_value)| {
            let reference_log_probability =
                f64::from(*reference_value - reference_max) - reference_log_sum;
            let candidate_log_probability =
                f64::from(*candidate_value - candidate_max) - candidate_log_sum;
            reference_log_probability.exp()
                * (reference_log_probability - candidate_log_probability)
        })
        .sum::<f64>();
    if divergence.is_finite() {
        Ok(divergence.max(0.0))
    } else {
        Err("KLD is non-finite".to_owned())
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = (quantile * (sorted.len().saturating_sub(1) as f64)).ceil() as usize;
    sorted[index]
}

fn compare_variant(
    id: String,
    sidecar: &VerifiedNvfp4Sidecar,
    reference: &VariantOutput,
    candidate: VariantOutput,
) -> Result<VariantReport, String> {
    if reference.positions.len() != candidate.positions.len()
        || reference.generations.len() != candidate.generations.len()
    {
        return Err(format!("{id} output inventory differs from B0"));
    }
    let mut positions = Vec::with_capacity(reference.positions.len());
    for (reference, candidate) in reference.positions.iter().zip(&candidate.positions) {
        if reference.case_id != candidate.case_id || reference.position != candidate.position {
            return Err(format!("{id} position identity differs from B0"));
        }
        let reference_top1 = top1(&reference.logits);
        let candidate_top1 = top1(&candidate.logits);
        let reference_top10 = top_k(&reference.logits, 10);
        let candidate_top10 = top_k(&candidate.logits, 10);
        let top10_overlap = reference_top10
            .iter()
            .filter(|token| candidate_top10.contains(token))
            .count();
        let max_abs_logit_error = reference
            .logits
            .iter()
            .zip(&candidate.logits)
            .map(|(left, right)| (*left - *right).abs())
            .fold(0.0_f32, f32::max);
        positions.push(PositionReport {
            case_id: reference.case_id.clone(),
            position: reference.position,
            reference_top1,
            candidate_top1,
            top1_match: reference_top1 == candidate_top1,
            top10_overlap,
            kld_bf16_to_candidate: kld(&reference.logits, &candidate.logits)?,
            max_abs_logit_error,
        });
    }
    let mut klds = positions
        .iter()
        .map(|position| position.kld_bf16_to_candidate)
        .collect::<Vec<_>>();
    klds.sort_by(f64::total_cmp);
    let worst = positions
        .iter()
        .max_by(|left, right| {
            left.kld_bf16_to_candidate
                .total_cmp(&right.kld_bf16_to_candidate)
        })
        .ok_or_else(|| "comparison set is empty".to_owned())?;
    let top1_matches = positions
        .iter()
        .filter(|position| position.top1_match)
        .count();
    let max_kld = *klds.last().expect("nonempty comparison set");
    let summary = MetricSummary {
        count: positions.len(),
        median_kld: percentile(&klds, 0.5),
        p90_kld: percentile(&klds, 0.9),
        max_kld,
        worst_case_id: worst.case_id.clone(),
        worst_position: worst.position,
        top1_matches,
        top1_match_rate: top1_matches as f64 / positions.len() as f64,
        min_top10_overlap: positions
            .iter()
            .map(|position| position.top10_overlap)
            .min()
            .unwrap_or(0),
        nonfinite_count: 0,
        kld_budget: KLD_BUDGET,
        budget_pass: max_kld <= KLD_BUDGET,
    };
    let generations = reference
        .generations
        .iter()
        .zip(candidate.generations)
        .map(|(reference, candidate)| {
            if reference.case_id != candidate.case_id {
                return Err(format!("{id} generation identity differs from B0"));
            }
            let first_divergence = reference
                .tokens
                .iter()
                .zip(&candidate.tokens)
                .position(|(left, right)| left != right);
            Ok(GenerationReport {
                case_id: reference.case_id.clone(),
                reference_tokens: reference.tokens.clone(),
                candidate_tokens: candidate.tokens,
                first_divergence,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(VariantReport {
        id,
        manifest_fingerprint: sidecar.manifest_fingerprint().to_owned(),
        artifact_sha256: sidecar.artifact_sha256().to_owned(),
        tensor_count: sidecar.tensors().count(),
        execution: candidate.execution,
        summary,
        positions,
        generations,
    })
}

fn run(config: Config) -> Result<Report, String> {
    let lock_bytes =
        std::fs::read(&config.lock).map_err(|error| format!("cannot read Gemma lock: {error}"))?;
    let lock = parse_gemma4_model_lock(&lock_bytes)
        .map_err(|error| format!("invalid Gemma lock: {error}"))?;
    let cache = lock
        .verify_cache(&config.cache)
        .map_err(|error| format!("Gemma cache verification failed: {error}"))?;
    let plan = build_verified_gemma4_weight_load_plan(&lock, &cache)
        .map_err(|error| format!("Gemma weight plan failed: {error}"))?;
    let prompt_bytes =
        std::fs::read(&config.prompts).map_err(|error| format!("cannot read prompts: {error}"))?;
    let prompts: PromptManifest = serde_json::from_slice(&prompt_bytes)
        .map_err(|error| format!("invalid prompt manifest: {error}"))?;
    validate_prompts(&prompts)?;
    let prompt_manifest_sha256 = format!("{:x}", Sha256::digest(&prompt_bytes));
    let mut verified = Vec::with_capacity(config.variants.len());
    for variant in &config.variants {
        let sidecar =
            verify_gemma4_nvfp4_sidecar(&variant.manifest, &variant.artifact, &config.lock, &lock)
                .map_err(|error| format!("{} sidecar verification failed: {error}", variant.id))?;
        if !config.allow_partial && sidecar.tensors().count() != 144 {
            return Err(format!(
                "{} must contain exactly 144 MLP tensors",
                variant.id
            ));
        }
        verified.push(Arc::new(sidecar));
    }
    let reference = execute_variant(
        &lock,
        &cache,
        &plan,
        &prompts,
        None,
        config.device_index,
        &config.target,
    )?;
    let comparison_position_count = reference.positions.len();
    let mut reports = Vec::with_capacity(config.variants.len());
    for (variant, sidecar) in config.variants.into_iter().zip(verified) {
        let output = execute_variant(
            &lock,
            &cache,
            &plan,
            &prompts,
            Some(Arc::clone(&sidecar)),
            config.device_index,
            &config.target,
        )?;
        reports.push(compare_variant(variant.id, &sidecar, &reference, output)?);
    }
    Ok(Report {
        schema_version: "phase15q-gemma-accuracy-v1",
        state: "PASS",
        model: "google/gemma-4-12B-it",
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        prompt_manifest_sha256,
        prompt_count: prompts.cases.len(),
        comparison_position_count,
        target: config.target,
        device_index: config.device_index,
        provider: "packed-dequant",
        arithmetic: "weight-E2M1/block-E4M3FN/tensor-FP32/BF16-activation/FP32-accumulate/FP16-KV",
        fallback_used: reference.execution.fallback_used
            || reports
                .iter()
                .any(|variant| variant.execution.fallback_used),
        reference_execution: reference.execution,
        variants: reports,
    })
}

fn main() -> ExitCode {
    match parse_config().and_then(run) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("cannot serialize Phase 15Q report: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Phase 15Q Gemma accuracy failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kld_is_zero_for_identical_logits_and_positive_otherwise() {
        let reference = [0.0, 1.0, -2.0, 0.5];
        assert!(kld(&reference, &reference).unwrap() <= f64::EPSILON);
        assert!(kld(&reference, &[1.0, 0.0, -2.0, 0.5]).unwrap() > 0.0);
    }

    #[test]
    fn ranking_ties_use_the_lowest_token_id() {
        let logits = [1.0, 3.0, 3.0, 2.0];
        assert_eq!(top1(&logits), 1);
        assert_eq!(top_k(&logits, 4), vec![1, 2, 3, 0]);
    }

    #[test]
    fn percentile_uses_the_conservative_upper_rank() {
        let values = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&values, 0.5), 2.0);
        assert_eq!(percentile(&values, 0.9), 4.0);
    }
}
