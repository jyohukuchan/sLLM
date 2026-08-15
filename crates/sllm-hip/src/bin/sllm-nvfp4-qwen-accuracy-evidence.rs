//! Phase 15 full-model BF16 versus weight-only NVFP4 evidence.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use sllm_core::{
    Backend, ExecutionSessionRequest, QwenResidentModel, VerifiedNvfp4Sidecar, build_qwen35_graph,
    build_qwen35_nvfp4_graph, build_verified_weight_load_plan, read_model_lock,
    verify_nvfp4_sidecar,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const CASES: [&[i32]; 3] = [&[9419], &[1, 17, 257], &[127, 128, 129, 255, 256, 257]];

#[derive(Serialize)]
struct ExecutionReport {
    load_elapsed_ms: u128,
    request_elapsed_ms: u128,
    model_resident_source_bytes: u64,
    model_resident_bytes: u64,
    session_high_water_bytes: u64,
    submission_count: u64,
    kernel_dispatch_count: u64,
    all_dispatches_hip: bool,
    fallback_used: bool,
}

#[derive(Serialize)]
struct CaseReport {
    input_token_ids: Vec<i32>,
    bf16_top1: usize,
    nvfp4_top1: usize,
    top1_match: bool,
    kld_bf16_to_nvfp4: f64,
    max_abs_logit_error: f32,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    provider: &'static str,
    arithmetic: &'static str,
    sidecar_fingerprint: String,
    sidecar_artifact_sha256: String,
    sidecar_size_bytes: u64,
    bf16_execution: ExecutionReport,
    nvfp4_execution: ExecutionReport,
    cases: Vec<CaseReport>,
    max_kld: f64,
    all_top1_match: bool,
    default_max_kld_budget: f64,
    default_eligible: bool,
    fallback_used: bool,
}

fn execute_logits(
    lock_path: &Path,
    cache_path: &Path,
    sidecar: Option<Arc<VerifiedNvfp4Sidecar>>,
    device_index: u32,
    target: &str,
) -> Result<(Vec<Vec<f32>>, ExecutionReport), String> {
    let lock = read_model_lock(lock_path).map_err(|error| error.to_string())?;
    let cache = Arc::new(
        lock.verify_cache(cache_path)
            .map_err(|error| error.to_string())?,
    );
    let plan = build_verified_weight_load_plan(&lock, &cache).map_err(|error| error.to_string())?;
    let source_bytes = plan.total_destination_bytes;
    let seed_graph = match &sidecar {
        Some(sidecar) => build_qwen35_nvfp4_graph(&lock, &plan, sidecar, 1, 7),
        None => build_qwen35_graph(&lock, &plan, 1, 7),
    }
    .map_err(|error| error.to_string())?;
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let request = ExecutionSessionRequest::new(device_index, target.to_owned())
        .map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| error.to_string())?;
    let result = (|| {
        let load_start = Instant::now();
        let resident = match &sidecar {
            Some(sidecar) => QwenResidentModel::new_nvfp4(
                Arc::clone(&session),
                seed_graph,
                plan.clone(),
                Arc::clone(&cache),
                Arc::clone(sidecar),
                COMPLETION_TIMEOUT,
            ),
            None => QwenResidentModel::new(
                Arc::clone(&session),
                seed_graph,
                plan.clone(),
                Arc::clone(&cache),
                COMPLETION_TIMEOUT,
            ),
        }
        .map_err(|error| error.to_string())?;
        let load_elapsed_ms = load_start.elapsed().as_millis();
        let resident_snapshot = resident.memory_snapshot();
        let model_resident_bytes = resident_snapshot.model_resident().current_bytes();
        let request_start = Instant::now();
        let mut logits = Vec::with_capacity(CASES.len());
        let mut submission_count = 0_u64;
        let mut kernel_dispatch_count = 0_u64;
        let mut all_dispatches_hip = true;
        let mut fallback_used = false;
        for input in CASES {
            let token_count = input.len() as u64;
            let graph = match &sidecar {
                Some(sidecar) => {
                    build_qwen35_nvfp4_graph(&lock, &plan, sidecar, token_count, token_count + 1)
                }
                None => build_qwen35_graph(&lock, &plan, token_count, token_count + 1),
            }
            .map_err(|error| error.to_string())?;
            let mut owner = resident
                .new_request(graph)
                .map_err(|error| error.to_string())?;
            let output = owner
                .prefill_with_last_logits(input)
                .map_err(|error| error.to_string())?;
            let values = output
                .last_logits()
                .ok_or_else(|| "full-model evidence did not publish logits".to_owned())?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err("full-model evidence produced non-finite logits".to_owned());
            }
            logits.push(values.to_vec());
            let audit = owner.audit_snapshot().map_err(|error| error.to_string())?;
            submission_count += audit.submission_count();
            kernel_dispatch_count += audit.kernel_dispatch_count();
            all_dispatches_hip &= audit.all_dispatches_hip();
            fallback_used |= audit.fallback_used();
        }
        let request_elapsed_ms = request_start.elapsed().as_millis();
        let session_high_water_bytes = resident.memory_snapshot().high_water_bytes();
        drop(resident);
        Ok((
            logits,
            ExecutionReport {
                load_elapsed_ms,
                request_elapsed_ms,
                model_resident_source_bytes: source_bytes,
                model_resident_bytes,
                session_high_water_bytes,
                submission_count,
                kernel_dispatch_count,
                all_dispatches_hip,
                fallback_used,
            },
        ))
    })();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| error.to_string())?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("full-model evidence session cleanup was nonzero".to_owned());
    }
    result
}

fn top1(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}

fn kld(reference: &[f32], candidate: &[f32]) -> f64 {
    let reference_max = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let candidate_max = candidate.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let reference_exp = reference
        .iter()
        .map(|value| f64::from(*value - reference_max).exp())
        .collect::<Vec<_>>();
    let candidate_exp = candidate
        .iter()
        .map(|value| f64::from(*value - candidate_max).exp())
        .collect::<Vec<_>>();
    let reference_sum: f64 = reference_exp.iter().sum();
    let candidate_sum: f64 = candidate_exp.iter().sum();
    reference_exp
        .iter()
        .zip(candidate_exp)
        .map(|(reference, candidate)| {
            let p = reference / reference_sum;
            let q = candidate / candidate_sum;
            if p == 0.0 { 0.0 } else { p * (p / q).ln() }
        })
        .sum()
}

fn run(arguments: &[String]) -> Result<Report, String> {
    if arguments.len() != 6 {
        return Err("usage: LOCK CACHE MANIFEST ARTIFACT DEVICE_INDEX TARGET".to_owned());
    }
    let lock_path = PathBuf::from(&arguments[0]);
    let cache_path = PathBuf::from(&arguments[1]);
    let manifest_path = PathBuf::from(&arguments[2]);
    let artifact_path = PathBuf::from(&arguments[3]);
    let lock = read_model_lock(&lock_path).map_err(|error| error.to_string())?;
    let sidecar = Arc::new(
        verify_nvfp4_sidecar(&manifest_path, &artifact_path, &lock_path, &lock)
            .map_err(|error| error.to_string())?,
    );
    let device_index = arguments[4]
        .parse::<u32>()
        .map_err(|_| "device index must be u32".to_owned())?;
    let target = arguments[5].clone();
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("target must be gfx1030 or gfx1201".to_owned());
    }
    let (bf16, bf16_execution) =
        execute_logits(&lock_path, &cache_path, None, device_index, &target)?;
    let (nvfp4, nvfp4_execution) = execute_logits(
        &lock_path,
        &cache_path,
        Some(Arc::clone(&sidecar)),
        device_index,
        &target,
    )?;
    if bf16_execution.fallback_used
        || nvfp4_execution.fallback_used
        || !bf16_execution.all_dispatches_hip
        || !nvfp4_execution.all_dispatches_hip
    {
        return Err("full-model execution audit reported fallback or non-HIP dispatch".to_owned());
    }
    let mut reports = Vec::with_capacity(CASES.len());
    for ((input, reference), candidate) in CASES.iter().zip(bf16).zip(nvfp4) {
        let bf16_top1 = top1(&reference);
        let nvfp4_top1 = top1(&candidate);
        let divergence = kld(&reference, &candidate);
        let max_abs_logit_error = reference
            .iter()
            .zip(&candidate)
            .map(|(left, right)| (*left - *right).abs())
            .fold(0.0_f32, f32::max);
        reports.push(CaseReport {
            input_token_ids: input.to_vec(),
            bf16_top1,
            nvfp4_top1,
            top1_match: bf16_top1 == nvfp4_top1,
            kld_bf16_to_nvfp4: divergence,
            max_abs_logit_error,
        });
    }
    let max_kld = reports
        .iter()
        .map(|case| case.kld_bf16_to_nvfp4)
        .fold(0.0, f64::max);
    let all_top1_match = reports.iter().all(|case| case.top1_match);
    let default_max_kld_budget = 0.05;
    let default_eligible = all_top1_match && max_kld <= default_max_kld_budget;
    let sidecar_size_bytes = artifact_path
        .metadata()
        .map_err(|error| error.to_string())?
        .len();
    Ok(Report {
        schema_version: "phase15-nvfp4-qwen-accuracy-v1",
        state: "PASS",
        target,
        device_index,
        provider: "packed-dequant",
        arithmetic: "weight-E2M1/block-E4M3FN/tensor-FP32/BF16-activation/FP32-accumulate",
        sidecar_fingerprint: sidecar.manifest_fingerprint().to_owned(),
        sidecar_artifact_sha256: sidecar.artifact_sha256().to_owned(),
        sidecar_size_bytes,
        bf16_execution,
        nvfp4_execution,
        cases: reports,
        max_kld,
        all_top1_match,
        default_max_kld_budget,
        default_eligible,
        fallback_used: false,
    })
}

fn main() -> ExitCode {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(report) => match serde_json::to_string(&report) {
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
            eprintln!("NVFP4 Qwen accuracy evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
