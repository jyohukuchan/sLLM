//! Phase 10 full-model BF16 versus FP8 logits evidence on an exact HIP target.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    Backend, ExecutionSessionRequest, QwenResidentModel, VerifiedFp8Sidecar,
    build_qwen35_fp8_graph, build_qwen35_graph, build_verified_weight_load_plan, read_model_lock,
    verify_fp8_sidecar,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const CASES: [&[i32]; 3] = [&[9419], &[1, 17, 257], &[127, 128, 129, 255, 256, 257]];

#[derive(Serialize)]
struct CaseReport {
    input_token_ids: Vec<i32>,
    bf16_top1: usize,
    fp8_top1: usize,
    top1_match: bool,
    kld_bf16_to_fp8: f64,
    max_abs_logit_error: f32,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    provider: &'static str,
    sidecar_fingerprint: String,
    cases: Vec<CaseReport>,
    max_kld: f64,
    all_top1_match: bool,
    fallback_used: bool,
}

fn execute_logits(
    lock_path: &Path,
    cache_path: &Path,
    sidecar: Option<Arc<VerifiedFp8Sidecar>>,
    device_index: u32,
    target: &str,
) -> Result<Vec<Vec<f32>>, String> {
    let lock = read_model_lock(lock_path).map_err(|error| error.to_string())?;
    let cache = Arc::new(
        lock.verify_cache(cache_path)
            .map_err(|error| error.to_string())?,
    );
    let plan = build_verified_weight_load_plan(&lock, &cache).map_err(|error| error.to_string())?;
    let seed_graph = match &sidecar {
        Some(sidecar) => build_qwen35_fp8_graph(&lock, &plan, sidecar, 1, 7),
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
        let resident = match &sidecar {
            Some(sidecar) => QwenResidentModel::new_fp8(
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
        let mut logits = Vec::with_capacity(CASES.len());
        for input in CASES {
            let token_count = input.len() as u64;
            let graph = match &sidecar {
                Some(sidecar) => {
                    build_qwen35_fp8_graph(&lock, &plan, sidecar, token_count, token_count + 1)
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
        }
        drop(resident);
        Ok(logits)
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
    let lock = read_model_lock(&lock_path).map_err(|error| error.to_string())?;
    let sidecar = Arc::new(
        verify_fp8_sidecar(
            Path::new(&arguments[2]),
            Path::new(&arguments[3]),
            &lock_path,
            &lock,
        )
        .map_err(|error| error.to_string())?,
    );
    let device_index = arguments[4]
        .parse::<u32>()
        .map_err(|_| "device index must be u32".to_owned())?;
    let target = arguments[5].clone();
    let bf16 = execute_logits(&lock_path, &cache_path, None, device_index, &target)?;
    let fp8 = execute_logits(
        &lock_path,
        &cache_path,
        Some(Arc::clone(&sidecar)),
        device_index,
        &target,
    )?;
    let mut reports = Vec::with_capacity(CASES.len());
    for ((input, reference), candidate) in CASES.iter().zip(bf16).zip(fp8) {
        let bf16_top1 = top1(&reference);
        let fp8_top1 = top1(&candidate);
        let divergence = kld(&reference, &candidate);
        let max_abs_logit_error = reference
            .iter()
            .zip(&candidate)
            .map(|(left, right)| (*left - *right).abs())
            .fold(0.0_f32, f32::max);
        reports.push(CaseReport {
            input_token_ids: input.to_vec(),
            bf16_top1,
            fp8_top1,
            top1_match: bf16_top1 == fp8_top1,
            kld_bf16_to_fp8: divergence,
            max_abs_logit_error,
        });
    }
    let max_kld = reports
        .iter()
        .map(|case| case.kld_bf16_to_fp8)
        .fold(0.0, f64::max);
    let all_top1_match = reports.iter().all(|case| case.top1_match);
    if !all_top1_match || max_kld > 0.05 {
        return Err(format!(
            "full-model FP8 accuracy gate failed: all_top1={all_top1_match} max_kld={max_kld}"
        ));
    }
    Ok(Report {
        schema_version: "phase10-fp8-qwen-accuracy-v1",
        state: "PASS",
        target,
        device_index,
        provider: "native-ocp-e4m3fn-outer-f32",
        sidecar_fingerprint: sidecar.manifest_fingerprint().to_owned(),
        cases: reports,
        max_kld,
        all_top1_match,
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
            eprintln!("FP8 Qwen accuracy evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
