//! Exact-artifact end-to-end Qwen3.5-35B-A3B MXFP4 GPU evidence.

use serde::Serialize;
use sllm_core::{
    Backend, ExecutionSessionRequest, QwenResidentModel, build_qwen35_moe_execution_graph,
    build_qwen35_moe_weight_load_plan, verify_qwen35_moe_artifact,
};
use sllm_hip::HipBackend;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

const GUARD: &str = "SLLM_QWEN35_MOE_GPU_EXECUTION";
const TIMEOUT: Duration = Duration::from_secs(600);
const PERFORMANCE_WARMUPS: usize = 2;
const PERFORMANCE_SAMPLES: usize = 11;

#[derive(Serialize)]
struct TimingDistribution {
    samples: usize,
    median_us: u128,
    mad_us: u128,
    p10_us: u128,
    p90_us: u128,
}

impl TimingDistribution {
    fn from_samples(mut samples: Vec<u128>) -> Result<Self, String> {
        if samples.len() != PERFORMANCE_SAMPLES {
            return Err("performance sample count differs from the fixed contract".to_owned());
        }
        samples.sort_unstable();
        let median_us = samples[samples.len() / 2];
        let mut deviations = samples
            .iter()
            .map(|sample| sample.abs_diff(median_us))
            .collect::<Vec<_>>();
        deviations.sort_unstable();
        let percentile = |percent: usize| samples[(samples.len() - 1) * percent / 100];
        Ok(Self {
            samples: samples.len(),
            median_us,
            mad_us: deviations[deviations.len() / 2],
            p10_us: percentile(10),
            p90_us: percentile(90),
        })
    }
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    artifact_root: String,
    model_fingerprint: String,
    plan_digest: String,
    plan_entries: usize,
    prompt_tokens: Vec<i32>,
    prefill_tokens: Vec<i32>,
    decode_token: i32,
    replay_prefill_tokens: Vec<i32>,
    deterministic_replay: bool,
    prefill_dispatches: u64,
    decode_dispatches: u64,
    prefill_sparse_moe_submissions: u64,
    decode_sparse_moe_submissions: u64,
    prefill_active_expert_pairs: u64,
    decode_active_expert_pairs: u64,
    fallback_used: bool,
    all_dispatches_hip: bool,
    artifact_verify_ms: u128,
    plan_graph_ms: u128,
    model_load_ms: u128,
    prefill_ms: u128,
    decode_ms: u128,
    replay_prefill_ms: u128,
    performance_warmups: usize,
    prefill_timing: TimingDistribution,
    decode_timing: TimingDistribution,
    model_resident_bytes: u64,
    request_state_bytes: u64,
    workspace_bytes: u64,
    high_water_bytes: u64,
    model_resident_high_water_bytes: u64,
    request_state_high_water_bytes: u64,
    workspace_high_water_bytes: u64,
    elapsed_ms: u128,
    cleanup_empty: bool,
}

fn run() -> Result<Report, String> {
    if env::var(GUARD).as_deref() != Ok("1") {
        return Err(format!("{GUARD}=1 is required"));
    }
    let mut args = env::args().skip(1);
    let device_index = args
        .next()
        .ok_or("device index is required")?
        .parse::<u32>()
        .map_err(|_| "device index must be U32")?;
    let target = args.next().ok_or("target is required")?;
    let artifact_root = PathBuf::from(args.next().ok_or("artifact root is required")?);
    if args.next().is_some() || !matches!(target.as_str(), "gfx1030" | "gfx1201" | "gfx942") {
        return Err("usage: DEVICE TARGET ARTIFACT_ROOT".to_owned());
    }
    let started = Instant::now();
    let artifact =
        Arc::new(verify_qwen35_moe_artifact(&artifact_root).map_err(|error| error.to_string())?);
    let artifact_verify_ms = started.elapsed().as_millis();
    let plan_started = Instant::now();
    let plan = build_qwen35_moe_weight_load_plan(&artifact).map_err(|error| error.to_string())?;
    let capacity = 17;
    let graph = build_qwen35_moe_execution_graph(&artifact, &plan, 3, capacity)
        .map_err(|error| error.to_string())?;
    let plan_graph_ms = plan_started.elapsed().as_millis();
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let load_started = Instant::now();
    let resident = QwenResidentModel::new_moe(
        Arc::clone(&session),
        graph.clone(),
        plan.clone(),
        Arc::clone(&artifact),
        TIMEOUT,
    )
    .map_err(|error| error.to_string())?;
    let model_load_ms = load_started.elapsed().as_millis();
    let prompt = vec![151_644, 8948, 198];
    let mut request = resident
        .new_request(graph.clone())
        .map_err(|error| error.to_string())?;
    let prefill_started = Instant::now();
    let prefill = request
        .prefill(&prompt)
        .map_err(|error| error.to_string())?;
    let prefill_ms = prefill_started.elapsed().as_millis();
    let prefill_audit = request
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    let decode_token = *prefill
        .token_ids()
        .last()
        .ok_or("prefill produced no token")?;
    let decode_started = Instant::now();
    let decode = request
        .decode(decode_token)
        .map_err(|error| error.to_string())?;
    let decode_ms = decode_started.elapsed().as_millis();
    let decode_audit = request
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    let mut replay = resident
        .new_request(graph.clone())
        .map_err(|error| error.to_string())?;
    let replay_started = Instant::now();
    let replay_prefill = replay.prefill(&prompt).map_err(|error| error.to_string())?;
    let replay_prefill_ms = replay_started.elapsed().as_millis();
    let replay_audit = replay.audit_snapshot().map_err(|error| error.to_string())?;
    let deterministic_replay = replay_prefill.token_ids() == prefill.token_ids();
    let mut fallback_used = prefill_audit.fallback_used()
        || decode_audit.fallback_used()
        || replay_audit.fallback_used();
    let mut all_dispatches_hip = prefill_audit.all_dispatches_hip()
        && decode_audit.all_dispatches_hip()
        && replay_audit.all_dispatches_hip();
    let prefill_sparse_moe_submissions = prefill_audit.sparse_moe_submission_count();
    let decode_sparse_moe_submissions = decode_audit
        .sparse_moe_submission_count()
        .checked_sub(prefill_sparse_moe_submissions)
        .ok_or("SparseMoe submission audit regressed across decode")?;
    let prefill_active_expert_pairs = prefill_audit.sparse_moe_active_pair_count();
    let decode_active_expert_pairs = decode_audit
        .sparse_moe_active_pair_count()
        .checked_sub(prefill_active_expert_pairs)
        .ok_or("SparseMoe active-pair audit regressed across decode")?;
    let mut prefill_samples = Vec::with_capacity(PERFORMANCE_SAMPLES);
    let mut decode_samples = Vec::with_capacity(PERFORMANCE_SAMPLES);
    for sample in 0..PERFORMANCE_WARMUPS + PERFORMANCE_SAMPLES {
        let mut trial = resident
            .new_request(graph.clone())
            .map_err(|error| error.to_string())?;
        let trial_prefill_started = Instant::now();
        let trial_prefill = trial.prefill(&prompt).map_err(|error| error.to_string())?;
        let trial_prefill_us = trial_prefill_started.elapsed().as_micros();
        if trial_prefill.token_ids() != prefill.token_ids() {
            return Err("performance prefill changed the fixed token oracle".to_owned());
        }
        let trial_decode_started = Instant::now();
        let trial_decode = trial
            .decode(decode_token)
            .map_err(|error| error.to_string())?;
        let trial_decode_us = trial_decode_started.elapsed().as_micros();
        if trial_decode.token_ids() != decode.token_ids() {
            return Err("performance decode changed the fixed token oracle".to_owned());
        }
        let trial_audit = trial.audit_snapshot().map_err(|error| error.to_string())?;
        fallback_used |= trial_audit.fallback_used();
        all_dispatches_hip &= trial_audit.all_dispatches_hip();
        if sample >= PERFORMANCE_WARMUPS {
            prefill_samples.push(trial_prefill_us);
            decode_samples.push(trial_decode_us);
        }
    }
    let prefill_timing = TimingDistribution::from_samples(prefill_samples)?;
    let decode_timing = TimingDistribution::from_samples(decode_samples)?;
    let memory = session.memory_snapshot();
    let model_resident_bytes = memory.model_resident().current_bytes();
    let request_state_bytes = memory.request_state().current_bytes();
    let workspace_bytes = memory.workspace().current_bytes();
    let high_water_bytes = memory.high_water_bytes();
    let model_resident_high_water_bytes = memory.model_resident().high_water_bytes();
    let request_state_high_water_bytes = memory.request_state().high_water_bytes();
    let workspace_high_water_bytes = memory.workspace().high_water_bytes();
    drop(replay);
    drop(request);
    drop(resident);
    let allocations_empty = session.memory_snapshot().current_bytes() == 0;
    let cleanup = session
        .shutdown(Duration::from_secs(30))
        .map_err(|error| error.to_string())?;
    let cleanup_empty =
        allocations_empty && cleanup.retryable_cleanup == 0 && cleanup.durable_quarantine == 0;
    if fallback_used
        || !all_dispatches_hip
        || !deterministic_replay
        || !cleanup_empty
        || prefill_sparse_moe_submissions != 40
        || decode_sparse_moe_submissions != 40
        || prefill_active_expert_pairs != 40 * prompt.len() as u64 * 8
        || decode_active_expert_pairs != 40 * 8
    {
        return Err("end-to-end MoE execution evidence failed closed".to_owned());
    }
    Ok(Report {
        schema_version: "sllm-qwen35-moe-execution-evidence-v1",
        state: "PASS",
        target,
        device_index,
        artifact_root: artifact.root().to_string_lossy().into_owned(),
        model_fingerprint: resident_model_fingerprint(),
        plan_digest: plan.digest_hex(),
        plan_entries: plan.entries.len(),
        prompt_tokens: prompt,
        prefill_tokens: prefill.token_ids().to_vec(),
        decode_token: decode.token_ids()[0],
        replay_prefill_tokens: replay_prefill.token_ids().to_vec(),
        deterministic_replay,
        prefill_dispatches: prefill_audit.kernel_dispatch_count(),
        decode_dispatches: decode_audit.kernel_dispatch_count()
            - prefill_audit.kernel_dispatch_count(),
        prefill_sparse_moe_submissions,
        decode_sparse_moe_submissions,
        prefill_active_expert_pairs,
        decode_active_expert_pairs,
        fallback_used,
        all_dispatches_hip,
        artifact_verify_ms,
        plan_graph_ms,
        model_load_ms,
        prefill_ms,
        decode_ms,
        replay_prefill_ms,
        performance_warmups: PERFORMANCE_WARMUPS,
        prefill_timing,
        decode_timing,
        model_resident_bytes,
        request_state_bytes,
        workspace_bytes,
        high_water_bytes,
        model_resident_high_water_bytes,
        request_state_high_water_bytes,
        workspace_high_water_bytes,
        elapsed_ms: started.elapsed().as_millis(),
        cleanup_empty,
    })
}

fn resident_model_fingerprint() -> String {
    sllm_core::QWEN35_MOE_MODEL_FINGERPRINT.to_owned()
}

fn main() -> ExitCode {
    match run() {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}
