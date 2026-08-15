//! Bounded direct-engine profiles for the reviewed Gemma 4 production path.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, Gemma4ResidentModel, build_verified_gemma4_weight_load_plan,
    parse_gemma4_model_lock,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const PROFILE_INPUT_TOKEN: i32 = 2;

struct Config {
    lock: PathBuf,
    cache: PathBuf,
    device_index: u32,
    target: String,
}

#[derive(Serialize)]
struct ProfileReport {
    name: &'static str,
    prefill_tokens: usize,
    generated_tokens: usize,
    generated_token_sha256: String,
    first_generated_token: i32,
    last_generated_token: i32,
    provision_ns: u64,
    prefill_ns: u64,
    ttft_ns: u64,
    decode_total_ns: u64,
    decode_tpot_min_ns: u64,
    decode_tpot_median_ns: u64,
    decode_tpot_max_ns: u64,
    e2e_ns: u64,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    model_resident_bytes: u64,
    request_state_bytes: u64,
    workspace_bytes: u64,
    peak_accounted_bytes: u64,
    submission_count: u64,
    kernel_dispatch_count: u64,
    segment_count: u64,
    boundary_count: u64,
    fallback_used: bool,
    cleanup_request_state_bytes: u64,
    cleanup_workspace_bytes: u64,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    model: &'static str,
    resolved_revision: String,
    lock_fingerprint: String,
    target: String,
    device_index: u32,
    upload_ns: u64,
    available_memory_bytes: Option<u64>,
    model_resident_bytes: u64,
    profiles: Vec<ProfileReport>,
    final_current_bytes: u64,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

fn parse_config() -> Result<Config, String> {
    let mut lock = None;
    let mut cache = None;
    let mut device_index = None;
    let mut target = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{argument} requires a value"))?;
        match argument.as_str() {
            "--lock" if lock.is_none() => lock = Some(PathBuf::from(value)),
            "--cache" if cache.is_none() => cache = Some(PathBuf::from(value)),
            "--device-index" if device_index.is_none() => {
                device_index = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
            }
            "--target" if target.is_none() && matches!(value.as_str(), "gfx1030" | "gfx1201") => {
                target = Some(value);
            }
            "--target" => return Err("--target must be exactly gfx1030 or gfx1201".to_owned()),
            _ => return Err(format!("duplicate or unknown argument: {argument}")),
        }
    }
    Ok(Config {
        lock: lock.ok_or_else(|| "missing --lock".to_owned())?,
        cache: cache.ok_or_else(|| "missing --cache".to_owned())?,
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
    })
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn profile(
    name: &'static str,
    prefill_tokens: usize,
    generated_tokens: usize,
    resident: &Gemma4ResidentModel,
) -> Result<ProfileReport, String> {
    let request_started = Instant::now();
    let state_capacity = prefill_tokens
        .checked_add(generated_tokens)
        .and_then(|count| u64::try_from(count).ok())
        .ok_or_else(|| "profile capacity overflowed".to_owned())?;
    let mut owner = resident
        .new_request(prefill_tokens as u64, state_capacity)
        .map_err(|error| format!("profile request provisioning failed: {error}"))?;
    let provision_ns = elapsed_ns(request_started);
    let allocated = owner.memory_snapshot();
    let input = vec![PROFILE_INPUT_TOKEN; prefill_tokens];
    let prefill_started = Instant::now();
    let output = owner
        .prefill(&input)
        .map_err(|error| format!("profile prefill failed: {error}"))?;
    let prefill_ns = elapsed_ns(prefill_started);
    let ttft_ns = elapsed_ns(request_started);
    let mut current = *output
        .token_ids()
        .last()
        .ok_or_else(|| "profile prefill returned no token".to_owned())?;
    let mut generated = vec![current];
    let mut decode_tpot = Vec::with_capacity(generated_tokens.saturating_sub(1));
    for _ in 1..generated_tokens {
        let started = Instant::now();
        let output = owner
            .decode(current)
            .map_err(|error| format!("profile decode failed: {error}"))?;
        decode_tpot.push(elapsed_ns(started));
        current = *output
            .token_ids()
            .last()
            .ok_or_else(|| "profile decode returned no token".to_owned())?;
        generated.push(current);
    }
    let e2e_ns = elapsed_ns(request_started);
    let audit = owner
        .audit_snapshot()
        .map_err(|error| format!("profile dispatch audit failed: {error}"))?;
    if audit.fallback_used() {
        return Err("profile used an execution fallback".to_owned());
    }
    let peak = owner.memory_snapshot();
    let mut digest = Sha256::new();
    for token in &generated {
        digest.update(token.to_le_bytes());
    }
    let generated_token_sha256 = format!("sha256:{:x}", digest.finalize());
    let decode_total_ns = decode_tpot.iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| "decode timing overflowed".to_owned())
    })?;
    decode_tpot.sort_unstable();
    let decode_tpot_min_ns = decode_tpot.first().copied().unwrap_or(0);
    let decode_tpot_median_ns = decode_tpot
        .get(decode_tpot.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(0);
    let decode_tpot_max_ns = decode_tpot.last().copied().unwrap_or(0);
    let prefill_tokens_per_second = if prefill_ns == 0 {
        0.0
    } else {
        prefill_tokens as f64 * 1_000_000_000.0 / prefill_ns as f64
    };
    let decode_tokens_per_second = if decode_total_ns == 0 {
        0.0
    } else {
        decode_tpot.len() as f64 * 1_000_000_000.0 / decode_total_ns as f64
    };
    let first_generated_token = generated[0];
    let last_generated_token = *generated.last().expect("generated output is nonempty");
    drop(owner);
    let cleanup = resident.memory_snapshot();
    if cleanup.request_state().current_bytes() != 0 || cleanup.workspace().current_bytes() != 0 {
        return Err("profile request cleanup retained dynamic memory".to_owned());
    }
    Ok(ProfileReport {
        name,
        prefill_tokens,
        generated_tokens,
        generated_token_sha256,
        first_generated_token,
        last_generated_token,
        provision_ns,
        prefill_ns,
        ttft_ns,
        decode_total_ns,
        decode_tpot_min_ns,
        decode_tpot_median_ns,
        decode_tpot_max_ns,
        e2e_ns,
        prefill_tokens_per_second,
        decode_tokens_per_second,
        model_resident_bytes: allocated.model_resident().current_bytes(),
        request_state_bytes: allocated.request_state().current_bytes(),
        workspace_bytes: allocated.workspace().current_bytes(),
        peak_accounted_bytes: peak.high_water_bytes(),
        submission_count: audit.submission_count(),
        kernel_dispatch_count: audit.kernel_dispatch_count(),
        segment_count: audit.segment_count(),
        boundary_count: audit.boundary_count(),
        fallback_used: audit.fallback_used(),
        cleanup_request_state_bytes: cleanup.request_state().current_bytes(),
        cleanup_workspace_bytes: cleanup.workspace().current_bytes(),
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
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("invalid execution request: {error}"))?;
    let session = Arc::new(
        backend
            .open_execution_session(request)
            .map_err(|error| format!("cannot open HIP execution session: {error}"))?,
    );
    let available_memory_bytes = session
        .available_memory_bytes()
        .map_err(|error| format!("cannot query available memory: {error}"))?;
    let upload_started = Instant::now();
    let resident = Gemma4ResidentModel::new(
        Arc::clone(&session),
        lock.clone(),
        plan,
        &cache,
        COMPLETION_TIMEOUT,
    )
    .map_err(|error| format!("Gemma resident load failed: {error}"))?;
    let upload_ns = elapsed_ns(upload_started);
    let model_resident_bytes = session.memory_snapshot().model_resident().current_bytes();
    let profiles = vec![
        profile("short-odd-3-17", 3, 17, &resident)?,
        profile("bounded-32-32", 32, 32, &resident)?,
    ];
    drop(resident);
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("session cleanup failed: {error}"))?;
    let final_current_bytes = session.memory_snapshot().current_bytes();
    if final_current_bytes != 0 || cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0
    {
        return Err("profile cleanup retained runtime resources".to_owned());
    }
    Ok(Report {
        schema_version: "gemma4-direct-profile-v1",
        state: "PASS",
        model: "google/gemma-4-12B",
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        target: config.target,
        device_index: config.device_index,
        upload_ns,
        available_memory_bytes,
        model_resident_bytes,
        profiles,
        final_current_bytes,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
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
                eprintln!("cannot serialize profile: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Gemma 4 profile failed: {error}");
            ExitCode::FAILURE
        }
    }
}
