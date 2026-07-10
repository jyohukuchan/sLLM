// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq8_embedding_runtime::QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::sq8_layer_runtime::{
    QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
};
use ullm_engine::sq8_model_head_runtime::{
    QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV, validate_qwen3_14b_sq8_r9700_device_info,
};
use ullm_engine::sq8_serving_runtime::{
    QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS, QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS,
    Qwen3Sq8ServingSession, Sq8CancellationToken, Sq8FinishReason, Sq8ReleaseOutcome,
    Sq8ServingAdvance, Sq8ServingRequest, Sq8ServingRuntimeStatus,
    load_qwen3_14b_sq8_serving_norms,
};
use ullm_runtime_sys::{RuntimeContext, device_count, device_info};

const UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
struct Options {
    artifact: PathBuf,
    package: PathBuf,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
    second_prompt_token_ids: Option<Vec<usize>>,
    second_max_new_tokens: usize,
    cancel_after_first_token: bool,
    oracle_capture_dir: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct ServingCaseResult {
    request_id: String,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
    generated_token_ids: Vec<usize>,
    prompt_progress_events: usize,
    execution_units: usize,
    terminal_reason: &'static str,
    release_outcome: &'static str,
    request_seconds: f64,
    reset_seconds: f64,
    oracle_capture: Option<OracleCaptureResult>,
}

#[derive(Debug, Serialize)]
struct OracleCaptureResult {
    position: usize,
    top1_token_id: usize,
    top1_logit: f32,
    final_hidden_file: PathBuf,
    final_hidden_f32_le_sha256: String,
    logits_file: PathBuf,
    logits_f32_le_sha256: String,
}

#[derive(Debug, Serialize)]
struct CancelledCaseResult {
    request_id: String,
    generated_before_cancel: Vec<usize>,
    release_outcome: &'static str,
    reset_seconds: f64,
}

#[derive(Debug, Serialize)]
struct ServingSmokeResult {
    schema_version: &'static str,
    passed: bool,
    requests: Vec<ServingCaseResult>,
    cancelled_request: Option<CancelledCaseResult>,
    load_seconds: f64,
    artifact_content_sha256: String,
    package_manifest_sha256: String,
    device: ServingDeviceResult,
    kv_cache_bytes: usize,
    cache_blocks: usize,
    context_tokens: usize,
    post_reset_status: &'static str,
    post_reset_active: usize,
    post_reset_waiting: usize,
    post_reset_allocated_blocks: usize,
    post_reset_cache_lengths_all_zero: bool,
}

#[derive(Debug, Serialize)]
struct ServingDeviceResult {
    device_id: i32,
    backend: String,
    name: String,
    gcn_arch_name: String,
    compute_major: i32,
    compute_minor: i32,
    total_global_mem: u64,
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    if let Some(directory) = &options.oracle_capture_dir {
        std::fs::create_dir(directory).map_err(|err| {
            format!(
                "failed to create new oracle capture directory {}: {err}",
                directory.display()
            )
        })?;
    }
    require_hip_kernel_guards()?;
    let artifact = read_sq8_canonical_artifact(&options.artifact)?;
    let norms = load_qwen3_14b_sq8_serving_norms(&options.package, UPLOAD_CHUNK_BYTES)
        .map_err(|err| err.to_string())?;
    let runtime_index = isolated_gfx1201_device()?;
    let mut context = RuntimeContext::create(runtime_index)?;
    let mut stream = context.create_stream()?;

    let load_start = Instant::now();
    let mut session = Qwen3Sq8ServingSession::load(
        &mut context,
        &mut stream,
        &artifact,
        &options.package,
        norms,
        UPLOAD_CHUNK_BYTES,
    )
    .map_err(|err| err.to_string())?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let mut requests = vec![run_completed_request(
        &mut session,
        &mut stream,
        "serving-smoke-1",
        options.prompt_token_ids.clone(),
        options.max_new_tokens,
        options.oracle_capture_dir.as_deref(),
    )?];
    if let Some(prompt_token_ids) = options.second_prompt_token_ids.clone() {
        requests.push(run_completed_request(
            &mut session,
            &mut stream,
            "serving-smoke-2",
            prompt_token_ids,
            options.second_max_new_tokens,
            options.oracle_capture_dir.as_deref(),
        )?);
    }
    let cancelled_request = if options.cancel_after_first_token {
        Some(run_cancel_after_first_token(
            &mut session,
            &mut stream,
            "serving-smoke-cancel",
            options.prompt_token_ids.clone(),
        )?)
    } else {
        None
    };
    let snapshot = reusable_snapshot(&session)?;
    let load_report = session.load_report();
    let result = ServingSmokeResult {
        schema_version: "ullm.sq8.serving_smoke.v2",
        passed: true,
        requests,
        cancelled_request,
        load_seconds,
        artifact_content_sha256: load_report.artifact_content_sha256.clone(),
        package_manifest_sha256: load_report.package_manifest_sha256.clone(),
        device: ServingDeviceResult {
            device_id: load_report.device.device_id,
            backend: load_report.device.backend.clone(),
            name: load_report.device.name.clone(),
            gcn_arch_name: load_report.device.gcn_arch_name.clone(),
            compute_major: load_report.device.compute_major,
            compute_minor: load_report.device.compute_minor,
            total_global_mem: load_report.device.total_global_mem,
        },
        kv_cache_bytes: load_report.total_kv_cache_bytes,
        cache_blocks: QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS,
        context_tokens: QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS,
        post_reset_status: serving_status_name(snapshot.status),
        post_reset_active: snapshot.scheduler_active,
        post_reset_waiting: snapshot.scheduler_waiting,
        post_reset_allocated_blocks: snapshot.allocator.allocated_blocks,
        post_reset_cache_lengths_all_zero: snapshot.cache_lengths.iter().all(|value| *value == 0),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&result)
            .map_err(|err| format!("failed to serialize serving smoke result: {err}"))?
    );
    Ok(())
}

fn run_completed_request(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    request_id: &str,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
    oracle_capture_dir: Option<&Path>,
) -> Result<ServingCaseResult, String> {
    let request = Sq8ServingRequest::greedy(request_id, prompt_token_ids.clone(), max_new_tokens);
    session
        .start(request, Sq8CancellationToken::new(), stream)
        .map_err(|err| err.to_string())?;
    let request_start = Instant::now();
    let mut generated_token_ids = Vec::new();
    let mut prompt_progress_events = 0_usize;
    let mut execution_units = 0_usize;
    let mut oracle_capture = None;
    let terminal_reason = loop {
        let advance = if oracle_capture_dir.is_some()
            && oracle_capture.is_none()
            && session.status() == Sq8ServingRuntimeStatus::Prefilling
        {
            let oracle = session
                .advance_prefill_oracle_synchronized(stream)
                .map_err(|err| err.to_string())?;
            if let Some(capture) = oracle.capture {
                oracle_capture = Some(persist_oracle_capture(
                    oracle_capture_dir.expect("checked above"),
                    request_id,
                    capture,
                )?);
            }
            oracle.advance
        } else {
            session
                .advance_synchronized(stream)
                .map_err(|err| err.to_string())?
        };
        execution_units += 1;
        match advance {
            Sq8ServingAdvance::PromptProgress { .. } => prompt_progress_events += 1,
            Sq8ServingAdvance::Token {
                token_id,
                terminal_reason,
                ..
            } => {
                generated_token_ids.push(token_id);
                if let Some(reason) = terminal_reason {
                    break reason;
                }
            }
            Sq8ServingAdvance::CancellationObserved => {
                return Err("serving smoke observed unexpected cancellation".into());
            }
        }
    };
    let request_seconds = request_start.elapsed().as_secs_f64();
    let reset_start = Instant::now();
    let release = session
        .finish_and_reset_synchronized(stream)
        .map_err(|err| err.to_string())?;
    let reset_seconds = reset_start.elapsed().as_secs_f64();
    reusable_snapshot(session)?;
    if !release.reset_complete
        || generated_token_ids.len() > max_new_tokens
        || generated_token_ids.is_empty()
    {
        return Err("serving smoke terminal/reset contract failed".into());
    }
    Ok(ServingCaseResult {
        request_id: request_id.to_string(),
        prompt_token_ids,
        max_new_tokens,
        generated_token_ids,
        prompt_progress_events,
        execution_units,
        terminal_reason: finish_reason_name(terminal_reason),
        release_outcome: release_outcome_name(release.outcome),
        request_seconds,
        reset_seconds,
        oracle_capture,
    })
}

fn persist_oracle_capture(
    directory: &Path,
    request_id: &str,
    capture: ullm_engine::sq8_serving_runtime::Sq8ServingOracleCapture,
) -> Result<OracleCaptureResult, String> {
    let final_hidden_file = directory.join(format!("{request_id}-final-hidden.f32le"));
    let logits_file = directory.join(format!("{request_id}-logits.f32le"));
    write_f32_le_create_new(&final_hidden_file, &capture.final_hidden)?;
    write_f32_le_create_new(&logits_file, &capture.logits)?;
    Ok(OracleCaptureResult {
        position: capture.position,
        top1_token_id: capture.top1.token_id,
        top1_logit: capture.top1.logit,
        final_hidden_file,
        final_hidden_f32_le_sha256: capture.final_hidden_f32_le_sha256,
        logits_file,
        logits_f32_le_sha256: capture.logits_f32_le_sha256,
    })
}

fn write_f32_le_create_new(path: &Path, values: &[f32]) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))?;
    let file = writer
        .into_inner()
        .map_err(|err| format!("failed to finish {}: {err}", path.display()))?;
    file.sync_all()
        .map_err(|err| format!("failed to sync {}: {err}", path.display()))
}

fn run_cancel_after_first_token(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    request_id: &str,
    prompt_token_ids: Vec<usize>,
) -> Result<CancelledCaseResult, String> {
    let cancel = Sq8CancellationToken::new();
    session
        .start(
            Sq8ServingRequest::greedy(request_id, prompt_token_ids, 8),
            cancel.clone(),
            stream,
        )
        .map_err(|err| err.to_string())?;
    let first_token = loop {
        match session
            .advance_synchronized(stream)
            .map_err(|err| err.to_string())?
        {
            Sq8ServingAdvance::PromptProgress { .. } => {}
            Sq8ServingAdvance::Token {
                token_id,
                terminal_reason: None,
                ..
            } => break token_id,
            Sq8ServingAdvance::Token { .. } => {
                return Err("cancel smoke request finished before cancellation".into());
            }
            Sq8ServingAdvance::CancellationObserved => {
                return Err("cancel smoke observed cancellation before flag publication".into());
            }
        }
    };
    cancel.cancel();
    if session
        .advance_synchronized(stream)
        .map_err(|err| err.to_string())?
        != Sq8ServingAdvance::CancellationObserved
    {
        return Err("cancel smoke published a token after cancellation observation".into());
    }
    let reset_start = Instant::now();
    let release = session
        .abort_and_reset_synchronized(stream)
        .map_err(|err| err.to_string())?;
    let reset_seconds = reset_start.elapsed().as_secs_f64();
    reusable_snapshot(session)?;
    if release.outcome != Sq8ReleaseOutcome::Cancelled || !release.reset_complete {
        return Err("cancel smoke release/reset contract failed".into());
    }
    Ok(CancelledCaseResult {
        request_id: request_id.to_string(),
        generated_before_cancel: vec![first_token],
        release_outcome: release_outcome_name(release.outcome),
        reset_seconds,
    })
}

fn reusable_snapshot(
    session: &Qwen3Sq8ServingSession,
) -> Result<ullm_engine::sq8_serving_runtime::Sq8ServingSnapshot, String> {
    let snapshot = session.snapshot();
    if session.status() != Sq8ServingRuntimeStatus::Ready
        || snapshot.scheduler_active != 0
        || snapshot.scheduler_waiting != 0
        || snapshot.allocator.allocated_blocks != 0
        || snapshot.cache_lengths.iter().any(|value| *value != 0)
    {
        return Err("serving smoke did not return to the reusable baseline".into());
    }
    Ok(snapshot)
}

fn parse_options() -> Result<Options, String> {
    let mut artifact = None;
    let mut package = None;
    let mut prompt_token_ids = vec![1_usize];
    let mut max_new_tokens = 1_usize;
    let mut second_prompt_token_ids = None;
    let mut second_max_new_tokens = 1_usize;
    let mut cancel_after_first_token = false;
    let mut oracle_capture_dir = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        match argument.to_str() {
            Some("--artifact") => {
                artifact = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--artifact requires a path".to_string())?,
                ));
            }
            Some("--package") => {
                package = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--package requires a path".to_string())?,
                ));
            }
            Some("--prompt-token-ids") => {
                let value = args
                    .next()
                    .ok_or_else(|| "--prompt-token-ids requires a value".to_string())?;
                prompt_token_ids = parse_token_ids(
                    value
                        .to_str()
                        .ok_or_else(|| "prompt token IDs must be UTF-8".to_string())?,
                )?;
            }
            Some("--max-new-tokens") => {
                let value = args
                    .next()
                    .ok_or_else(|| "--max-new-tokens requires a value".to_string())?;
                max_new_tokens = value
                    .to_str()
                    .ok_or_else(|| "max-new-tokens must be UTF-8".to_string())?
                    .parse::<usize>()
                    .map_err(|err| format!("invalid max-new-tokens: {err}"))?;
            }
            Some("--second-prompt-token-ids") => {
                let value = args
                    .next()
                    .ok_or_else(|| "--second-prompt-token-ids requires a value".to_string())?;
                second_prompt_token_ids =
                    Some(parse_token_ids(value.to_str().ok_or_else(|| {
                        "second prompt token IDs must be UTF-8".to_string()
                    })?)?);
            }
            Some("--second-max-new-tokens") => {
                let value = args
                    .next()
                    .ok_or_else(|| "--second-max-new-tokens requires a value".to_string())?;
                second_max_new_tokens = value
                    .to_str()
                    .ok_or_else(|| "second max-new-tokens must be UTF-8".to_string())?
                    .parse::<usize>()
                    .map_err(|err| format!("invalid second max-new-tokens: {err}"))?;
            }
            Some("--cancel-after-first-token") => cancel_after_first_token = true,
            Some("--oracle-capture-dir") => {
                oracle_capture_dir =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--oracle-capture-dir requires a path".to_string()
                    })?));
            }
            Some(value) => return Err(format!("unknown argument: {value}")),
            None => return Err("arguments must be UTF-8".into()),
        }
    }
    Ok(Options {
        artifact: artifact.ok_or_else(|| "--artifact is required".to_string())?,
        package: package.ok_or_else(|| "--package is required".to_string())?,
        prompt_token_ids,
        max_new_tokens,
        second_prompt_token_ids,
        second_max_new_tokens,
        cancel_after_first_token,
        oracle_capture_dir,
    })
}

fn parse_token_ids(value: &str) -> Result<Vec<usize>, String> {
    if value.is_empty() {
        return Err("prompt token IDs must not be empty".into());
    }
    value
        .split(',')
        .map(|part| {
            part.parse::<usize>()
                .map_err(|err| format!("invalid prompt token ID {part:?}: {err}"))
        })
        .collect()
}

fn isolated_gfx1201_device() -> Result<u32, String> {
    let mut devices = Vec::new();
    for index in 1..device_count()? {
        let info = device_info(index)
            .map_err(|err| format!("failed to inspect runtime device {index}: {err}"))?;
        if info.backend == "hip" {
            devices.push((index, info));
        }
    }
    if devices.len() != 1 {
        return Err(format!(
            "serving smoke requires exactly one visible HIP device, found {}",
            devices.len()
        ));
    }
    let (runtime_index, device) = devices.pop().expect("one device");
    validate_qwen3_14b_sq8_r9700_device_info(&device)?;
    if device.device_id != 0 {
        return Err(format!(
            "serving smoke requires isolated HIP device 0, got {}",
            device.device_id
        ));
    }
    Ok(runtime_index)
}

fn require_hip_kernel_guards() -> Result<(), String> {
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV)
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    let invalid = names
        .into_iter()
        .filter(|name| std::env::var(name).ok().as_deref() != Some("1"))
        .collect::<Vec<_>>();
    if invalid.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "serving smoke requires these HIP guards to equal 1: {}",
            invalid.join(",")
        ))
    }
}

fn finish_reason_name(reason: Sq8FinishReason) -> &'static str {
    match reason {
        Sq8FinishReason::Stop => "stop",
        Sq8FinishReason::Length => "length",
    }
}

fn release_outcome_name(outcome: Sq8ReleaseOutcome) -> &'static str {
    match outcome {
        Sq8ReleaseOutcome::Stop => "stop",
        Sq8ReleaseOutcome::Length => "length",
        Sq8ReleaseOutcome::Cancelled => "cancelled",
    }
}

fn serving_status_name(status: Sq8ServingRuntimeStatus) -> &'static str {
    match status {
        Sq8ServingRuntimeStatus::Ready => "ready",
        Sq8ServingRuntimeStatus::Prefilling => "prefilling",
        Sq8ServingRuntimeStatus::Decoding => "decoding",
        Sq8ServingRuntimeStatus::Finishing => "finishing",
        Sq8ServingRuntimeStatus::Cancelling => "cancelling",
        Sq8ServingRuntimeStatus::Resetting => "resetting",
        Sq8ServingRuntimeStatus::Failed => "failed",
    }
}
