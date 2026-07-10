// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq8_embedding_runtime::QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::sq8_layer_runtime::{
    QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV,
    QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_SQ8_PREFILL_CHUNK_TOKENS,
    QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
};
use ullm_engine::sq8_model_head_runtime::{
    QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV, validate_qwen3_14b_sq8_r9700_device_info,
};
use ullm_engine::sq8_serving_runtime::{
    QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS, QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS,
    QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS, Qwen3Sq8ServingSession, Sq8CancellationToken,
    Sq8FinishReason, Sq8ReleaseOutcome, Sq8ServingAdvance, Sq8ServingPrefillMode,
    Sq8ServingRequest, Sq8ServingRuntimeStatus, load_qwen3_14b_sq8_serving_norms,
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
    prompt_lengths: Option<Vec<usize>>,
    prefill_mode: Sq8ServingPrefillMode,
    cancel_after_first_token: bool,
    cancel_after_prompt_progress: Option<usize>,
    oracle_capture_dir: Option<PathBuf>,
    result_json: Option<PathBuf>,
}

#[derive(Debug)]
struct ServingCase {
    request_id: String,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ServingCaseResult {
    request_id: String,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
    generated_token_ids: Vec<usize>,
    prompt_progress_events: usize,
    execution_units: usize,
    processed_prompt_tokens: usize,
    execution_calls: usize,
    prefill_execution_units: Vec<PrefillExecutionUnitResult>,
    reserved_context_tokens: usize,
    terminal_sequence_tokens: usize,
    terminal_status: &'static str,
    terminal_expected_cache_len: usize,
    terminal_cache_lengths: Vec<usize>,
    terminal_cache_lengths_all_expected: bool,
    terminal_last_cache_position: usize,
    terminal_last_logical_block: usize,
    terminal_scheduler_active: usize,
    terminal_scheduler_waiting: usize,
    terminal_allocated_blocks: usize,
    terminal_reason: &'static str,
    release_outcome: &'static str,
    request_seconds: f64,
    reset_seconds: f64,
    oracle_capture: Option<OracleCaptureResult>,
}

#[derive(Debug, Serialize)]
struct PrefillExecutionUnitResult {
    start_position: usize,
    width: usize,
    end_position: usize,
    final_prompt_unit: bool,
    cache_lengths: Vec<usize>,
    cache_lengths_all_expected: bool,
    last_cache_position: usize,
    last_logical_block: usize,
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
    cancellation_phase: &'static str,
    prompt_tokens: usize,
    prompt_progress_before_cancel: usize,
    generated_before_cancel: Vec<usize>,
    execution_units_before_cancel: usize,
    status_before_cancel: &'static str,
    cache_lengths_before_cancel: Vec<usize>,
    scheduler_active_before_cancel: usize,
    scheduler_waiting_before_cancel: usize,
    allocated_blocks_before_cancel: usize,
    status_after_observation: &'static str,
    prompt_progress_after_observation: usize,
    generated_tokens_after_observation: usize,
    cache_lengths_after_observation: Vec<usize>,
    scheduler_active_after_observation: usize,
    scheduler_waiting_after_observation: usize,
    allocated_blocks_after_observation: usize,
    release_outcome: &'static str,
    reset_seconds: f64,
}

#[derive(Debug, Serialize)]
struct ServingSmokeResult {
    schema_version: &'static str,
    prefill_mode: &'static str,
    prefill_chunk_tokens: usize,
    prefill_implementation: String,
    runner_git_commit: String,
    runner_worktree_clean: bool,
    runner_binary_sha256: String,
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
    post_reset_cache_lengths: Vec<usize>,
    post_reset_cache_lengths_all_zero: bool,
}

#[derive(Debug)]
struct RunnerIdentity {
    git_commit: String,
    worktree_clean: bool,
    binary_sha256: String,
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
    let runner_identity = runner_identity()?;
    if let Some(directory) = &options.oracle_capture_dir {
        std::fs::create_dir(directory).map_err(|err| {
            format!(
                "failed to create new oracle capture directory {}: {err}",
                directory.display()
            )
        })?;
    }
    require_hip_kernel_guards(options.prefill_mode)?;
    let artifact = read_sq8_canonical_artifact(&options.artifact)?;
    let norms = load_qwen3_14b_sq8_serving_norms(&options.package, UPLOAD_CHUNK_BYTES)
        .map_err(|err| err.to_string())?;
    let runtime_index = isolated_gfx1201_device()?;
    let mut context = RuntimeContext::create(runtime_index)?;
    let mut stream = context.create_stream()?;

    let load_start = Instant::now();
    let mut session = Qwen3Sq8ServingSession::load_with_prefill_mode(
        &mut context,
        &mut stream,
        &artifact,
        &options.package,
        norms,
        UPLOAD_CHUNK_BYTES,
        options.prefill_mode,
    )
    .map_err(|err| err.to_string())?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let cases = serving_cases(&options)?;
    let cancellation_prompt = cases
        .first()
        .expect("serving cases are nonempty")
        .prompt_token_ids
        .clone();
    let mut requests = Vec::with_capacity(cases.len());
    let evidence_root = options.result_json.as_deref().and_then(Path::parent);
    for case in cases {
        requests.push(run_completed_request(
            &mut session,
            &mut stream,
            &case.request_id,
            case.prompt_token_ids,
            case.max_new_tokens,
            options.oracle_capture_dir.as_deref(),
            evidence_root,
        )?);
    }
    let cancelled_request = match (
        options.cancel_after_first_token,
        options.cancel_after_prompt_progress,
    ) {
        (true, None) => Some(run_cancel_after_first_token(
            &mut session,
            &mut stream,
            "serving-smoke-cancel",
            cancellation_prompt,
        )?),
        (false, Some(prompt_progress)) => Some(run_cancel_during_prefill(
            &mut session,
            &mut stream,
            "serving-smoke-prefill-cancel",
            cancellation_prompt,
            prompt_progress,
        )?),
        (false, None) => None,
        (true, Some(_)) => return Err("cancellation modes must be mutually exclusive".into()),
    };
    let snapshot = reusable_snapshot(&session)?;
    let load_report = session.load_report();
    let result = ServingSmokeResult {
        schema_version: match options.prefill_mode {
            Sq8ServingPrefillMode::SequentialM1 => "ullm.sq8.serving_smoke.v2",
            Sq8ServingPrefillMode::FixedM8Chunks => "ullm.sq8.serving_chunks.v3",
        },
        prefill_mode: prefill_mode_name(options.prefill_mode),
        prefill_chunk_tokens: load_report.prefill_chunk_tokens,
        prefill_implementation: load_report.prefill_implementation.clone(),
        runner_git_commit: runner_identity.git_commit,
        runner_worktree_clean: runner_identity.worktree_clean,
        runner_binary_sha256: runner_identity.binary_sha256,
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
        post_reset_cache_lengths: snapshot.cache_lengths.clone(),
        post_reset_cache_lengths_all_zero: snapshot.cache_lengths.iter().all(|value| *value == 0),
    };
    let serialized = serde_json::to_string_pretty(&result)
        .map_err(|err| format!("failed to serialize serving smoke result: {err}"))?;
    if let Some(path) = &options.result_json {
        write_bytes_create_new(path, serialized.as_bytes())?;
    }
    println!("{serialized}");
    Ok(())
}

fn run_completed_request(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    request_id: &str,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
    oracle_capture_dir: Option<&Path>,
    evidence_root: Option<&Path>,
) -> Result<ServingCaseResult, String> {
    let request = Sq8ServingRequest::greedy(request_id, prompt_token_ids.clone(), max_new_tokens);
    session
        .start(request, Sq8CancellationToken::new(), stream)
        .map_err(|err| err.to_string())?;
    let request_start = Instant::now();
    let prefill_mode = session.prefill_mode();
    let mut generated_token_ids = Vec::new();
    let mut prompt_progress_events = 0_usize;
    let mut execution_units = 0_usize;
    let mut prefill_execution_units = Vec::new();
    let mut oracle_capture = None;
    let terminal_reason = loop {
        let prefill_before =
            (session.status() == Sq8ServingRuntimeStatus::Prefilling).then(|| session.snapshot());
        let capture_directory = match oracle_capture_dir {
            Some(directory)
                if oracle_capture.is_none()
                    && session.status() == Sq8ServingRuntimeStatus::Prefilling =>
            {
                Some(directory)
            }
            _ => None,
        };
        let advance = if let Some(capture_directory) = capture_directory {
            let oracle = session
                .advance_prefill_oracle_synchronized(stream)
                .map_err(|err| err.to_string())?;
            if let Some(capture) = oracle.capture {
                oracle_capture = Some(persist_oracle_capture(
                    capture_directory,
                    request_id,
                    capture,
                    evidence_root,
                )?);
            }
            oracle.advance
        } else {
            session
                .advance_synchronized(stream)
                .map_err(|err| err.to_string())?
        };
        execution_units += 1;
        if let Some(before) = prefill_before {
            let after = session.snapshot();
            let start_position = before.prompt_tokens_processed;
            let end_position = after.prompt_tokens_processed;
            let width = end_position
                .checked_sub(start_position)
                .ok_or_else(|| "serving smoke prefill progress moved backwards".to_string())?;
            if width == 0 {
                return Err("serving smoke prefill execution made no token progress".into());
            }
            let cache_lengths_all_expected = after
                .cache_lengths
                .iter()
                .all(|length| *length == end_position);
            if !cache_lengths_all_expected {
                return Err(format!(
                    "serving smoke prefill cache mismatch after {start_position}..{end_position}: {:?}",
                    after.cache_lengths
                ));
            }
            let last_cache_position = end_position
                .checked_sub(1)
                .ok_or_else(|| "serving smoke prefill cache position underflows".to_string())?;
            prefill_execution_units.push(PrefillExecutionUnitResult {
                start_position,
                width,
                end_position,
                final_prompt_unit: end_position == prompt_token_ids.len(),
                cache_lengths: after.cache_lengths,
                cache_lengths_all_expected,
                last_cache_position,
                last_logical_block: last_cache_position / QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS,
            });
        }
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
    let terminal_snapshot = session.snapshot();
    let reserved_context_tokens = prompt_token_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| "serving smoke reserved context overflows".to_string())?;
    let terminal_sequence_tokens = prompt_token_ids
        .len()
        .checked_add(generated_token_ids.len())
        .ok_or_else(|| "serving smoke terminal sequence length overflows".to_string())?;
    let terminal_expected_cache_len = terminal_sequence_tokens
        .checked_sub(1)
        .ok_or_else(|| "serving smoke terminal cache length underflows".to_string())?;
    let terminal_cache_lengths_all_expected = terminal_snapshot
        .cache_lengths
        .iter()
        .all(|length| *length == terminal_expected_cache_len);
    let expected_prefill_calls =
        expected_prefill_execution_calls(prompt_token_ids.len(), prefill_mode)?;
    let expected_decode_calls = generated_token_ids
        .len()
        .checked_sub(1)
        .ok_or_else(|| "serving smoke emitted no generated token".to_string())?;
    let expected_execution_calls = expected_prefill_calls
        .checked_add(expected_decode_calls)
        .ok_or_else(|| "serving smoke execution call count overflows".to_string())?;
    if terminal_snapshot.status != Sq8ServingRuntimeStatus::Finishing
        || terminal_snapshot.active_request_id.as_deref() != Some(request_id)
        || terminal_snapshot.prompt_tokens != prompt_token_ids.len()
        || terminal_snapshot.prompt_tokens_processed != prompt_token_ids.len()
        || terminal_snapshot.generated_tokens != generated_token_ids.len()
        || terminal_snapshot.scheduler_active != 1
        || terminal_snapshot.scheduler_waiting != 0
        || terminal_snapshot.allocator.allocated_blocks != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
        || !terminal_cache_lengths_all_expected
        || prompt_progress_events != expected_prefill_calls - 1
        || prefill_execution_units.len() != expected_prefill_calls
        || prefill_execution_units
            .iter()
            .map(|unit| unit.width)
            .sum::<usize>()
            != prompt_token_ids.len()
        || execution_units != expected_execution_calls
        || reserved_context_tokens > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS
    {
        return Err(format!(
            "serving smoke terminal snapshot mismatch: snapshot={terminal_snapshot:?} "
        ));
    }
    let terminal_last_cache_position = terminal_expected_cache_len - 1;
    let terminal_last_logical_block =
        terminal_last_cache_position / QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS;
    let reset_start = Instant::now();
    let release = session
        .finish_and_reset_synchronized(stream)
        .map_err(|err| err.to_string())?;
    let reset_seconds = reset_start.elapsed().as_secs_f64();
    reusable_snapshot(session)?;
    if !release.reset_complete
        || generated_token_ids.len() > max_new_tokens
        || generated_token_ids.is_empty()
        || release.request_id != request_id
        || release.prompt_tokens != prompt_token_ids.len()
        || release.generated_tokens != generated_token_ids.len()
        || release.outcome
            != match terminal_reason {
                Sq8FinishReason::Stop => Sq8ReleaseOutcome::Stop,
                Sq8FinishReason::Length => Sq8ReleaseOutcome::Length,
            }
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
        processed_prompt_tokens: terminal_snapshot.prompt_tokens_processed,
        execution_calls: execution_units,
        prefill_execution_units,
        reserved_context_tokens,
        terminal_sequence_tokens,
        terminal_status: serving_status_name(terminal_snapshot.status),
        terminal_expected_cache_len,
        terminal_cache_lengths: terminal_snapshot.cache_lengths,
        terminal_cache_lengths_all_expected,
        terminal_last_cache_position,
        terminal_last_logical_block,
        terminal_scheduler_active: terminal_snapshot.scheduler_active,
        terminal_scheduler_waiting: terminal_snapshot.scheduler_waiting,
        terminal_allocated_blocks: terminal_snapshot.allocator.allocated_blocks,
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
    evidence_root: Option<&Path>,
) -> Result<OracleCaptureResult, String> {
    let final_hidden_file = directory.join(format!("{request_id}-final-hidden.f32le"));
    let logits_file = directory.join(format!("{request_id}-logits.f32le"));
    write_f32_le_create_new(&final_hidden_file, &capture.final_hidden)?;
    write_f32_le_create_new(&logits_file, &capture.logits)?;
    let recorded_final_hidden = evidence_path(&final_hidden_file, evidence_root);
    let recorded_logits = evidence_path(&logits_file, evidence_root);
    Ok(OracleCaptureResult {
        position: capture.position,
        top1_token_id: capture.top1.token_id,
        top1_logit: capture.top1.logit,
        final_hidden_file: recorded_final_hidden,
        final_hidden_f32_le_sha256: capture.final_hidden_f32_le_sha256,
        logits_file: recorded_logits,
        logits_f32_le_sha256: capture.logits_f32_le_sha256,
    })
}

fn evidence_path(path: &Path, evidence_root: Option<&Path>) -> PathBuf {
    evidence_root
        .and_then(|root| path.strip_prefix(root).ok())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| path.to_path_buf())
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

fn write_bytes_create_new(path: &Path, payload: &[u8]) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(payload)
        .and_then(|_| writer.write_all(b"\n"))
        .and_then(|_| writer.flush())
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    let file = writer
        .into_inner()
        .map_err(|err| format!("failed to finish {}: {err}", path.display()))?;
    file.sync_all()
        .map_err(|err| format!("failed to sync {}: {err}", path.display()))
}

fn runner_identity() -> Result<RunnerIdentity, String> {
    let root_output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|err| format!("failed to inspect runner git root: {err}"))?;
    if !root_output.status.success() {
        return Err("runner evidence requires a git worktree".into());
    }
    let root = String::from_utf8(root_output.stdout)
        .map_err(|err| format!("runner git root is not UTF-8: {err}"))?;
    let root = root.trim();
    let commit_output = Command::new("git")
        .args(["-C", root, "rev-parse", "HEAD"])
        .output()
        .map_err(|err| format!("failed to inspect runner git commit: {err}"))?;
    if !commit_output.status.success() {
        return Err("failed to resolve runner git commit".into());
    }
    let git_commit = String::from_utf8(commit_output.stdout)
        .map_err(|err| format!("runner git commit is not UTF-8: {err}"))?
        .trim()
        .to_string();
    if git_commit.len() != 40 || !git_commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("runner git commit is invalid: {git_commit:?}"));
    }
    let status_output = Command::new("git")
        .args(["-C", root, "status", "--porcelain", "--untracked-files=all"])
        .output()
        .map_err(|err| format!("failed to inspect runner worktree status: {err}"))?;
    if !status_output.status.success() {
        return Err("failed to inspect runner worktree status".into());
    }
    let status = String::from_utf8(status_output.stdout)
        .map_err(|err| format!("runner worktree status is not UTF-8: {err}"))?;
    let worktree_clean = status
        .lines()
        .all(|line| line == "?? .rocprofv3/" || line.starts_with("?? .rocprofv3/"));

    let executable = std::env::current_exe()
        .map_err(|err| format!("failed to resolve runner executable: {err}"))?;
    let mut file = File::open(&executable).map_err(|err| {
        format!(
            "failed to open runner executable {}: {err}",
            executable.display()
        )
    })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to hash runner executable: {err}"))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(RunnerIdentity {
        git_commit,
        worktree_clean,
        binary_sha256: format!("{:x}", digest.finalize()),
    })
}

fn run_cancel_after_first_token(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    request_id: &str,
    prompt_token_ids: Vec<usize>,
) -> Result<CancelledCaseResult, String> {
    let prompt_tokens = prompt_token_ids.len();
    let expected_prefill_calls =
        expected_prefill_execution_calls(prompt_tokens, session.prefill_mode())?;
    let cancel = Sq8CancellationToken::new();
    session
        .start(
            Sq8ServingRequest::greedy(request_id, prompt_token_ids, 8),
            cancel.clone(),
            stream,
        )
        .map_err(|err| err.to_string())?;
    let mut prompt_progress_before_cancel = 0_usize;
    let mut execution_units_before_cancel = 0_usize;
    let first_token = loop {
        let advance = session
            .advance_synchronized(stream)
            .map_err(|err| err.to_string())?;
        execution_units_before_cancel += 1;
        match advance {
            Sq8ServingAdvance::PromptProgress { .. } => prompt_progress_before_cancel += 1,
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
    let before_cancel = session.snapshot();
    if before_cancel.status != Sq8ServingRuntimeStatus::Decoding
        || before_cancel.prompt_tokens != prompt_tokens
        || before_cancel.prompt_tokens_processed != prompt_tokens
        || before_cancel.generated_tokens != 1
        || before_cancel.scheduler_active != 1
        || before_cancel.scheduler_waiting != 0
        || before_cancel.allocator.allocated_blocks != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
        || before_cancel
            .cache_lengths
            .iter()
            .any(|length| *length != prompt_tokens)
        || prompt_progress_before_cancel != expected_prefill_calls - 1
        || execution_units_before_cancel != expected_prefill_calls
    {
        return Err(format!(
            "decode cancel precondition snapshot mismatch: {before_cancel:?}"
        ));
    }
    cancel.cancel();
    if session
        .advance_synchronized(stream)
        .map_err(|err| err.to_string())?
        != Sq8ServingAdvance::CancellationObserved
    {
        return Err("cancel smoke published a token after cancellation observation".into());
    }
    let after_observation = session.snapshot();
    if after_observation.status != Sq8ServingRuntimeStatus::Cancelling
        || after_observation.active_request_id != before_cancel.active_request_id
        || after_observation.prompt_tokens != before_cancel.prompt_tokens
        || after_observation.prompt_tokens_processed != before_cancel.prompt_tokens_processed
        || after_observation.generated_tokens != before_cancel.generated_tokens
        || after_observation.cache_lengths != before_cancel.cache_lengths
        || after_observation.scheduler_active != before_cancel.scheduler_active
        || after_observation.scheduler_waiting != before_cancel.scheduler_waiting
        || after_observation.allocator != before_cancel.allocator
    {
        return Err(format!(
            "decode cancellation observation mutated request state: before={before_cancel:?} after={after_observation:?}"
        ));
    }
    let reset_start = Instant::now();
    let release = session
        .abort_and_reset_synchronized(stream)
        .map_err(|err| err.to_string())?;
    let reset_seconds = reset_start.elapsed().as_secs_f64();
    reusable_snapshot(session)?;
    if release.outcome != Sq8ReleaseOutcome::Cancelled
        || !release.reset_complete
        || release.request_id != request_id
        || release.prompt_tokens != prompt_tokens
        || release.generated_tokens != 1
    {
        return Err("cancel smoke release/reset contract failed".into());
    }
    Ok(CancelledCaseResult {
        request_id: request_id.to_string(),
        cancellation_phase: "decode",
        prompt_tokens,
        prompt_progress_before_cancel,
        generated_before_cancel: vec![first_token],
        execution_units_before_cancel,
        status_before_cancel: serving_status_name(before_cancel.status),
        cache_lengths_before_cancel: before_cancel.cache_lengths,
        scheduler_active_before_cancel: before_cancel.scheduler_active,
        scheduler_waiting_before_cancel: before_cancel.scheduler_waiting,
        allocated_blocks_before_cancel: before_cancel.allocator.allocated_blocks,
        status_after_observation: serving_status_name(after_observation.status),
        prompt_progress_after_observation: after_observation.prompt_tokens_processed,
        generated_tokens_after_observation: after_observation.generated_tokens,
        cache_lengths_after_observation: after_observation.cache_lengths,
        scheduler_active_after_observation: after_observation.scheduler_active,
        scheduler_waiting_after_observation: after_observation.scheduler_waiting,
        allocated_blocks_after_observation: after_observation.allocator.allocated_blocks,
        release_outcome: release_outcome_name(release.outcome),
        reset_seconds,
    })
}

fn run_cancel_during_prefill(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    request_id: &str,
    prompt_token_ids: Vec<usize>,
    cancel_after_prompt_progress: usize,
) -> Result<CancelledCaseResult, String> {
    let prompt_tokens = prompt_token_ids.len();
    if cancel_after_prompt_progress == 0 || cancel_after_prompt_progress >= prompt_tokens {
        return Err(format!(
            "prefill cancellation progress must be in 1..{prompt_tokens}, got {cancel_after_prompt_progress}"
        ));
    }
    let cancel = Sq8CancellationToken::new();
    session
        .start(
            Sq8ServingRequest::greedy(request_id, prompt_token_ids, 1),
            cancel.clone(),
            stream,
        )
        .map_err(|err| err.to_string())?;
    let mut observed_progress = 0_usize;
    let mut execution_calls_before_cancel = 0_usize;
    while observed_progress < cancel_after_prompt_progress {
        match session
            .advance_synchronized(stream)
            .map_err(|err| err.to_string())?
        {
            Sq8ServingAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            } if prompt_tokens_processed > observed_progress
                && prompt_tokens_processed <= cancel_after_prompt_progress
                && cache_len == prompt_tokens_processed
                && execution_width == prompt_tokens_processed - observed_progress =>
            {
                observed_progress = prompt_tokens_processed;
                execution_calls_before_cancel += 1;
            }
            advance => {
                return Err(format!(
                    "prefill cancel could not reach exact progress {cancel_after_prompt_progress} from {observed_progress}: {advance:?}"
                ));
            }
        }
    }
    let before_cancel = session.snapshot();
    if before_cancel.status != Sq8ServingRuntimeStatus::Prefilling
        || before_cancel.prompt_tokens != prompt_tokens
        || before_cancel.prompt_tokens_processed != cancel_after_prompt_progress
        || before_cancel.generated_tokens != 0
        || before_cancel.scheduler_active != 1
        || before_cancel.scheduler_waiting != 0
        || before_cancel.allocator.allocated_blocks != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
        || before_cancel
            .cache_lengths
            .iter()
            .any(|length| *length != cancel_after_prompt_progress)
    {
        return Err(format!(
            "prefill cancel precondition snapshot mismatch: {before_cancel:?}"
        ));
    }
    cancel.cancel();
    if session
        .advance_synchronized(stream)
        .map_err(|err| err.to_string())?
        != Sq8ServingAdvance::CancellationObserved
    {
        return Err("prefill cancel published progress or a token after cancellation".into());
    }
    let cancelling = session.snapshot();
    if cancelling.status != Sq8ServingRuntimeStatus::Cancelling
        || cancelling.active_request_id != before_cancel.active_request_id
        || cancelling.prompt_tokens != before_cancel.prompt_tokens
        || cancelling.prompt_tokens_processed != cancel_after_prompt_progress
        || cancelling.generated_tokens != 0
        || cancelling.cache_lengths != before_cancel.cache_lengths
        || cancelling.scheduler_active != before_cancel.scheduler_active
        || cancelling.scheduler_waiting != before_cancel.scheduler_waiting
        || cancelling.allocator != before_cancel.allocator
    {
        return Err(format!(
            "prefill cancellation observation mutated progress: {cancelling:?}"
        ));
    }
    let reset_start = Instant::now();
    let release = session
        .abort_and_reset_synchronized(stream)
        .map_err(|err| err.to_string())?;
    let reset_seconds = reset_start.elapsed().as_secs_f64();
    reusable_snapshot(session)?;
    if release.outcome != Sq8ReleaseOutcome::Cancelled
        || !release.reset_complete
        || release.request_id != request_id
        || release.prompt_tokens != prompt_tokens
        || release.generated_tokens != 0
    {
        return Err("prefill cancel release/reset contract failed".into());
    }
    Ok(CancelledCaseResult {
        request_id: request_id.to_string(),
        cancellation_phase: "prefill",
        prompt_tokens,
        prompt_progress_before_cancel: cancel_after_prompt_progress,
        generated_before_cancel: Vec::new(),
        execution_units_before_cancel: execution_calls_before_cancel,
        status_before_cancel: serving_status_name(before_cancel.status),
        cache_lengths_before_cancel: before_cancel.cache_lengths,
        scheduler_active_before_cancel: before_cancel.scheduler_active,
        scheduler_waiting_before_cancel: before_cancel.scheduler_waiting,
        allocated_blocks_before_cancel: before_cancel.allocator.allocated_blocks,
        status_after_observation: serving_status_name(cancelling.status),
        prompt_progress_after_observation: cancelling.prompt_tokens_processed,
        generated_tokens_after_observation: cancelling.generated_tokens,
        cache_lengths_after_observation: cancelling.cache_lengths,
        scheduler_active_after_observation: cancelling.scheduler_active,
        scheduler_waiting_after_observation: cancelling.scheduler_waiting,
        allocated_blocks_after_observation: cancelling.allocator.allocated_blocks,
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

fn serving_cases(options: &Options) -> Result<Vec<ServingCase>, String> {
    if let Some(lengths) = &options.prompt_lengths {
        return lengths
            .iter()
            .copied()
            .map(|prompt_tokens| {
                Ok(ServingCase {
                    request_id: format!("serving-smoke-p{prompt_tokens:04}"),
                    prompt_token_ids: (1..=prompt_tokens).collect(),
                    max_new_tokens: options.max_new_tokens,
                })
            })
            .collect();
    }
    let mut cases = vec![ServingCase {
        request_id: "serving-smoke-1".to_string(),
        prompt_token_ids: options.prompt_token_ids.clone(),
        max_new_tokens: options.max_new_tokens,
    }];
    if let Some(prompt_token_ids) = &options.second_prompt_token_ids {
        cases.push(ServingCase {
            request_id: "serving-smoke-2".to_string(),
            prompt_token_ids: prompt_token_ids.clone(),
            max_new_tokens: options.second_max_new_tokens,
        });
    }
    Ok(cases)
}

fn parse_options() -> Result<Options, String> {
    let mut artifact = None;
    let mut package = None;
    let mut prompt_token_ids = vec![1_usize];
    let mut max_new_tokens = 1_usize;
    let mut second_prompt_token_ids = None;
    let mut second_max_new_tokens = 1_usize;
    let mut prompt_lengths = None;
    let mut prefill_mode = Sq8ServingPrefillMode::SequentialM1;
    let mut prompt_token_ids_explicit = false;
    let mut cancel_after_first_token = false;
    let mut cancel_after_prompt_progress = None;
    let mut oracle_capture_dir = None;
    let mut result_json = None;
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
                prompt_token_ids_explicit = true;
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
            Some("--prompt-lengths") => {
                let value = args
                    .next()
                    .ok_or_else(|| "--prompt-lengths requires a value".to_string())?;
                prompt_lengths =
                    Some(parse_prompt_lengths(value.to_str().ok_or_else(|| {
                        "prompt lengths must be UTF-8".to_string()
                    })?)?);
            }
            Some("--prefill-mode") => {
                let value = args
                    .next()
                    .ok_or_else(|| "--prefill-mode requires a value".to_string())?;
                prefill_mode = parse_prefill_mode(
                    value
                        .to_str()
                        .ok_or_else(|| "prefill mode must be UTF-8".to_string())?,
                )?;
            }
            Some("--cancel-after-first-token") => cancel_after_first_token = true,
            Some("--cancel-after-prompt-progress") => {
                let value = args
                    .next()
                    .ok_or_else(|| "--cancel-after-prompt-progress requires a value".to_string())?;
                cancel_after_prompt_progress = Some(
                    value
                        .to_str()
                        .ok_or_else(|| "cancel progress must be UTF-8".to_string())?
                        .parse::<usize>()
                        .map_err(|err| format!("invalid cancel progress: {err}"))?,
                );
            }
            Some("--oracle-capture-dir") => {
                oracle_capture_dir =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--oracle-capture-dir requires a path".to_string()
                    })?));
            }
            Some("--result-json") => {
                result_json =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--result-json requires a path".to_string()
                    })?));
            }
            Some(value) => return Err(format!("unknown argument: {value}")),
            None => return Err("arguments must be UTF-8".into()),
        }
    }
    if prompt_lengths.is_some() && (prompt_token_ids_explicit || second_prompt_token_ids.is_some())
    {
        return Err(
            "--prompt-lengths cannot be combined with explicit prompt token ID arguments".into(),
        );
    }
    if cancel_after_first_token && cancel_after_prompt_progress.is_some() {
        return Err("cancellation modes must be mutually exclusive".into());
    }
    Ok(Options {
        artifact: artifact.ok_or_else(|| "--artifact is required".to_string())?,
        package: package.ok_or_else(|| "--package is required".to_string())?,
        prompt_token_ids,
        max_new_tokens,
        second_prompt_token_ids,
        second_max_new_tokens,
        prompt_lengths,
        prefill_mode,
        cancel_after_first_token,
        cancel_after_prompt_progress,
        oracle_capture_dir,
        result_json,
    })
}

fn parse_prefill_mode(value: &str) -> Result<Sq8ServingPrefillMode, String> {
    match value {
        "all-m1" => Ok(Sq8ServingPrefillMode::SequentialM1),
        "m8-chunk8" => Ok(Sq8ServingPrefillMode::FixedM8Chunks),
        _ => Err(format!(
            "prefill mode must be all-m1 or m8-chunk8, got {value:?}"
        )),
    }
}

fn prefill_mode_name(mode: Sq8ServingPrefillMode) -> &'static str {
    match mode {
        Sq8ServingPrefillMode::SequentialM1 => "all-m1",
        Sq8ServingPrefillMode::FixedM8Chunks => "m8-chunk8",
    }
}

fn expected_prefill_execution_calls(
    prompt_tokens: usize,
    mode: Sq8ServingPrefillMode,
) -> Result<usize, String> {
    if prompt_tokens == 0 {
        return Err("prefill execution call count requires a nonempty prompt".into());
    }
    Ok(match mode {
        Sq8ServingPrefillMode::SequentialM1 => prompt_tokens,
        Sq8ServingPrefillMode::FixedM8Chunks => {
            prompt_tokens / QWEN3_14B_SQ8_PREFILL_CHUNK_TOKENS
                + prompt_tokens % QWEN3_14B_SQ8_PREFILL_CHUNK_TOKENS
        }
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

fn parse_prompt_lengths(value: &str) -> Result<Vec<usize>, String> {
    if value.is_empty() {
        return Err("prompt lengths must not be empty".into());
    }
    let mut lengths = Vec::new();
    for part in value.split(',') {
        let length = part
            .parse::<usize>()
            .map_err(|err| format!("invalid prompt length {part:?}: {err}"))?;
        if length == 0 || length > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS {
            return Err(format!(
                "prompt length must be in 1..={QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS}, got {length}"
            ));
        }
        if lengths.contains(&length) {
            return Err(format!("duplicate prompt length: {length}"));
        }
        lengths.push(length);
    }
    Ok(lengths)
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

fn require_hip_kernel_guards(mode: Sq8ServingPrefillMode) -> Result<(), String> {
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV)
        .collect::<Vec<_>>();
    if mode == Sq8ServingPrefillMode::FixedM8Chunks {
        names.extend(QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV);
    }
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
