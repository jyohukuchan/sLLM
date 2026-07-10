// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::Value;
use std::fs;
use std::process;
use ullm_engine::sq8_serving_runtime::{
    QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS, QWEN3_14B_SQ8_SERVING_TOP_K, Sq8ServingSnapshot,
};

const RAW_SCHEMA_V1: &str = "ullm.sq8.serving_performance.raw.v1";
const RAW_SCHEMA_V2: &str = "ullm.sq8.serving_performance.raw.v2";
const WARMUP_RUNS: usize = 2;
const MEASURED_RUNS: usize = 5;
const TTFT_MAX_NEW_TOKENS: usize = 512;
const TTFT_PROMPT_LENGTHS: [usize; 5] = [32, 128, 512, 2048, 3584];
const DECODE_PROMPT_TOKENS: usize = 32;
const DECODE_GENERATED_TOKENS: usize = 64;
const DECODE_TOKENS: usize = DECODE_GENERATED_TOKENS - 1;
const AMD_SMI_GPU_INDEX: u64 = 2;
const R9700_KFD_GPU_ID: u64 = 51_545;
const R9700_BDF: &str = "0000:47:00.0";
const R9700_UUID: &str = "a8ff7551-0000-1000-80e9-ddefa2d60f55";
const DECODE_P50_TOKENS_PER_SECOND_MINIMUM: f64 = 15.0;
const DECODE_P95_INTER_TOKEN_SECONDS_MAXIMUM: f64 = 0.100;

#[derive(Debug, Serialize)]
struct PerformanceResult {
    schema_version: &'static str,
    runner_git_commit: String,
    runner_worktree_clean: bool,
    runner_binary_sha256: String,
    prefill_mode: &'static str,
    prefill_chunk_tokens: usize,
    prefill_implementation: String,
    warmup_runs: usize,
    measured_runs: usize,
    percentile_method: &'static str,
    vram_policy: &'static str,
    timer: TimerContract,
    sampling: SamplingContract,
    load_seconds: f64,
    artifact_content_sha256: String,
    package_manifest_sha256: String,
    device: ServingDeviceResult,
    kv_cache_bytes: usize,
    cache_blocks: usize,
    context_tokens: usize,
    environment: PerformanceEnvironment,
    initial_vram: VramCapture,
    ttft_cases: Vec<TtftCaseResult>,
    decode_case: DecodeCaseResult,
    final_vram: VramCapture,
    final_snapshot: PerformanceSnapshot,
}

#[derive(Debug, Serialize)]
struct TtftCaseResult {
    prompt_tokens: usize,
    max_new_tokens: usize,
    prompt_token_pattern: &'static str,
    prompt_token_ids_u32_le_sha256: String,
    p50_limit_seconds: f64,
    p95_limit_seconds: f64,
    metric_before: GpuMetricCapture,
    samples: Vec<TtftSample>,
    metric_after: GpuMetricCapture,
}

#[derive(Debug, Serialize)]
struct TtftSample {
    phase: &'static str,
    sample_index: usize,
    request_id: String,
    request_start_elapsed_ns: u64,
    first_token_elapsed_ns: u64,
    ttft_ns: u64,
    first_token_id: usize,
    first_token_cache_len: usize,
    prompt_execution_calls: usize,
    prompt_progress_events: usize,
    first_token_snapshot: PerformanceSnapshot,
    cancel_set_elapsed_ns: u64,
    cancellation_observed_elapsed_ns: u64,
    cancellation_snapshot: PerformanceSnapshot,
    reset_start_elapsed_ns: u64,
    reset_end_elapsed_ns: u64,
    reset_ns: u64,
    release_outcome: &'static str,
    release_generated_tokens: usize,
    release_reset_complete: bool,
    post_reset_snapshot: PerformanceSnapshot,
    vram_after_reset: VramCapture,
}

#[derive(Debug, Serialize)]
struct DecodeCaseResult {
    prompt_tokens: usize,
    max_new_tokens: usize,
    generated_tokens: usize,
    prompt_token_pattern: &'static str,
    prompt_token_ids_u32_le_sha256: String,
    decode_tokens_per_sample: usize,
    p50_tokens_per_second_minimum: f64,
    p95_inter_token_seconds_maximum: f64,
    p95_inter_token_pooling: &'static str,
    metric_before: GpuMetricCapture,
    samples: Vec<DecodeSample>,
    metric_after: GpuMetricCapture,
}

#[derive(Debug, Serialize)]
struct DecodeSample {
    phase: &'static str,
    sample_index: usize,
    request_id: String,
    request_start_elapsed_ns: u64,
    first_token_elapsed_ns: u64,
    last_token_elapsed_ns: u64,
    ttft_ns: u64,
    decode_duration_ns: u64,
    prompt_execution_calls: usize,
    prompt_progress_events: usize,
    execution_calls: usize,
    generated: Vec<GeneratedAvailability>,
    terminal_snapshot: PerformanceSnapshot,
    reset_start_elapsed_ns: u64,
    reset_end_elapsed_ns: u64,
    reset_ns: u64,
    release_outcome: &'static str,
    release_generated_tokens: usize,
    release_reset_complete: bool,
    post_reset_snapshot: PerformanceSnapshot,
    vram_after_reset: VramCapture,
}

#[derive(Debug, Serialize)]
struct GeneratedAvailability {
    generated_index: usize,
    token_id: usize,
    cache_len: usize,
    available_elapsed_ns: u64,
    terminal_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct PerformanceSnapshot {
    status: &'static str,
    active_request_id: Option<String>,
    prompt_tokens: usize,
    prompt_tokens_processed: usize,
    generated_tokens: usize,
    cache_lengths: Vec<usize>,
    scheduler_active: usize,
    scheduler_waiting: usize,
    block_size_tokens: u32,
    total_blocks: u32,
    free_blocks: usize,
    allocated_blocks: usize,
    free_runs: usize,
    largest_free_run: usize,
}

#[derive(Debug, Serialize)]
struct VramCapture {
    amd_smi_command_start_elapsed_ns: u64,
    amd_smi_command_end_elapsed_ns: u64,
    captured_elapsed_ns: u64,
    worker_pid: u32,
    amd_smi_gpu_index: u64,
    amd_smi_mem_usage_bytes: u64,
    amd_smi_process_raw_json: String,
    amd_smi_process_raw_sha256: String,
    kfd_gpu_id: u64,
    kfd_vram_bytes: u64,
    kfd_processes: Vec<KfdProcessVram>,
    unrelated_positive_kfd_pids: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct KfdProcessVram {
    pid: u32,
    vram_bytes: u64,
}

#[derive(Debug, Serialize)]
struct GpuMetricCapture {
    command_start_elapsed_ns: u64,
    command_end_elapsed_ns: u64,
    captured_elapsed_ns: u64,
    gpu_index: u64,
    hotspot_temperature_c: f64,
    socket_power_w: f64,
    gfx_clock_mhz: f64,
    memory_clock_mhz: f64,
    fabric_clock_mhz: f64,
    raw_json: String,
    raw_sha256: String,
}

#[derive(Debug, Serialize)]
struct PerformanceEnvironment {
    hip_visible_devices: String,
    amd_smi_version_raw: String,
    amd_smi_version_raw_sha256: String,
    amd_smi_list_raw_json: String,
    amd_smi_list_raw_sha256: String,
    target_gpu_index: u64,
    target_gpu_bdf: String,
    target_gpu_uuid: String,
    target_kfd_gpu_id: u64,
}

#[derive(Debug, Serialize)]
struct TimerContract {
    clock: &'static str,
    ttft_start: &'static str,
    ttft_end: &'static str,
    fixture_construction_included: bool,
    model_load_included: bool,
    cleanup_included: bool,
}

#[derive(Debug, Serialize)]
struct SamplingContract {
    method: &'static str,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    seed: i64,
    eos_token_ids: [usize; 2],
}

pub(super) fn validate_output_path(options: &Options) -> Result<(), String> {
    let path = options
        .result_json
        .as_ref()
        .ok_or_else(|| "performance gate requires --result-json".to_string())?;
    match fs::symlink_metadata(path) {
        Ok(_) => {
            return Err(format!(
                "performance result already exists: {}",
                path.display()
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "failed to inspect performance result {}: {error}",
                path.display()
            ));
        }
    }
    let parent = path.parent().filter(|value| !value.as_os_str().is_empty());
    let parent = parent.unwrap_or_else(|| Path::new("."));
    let metadata = fs::metadata(parent).map_err(|error| {
        format!(
            "failed to inspect performance result parent {}: {error}",
            parent.display()
        )
    })?;
    if !metadata.is_dir() {
        return Err(format!(
            "performance result parent is not a directory: {}",
            parent.display()
        ));
    }
    Ok(())
}

fn performance_schema_version(mode: Sq8ServingPrefillMode) -> Result<&'static str, String> {
    match mode {
        Sq8ServingPrefillMode::FixedM8Chunks => Ok(RAW_SCHEMA_V1),
        Sq8ServingPrefillMode::FixedM128Chunks => Ok(RAW_SCHEMA_V2),
        _ => Err("performance gate requires the fixed M=8 or M=128 prefill mode".into()),
    }
}

fn performance_prompt_execution_calls(
    prompt_tokens: usize,
    prefill_chunk_tokens: usize,
) -> Result<usize, String> {
    if prompt_tokens == 0 || prefill_chunk_tokens == 0 {
        return Err("performance prompt execution count requires nonzero dimensions".into());
    }
    (prompt_tokens / prefill_chunk_tokens)
        .checked_add(prompt_tokens % prefill_chunk_tokens)
        .ok_or_else(|| "performance prompt execution call count overflows".to_string())
}

#[allow(clippy::too_many_arguments)]
fn validate_performance_prompt_progress(
    request_id: &str,
    prompt_tokens: usize,
    prefill_chunk_tokens: usize,
    expected_prompt_tokens_processed: &mut usize,
    prompt_tokens_processed: usize,
    cache_len: usize,
    execution_width: usize,
) -> Result<(), String> {
    let remaining = prompt_tokens
        .checked_sub(*expected_prompt_tokens_processed)
        .ok_or_else(|| format!("prompt progress moved beyond {request_id}"))?;
    let expected_width = if remaining >= prefill_chunk_tokens {
        prefill_chunk_tokens
    } else {
        1
    };
    let expected_processed = expected_prompt_tokens_processed
        .checked_add(expected_width)
        .ok_or_else(|| format!("prompt progress overflows for {request_id}"))?;
    if execution_width != expected_width
        || prompt_tokens_processed != expected_processed
        || cache_len != expected_processed
        || expected_processed >= prompt_tokens
    {
        return Err(format!(
            "prompt progress mismatch for {request_id}: width={execution_width} \
             processed={prompt_tokens_processed} cache_len={cache_len} \
             expected_width={expected_width} expected_processed={expected_processed}"
        ));
    }
    *expected_prompt_tokens_processed = expected_processed;
    Ok(())
}

pub(super) fn run(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    options: &Options,
    runner_identity: RunnerIdentity,
    load_seconds: f64,
) -> Result<(), String> {
    validate_output_path(options)?;
    let load_report = session.load_report();
    let prefill_mode = load_report.prefill_mode;
    let prefill_chunk_tokens = load_report.prefill_chunk_tokens;
    let schema_version = performance_schema_version(prefill_mode)?;
    let origin = Instant::now();
    let environment = capture_environment()?;
    let initial_vram = capture_vram(&origin)?;
    let mut ttft_cases = Vec::with_capacity(TTFT_PROMPT_LENGTHS.len());
    for prompt_tokens in TTFT_PROMPT_LENGTHS {
        let (p50_limit_seconds, p95_limit_seconds) = ttft_limits(prompt_tokens)?;
        let metric_before = capture_gpu_metric(&origin)?;
        let mut samples = Vec::with_capacity(WARMUP_RUNS + MEASURED_RUNS);
        for (phase, count) in [("warmup", WARMUP_RUNS), ("measured", MEASURED_RUNS)] {
            for sample_index in 0..count {
                samples.push(run_ttft_sample(
                    session,
                    stream,
                    &origin,
                    prompt_tokens,
                    prefill_chunk_tokens,
                    phase,
                    sample_index,
                )?);
            }
        }
        let metric_after = capture_gpu_metric(&origin)?;
        ttft_cases.push(TtftCaseResult {
            prompt_tokens,
            max_new_tokens: TTFT_MAX_NEW_TOKENS,
            prompt_token_pattern: "ascending_u32_1_through_prompt_tokens",
            prompt_token_ids_u32_le_sha256: ascending_prompt_sha256(prompt_tokens)?,
            p50_limit_seconds,
            p95_limit_seconds,
            metric_before,
            samples,
            metric_after,
        });
    }

    let decode_metric_before = capture_gpu_metric(&origin)?;
    let mut decode_samples = Vec::with_capacity(WARMUP_RUNS + MEASURED_RUNS);
    for (phase, count) in [("warmup", WARMUP_RUNS), ("measured", MEASURED_RUNS)] {
        for sample_index in 0..count {
            decode_samples.push(run_decode_sample(
                session,
                stream,
                &origin,
                prefill_chunk_tokens,
                phase,
                sample_index,
            )?);
        }
    }
    let decode_metric_after = capture_gpu_metric(&origin)?;
    let decode_case = DecodeCaseResult {
        prompt_tokens: DECODE_PROMPT_TOKENS,
        max_new_tokens: DECODE_GENERATED_TOKENS,
        generated_tokens: DECODE_GENERATED_TOKENS,
        prompt_token_pattern: "ascending_u32_1_through_prompt_tokens",
        prompt_token_ids_u32_le_sha256: ascending_prompt_sha256(DECODE_PROMPT_TOKENS)?,
        decode_tokens_per_sample: DECODE_TOKENS,
        p50_tokens_per_second_minimum: DECODE_P50_TOKENS_PER_SECOND_MINIMUM,
        p95_inter_token_seconds_maximum: DECODE_P95_INTER_TOKEN_SECONDS_MAXIMUM,
        p95_inter_token_pooling: "all_measured_inter_token_intervals",
        metric_before: decode_metric_before,
        samples: decode_samples,
        metric_after: decode_metric_after,
    };
    let final_snapshot = reusable_performance_snapshot(session)?;
    let final_vram = capture_vram(&origin)?;
    let load_report = session.load_report();
    let result = PerformanceResult {
        schema_version,
        runner_git_commit: runner_identity.git_commit,
        runner_worktree_clean: runner_identity.worktree_clean,
        runner_binary_sha256: runner_identity.binary_sha256,
        prefill_mode: prefill_mode_name(load_report.prefill_mode),
        prefill_chunk_tokens: load_report.prefill_chunk_tokens,
        prefill_implementation: load_report.prefill_implementation.clone(),
        warmup_runs: WARMUP_RUNS,
        measured_runs: MEASURED_RUNS,
        percentile_method: "linear_interpolation_rank_(n-1)*p",
        vram_policy: "record_and_cross_check_each_sample_no_stability_gate",
        timer: TimerContract {
            clock: "std::time::Instant_monotonic",
            ttft_start: "immediately_before_session.start",
            ttft_end: "immediately_after_first_token_return_before_snapshot",
            fixture_construction_included: false,
            model_load_included: false,
            cleanup_included: false,
        },
        sampling: SamplingContract {
            method: "greedy_temperature_zero",
            temperature: 0.0,
            top_p: 1.0,
            top_k: QWEN3_14B_SQ8_SERVING_TOP_K,
            seed: 0,
            eos_token_ids: QWEN3_14B_SQ8_SERVING_EOS_TOKEN_IDS,
        },
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
        environment,
        initial_vram,
        ttft_cases,
        decode_case,
        final_vram,
        final_snapshot,
    };
    let serialized = serde_json::to_string_pretty(&result)
        .map_err(|error| format!("failed to serialize performance result: {error}"))?;
    write_bytes_create_new(
        options
            .result_json
            .as_deref()
            .expect("performance output was validated"),
        serialized.as_bytes(),
    )?;
    println!("{serialized}");
    Ok(())
}

fn run_ttft_sample(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    origin: &Instant,
    prompt_tokens: usize,
    prefill_chunk_tokens: usize,
    phase: &'static str,
    sample_index: usize,
) -> Result<TtftSample, String> {
    let request_id = format!("perf-ttft-p{prompt_tokens:04}-{phase}-{sample_index}");
    let request = Sq8ServingRequest::greedy(
        &request_id,
        (1..=prompt_tokens).collect(),
        TTFT_MAX_NEW_TOKENS,
    );
    let cancel = Sq8CancellationToken::new();
    let expected_prompt_calls =
        performance_prompt_execution_calls(prompt_tokens, prefill_chunk_tokens)?;
    let mut prompt_execution_calls = 0_usize;
    let mut prompt_progress_events = 0_usize;
    let mut expected_prompt_tokens_processed = 0_usize;
    let request_start_elapsed_ns = elapsed_ns(origin)?;
    session
        .start(request, cancel.clone(), stream)
        .map_err(|error| error.to_string())?;
    let (first_token_id, first_token_cache_len, first_token_elapsed_ns) = loop {
        let advance = session
            .advance_synchronized(stream)
            .map_err(|error| error.to_string())?;
        prompt_execution_calls += 1;
        match advance {
            Sq8ServingAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            } => {
                prompt_progress_events += 1;
                validate_performance_prompt_progress(
                    &request_id,
                    prompt_tokens,
                    prefill_chunk_tokens,
                    &mut expected_prompt_tokens_processed,
                    prompt_tokens_processed,
                    cache_len,
                    execution_width,
                )?;
            }
            Sq8ServingAdvance::Token {
                token_id,
                generated_index,
                cache_len,
                terminal_reason,
            } => {
                let available_elapsed_ns = elapsed_ns(origin)?;
                if generated_index != 0 || cache_len != prompt_tokens || terminal_reason.is_some() {
                    return Err(format!(
                        "TTFT request did not produce a cancellable first token: {advance:?}"
                    ));
                }
                break (token_id, cache_len, available_elapsed_ns);
            }
            advance => {
                return Err(format!(
                    "TTFT request did not produce a cancellable first token: {advance:?}"
                ));
            }
        }
    };
    if prompt_execution_calls != expected_prompt_calls
        || prompt_progress_events + 1 != expected_prompt_calls
    {
        return Err(format!(
            "TTFT prompt execution count mismatch for {request_id}: \
             calls={prompt_execution_calls} progress={prompt_progress_events}"
        ));
    }
    let first_snapshot_raw = session.snapshot();
    validate_active_snapshot(
        &first_snapshot_raw,
        Sq8ServingRuntimeStatus::Decoding,
        prompt_tokens,
        1,
        prompt_tokens,
        &request_id,
    )?;
    let first_token_snapshot = performance_snapshot(first_snapshot_raw);

    cancel.cancel();
    let cancel_set_elapsed_ns = elapsed_ns(origin)?;
    let cancellation = session
        .advance_synchronized(stream)
        .map_err(|error| error.to_string())?;
    let cancellation_observed_elapsed_ns = elapsed_ns(origin)?;
    if cancellation != Sq8ServingAdvance::CancellationObserved {
        return Err(format!(
            "TTFT request did not observe cancellation: {cancellation:?}"
        ));
    }
    let cancellation_raw = session.snapshot();
    validate_active_snapshot(
        &cancellation_raw,
        Sq8ServingRuntimeStatus::Cancelling,
        prompt_tokens,
        1,
        prompt_tokens,
        &request_id,
    )?;
    let cancellation_snapshot = performance_snapshot(cancellation_raw);
    let reset_start_elapsed_ns = elapsed_ns(origin)?;
    let release = session
        .abort_and_reset_synchronized(stream)
        .map_err(|error| error.to_string())?;
    let reset_end_elapsed_ns = elapsed_ns(origin)?;
    if release.outcome != Sq8ReleaseOutcome::Cancelled
        || !release.reset_complete
        || release.request_id != request_id
        || release.prompt_tokens != prompt_tokens
        || release.generated_tokens != 1
    {
        return Err(format!("TTFT release contract failed: {release:?}"));
    }
    let post_reset_snapshot = reusable_performance_snapshot(session)?;
    let vram_after_reset = capture_vram(origin)?;
    Ok(TtftSample {
        phase,
        sample_index,
        request_id,
        request_start_elapsed_ns,
        first_token_elapsed_ns,
        ttft_ns: first_token_elapsed_ns
            .checked_sub(request_start_elapsed_ns)
            .ok_or_else(|| "TTFT timestamp moved backwards".to_string())?,
        first_token_id,
        first_token_cache_len,
        prompt_execution_calls,
        prompt_progress_events,
        first_token_snapshot,
        cancel_set_elapsed_ns,
        cancellation_observed_elapsed_ns,
        cancellation_snapshot,
        reset_start_elapsed_ns,
        reset_end_elapsed_ns,
        reset_ns: reset_end_elapsed_ns
            .checked_sub(reset_start_elapsed_ns)
            .ok_or_else(|| "TTFT reset timestamp moved backwards".to_string())?,
        release_outcome: release_outcome_name(release.outcome),
        release_generated_tokens: release.generated_tokens,
        release_reset_complete: release.reset_complete,
        post_reset_snapshot,
        vram_after_reset,
    })
}

fn run_decode_sample(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    origin: &Instant,
    prefill_chunk_tokens: usize,
    phase: &'static str,
    sample_index: usize,
) -> Result<DecodeSample, String> {
    let request_id = format!("perf-decode-p0032-g0064-{phase}-{sample_index}");
    let request = Sq8ServingRequest::greedy(
        &request_id,
        (1..=DECODE_PROMPT_TOKENS).collect(),
        DECODE_GENERATED_TOKENS,
    );
    let cancel = Sq8CancellationToken::new();
    let expected_prompt_calls =
        performance_prompt_execution_calls(DECODE_PROMPT_TOKENS, prefill_chunk_tokens)?;
    let mut prompt_progress_events = 0_usize;
    let mut expected_prompt_tokens_processed = 0_usize;
    let mut execution_calls = 0_usize;
    let mut generated = Vec::with_capacity(DECODE_GENERATED_TOKENS);
    let request_start_elapsed_ns = elapsed_ns(origin)?;
    session
        .start(request, cancel, stream)
        .map_err(|error| error.to_string())?;
    while generated.len() < DECODE_GENERATED_TOKENS {
        let advance = session
            .advance_synchronized(stream)
            .map_err(|error| error.to_string())?;
        execution_calls += 1;
        match advance {
            Sq8ServingAdvance::PromptProgress {
                prompt_tokens_processed,
                cache_len,
                execution_width,
            } => {
                prompt_progress_events += 1;
                validate_performance_prompt_progress(
                    &request_id,
                    DECODE_PROMPT_TOKENS,
                    prefill_chunk_tokens,
                    &mut expected_prompt_tokens_processed,
                    prompt_tokens_processed,
                    cache_len,
                    execution_width,
                )?;
            }
            Sq8ServingAdvance::Token {
                token_id,
                generated_index,
                cache_len,
                terminal_reason,
            } => {
                let available_elapsed_ns = elapsed_ns(origin)?;
                let expected_index = generated.len();
                let expected_cache_len = DECODE_PROMPT_TOKENS + expected_index;
                let expected_reason = (expected_index + 1 == DECODE_GENERATED_TOKENS)
                    .then_some(Sq8FinishReason::Length);
                if generated_index != expected_index
                    || cache_len != expected_cache_len
                    || terminal_reason != expected_reason
                {
                    return Err(format!(
                        "decode token transition mismatch for {request_id}: {advance:?}"
                    ));
                }
                generated.push(GeneratedAvailability {
                    generated_index,
                    token_id,
                    cache_len,
                    available_elapsed_ns,
                    terminal_reason: terminal_reason.map(finish_reason_name),
                });
            }
            Sq8ServingAdvance::CancellationObserved => {
                return Err(format!(
                    "decode request was unexpectedly cancelled: {request_id}"
                ));
            }
        }
    }
    let expected_execution_calls = expected_prompt_calls + DECODE_TOKENS;
    if prompt_progress_events + 1 != expected_prompt_calls
        || execution_calls != expected_execution_calls
    {
        return Err(format!(
            "decode execution count mismatch for {request_id}: \
             calls={execution_calls} progress={prompt_progress_events}"
        ));
    }
    let terminal_raw = session.snapshot();
    validate_active_snapshot(
        &terminal_raw,
        Sq8ServingRuntimeStatus::Finishing,
        DECODE_PROMPT_TOKENS,
        DECODE_GENERATED_TOKENS,
        DECODE_PROMPT_TOKENS + DECODE_TOKENS,
        &request_id,
    )?;
    let terminal_snapshot = performance_snapshot(terminal_raw);
    let first_token_elapsed_ns = generated
        .first()
        .ok_or_else(|| "decode sample emitted no token".to_string())?
        .available_elapsed_ns;
    let last_token_elapsed_ns = generated
        .last()
        .ok_or_else(|| "decode sample emitted no final token".to_string())?
        .available_elapsed_ns;
    let reset_start_elapsed_ns = elapsed_ns(origin)?;
    let release = session
        .finish_and_reset_synchronized(stream)
        .map_err(|error| error.to_string())?;
    let reset_end_elapsed_ns = elapsed_ns(origin)?;
    if release.outcome != Sq8ReleaseOutcome::Length
        || !release.reset_complete
        || release.request_id != request_id
        || release.prompt_tokens != DECODE_PROMPT_TOKENS
        || release.generated_tokens != DECODE_GENERATED_TOKENS
    {
        return Err(format!("decode release contract failed: {release:?}"));
    }
    let post_reset_snapshot = reusable_performance_snapshot(session)?;
    let vram_after_reset = capture_vram(origin)?;
    Ok(DecodeSample {
        phase,
        sample_index,
        request_id,
        request_start_elapsed_ns,
        first_token_elapsed_ns,
        last_token_elapsed_ns,
        ttft_ns: first_token_elapsed_ns
            .checked_sub(request_start_elapsed_ns)
            .ok_or_else(|| "decode TTFT timestamp moved backwards".to_string())?,
        decode_duration_ns: last_token_elapsed_ns
            .checked_sub(first_token_elapsed_ns)
            .ok_or_else(|| "decode timestamps moved backwards".to_string())?,
        prompt_execution_calls: expected_prompt_calls,
        prompt_progress_events,
        execution_calls,
        generated,
        terminal_snapshot,
        reset_start_elapsed_ns,
        reset_end_elapsed_ns,
        reset_ns: reset_end_elapsed_ns
            .checked_sub(reset_start_elapsed_ns)
            .ok_or_else(|| "decode reset timestamp moved backwards".to_string())?,
        release_outcome: release_outcome_name(release.outcome),
        release_generated_tokens: release.generated_tokens,
        release_reset_complete: release.reset_complete,
        post_reset_snapshot,
        vram_after_reset,
    })
}

fn validate_active_snapshot(
    snapshot: &Sq8ServingSnapshot,
    status: Sq8ServingRuntimeStatus,
    prompt_tokens: usize,
    generated_tokens: usize,
    cache_len: usize,
    request_id: &str,
) -> Result<(), String> {
    if snapshot.status != status
        || snapshot.active_request_id.as_deref() != Some(request_id)
        || snapshot.prompt_tokens != prompt_tokens
        || snapshot.prompt_tokens_processed != prompt_tokens
        || snapshot.generated_tokens != generated_tokens
        || snapshot.cache_lengths.len() != 40
        || snapshot
            .cache_lengths
            .iter()
            .any(|length| *length != cache_len)
        || snapshot.scheduler_active != 1
        || snapshot.scheduler_waiting != 0
        || snapshot.allocator.block_size_tokens
            != u32::try_from(QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS)
                .expect("serving block tokens fit u32")
        || snapshot.allocator.total_blocks
            != u32::try_from(QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS)
                .expect("serving cache blocks fit u32")
        || snapshot.allocator.free_blocks != 0
        || snapshot.allocator.allocated_blocks != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
        || snapshot.allocator.free_runs != 0
        || snapshot.allocator.largest_free_run != 0
    {
        return Err(format!(
            "performance active snapshot mismatch for {request_id}: {snapshot:?}"
        ));
    }
    Ok(())
}

fn reusable_performance_snapshot(
    session: &Qwen3Sq8ServingSession,
) -> Result<PerformanceSnapshot, String> {
    let snapshot = reusable_snapshot(session)?;
    if snapshot.allocator.block_size_tokens
        != u32::try_from(QWEN3_14B_SQ8_SERVING_BLOCK_TOKENS).expect("serving block tokens fit u32")
        || snapshot.allocator.total_blocks
            != u32::try_from(QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS)
                .expect("serving cache blocks fit u32")
        || snapshot.allocator.free_blocks != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
        || snapshot.allocator.free_runs != 1
        || snapshot.allocator.largest_free_run != QWEN3_14B_SQ8_SERVING_CACHE_BLOCKS
    {
        return Err(format!(
            "performance reusable allocator baseline mismatch: {:?}",
            snapshot.allocator
        ));
    }
    Ok(performance_snapshot(snapshot))
}

fn performance_snapshot(snapshot: Sq8ServingSnapshot) -> PerformanceSnapshot {
    PerformanceSnapshot {
        status: serving_status_name(snapshot.status),
        active_request_id: snapshot.active_request_id,
        prompt_tokens: snapshot.prompt_tokens,
        prompt_tokens_processed: snapshot.prompt_tokens_processed,
        generated_tokens: snapshot.generated_tokens,
        cache_lengths: snapshot.cache_lengths,
        scheduler_active: snapshot.scheduler_active,
        scheduler_waiting: snapshot.scheduler_waiting,
        block_size_tokens: snapshot.allocator.block_size_tokens,
        total_blocks: snapshot.allocator.total_blocks,
        free_blocks: snapshot.allocator.free_blocks,
        allocated_blocks: snapshot.allocator.allocated_blocks,
        free_runs: snapshot.allocator.free_runs,
        largest_free_run: snapshot.allocator.largest_free_run,
    }
}

fn capture_vram(origin: &Instant) -> Result<VramCapture, String> {
    let worker_pid = process::id();
    let amd_smi_command_start_elapsed_ns = elapsed_ns(origin)?;
    let raw = command_stdout("amd-smi", &["process", "--gpu", "2", "--general", "--json"])?;
    let amd_smi_command_end_elapsed_ns = elapsed_ns(origin)?;
    let document: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse amd-smi process JSON: {error}"))?;
    let gpu = document
        .as_array()
        .filter(|values| values.len() == 1)
        .and_then(|values| values.first())
        .and_then(Value::as_object)
        .ok_or_else(|| "amd-smi process JSON must contain one GPU object".to_string())?;
    let gpu_index = gpu
        .get("gpu")
        .and_then(Value::as_u64)
        .ok_or_else(|| "amd-smi process GPU index is missing".to_string())?;
    let processes = gpu
        .get("process_list")
        .and_then(Value::as_array)
        .filter(|values| values.len() == 1)
        .ok_or_else(|| "R9700 must contain exactly one AMD SMI process record".to_string())?;
    let info = processes[0]
        .get("process_info")
        .and_then(Value::as_object)
        .ok_or_else(|| "AMD SMI process record is not an object".to_string())?;
    let amd_pid = info
        .get("pid")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| "AMD SMI process PID is invalid".to_string())?;
    let memory = info
        .get("mem_usage")
        .and_then(Value::as_object)
        .ok_or_else(|| "AMD SMI process memory is missing".to_string())?;
    let amd_smi_mem_usage_bytes = memory
        .get("value")
        .and_then(Value::as_u64)
        .ok_or_else(|| "AMD SMI process memory value is invalid".to_string())?;
    if gpu_index != AMD_SMI_GPU_INDEX
        || amd_pid != worker_pid
        || memory.get("unit").and_then(Value::as_str) != Some("B")
        || amd_smi_mem_usage_bytes == 0
    {
        return Err(format!(
            "AMD SMI does not identify the sole R9700 worker: gpu={gpu_index} \
             pid={amd_pid} expected_pid={worker_pid} bytes={amd_smi_mem_usage_bytes}"
        ));
    }

    let mut kfd_processes = Vec::new();
    let root = Path::new("/sys/class/kfd/kfd/proc");
    for entry in fs::read_dir(root)
        .map_err(|error| format!("failed to enumerate {}: {error}", root.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to read KFD process entry: {error}"))?;
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| "KFD process directory name is not UTF-8".to_string())?;
        let Ok(pid) = name.parse::<u32>() else {
            continue;
        };
        let path = entry.path().join(format!("vram_{R9700_KFD_GPU_ID}"));
        let value = match fs::read_to_string(&path) {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(format!("failed to read {}: {error}", path.display()));
            }
        };
        let vram_bytes = value
            .trim()
            .parse::<u64>()
            .map_err(|error| format!("invalid KFD VRAM value in {}: {error}", path.display()))?;
        kfd_processes.push(KfdProcessVram { pid, vram_bytes });
    }
    kfd_processes.sort_by_key(|entry| entry.pid);
    let kfd_vram_bytes = kfd_processes
        .iter()
        .find(|entry| entry.pid == worker_pid)
        .map(|entry| entry.vram_bytes)
        .ok_or_else(|| "worker has no R9700 KFD VRAM record".to_string())?;
    let unrelated_positive_kfd_pids = kfd_processes
        .iter()
        .filter(|entry| entry.pid != worker_pid && entry.vram_bytes > 0)
        .map(|entry| entry.pid)
        .collect::<Vec<_>>();
    if !unrelated_positive_kfd_pids.is_empty()
        || kfd_vram_bytes == 0
        || kfd_vram_bytes != amd_smi_mem_usage_bytes
    {
        return Err(format!(
            "R9700 VRAM isolation failed: amd={amd_smi_mem_usage_bytes} \
             kfd={kfd_vram_bytes} unrelated={unrelated_positive_kfd_pids:?}"
        ));
    }
    Ok(VramCapture {
        amd_smi_command_start_elapsed_ns,
        amd_smi_command_end_elapsed_ns,
        captured_elapsed_ns: elapsed_ns(origin)?,
        worker_pid,
        amd_smi_gpu_index: gpu_index,
        amd_smi_mem_usage_bytes,
        amd_smi_process_raw_sha256: sha256_bytes(raw.as_bytes()),
        amd_smi_process_raw_json: raw,
        kfd_gpu_id: R9700_KFD_GPU_ID,
        kfd_vram_bytes,
        kfd_processes,
        unrelated_positive_kfd_pids,
    })
}

fn capture_environment() -> Result<PerformanceEnvironment, String> {
    let hip_visible_devices = std::env::var("HIP_VISIBLE_DEVICES")
        .map_err(|_| "performance gate requires HIP_VISIBLE_DEVICES=1".to_string())?;
    if hip_visible_devices != "1" {
        return Err(format!(
            "performance gate requires HIP_VISIBLE_DEVICES=1, got {hip_visible_devices:?}"
        ));
    }
    let version = command_stdout("amd-smi", &["version"])?;
    for required in [
        "AMDSMI Tool: 26.2.2+e1a6bc5663",
        "AMDSMI Library version: 26.2.2",
        "ROCm version: 7.2.1",
    ] {
        if !version.contains(required) {
            return Err(format!(
                "amd-smi version no longer matches the frozen environment: missing {required:?}"
            ));
        }
    }
    let list = command_stdout("amd-smi", &["list", "--json"])?;
    let document: Value = serde_json::from_str(&list)
        .map_err(|error| format!("failed to parse amd-smi list JSON: {error}"))?;
    let entries = document
        .as_array()
        .ok_or_else(|| "amd-smi list JSON root is not an array".to_string())?;
    let matches = entries
        .iter()
        .filter(|entry| entry.get("gpu").and_then(Value::as_u64) == Some(AMD_SMI_GPU_INDEX))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "amd-smi list must contain exactly one GPU {} entry, found {}",
            AMD_SMI_GPU_INDEX,
            matches.len()
        ));
    }
    let target = matches[0];
    let bdf = target
        .get("bdf")
        .and_then(Value::as_str)
        .ok_or_else(|| "R9700 BDF is missing from amd-smi list".to_string())?;
    let uuid = target
        .get("uuid")
        .and_then(Value::as_str)
        .ok_or_else(|| "R9700 UUID is missing from amd-smi list".to_string())?;
    let kfd_gpu_id = target
        .get("kfd_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| "R9700 KFD ID is missing from amd-smi list".to_string())?;
    if bdf != R9700_BDF || uuid != R9700_UUID || kfd_gpu_id != R9700_KFD_GPU_ID {
        return Err(format!(
            "amd-smi GPU2 identity mismatch: bdf={bdf} uuid={uuid} kfd_id={kfd_gpu_id}"
        ));
    }
    Ok(PerformanceEnvironment {
        hip_visible_devices,
        amd_smi_version_raw_sha256: sha256_bytes(version.as_bytes()),
        amd_smi_version_raw: version,
        amd_smi_list_raw_sha256: sha256_bytes(list.as_bytes()),
        amd_smi_list_raw_json: list,
        target_gpu_index: AMD_SMI_GPU_INDEX,
        target_gpu_bdf: bdf.to_string(),
        target_gpu_uuid: uuid.to_string(),
        target_kfd_gpu_id: kfd_gpu_id,
    })
}

fn capture_gpu_metric(origin: &Instant) -> Result<GpuMetricCapture, String> {
    let command_start_elapsed_ns = elapsed_ns(origin)?;
    let raw = command_stdout("amd-smi", &["metric", "--gpu", "2", "--json"])?;
    let command_end_elapsed_ns = elapsed_ns(origin)?;
    let document: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse amd-smi metric JSON: {error}"))?;
    let gpu = document
        .pointer("/gpu_data/0")
        .and_then(Value::as_object)
        .ok_or_else(|| "amd-smi metric JSON is missing gpu_data[0]".to_string())?;
    let gpu_index = gpu
        .get("gpu")
        .and_then(Value::as_u64)
        .ok_or_else(|| "amd-smi metric GPU index is missing".to_string())?;
    if gpu_index != AMD_SMI_GPU_INDEX {
        return Err(format!(
            "amd-smi metric returned GPU {gpu_index}, expected {AMD_SMI_GPU_INDEX}"
        ));
    }
    Ok(GpuMetricCapture {
        command_start_elapsed_ns,
        command_end_elapsed_ns,
        captured_elapsed_ns: command_end_elapsed_ns,
        gpu_index,
        hotspot_temperature_c: metric_quantity(&document, "/gpu_data/0/temperature/hotspot", "C")?,
        socket_power_w: metric_quantity(&document, "/gpu_data/0/power/socket_power", "W")?,
        gfx_clock_mhz: metric_quantity(&document, "/gpu_data/0/clock/gfx_0/clk", "MHz")?,
        memory_clock_mhz: metric_quantity(&document, "/gpu_data/0/clock/mem_0/clk", "MHz")?,
        fabric_clock_mhz: metric_quantity(&document, "/gpu_data/0/clock/fclk_0/clk", "MHz")?,
        raw_sha256: sha256_bytes(raw.as_bytes()),
        raw_json: raw,
    })
}

fn metric_quantity(document: &Value, pointer: &str, expected_unit: &str) -> Result<f64, String> {
    let quantity = document
        .pointer(pointer)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("amd-smi metric is missing {pointer}"))?;
    let value = quantity
        .get("value")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| format!("amd-smi metric value is invalid at {pointer}"))?;
    if quantity.get("unit").and_then(Value::as_str) != Some(expected_unit) {
        return Err(format!(
            "amd-smi metric unit at {pointer} is not {expected_unit}"
        ));
    }
    Ok(value)
}

fn command_stdout(program: &str, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| format!("failed to execute {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("{program} output is not UTF-8: {error}"))
}

fn elapsed_ns(origin: &Instant) -> Result<u64, String> {
    u64::try_from(origin.elapsed().as_nanos())
        .map_err(|_| "performance monotonic timestamp exceeds u64".to_string())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn ascending_prompt_sha256(prompt_tokens: usize) -> Result<String, String> {
    let mut digest = Sha256::new();
    for token_id in 1..=prompt_tokens {
        let token_id = u32::try_from(token_id)
            .map_err(|_| format!("prompt token ID {token_id} does not fit u32"))?;
        digest.update(token_id.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn ttft_limits(prompt_tokens: usize) -> Result<(f64, f64), String> {
    match prompt_tokens {
        32 => Ok((2.5, 3.0)),
        128 => Ok((4.0, 5.0)),
        512 => Ok((10.0, 12.0)),
        2048 => Ok((30.0, 35.0)),
        3584 => Ok((50.0, 60.0)),
        _ => Err(format!("no TTFT gate is frozen for prompt {prompt_tokens}")),
    }
}
