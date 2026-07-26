// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Isolated R9700-only full-model steady-decode measurement for the SQ8_0
//! paged attention path.  It intentionally times only the 16 feedback-token
//! advances after a 1024-token M=128 seed and four unmeasured decode warmups.

use serde::Serialize;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq8_embedding_runtime::QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::sq8_layer_runtime::{
    QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV,
    QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
};
use ullm_engine::sq8_model_head_runtime::{
    QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV, validate_qwen3_14b_sq8_r9700_device_info,
};
use ullm_engine::sq8_serving_runtime::{
    QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS, Qwen3Sq8ServingSession, Sq8CancellationToken,
    Sq8FinishReason, Sq8ReleaseOutcome, Sq8ServingAdvance, Sq8ServingPrefillMode,
    Sq8ServingRequest, load_qwen3_14b_sq8_serving_norms,
};
use ullm_runtime_sys::{DeviceInfo, RuntimeContext, RuntimeStream, device_count, device_info};

const UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_PROMPT_TOKENS: usize = 1024;
const DEFAULT_WARMUP_STEPS: usize = 4;
const DEFAULT_MEASURED_STEPS: usize = 16;
const DEFAULT_REPEATS: usize = 5;
const WAVE_SCALAR_ENV: &str = "ULLM_EXPERIMENTAL_PAGED_DECODE_WAVE_SCALAR_SOFTMAX";

#[derive(Debug)]
struct Options {
    artifact: PathBuf,
    package: PathBuf,
    output: PathBuf,
    prompt_tokens: usize,
    warmup_steps: usize,
    measured_steps: usize,
    repeats: usize,
}

#[derive(Debug, Serialize)]
struct ResultFile {
    schema_version: &'static str,
    scope: &'static str,
    input_token_pattern: &'static str,
    test_only_ignore_eos: bool,
    wave_scalar_softmax_enabled: bool,
    runner_git_head: String,
    device: DeviceSummary,
    prefill_mode: &'static str,
    prompt_tokens: usize,
    warmup_decode_steps: usize,
    measured_decode_steps: usize,
    repeats: usize,
    load_seconds: f64,
    artifact_content_sha256: String,
    package_manifest_sha256: String,
    samples: Vec<Sample>,
    mean_seconds_per_measured_steps: f64,
    mean_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
struct DeviceSummary {
    runtime_index: u32,
    device_id: i32,
    backend: String,
    name: String,
    gcn_arch_name: String,
    compute_major: i32,
    compute_minor: i32,
    total_global_mem: u64,
}

#[derive(Debug, Serialize)]
struct Sample {
    repeat_index: usize,
    cache_len_start: usize,
    cache_len_end: usize,
    elapsed_seconds: f64,
    tokens_per_second: f64,
    generated_token_ids: Vec<usize>,
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    validate_options(&options)?;
    require_hip_kernel_guards()?;

    let artifact = read_sq8_canonical_artifact(&options.artifact)?;
    let norms = load_qwen3_14b_sq8_serving_norms(&options.package, UPLOAD_CHUNK_BYTES)
        .map_err(|error| error.to_string())?;
    let (runtime_index, device) = isolated_gfx1201_device()?;
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
        Sq8ServingPrefillMode::FixedM128Chunks,
    )
    .map_err(|error| error.to_string())?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let (artifact_content_sha256, package_manifest_sha256) = {
        let load_report = session.load_report();
        (
            load_report.artifact_content_sha256.clone(),
            load_report.package_manifest_sha256.clone(),
        )
    };

    let mut samples = Vec::with_capacity(options.repeats);
    let mut canonical_tokens = None;
    for repeat_index in 0..options.repeats {
        let sample = run_sample(&mut session, &mut context, &mut stream, &options, repeat_index)?;
        if let Some(expected) = &canonical_tokens {
            if expected != &sample.generated_token_ids {
                return Err(format!(
                    "repeat {repeat_index} changed deterministic generated tokens: expected={expected:?} actual={:?}",
                    sample.generated_token_ids
                ));
            }
        } else {
            canonical_tokens = Some(sample.generated_token_ids.clone());
        }
        samples.push(sample);
    }

    let mean_seconds_per_measured_steps = samples
        .iter()
        .map(|sample| sample.elapsed_seconds)
        .sum::<f64>()
        / samples.len() as f64;
    let mean_tokens_per_second = options.measured_steps as f64 / mean_seconds_per_measured_steps;
    let result = ResultFile {
        schema_version: "ullm.sq8_0.paged_decode_steady_bench.v1",
        scope: "unprofiled full-model M=1 decode only; load, M=128 seed prefill, and four warmups are excluded",
        input_token_pattern: "ascending_u32_1_through_prompt_tokens",
        test_only_ignore_eos: true,
        wave_scalar_softmax_enabled: env::var_os(WAVE_SCALAR_ENV).is_some(),
        runner_git_head: git_head(),
        device: DeviceSummary {
            runtime_index,
            device_id: device.device_id,
            backend: device.backend,
            name: device.name,
            gcn_arch_name: device.gcn_arch_name,
            compute_major: device.compute_major,
            compute_minor: device.compute_minor,
            total_global_mem: device.total_global_mem,
        },
        prefill_mode: "m128-chunk128",
        prompt_tokens: options.prompt_tokens,
        warmup_decode_steps: options.warmup_steps,
        measured_decode_steps: options.measured_steps,
        repeats: options.repeats,
        load_seconds,
        artifact_content_sha256,
        package_manifest_sha256,
        samples,
        mean_seconds_per_measured_steps,
        mean_tokens_per_second,
    };
    let serialized = serde_json::to_string_pretty(&result)
        .map_err(|error| format!("failed to serialize steady-decode result: {error}"))?;
    fs::write(&options.output, &serialized)
        .map_err(|error| format!("failed to write {}: {error}", options.output.display()))?;
    println!("{serialized}");
    Ok(())
}

fn run_sample(
    session: &mut Qwen3Sq8ServingSession,
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
    options: &Options,
    repeat_index: usize,
) -> Result<Sample, String> {
    let total_generated = 1usize
        .checked_add(options.warmup_steps)
        .and_then(|value| value.checked_add(options.measured_steps))
        .ok_or_else(|| "generated-token count overflows".to_string())?;
    let request = Sq8ServingRequest::greedy_ignore_eos_for_testing(
        format!("sq8_0-steady-r{repeat_index}"),
        (1..=options.prompt_tokens).collect(),
        total_generated,
    );
    session
        .start(context, request, Sq8CancellationToken::new(), stream)
        .map_err(|error| error.to_string())?;

    let mut generated_token_ids = Vec::with_capacity(total_generated);
    let initial = advance_expect_token(session, stream, 0, options.prompt_tokens, false)?;
    generated_token_ids.push(initial.0);
    for generated_index in 1..=options.warmup_steps {
        let (token_id, _) = advance_expect_token(
            session,
            stream,
            generated_index,
            options.prompt_tokens + generated_index,
            false,
        )?;
        generated_token_ids.push(token_id);
    }

    let cache_len_start = options.prompt_tokens + options.warmup_steps;
    let timer = Instant::now();
    for offset in 0..options.measured_steps {
        let generated_index = options.warmup_steps + 1 + offset;
        let final_step = offset + 1 == options.measured_steps;
        let (token_id, _) = advance_expect_token(
            session,
            stream,
            generated_index,
            options.prompt_tokens + generated_index,
            final_step,
        )?;
        generated_token_ids.push(token_id);
    }
    let elapsed_seconds = timer.elapsed().as_secs_f64();
    let cache_len_end = options.prompt_tokens + options.warmup_steps + options.measured_steps;
    let release = session
        .finish_and_reset_synchronized(stream)
        .map_err(|error| error.to_string())?;
    if !release.reset_complete || release.outcome != Sq8ReleaseOutcome::Length {
        return Err(format!(
            "steady-decode request did not reset after length: {release:?}"
        ));
    }
    Ok(Sample {
        repeat_index,
        cache_len_start,
        cache_len_end,
        elapsed_seconds,
        tokens_per_second: options.measured_steps as f64 / elapsed_seconds,
        generated_token_ids,
    })
}

fn advance_expect_token(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut RuntimeStream,
    expected_index: usize,
    expected_cache_len: usize,
    terminal: bool,
) -> Result<(usize, usize), String> {
    loop {
        match session
            .advance_synchronized(stream)
            .map_err(|error| error.to_string())?
        {
            Sq8ServingAdvance::PromptProgress { .. } => continue,
            Sq8ServingAdvance::Token {
                token_id,
                generated_index,
                cache_len,
                terminal_reason,
            } => {
                let expected_reason = terminal.then_some(Sq8FinishReason::Length);
                if generated_index != expected_index
                    || cache_len != expected_cache_len
                    || terminal_reason != expected_reason
                {
                    return Err(format!(
                        "unexpected token transition: index={generated_index} cache_len={cache_len} terminal={terminal_reason:?}; expected index={expected_index} cache_len={expected_cache_len} terminal={expected_reason:?}"
                    ));
                }
                return Ok((token_id, cache_len));
            }
            other => return Err(format!("unexpected serving advance: {other:?}")),
        }
    }
}

fn parse_options() -> Result<Options, String> {
    let mut artifact = None;
    let mut package = None;
    let mut output = None;
    let mut prompt_tokens = DEFAULT_PROMPT_TOKENS;
    let mut warmup_steps = DEFAULT_WARMUP_STEPS;
    let mut measured_steps = DEFAULT_MEASURED_STEPS;
    let mut repeats = DEFAULT_REPEATS;
    let mut args = env::args_os().skip(1);
    while let Some(argument) = args.next() {
        let argument = argument
            .to_str()
            .ok_or_else(|| "arguments must be UTF-8".to_string())?;
        match argument {
            "--artifact" => artifact = Some(PathBuf::from(option_value(&mut args, "--artifact")?)),
            "--package" => package = Some(PathBuf::from(option_value(&mut args, "--package")?)),
            "--output" => output = Some(PathBuf::from(option_value(&mut args, "--output")?)),
            "--prompt-tokens" => {
                prompt_tokens = parse_positive(
                    &option_value(&mut args, "--prompt-tokens")?,
                    "--prompt-tokens",
                )?
            }
            "--warmup-steps" => {
                warmup_steps = parse_nonnegative(
                    &option_value(&mut args, "--warmup-steps")?,
                    "--warmup-steps",
                )?
            }
            "--measured-steps" => {
                measured_steps = parse_positive(
                    &option_value(&mut args, "--measured-steps")?,
                    "--measured-steps",
                )?
            }
            "--repeats" => {
                repeats = parse_positive(&option_value(&mut args, "--repeats")?, "--repeats")?
            }
            "--help" => {
                return Err(
                    "usage: sq8_0_paged_decode_steady_bench --artifact DIR --package DIR --output JSON [--prompt-tokens N] [--warmup-steps N] [--measured-steps N] [--repeats N]"
                        .into(),
                );
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Options {
        artifact: artifact.ok_or_else(|| "--artifact is required".to_string())?,
        package: package.ok_or_else(|| "--package is required".to_string())?,
        output: output.ok_or_else(|| "--output is required".to_string())?,
        prompt_tokens,
        warmup_steps,
        measured_steps,
        repeats,
    })
}

fn option_value<I>(args: &mut I, name: &str) -> Result<String, String>
where
    I: Iterator<Item = OsString>,
{
    args.next()
        .ok_or_else(|| format!("{name} requires a value"))?
        .into_string()
        .map_err(|_| format!("{name} must be UTF-8"))
}

fn parse_positive(value: &str, name: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(parsed)
}

fn parse_nonnegative(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn validate_options(options: &Options) -> Result<(), String> {
    if !options.artifact.is_dir() || !options.package.is_dir() {
        return Err("--artifact and --package must be readable directories".into());
    }
    if options.output.exists() {
        return Err(format!(
            "refusing to overwrite existing result: {}",
            options.output.display()
        ));
    }
    if !options
        .output
        .parent()
        .is_some_and(|parent| parent.is_dir())
    {
        return Err(format!(
            "output parent must already exist: {}",
            options.output.display()
        ));
    }
    let total = options
        .prompt_tokens
        .checked_add(1)
        .and_then(|value| value.checked_add(options.warmup_steps))
        .and_then(|value| value.checked_add(options.measured_steps))
        .ok_or_else(|| "prompt/generation length overflows".to_string())?;
    if total > QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS {
        return Err(format!(
            "prompt plus generated tokens exceeds context: {total} > {QWEN3_14B_SQ8_SERVING_CONTEXT_TOKENS}"
        ));
    }
    Ok(())
}

fn isolated_gfx1201_device() -> Result<(u32, DeviceInfo), String> {
    let mut devices = Vec::new();
    for index in 1..device_count()? {
        let info = device_info(index)
            .map_err(|error| format!("failed to inspect runtime device {index}: {error}"))?;
        if info.backend == "hip" {
            devices.push((index, info));
        }
    }
    if devices.len() != 1 {
        return Err(format!(
            "steady-decode bench requires exactly one visible HIP device, found {}",
            devices.len()
        ));
    }
    let (runtime_index, device) = devices.pop().expect("exactly one device");
    validate_qwen3_14b_sq8_r9700_device_info(&device)?;
    if device.device_id != 0 {
        return Err(format!(
            "steady-decode bench requires isolated HIP device 0, got {}",
            device.device_id
        ));
    }
    Ok((runtime_index, device))
}

fn require_hip_kernel_guards() -> Result<(), String> {
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV)
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    let invalid = names
        .into_iter()
        .filter(|name| env::var(name).ok().as_deref() != Some("1"))
        .collect::<Vec<_>>();
    if invalid.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "steady-decode bench requires these HIP guards to equal 1: {}",
            invalid.join(",")
        ))
    }
}

fn git_head() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}
