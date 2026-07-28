// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Resident-BF16 Gemma4 E2B validation and throughput evidence driver.
//!
//! This is deliberately separate from serving and from the architecture trace
//! writer.  It loads the text-decoder weights once, refuses an occupied GPU
//! measurement window, and records the exact cache/no-cache and sliding-window
//! checks used for the resident execution path.

use serde_json::{json, Value};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::thread;
use std::time::{Duration, Instant};
use ullm_engine::gemma4_text_executor::{
    Gemma4ResidentHostProfile, Gemma4ResidentKvCacheSnapshot, Gemma4ResidentLogicalBytes,
    Gemma4ResidentPrimitiveHostProfile, Gemma4TextExecutor,
};

const R9700_AMD_SMI_INDEX: &str = "2";
const AMD_SMI: &str = "/opt/rocm/bin/amd-smi";
const DEFAULT_COOLDOWN_HOTSPOT_C: f64 = 55.0;
const DEFAULT_COOLDOWN_TIMEOUT_SECONDS: u64 = 900;
const DEFAULT_BENCHMARK_REPEATS: usize = 3;
const DEFAULT_BENCHMARK_PROMPT_TOKENS: usize = 6;
const DEFAULT_BENCHMARK_DECODE_TOKENS: usize = 4;
const DEFAULT_BENCHMARK_PROMPT_TOKEN_ID: u32 = 2;
const DEFAULT_CONTINUATION_TOKENS: usize = 128;
const GEMMA4_SLIDING_ATTENTION_LAYERS: usize = 28;
const GEMMA4_FULL_ATTENTION_LAYERS: usize = 7;
const GEMMA4_PREFILL_QUERY_TILE_TOKENS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Validation,
    SlidingBoundary,
    Benchmark,
    AttentionProfile,
    Continuation,
}

impl Mode {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "validation" => Ok(Self::Validation),
            "sliding-boundary" => Ok(Self::SlidingBoundary),
            "benchmark" => Ok(Self::Benchmark),
            "attention-profile" => Ok(Self::AttentionProfile),
            "continuation" => Ok(Self::Continuation),
            _ => Err(format!(
                "--mode must be validation, sliding-boundary, benchmark, attention-profile, or continuation, got {raw:?}"
            )),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::SlidingBoundary => "sliding-boundary",
            Self::Benchmark => "benchmark",
            Self::AttentionProfile => "attention-profile",
            Self::Continuation => "continuation",
        }
    }
}

#[derive(Debug)]
struct Options {
    model_dir: PathBuf,
    output: PathBuf,
    mode: Mode,
    benchmark_repeats: usize,
    benchmark_prompt_tokens: usize,
    benchmark_decode_tokens: usize,
    benchmark_prompt_token_id: u32,
    continuation_tokens: usize,
    cooldown_hotspot_c: f64,
    cooldown_timeout_seconds: u64,
}

#[derive(Debug, Clone, Copy)]
struct TraceCase {
    name: &'static str,
    initial_token_ids: &'static [u32],
    expected_generated_token_ids: &'static [u32],
    expected_text: &'static str,
}

const CAPITAL_FRANCE: TraceCase = TraceCase {
    name: "capital-france",
    initial_token_ids: &[2, 818, 5279, 529, 7001, 563],
    expected_generated_token_ids: &[9079, 236761, 108, 818],
    expected_text: "The capital of France is Paris.\\n\\nThe",
};

const ONCE_UPON_A_TIME: TraceCase = TraceCase {
    name: "once-upon-a-time",
    initial_token_ids: &[2, 14946, 3324, 496, 990, 236764],
    expected_generated_token_ids: &[528, 496, 1902, 1298],
    expected_text: "Once upon a time, in a world where",
};

fn usage() -> &'static str {
    "usage: ullm-gemma4-resident --model-dir PATH --output PATH --mode validation|sliding-boundary|benchmark|attention-profile|continuation [--benchmark-repeats N] [--benchmark-prompt-tokens N] [--benchmark-decode-tokens N] [--benchmark-prompt-token-id ID] [--continuation-tokens N] [--cooldown-hotspot-c C] [--cooldown-timeout-seconds N]"
}

fn main() -> ExitCode {
    match parse_options().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-gemma4-resident: {error}");
            ExitCode::from(2)
        }
    }
}

fn parse_options() -> Result<Options, String> {
    let mut model_dir = None;
    let mut output = None;
    let mut mode = None;
    let mut benchmark_repeats = DEFAULT_BENCHMARK_REPEATS;
    let mut benchmark_prompt_tokens = DEFAULT_BENCHMARK_PROMPT_TOKENS;
    let mut benchmark_decode_tokens = DEFAULT_BENCHMARK_DECODE_TOKENS;
    let mut benchmark_prompt_token_id = DEFAULT_BENCHMARK_PROMPT_TOKEN_ID;
    let mut continuation_tokens = DEFAULT_CONTINUATION_TOKENS;
    let mut cooldown_hotspot_c = DEFAULT_COOLDOWN_HOTSPOT_C;
    let mut cooldown_timeout_seconds = DEFAULT_COOLDOWN_TIMEOUT_SECONDS;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model-dir" => {
                if model_dir
                    .replace(PathBuf::from(next_argument("--model-dir", &mut arguments)?))
                    .is_some()
                {
                    return Err(format!(
                        "--model-dir was supplied more than once; {}",
                        usage()
                    ));
                }
            }
            "--output" => {
                if output
                    .replace(PathBuf::from(next_argument("--output", &mut arguments)?))
                    .is_some()
                {
                    return Err(format!("--output was supplied more than once; {}", usage()));
                }
            }
            "--mode" => {
                let value = Mode::parse(&next_argument("--mode", &mut arguments)?)?;
                if mode.replace(value).is_some() {
                    return Err(format!("--mode was supplied more than once; {}", usage()));
                }
            }
            "--benchmark-repeats" => {
                benchmark_repeats = parse_positive_usize(
                    "--benchmark-repeats",
                    &next_argument("--benchmark-repeats", &mut arguments)?,
                )?;
            }
            "--benchmark-prompt-tokens" => {
                benchmark_prompt_tokens = parse_positive_usize(
                    "--benchmark-prompt-tokens",
                    &next_argument("--benchmark-prompt-tokens", &mut arguments)?,
                )?;
            }
            "--benchmark-decode-tokens" => {
                benchmark_decode_tokens = parse_positive_usize(
                    "--benchmark-decode-tokens",
                    &next_argument("--benchmark-decode-tokens", &mut arguments)?,
                )?;
            }
            "--benchmark-prompt-token-id" => {
                let raw = next_argument("--benchmark-prompt-token-id", &mut arguments)?;
                benchmark_prompt_token_id = raw.parse::<u32>().map_err(|_| {
                    format!("--benchmark-prompt-token-id must be a u32, got {raw:?}")
                })?;
            }
            "--continuation-tokens" => {
                continuation_tokens = parse_positive_usize(
                    "--continuation-tokens",
                    &next_argument("--continuation-tokens", &mut arguments)?,
                )?;
            }
            "--cooldown-hotspot-c" => {
                let raw = next_argument("--cooldown-hotspot-c", &mut arguments)?;
                cooldown_hotspot_c = raw.parse::<f64>().map_err(|_| {
                    format!("--cooldown-hotspot-c must be a finite positive number, got {raw:?}")
                })?;
                if !cooldown_hotspot_c.is_finite() || cooldown_hotspot_c <= 0.0 {
                    return Err(format!(
                        "--cooldown-hotspot-c must be a finite positive number, got {cooldown_hotspot_c}"
                    ));
                }
            }
            "--cooldown-timeout-seconds" => {
                cooldown_timeout_seconds = parse_positive_u64(
                    "--cooldown-timeout-seconds",
                    &next_argument("--cooldown-timeout-seconds", &mut arguments)?,
                )?;
            }
            "--help" | "-h" => return Err(usage().to_string()),
            _ => return Err(format!("unknown argument {argument:?}; {}", usage())),
        }
    }
    Ok(Options {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required; {}", usage()))?,
        output: output.ok_or_else(|| format!("--output is required; {}", usage()))?,
        mode: mode.ok_or_else(|| format!("--mode is required; {}", usage()))?,
        benchmark_repeats,
        benchmark_prompt_tokens,
        benchmark_decode_tokens,
        benchmark_prompt_token_id,
        continuation_tokens,
        cooldown_hotspot_c,
        cooldown_timeout_seconds,
    })
}

fn next_argument(
    name: &str,
    arguments: &mut impl Iterator<Item = String>,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{name} requires a value; {}", usage()))
}

fn parse_positive_usize(name: &str, raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer, got {raw:?}"))?;
    if value == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(value)
}

fn parse_positive_u64(name: &str, raw: &str) -> Result<u64, String> {
    let value = raw
        .parse::<u64>()
        .map_err(|_| format!("{name} must be a positive integer, got {raw:?}"))?;
    if value == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(value)
}

fn run(options: Options) -> Result<(), String> {
    if options.output.exists() {
        return Err(format!(
            "output already exists; refusing to overwrite {}",
            options.output.display()
        ));
    }
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create output directory {}: {error}",
                parent.display()
            )
        })?;
    }

    let preflight = preflight()?;
    let launch_started = Instant::now();
    let load_started = Instant::now();
    let mut executor = Gemma4TextExecutor::load_resident(&options.model_dir)?;
    let load_elapsed_seconds = load_started.elapsed().as_secs_f64();
    let plan = executor
        .resident_memory_plan()
        .cloned()
        .ok_or_else(|| "resident Gemma4 executor did not return a memory plan".to_string())?;
    let after_load_telemetry = capture_telemetry();
    let cooldown = wait_for_cooldown(options.cooldown_hotspot_c, options.cooldown_timeout_seconds)?;
    let before_workload_telemetry = capture_telemetry();
    // Validation probes temporarily enable and then restore individual
    // dispatch controls.  Preserve the caller's exact launch configuration
    // before they run so the evidence cannot misleadingly report a rollback.
    let environment = driver_environment_json();

    let workload = match options.mode {
        Mode::Validation => validation_workload(&mut executor)?,
        Mode::SlidingBoundary => sliding_boundary_workload(&mut executor)?,
        Mode::Benchmark => benchmark_workload(
            &mut executor,
            options.benchmark_repeats,
            options.benchmark_prompt_tokens,
            options.benchmark_decode_tokens,
            options.benchmark_prompt_token_id,
        )?,
        Mode::AttentionProfile => attention_profile_workload(
            &mut executor,
            options.benchmark_prompt_tokens,
            options.benchmark_prompt_token_id,
        )?,
        Mode::Continuation => continuation_workload(&mut executor, options.continuation_tokens)?,
    };
    let after_workload_telemetry = capture_telemetry();
    let snapshot = executor.resident_kv_cache_snapshot()?;
    let actual_weight_bytes = executor
        .resident_weight_bytes()
        .ok_or_else(|| "resident Gemma4 executor did not retain weights".to_string())?;
    let actual_kv_bytes = executor
        .device_kv_bytes()?
        .ok_or_else(|| "resident Gemma4 executor did not retain device K/V".to_string())?;
    let actual_transient_bytes = executor
        .device_transient_bytes()?
        .ok_or_else(|| "resident Gemma4 executor did not expose transient buffers".to_string())?;
    let actual_total_bytes = executor
        .resident_device_allocation_bytes()?
        .ok_or_else(|| {
            "resident Gemma4 executor did not expose allocation accounting".to_string()
        })?;
    let mlp_validation = executor.resident_mlp_validation();
    let rope_validation = executor.resident_rope_validation();
    let device = executor.device();
    let report = json!({
        "schema_version": "ullm.gemma4_e2b_resident.v0.1",
        "producer": "ullm-gemma4-resident",
        "mode": options.mode.as_str(),
        "model": {
            "model_dir": executor.source_model_dir(),
            "config_sha256": executor.config_sha256(),
            "weight_format": "source BF16 safetensors, resident text decoder",
            "activation_dtype": "F32",
        },
        "device": {
            "runtime_index": device.runtime_index,
            "device_id": device.device_id,
            "backend": device.backend,
            "name": device.name,
            "gcn_arch_name": device.gcn_arch_name,
            "compute": [device.compute_major, device.compute_minor],
            "total_global_mem_bytes": device.total_global_mem,
        },
        "environment": environment,
        "preflight": preflight,
        "cooldown": cooldown,
        "telemetry": {
            "after_resident_load": after_load_telemetry,
            "before_workload": before_workload_telemetry,
            "after_workload": after_workload_telemetry,
        },
        "resident_memory": memory_json(&plan, device.total_global_mem, actual_weight_bytes, actual_kv_bytes, actual_transient_bytes, actual_total_bytes)?,
        "final_kv_snapshot": snapshot_json(snapshot),
        "workload": workload,
        "mlp_region_validation": {
            "enabled": env::var("ULLM_GEMMA4_VALIDATE_DEVICE_MLP").ok().as_deref() == Some("1"),
            "calls": mlp_validation.calls,
            "elements": mlp_validation.elements,
            "max_abs": mlp_validation.max_abs,
            "max_rel": mlp_validation.max_rel,
            "reference": "unchanged host pre-FF RMSNorm -> MLP -> post-FF RMSNorm -> residual-add sequence on the same captured attention residuals",
        },
        "gemma_proportional_rope_validation": {
            "enabled": env::var("ULLM_GEMMA4_VALIDATE_PROPORTIONAL_ROPE").ok().as_deref() == Some("1"),
            "calls": rope_validation.calls,
            "elements": rope_validation.elements,
            "max_abs": rope_validation.max_abs,
            "max_rel": rope_validation.max_rel,
            "rotated_channels": {
                "max_abs": rope_validation.rotated_max_abs,
                "max_rel": rope_validation.rotated_max_rel,
            },
            "unrotated_channels": {
                "max_abs": rope_validation.unrotated_max_abs,
                "max_rel": rope_validation.unrotated_max_rel,
                "contract": "exact pass-through, including channels on both partial-pair boundaries",
            },
            "reference": "unchanged host apply_gemma4_rope_in_place on the same real normalized Q/K activations; exponent denominator is head_dim and unrotated tail is copied unchanged",
        },
        "timing": {
            "resident_load_seconds": load_elapsed_seconds,
            "wall_seconds_including_load_and_workload": launch_started.elapsed().as_secs_f64(),
            "clock": "std::time::Instant monotonic wall clock; no profiler range is used as throughput",
        },
    });
    write_json_new(&options.output, &report)?;
    println!(
        "Gemma4 resident {} evidence written to {} (load {:.1}s)",
        options.mode.as_str(),
        options.output.display(),
        load_elapsed_seconds,
    );
    Ok(())
}

fn continuation_workload(
    executor: &mut Gemma4TextExecutor,
    continuation_tokens: usize,
) -> Result<Value, String> {
    if continuation_tokens < CAPITAL_FRANCE.expected_generated_token_ids.len() {
        return Err(format!(
            "--continuation-tokens must be at least {} to verify the known-good trace",
            CAPITAL_FRANCE.expected_generated_token_ids.len()
        ));
    }
    let generation = cached_generation(
        executor,
        CAPITAL_FRANCE.initial_token_ids,
        continuation_tokens,
    )?;
    let generated = generation.generated_token_ids;
    if generated[..CAPITAL_FRANCE.expected_generated_token_ids.len()]
        != CAPITAL_FRANCE.expected_generated_token_ids[..]
    {
        return Err(format!(
            "continuation first four greedy IDs differ from BL trace: expected {:?}, got {:?}",
            CAPITAL_FRANCE.expected_generated_token_ids,
            &generated[..CAPITAL_FRANCE.expected_generated_token_ids.len()]
        ));
    }
    let final_position = executor.position();
    let ring_rollover = if continuation_tokens >= executor.config().sliding_window {
        let snapshot = executor
            .resident_kv_cache_snapshot()?
            .ok_or_else(|| "continuation has no device K/V snapshot".to_string())?;
        assert_sliding_boundary_state(&snapshot, executor.config().sliding_window, final_position)?;
        json!({
            "result": "passed",
            "sliding_window": executor.config().sliding_window,
            "final_position": final_position,
            "method": "cached M=1 decode continued beyond the sliding-ring capacity and every sliding source retained capacity/window cache_len with the exact absolute position",
        })
    } else {
        json!({
            "result": "not-requested",
            "minimum_continuation_tokens": executor.config().sliding_window,
        })
    };
    Ok(json!({
        "kind": "known-good-cached-continuation",
        "initial_token_ids": CAPITAL_FRANCE.initial_token_ids,
        "expected_first_four_generated_token_ids_from_bl_trace": CAPITAL_FRANCE.expected_generated_token_ids,
        "generated_token_ids": generated,
        "first_four_match_bl_trace": true,
        "ring_rollover": ring_rollover,
        "cached_decode": generation.json,
    }))
}

fn driver_environment_json() -> Value {
    json!({
        "HIP_VISIBLE_DEVICES": env::var("HIP_VISIBLE_DEVICES").ok(),
        "ULLM_HIP_VISIBLE_DEVICES": env::var("ULLM_HIP_VISIBLE_DEVICES").ok(),
        "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL": env::var("ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL").ok(),
        "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL": env::var("ULLM_REQUIRE_HIP_BF16_ROW_KERNEL").ok(),
        "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL": env::var("ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL").ok(),
        "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL": env::var("ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL").ok(),
        "ULLM_REQUIRE_HIP_RMSNORM_KERNEL": env::var("ULLM_REQUIRE_HIP_RMSNORM_KERNEL").ok(),
        "ULLM_REQUIRE_HIP_ADD_KERNEL": env::var("ULLM_REQUIRE_HIP_ADD_KERNEL").ok(),
        "ULLM_REQUIRE_HIP_ROPE_KERNEL": env::var("ULLM_REQUIRE_HIP_ROPE_KERNEL").ok(),
        "ULLM_REQUIRE_HIP_GEMMA_PROPORTIONAL_ROPE_KERNEL": env::var("ULLM_REQUIRE_HIP_GEMMA_PROPORTIONAL_ROPE_KERNEL").ok(),
        "ULLM_GEMMA4_PREFILL_LAYER_MAJOR": env::var("ULLM_GEMMA4_PREFILL_LAYER_MAJOR").ok(),
        "ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED": env::var("ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED").ok(),
        "ULLM_GEMMA4_FULL_ATTN_SPLIT_KV": env::var("ULLM_GEMMA4_FULL_ATTN_SPLIT_KV").ok(),
        "ULLM_GEMMA4_SLIDING_ATTN_SPLIT_KV": env::var("ULLM_GEMMA4_SLIDING_ATTN_SPLIT_KV").ok(),
    })
}

fn preflight() -> Result<Value, String> {
    let pgrep = capture_command(
        "pgrep",
        [
            "-af",
            "ullm-sq8-r9700|run_measurements.py|llama-bench|llama-server|promote-served-model|ullm-aq4-worker",
        ],
    );
    let service = capture_command("systemctl", ["is-active", "ullm-openai.service"]);
    let service_state = service
        .get("stdout")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if !matches!(service_state, "inactive" | "failed") {
        return Err(format!(
            "refusing GPU work while ullm-openai.service is {service_state:?}; wait for inactive or failed"
        ));
    }
    let process = capture_command(
        AMD_SMI,
        [
            "process",
            "--gpu",
            R9700_AMD_SMI_INDEX,
            "--general",
            "--json",
        ],
    );
    let process_text = process
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !process_text.contains("No running processes detected") {
        return Err(
            "refusing GPU work because amd-smi reports one or more R9700 processes".to_string(),
        );
    }
    Ok(json!({
        "required_pgrep": pgrep_summary(&pgrep),
        "ullm_openai_service": service,
        "r9700_process": process,
        "decision": "accepted: service is not running (inactive or failed) and amd-smi reported no R9700 process",
        "note": "The required pgrep command was executed. Its raw output is deliberately summarized because its broad pattern can match an orchestration prompt rather than an executable workload; amd-smi process state is the authoritative R9700 occupancy check.",
    }))
}

fn pgrep_summary(capture: &Value) -> Value {
    let stdout = capture
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let executable_like_matches = stdout
        .lines()
        .filter(|line| !line.contains("codex exec") && !line.contains("/bin/bash -c"))
        .map(str::to_string)
        .collect::<Vec<_>>();
    json!({
        "program": capture.get("program"),
        "args": capture.get("args"),
        "exit_code": capture.get("exit_code"),
        "raw_match_line_count": stdout.lines().count(),
        "non_orchestrator_match_lines": executable_like_matches,
    })
}

fn capture_telemetry() -> Value {
    capture_command(
        AMD_SMI,
        [
            "metric",
            "--gpu",
            R9700_AMD_SMI_INDEX,
            "--temperature",
            "--clock",
            "--power",
            "--violation",
            "--mem-usage",
            "--json",
        ],
    )
}

fn wait_for_cooldown(threshold_c: f64, timeout_seconds: u64) -> Result<Value, String> {
    let started = Instant::now();
    let mut samples = Vec::new();
    loop {
        let sample = capture_telemetry();
        let hotspot_c = hotspot_celsius(&sample);
        samples.push(sample);
        if let Some(hotspot_c) = hotspot_c {
            if hotspot_c <= threshold_c {
                return Ok(json!({
                    "threshold_hotspot_c": threshold_c,
                    "samples": samples,
                    "elapsed_seconds": started.elapsed().as_secs_f64(),
                    "decision": "temperature at or below threshold before measured work",
                }));
            }
        }
        if started.elapsed() >= Duration::from_secs(timeout_seconds) {
            return Err(format!(
                "R9700 hotspot did not cool to {threshold_c:.1} C within {timeout_seconds}s"
            ));
        }
        thread::sleep(Duration::from_secs(5));
    }
}

fn hotspot_celsius(capture: &Value) -> Option<f64> {
    capture
        .get("parsed_stdout")?
        .get("gpu_data")?
        .as_array()?
        .iter()
        .find_map(|gpu| {
            gpu.get("temperature")?
                .get("hotspot")?
                .get("value")?
                .as_f64()
        })
}

fn capture_command<const N: usize>(program: &str, args: [&str; N]) -> Value {
    let json_args = args.to_vec();
    match Command::new(program).args(args).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            let parsed_stdout = serde_json::from_str::<Value>(&stdout).ok();
            json!({
                "program": program,
                "args": json_args,
                "exit_code": output.status.code(),
                "stdout": stdout,
                "stderr": stderr,
                "parsed_stdout": parsed_stdout,
            })
        }
        Err(error) => json!({
            "program": program,
            "args": json_args,
            "spawn_error": error.to_string(),
        }),
    }
}

fn validation_workload(executor: &mut Gemma4TextExecutor) -> Result<Value, String> {
    let attention_region_differential = attention_region_differential(executor)?;
    let causal_mask_future_token_probe = causal_mask_future_token_probe(executor)?;
    let causal_mask_sliding_window_probe = causal_mask_sliding_window_probe(executor)?;
    let mut cases = Vec::new();
    let mut capital_cached_ids = None;
    for case in [CAPITAL_FRANCE, ONCE_UPON_A_TIME] {
        let cached = cached_generation(
            executor,
            case.initial_token_ids,
            case.expected_generated_token_ids.len(),
        )?;
        if cached.generated_token_ids != case.expected_generated_token_ids {
            return Err(format!(
                "resident cached {} greedy IDs differ from BL trace: expected {:?}, got {:?}",
                case.name, case.expected_generated_token_ids, cached.generated_token_ids
            ));
        }
        let reprefill = full_reprefill_generation(
            executor,
            case.initial_token_ids,
            case.expected_generated_token_ids.len(),
        )?;
        if reprefill.generated_token_ids != cached.generated_token_ids {
            return Err(format!(
                "resident cache/no-cache greedy IDs differ for {}: cached {:?}, full-reprefill {:?}",
                case.name, cached.generated_token_ids, reprefill.generated_token_ids
            ));
        }
        if case.name == CAPITAL_FRANCE.name {
            capital_cached_ids = Some(cached.generated_token_ids.clone());
        }
        cases.push(json!({
            "case": case.name,
            "initial_token_ids": case.initial_token_ids,
            "expected_generated_token_ids_from_bl_trace": case.expected_generated_token_ids,
            "expected_decoded_text_from_bl_trace": case.expected_text,
            "cached_decode": cached.json,
            "full_reprefill_without_cross_step_cache": reprefill.json,
            "greedy_ids_match_bl_trace": true,
            "cache_and_full_reprefill_match": true,
        }));
    }
    let snapshot = executor
        .resident_kv_cache_snapshot()?
        .ok_or_else(|| "validation requires a resident K/V snapshot".to_string())?;
    assert_e2b_shared_sources(&snapshot)?;
    let capital_cached_ids = capital_cached_ids
        .ok_or_else(|| "validation did not retain the capital-France cached result".to_string())?;
    let unshared = executor.unshared_kv_reference_generation(
        CAPITAL_FRANCE.initial_token_ids,
        CAPITAL_FRANCE.expected_generated_token_ids.len(),
    )?;
    let unshared_ids_equal_shared = unshared.generated_token_ids == capital_cached_ids;
    Ok(json!({
        "kind": "resident-greedy-and-cache-equivalence",
        "attention_region_differential": attention_region_differential,
        "causal_mask_future_token_probe": causal_mask_future_token_probe,
        "causal_mask_sliding_window_probe": causal_mask_sliding_window_probe,
        "cases": cases,
        "shared_kv_source_check": {
            "result": "passed",
            "method": "device-cache snapshot contains only non-sharing source layers; every layer 15..34 maps by attention kind to layer 13 (local) or layer 14 (full).",
            "snapshot": snapshot_json(Some(snapshot)),
        },
        "sharing_disabled_physical_kv_reference": {
            "method": "The diagnostic temporarily replaces the resident source-cache topology with independent host K/V caches for all layers and evaluates physical K/V projections for shared layers. HF does not use this topology; it is included only to make the source selection contrast explicit.",
            "normal_shared_generated_token_ids": capital_cached_ids,
            "physical_kv_reprojected_generated_token_ids": unshared.generated_token_ids,
            "physical_kv_reprojected_top1_logits": unshared.top1_logits,
            "generated_ids_equal_to_normal_shared_path": unshared_ids_equal_shared,
        },
    }))
}

/// Changes key 0 and observes layer 0's query 512 result.  At that point the
/// local reader's 512-token window starts at key 1, so a reader that leaks the
/// excluded j-512 key changes the output.  Query 512 deliberately starts a
/// full M=128 prefill chunk and therefore exercises the split-reader dispatch,
/// rather than a final M=1 tail.
fn causal_mask_sliding_window_probe(executor: &mut Gemma4TextExecutor) -> Result<Value, String> {
    const QUERY_INDEX: usize = 512;
    const PROMPT_TOKENS: usize = 640;
    const FIRST_KEY: u32 = 818;
    const SECOND_KEY: u32 = 5279;
    let prior_layer_major = env::var_os("ULLM_GEMMA4_PREFILL_LAYER_MAJOR");
    let prior_ring_batched = env::var_os("ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED");
    unsafe { env::set_var("ULLM_GEMMA4_PREFILL_LAYER_MAJOR", "1") };
    unsafe { env::set_var("ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED", "1") };
    let mut first_tokens = vec![2_u32; PROMPT_TOKENS];
    first_tokens[0] = FIRST_KEY;
    executor.reset();
    let first = executor.prefill(&first_tokens)?;
    let mut second_tokens = first_tokens;
    second_tokens[0] = SECOND_KEY;
    executor.reset();
    let second = executor.prefill(&second_tokens)?;
    restore_env_var(
        "ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED",
        prior_ring_batched,
    );
    restore_env_var("ULLM_GEMMA4_PREFILL_LAYER_MAJOR", prior_layer_major);

    let hidden = executor.config().decoder.hidden_size;
    let range = QUERY_INDEX
        .checked_mul(hidden)
        .and_then(|start| start.checked_add(hidden).map(|end| start..end))
        .ok_or_else(|| "sliding-window causal probe row range overflows".to_string())?;
    let (max_abs, max_rel) = max_error(
        first.layer_outputs[0][range.clone()].iter().copied(),
        second.layer_outputs[0][range].iter().copied(),
    )?;
    if max_abs != 0.0 {
        return Err(format!(
            "sliding-window causal probe failed: query {QUERY_INDEX} observed excluded key 0 (max_abs={max_abs}, max_rel={max_rel})"
        ));
    }
    Ok(json!({
        "result": "passed",
        "method": "Two 640-token M=128-chunk prefills differ only at key 0; layer 0 query 512 must be bit-identical because the 512-token sliding window excludes key j-512.",
        "query_index": QUERY_INDEX,
        "excluded_key_index": 0,
        "prefill_chunk_rows": 128,
        "layer_checked": 0,
        "layer_output_max_abs": max_abs,
        "layer_output_max_rel": max_rel,
    }))
}

/// Changes only key j+1 in a complete query tile.  Query j is identical in
/// both inputs, so a batched reader that exposes a future K/V row fails before
/// any plausible-looking continuation can hide the leak.
fn causal_mask_future_token_probe(executor: &mut Gemma4TextExecutor) -> Result<Value, String> {
    // M=3 used to exercise the M=1 fallback because split-KV intentionally
    // starts at M=4.  A complete tile instead reaches the ring split reader.
    const PROMPT_TOKENS: usize = GEMMA4_PREFILL_QUERY_TILE_TOKENS;
    const QUERY_INDEX: usize = 64;
    const FUTURE_INDEX: usize = QUERY_INDEX + 1;
    const FIRST_FUTURE: u32 = 5279;
    const SECOND_FUTURE: u32 = 7001;
    let prior_layer_major = env::var_os("ULLM_GEMMA4_PREFILL_LAYER_MAJOR");
    let prior_ring_batched = env::var_os("ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED");
    unsafe { env::set_var("ULLM_GEMMA4_PREFILL_LAYER_MAJOR", "1") };
    unsafe { env::set_var("ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED", "1") };
    let mut tokens = vec![2_u32; PROMPT_TOKENS];
    tokens[FUTURE_INDEX] = FIRST_FUTURE;
    executor.reset();
    let first = executor.prefill(&tokens)?;
    tokens[FUTURE_INDEX] = SECOND_FUTURE;
    executor.reset();
    let second = executor.prefill(&tokens)?;
    let hidden = executor.config().decoder.hidden_size;
    let query_range = QUERY_INDEX
        .checked_mul(hidden)
        .and_then(|start| start.checked_add(hidden).map(|end| start..end))
        .ok_or_else(|| "causal-mask query width overflows".to_string())?;
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (layer, (first_layer, second_layer)) in first
        .layer_outputs
        .iter()
        .zip(second.layer_outputs.iter())
        .enumerate()
    {
        let (abs, rel) = max_error(
            first_layer[query_range.clone()].iter().copied(),
            second_layer[query_range.clone()].iter().copied(),
        )?;
        if abs != 0.0 {
            return Err(format!(
                "causal-mask future-token probe failed at layer {layer}: changing key {FUTURE_INDEX} changed query {QUERY_INDEX} (max_abs={abs}, max_rel={rel})"
            ));
        }
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    let output = json!({
        "result": "passed",
        "method": "Two 128-token M=128 prefills differ only at key j+1. Query j must be bit-identical in every layer; this reaches the ring-batched split reader rather than the M=1 fallback.",
        "query_index": QUERY_INDEX,
        "excluded_future_key_index": FUTURE_INDEX,
        "first_future_token_id": FIRST_FUTURE,
        "second_future_token_id": SECOND_FUTURE,
        "prefill_chunk_rows": PROMPT_TOKENS,
        "layer_output_query_max_abs": max_abs,
        "layer_output_query_max_rel": max_rel,
    });
    restore_env_var(
        "ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED",
        prior_ring_batched,
    );
    restore_env_var("ULLM_GEMMA4_PREFILL_LAYER_MAJOR", prior_layer_major);
    Ok(output)
}

fn restore_env_var(name: &str, value: Option<std::ffi::OsString>) {
    unsafe {
        match value {
            Some(value) => env::set_var(name, value),
            None => env::remove_var(name),
        }
    }
}

fn attention_region_differential(executor: &mut Gemma4TextExecutor) -> Result<Value, String> {
    executor.reset();
    unsafe { env::remove_var("ULLM_GEMMA4_DISABLE_ATTENTION_REGION") };
    unsafe { env::remove_var("ULLM_GEMMA4_DISABLE_PLE_REGION") };
    unsafe { env::set_var("ULLM_GEMMA4_PREFILL_LAYER_MAJOR", "1") };
    let resident = executor.prefill(CAPITAL_FRANCE.initial_token_ids)?;
    executor.reset();
    unsafe { env::set_var("ULLM_GEMMA4_DISABLE_ATTENTION_REGION", "1") };
    unsafe { env::set_var("ULLM_GEMMA4_DISABLE_PLE_REGION", "1") };
    let host = executor.prefill(CAPITAL_FRANCE.initial_token_ids)?;
    unsafe { env::remove_var("ULLM_GEMMA4_DISABLE_ATTENTION_REGION") };
    unsafe { env::remove_var("ULLM_GEMMA4_DISABLE_PLE_REGION") };
    unsafe { env::remove_var("ULLM_GEMMA4_PREFILL_LAYER_MAJOR") };
    let (hidden_abs, hidden_rel) = max_error(
        resident.layer_outputs.iter().flatten().copied(),
        host.layer_outputs.iter().flatten().copied(),
    )?;
    let (final_abs, final_rel) = max_error(
        resident.final_norm.iter().copied(),
        host.final_norm.iter().copied(),
    )?;
    let (logit_abs, logit_rel) = max_error(
        resident.logits_last.iter().copied(),
        host.logits_last.iter().copied(),
    )?;
    Ok(json!({
        "reference": "unchanged host attention/PLE path on the same six real captured prompt activations; multi-layer hidden/final/logit comparison",
        "input_token_ids": CAPITAL_FRANCE.initial_token_ids,
        "layer_output_max_abs": hidden_abs,
        "layer_output_max_rel": hidden_rel,
        "final_norm_max_abs": final_abs,
        "final_norm_max_rel": final_rel,
        "logits_max_abs": logit_abs,
        "logits_max_rel": logit_rel,
        "resident_top1": {"token_id": resident.top1.token_id, "logit": resident.top1.logit},
        "host_top1": {"token_id": host.top1.token_id, "logit": host.top1.logit},
    }))
}

fn max_error(
    actual: impl Iterator<Item = f32>,
    expected: impl Iterator<Item = f32>,
) -> Result<(f32, f32), String> {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut count = 0_usize;
    for (actual, expected) in actual.zip(expected) {
        let abs = (actual - expected).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / expected.abs().max(f32::MIN_POSITIVE));
        count = count.saturating_add(1);
    }
    if count == 0 {
        return Err("attention region differential compared zero values".into());
    }
    Ok((max_abs, max_rel))
}

struct GenerationResult {
    generated_token_ids: Vec<u32>,
    json: Value,
}

fn cached_generation(
    executor: &mut Gemma4TextExecutor,
    initial: &[u32],
    new_tokens: usize,
) -> Result<GenerationResult, String> {
    executor.reset();
    executor.reset_resident_logical_bytes();
    let mut generated_token_ids = Vec::with_capacity(new_tokens);
    let mut steps = Vec::with_capacity(new_tokens);
    let mut started = Instant::now();
    let trace = executor.prefill(initial)?;
    steps.push(json!({
        "operation": "prefill",
        "input_token_count": initial.len(),
        "top1_token_id": trace.top1.token_id,
        "top1_logit": trace.top1.logit,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    }));
    generated_token_ids.push(trace.top1.token_id);
    for _ in 1..new_tokens {
        let input_token_id = *generated_token_ids
            .last()
            .ok_or_else(|| "cached generation lost its previous token".to_string())?;
        started = Instant::now();
        let trace = executor.decode(input_token_id)?;
        steps.push(json!({
            "operation": "decode",
            "input_token_id": input_token_id,
            "top1_token_id": trace.top1.token_id,
            "top1_logit": trace.top1.logit,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        }));
        generated_token_ids.push(trace.top1.token_id);
    }
    let logical = executor.resident_logical_bytes().ok_or_else(|| {
        "cached generation did not expose resident logical accounting".to_string()
    })?;
    Ok(GenerationResult {
        generated_token_ids: generated_token_ids.clone(),
        json: json!({
            "generated_token_ids": generated_token_ids,
            "steps": steps,
            "logical_lower_bound": logical_bytes_json(logical)?,
            "final_position": executor.position(),
        }),
    })
}

fn full_reprefill_generation(
    executor: &mut Gemma4TextExecutor,
    initial: &[u32],
    new_tokens: usize,
) -> Result<GenerationResult, String> {
    let mut prefix = initial.to_vec();
    let mut generated_token_ids = Vec::with_capacity(new_tokens);
    let mut steps = Vec::with_capacity(new_tokens);
    executor.reset_resident_logical_bytes();
    for _ in 0..new_tokens {
        executor.reset();
        let started = Instant::now();
        let trace = executor.prefill(&prefix)?;
        steps.push(json!({
            "operation": "prefill-from-token-zero",
            "input_token_count": prefix.len(),
            "top1_token_id": trace.top1.token_id,
            "top1_logit": trace.top1.logit,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        }));
        generated_token_ids.push(trace.top1.token_id);
        prefix.push(trace.top1.token_id);
    }
    let logical = executor
        .resident_logical_bytes()
        .ok_or_else(|| "full re-prefill did not expose resident logical accounting".to_string())?;
    Ok(GenerationResult {
        generated_token_ids: generated_token_ids.clone(),
        json: json!({
            "generated_token_ids": generated_token_ids,
            "steps": steps,
            "logical_lower_bound": logical_bytes_json(logical)?,
            "cache_reset_before_each_step": true,
        }),
    })
}

fn sliding_boundary_workload(executor: &mut Gemma4TextExecutor) -> Result<Value, String> {
    let window = executor.config().sliding_window;
    // Four complete ring rotations exercise both overwrite addressing and the
    // promoted M=128 prefill path at the requested N=2048 acceptance length.
    let sequence_len = window
        .checked_mul(4)
        .ok_or_else(|| "sliding-window boundary sequence length overflows usize".to_string())?;
    let boundary_tokens = vec![2_u32; sequence_len];

    executor.reset();
    executor.reset_resident_logical_bytes();
    let cached_started = Instant::now();
    let mut trace = executor.prefill(&boundary_tokens[..1])?;
    for _ in 1..sequence_len {
        trace = executor.decode(2)?;
    }
    let cached_elapsed_seconds = cached_started.elapsed().as_secs_f64();
    let cached_top1 = trace.top1;
    let cached_snapshot = executor
        .resident_kv_cache_snapshot()?
        .ok_or_else(|| "sliding-boundary cached route has no device K/V snapshot".to_string())?;
    assert_sliding_boundary_state(&cached_snapshot, window, sequence_len)?;
    let cached_logical = executor
        .resident_logical_bytes()
        .ok_or_else(|| "sliding-boundary cached route has no logical accounting".to_string())?;

    executor.reset();
    executor.reset_resident_logical_bytes();
    let reprefill_started = Instant::now();
    let reprefill_trace = executor.prefill(&boundary_tokens)?;
    let reprefill_elapsed_seconds = reprefill_started.elapsed().as_secs_f64();
    let reprefill_snapshot = executor.resident_kv_cache_snapshot()?.ok_or_else(|| {
        "sliding-boundary re-prefill route has no device K/V snapshot".to_string()
    })?;
    assert_sliding_boundary_state(&reprefill_snapshot, window, sequence_len)?;
    if reprefill_trace.top1.token_id != cached_top1.token_id {
        return Err(format!(
            "sliding-window boundary cache/no-cache top1 differs at length {sequence_len}: cached={} reprefill={}",
            cached_top1.token_id, reprefill_trace.top1.token_id
        ));
    }
    let reprefill_logical = executor
        .resident_logical_bytes()
        .ok_or_else(|| "sliding-boundary re-prefill has no logical accounting".to_string())?;
    Ok(json!({
        "kind": "sliding-window-boundary",
        "sliding_window_from_config": window,
        "sequence_length": sequence_len,
        "input_token_pattern": {"token_id": 2, "description": "BOS token repeated so position/RoPE and cache order, not tokenizer variability, exercise the boundary"},
        "cached_m1_route": {
            "top1_token_id": cached_top1.token_id,
            "top1_logit": cached_top1.logit,
            "elapsed_seconds": cached_elapsed_seconds,
            "logical_lower_bound": logical_bytes_json(cached_logical)?,
            "kv_snapshot": snapshot_json(Some(cached_snapshot)),
        },
        "full_mn_reprefill_route": {
            "top1_token_id": reprefill_trace.top1.token_id,
            "top1_logit": reprefill_trace.top1.logit,
            "elapsed_seconds": reprefill_elapsed_seconds,
            "logical_lower_bound": logical_bytes_json(reprefill_logical)?,
            "kv_snapshot": snapshot_json(Some(reprefill_snapshot)),
        },
        "cache_and_full_reprefill_top1_match": true,
    }))
}

/// One cold-cache prefill with no warmup or decode.  This deliberately exists
/// for profiler attribution: the trace contains exactly `35 * N` Gemma4
/// paged-reader launches, rather than a warmup and a follow-on decode.
fn attention_profile_workload(
    executor: &mut Gemma4TextExecutor,
    prompt_tokens: usize,
    prompt_token_id: u32,
) -> Result<Value, String> {
    let prompt = vec![prompt_token_id; prompt_tokens];
    executor.reset();
    executor.reset_resident_logical_bytes();
    executor.reset_resident_host_profile();
    let started = Instant::now();
    let trace = executor.prefill(&prompt)?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let logical = executor
        .resident_logical_bytes()
        .ok_or_else(|| "attention profile has no logical accounting".to_string())?;
    let host_profile = executor
        .resident_host_profile()
        .ok_or_else(|| "attention profile has no resident host profile".to_string())?;
    let expected_attention_calls = expected_prefill_reader_calls(prompt_tokens)?;
    if usize::try_from(logical.attention_calls).ok() != Some(expected_attention_calls) {
        return Err(format!(
            "attention profile expected {expected_attention_calls} paged-reader calls, got {}",
            logical.attention_calls
        ));
    }
    Ok(json!({
        "kind": "resident-attention-profile",
        "method": "one cold-cache prefill only; no warmup or decode is included",
        "prompt": {
            "token_count": prompt_tokens,
            "token_id": prompt_token_id,
        },
        "elapsed_seconds": elapsed_seconds,
        "prefill_tokens_per_second": (prompt_tokens as f64) / elapsed_seconds,
        "top1_token_id": trace.top1.token_id,
        "logical_lower_bound": logical_bytes_json(logical)?,
        "expected_paged_reader_calls": expected_attention_calls,
        "host_profile": host_profile_json(host_profile),
    }))
}

/// The profiling workload reports *physical reader launches*. Both the 28
/// local layers and seven full-attention layers emit one reader per query tile
/// by default; keep the legacy local M=1 count when its documented rollback is
/// active so this diagnostic validates either execution route honestly.
fn expected_prefill_reader_calls(prompt_tokens: usize) -> Result<usize, String> {
    let sliding = if env::var("ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED")
        .ok()
        .as_deref()
        != Some("0")
    {
        let tile_tokens = env::var("ULLM_GEMMA4_PREFILL_ACTIVATION_CHUNK_TOKENS")
            .ok().map(|value| value.parse::<usize>())
            .transpose().map_err(|_| "ULLM_GEMMA4_PREFILL_ACTIVATION_CHUNK_TOKENS must be an integer".to_string())?
            .unwrap_or(GEMMA4_PREFILL_QUERY_TILE_TOKENS);
        prompt_tokens.div_ceil(tile_tokens).checked_mul(GEMMA4_SLIDING_ATTENTION_LAYERS)
            .ok_or_else(|| "batched sliding reader launch count overflows usize".to_string())?
    } else {
        prompt_tokens.checked_mul(GEMMA4_SLIDING_ATTENTION_LAYERS)
            .ok_or_else(|| "sliding reader launch count overflows usize".to_string())?
    };
    let full = if env::var("ULLM_GEMMA4_PREFILL_LAYER_MAJOR").ok().as_deref() == Some("0") {
        prompt_tokens
            .checked_mul(GEMMA4_FULL_ATTENTION_LAYERS)
            .ok_or_else(|| "full reader launch count overflows usize".to_string())?
    } else {
        let tile_tokens = match env::var("ULLM_GEMMA4_PREFILL_ACTIVATION_CHUNK_TOKENS") {
            Ok(value) => value.parse::<usize>().map_err(|_| {
                "ULLM_GEMMA4_PREFILL_ACTIVATION_CHUNK_TOKENS must be an integer".to_string()
            })?,
            Err(env::VarError::NotPresent) => GEMMA4_PREFILL_QUERY_TILE_TOKENS,
            Err(error) => {
                return Err(format!(
                    "failed to read ULLM_GEMMA4_PREFILL_ACTIVATION_CHUNK_TOKENS: {error}"
                ));
            }
        };
        if !(1..=GEMMA4_PREFILL_QUERY_TILE_TOKENS).contains(&tile_tokens) {
            return Err(format!(
                "ULLM_GEMMA4_PREFILL_ACTIVATION_CHUNK_TOKENS must be in 1..={GEMMA4_PREFILL_QUERY_TILE_TOKENS}, got {tile_tokens}"
            ));
        }
        prompt_tokens
            .div_ceil(tile_tokens)
            .checked_mul(GEMMA4_FULL_ATTENTION_LAYERS)
            .ok_or_else(|| "batched full reader launch count overflows usize".to_string())?
    };
    sliding
        .checked_add(full)
        .ok_or_else(|| "total reader launch count overflows usize".to_string())
}

fn benchmark_workload(
    executor: &mut Gemma4TextExecutor,
    repeats: usize,
    prompt_tokens: usize,
    decode_tokens_per_repeat: usize,
    prompt_token_id: u32,
) -> Result<Value, String> {
    let prompt = vec![prompt_token_id; prompt_tokens];
    executor.reset();
    let warmup = executor.prefill(&prompt)?;
    let mut prefill_runs = Vec::with_capacity(repeats);
    let mut prefill_tokens = 0_usize;
    let mut prefill_seconds = 0.0_f64;
    for _ in 0..repeats {
        executor.reset();
        executor.reset_resident_logical_bytes();
        executor.reset_resident_host_profile();
        let started = Instant::now();
        let trace = executor.prefill(&prompt)?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let logical = executor
            .resident_logical_bytes()
            .ok_or_else(|| "prefill run has no logical accounting".to_string())?;
        let host_profile = executor
            .resident_host_profile()
            .ok_or_else(|| "prefill run has no resident host profile".to_string())?;
        prefill_tokens = prefill_tokens
            .checked_add(prompt.len())
            .ok_or_else(|| "prefill token count overflows usize".to_string())?;
        prefill_seconds += elapsed_seconds;
        prefill_runs.push(json!({
            "input_tokens": prompt.len(),
            "elapsed_seconds": elapsed_seconds,
            "top1_token_id": trace.top1.token_id,
            "logical_lower_bound": logical_bytes_json(logical)?,
            "host_profile": host_profile_json(host_profile),
        }));
    }

    let mut decode_runs = Vec::with_capacity(repeats);
    let mut decode_tokens = 0_usize;
    let mut decode_seconds = 0.0_f64;
    for _ in 0..repeats {
        executor.reset();
        let prefill_trace = executor.prefill(&prompt)?;
        let mut input_token_id = prefill_trace.top1.token_id;
        executor.reset_resident_logical_bytes();
        executor.reset_resident_host_profile();
        let started = Instant::now();
        let mut context_lengths_after_append = Vec::with_capacity(decode_tokens_per_repeat);
        let mut generated = Vec::with_capacity(decode_tokens_per_repeat);
        for _ in 0..decode_tokens_per_repeat {
            let trace = executor.decode(input_token_id)?;
            context_lengths_after_append.push(executor.position());
            input_token_id = trace.top1.token_id;
            generated.push(input_token_id);
        }
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let logical = executor
            .resident_logical_bytes()
            .ok_or_else(|| "decode run has no logical accounting".to_string())?;
        let host_profile = executor
            .resident_host_profile()
            .ok_or_else(|| "decode run has no resident host profile".to_string())?;
        decode_tokens = decode_tokens
            .checked_add(decode_tokens_per_repeat)
            .ok_or_else(|| "decode token count overflows usize".to_string())?;
        decode_seconds += elapsed_seconds;
        decode_runs.push(json!({
            "generated_tokens": generated,
            "context_lengths_after_append": context_lengths_after_append,
            "elapsed_seconds": elapsed_seconds,
            "logical_lower_bound": logical_bytes_json(logical)?,
            "host_profile": host_profile_json(host_profile),
        }));
    }
    let prefill_tok_s = (prefill_tokens as f64) / prefill_seconds;
    let decode_tok_s = (decode_tokens as f64) / decode_seconds;
    Ok(json!({
        "kind": "resident-throughput",
        "prompt": {
            "token_count": prompt.len(),
            "token_id": prompt_token_id,
            "construction": "one fixed valid Gemma tokenizer token repeated to the requested context length; this makes every repeat and both before/after builds use the exact same token IDs",
        },
        "warmup": {
            "prompt_token_count": prompt.len(),
            "top1_token_id": warmup.top1.token_id,
            "excluded_from_timing": true,
        },
        "prefill": {
            "runs": prefill_runs,
            "accounting": "Each timed operation is an M=N prefill of the declared fixed-token prompt after resident load. Total input tokens divided by total monotonic wall time; load, cooldown, warmup, and profiler time are excluded.",
            "total_tokens": prefill_tokens,
            "total_elapsed_seconds": prefill_seconds,
            "tok_per_second": prefill_tok_s,
            "milliseconds_per_token": 1000.0 / prefill_tok_s,
        },
        "decode": {
            "runs": decode_runs,
            "tokens_per_repeat": decode_tokens_per_repeat,
            "accounting": "Each timed operation is M=1 decode after an untimed prefill of the declared fixed-token prompt. Generated tokens per repeat are divided by total monotonic wall time; load, the setup prefill, cooldown, warmup, and profiler time are excluded.",
            "total_tokens": decode_tokens,
            "total_elapsed_seconds": decode_seconds,
            "tok_per_second": decode_tok_s,
            "milliseconds_per_token": 1000.0 / decode_tok_s,
        },
        "logical_stream_definition": "For each timed run, BF16 source weights read by each resident matvec or BF16 row operation plus F32 K/V writes and unique K+V cache reads per attention invocation. It is a logical lower bound, not a physical-HBM counter; activation transfers, output writes, page tables, allocator traffic, L2 reuse, and kernel-launch overhead are excluded.",
    }))
}

fn assert_e2b_shared_sources(snapshot: &Gemma4ResidentKvCacheSnapshot) -> Result<(), String> {
    if snapshot.shared_layer_sources.len() != 20 {
        return Err(format!(
            "E2B expected 20 shared K/V layers, observed {}",
            snapshot.shared_layer_sources.len()
        ));
    }
    for shared in &snapshot.shared_layer_sources {
        let expected = match shared.layer_kind.as_str() {
            "sliding_attention" => 13,
            "full_attention" => 14,
            other => {
                return Err(format!(
                    "E2B shared layer {} has unsupported kind {other}",
                    shared.layer_index
                ));
            }
        };
        if shared.layer_index < 15 || shared.source_layer_index != expected {
            return Err(format!(
                "E2B shared K/V source mismatch at layer {}: kind={} source={} expected={expected}",
                shared.layer_index, shared.layer_kind, shared.source_layer_index
            ));
        }
    }
    Ok(())
}

fn assert_sliding_boundary_state(
    snapshot: &Gemma4ResidentKvCacheSnapshot,
    window: usize,
    sequence_len: usize,
) -> Result<(), String> {
    for source in &snapshot.source_layers {
        if source.layer_kind == "sliding_attention"
            && (source.capacity_tokens != window
                || source.cache_len != window
                || source.absolute_len != sequence_len)
        {
            return Err(format!(
                "sliding source layer {} state is capacity={} cache_len={} absolute_len={}, expected {window}/{window}/{sequence_len}",
                source.layer_index, source.capacity_tokens, source.cache_len, source.absolute_len
            ));
        }
    }
    Ok(())
}

fn logical_bytes_json(bytes: Gemma4ResidentLogicalBytes) -> Result<Value, String> {
    Ok(json!({
        "bf16_weight_bytes": bytes.bf16_weight_bytes,
        "kv_read_bytes": bytes.kv_read_bytes,
        "kv_write_bytes": bytes.kv_write_bytes,
        "total_bytes": bytes.total_bytes()?,
        "matvec_calls": bytes.matvec_calls,
        "bf16_row_reads": bytes.bf16_row_reads,
        "attention_calls": bytes.attention_calls,
    }))
}

fn host_profile_json(profile: Gemma4ResidentHostProfile) -> Value {
    json!({
        "units": "nanoseconds measured with std::time::Instant; primitive_ns is inclusive",
        "token_forward_ns": profile.token_forward_ns,
        "primitive_ns": profile.primitive_ns,
        "executor_other_ns": profile.executor_other_ns,
        "input_encode_ns": profile.input_encode_ns,
        "output_allocation_ns": profile.output_allocation_ns,
        "buffer_ensure_ns": profile.buffer_ensure_ns,
        "buffer_allocate_ns": profile.buffer_allocate_ns,
        "h2d_submit_ns": profile.h2d_submit_ns,
        "kernel_submit_ns": profile.kernel_submit_ns,
        "d2h_submit_ns": profile.d2h_submit_ns,
        "stream_synchronize_ns": profile.stream_synchronize_ns,
        "output_decode_validate_ns": profile.output_decode_validate_ns,
        "kv_table_host_ns": profile.kv_table_host_ns,
        "gemma_batched_matmul": {
            "primitive_ns": profile.gemma_batched_matmul_ns,
            "calls": profile.gemma_batched_matmul_calls,
            "units": "nanoseconds measured with std::time::Instant; end-to-end host round trip including H2D, kernel, D2H, and synchronization",
        },
        "calls": {
            "matvec": profile.matvec_calls,
            "row": profile.row_calls,
            "attention": profile.attention_calls,
            "kv_write": profile.kv_write_calls,
        },
        "by_primitive": {
            "matvec": primitive_host_profile_json(profile.matvec),
            "bf16_row": primitive_host_profile_json(profile.bf16_row),
            "attention": primitive_host_profile_json(profile.attention),
            "kv_write": primitive_host_profile_json(profile.kv_write),
        },
        "attention_region": {
            "units": "nanoseconds measured with std::time::Instant; complete resident attention region including its final synchronization",
            "sliding": {
                "primitive_ns": profile.attention_region.sliding_ns,
                "calls": profile.attention_region.sliding_calls,
                "layer_indices_per_token": 28,
                "geometry": "8Q/1KV/256/256, local window 512",
            },
            "full": {
                "primitive_ns": profile.attention_region.full_ns,
                "calls": profile.attention_region.full_calls,
                "layer_indices_per_token": 7,
                "geometry": "8Q/1KV/512/512, full context",
            },
        },
        "attention_components": {
            "units": "nanoseconds measured with std::time::Instant; reader_round_trip_ns includes reader staging, kernel, D2H, and its final stream synchronization; GPU-kernel-only reader time is measured separately with rocprof",
            "sliding": attention_component_host_profile_json(profile.attention_components.sliding),
            "full": attention_component_host_profile_json(profile.attention_components.full),
        },
    })
}

fn attention_component_host_profile_json(
    profile: ullm_engine::gemma4_text_executor::Gemma4ResidentAttentionComponentHostProfile,
) -> Value {
    json!({
        "input_rms_norm_ns": profile.input_rms_norm_ns,
        "q_projection_ns": profile.q_projection_ns,
        "k_projection_ns": profile.k_projection_ns,
        "v_projection_ns": profile.v_projection_ns,
        "rope_and_head_norm_ns": profile.rope_and_head_norm_ns,
        "kv_write_ns": profile.kv_write_ns,
        "reader_round_trip_ns": profile.reader_round_trip_ns,
        "output_projection_ns": profile.output_projection_ns,
        "post_attention_norm_ns": profile.post_attention_norm_ns,
        "residual_ns": profile.residual_ns,
        "residual_host_or_sync_ns": profile.residual_host_or_sync_ns,
        "calls": profile.calls,
    })
}

fn primitive_host_profile_json(profile: Gemma4ResidentPrimitiveHostProfile) -> Value {
    json!({
        "primitive_ns": profile.primitive_ns,
        "input_encode_ns": profile.input_encode_ns,
        "output_allocation_ns": profile.output_allocation_ns,
        "h2d_submit_ns": profile.h2d_submit_ns,
        "kernel_submit_ns": profile.kernel_submit_ns,
        "d2h_submit_ns": profile.d2h_submit_ns,
        "stream_synchronize_ns": profile.stream_synchronize_ns,
        "output_decode_validate_ns": profile.output_decode_validate_ns,
        "kv_table_host_ns": profile.kv_table_host_ns,
        "calls": profile.calls,
    })
}

fn snapshot_json(snapshot: Option<Gemma4ResidentKvCacheSnapshot>) -> Value {
    let Some(snapshot) = snapshot else {
        return Value::Null;
    };
    json!({
        "source_layers": snapshot.source_layers.iter().map(|source| json!({
            "layer_index": source.layer_index,
            "layer_kind": source.layer_kind,
            "capacity_tokens": source.capacity_tokens,
            "cache_len": source.cache_len,
            "absolute_len": source.absolute_len,
            "allocated_bytes": source.allocated_bytes,
        })).collect::<Vec<_>>(),
        "shared_layer_sources": snapshot.shared_layer_sources.iter().map(|shared| json!({
            "layer_index": shared.layer_index,
            "layer_kind": shared.layer_kind,
            "source_layer_index": shared.source_layer_index,
        })).collect::<Vec<_>>(),
    })
}

fn memory_json(
    plan: &ullm_engine::gemma4_text_executor::Gemma4ResidentMemoryPlan,
    device_total_bytes: u64,
    actual_weight_bytes: u64,
    actual_kv_bytes: u64,
    actual_transient_bytes: u64,
    actual_total_bytes: u64,
) -> Result<Value, String> {
    let mut contexts = vec![1_usize, plan.local_kv_capacity_tokens, 4096, 32_768];
    if !contexts.contains(&plan.max_context_tokens) {
        contexts.push(plan.max_context_tokens);
    }
    contexts.sort_unstable();
    contexts.dedup();
    let context_table = contexts
        .into_iter()
        .map(|context_tokens| {
            let kv_bytes = plan.estimated_kv_bytes(context_tokens)?;
            let device_bytes = plan.estimated_device_bytes(context_tokens)?;
            let headroom_bytes = device_total_bytes
                .checked_sub(device_bytes)
                .ok_or_else(|| {
                    format!("Gemma4 planned device bytes exceed R9700 at context {context_tokens}")
                })?;
            Ok(json!({
                "context_tokens": context_tokens,
                "demand_kv_bytes": kv_bytes,
                "demand_total_device_bytes": device_bytes,
                "headroom_to_reported_total_vram_bytes": headroom_bytes,
                "fits": true,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let max_kv = plan.estimated_kv_bytes(plan.max_context_tokens)?;
    let full_file_plus_max = plan
        .source_model_file_bytes
        .checked_add(max_kv)
        .and_then(|bytes| bytes.checked_add(plan.device_transient_bytes))
        .ok_or_else(|| "Gemma4 full-source fit calculation overflows u64".to_string())?;
    Ok(json!({
        "source_model_file_bytes": plan.source_model_file_bytes,
        "source_payload_bytes": plan.source_payload_bytes,
        "resident_checkpoint_weight_bytes_planned": plan.resident_checkpoint_weight_bytes,
        "resident_checkpoint_tensor_count_planned": plan.resident_checkpoint_tensor_count,
        "text_decoder_weight_bytes_executed": plan.text_weight_bytes,
        "text_decoder_tensor_count_executed": plan.text_tensor_count,
        "ple_weight_bytes_included_in_text_resident_plan": plan.ple_weight_bytes,
        "unexecuted_multimodal_payload_bytes": plan.unexecuted_multimodal_weight_bytes,
        "kv_geometry": {
            "local_kv_source_layers": plan.local_kv_source_layers,
            "full_kv_source_layers": plan.full_kv_source_layers,
            "local_kv_capacity_tokens": plan.local_kv_capacity_tokens,
            "local_kv_fixed_bytes_including_tables": plan.local_kv_bytes,
            "full_kv_bytes_per_token": plan.full_kv_bytes_per_token,
            "full_kv_page_table_bytes_per_token": plan.page_table_bytes_per_full_token,
        },
        "transient_device_bytes_planned": plan.device_transient_bytes,
        "context_demand_table": context_table,
        "config_max_context_tokens": plan.max_context_tokens,
        "full_source_file_plus_max_context_kv_and_planned_transient_bytes": full_file_plus_max,
        "full_source_file_plus_max_context_fits_reported_vram": full_file_plus_max <= device_total_bytes,
        "actual_runtimebuffer_accounting": {
            "resident_checkpoint_weight_bytes": actual_weight_bytes,
            "device_kv_bytes_at_config_max_capacity": actual_kv_bytes,
            "temporary_buffer_bytes_after_workload": actual_transient_bytes,
            "total_runtimebuffer_bytes": actual_total_bytes,
            "reported_total_vram_bytes": device_total_bytes,
            "headroom_to_reported_total_vram_bytes": device_total_bytes.checked_sub(actual_total_bytes),
            "scope_note": "RuntimeBuffer accounting excludes HIP context/allocator overhead. amd-smi telemetry is captured while the resident allocation is live.",
        },
    }))
}

fn write_json_new(path: &PathBuf, value: &Value) -> Result<(), String> {
    let encoded = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to serialize resident evidence JSON: {error}"))?;
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("failed to create evidence {}: {error}", path.display()))?;
    file.write_all(&encoded)
        .map_err(|error| format!("failed to write evidence {}: {error}", path.display()))?;
    file.write_all(b"\n")
        .map_err(|error| format!("failed to finish evidence {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync evidence {}: {error}", path.display()))?;
    Ok(())
}
