// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Offline, decode-first Qwen3.5-35B-A3B AQ4_0 MoE generation driver.
//!
//! The caller is responsible for isolating the R9700, for example with
//! `HIP_VISIBLE_DEVICES=2`; this binary then requires that runtime-visible
//! device 0 reports `gfx1201`.  It never starts or contacts a serving service.

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ullm_engine::kv_cache_dtype::KvCacheDtypes;
use ullm_engine::qwen35_moe_aq4_runtime::{
    Qwen35MoeAq4ModelLoadConfig, Qwen35MoeAq4Runtime, QWEN35_MOE_DEFAULT_CONTEXT_LENGTH,
    QWEN35_MOE_DEFAULT_KV_BLOCK_SIZE,
};

const DEFAULT_PACKAGE: &str =
    "/home/homelab1/datapool/ullm/product/qwen35-35b-a3b-aq4_0-g8-moe-v0.2/package";

#[derive(Debug, Clone)]
struct Args {
    package_dir: PathBuf,
    prompt_token_ids: Vec<usize>,
    max_new_tokens: usize,
    device_index: u32,
    context_length: usize,
    kv_block_size: usize,
    output: Option<PathBuf>,
    hold_seconds: u64,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-qwen35-moe-aq4-generate: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    // Parse the exact environment contract once for the evidence record.  The
    // resident full-attention layers read the same selectors while allocating
    // their persistent K/V cache, so an invalid request fails before any
    // weight residency attempt.
    let kv_cache_dtypes = KvCacheDtypes::from_env()?;
    let load_started = Instant::now();
    let mut config =
        Qwen35MoeAq4ModelLoadConfig::production_sized(args.package_dir.clone(), args.device_index);
    config.context_length = args.context_length;
    config.kv_block_size = args.kv_block_size;
    let mut runtime = Qwen35MoeAq4Runtime::load(config)?;
    let load_wall_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let residency = runtime.residency();
    let generation = runtime.generate_greedy(&args.prompt_token_ids, args.max_new_tokens)?;
    let router_verification = runtime.verify_last_token_routes()?;
    let strict_mismatches = router_verification
        .iter()
        .filter(|entry| entry.strict_order_match == Some(false))
        .map(|entry| entry.layer_index)
        .collect::<Vec<_>>();
    if !strict_mismatches.is_empty() {
        return Err(format!(
            "independent raw-BF16 router verification disagreed on tie-free layers {strict_mismatches:?}"
        ));
    }

    let decode_wall_seconds = generation.decode_wall_ms / 1000.0;
    let output = json!({
        "schema": "ullm.qwen35_moe_aq4_generation.v0.1",
        "format_id": "AQ4_0",
        "model": "Qwen3.5-35B-A3B",
        "device": {
            "runtime_visible_index": args.device_index,
            "required_architecture": "gfx1201"
        },
        "package_dir": args.package_dir,
        "kv_cache": {
            "key_dtype": kv_cache_dtypes.key.as_str(),
            "value_dtype": kv_cache_dtypes.value.as_str(),
        },
        "load_wall_ms": load_wall_ms,
        "residency": {
            "declared_package_bytes": residency.declared_package_bytes,
            "device_total_global_mem_bytes": residency.device_total_global_mem_bytes,
            "context_length": residency.context_length,
            "kv_block_size": residency.kv_block_size,
            "cache_blocks": residency.cache_blocks,
            "resident_expert_payload_bytes": residency.resident_expert_payload_bytes,
            "shared_moe_decode_workspace_bytes": residency.shared_moe_decode_workspace_bytes
        },
        "generation": {
            "prompt_token_ids": generation.prompt_token_ids,
            "generated_token_ids": generation.generated_token_ids,
            "wall_ms": generation.wall_ms,
            "prompt_wall_ms": generation.prompt_wall_ms,
            "decode_wall_ms": generation.decode_wall_ms,
            "decode_tokens_per_second": (args.max_new_tokens > 0 && decode_wall_seconds > 0.0)
                .then(|| args.max_new_tokens as f64 / decode_wall_seconds),
            "final_position": generation.final_step.position,
            "final_step_wall_ms": generation.final_step.wall_ms,
            "final_top_logits": generation.final_step.top_logits.iter().map(|entry| json!({
                "token_id": entry.token_id,
                "logit": entry.logit,
            })).collect::<Vec<_>>(),
            "final_routes": generation.final_step.routes.iter().map(|route| json!({
                "layer_index": route.layer_index,
                "selected_expert_ids": route.selected_expert_ids,
                "routing_scores": route.routing_scores,
                "boundary_tie_flags": route.boundary_tie_flags,
            })).collect::<Vec<_>>()
        },
        "router_verification": router_verification.iter().map(|entry| json!({
            "layer_index": entry.layer_index,
            "runtime_selected_expert_ids": entry.runtime_selected_expert_ids,
            "reference_selected_expert_ids": entry.reference_selected_expert_ids,
            "runtime_boundary_tie_flags": entry.runtime_boundary_tie_flags,
            "reference_boundary_tie_flags": entry.reference_boundary_tie_flags,
            "strict_order_match": entry.strict_order_match,
            "routing_score_sum": entry.routing_score_sum,
        })).collect::<Vec<_>>(),
        "router_verification_tie_free_mismatches": strict_mismatches,
        "residency_hold_seconds": args.hold_seconds,
    });
    write_result(args.output.as_ref(), &output)?;
    if args.hold_seconds != 0 {
        std::thread::sleep(std::time::Duration::from_secs(args.hold_seconds));
        // Retain the model allocations for external, read-only VRAM sampling.
        std::hint::black_box(runtime.position());
    }
    Ok(())
}

fn write_result(output_path: Option<&PathBuf>, value: &Value) -> Result<(), String> {
    let encoded = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize generation result: {error}"))?;
    if let Some(path) = output_path {
        fs::write(path, format!("{encoded}\n"))
            .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    }
    println!("{encoded}");
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut package_dir = PathBuf::from(DEFAULT_PACKAGE);
    let mut prompt_token_ids = None;
    let mut max_new_tokens = 8_usize;
    let mut device_index = 0_u32;
    let mut context_length = QWEN35_MOE_DEFAULT_CONTEXT_LENGTH;
    let mut kv_block_size = QWEN35_MOE_DEFAULT_KV_BLOCK_SIZE;
    let mut output = None;
    let mut hold_seconds = 0_u64;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--package" => {
                package_dir = PathBuf::from(required_value(&mut arguments, "--package")?)
            }
            "--prompt-token-ids" => {
                prompt_token_ids = Some(parse_token_ids(&required_value(
                    &mut arguments,
                    "--prompt-token-ids",
                )?)?)
            }
            "--new-tokens" => {
                max_new_tokens = parse_usize(
                    "--new-tokens",
                    &required_value(&mut arguments, "--new-tokens")?,
                )?
            }
            "--device-index" => {
                device_index = required_value(&mut arguments, "--device-index")?
                    .parse::<u32>()
                    .map_err(|error| format!("invalid --device-index: {error}"))?
            }
            "--context-length" => {
                context_length = parse_usize(
                    "--context-length",
                    &required_value(&mut arguments, "--context-length")?,
                )?
            }
            "--kv-block-size" => {
                kv_block_size = parse_usize(
                    "--kv-block-size",
                    &required_value(&mut arguments, "--kv-block-size")?,
                )?
            }
            "--output" => output = Some(PathBuf::from(required_value(&mut arguments, "--output")?)),
            "--hold-seconds" => {
                hold_seconds = required_value(&mut arguments, "--hold-seconds")?
                    .parse::<u64>()
                    .map_err(|error| format!("invalid --hold-seconds: {error}"))?
            }
            "--help" | "-h" => return Err(usage().to_string()),
            _ => return Err(format!("unknown argument {argument:?}; {}", usage())),
        }
    }
    let prompt_token_ids =
        prompt_token_ids.ok_or_else(|| format!("--prompt-token-ids is required; {}", usage()))?;
    if prompt_token_ids.is_empty() || context_length == 0 || kv_block_size == 0 {
        return Err("prompt token IDs, context length, and KV block size must be nonzero".into());
    }
    if prompt_token_ids
        .len()
        .checked_add(max_new_tokens)
        .is_none_or(|total| total > context_length)
    {
        return Err("prompt plus requested new tokens exceeds --context-length".into());
    }
    Ok(Args {
        package_dir,
        prompt_token_ids,
        max_new_tokens,
        device_index,
        context_length,
        kv_block_size,
        output,
        hold_seconds,
    })
}

fn required_value(
    arguments: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_usize(name: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} value {value:?}: {error}"))
}

fn parse_token_ids(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|piece| parse_usize("--prompt-token-ids", piece.trim()))
        .collect()
}

fn usage() -> &'static str {
    "usage: ullm-qwen35-moe-aq4-generate --prompt-token-ids ID[,ID...] [--package DIR] [--new-tokens N] [--device-index N] [--context-length N] [--kv-block-size N] [--output FILE] [--hold-seconds N]"
}
