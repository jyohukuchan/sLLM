// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Read-only regression probe for the existing Qwen3.5-9B AQ4_0 resident path.
//!
//! It deliberately invokes the public production model runtime without a
//! worker, gateway, service mutation, or active-manifest read/write.  It is
//! used to demonstrate that MoE loader work did not alter the dense path.

use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::Qwen35Aq4ModelLoadConfig;
use ullm_engine::qwen35_aq4_model_runtime::Qwen35Aq4ModelRuntime;
use ullm_engine::qwen35_aq4_session::{QWEN35_AQ4_ROPE_BASE, QWEN35_AQ4_ROTARY_DIM};

const DEFAULT_PACKAGE: &str = "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package";

#[derive(Debug)]
struct Args {
    package_dir: PathBuf,
    token_ids: Vec<usize>,
    device_index: u32,
    context_length: usize,
    expected_top1: Option<usize>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-qwen35-aq4-baseline-probe: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.token_ids.len() > args.context_length {
        return Err("--token-ids exceeds --context-length".into());
    }
    let mut runtime = Qwen35Aq4ModelRuntime::load(Qwen35Aq4ModelLoadConfig {
        package_dir: args.package_dir.clone(),
        device_index: args.device_index,
        expected_architecture: Some("gfx1201".into()),
        chunk_bytes: 64 * 1024 * 1024,
        context_length: args.context_length,
        kv_block_size: 256,
        layer_indices: None,
        lm_head_mode: PackageLmHeadMode::GpuResidentF32,
        lm_head_chunk_rows: 1024,
    })?;
    let mut top_logits = Vec::new();
    for (position, token_id) in args.token_ids.iter().copied().enumerate() {
        runtime.dispatch_token(
            token_id,
            QWEN35_AQ4_ROTARY_DIM,
            QWEN35_AQ4_ROPE_BASE,
            position,
            position,
            false,
            "Qwen3.5 AQ4 baseline regression",
        )?;
        top_logits = runtime.top_logits_from_last_layer(10, "Qwen3.5 AQ4 baseline regression")?;
    }
    let top1 = top_logits
        .first()
        .ok_or_else(|| "Qwen3.5 AQ4 baseline returned no logits".to_string())?;
    if let Some(expected) = args.expected_top1
        && top1.token_id != expected
    {
        return Err(format!(
            "Qwen3.5 AQ4 baseline top-1 changed: got {} expected {expected}",
            top1.token_id
        ));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "ullm.qwen35_aq4_baseline_probe.v0.1",
            "format_id": "AQ4_0",
            "package_dir": args.package_dir,
            "device_index": args.device_index,
            "backend": runtime.backend(),
            "device_name": runtime.device_name(),
            "token_count": args.token_ids.len(),
            "top1": {"token_id": top1.token_id, "logit": top1.logit},
            "top10": top_logits.iter().map(|entry| json!({
                "token_id": entry.token_id,
                "logit": entry.logit,
            })).collect::<Vec<_>>(),
            "expected_top1": args.expected_top1,
        }))
        .map_err(|error| format!("failed to serialize baseline probe: {error}"))?
    );
    runtime.reset_all_request_state_synchronized()
}

fn parse_args() -> Result<Args, String> {
    let mut package_dir = PathBuf::from(DEFAULT_PACKAGE);
    let mut token_ids = None;
    let mut device_index = 0_u32;
    let mut context_length = 256_usize;
    let mut expected_top1 = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--package" => package_dir = PathBuf::from(required(&mut arguments, "--package")?),
            "--token-ids" => {
                token_ids = Some(parse_ids(&required(&mut arguments, "--token-ids")?)?)
            }
            "--device-index" => {
                device_index = required(&mut arguments, "--device-index")?
                    .parse()
                    .map_err(|error| format!("invalid --device-index: {error}"))?
            }
            "--context-length" => {
                context_length = parse_usize(
                    "--context-length",
                    &required(&mut arguments, "--context-length")?,
                )?
            }
            "--expected-top1" => {
                expected_top1 = Some(parse_usize(
                    "--expected-top1",
                    &required(&mut arguments, "--expected-top1")?,
                )?)
            }
            "--help" | "-h" => return Err(usage().to_string()),
            _ => return Err(format!("unknown argument {argument:?}; {}", usage())),
        }
    }
    let token_ids = token_ids.ok_or_else(|| format!("--token-ids is required; {}", usage()))?;
    if token_ids.is_empty() || context_length == 0 {
        return Err("--token-ids and --context-length must be nonzero".into());
    }
    Ok(Args {
        package_dir,
        token_ids,
        device_index,
        context_length,
        expected_top1,
    })
}

fn required(arguments: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{name} needs a value"))
}

fn parse_usize(name: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

fn parse_ids(value: &str) -> Result<Vec<usize>, String> {
    value
        .split(',')
        .map(|part| parse_usize("--token-ids", part.trim()))
        .collect()
}

fn usage() -> &'static str {
    "usage: ullm-qwen35-aq4-baseline-probe --token-ids ID[,ID...] [--package DIR] [--device-index N] [--context-length N] [--expected-top1 ID]"
}
