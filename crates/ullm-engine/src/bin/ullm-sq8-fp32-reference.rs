// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Execute a fixed CPU-only, strict artifact-FP32 Qwen3-14B reference run.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;
use ullm_engine::sq8_fp32_reference::{
    ARTIFACT_FP32_REFERENCE_SCHEMA_VERSION, ArtifactFp32ForwardSummary,
    ArtifactFp32ProjectionCrossCheck, ArtifactFp32ReferenceIdentity, ArtifactFp32ReferenceModel,
    QWEN3_14B_FP32_REFERENCE_DEFAULT_THREADS, QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT,
    duration_seconds, process_peak_rss_kib, write_forward_capture,
};

#[derive(Debug)]
struct Options {
    artifact: PathBuf,
    package: PathBuf,
    output: PathBuf,
    token_id: u32,
    forwards: usize,
    threads: usize,
    max_context: usize,
    capture: bool,
    cross_check_layer0_q: bool,
}

#[derive(Debug, Serialize)]
struct StepRecord {
    forward_index: usize,
    elapsed_seconds: f64,
    summary: ArtifactFp32ForwardSummary,
}

#[derive(Debug, Serialize)]
struct ProjectionCrossCheckRecord {
    elapsed_seconds: f64,
    result: ArtifactFp32ProjectionCrossCheck,
}

#[derive(Debug, Serialize)]
struct RunReceipt {
    schema_version: &'static str,
    status: &'static str,
    execution_backend: &'static str,
    seed: u64,
    executable: String,
    executable_sha256: String,
    cpu_model: Option<String>,
    identity: ArtifactFp32ReferenceIdentity,
    initialization_elapsed_seconds: f64,
    cross_check_layer0_q: Option<ProjectionCrossCheckRecord>,
    forward_steps: Vec<StepRecord>,
    peak_rss_kib: Option<u64>,
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    if options.output.exists() {
        return Err(format!(
            "output path already exists; refusing to clobber {}",
            options.output.display()
        ));
    }
    fs::create_dir_all(&options.output).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            options.output.display()
        )
    })?;

    let initialization_start = Instant::now();
    let model =
        ArtifactFp32ReferenceModel::open(&options.artifact, &options.package, options.threads)?;
    let initialization_elapsed = initialization_start.elapsed();
    let identity = model.identity();
    let mut session = model.session(options.max_context)?;
    let cross_check_layer0_q = if options.cross_check_layer0_q {
        let cross_check_start = Instant::now();
        let result = model.cross_check_layer0_q_projection(options.token_id)?;
        Some(ProjectionCrossCheckRecord {
            elapsed_seconds: duration_seconds(cross_check_start.elapsed()),
            result,
        })
    } else {
        None
    };
    let mut next_token = options.token_id;
    let mut forward_steps = Vec::with_capacity(options.forwards);
    for forward_index in 0..options.forwards {
        let start = Instant::now();
        let forward = session.forward_token(next_token)?;
        let elapsed = start.elapsed();
        if options.capture {
            let capture_dir = options.output.join(format!("forward-{forward_index:04}"));
            write_forward_capture(&capture_dir, &identity, &forward)?;
        }
        next_token = forward.summary.greedy_token_id;
        forward_steps.push(StepRecord {
            forward_index,
            elapsed_seconds: duration_seconds(elapsed),
            summary: forward.summary,
        });
    }

    let executable = std::env::current_exe()
        .map_err(|err| format!("failed to resolve current executable: {err}"))?;
    let receipt = RunReceipt {
        schema_version: ARTIFACT_FP32_REFERENCE_SCHEMA_VERSION,
        status: "ok",
        execution_backend: "cpu_only_no_runtime_context",
        seed: 0,
        executable: executable.display().to_string(),
        executable_sha256: sha256_file(&executable)?,
        cpu_model: cpu_model(),
        identity,
        initialization_elapsed_seconds: duration_seconds(initialization_elapsed),
        cross_check_layer0_q,
        forward_steps,
        peak_rss_kib: process_peak_rss_kib()?,
    };
    write_json_create_new(&options.output.join("run.json"), &receipt)?;
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let artifact = PathBuf::from(args.next().ok_or_else(usage)?);
    let package = PathBuf::from(args.next().ok_or_else(usage)?);
    let output = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut token_id = 1_u32;
    let mut forwards = 1_usize;
    let mut threads = QWEN3_14B_FP32_REFERENCE_DEFAULT_THREADS;
    let mut max_context = QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT;
    let mut capture = true;
    let mut cross_check_layer0_q = false;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--token-id" => token_id = parse_value(args.next(), "--token-id")?,
            "--forwards" => forwards = parse_value(args.next(), "--forwards")?,
            "--threads" => threads = parse_value(args.next(), "--threads")?,
            "--max-context" => max_context = parse_value(args.next(), "--max-context")?,
            "--no-capture" => capture = false,
            "--cross-check-layer0-q" => cross_check_layer0_q = true,
            "--help" | "-h" => return Err(usage()),
            _ => return Err(format!("unknown argument {argument:?}\n{}", usage())),
        }
    }
    if forwards == 0 {
        return Err("--forwards must be greater than zero".into());
    }
    if forwards > max_context {
        return Err("--forwards must not exceed --max-context".into());
    }
    Ok(Options {
        artifact,
        package,
        output,
        token_id,
        forwards,
        threads,
        max_context,
        capture,
        cross_check_layer0_q,
    })
}

fn parse_value<T>(value: Option<String>, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse::<T>()
        .map_err(|err| format!("invalid {name}: {err}"))
}

fn usage() -> String {
    "usage: ullm-sq8-fp32-reference ARTIFACT_DIR PACKAGE_DIR OUTPUT_DIR [--token-id N] [--forwards N] [--threads N] [--max-context N] [--no-capture] [--cross-check-layer0-q]".to_string()
}

fn sha256_file(path: &PathBuf) -> Result<String, String> {
    let mut file = fs::File::open(path)
        .map_err(|err| format!("failed to open executable {}: {err}", path.display()))?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to hash executable {}: {err}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn cpu_model() -> Option<String> {
    let content = fs::read_to_string("/proc/cpuinfo").ok()?;
    content.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name").then(|| value.trim().to_string())
    })
}

fn write_json_create_new(path: &PathBuf, value: &impl Serialize) -> Result<(), String> {
    let payload = serde_json::to_vec_pretty(value)
        .map_err(|err| format!("failed to serialize run receipt: {err}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    file.write_all(&payload)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    file.write_all(b"\n")
        .map_err(|err| format!("failed to finish {}: {err}", path.display()))?;
    Ok(())
}
