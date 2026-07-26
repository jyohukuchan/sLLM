// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Execute the standalone gfx1201 HIP/hipBLAS F32 control for the canonical
//! Qwen3-14B `SQ8_0` artifact.  The C++ control itself rejects any device
//! selection other than the pinned R9700 token.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use ullm_engine::sq8_fp32_reference::{
    ArtifactFp32ForwardSummary, QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT, duration_seconds,
    process_peak_rss_kib,
};
use ullm_engine::sq8_gpu_fp32_reference::{
    ARTIFACT_GPU_FP32_REFERENCE_SCHEMA_VERSION, ArtifactGpuFp32ReferenceIdentity,
    ArtifactGpuFp32ReferenceModel, write_gpu_forward_capture,
};

#[derive(Debug)]
struct Options {
    artifact: PathBuf,
    package: PathBuf,
    output: PathBuf,
    token_id: u32,
    forwards: Option<usize>,
    max_context: usize,
    capture: bool,
    teacher_forcing_run: Option<PathBuf>,
    determinism_replay: bool,
}

#[derive(Debug, Serialize)]
struct StepRecord {
    forward_index: usize,
    elapsed_seconds: f64,
    summary: ArtifactFp32ForwardSummary,
}

#[derive(Debug, Serialize)]
struct DeterminismReplay {
    replay_elapsed_seconds: f64,
    replayed_positions: usize,
    all_output_hashes_equal: bool,
}

#[derive(Debug, Serialize)]
struct RunReceipt {
    schema_version: &'static str,
    status: &'static str,
    execution_backend: &'static str,
    seed: u64,
    executable: String,
    executable_sha256: String,
    hip_visible_devices: Option<String>,
    ullm_hip_visible_devices: Option<String>,
    identity: ArtifactGpuFp32ReferenceIdentity,
    initialization_elapsed_seconds: f64,
    input_mode: &'static str,
    teacher_forcing_run: Option<String>,
    teacher_forcing_run_sha256: Option<String>,
    forward_steps: Vec<StepRecord>,
    determinism_replay: Option<DeterminismReplay>,
    peak_rss_kib: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct TeacherRunReceipt {
    forward_steps: Vec<TeacherStep>,
}

#[derive(Debug, Deserialize)]
struct TeacherStep {
    summary: TeacherSummary,
}

#[derive(Debug, Deserialize)]
struct TeacherSummary {
    input_token_id: u32,
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    if options.output.exists() {
        return Err(format!(
            "output path already exists; refusing to clobber {}",
            options.output.display()
        ));
    }
    let teacher_inputs = options
        .teacher_forcing_run
        .as_ref()
        .map(load_teacher_inputs)
        .transpose()?;
    let forwards = options
        .forwards
        .unwrap_or_else(|| teacher_inputs.as_ref().map_or(1, Vec::len));
    if forwards == 0 {
        return Err("--forwards must be greater than zero".to_string());
    }
    if forwards > options.max_context {
        return Err("--forwards must not exceed --max-context".to_string());
    }
    if let Some(inputs) = &teacher_inputs {
        if inputs.len() < forwards {
            return Err(format!(
                "teacher-forcing run contains {} positions, fewer than requested --forwards {forwards}",
                inputs.len()
            ));
        }
    }
    fs::create_dir_all(&options.output).map_err(|error| {
        format!(
            "failed to create output directory {}: {error}",
            options.output.display()
        )
    })?;

    let initialization_start = Instant::now();
    let mut model = ArtifactGpuFp32ReferenceModel::open(
        &options.artifact,
        &options.package,
        options.max_context,
    )?;
    let initialization_elapsed = initialization_start.elapsed();
    let identity = model.identity().clone();
    let mut input_tokens = Vec::with_capacity(forwards);
    let mut next_token = options.token_id;
    let mut forward_steps = Vec::with_capacity(forwards);
    for forward_index in 0..forwards {
        let input_token = teacher_inputs
            .as_ref()
            .and_then(|inputs| inputs.get(forward_index).copied())
            .unwrap_or(next_token);
        let start = Instant::now();
        let forward = model.forward_token(input_token)?;
        let elapsed = start.elapsed();
        if options.capture {
            let capture_dir = options.output.join(format!("forward-{forward_index:04}"));
            write_gpu_forward_capture(&capture_dir, &identity, &forward)?;
        }
        next_token = forward.summary.greedy_token_id;
        input_tokens.push(input_token);
        forward_steps.push(StepRecord {
            forward_index,
            elapsed_seconds: duration_seconds(elapsed),
            summary: forward.summary,
        });
    }

    let determinism_replay = if options.determinism_replay {
        let replay_start = Instant::now();
        model.reset()?;
        for (index, input_token) in input_tokens.iter().copied().enumerate() {
            let replay = model.forward_token(input_token)?;
            let expected = forward_steps
                .get(index)
                .ok_or_else(|| "GPU F32 determinism replay index escaped first run".to_string())?;
            if !summary_hashes_equal(&expected.summary, &replay.summary) {
                return Err(format!(
                    "GPU F32 determinism replay hash mismatch at position {index}"
                ));
            }
        }
        Some(DeterminismReplay {
            replay_elapsed_seconds: duration_seconds(replay_start.elapsed()),
            replayed_positions: input_tokens.len(),
            all_output_hashes_equal: true,
        })
    } else {
        None
    };

    let executable = std::env::current_exe()
        .map_err(|error| format!("failed to resolve current executable: {error}"))?;
    let teacher_forcing_run_sha256 = options
        .teacher_forcing_run
        .as_ref()
        .map(|path| sha256_file(path))
        .transpose()?;
    let receipt = RunReceipt {
        schema_version: ARTIFACT_GPU_FP32_REFERENCE_SCHEMA_VERSION,
        status: "ok",
        execution_backend: "standalone_gfx1201_hipblas_f32_control",
        seed: 0,
        executable: executable.display().to_string(),
        executable_sha256: sha256_file(&executable)?,
        hip_visible_devices: std::env::var("HIP_VISIBLE_DEVICES").ok(),
        ullm_hip_visible_devices: std::env::var("ULLM_HIP_VISIBLE_DEVICES").ok(),
        identity,
        initialization_elapsed_seconds: duration_seconds(initialization_elapsed),
        input_mode: if teacher_inputs.is_some() {
            "teacher_forced_from_cpu_reference"
        } else {
            "self_greedy_feedback"
        },
        teacher_forcing_run: options
            .teacher_forcing_run
            .as_ref()
            .map(|path| path.display().to_string()),
        teacher_forcing_run_sha256,
        forward_steps,
        determinism_replay,
        peak_rss_kib: process_peak_rss_kib()?,
    };
    write_json_create_new(&options.output.join("run.json"), &receipt)
}

fn summary_hashes_equal(
    left: &ArtifactFp32ForwardSummary,
    right: &ArtifactFp32ForwardSummary,
) -> bool {
    left.position == right.position
        && left.input_token_id == right.input_token_id
        && left.greedy_token_id == right.greedy_token_id
        && left.logits_f32le_sha256 == right.logits_f32le_sha256
        && left.final_hidden_f32le_sha256 == right.final_hidden_f32le_sha256
        && left.layer_hidden_f32le_sha256 == right.layer_hidden_f32le_sha256
}

fn load_teacher_inputs(path: &PathBuf) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|error| {
        format!(
            "failed to read teacher-forcing run {}: {error}",
            path.display()
        )
    })?;
    let receipt: TeacherRunReceipt = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "failed to parse teacher-forcing run {}: {error}",
            path.display()
        )
    })?;
    if receipt.forward_steps.is_empty() {
        return Err("teacher-forcing run has no forward steps".to_string());
    }
    Ok(receipt
        .forward_steps
        .into_iter()
        .map(|step| step.summary.input_token_id)
        .collect())
}

fn parse_options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let artifact = PathBuf::from(args.next().ok_or_else(usage)?);
    let package = PathBuf::from(args.next().ok_or_else(usage)?);
    let output = PathBuf::from(args.next().ok_or_else(usage)?);
    let mut token_id = 1_u32;
    let mut forwards = None;
    let mut max_context = QWEN3_14B_FP32_REFERENCE_MAX_CONTEXT;
    let mut capture = true;
    let mut teacher_forcing_run = None;
    let mut determinism_replay = false;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--token-id" => token_id = parse_value(args.next(), "--token-id")?,
            "--forwards" => forwards = Some(parse_value(args.next(), "--forwards")?),
            "--max-context" => max_context = parse_value(args.next(), "--max-context")?,
            "--no-capture" => capture = false,
            "--teacher-forcing-run" => {
                teacher_forcing_run =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--teacher-forcing-run requires a path".to_string()
                    })?));
            }
            "--determinism-replay" => determinism_replay = true,
            "--help" | "-h" => return Err(usage()),
            _ => return Err(format!("unknown argument {argument:?}\n{}", usage())),
        }
    }
    Ok(Options {
        artifact,
        package,
        output,
        token_id,
        forwards,
        max_context,
        capture,
        teacher_forcing_run,
        determinism_replay,
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
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn usage() -> String {
    "usage: ullm-sq8-gpu-fp32-reference ARTIFACT_DIR PACKAGE_DIR OUTPUT_DIR [--token-id N] [--forwards N] [--max-context N] [--teacher-forcing-run CPU_RUN_JSON] [--determinism-replay] [--no-capture]".to_string()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("failed to open {} for SHA-256: {error}", path.display()))?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_json_create_new(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let payload = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to serialize GPU F32 run receipt: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    file.write_all(&payload)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    file.write_all(b"\n")
        .map_err(|error| format!("failed to finish {}: {error}", path.display()))?;
    Ok(())
}
