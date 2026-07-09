// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::path::PathBuf;
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq_reference::{
    Sq8CorrectnessExecutionPath, Sq8CorrectnessFallbackState, Sq8CorrectnessReport,
    build_sq8_correctness_report, run_sq8_reference_projection, sq8_f32_le_sha256,
    sq8_reference_activation,
};
use ullm_engine::sq_runtime::{
    SQ8_CANONICAL_UPLOAD_CHUNK_BYTES, load_sq8_canonical_resident_tensor,
    run_sq8_canonical_resident_projection_f32,
};
use ullm_runtime_sys::{RuntimeContext, SqFp8ExecutionPath};

#[derive(Debug)]
struct Args {
    artifact: PathBuf,
    tensor: String,
    device_index: u32,
    report: Option<PathBuf>,
}

#[derive(Serialize)]
struct DeviceReport {
    requested_index: u32,
    runtime_device_id: i32,
    backend: String,
    name: String,
    compute_major: i32,
    compute_minor: i32,
    gcn_arch_name: String,
}

#[derive(Serialize)]
struct TensorReport {
    name: String,
    rows: usize,
    cols: usize,
    scale_rows: usize,
    scale_cols: usize,
    block_rows: usize,
    block_cols: usize,
}

#[derive(Serialize)]
struct ReferenceLinearReport {
    schema_version: &'static str,
    artifact: String,
    artifact_schema_version: String,
    artifact_content_sha256: String,
    tensor: TensorReport,
    input_f32_le_sha256: String,
    oracle_output_f32_le_sha256: String,
    runtime_output_f32_le_sha256: String,
    runtime_execution_path: &'static str,
    device: DeviceReport,
    correctness: Sq8CorrectnessReport,
    oracle_preview: Vec<f32>,
    runtime_preview: Vec<f32>,
}

fn take_value(values: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    values
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut artifact = None;
    let mut tensor = None;
    let mut device_index = None;
    let mut report = None;
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--artifact" => artifact = Some(PathBuf::from(take_value(&mut values, &flag)?)),
            "--tensor" => tensor = Some(take_value(&mut values, &flag)?),
            "--device-index" => {
                let value = take_value(&mut values, &flag)?;
                device_index = Some(
                    value
                        .parse::<u32>()
                        .map_err(|err| format!("invalid --device-index {value:?}: {err}"))?,
                );
            }
            "--report" => report = Some(PathBuf::from(take_value(&mut values, &flag)?)),
            "--help" | "-h" => {
                println!(
                    "usage: sq8_reference_linear --artifact DIR --tensor NAME --device-index N [--report FILE]"
                );
                return Ok(None);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Some(Args {
        artifact: artifact.ok_or_else(|| "--artifact is required".to_string())?,
        tensor: tensor.ok_or_else(|| "--tensor is required".to_string())?,
        device_index: device_index.ok_or_else(|| "--device-index is required".to_string())?,
        report,
    }))
}

fn main() -> Result<(), String> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let artifact = read_sq8_canonical_artifact(&args.artifact)?;
    let pair = artifact.tensor_pair(&args.tensor)?;
    let cols = usize::try_from(pair.shape[1])
        .map_err(|_| format!("SQ8 tensor {} cols do not fit usize", pair.name))?;
    let input = sq8_reference_activation(cols);
    let oracle = run_sq8_reference_projection(&artifact, &args.tensor, &input)?;

    let mut context = RuntimeContext::create(args.device_index)?;
    let device = context.device_info()?;
    let mut stream = context.create_stream()?;
    let resident = load_sq8_canonical_resident_tensor(
        &mut context,
        &mut stream,
        &artifact,
        &args.tensor,
        SQ8_CANONICAL_UPLOAD_CHUNK_BYTES,
    )?;
    let runtime =
        run_sq8_canonical_resident_projection_f32(&mut context, &mut stream, &resident, &input)?;
    let (report_execution_path, fallback_state, runtime_execution_path) =
        match runtime.execution_path {
            SqFp8ExecutionPath::CpuReference => (
                Sq8CorrectnessExecutionPath::RuntimeCpuReference,
                Sq8CorrectnessFallbackState::NotApplicable,
                "cpu_reference",
            ),
            SqFp8ExecutionPath::HipKernel => (
                Sq8CorrectnessExecutionPath::RuntimeHipKernel,
                Sq8CorrectnessFallbackState::NotUsed,
                "hip_kernel",
            ),
        };
    let correctness = build_sq8_correctness_report(
        &artifact,
        &args.tensor,
        &input,
        &oracle.output,
        &runtime.output,
        report_execution_path,
        fallback_state,
    )?;
    let result = ReferenceLinearReport {
        schema_version: "sq8-reference-correctness-result-v0.1",
        artifact: args.artifact.display().to_string(),
        artifact_schema_version: artifact.manifest().schema_version.clone(),
        artifact_content_sha256: artifact.manifest().integrity.content_sha256.clone(),
        tensor: TensorReport {
            name: resident.tensor_name.clone(),
            rows: resident.rows,
            cols: resident.cols,
            scale_rows: resident.scale_rows,
            scale_cols: resident.scale_cols,
            block_rows: resident.block_rows,
            block_cols: resident.block_cols,
        },
        input_f32_le_sha256: oracle.input_f32_le_sha256.clone(),
        oracle_output_f32_le_sha256: sq8_f32_le_sha256(&oracle.output)?,
        runtime_output_f32_le_sha256: sq8_f32_le_sha256(&runtime.output)?,
        runtime_execution_path,
        device: DeviceReport {
            requested_index: args.device_index,
            runtime_device_id: device.device_id,
            backend: device.backend,
            name: device.name,
            compute_major: device.compute_major,
            compute_minor: device.compute_minor,
            gcn_arch_name: device.gcn_arch_name,
        },
        correctness,
        oracle_preview: oracle.output.iter().copied().take(8).collect(),
        runtime_preview: runtime.output.iter().copied().take(8).collect(),
    };
    let json = serde_json::to_string_pretty(&result)
        .map_err(|err| format!("failed to serialize SQ8 reference report: {err}"))?;
    if let Some(path) = args.report {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create report directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        std::fs::write(&path, format!("{json}\n"))
            .map_err(|err| format!("failed to write report {}: {err}", path.display()))?;
    }
    println!("{json}");
    if !result.correctness.passed {
        return Err("SQ8 reference correctness gate failed".to_string());
    }
    Ok(())
}
