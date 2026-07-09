// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;
use ullm_engine::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq_reference::{sq8_f32_le_sha256, sq8_reference_activation};
use ullm_engine::sq_runtime::{
    SQ8_CANONICAL_UPLOAD_CHUNK_BYTES, load_sq8_canonical_resident_tensor,
};
use ullm_runtime_sys::{RuntimeContext, SqFp8ExecutionPath, sq_fp8_matvec_block2d_batch_f32};

#[derive(Debug)]
struct Args {
    artifact_dir: PathBuf,
    tensor: String,
    device_index: u32,
    m: usize,
    warmups: usize,
    repeats: usize,
}

#[derive(Serialize)]
struct DeviceReport {
    requested_index: u32,
    runtime_device_id: i32,
    backend: String,
    name: String,
    gcn_arch_name: String,
}

#[derive(Serialize)]
struct TimingReport {
    source: &'static str,
    warmups: usize,
    repeats: usize,
    p50_ms: f64,
    p95_ms: f64,
    aggregate_tflops_p50: f64,
}

#[derive(Serialize)]
struct BenchmarkReport {
    schema_version: &'static str,
    profile: &'static str,
    artifact_dir: String,
    artifact_content_sha256: String,
    tensor: String,
    m: usize,
    n: usize,
    k: usize,
    input_f32_le_sha256: String,
    output_f32_le_sha256: String,
    output_nonfinite: usize,
    execution_path: &'static str,
    fallback_state: &'static str,
    device: DeviceReport,
    timing: TimingReport,
    passed: bool,
}

fn take_value(values: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    values
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_positive(value: String, flag: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag} {value:?}: {err}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut artifact_dir = None;
    let mut tensor = None;
    let mut device_index = None;
    let mut m = None;
    let mut warmups = 5;
    let mut repeats = 20;
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--artifact-dir" => {
                artifact_dir = Some(PathBuf::from(take_value(&mut values, &flag)?));
            }
            "--tensor" => tensor = Some(take_value(&mut values, &flag)?),
            "--device-index" => {
                let value = take_value(&mut values, &flag)?;
                device_index = Some(
                    value
                        .parse::<u32>()
                        .map_err(|err| format!("invalid --device-index {value:?}: {err}"))?,
                );
            }
            "--m" => m = Some(parse_positive(take_value(&mut values, &flag)?, &flag)?),
            "--warmups" => {
                let value = take_value(&mut values, &flag)?;
                warmups = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --warmups {value:?}: {err}"))?;
                if warmups > 100_000 {
                    return Err("--warmups must not exceed 100000".to_string());
                }
            }
            "--repeats" => {
                repeats = parse_positive(take_value(&mut values, &flag)?, &flag)?;
                if repeats > 100_000 {
                    return Err("--repeats must not exceed 100000".to_string());
                }
            }
            "--help" | "-h" => {
                println!(
                    "usage: sq8_reference_batch_bench --artifact-dir DIR --tensor NAME --device-index N --m M [--warmups N] [--repeats N]"
                );
                return Ok(None);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Some(Args {
        artifact_dir: artifact_dir.ok_or_else(|| "--artifact-dir is required".to_string())?,
        tensor: tensor.ok_or_else(|| "--tensor is required".to_string())?,
        device_index: device_index.ok_or_else(|| "--device-index is required".to_string())?,
        m: m.ok_or_else(|| "--m is required".to_string())?,
        warmups,
        repeats,
    }))
}

fn percentile(mut samples: Vec<f64>, probability: f64) -> Result<f64, String> {
    if samples.is_empty() {
        return Err("cannot calculate percentile of empty samples".to_string());
    }
    samples.sort_by(f64::total_cmp);
    let position = probability * (samples.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Ok(samples[lower] * (1.0 - fraction) + samples[upper] * fraction)
}

fn execution_path_label(path: SqFp8ExecutionPath) -> &'static str {
    match path {
        SqFp8ExecutionPath::CpuReference => "cpu_reference",
        SqFp8ExecutionPath::HipKernel => "hip_kernel",
    }
}

fn main() -> Result<(), String> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let artifact = read_sq8_canonical_artifact(&args.artifact_dir)?;
    let pair = artifact.tensor_pair(&args.tensor)?;
    let n = usize::try_from(pair.shape[0])
        .map_err(|_| format!("SQ8 tensor {} N does not fit usize", pair.name))?;
    let k = usize::try_from(pair.shape[1])
        .map_err(|_| format!("SQ8 tensor {} K does not fit usize", pair.name))?;
    let input_elements = args
        .m
        .checked_mul(k)
        .ok_or_else(|| format!("activation shape [{},{}] overflows usize", args.m, k))?;
    let output_elements = args
        .m
        .checked_mul(n)
        .ok_or_else(|| format!("output shape [{},{}] overflows usize", args.m, n))?;
    let input = sq8_reference_activation(input_elements);
    let input_bytes = encode_f32_to_bytes(&input);

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
    let mut input_buffer = context.alloc_buffer(input_bytes.len())?;
    let output_bytes_len = output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "reference output byte size overflows usize".to_string())?;
    let mut output_buffer = context.alloc_buffer(output_bytes_len)?;
    input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream))?;
    stream.synchronize()?;

    let mut observed_path = None;
    let mut run_once = || -> Result<SqFp8ExecutionPath, String> {
        let path = sq_fp8_matvec_block2d_batch_f32(
            &resident.payload_buffer,
            &resident.scale_buffer,
            &input_buffer,
            resident.rows,
            resident.cols,
            resident.block_rows,
            resident.block_cols,
            args.m,
            &mut output_buffer,
            Some(&mut stream),
        )?;
        stream.synchronize()?;
        if observed_path.is_some_and(|observed| observed != path) {
            return Err("SQ8 reference execution path changed between repeats".to_string());
        }
        observed_path = Some(path);
        Ok(path)
    };
    for _ in 0..args.warmups {
        run_once()?;
    }
    let mut samples_ms = Vec::with_capacity(args.repeats);
    for _ in 0..args.repeats {
        let start = Instant::now();
        run_once()?;
        samples_ms.push(start.elapsed().as_secs_f64() * 1.0e3);
    }
    let execution_path = observed_path.ok_or_else(|| "reference path did not run".to_string())?;

    let mut output_bytes = vec![0_u8; output_bytes_len];
    output_buffer.copy_to_host(0, &mut output_bytes, Some(&mut stream))?;
    stream.synchronize()?;
    let output = decode_f32_le_values(&output_bytes);
    let output_nonfinite = output.iter().filter(|value| !value.is_finite()).count();
    let p50_ms = percentile(samples_ms.clone(), 0.50)?;
    let p95_ms = percentile(samples_ms, 0.95)?;
    let operations = 2.0 * args.m as f64 * n as f64 * k as f64;
    let aggregate_tflops_p50 = operations / (p50_ms * 1.0e9);
    let passed = execution_path == SqFp8ExecutionPath::HipKernel
        && output_nonfinite == 0
        && device.backend == "hip";
    let result = BenchmarkReport {
        schema_version: "sq8-reference-batch-benchmark-v0.1",
        profile: "reference_w8a16",
        artifact_dir: artifact.artifact_dir().display().to_string(),
        artifact_content_sha256: artifact.manifest().integrity.content_sha256.clone(),
        tensor: pair.name.clone(),
        m: args.m,
        n,
        k,
        input_f32_le_sha256: sq8_f32_le_sha256(&input)?,
        output_f32_le_sha256: sq8_f32_le_sha256(&output)?,
        output_nonfinite,
        execution_path: execution_path_label(execution_path),
        fallback_state: if execution_path == SqFp8ExecutionPath::HipKernel {
            "not_used"
        } else {
            "used"
        },
        device: DeviceReport {
            requested_index: args.device_index,
            runtime_device_id: device.device_id,
            backend: device.backend,
            name: device.name,
            gcn_arch_name: device.gcn_arch_name,
        },
        timing: TimingReport {
            source: "host_monotonic_launch_plus_sync",
            warmups: args.warmups,
            repeats: args.repeats,
            p50_ms,
            p95_ms,
            aggregate_tflops_p50,
        },
        passed,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&result)
            .map_err(|err| format!("failed to serialize SQ8 reference benchmark: {err}"))?
    );
    if !passed {
        return Err("SQ8 reference batch benchmark gate failed".to_string());
    }
    Ok(())
}
