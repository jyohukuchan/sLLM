// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

use ullm_runtime_sys::{
    RuntimeContext, Sq8CkQuantizedActivation, device_count, device_info,
    sq8_ck_projection_buffer_bytes, sq8_ck_projection_f32,
};

const RELATIVE_L2_LIMIT: f64 = 5.0e-3;
const COSINE_LIMIT: f64 = 0.9999;

fn main() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let fixture_dir = PathBuf::from(args.next().ok_or_else(usage)?);
    let weight_path = PathBuf::from(args.next().ok_or_else(usage)?);
    let weight_scale_bf16_path = PathBuf::from(args.next().ok_or_else(usage)?);
    let m = parse_usize(args.next(), "M")?;
    let n = parse_usize(args.next(), "N")?;
    let k = parse_usize(args.next(), "K")?;
    if args.next().is_some() {
        return Err(usage());
    }

    let activation_bytes = read_exact(
        &fixture_dir.join("activation.f32le"),
        checked_bytes(m, k, std::mem::size_of::<f32>(), "activation")?,
    )?;
    let expected_quantized = read_exact(
        &fixture_dir.join("activation.f8"),
        checked_bytes(m, k, 1, "quantized activation")?,
    )?;
    let expected_scales = read_exact(
        &fixture_dir.join("activation_scales.f32le"),
        checked_bytes(m, k / 128, std::mem::size_of::<f32>(), "activation scales")?,
    )?;
    let oracle = decode_f32(read_exact(
        &fixture_dir.join("oracle_output.f32le"),
        checked_bytes(m, n, std::mem::size_of::<f32>(), "oracle")?,
    )?)?;
    let weight = read_exact(&weight_path, checked_bytes(n, k, 1, "weight")?)?;
    let weight_scale_bf16 = read_exact(
        &weight_scale_bf16_path,
        checked_bytes(
            n / 128,
            k / 128,
            std::mem::size_of::<u16>(),
            "weight scales",
        )?,
    )?;
    let weight_scales = decode_bf16_to_f32_bytes(&weight_scale_bf16);

    let hip_index = (1..device_count()?)
        .find(|index| device_info(*index).is_ok_and(|info| info.backend == "hip"))
        .ok_or_else(|| "one isolated HIP device was not found".to_string())?;
    let mut context = RuntimeContext::create(hip_index)?;
    let info = context.device_info()?;
    if info.compute_major != 12 || info.compute_minor != 0 {
        return Err(format!(
            "fixture requires compute 12.0; selected {}.{}",
            info.compute_major, info.compute_minor
        ));
    }
    let mut stream = context.create_stream()?;

    let mut input = context.alloc_buffer(activation_bytes.len())?;
    input.copy_from_host(0, &activation_bytes, Some(&mut stream))?;
    let mut quantized = Sq8CkQuantizedActivation::allocate(&mut context, m, k)?;
    quantized.quantize_f32(&input, Some(&mut stream))?;
    let mut device_weight = context.alloc_buffer(weight.len())?;
    device_weight.copy_from_host(0, &weight, Some(&mut stream))?;
    let mut device_weight_scales = context.alloc_buffer(weight_scales.len())?;
    device_weight_scales.copy_from_host(0, &weight_scales, Some(&mut stream))?;
    stream.synchronize()?;
    drop(weight);
    drop(weight_scales);

    let (workspace_bytes, output_bytes) = sq8_ck_projection_buffer_bytes(m, n)?;
    let mut workspace = context.alloc_buffer(workspace_bytes)?;
    let mut output = context.alloc_buffer(output_bytes)?;
    let implementation = sq8_ck_projection_f32(
        &quantized,
        &device_weight,
        &device_weight_scales,
        n,
        &mut workspace,
        &mut output,
        Some(&mut stream),
    )?;
    stream.synchronize()?;

    let mut actual_quantized = vec![0_u8; quantized.quantized_bytes()];
    let mut actual_scales = vec![0_u8; quantized.scale_bytes()];
    let mut actual_output_bytes = vec![0_u8; output_bytes];
    quantized
        .quantized_buffer()
        .copy_to_host(0, &mut actual_quantized, Some(&mut stream))?;
    quantized
        .scale_buffer()
        .copy_to_host(0, &mut actual_scales, Some(&mut stream))?;
    output.copy_to_host(0, &mut actual_output_bytes, Some(&mut stream))?;
    stream.synchronize()?;

    if actual_quantized != expected_quantized {
        return Err("GPU quantized activation is not byte-exact to the fixture".to_string());
    }
    if actual_scales != expected_scales {
        return Err("GPU activation scales are not bit-exact to the fixture".to_string());
    }
    let actual_output = decode_f32(actual_output_bytes)?;
    let metrics = metrics(&actual_output, &oracle)?;
    if metrics.nonfinite != 0
        || metrics.relative_l2 > RELATIVE_L2_LIMIT
        || metrics.cosine < COSINE_LIMIT
    {
        return Err(format!(
            "fixture numerical gate failed: nonfinite={} rel_l2={} cosine={}",
            metrics.nonfinite, metrics.relative_l2, metrics.cosine
        ));
    }

    println!(
        "passed=true m={m} n={n} k={k} implementation={implementation:?} \
         quantized_byte_exact=true scale_bit_exact=true nonfinite={} max_abs={:.9} \
         relative_l2={:.9} cosine={:.12}",
        metrics.nonfinite, metrics.max_abs, metrics.relative_l2, metrics.cosine
    );
    Ok(())
}

fn usage() -> String {
    "usage: sq8_ck_fixture FIXTURE_DIR WEIGHT_F8 WEIGHT_SCALE_BF16 M N K".to_string()
}

fn parse_usize(value: Option<std::ffi::OsString>, label: &str) -> Result<usize, String> {
    value
        .ok_or_else(usage)?
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|error| format!("invalid {label}: {error}"))
}

fn checked_bytes(rows: usize, cols: usize, width: usize, label: &str) -> Result<usize, String> {
    rows.checked_mul(cols)
        .and_then(|elements| elements.checked_mul(width))
        .ok_or_else(|| format!("{label} byte size overflows"))
}

fn read_exact(path: &Path, expected: usize) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes; expected {expected}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn decode_bf16_to_f32_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len() * 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        output.extend_from_slice(&f32::from_bits(u32::from(bits) << 16).to_le_bytes());
    }
    output
}

fn decode_f32(bytes: Vec<u8>) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err("f32 byte length is not aligned".to_string());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

struct Metrics {
    nonfinite: usize,
    max_abs: f64,
    relative_l2: f64,
    cosine: f64,
}

fn metrics(actual: &[f32], expected: &[f32]) -> Result<Metrics, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("output and oracle lengths differ or are empty".to_string());
    }
    let mut nonfinite = 0_usize;
    let mut max_abs = 0.0_f64;
    let mut squared_error = 0.0_f64;
    let mut squared_expected = 0.0_f64;
    let mut squared_actual = 0.0_f64;
    let mut dot = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() {
            nonfinite += 1;
            continue;
        }
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let error = actual - expected;
        max_abs = max_abs.max(error.abs());
        squared_error += error * error;
        squared_expected += expected * expected;
        squared_actual += actual * actual;
        dot += actual * expected;
    }
    let relative_l2 = (squared_error / squared_expected).sqrt();
    let cosine = dot / (squared_actual.sqrt() * squared_expected.sqrt());
    Ok(Metrics {
        nonfinite,
        max_abs,
        relative_l2,
        cosine,
    })
}
