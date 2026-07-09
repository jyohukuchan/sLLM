// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use ullm_engine::loader::read_named_passthrough_f32;
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq_optimized_reference::quantize_sq8_dynamic_activation;
use ullm_engine::sq_reference::{
    Sq8CorrectnessMetrics, compare_sq8_correctness, sq8_f32_le_sha256,
};
use ullm_engine::sq8_layer_oracle::{
    QWEN3_14B_HIDDEN_SIZE, Sq8LayerNormWeights, Sq8LayerOracleOptions, Sq8LayerProjectionNames,
    run_qwen3_14b_sq8_layer_oracle_with_options,
};
use ullm_engine::sq8_layer_runtime::{
    QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS, QWEN3_14B_SQ8_LAYER_PROJECTIONS,
    QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV, Qwen3Sq8LayerConfig, Qwen3Sq8LayerNormValues,
    Qwen3Sq8LayerWorkspace, Sq8LayerExecutionProfile, Sq8LayerExecutionReport,
    Sq8LayerProjectionExecution, Sq8LayerQuantizedActivationTrace, Sq8LayerRuntimeTrace,
    load_qwen3_14b_sq8_layer_weights, qwen3_sq8_layer_tensor_names,
};
use ullm_runtime_sys::{
    DeviceInfo, RuntimeBuffer, RuntimeContext, RuntimeStream, Sq8CkImplementation, device_count,
    device_info,
};

const SCHEMA_VERSION: &str = "ullm.sq8.layer.v1";
const UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
struct Options {
    artifact: PathBuf,
    package: PathBuf,
    input: PathBuf,
    output: PathBuf,
    layer: usize,
    m: usize,
    warmups: usize,
    repeats: usize,
}

#[derive(Debug, Serialize)]
struct TimingSummary {
    samples_ms: Vec<f64>,
    p50_ms: f64,
    p95_ms: f64,
}

#[derive(Debug, Serialize)]
struct TensorCheck {
    metrics: Sq8CorrectnessMetrics,
    max_relative_l2: f64,
    min_cosine: f64,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ActivationCheck {
    m: usize,
    k: usize,
    encoded_byte_exact: bool,
    scale_bit_exact: bool,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ProjectionExecutions {
    q: String,
    k: String,
    v: String,
    o: String,
    gate: String,
    up: String,
    down: String,
}

#[derive(Debug, Serialize)]
struct OutputHealth {
    elements: usize,
    nonfinite: usize,
    minimum: f32,
    maximum: f32,
    max_abs: f32,
    f32_le_sha256: String,
}

#[derive(Debug, Serialize)]
struct LayerResult {
    schema_version: &'static str,
    passed: bool,
    artifact_content_sha256: String,
    layer_index: usize,
    sequence_len: usize,
    position_offset: usize,
    input_f32_le_sha256: String,
    device: DeviceResult,
    contracts: Contracts,
    optimized_executions: ProjectionExecutions,
    reference_executions: ProjectionExecutions,
    activation_checks: BTreeMap<String, ActivationCheck>,
    tensor_checks: BTreeMap<String, TensorCheck>,
    reference_vs_optimized_oracle_final_check: TensorCheck,
    optimized_output_health: OutputHealth,
    oracle_elapsed_ms: f64,
    optimized_timing: TimingSummary,
    reference_timing: TimingSummary,
    optimized_speedup: f64,
    oracle_trace: ullm_engine::sq8_layer_oracle::Sq8LayerOracleTrace,
}

#[derive(Debug, Serialize)]
struct DeviceResult {
    runtime_index: u32,
    backend_device_id: i32,
    backend: String,
    name: String,
    compute_major: i32,
    compute_minor: i32,
}

#[derive(Debug, Serialize)]
struct Contracts {
    optimized_profile: &'static str,
    reference_profile: &'static str,
    projection_output: &'static str,
    activation_quantizations: usize,
    projection_calls: usize,
    fallback_used: bool,
    timed_path_host_staging: bool,
    required_hip_kernel_env: Vec<&'static str>,
    warmups: usize,
    repeats: usize,
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    let config = Qwen3Sq8LayerConfig::qwen3_14b(options.m, 0)?;
    let input = read_input(&options.input, options.m)?;
    let input_hash = sq8_f32_le_sha256(&input)?;
    let artifact = read_sq8_canonical_artifact(&options.artifact)?;
    let names = qwen3_sq8_layer_tensor_names(options.layer);
    let norms = read_norms(&options.package, options.layer)?;

    let oracle_started = Instant::now();
    let oracle = run_qwen3_14b_sq8_layer_oracle_with_options(
        &artifact,
        projection_names(&names),
        norm_refs(&norms),
        &input,
        options.m,
        Sq8LayerOracleOptions::default(),
    )?;
    let oracle_elapsed_ms = oracle_started.elapsed().as_secs_f64() * 1000.0;

    let (runtime_index, device) = isolated_hip_device()?;
    let mut context = RuntimeContext::create(runtime_index)?;
    let mut stream = context.create_stream()?;
    let weights = load_qwen3_14b_sq8_layer_weights(
        &mut context,
        &mut stream,
        &artifact,
        options.layer,
        &norms,
        UPLOAD_CHUNK_BYTES,
    )?;
    let mut workspace = Qwen3Sq8LayerWorkspace::allocate(&mut context, config)?;
    let input_bytes = f32_bytes(&input);
    let mut input_buffer = context.alloc_buffer(input_bytes.len())?;
    input_buffer.copy_from_host(0, &input_bytes, Some(&mut stream))?;
    stream.synchronize()?;
    drop(input_bytes);

    for _ in 0..options.warmups {
        run_and_sync(
            &mut workspace,
            &weights,
            &input_buffer,
            Sq8LayerExecutionProfile::ReferenceW8a16Block2d,
            &mut stream,
        )?;
        run_and_sync(
            &mut workspace,
            &weights,
            &input_buffer,
            Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
            &mut stream,
        )?;
    }

    let mut optimized_samples = Vec::with_capacity(options.repeats);
    let mut reference_samples = Vec::with_capacity(options.repeats);
    let mut optimized_report = None;
    let mut reference_report = None;
    for repeat in 0..options.repeats {
        let profiles = if repeat.is_multiple_of(2) {
            [
                Sq8LayerExecutionProfile::ReferenceW8a16Block2d,
                Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
            ]
        } else {
            [
                Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
                Sq8LayerExecutionProfile::ReferenceW8a16Block2d,
            ]
        };
        for profile in profiles {
            let started = Instant::now();
            let report = run_and_sync(
                &mut workspace,
                &weights,
                &input_buffer,
                profile,
                &mut stream,
            )?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            match profile {
                Sq8LayerExecutionProfile::ReferenceW8a16Block2d => {
                    reference_samples.push(elapsed_ms);
                    verify_stable_report(&mut reference_report, &report)?;
                }
                Sq8LayerExecutionProfile::Rdna4W8a8BlockCk => {
                    optimized_samples.push(elapsed_ms);
                    verify_stable_report(&mut optimized_report, &report)?;
                }
            }
        }
    }
    let optimized_report =
        optimized_report.ok_or_else(|| "missing optimized report".to_string())?;
    let reference_report =
        reference_report.ok_or_else(|| "missing reference report".to_string())?;
    validate_execution_reports(&optimized_report, &reference_report, options.m)?;

    run_and_sync(
        &mut workspace,
        &weights,
        &input_buffer,
        Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
        &mut stream,
    )?;
    let runtime_trace = workspace.read_trace(&mut stream)?;
    let optimized_output_health = output_health(&runtime_trace.output)?;
    let tensor_checks = compare_intermediates(&oracle, &runtime_trace)?;
    let activation_checks = validate_activations(options.m, &runtime_trace)?;

    run_and_sync(
        &mut workspace,
        &weights,
        &input_buffer,
        Sq8LayerExecutionProfile::ReferenceW8a16Block2d,
        &mut stream,
    )?;
    let reference_output = workspace.read_trace(&mut stream)?.output;
    let reference_vs_optimized_oracle_final_check =
        tensor_check(&oracle.output, &reference_output, 2.0e-2, 0.999)?;

    let optimized_timing = timing_summary(optimized_samples)?;
    let reference_timing = timing_summary(reference_samples)?;
    let optimized_speedup = reference_timing.p50_ms / optimized_timing.p50_ms;
    let checks_passed = tensor_checks.values().all(|check| check.passed)
        && activation_checks.values().all(|check| check.passed);
    let passed = checks_passed
        && optimized_report.all_ck()
        && reference_report.all_reference_hip()
        && !optimized_report.fallback_used
        && !reference_report.fallback_used
        && reference_vs_optimized_oracle_final_check
            .metrics
            .nonfinite_count
            == 0
        && optimized_speedup > 1.0
        && optimized_output_health.nonfinite == 0;

    let result = LayerResult {
        schema_version: SCHEMA_VERSION,
        passed,
        artifact_content_sha256: artifact.manifest().integrity.content_sha256.clone(),
        layer_index: options.layer,
        sequence_len: options.m,
        position_offset: 0,
        input_f32_le_sha256: input_hash,
        device: DeviceResult {
            runtime_index,
            backend_device_id: device.device_id,
            backend: device.backend,
            name: device.name,
            compute_major: device.compute_major,
            compute_minor: device.compute_minor,
        },
        contracts: Contracts {
            optimized_profile: "rdna4_w8a8_block_ck",
            reference_profile: "reference_w8a16_block2d",
            projection_output: "bf16_rne_then_f32",
            activation_quantizations: optimized_report.activation_quantizations,
            projection_calls: optimized_report.projection_calls,
            fallback_used: optimized_report.fallback_used || reference_report.fallback_used,
            timed_path_host_staging: false,
            required_hip_kernel_env: QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV.to_vec(),
            warmups: options.warmups,
            repeats: options.repeats,
        },
        optimized_executions: projection_executions(&optimized_report),
        reference_executions: projection_executions(&reference_report),
        activation_checks,
        tensor_checks,
        reference_vs_optimized_oracle_final_check,
        optimized_output_health,
        oracle_elapsed_ms,
        optimized_timing,
        reference_timing,
        optimized_speedup,
        oracle_trace: oracle.trace,
    };
    write_json_no_clobber(&options.output, &result)?;
    if !result.passed {
        return Err(format!(
            "SQ8 layer gate failed; evidence was written to {}: speedup={optimized_speedup:.6}",
            options.output.display()
        ));
    }
    println!(
        "passed=true output={} optimized_p50_ms={:.6} reference_p50_ms={:.6} speedup={:.6} final_rel_l2={:.9} final_cosine={:.12}",
        options.output.display(),
        result.optimized_timing.p50_ms,
        result.reference_timing.p50_ms,
        result.optimized_speedup,
        result.tensor_checks["output"].metrics.relative_l2,
        result.tensor_checks["output"].metrics.cosine_similarity,
    );
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut args = std::env::args_os().skip(1);
    let artifact = PathBuf::from(args.next().ok_or_else(usage)?);
    let package = PathBuf::from(args.next().ok_or_else(usage)?);
    let input = PathBuf::from(args.next().ok_or_else(usage)?);
    let output = PathBuf::from(args.next().ok_or_else(usage)?);
    let layer = parse_optional_usize(args.next(), 0, "layer")?;
    let m = parse_optional_usize(args.next(), 8, "M")?;
    let warmups = parse_optional_usize(args.next(), 3, "warmups")?;
    let repeats = parse_optional_usize(args.next(), 10, "repeats")?;
    if args.next().is_some() {
        return Err(usage());
    }
    if warmups == 0 || repeats < 3 || repeats > 1000 {
        return Err("warmups must be positive and repeats must be in 3..=1000".to_string());
    }
    if output.exists() {
        return Err(format!("output already exists: {}", output.display()));
    }
    Ok(Options {
        artifact,
        package,
        input,
        output,
        layer,
        m,
        warmups,
        repeats,
    })
}

fn usage() -> String {
    "usage: sq8_ck_layer ARTIFACT_DIR THIN_PACKAGE INPUT_F32LE OUTPUT_JSON [LAYER=0] [M=8] [WARMUPS=3] [REPEATS=10]".to_string()
}

fn parse_optional_usize(
    value: Option<std::ffi::OsString>,
    default: usize,
    label: &str,
) -> Result<usize, String> {
    match value {
        Some(value) => value
            .to_string_lossy()
            .parse::<usize>()
            .map_err(|err| format!("invalid {label}: {err}")),
        None => Ok(default),
    }
}

fn read_input(path: &Path, m: usize) -> Result<Vec<f32>, String> {
    let elements = m
        .checked_mul(QWEN3_14B_HIDDEN_SIZE)
        .ok_or_else(|| "input element count overflows".to_string())?;
    let expected = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "input byte size overflows".to_string())?;
    let bytes = std::fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes; expected {expected}",
            path.display(),
            bytes.len()
        ));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "input contains non-finite value {value} at {index}"
        ));
    }
    Ok(values)
}

fn read_norms(package: &Path, layer: usize) -> Result<Qwen3Sq8LayerNormValues, String> {
    let prefix = format!("model.layers.{layer}");
    Ok(Qwen3Sq8LayerNormValues {
        input: read_named_passthrough_f32(
            package,
            &format!("{prefix}.input_layernorm.weight"),
            UPLOAD_CHUNK_BYTES,
        )?
        .values,
        post_attention: read_named_passthrough_f32(
            package,
            &format!("{prefix}.post_attention_layernorm.weight"),
            UPLOAD_CHUNK_BYTES,
        )?
        .values,
        q: read_named_passthrough_f32(
            package,
            &format!("{prefix}.self_attn.q_norm.weight"),
            UPLOAD_CHUNK_BYTES,
        )?
        .values,
        k: read_named_passthrough_f32(
            package,
            &format!("{prefix}.self_attn.k_norm.weight"),
            UPLOAD_CHUNK_BYTES,
        )?
        .values,
    })
}

fn projection_names<'a>(names: &'a [String; 7]) -> Sq8LayerProjectionNames<'a> {
    Sq8LayerProjectionNames {
        q_proj: &names[0],
        k_proj: &names[1],
        v_proj: &names[2],
        o_proj: &names[3],
        gate_proj: &names[4],
        up_proj: &names[5],
        down_proj: &names[6],
    }
}

fn norm_refs(norms: &Qwen3Sq8LayerNormValues) -> Sq8LayerNormWeights<'_> {
    Sq8LayerNormWeights {
        input: &norms.input,
        post_attention: &norms.post_attention,
        q: &norms.q,
        k: &norms.k,
    }
}

fn isolated_hip_device() -> Result<(u32, DeviceInfo), String> {
    let devices = (1..device_count()?)
        .filter_map(|index| device_info(index).ok().map(|info| (index, info)))
        .filter(|(_, info)| info.backend == "hip")
        .collect::<Vec<_>>();
    if devices.len() != 1 {
        return Err(format!(
            "SQ8 CK layer requires exactly one runtime HIP device, found {}",
            devices.len()
        ));
    }
    let (index, info) = devices.into_iter().next().expect("one device");
    if info.compute_major != 12 || info.compute_minor != 0 {
        return Err(format!(
            "SQ8 CK layer requires compute 12.0, got {}.{}",
            info.compute_major, info.compute_minor
        ));
    }
    Ok((index, info))
}

fn run_and_sync(
    workspace: &mut Qwen3Sq8LayerWorkspace,
    weights: &ullm_engine::sq8_layer_runtime::Qwen3Sq8LayerWeights,
    input: &RuntimeBuffer,
    profile: Sq8LayerExecutionProfile,
    stream: &mut RuntimeStream,
) -> Result<Sq8LayerExecutionReport, String> {
    workspace.run_synchronized(weights, input, profile, stream)
}

fn verify_stable_report(
    expected: &mut Option<Sq8LayerExecutionReport>,
    actual: &Sq8LayerExecutionReport,
) -> Result<(), String> {
    match expected {
        Some(expected) if expected != actual => {
            Err("SQ8 layer execution report changed between repeats".to_string())
        }
        Some(_) => Ok(()),
        None => {
            *expected = Some(actual.clone());
            Ok(())
        }
    }
}

fn validate_execution_reports(
    optimized: &Sq8LayerExecutionReport,
    reference: &Sq8LayerExecutionReport,
    m: usize,
) -> Result<(), String> {
    if optimized.activation_quantizations != QWEN3_14B_SQ8_LAYER_ACTIVATION_QUANTIZATIONS
        || optimized.projection_calls != QWEN3_14B_SQ8_LAYER_PROJECTIONS
        || optimized.fallback_used
        || !optimized.all_ck()
    {
        return Err(format!("invalid optimized execution report: {optimized:?}"));
    }
    if reference.activation_quantizations != 0
        || reference.projection_calls != QWEN3_14B_SQ8_LAYER_PROJECTIONS
        || reference.fallback_used
        || !reference.all_reference_hip()
    {
        return Err(format!("invalid reference execution report: {reference:?}"));
    }
    let default128 =
        Sq8LayerProjectionExecution::Ck(Sq8CkImplementation::MemV1DefaultTile16x128x128);
    let gate_up = Sq8LayerProjectionExecution::Ck(if m == 128 {
        Sq8CkImplementation::MemV1DefaultTile16x256x128
    } else {
        Sq8CkImplementation::MemV1KPaddingTile16x128x256
    });
    let down = Sq8LayerProjectionExecution::Ck(if m == 128 {
        Sq8CkImplementation::MemV1DefaultTile16x128x128
    } else {
        Sq8CkImplementation::MemV1DefaultTile16x128x256
    });
    if [optimized.q, optimized.k, optimized.v, optimized.o]
        .into_iter()
        .any(|execution| execution != default128)
        || optimized.gate != gate_up
        || optimized.up != gate_up
        || optimized.down != down
    {
        return Err(format!(
            "optimized dispatch does not match the measured table: {optimized:?}"
        ));
    }
    Ok(())
}

fn compare_intermediates(
    oracle: &ullm_engine::sq8_layer_oracle::Sq8LayerOracleOutput,
    actual: &Sq8LayerRuntimeTrace,
) -> Result<BTreeMap<String, TensorCheck>, String> {
    let expected = &oracle.intermediates;
    let mut checks = BTreeMap::new();
    insert_check(
        &mut checks,
        "input_norm",
        &expected.input_norm,
        &actual.input_normed,
        1.0e-4,
        0.999_999,
    )?;
    insert_check(
        &mut checks,
        "q_projected",
        &expected.q_projected,
        &actual.q_projected,
        5.0e-3,
        0.9999,
    )?;
    insert_check(
        &mut checks,
        "k_projected",
        &expected.k_projected,
        &actual.k_projected,
        5.0e-3,
        0.9999,
    )?;
    insert_check(
        &mut checks,
        "v_projected",
        &expected.v_projected,
        &actual.v_projected,
        5.0e-3,
        0.9999,
    )?;
    insert_check(
        &mut checks,
        "q_norm",
        &expected.q_norm,
        &actual.q_normed,
        7.5e-3,
        0.9998,
    )?;
    insert_check(
        &mut checks,
        "k_norm",
        &expected.k_norm,
        &actual.k_normed,
        7.5e-3,
        0.9998,
    )?;
    insert_check(
        &mut checks,
        "q_rope",
        &expected.q_rope,
        &actual.q_rope,
        7.5e-3,
        0.9998,
    )?;
    insert_check(
        &mut checks,
        "k_rope",
        &expected.k_rope,
        &actual.k_rope,
        7.5e-3,
        0.9998,
    )?;
    insert_check(
        &mut checks,
        "attention",
        &expected.attention,
        &actual.attention,
        1.0e-2,
        0.9995,
    )?;
    insert_check(
        &mut checks,
        "o_projected",
        &expected.o_projected,
        &actual.o_projected,
        2.0e-2,
        0.999,
    )?;
    insert_check(
        &mut checks,
        "attention_residual",
        &expected.attention_residual,
        &actual.attention_residual,
        2.0e-2,
        0.999,
    )?;
    insert_check(
        &mut checks,
        "post_attention_norm",
        &expected.post_attention_norm,
        &actual.post_normed,
        2.0e-2,
        0.999,
    )?;
    insert_check(
        &mut checks,
        "gate_projected",
        &expected.gate_projected,
        &actual.gate_projected,
        2.0e-2,
        0.999,
    )?;
    insert_check(
        &mut checks,
        "up_projected",
        &expected.up_projected,
        &actual.up_projected,
        2.0e-2,
        0.999,
    )?;
    insert_check(
        &mut checks,
        "silu_gate_mul_up",
        &expected.silu_gate_mul_up,
        &actual.mlp_activation,
        2.0e-2,
        0.999,
    )?;
    insert_check(
        &mut checks,
        "down_projected",
        &expected.down_projected,
        &actual.down_projected,
        2.0e-2,
        0.999,
    )?;
    insert_check(
        &mut checks,
        "output",
        &oracle.output,
        &actual.output,
        2.0e-2,
        0.999,
    )?;
    Ok(checks)
}

fn insert_check(
    checks: &mut BTreeMap<String, TensorCheck>,
    name: &str,
    expected: &[f32],
    actual: &[f32],
    max_relative_l2: f64,
    min_cosine: f64,
) -> Result<(), String> {
    let check = tensor_check(expected, actual, max_relative_l2, min_cosine)?;
    if checks.insert(name.to_string(), check).is_some() {
        return Err(format!("duplicate tensor check {name}"));
    }
    Ok(())
}

fn tensor_check(
    expected: &[f32],
    actual: &[f32],
    max_relative_l2: f64,
    min_cosine: f64,
) -> Result<TensorCheck, String> {
    let metrics = compare_sq8_correctness(expected, actual)?;
    let passed = metrics.nonfinite_count == 0
        && metrics.relative_l2.is_finite()
        && metrics.relative_l2 <= max_relative_l2
        && metrics.cosine_similarity.is_finite()
        && metrics.cosine_similarity >= min_cosine;
    Ok(TensorCheck {
        metrics,
        max_relative_l2,
        min_cosine,
        passed,
    })
}

fn validate_activations(
    m: usize,
    trace: &Sq8LayerRuntimeTrace,
) -> Result<BTreeMap<String, ActivationCheck>, String> {
    let mut checks = BTreeMap::new();
    activation_check(
        &mut checks,
        "input_norm_qkv",
        &trace.input_normed,
        trace.qkv_activation.as_ref(),
        m,
    )?;
    activation_check(
        &mut checks,
        "attention_o",
        &trace.attention,
        trace.o_activation.as_ref(),
        m,
    )?;
    activation_check(
        &mut checks,
        "post_norm_gate_up",
        &trace.post_normed,
        trace.gate_up_activation.as_ref(),
        m,
    )?;
    activation_check(
        &mut checks,
        "mlp_down",
        &trace.mlp_activation,
        trace.down_activation.as_ref(),
        m,
    )?;
    Ok(checks)
}

fn activation_check(
    checks: &mut BTreeMap<String, ActivationCheck>,
    name: &str,
    input: &[f32],
    actual: Option<&Sq8LayerQuantizedActivationTrace>,
    m: usize,
) -> Result<(), String> {
    let actual = actual.ok_or_else(|| format!("missing optimized activation {name}"))?;
    if actual.m != m || input.len() != m * actual.k {
        return Err(format!("activation {name} shape mismatch"));
    }
    let expected = quantize_sq8_dynamic_activation(input, m, actual.k)?;
    let encoded_byte_exact = expected.values() == actual.values;
    let scale_bit_exact = f32_bytes(expected.scales()) == f32_bytes(&actual.scales);
    let check = ActivationCheck {
        m,
        k: actual.k,
        encoded_byte_exact,
        scale_bit_exact,
        passed: encoded_byte_exact && scale_bit_exact,
    };
    if checks.insert(name.to_string(), check).is_some() {
        return Err(format!("duplicate activation check {name}"));
    }
    Ok(())
}

fn projection_executions(report: &Sq8LayerExecutionReport) -> ProjectionExecutions {
    ProjectionExecutions {
        q: format!("{:?}", report.q),
        k: format!("{:?}", report.k),
        v: format!("{:?}", report.v),
        o: format!("{:?}", report.o),
        gate: format!("{:?}", report.gate),
        up: format!("{:?}", report.up),
        down: format!("{:?}", report.down),
    }
}

fn timing_summary(mut samples_ms: Vec<f64>) -> Result<TimingSummary, String> {
    if samples_ms.is_empty()
        || samples_ms
            .iter()
            .any(|sample| !sample.is_finite() || *sample <= 0.0)
    {
        return Err("timing samples must be finite, positive, and non-empty".to_string());
    }
    let mut sorted = samples_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let p50_ms = percentile(&sorted, 0.50);
    let p95_ms = percentile(&sorted, 0.95);
    samples_ms.shrink_to_fit();
    Ok(TimingSummary {
        samples_ms,
        p50_ms,
        p95_ms,
    })
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let position = fraction * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn output_health(values: &[f32]) -> Result<OutputHealth, String> {
    if values.is_empty() {
        return Err("output health requires non-empty values".to_string());
    }
    let nonfinite = values.iter().filter(|value| !value.is_finite()).count();
    let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
    Ok(OutputHealth {
        elements: values.len(),
        nonfinite,
        minimum,
        maximum,
        max_abs,
        f32_le_sha256: sq8_f32_le_sha256(values)?,
    })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_json_no_clobber(path: &Path, result: &LayerResult) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("output has no parent: {}", path.display()))?;
    if !parent.is_dir() {
        return Err(format!(
            "output parent is not a directory: {}",
            parent.display()
        ));
    }
    let mut payload = serde_json::to_vec_pretty(result)
        .map_err(|err| format!("failed to serialize layer result: {err}"))?;
    payload.push(b'\n');
    let temporary = parent.join(format!(
        ".{}.tmp.{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "output filename is not UTF-8".to_string())?,
        std::process::id()
    ));
    let write_result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|err| format!("failed to create {}: {err}", temporary.display()))?;
        file.write_all(&payload)
            .map_err(|err| format!("failed to write {}: {err}", temporary.display()))?;
        file.sync_all()
            .map_err(|err| format!("failed to sync {}: {err}", temporary.display()))?;
        std::fs::hard_link(&temporary, path).map_err(|err| {
            format!(
                "failed to publish {} without clobbering: {err}",
                path.display()
            )
        })?;
        Ok::<(), String>(())
    })();
    let _ = std::fs::remove_file(&temporary);
    write_result
}
