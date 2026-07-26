// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Independent correctness verifier for the loader-free MoE runtime ABI.
//!
//! It has two modes:
//! - default: runs a synthetic complete MoE block through every CPU-reference
//!   stage and, with `--gpu`, every gfx1201 primitive;
//! - `--route-fixture DIR`: consumes an HF-produced real-router fixture and
//!   compares top-k IDs before optionally checking the GPU routing primitive;
//! - `--grouped-gemm-fixture DIR` / `--decode-gemm-fixture DIR`: consume a
//!   compact real 3-D BF16 expert slice through the prefill or decode path.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use ullm_runtime_sys::{
    MoeRouting, MoeShape, MoeWeightDtype, RuntimeBuffer, RuntimeContext, RuntimeStream, add_f32,
    device_count, moe_decode_gemm_f32, moe_decode_gemm_reference_raw_f32, moe_gated_silu_f32,
    moe_gated_silu_reference_f32, moe_gather_f32, moe_gather_reference_f32, moe_grouped_gemm_f32,
    moe_grouped_gemm_reference_f32, moe_grouped_gemm_reference_raw_f32, moe_route_f32,
    moe_route_reference_with_weight_dtype_f32, moe_scatter_weighted_f32,
    moe_scatter_weighted_reference_f32, moe_sigmoid_gate_f32, moe_sigmoid_gate_reference_f32,
};

#[derive(Debug)]
struct Options {
    gpu: bool,
    device: Option<u32>,
    report: PathBuf,
    route_fixture: Option<PathBuf>,
    grouped_gemm_fixture: Option<PathBuf>,
    decode_gemm_fixture: Option<PathBuf>,
    expect_boundary_tie: bool,
}

fn usage() -> &'static str {
    "usage: moe_runtime_verify --report PATH [--gpu] [--device N] [--route-fixture DIR | --grouped-gemm-fixture DIR | --decode-gemm-fixture DIR] [--expect-boundary-tie]"
}

fn parse_options() -> Result<Options, String> {
    let mut gpu = false;
    let mut device = None;
    let mut report = None;
    let mut route_fixture = None;
    let mut grouped_gemm_fixture = None;
    let mut decode_gemm_fixture = None;
    let mut expect_boundary_tie = false;
    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--gpu" => gpu = true,
            "--device" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--device requires an integer".to_string())?;
                device = Some(
                    value
                        .to_string_lossy()
                        .parse::<u32>()
                        .map_err(|_| "--device must be an unsigned integer".to_string())?,
                );
            }
            "--report" => {
                report = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--report requires a path".to_string())?,
                ));
            }
            "--route-fixture" => {
                route_fixture =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--route-fixture requires a directory".to_string()
                    })?));
            }
            "--grouped-gemm-fixture" => {
                grouped_gemm_fixture =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--grouped-gemm-fixture requires a directory".to_string()
                    })?));
            }
            "--decode-gemm-fixture" => {
                decode_gemm_fixture =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--decode-gemm-fixture requires a directory".to_string()
                    })?));
            }
            "--expect-boundary-tie" => expect_boundary_tie = true,
            "--help" | "-h" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}; {}", usage())),
        }
    }
    let options = Options {
        gpu,
        device,
        report: report.ok_or_else(|| "--report is required".to_string())?,
        route_fixture,
        grouped_gemm_fixture,
        decode_gemm_fixture,
        expect_boundary_tie,
    };
    let fixture_count = usize::from(options.route_fixture.is_some())
        + usize::from(options.grouped_gemm_fixture.is_some())
        + usize::from(options.decode_gemm_fixture.is_some());
    if fixture_count > 1 {
        return Err(
            "--route-fixture, --grouped-gemm-fixture, and --decode-gemm-fixture are mutually exclusive"
                .to_string(),
        );
    }
    if options.expect_boundary_tie && options.route_fixture.is_none() {
        return Err("--expect-boundary-tie requires --route-fixture".to_string());
    }
    Ok(options)
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    // T is used only for F32/I32/U32 vectors in this verifier, all plain data.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn f32_from_bytes(bytes: &[u8], label: &str) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(format!("{label} byte length is not F32-aligned"));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn i32_from_bytes(bytes: &[u8], label: &str) -> Result<Vec<i32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<i32>()) {
        return Err(format!("{label} byte length is not I32-aligned"));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn write_report(path: &Path, body: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "cannot create report directory {}: {error}",
                parent.display()
            )
        })?;
    }
    if path.exists() {
        return Err(format!(
            "refusing to overwrite existing report {}",
            path.display()
        ));
    }
    fs::write(path, body)
        .map_err(|error| format!("cannot write report {}: {error}", path.display()))
}

fn gfx1201_runtime_device(requested: Option<u32>) -> Result<u32, String> {
    if let Some(index) = requested {
        let info = ullm_runtime_sys::device_info(index)?;
        if info.backend == "hip" && info.gcn_arch_name.starts_with("gfx1201") {
            return Ok(index);
        }
        return Err(format!(
            "requested runtime device {index} is not the permitted gfx1201/R9700 (backend={} arch={})",
            info.backend, info.gcn_arch_name
        ));
    }
    for index in 1..device_count()? {
        let info = ullm_runtime_sys::device_info(index)?;
        if info.backend == "hip" && info.gcn_arch_name.starts_with("gfx1201") {
            return Ok(index);
        }
    }
    Err("no permitted gfx1201/R9700 HIP runtime device is visible".to_string())
}

fn values(count: usize, phase: f32) -> Vec<f32> {
    (0..count)
        .map(|index| ((index as f32 + 1.0) * 0.173 + phase).sin() * 0.37)
        .collect()
}

fn f32_to_bf16_roundtrip(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return value;
    }
    let rounded = bits.wrapping_add(0x7fff).wrapping_add((bits >> 16) & 1);
    f32::from_bits(rounded & 0xffff_0000)
}

fn encode_weight_values(values: &[f32], dtype: MoeWeightDtype) -> Vec<u8> {
    match dtype {
        MoeWeightDtype::F32 => as_bytes(values).to_vec(),
        MoeWeightDtype::Bf16 => {
            let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u16>());
            for value in values {
                let rounded = f32_to_bf16_roundtrip(*value);
                bytes.extend_from_slice(&((rounded.to_bits() >> 16) as u16).to_ne_bytes());
            }
            bytes
        }
    }
}

struct SyntheticWeights {
    router: Vec<f32>,
    expert_gate_up: Vec<f32>,
    expert_down: Vec<f32>,
    shared_gate_up: Vec<f32>,
    shared_down: Vec<f32>,
    shared_expert_gate: Vec<f32>,
}

struct CpuStages {
    routing: MoeRouting,
    gathered: Vec<f32>,
    expert_gate_up: Vec<f32>,
    activated: Vec<f32>,
    expert_output: Vec<f32>,
    routed: Vec<f32>,
    shared_gate_up: Vec<f32>,
    shared_active: Vec<f32>,
    shared_output: Vec<f32>,
    shared_gate_values: Vec<f32>,
    gated_shared: Vec<f32>,
    final_output: Vec<f32>,
}

fn make_synthetic_weights(shape: MoeShape) -> SyntheticWeights {
    let shared_gate = values(shape.shared_intermediate_size * shape.hidden_size, 0.5);
    let shared_up = values(shape.shared_intermediate_size * shape.hidden_size, 0.6);
    let mut shared_gate_up = vec![0.0_f32; 2 * shape.shared_intermediate_size * shape.hidden_size];
    for row in 0..shape.shared_intermediate_size {
        let src = row * shape.hidden_size;
        let gate_dst = row * shape.hidden_size;
        let up_dst = (shape.shared_intermediate_size + row) * shape.hidden_size;
        shared_gate_up[gate_dst..gate_dst + shape.hidden_size]
            .copy_from_slice(&shared_gate[src..src + shape.hidden_size]);
        shared_gate_up[up_dst..up_dst + shape.hidden_size]
            .copy_from_slice(&shared_up[src..src + shape.hidden_size]);
    }
    SyntheticWeights {
        router: values(shape.num_experts * shape.hidden_size, 0.2),
        expert_gate_up: values(
            shape.num_experts * 2 * shape.intermediate_size * shape.hidden_size,
            0.3,
        ),
        expert_down: values(
            shape.num_experts * shape.hidden_size * shape.intermediate_size,
            0.4,
        ),
        shared_gate_up,
        shared_down: values(shape.hidden_size * shape.shared_intermediate_size, 0.7),
        shared_expert_gate: values(shape.hidden_size, 0.8),
    }
}

#[allow(clippy::too_many_arguments)]
fn grouped_reference_for_dtype(
    weights: &[f32],
    weight_dtype: MoeWeightDtype,
    expert_ids: &[i32],
    input: &[f32],
    assignments: usize,
    num_experts: usize,
    rows_per_expert: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    match weight_dtype {
        MoeWeightDtype::F32 => moe_grouped_gemm_reference_f32(
            weights,
            expert_ids,
            input,
            assignments,
            num_experts,
            rows_per_expert,
            cols,
        ),
        MoeWeightDtype::Bf16 => moe_grouped_gemm_reference_raw_f32(
            &encode_weight_values(weights, MoeWeightDtype::Bf16),
            MoeWeightDtype::Bf16,
            expert_ids,
            input,
            assignments,
            num_experts,
            rows_per_expert,
            cols,
        ),
    }
}

fn cpu_stages(
    shape: MoeShape,
    hidden: &[f32],
    weights: &SyntheticWeights,
    weight_dtype: MoeWeightDtype,
) -> Result<CpuStages, String> {
    let router = match weight_dtype {
        MoeWeightDtype::F32 => weights.router.clone(),
        MoeWeightDtype::Bf16 => weights
            .router
            .iter()
            .map(|value| f32_to_bf16_roundtrip(*value))
            .collect(),
    };
    let routing = moe_route_reference_with_weight_dtype_f32(shape, hidden, &router, weight_dtype)?;
    if routing.has_boundary_tie() {
        return Err("synthetic routing unexpectedly has a top-k boundary tie".to_string());
    }
    let assignments = shape.tokens * shape.top_k;
    let gathered = moe_gather_reference_f32(hidden, shape.tokens, shape.hidden_size, shape.top_k)?;
    let expert_gate_up = grouped_reference_for_dtype(
        &weights.expert_gate_up,
        weight_dtype,
        &routing.selected_expert_ids,
        &gathered,
        assignments,
        shape.num_experts,
        2 * shape.intermediate_size,
        shape.hidden_size,
    )?;
    let activated =
        moe_gated_silu_reference_f32(&expert_gate_up, assignments, shape.intermediate_size)?;
    let expert_output = grouped_reference_for_dtype(
        &weights.expert_down,
        weight_dtype,
        &routing.selected_expert_ids,
        &activated,
        assignments,
        shape.num_experts,
        shape.hidden_size,
        shape.intermediate_size,
    )?;
    let routed = moe_scatter_weighted_reference_f32(
        &expert_output,
        &routing.routing_scores,
        shape.tokens,
        shape.top_k,
        shape.hidden_size,
    )?;
    let shared_ids = vec![0_i32; shape.tokens];
    let shared_gate_up = grouped_reference_for_dtype(
        &weights.shared_gate_up,
        weight_dtype,
        &shared_ids,
        hidden,
        shape.tokens,
        1,
        2 * shape.shared_intermediate_size,
        shape.hidden_size,
    )?;
    let shared_active = moe_gated_silu_reference_f32(
        &shared_gate_up,
        shape.tokens,
        shape.shared_intermediate_size,
    )?;
    let shared_output = grouped_reference_for_dtype(
        &weights.shared_down,
        weight_dtype,
        &shared_ids,
        &shared_active,
        shape.tokens,
        1,
        shape.hidden_size,
        shape.shared_intermediate_size,
    )?;
    let shared_gate_values = grouped_reference_for_dtype(
        &weights.shared_expert_gate,
        weight_dtype,
        &shared_ids,
        hidden,
        shape.tokens,
        1,
        1,
        shape.hidden_size,
    )?;
    let gated_shared = moe_sigmoid_gate_reference_f32(
        &shared_gate_values,
        &shared_output,
        shape.tokens,
        shape.hidden_size,
    )?;
    let final_output = routed
        .iter()
        .zip(&gated_shared)
        .map(|(routed, shared)| routed + shared)
        .collect();
    Ok(CpuStages {
        routing,
        gathered,
        expert_gate_up,
        activated,
        expert_output,
        routed,
        shared_gate_up,
        shared_active,
        shared_output,
        shared_gate_values,
        gated_shared,
        final_output,
    })
}

fn upload(
    context: &mut RuntimeContext,
    values: &[u8],
    stream: &mut RuntimeStream,
) -> Result<RuntimeBuffer, String> {
    let mut buffer = context.alloc_buffer(values.len())?;
    buffer.copy_from_host(0, values, Some(stream))?;
    Ok(buffer)
}

fn allocate_f32(context: &mut RuntimeContext, elements: usize) -> Result<RuntimeBuffer, String> {
    context.alloc_buffer(
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "F32 allocation size overflows".to_string())?,
    )
}

fn allocate_i32(context: &mut RuntimeContext, elements: usize) -> Result<RuntimeBuffer, String> {
    context.alloc_buffer(
        elements
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| "I32 allocation size overflows".to_string())?,
    )
}

fn allocate_u32(context: &mut RuntimeContext, elements: usize) -> Result<RuntimeBuffer, String> {
    context.alloc_buffer(
        elements
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "U32 allocation size overflows".to_string())?,
    )
}

fn download_f32(
    buffer: &RuntimeBuffer,
    values: usize,
    stream: &mut RuntimeStream,
) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0_u8; values * std::mem::size_of::<f32>()];
    buffer.copy_to_host(0, &mut bytes, Some(stream))?;
    stream.synchronize()?;
    f32_from_bytes(&bytes, "downloaded F32")
}

fn download_i32(
    buffer: &RuntimeBuffer,
    values: usize,
    stream: &mut RuntimeStream,
) -> Result<Vec<i32>, String> {
    let mut bytes = vec![0_u8; values * std::mem::size_of::<i32>()];
    buffer.copy_to_host(0, &mut bytes, Some(stream))?;
    stream.synchronize()?;
    i32_from_bytes(&bytes, "downloaded I32")
}

fn download_u32(
    buffer: &RuntimeBuffer,
    values: usize,
    stream: &mut RuntimeStream,
) -> Result<Vec<u32>, String> {
    let mut bytes = vec![0_u8; values * std::mem::size_of::<u32>()];
    buffer.copy_to_host(0, &mut bytes, Some(stream))?;
    stream.synchronize()?;
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn max_abs(actual: &[f32], expected: &[f32], label: &str) -> Result<f32, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} length mismatch: {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max))
}

struct RuntimeRoute {
    runtime_device_index: u32,
    device_id: i32,
    backend: String,
    arch: String,
    ids: Vec<i32>,
    scores: Vec<f32>,
    flags: Vec<u32>,
}

fn execute_runtime_route(
    hidden: &[f32],
    router_weights: &[u8],
    weight_dtype: MoeWeightDtype,
    tokens: usize,
    hidden_size: usize,
    num_experts: usize,
    top_k: usize,
    runtime_device_index: u32,
) -> Result<RuntimeRoute, String> {
    let mut context = RuntimeContext::create(runtime_device_index)?;
    let info = context.device_info()?;
    let mut stream = context.create_stream()?;
    let hidden_buffer = upload(&mut context, as_bytes(hidden), &mut stream)?;
    let router_buffer = upload(&mut context, router_weights, &mut stream)?;
    let selected = tokens * top_k;
    let mut scores = allocate_f32(&mut context, selected)?;
    let mut ids = allocate_i32(&mut context, selected)?;
    let mut flags = allocate_u32(&mut context, tokens)?;
    moe_route_f32(
        &hidden_buffer,
        &router_buffer,
        weight_dtype,
        tokens,
        hidden_size,
        num_experts,
        top_k,
        &mut scores,
        &mut ids,
        &mut flags,
        Some(&mut stream),
    )?;
    Ok(RuntimeRoute {
        runtime_device_index,
        device_id: info.device_id,
        backend: info.backend,
        arch: info.gcn_arch_name,
        ids: download_i32(&ids, selected, &mut stream)?,
        scores: download_f32(&scores, selected, &mut stream)?,
        flags: download_u32(&flags, tokens, &mut stream)?,
    })
}

struct RuntimeGroupedGemm {
    runtime_device_index: u32,
    device_id: i32,
    backend: String,
    arch: String,
    output: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
enum GemmPath {
    Decode,
    Prefill,
}

#[allow(clippy::too_many_arguments)]
fn runtime_gemm(
    path: GemmPath,
    weight_buffer: &RuntimeBuffer,
    weight_dtype: MoeWeightDtype,
    expert_ids_buffer: &RuntimeBuffer,
    input_buffer: &RuntimeBuffer,
    assignments: usize,
    num_experts: usize,
    rows_per_expert: usize,
    cols: usize,
    output_buffer: &mut RuntimeBuffer,
    stream: &mut RuntimeStream,
) -> Result<(), String> {
    match path {
        GemmPath::Decode => moe_decode_gemm_f32(
            weight_buffer,
            weight_dtype,
            expert_ids_buffer,
            input_buffer,
            assignments,
            num_experts,
            rows_per_expert,
            cols,
            output_buffer,
            Some(stream),
        ),
        GemmPath::Prefill => moe_grouped_gemm_f32(
            weight_buffer,
            weight_dtype,
            expert_ids_buffer,
            input_buffer,
            assignments,
            num_experts,
            rows_per_expert,
            cols,
            output_buffer,
            Some(stream),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_runtime_grouped_gemm(
    weights: &[u8],
    expert_ids: &[i32],
    input: &[f32],
    assignments: usize,
    num_experts: usize,
    rows_per_expert: usize,
    cols: usize,
    runtime_device_index: u32,
    path: GemmPath,
) -> Result<RuntimeGroupedGemm, String> {
    let mut context = RuntimeContext::create(runtime_device_index)?;
    let info = context.device_info()?;
    let mut stream = context.create_stream()?;
    let weight_buffer = upload(&mut context, weights, &mut stream)?;
    let ids_buffer = upload(&mut context, as_bytes(expert_ids), &mut stream)?;
    let input_buffer = upload(&mut context, as_bytes(input), &mut stream)?;
    let mut output_buffer = allocate_f32(
        &mut context,
        assignments
            .checked_mul(rows_per_expert)
            .ok_or_else(|| "grouped-GEMM output element count overflows".to_string())?,
    )?;
    runtime_gemm(
        path,
        &weight_buffer,
        MoeWeightDtype::Bf16,
        &ids_buffer,
        &input_buffer,
        assignments,
        num_experts,
        rows_per_expert,
        cols,
        &mut output_buffer,
        &mut stream,
    )?;
    Ok(RuntimeGroupedGemm {
        runtime_device_index,
        device_id: info.device_id,
        backend: info.backend,
        arch: info.gcn_arch_name,
        output: download_f32(&output_buffer, assignments * rows_per_expert, &mut stream)?,
    })
}

fn verify_runtime_synthetic(
    shape: MoeShape,
    hidden: &[f32],
    weights: &SyntheticWeights,
    cpu: &CpuStages,
    weight_dtype: MoeWeightDtype,
    gemm_path: GemmPath,
    device: u32,
    require_gfx1201: bool,
) -> Result<String, String> {
    let mut context = RuntimeContext::create(device)?;
    let info = context.device_info()?;
    if require_gfx1201 && (info.backend != "hip" || !info.gcn_arch_name.starts_with("gfx1201")) {
        return Err(format!(
            "--gpu requires the permitted gfx1201/R9700 device, got backend={} arch={}",
            info.backend, info.gcn_arch_name
        ));
    }
    let mut stream = context.create_stream()?;
    let assignments = shape.tokens * shape.top_k;
    let hidden_buffer = upload(&mut context, as_bytes(hidden), &mut stream)?;
    let router_buffer = upload(
        &mut context,
        &encode_weight_values(&weights.router, weight_dtype),
        &mut stream,
    )?;
    let expert_gate_up_weight = upload(
        &mut context,
        &encode_weight_values(&weights.expert_gate_up, weight_dtype),
        &mut stream,
    )?;
    let expert_down_weight = upload(
        &mut context,
        &encode_weight_values(&weights.expert_down, weight_dtype),
        &mut stream,
    )?;
    let shared_ids_host = vec![0_i32; shape.tokens];
    let shared_ids = upload(&mut context, as_bytes(&shared_ids_host), &mut stream)?;
    let shared_gate_up_weight = upload(
        &mut context,
        &encode_weight_values(&weights.shared_gate_up, weight_dtype),
        &mut stream,
    )?;
    let shared_down_weight = upload(
        &mut context,
        &encode_weight_values(&weights.shared_down, weight_dtype),
        &mut stream,
    )?;
    let shared_gate_weight = upload(
        &mut context,
        &encode_weight_values(&weights.shared_expert_gate, weight_dtype),
        &mut stream,
    )?;

    let mut routing_scores = allocate_f32(&mut context, assignments)?;
    let mut selected_ids = allocate_i32(&mut context, assignments)?;
    let mut tie_flags = allocate_u32(&mut context, shape.tokens)?;
    let mut gathered = allocate_f32(&mut context, assignments * shape.hidden_size)?;
    let mut expert_gate_up = allocate_f32(&mut context, assignments * 2 * shape.intermediate_size)?;
    let mut activated = allocate_f32(&mut context, assignments * shape.intermediate_size)?;
    let mut expert_output = allocate_f32(&mut context, assignments * shape.hidden_size)?;
    let mut routed = allocate_f32(&mut context, shape.tokens * shape.hidden_size)?;
    let mut shared_gate_up = allocate_f32(
        &mut context,
        shape.tokens * 2 * shape.shared_intermediate_size,
    )?;
    let mut shared_active =
        allocate_f32(&mut context, shape.tokens * shape.shared_intermediate_size)?;
    let mut shared_output = allocate_f32(&mut context, shape.tokens * shape.hidden_size)?;
    let mut shared_gate_values = allocate_f32(&mut context, shape.tokens)?;
    let mut gated_shared = allocate_f32(&mut context, shape.tokens * shape.hidden_size)?;
    let mut final_output = allocate_f32(&mut context, shape.tokens * shape.hidden_size)?;

    moe_route_f32(
        &hidden_buffer,
        &router_buffer,
        weight_dtype,
        shape.tokens,
        shape.hidden_size,
        shape.num_experts,
        shape.top_k,
        &mut routing_scores,
        &mut selected_ids,
        &mut tie_flags,
        Some(&mut stream),
    )?;
    moe_gather_f32(
        &hidden_buffer,
        shape.tokens,
        shape.hidden_size,
        shape.top_k,
        &mut gathered,
        Some(&mut stream),
    )?;
    runtime_gemm(
        gemm_path,
        &expert_gate_up_weight,
        weight_dtype,
        &selected_ids,
        &gathered,
        assignments,
        shape.num_experts,
        2 * shape.intermediate_size,
        shape.hidden_size,
        &mut expert_gate_up,
        &mut stream,
    )?;
    moe_gated_silu_f32(
        &expert_gate_up,
        assignments,
        shape.intermediate_size,
        &mut activated,
        Some(&mut stream),
    )?;
    runtime_gemm(
        gemm_path,
        &expert_down_weight,
        weight_dtype,
        &selected_ids,
        &activated,
        assignments,
        shape.num_experts,
        shape.hidden_size,
        shape.intermediate_size,
        &mut expert_output,
        &mut stream,
    )?;
    moe_scatter_weighted_f32(
        &expert_output,
        &routing_scores,
        shape.tokens,
        shape.top_k,
        shape.hidden_size,
        &mut routed,
        Some(&mut stream),
    )?;
    runtime_gemm(
        gemm_path,
        &shared_gate_up_weight,
        weight_dtype,
        &shared_ids,
        &hidden_buffer,
        shape.tokens,
        1,
        2 * shape.shared_intermediate_size,
        shape.hidden_size,
        &mut shared_gate_up,
        &mut stream,
    )?;
    moe_gated_silu_f32(
        &shared_gate_up,
        shape.tokens,
        shape.shared_intermediate_size,
        &mut shared_active,
        Some(&mut stream),
    )?;
    runtime_gemm(
        gemm_path,
        &shared_down_weight,
        weight_dtype,
        &shared_ids,
        &shared_active,
        shape.tokens,
        1,
        shape.hidden_size,
        shape.shared_intermediate_size,
        &mut shared_output,
        &mut stream,
    )?;
    runtime_gemm(
        gemm_path,
        &shared_gate_weight,
        weight_dtype,
        &shared_ids,
        &hidden_buffer,
        shape.tokens,
        1,
        1,
        shape.hidden_size,
        &mut shared_gate_values,
        &mut stream,
    )?;
    moe_sigmoid_gate_f32(
        &shared_gate_values,
        &shared_output,
        shape.tokens,
        shape.hidden_size,
        &mut gated_shared,
        Some(&mut stream),
    )?;
    add_f32(
        &routed,
        &gated_shared,
        shape.tokens * shape.hidden_size,
        &mut final_output,
        Some(&mut stream),
    )?;

    let gpu_ids = download_i32(&selected_ids, assignments, &mut stream)?;
    let gpu_scores = download_f32(&routing_scores, assignments, &mut stream)?;
    let gpu_flags = download_u32(&tie_flags, shape.tokens, &mut stream)?;
    let checks = [
        (
            "gather",
            download_f32(&gathered, cpu.gathered.len(), &mut stream)?,
            &cpu.gathered,
        ),
        (
            "expert_gate_up",
            download_f32(&expert_gate_up, cpu.expert_gate_up.len(), &mut stream)?,
            &cpu.expert_gate_up,
        ),
        (
            "activated",
            download_f32(&activated, cpu.activated.len(), &mut stream)?,
            &cpu.activated,
        ),
        (
            "expert_output",
            download_f32(&expert_output, cpu.expert_output.len(), &mut stream)?,
            &cpu.expert_output,
        ),
        (
            "routed",
            download_f32(&routed, cpu.routed.len(), &mut stream)?,
            &cpu.routed,
        ),
        (
            "shared_gate_up",
            download_f32(&shared_gate_up, cpu.shared_gate_up.len(), &mut stream)?,
            &cpu.shared_gate_up,
        ),
        (
            "shared_active",
            download_f32(&shared_active, cpu.shared_active.len(), &mut stream)?,
            &cpu.shared_active,
        ),
        (
            "shared_output",
            download_f32(&shared_output, cpu.shared_output.len(), &mut stream)?,
            &cpu.shared_output,
        ),
        (
            "shared_gate",
            download_f32(
                &shared_gate_values,
                cpu.shared_gate_values.len(),
                &mut stream,
            )?,
            &cpu.shared_gate_values,
        ),
        (
            "gated_shared",
            download_f32(&gated_shared, cpu.gated_shared.len(), &mut stream)?,
            &cpu.gated_shared,
        ),
        (
            "final_output",
            download_f32(&final_output, cpu.final_output.len(), &mut stream)?,
            &cpu.final_output,
        ),
    ];
    if gpu_ids != cpu.routing.selected_expert_ids {
        return Err(format!(
            "GPU routing IDs differ: {gpu_ids:?} != {:?}",
            cpu.routing.selected_expert_ids
        ));
    }
    if gpu_flags != cpu.routing.boundary_tie_flags {
        return Err(format!(
            "GPU routing tie flags differ: {gpu_flags:?} != {:?}",
            cpu.routing.boundary_tie_flags
        ));
    }
    let score_error = max_abs(&gpu_scores, &cpu.routing.routing_scores, "routing scores")?;
    if score_error > 2.0e-5 {
        return Err(format!(
            "GPU routing score max abs error {score_error:e} exceeds 2e-5"
        ));
    }
    let mut rows = Vec::new();
    rows.push(format!("routing_scores={score_error:.9e}"));
    for (label, actual, expected) in checks {
        let error = max_abs(&actual, expected, label)?;
        if error > 2.0e-4 {
            return Err(format!("GPU {label} max abs error {error:e} exceeds 2e-4"));
        }
        rows.push(format!("{label}={error:.9e}"));
    }
    Ok(format!(
        "{{\"runtime_device_index\":{},\"device_id\":{},\"backend\":{:?},\"name\":{:?},\"arch\":{:?},\"max_abs\":{{{}}}}}",
        device,
        info.device_id,
        info.backend,
        info.name,
        info.gcn_arch_name,
        rows.into_iter()
            .map(|row| {
                let (key, value) = row.split_once('=').unwrap();
                format!("\"{key}\":{value}")
            })
            .collect::<Vec<_>>()
            .join(",")
    ))
}

fn run_synthetic_path(
    options: &Options,
    shape: MoeShape,
    phase: f32,
    gemm_path: GemmPath,
) -> Result<String, String> {
    let hidden = values(shape.tokens * shape.hidden_size, phase);
    let weights = make_synthetic_weights(shape);
    let f32_cpu = cpu_stages(shape, &hidden, &weights, MoeWeightDtype::F32)?;
    let bf16_cpu = cpu_stages(shape, &hidden, &weights, MoeWeightDtype::Bf16)?;
    if f32_cpu.routing.has_boundary_tie() || bf16_cpu.routing.has_boundary_tie() {
        return Err("synthetic routing unexpectedly has a top-k boundary tie".to_string());
    }
    let f32_cpu_runtime = verify_runtime_synthetic(
        shape,
        &hidden,
        &weights,
        &f32_cpu,
        MoeWeightDtype::F32,
        gemm_path,
        0,
        false,
    )?;
    let bf16_cpu_runtime = verify_runtime_synthetic(
        shape,
        &hidden,
        &weights,
        &bf16_cpu,
        MoeWeightDtype::Bf16,
        gemm_path,
        0,
        false,
    )?;
    let (f32_gpu, bf16_gpu) = if options.gpu {
        let device = gfx1201_runtime_device(options.device)?;
        (
            verify_runtime_synthetic(
                shape,
                &hidden,
                &weights,
                &f32_cpu,
                MoeWeightDtype::F32,
                gemm_path,
                device,
                true,
            )?,
            verify_runtime_synthetic(
                shape,
                &hidden,
                &weights,
                &bf16_cpu,
                MoeWeightDtype::Bf16,
                gemm_path,
                device,
                true,
            )?,
        )
    } else {
        ("null".to_string(), "null".to_string())
    };
    Ok(format!(
        "{{\"shape\":{{\"tokens\":{},\"hidden_size\":{},\"num_experts\":{},\"top_k\":{},\"intermediate_size\":{},\"shared_intermediate_size\":{}}},\"f32\":{{\"cpu_reference\":{{\"routing_boundary_tie\":false,\"output_elements\":{}}},\"cpu_runtime\":{f32_cpu_runtime},\"gpu\":{f32_gpu}}},\"bf16\":{{\"cpu_reference\":{{\"routing_boundary_tie\":false,\"output_elements\":{}}},\"cpu_runtime\":{bf16_cpu_runtime},\"gpu\":{bf16_gpu}}}}}",
        shape.tokens,
        shape.hidden_size,
        shape.num_experts,
        shape.top_k,
        shape.intermediate_size,
        shape.shared_intermediate_size,
        f32_cpu.final_output.len(),
        bf16_cpu.final_output.len(),
    ))
}

fn run_synthetic(options: &Options) -> Result<String, String> {
    let prefill_shape = MoeShape {
        tokens: 5,
        hidden_size: 16,
        num_experts: 7,
        top_k: 3,
        intermediate_size: 9,
        shared_intermediate_size: 7,
    };
    let decode_shape = MoeShape {
        tokens: 1,
        hidden_size: 16,
        num_experts: 7,
        top_k: 3,
        intermediate_size: 9,
        shared_intermediate_size: 7,
    };
    let prefill = run_synthetic_path(options, prefill_shape, 0.1, GemmPath::Prefill)?;
    let decode = run_synthetic_path(options, decode_shape, 0.73, GemmPath::Decode)?;
    Ok(format!(
        "{{\"schema\":\"ullm.moe_runtime_verify.v1\",\"mode\":\"synthetic-full\",\"passed\":true,\"prefill\":{prefill},\"decode\":{decode}}}\n"
    ))
}

fn read_shape(dir: &Path) -> Result<(usize, usize, usize, usize), String> {
    let raw = fs::read_to_string(dir.join("shape.txt"))
        .map_err(|error| format!("cannot read route fixture shape: {error}"))?;
    let values = raw
        .split_whitespace()
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            "route fixture shape.txt must contain tokens hidden experts top_k".to_string()
        })?;
    if values.len() != 4 {
        return Err("route fixture shape.txt must contain exactly four integers".to_string());
    }
    Ok((values[0], values[1], values[2], values[3]))
}

fn read_grouped_gemm_shape(dir: &Path) -> Result<(usize, usize, usize, usize), String> {
    let raw = fs::read_to_string(dir.join("shape.txt"))
        .map_err(|error| format!("cannot read grouped-GEMM fixture shape: {error}"))?;
    let values = raw
        .split_whitespace()
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            "grouped-GEMM shape.txt must contain assignments experts rows cols".to_string()
        })?;
    if values.len() != 4 || values.contains(&0) {
        return Err("grouped-GEMM shape.txt must contain four nonzero integers".to_string());
    }
    Ok((values[0], values[1], values[2], values[3]))
}

fn run_grouped_gemm_fixture(
    options: &Options,
    dir: &Path,
    path: GemmPath,
) -> Result<String, String> {
    let (assignments, num_experts, rows_per_expert, cols) = read_grouped_gemm_shape(dir)?;
    let weights = fs::read(dir.join("weights.bf16"))
        .map_err(|error| format!("cannot read raw BF16 grouped-GEMM weights: {error}"))?;
    let expert_ids = i32_from_bytes(
        &fs::read(dir.join("expert_ids.i32")).map_err(|error| error.to_string())?,
        "grouped-GEMM expert IDs",
    )?;
    let input = f32_from_bytes(
        &fs::read(dir.join("input.f32")).map_err(|error| error.to_string())?,
        "grouped-GEMM input",
    )?;
    let expected = f32_from_bytes(
        &fs::read(dir.join("expected.f32")).map_err(|error| error.to_string())?,
        "grouped-GEMM HF expected output",
    )?;
    let reference = match path {
        GemmPath::Decode => moe_decode_gemm_reference_raw_f32(
            &weights,
            MoeWeightDtype::Bf16,
            &expert_ids,
            &input,
            assignments,
            num_experts,
            rows_per_expert,
            cols,
        )?,
        GemmPath::Prefill => moe_grouped_gemm_reference_raw_f32(
            &weights,
            MoeWeightDtype::Bf16,
            &expert_ids,
            &input,
            assignments,
            num_experts,
            rows_per_expert,
            cols,
        )?,
    };
    let path_label = match path {
        GemmPath::Decode => "decode-GEMM",
        GemmPath::Prefill => "prefill grouped-GEMM",
    };
    let expected_error = max_abs(&reference, &expected, "HF GEMM expected output")?;
    if expected_error > 2.0e-5 {
        return Err(format!(
            "CPU reference differs from HF {path_label} fixture by {expected_error:e}, exceeds 2e-5"
        ));
    }
    let cpu_runtime = execute_runtime_grouped_gemm(
        &weights,
        &expert_ids,
        &input,
        assignments,
        num_experts,
        rows_per_expert,
        cols,
        0,
        path,
    )?;
    let cpu_error = max_abs(&cpu_runtime.output, &reference, "CPU C ABI GEMM output")?;
    if cpu_error != 0.0 {
        return Err(format!("CPU C ABI {path_label} differs by {cpu_error:e}"));
    }
    let gpu = if options.gpu {
        let runtime = execute_runtime_grouped_gemm(
            &weights,
            &expert_ids,
            &input,
            assignments,
            num_experts,
            rows_per_expert,
            cols,
            gfx1201_runtime_device(options.device)?,
            path,
        )?;
        if runtime.backend != "hip" || !runtime.arch.starts_with("gfx1201") {
            return Err(format!(
                "--gpu {path_label} fixture selected non-gfx1201 runtime device {} {}",
                runtime.backend, runtime.arch
            ));
        }
        let error = max_abs(&runtime.output, &reference, "GPU GEMM output")?;
        if error > 2.0e-5 {
            return Err(format!("GPU {path_label} error {error:e} exceeds 2e-5"));
        }
        format!(
            "{{\"runtime_device_index\":{},\"device_id\":{},\"max_abs\":{error:.9e}}}",
            runtime.runtime_device_index, runtime.device_id
        )
    } else {
        "null".to_string()
    };
    Ok(format!(
        "{{\"schema\":\"ullm.moe_runtime_verify.v1\",\"mode\":{:?},\"passed\":true,\"weight_dtype\":\"bf16\",\"shape\":{{\"assignments\":{},\"num_experts\":{},\"rows_per_expert\":{},\"cols\":{}}},\"hf_expected_max_abs\":{expected_error:.9e},\"cpu_runtime\":{{\"runtime_device_index\":{},\"device_id\":{},\"backend\":{:?},\"max_abs\":{cpu_error:.9e}}},\"gpu\":{gpu}}}\n",
        match path {
            GemmPath::Decode => "hf-decode-gemm-fixture",
            GemmPath::Prefill => "hf-prefill-grouped-gemm-fixture",
        },
        assignments,
        num_experts,
        rows_per_expert,
        cols,
        cpu_runtime.runtime_device_index,
        cpu_runtime.device_id,
        cpu_runtime.backend,
    ))
}

fn run_route_fixture(options: &Options, dir: &Path) -> Result<String, String> {
    let (tokens, hidden_size, num_experts, top_k) = read_shape(dir)?;
    let shape = MoeShape {
        tokens,
        hidden_size,
        num_experts,
        top_k,
        intermediate_size: 1,
        shared_intermediate_size: 1,
    };
    let hidden = f32_from_bytes(
        &fs::read(dir.join("hidden.f32")).map_err(|e| e.to_string())?,
        "fixture hidden",
    )?;
    let router = f32_from_bytes(
        &fs::read(dir.join("router.f32")).map_err(|e| e.to_string())?,
        "fixture router",
    )?;
    let router_bf16 = fs::read(dir.join("router.bf16"))
        .map_err(|error| format!("cannot read fixture raw BF16 router: {error}"))?;
    let expected_ids = i32_from_bytes(
        &fs::read(dir.join("expected_indices.i32")).map_err(|e| e.to_string())?,
        "fixture expected indices",
    )?;
    let expected_scores = f32_from_bytes(
        &fs::read(dir.join("expected_scores.f32")).map_err(|e| e.to_string())?,
        "fixture expected scores",
    )?;
    let routing =
        moe_route_reference_with_weight_dtype_f32(shape, &hidden, &router, MoeWeightDtype::Bf16)?;
    let ids_match = routing.selected_expert_ids == expected_ids;
    let score_error = max_abs(
        &routing.routing_scores,
        &expected_scores,
        "HF routing score",
    )?;
    let tie_detected = routing.has_boundary_tie();
    if options.expect_boundary_tie {
        if !tie_detected {
            return Err(
                "expected a top-k boundary tie but the runtime did not flag one".to_string(),
            );
        }
    } else if tie_detected {
        return Err("real HF routing fixture has a top-k boundary tie".to_string());
    } else if !ids_match {
        return Err(format!(
            "CPU route IDs differ from HF: {:?} != {:?}",
            routing.selected_expert_ids, expected_ids
        ));
    }
    let cpu_runtime = execute_runtime_route(
        &hidden,
        &router_bf16,
        MoeWeightDtype::Bf16,
        tokens,
        hidden_size,
        num_experts,
        top_k,
        0,
    )?;
    if cpu_runtime.ids != routing.selected_expert_ids
        || cpu_runtime.flags != routing.boundary_tie_flags
    {
        return Err("CPU C ABI route result differs from the Rust reference".to_string());
    }
    let cpu_runtime_error = max_abs(
        &cpu_runtime.scores,
        &routing.routing_scores,
        "CPU C ABI route scores",
    )?;
    if cpu_runtime_error > 2.0e-5 {
        return Err(format!(
            "CPU C ABI routing score error {cpu_runtime_error:e} exceeds 2e-5"
        ));
    }
    let gpu = if options.gpu {
        let gpu_runtime = execute_runtime_route(
            &hidden,
            &router_bf16,
            MoeWeightDtype::Bf16,
            tokens,
            hidden_size,
            num_experts,
            top_k,
            gfx1201_runtime_device(options.device)?,
        )?;
        if gpu_runtime.backend != "hip" || !gpu_runtime.arch.starts_with("gfx1201") {
            return Err(format!(
                "--gpu fixture selected non-gfx1201 runtime device {} {}",
                gpu_runtime.backend, gpu_runtime.arch
            ));
        }
        if gpu_runtime.ids != routing.selected_expert_ids
            || gpu_runtime.flags != routing.boundary_tie_flags
        {
            return Err("GPU route result differs from CPU reference".to_string());
        }
        let gpu_error = max_abs(
            &gpu_runtime.scores,
            &routing.routing_scores,
            "GPU route scores",
        )?;
        if gpu_error > 2.0e-5 {
            return Err(format!(
                "GPU fixture routing score error {gpu_error:e} exceeds 2e-5"
            ));
        }
        format!(
            "{{\"runtime_device_index\":{},\"device_id\":{},\"score_max_abs\":{gpu_error:.9e}}}",
            gpu_runtime.runtime_device_index, gpu_runtime.device_id
        )
    } else {
        "null".to_string()
    };
    Ok(format!(
        "{{\"schema\":\"ullm.moe_runtime_verify.v1\",\"mode\":\"hf-route-fixture\",\"passed\":true,\"hf_indices_match_cpu\":{},\"hf_score_max_abs\":{score_error:.9e},\"boundary_tie_detected\":{},\"cpu_runtime\":{{\"runtime_device_index\":{},\"device_id\":{},\"backend\":{:?},\"score_max_abs\":{cpu_runtime_error:.9e}}},\"gpu\":{gpu} }}\n",
        ids_match,
        tie_detected,
        cpu_runtime.runtime_device_index,
        cpu_runtime.device_id,
        cpu_runtime.backend,
    ))
}

fn main() {
    let result = (|| -> Result<(), String> {
        let options = parse_options()?;
        let report = match &options.route_fixture {
            Some(dir) => run_route_fixture(&options, dir)?,
            None => match &options.grouped_gemm_fixture {
                Some(dir) => run_grouped_gemm_fixture(&options, dir, GemmPath::Prefill)?,
                None => match &options.decode_gemm_fixture {
                    Some(dir) => run_grouped_gemm_fixture(&options, dir, GemmPath::Decode)?,
                    None => run_synthetic(&options)?,
                },
            },
        };
        write_report(&options.report, &report)?;
        print!("{report}");
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("moe_runtime_verify: {error}");
        std::process::exit(1);
    }
}
