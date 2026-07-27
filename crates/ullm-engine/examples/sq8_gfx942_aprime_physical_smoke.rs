// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Physical-only `SQ8_0` gfx942 A′ smoke test.
//!
//! Do not run this on the development host.  It deliberately requires one
//! `HIP_VISIBLE_DEVICES` token and an exact gfx942 `gcnArchName`, then performs
//! five short deterministic projections plus a one-wave FNUZ fragment dump.
//! There is no artifact path and no serving integration: its inputs and CPU
//! expectations are constructed locally so a rented MI300X can decide the
//! A′ ABI/scale/fragment questions in a few minutes.

use std::collections::BTreeSet;
use ullm_engine::sq8_gfx942_aprime::{
    SQ8_0_BLOCK_K, SQ8_0_BLOCK_N, Sq8_0FnuzPrepackedMatrix, Sq8_0OcpBlockScaledMatrix,
    infer_sq8_0_fragment_lane_map, sq8_0_fnuz_fragment_probe_fixture,
};
use ullm_runtime_sys::{
    DeviceInfo, RuntimeContext, RuntimeStream, SQ8_GFX942_APRIME_FRAGMENT_A_BYTES,
    SQ8_GFX942_APRIME_FRAGMENT_B_BYTES, SQ8_GFX942_APRIME_FRAGMENT_LANE_F32_BYTES,
    SQ8_GFX942_APRIME_FRAGMENT_MATRIX_F32_BYTES, device_count, device_info,
    sq8_gfx942_aprime_fragment_probe_fnuz as fragment_probe,
    sq8_gfx942_aprime_implementation_for_shape, sq8_gfx942_aprime_is_selected_for_device,
    sq8_gfx942_aprime_projection_buffer_bytes, sq8_gfx942_aprime_projection_fnuz_prepacked_f32,
    sq8_gfx942_control_buffer_bytes, sq8_gfx942_control_dequant_ocp_bf16_projection_f32,
};

const B_CONTROL_ABSOLUTE_TOLERANCE: f32 = 1.0e-5;
const B_CONTROL_RELATIVE_TOLERANCE: f32 = 1.0e-5;
// A′ writes CK's BF16 C workspace before its diagnostic FP32 copy.  The
// comparison therefore has the expected BF16-output allowance; B remains the
// F32-accumulation correctness control above.
const APRIME_ABSOLUTE_TOLERANCE: f32 = 0.125;
const APRIME_RELATIVE_TOLERANCE: f32 = 0.008;

#[derive(Clone, Copy)]
struct ProjectionCase {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
}

const PROJECTION_CASES: [ProjectionCase; 5] = [
    // ID 1, q/k/v/o family with logical M tail.
    ProjectionCase {
        name: "k_or_v_tail_id1",
        m: 1,
        n: 1024,
        k: 5120,
    },
    // ID 1, q/o full-tile family.
    ProjectionCase {
        name: "q_or_o_full_id1",
        m: 16,
        n: 5120,
        k: 5120,
    },
    // ID 2, gate/up KPadding with a logical M tail.
    ProjectionCase {
        name: "gate_or_up_tail_id2",
        m: 1,
        n: 17408,
        k: 5120,
    },
    // ID 3, gate/up full M=128 tile.
    ProjectionCase {
        name: "gate_or_up_full_id3",
        m: 128,
        n: 17408,
        k: 5120,
    },
    // ID 4, down projection logical M tail and K=17408.
    ProjectionCase {
        name: "down_tail_id4",
        m: 1,
        n: 5120,
        k: 17408,
    },
];

struct ProjectionFixture {
    activation_ocp: Vec<u8>,
    activation_scales_f32: Vec<f32>,
    weight_ocp: Vec<u8>,
    weight_scales_f32: Vec<f32>,
    expected_f32: Vec<f32>,
}

#[derive(Debug)]
struct ErrorStats {
    maximum_absolute: f32,
    maximum_relative: f32,
    maximum_index: usize,
}

fn main() -> Result<(), String> {
    let (runtime_index, device) = isolated_exact_gfx942_device()?;
    println!(
        "physical-only SQ8_0 gfx942 A′ smoke: device {} ({}, {})",
        runtime_index, device.name, device.gcn_arch_name
    );

    let mut context = RuntimeContext::create(runtime_index)?;
    let mut stream = context.create_stream()?;
    run_fragment_probe(&mut context, &mut stream)?;
    for case in PROJECTION_CASES {
        run_projection_case(case, &mut context, &mut stream)?;
    }
    println!("SQ8_0 gfx942 A′ physical smoke passed");
    Ok(())
}

fn isolated_exact_gfx942_device() -> Result<(u32, DeviceInfo), String> {
    let visible = std::env::var("HIP_VISIBLE_DEVICES").map_err(|_| {
        "physical SQ8_0 gfx942 smoke requires HIP_VISIBLE_DEVICES to name exactly one device"
            .to_string()
    })?;
    if visible.is_empty() || visible.contains(',') {
        return Err(
            "physical SQ8_0 gfx942 smoke requires exactly one HIP_VISIBLE_DEVICES token"
                .to_string(),
        );
    }
    // The uLLM runtime always exposes a CPU device at index 0 and HIP devices
    // after it, so a single visible GPU yields a total count of 2. Select the
    // unique device that the fail-closed gfx942 selector accepts, and require
    // that exactly one such device exists.
    let total = device_count()?;
    let mut selected: Option<(u32, DeviceInfo)> = None;
    for index in 0..total {
        let candidate = device_info(index)?;
        if sq8_gfx942_aprime_is_selected_for_device(&candidate) {
            if selected.is_some() {
                return Err(
                    "physical SQ8_0 gfx942 smoke requires exactly one selectable gfx942 device"
                        .to_string(),
                );
            }
            selected = Some((index, candidate));
        }
    }
    let (index, device) = selected.ok_or_else(|| {
        format!(
            "physical SQ8_0 gfx942 smoke is fail-closed: no exact HIP gfx942 device among {total} runtime devices"
        )
    })?;
    Ok((index, device))
}

fn run_fragment_probe(
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
) -> Result<(), String> {
    let fixture = sq8_0_fnuz_fragment_probe_fixture()?;
    let mut a = context.alloc_buffer(SQ8_GFX942_APRIME_FRAGMENT_A_BYTES)?;
    let mut b = context.alloc_buffer(SQ8_GFX942_APRIME_FRAGMENT_B_BYTES)?;
    let mut matrix = context.alloc_buffer(SQ8_GFX942_APRIME_FRAGMENT_MATRIX_F32_BYTES)?;
    let mut lanes = context.alloc_buffer(SQ8_GFX942_APRIME_FRAGMENT_LANE_F32_BYTES)?;
    a.copy_from_host(0, &fixture.a_fnuz_16x32_row_major, Some(stream))?;
    b.copy_from_host(0, &fixture.b_fnuz_32x16_column_major, Some(stream))?;
    fragment_probe(&a, &b, &mut matrix, &mut lanes, Some(stream))?;
    stream.synchronize()?;

    let mut matrix_bytes = vec![0_u8; SQ8_GFX942_APRIME_FRAGMENT_MATRIX_F32_BYTES];
    let mut lane_bytes = vec![0_u8; SQ8_GFX942_APRIME_FRAGMENT_LANE_F32_BYTES];
    matrix.copy_to_host(0, &mut matrix_bytes, Some(stream))?;
    lanes.copy_to_host(0, &mut lane_bytes, Some(stream))?;
    stream.synchronize()?;
    let actual_matrix = f32_from_le_bytes(&matrix_bytes)?;
    let actual_lanes = f32_from_le_bytes(&lane_bytes)?;
    let stats = verify_close(
        "FNUZ fragment logical matrix",
        &actual_matrix,
        &fixture.expected_matrix_f32_16x16,
        0.002,
        1.0e-5,
    )?;
    let lane_map = infer_sq8_0_fragment_lane_map(&actual_matrix, &actual_lanes)?;
    let locations: BTreeSet<(usize, usize)> = lane_map
        .iter()
        .map(|entry| (entry.row, entry.column))
        .collect();
    if locations.len() != 16 * 16 {
        return Err(format!(
            "FNUZ fragment raw dump does not cover each logical coordinate exactly once: {} unique",
            locations.len()
        ));
    }
    println!(
        "fragment probe passed: logical max_abs={:.6} max_rel={:.6}; inferred {} lane/register coordinates",
        stats.maximum_absolute,
        stats.maximum_relative,
        lane_map.len()
    );
    Ok(())
}

fn run_projection_case(
    case: ProjectionCase,
    context: &mut RuntimeContext,
    stream: &mut RuntimeStream,
) -> Result<(), String> {
    let fixture = structured_fixture(case)?;
    let activation = Sq8_0OcpBlockScaledMatrix::activation(
        &fixture.activation_ocp,
        &fixture.activation_scales_f32,
        case.m,
        case.k,
    )?;
    let weight = Sq8_0OcpBlockScaledMatrix::weight(
        &fixture.weight_ocp,
        &fixture.weight_scales_f32,
        case.n,
        case.k,
    )?;
    let activation_fnuz = Sq8_0FnuzPrepackedMatrix::from_ocp(activation)?;
    let weight_fnuz = Sq8_0FnuzPrepackedMatrix::from_ocp(weight)?;
    if !fixture.activation_ocp.contains(&0x80) || !fixture.weight_ocp.contains(&0x80) {
        return Err(
            "physical SQ8_0 fixture must exercise OCP negative-zero normalization".to_string(),
        );
    }
    if activation_fnuz.payload.iter().any(|byte| *byte == 0x80)
        || weight_fnuz.payload.iter().any(|byte| *byte == 0x80)
    {
        return Err("FNUZ prepack left an invalid 0x80 byte in the physical fixture".to_string());
    }

    let activation_scale_bytes = f32_to_le_bytes(&fixture.activation_scales_f32);
    let weight_scale_bytes = f32_to_le_bytes(&fixture.weight_scales_f32);
    let activation_fnuz_scale_bytes = f32_to_le_bytes(&activation_fnuz.scales_f32_x2);
    let weight_fnuz_scale_bytes = f32_to_le_bytes(&weight_fnuz.scales_f32_x2);

    let mut activation_ocp_buffer = context.alloc_buffer(fixture.activation_ocp.len())?;
    let mut activation_scale_buffer = context.alloc_buffer(activation_scale_bytes.len())?;
    let mut weight_ocp_buffer = context.alloc_buffer(fixture.weight_ocp.len())?;
    let mut weight_scale_buffer = context.alloc_buffer(weight_scale_bytes.len())?;
    let mut activation_fnuz_buffer = context.alloc_buffer(activation_fnuz.payload.len())?;
    let mut activation_fnuz_scale_buffer =
        context.alloc_buffer(activation_fnuz_scale_bytes.len())?;
    let mut weight_fnuz_buffer = context.alloc_buffer(weight_fnuz.payload.len())?;
    let mut weight_fnuz_scale_buffer = context.alloc_buffer(weight_fnuz_scale_bytes.len())?;

    activation_ocp_buffer.copy_from_host(0, &fixture.activation_ocp, Some(stream))?;
    activation_scale_buffer.copy_from_host(0, &activation_scale_bytes, Some(stream))?;
    weight_ocp_buffer.copy_from_host(0, &fixture.weight_ocp, Some(stream))?;
    weight_scale_buffer.copy_from_host(0, &weight_scale_bytes, Some(stream))?;
    activation_fnuz_buffer.copy_from_host(0, &activation_fnuz.payload, Some(stream))?;
    activation_fnuz_scale_buffer.copy_from_host(0, &activation_fnuz_scale_bytes, Some(stream))?;
    weight_fnuz_buffer.copy_from_host(0, &weight_fnuz.payload, Some(stream))?;
    weight_fnuz_scale_buffer.copy_from_host(0, &weight_fnuz_scale_bytes, Some(stream))?;

    let (aprime_workspace_bytes, aprime_output_bytes) =
        sq8_gfx942_aprime_projection_buffer_bytes(case.m, case.n)?;
    let (control_activation_bf16_bytes, control_weight_bf16_bytes, control_output_bytes) =
        sq8_gfx942_control_buffer_bytes(case.m, case.n, case.k)?;
    let mut aprime_workspace = context.alloc_buffer(aprime_workspace_bytes)?;
    let mut aprime_output = context.alloc_buffer(aprime_output_bytes)?;
    let mut control_activation_bf16 = context.alloc_buffer(control_activation_bf16_bytes)?;
    let mut control_weight_bf16 = context.alloc_buffer(control_weight_bf16_bytes)?;
    let mut control_output = context.alloc_buffer(control_output_bytes)?;

    let expected_implementation =
        sq8_gfx942_aprime_implementation_for_shape(case.m, case.n, case.k).ok_or_else(|| {
            format!(
                "physical case {} is outside the A′ dispatch table",
                case.name
            )
        })?;
    let selected_implementation = sq8_gfx942_aprime_projection_fnuz_prepacked_f32(
        &activation_fnuz_buffer,
        &activation_fnuz_scale_buffer,
        &weight_fnuz_buffer,
        &weight_fnuz_scale_buffer,
        case.m,
        case.n,
        case.k,
        &mut aprime_workspace,
        &mut aprime_output,
        Some(stream),
    )?;
    // Optional timed repeat of the A' projection only (correctness already
    // verified above). Enabled with ULLM_SMOKE_TIME_REPEATS=<n>.
    if let Ok(reps) = std::env::var("ULLM_SMOKE_TIME_REPEATS") {
        let reps: u32 = reps.parse().unwrap_or(0);
        if reps > 0 {
            // warmup
            for _ in 0..3 {
                sq8_gfx942_aprime_projection_fnuz_prepacked_f32(
                    &activation_fnuz_buffer, &activation_fnuz_scale_buffer,
                    &weight_fnuz_buffer, &weight_fnuz_scale_buffer,
                    case.m, case.n, case.k,
                    &mut aprime_workspace, &mut aprime_output, Some(stream),
                )?;
            }
            stream.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                sq8_gfx942_aprime_projection_fnuz_prepacked_f32(
                    &activation_fnuz_buffer, &activation_fnuz_scale_buffer,
                    &weight_fnuz_buffer, &weight_fnuz_scale_buffer,
                    case.m, case.n, case.k,
                    &mut aprime_workspace, &mut aprime_output, Some(stream),
                )?;
            }
            stream.synchronize()?;
            let el = t0.elapsed().as_secs_f64();
            let per = el / reps as f64;
            let flop = 2.0 * case.m as f64 * case.n as f64 * case.k as f64;
            let bytes = (case.n as f64 * case.k as f64) + (case.m as f64 * case.k as f64);
            println!(
                "TIMING {} M/N/K={}/{}/{} reps={} per_call_ms={:.6} TFLOPS={:.3} weight_GB/s={:.1}",
                case.name, case.m, case.n, case.k, reps,
                per * 1e3, flop / per / 1e12, bytes / per / 1e9
            );
        }
    }
    if selected_implementation != expected_implementation {
        return Err(format!(
            "A′ physical case {} selected {selected_implementation:?}; expected {expected_implementation:?}",
            case.name
        ));
    }
    sq8_gfx942_control_dequant_ocp_bf16_projection_f32(
        &activation_ocp_buffer,
        &activation_scale_buffer,
        &weight_ocp_buffer,
        &weight_scale_buffer,
        case.m,
        case.n,
        case.k,
        &mut control_activation_bf16,
        &mut control_weight_bf16,
        &mut control_output,
        Some(stream),
    )?;
    stream.synchronize()?;

    let mut aprime_bytes = vec![0_u8; aprime_output_bytes];
    let mut control_bytes = vec![0_u8; control_output_bytes];
    aprime_output.copy_to_host(0, &mut aprime_bytes, Some(stream))?;
    control_output.copy_to_host(0, &mut control_bytes, Some(stream))?;
    stream.synchronize()?;
    let aprime = f32_from_le_bytes(&aprime_bytes)?;
    let control = f32_from_le_bytes(&control_bytes)?;
    // The B control path is an independent dequant-to-BF16 reference. When it is
    // known-broken we still want the A' verdict, so allow skipping only the B
    // comparison. A' is never skipped.
    let skip_b = std::env::var("ULLM_SMOKE_SKIP_B_CONTROL").is_ok();
    let control_stats = if skip_b {
        eprintln!("{} B control SKIPPED by ULLM_SMOKE_SKIP_B_CONTROL", case.name);
        verify_close(
            &format!("{} B OCP-to-BF16 control (skipped)", case.name),
            &control,
            &control,
            B_CONTROL_ABSOLUTE_TOLERANCE,
            B_CONTROL_RELATIVE_TOLERANCE,
        )?
    } else {
        verify_close(
            &format!("{} B OCP-to-BF16 control", case.name),
            &control,
            &fixture.expected_f32,
            B_CONTROL_ABSOLUTE_TOLERANCE,
            B_CONTROL_RELATIVE_TOLERANCE,
        )?
    };
    if case.name == "k_or_v_tail_id1" {
        println!(
            "B control hardware layout sentinel: first={:.5} expected={:.5}",
            control[0], fixture.expected_f32[0]
        );
    }
    let aprime_stats = verify_close(
        &format!("{} A′ FNUZ/CK", case.name),
        &aprime,
        &fixture.expected_f32,
        APRIME_ABSOLUTE_TOLERANCE,
        APRIME_RELATIVE_TOLERANCE,
    )?;
    // A'-versus-B is only meaningful when B itself is trusted.
    let pair_stats = if skip_b {
        verify_close(
            &format!("{} A′ versus B (skipped)", case.name),
            &aprime,
            &aprime,
            APRIME_ABSOLUTE_TOLERANCE,
            APRIME_RELATIVE_TOLERANCE,
        )?
    } else {
        verify_close(
            &format!("{} A′ versus B", case.name),
            &aprime,
            &control,
            APRIME_ABSOLUTE_TOLERANCE,
            APRIME_RELATIVE_TOLERANCE,
        )?
    };
    println!(
        "{} ({:?}, M/N/K={}/{}/{}): B max_abs={:.6} max_rel={:.6}; A′ max_abs={:.6} max_rel={:.6}; A′-B max_abs={:.6} max_rel={:.6}",
        case.name,
        selected_implementation,
        case.m,
        case.n,
        case.k,
        control_stats.maximum_absolute,
        control_stats.maximum_relative,
        aprime_stats.maximum_absolute,
        aprime_stats.maximum_relative,
        pair_stats.maximum_absolute,
        pair_stats.maximum_relative,
    );
    Ok(())
}

/// Constructs a sparse, two-K-block fixture and its CPU-computed analytic
/// expectation.  The physical kernels still traverse each real model shape,
/// while host expectation construction stays O(M*N + M*K/128 + N*K/16384)
/// instead of O(M*N*K).  Both first and final K128 blocks are nonzero, so a
/// K-block stride or tail assumption cannot hide behind an all-zero payload.
fn structured_fixture(case: ProjectionCase) -> Result<ProjectionFixture, String> {
    if case.m == 0
        || case.n == 0
        || !case.n.is_multiple_of(SQ8_0_BLOCK_N)
        || !case.k.is_multiple_of(SQ8_0_BLOCK_K)
    {
        return Err(format!("invalid SQ8_0 physical shape for {}", case.name));
    }
    let k_blocks = case.k / SQ8_0_BLOCK_K;
    if k_blocks < 2 {
        return Err("SQ8_0 physical fixture requires at least two K128 blocks".to_string());
    }
    let n_blocks = case.n / SQ8_0_BLOCK_N;
    let final_k = (k_blocks - 1) * SQ8_0_BLOCK_K;
    // Negative OCP zero deliberately fills unused positions.  The A′ prepack
    // must normalize every such byte to FNUZ +0 before CK sees it.
    let mut activation_ocp = vec![0x80_u8; case.m * case.k];
    let mut weight_ocp = vec![0x80_u8; case.n * case.k];
    let mut activation_scales_f32 = vec![0.0_f32; case.m * k_blocks];
    let mut weight_scales_f32 = vec![0.0_f32; n_blocks * k_blocks];

    for row in 0..case.m {
        for k_block in 0..k_blocks {
            activation_scales_f32[row * k_blocks + k_block] = scale_pattern(row + k_block);
        }
        let code = nonzero_ocp_code(row);
        activation_ocp[row * case.k] = code;
        activation_ocp[row * case.k + final_k] = code;
    }
    for n_block in 0..n_blocks {
        for k_block in 0..k_blocks {
            weight_scales_f32[n_block * k_blocks + k_block] =
                scale_pattern(2 * n_block + 3 * k_block + 1);
        }
        for column in n_block * SQ8_0_BLOCK_N..(n_block + 1) * SQ8_0_BLOCK_N {
            let code = nonzero_ocp_code(column);
            weight_ocp[column * case.k] = code;
            weight_ocp[column * case.k + final_k] = code;
        }
    }

    let mut expected_f32 = vec![0.0_f32; case.m * case.n];
    for row in 0..case.m {
        let a_code = nonzero_ocp_value(row);
        for n_block in 0..n_blocks {
            let block_sum = activation_scales_f32[row * k_blocks]
                * weight_scales_f32[n_block * k_blocks]
                + activation_scales_f32[row * k_blocks + k_blocks - 1]
                    * weight_scales_f32[n_block * k_blocks + k_blocks - 1];
            for column in n_block * SQ8_0_BLOCK_N..(n_block + 1) * SQ8_0_BLOCK_N {
                expected_f32[row * case.n + column] =
                    a_code * nonzero_ocp_value(column) * block_sum;
            }
        }
    }
    Ok(ProjectionFixture {
        activation_ocp,
        activation_scales_f32,
        weight_ocp,
        weight_scales_f32,
        expected_f32,
    })
}

fn nonzero_ocp_code(index: usize) -> u8 {
    match index % 4 {
        0 => 0x30, // 0.5
        1 => 0x38, // 1.0
        2 => 0x40, // 2.0
        _ => 0x48, // 4.0
    }
}

fn nonzero_ocp_value(index: usize) -> f32 {
    match index % 4 {
        0 => 0.5,
        1 => 1.0,
        2 => 2.0,
        _ => 4.0,
    }
}

fn scale_pattern(index: usize) -> f32 {
    match index % 4 {
        0 => 0.25,
        1 => 0.5,
        2 => 1.0,
        _ => 2.0,
    }
}

fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f32_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(format!(
            "F32 byte payload has invalid length {}",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn verify_close(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> Result<ErrorStats, String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: output length {} differs from expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut stats = ErrorStats {
        maximum_absolute: 0.0,
        maximum_relative: 0.0,
        maximum_index: 0,
    };
    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        if !observed.is_finite() {
            return Err(format!("{label}: non-finite output at {index}: {observed}"));
        }
        let absolute = (observed - reference).abs();
        let relative = absolute / reference.abs().max(1.0);
        if absolute > stats.maximum_absolute {
            stats.maximum_absolute = absolute;
            stats.maximum_relative = relative;
            stats.maximum_index = index;
        }
        if relative > stats.maximum_relative {
            stats.maximum_relative = relative;
        }
        if absolute > absolute_tolerance && relative > relative_tolerance {
            return Err(format!(
                "{label}: mismatch at {index}: observed={observed}, expected={reference}, abs={absolute}, rel={relative}, allowed abs={absolute_tolerance}, rel={relative_tolerance}"
            ));
        }
    }
    let _ = stats.maximum_index;
    Ok(stats)
}
