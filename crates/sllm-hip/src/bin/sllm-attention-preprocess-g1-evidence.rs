//! Bounded semantic G1 evidence for the public C3a1 attention-preprocess path.
//!
//! This runner performs no CPU fallback. Its CPU work is an independent scalar
//! oracle used only after the owned HIP output readbacks have completed.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, AttentionPreprocessContract, AttentionPreprocessPositionMode, Backend,
    BoundSemanticOp, DType, DispatchEvidence, ExecutionSessionRequest, ExecutionState,
    SemanticOpDescriptor, TensorView,
};
use sllm_hip::HipBackend;

const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);
const HIP_BACKEND: u32 = 1;
const ATTENTION_KERNEL_ID: u32 = 1;
const WAVE32_KERNEL_ID: u32 = 2;
const WORKGROUP_SIZE: u32 = 1;
const WAVE32_WORKGROUP_SIZE: u32 = 32;
const Q_HEADS: usize = 16;
const K_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const QGATE_HEAD_WIDTH: usize = 512;
const ROTARY_DIM: usize = 64;
const EPSILON: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10_000_000.0;
const FINITE_BF16_ULP_BOUND: u16 = 2;
const KERNEL_SYMBOL: &str = "attention_preprocess.headwise_norm_rope.v1";
const DEVICE_SYMBOL: &str = "sllm_attention_preprocess_headwise_norm_rope_v1";
const WAVE32_KERNEL_SYMBOL: &str = "attention_preprocess.headwise_norm_rope.wave32.v1";
const WAVE32_DEVICE_SYMBOL: &str = "sllm_attention_preprocess_headwise_norm_rope_wave32_v1";

fn default_on_env(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|value| value == "1")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Case {
    m: usize,
    start_position: u32,
    mode: AttentionPreprocessPositionMode,
}

// This is deliberately non-Cartesian. It includes prefill reset, decode
// continuation, every requested M, and all requested absolute boundaries.
const CASES: [Case; 8] = [
    Case {
        m: 1,
        start_position: 0,
        mode: AttentionPreprocessPositionMode::Prefill,
    },
    Case {
        m: 3,
        start_position: 0,
        mode: AttentionPreprocessPositionMode::Prefill,
    },
    Case {
        m: 17,
        start_position: 0,
        mode: AttentionPreprocessPositionMode::Prefill,
    },
    Case {
        m: 1,
        start_position: 1,
        mode: AttentionPreprocessPositionMode::DecodeContinuation,
    },
    Case {
        m: 3,
        start_position: 3,
        mode: AttentionPreprocessPositionMode::DecodeContinuation,
    },
    Case {
        m: 17,
        start_position: 255,
        mode: AttentionPreprocessPositionMode::DecodeContinuation,
    },
    Case {
        m: 1,
        start_position: 256,
        mode: AttentionPreprocessPositionMode::DecodeContinuation,
    },
    Case {
        m: 3,
        start_position: 257,
        mode: AttentionPreprocessPositionMode::DecodeContinuation,
    },
];

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
}

#[derive(Default)]
struct ComparisonObservation {
    max_finite_ulp: u16,
    max_finite_abs_error: f64,
}

#[derive(Serialize)]
struct CaseEvidence {
    m: usize,
    start_position: u32,
    last_position: u32,
    position_mode: &'static str,
    dispatch_id: u64,
    dispatch_count: u32,
    kernel_id: u32,
    workgroup_size_x: u32,
    grid_size_x: u32,
    row_count: u64,
    normalized_size: u64,
    kernel_symbol: String,
    device_symbol: String,
    exact_gate_match: bool,
    exact_qk_classification_sign: bool,
    max_finite_ulp: u16,
    max_finite_abs_error: f64,
}

#[derive(Serialize)]
struct ComparisonReport {
    finite_policy: &'static str,
    finite_bf16_ulp_bound: u16,
    special_policy: &'static str,
    gate_policy: &'static str,
    max_observed_finite_ulp: u16,
    max_observed_finite_abs_error: f64,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    pass: bool,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    cpu_fallback_used: bool,
    fallback: bool,
    fallback_allowed: bool,
    fallback_used: bool,
    rounding_order_changed: bool,
    operations: usize,
    dispatch_count: usize,
    kernel_dispatches: usize,
    comparison: ComparisonReport,
    cases: Vec<CaseEvidence>,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

fn parse_config() -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--device-index" => {
                if device_index.is_some() {
                    return Err("duplicate --device-index".to_owned());
                }
                device_index = Some(
                    arguments
                        .next()
                        .ok_or_else(|| "--device-index requires a value".to_owned())?
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
            }
            "--target" => {
                if target.is_some() {
                    return Err("duplicate --target".to_owned());
                }
                let value = arguments
                    .next()
                    .ok_or_else(|| "--target requires a value".to_owned())?;
                if !matches!(value.as_str(), "gfx1030" | "gfx1201" | "gfx942") {
                    return Err(
                        "--target must be the exact gfx1030, gfx1201, or gfx942 target".to_owned(),
                    );
                }
                target = Some(value);
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
    })
}

/// Convert binary32 to BF16 with the exact native round-to-nearest-even rule.
fn float_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        if bits & 0x007f_ffff != 0 {
            let sign = ((bits >> 16) as u16) & 0x8000;
            let payload = ((bits >> 16) as u16) & 0x003f;
            return sign | 0x7fc0 | payload;
        }
        return (bits >> 16) as u16;
    }
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

fn bf16_to_float(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn words_to_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn i32_to_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u16>, String> {
    let chunks = bytes.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err("BF16 readback has an odd byte count".to_owned());
    }
    Ok(chunks
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

const SPECIAL_VALUES: [u16; 7] = [
    0x0000, // +0
    0x8000, // -0
    0x0001, // smallest positive BF16 subnormal
    0x7f00, // large finite value
    0x7fc1, // positive NaN
    0x7f80, // positive infinity
    0xff80, // negative infinity
];

fn generated_word(index: usize, seed: usize) -> u16 {
    let value =
        ((index.wrapping_mul(37).wrapping_add(seed.wrapping_mul(13)) % 251) as f32 - 125.0) / 32.0;
    float_to_bf16_rne(value)
}

fn inject_specials(values: &mut [u16], seed: usize) {
    for (index, value) in SPECIAL_VALUES.iter().copied().enumerate() {
        values[index] = if index == 6 && seed % 2 == 1 {
            0xff80
        } else {
            value
        };
    }
}

fn make_packed_q_gate(m: usize, case_index: usize) -> Vec<u16> {
    let mut packed = vec![0_u16; m * Q_HEADS * QGATE_HEAD_WIDTH];
    for row in 0..m {
        for head in 0..Q_HEADS {
            let offset = (row * Q_HEADS + head) * QGATE_HEAD_WIDTH;
            for dim in 0..HEAD_DIM {
                packed[offset + dim] = generated_word(
                    row * Q_HEADS * HEAD_DIM + head * HEAD_DIM + dim,
                    case_index + 1,
                );
                packed[offset + HEAD_DIM + dim] = generated_word(
                    row * Q_HEADS * HEAD_DIM + head * HEAD_DIM + dim,
                    case_index + 101,
                );
            }
        }
    }
    inject_specials(&mut packed, case_index);
    packed
}

fn make_k(m: usize, case_index: usize) -> Vec<u16> {
    let mut values = (0..m * K_HEADS * HEAD_DIM)
        .map(|index| generated_word(index, case_index + 211))
        .collect::<Vec<_>>();
    inject_specials(&mut values, case_index + 2);
    values
}

fn make_scale(heads: usize, case_index: usize) -> Vec<u16> {
    let mut values = (0..heads * HEAD_DIM)
        .map(|index| generated_word(index, case_index + 307))
        .collect::<Vec<_>>();
    inject_specials(&mut values, case_index + 3);
    values
}

fn positions(case: Case) -> Vec<i32> {
    (0..case.m)
        .map(|index| case.start_position as i32 + index as i32)
        .collect()
}

fn rotate_neox(values: &mut [f32], position: i32) {
    for pair in 0..ROTARY_DIM / 2 {
        let exponent = -((2 * pair) as f32) / ROTARY_DIM as f32;
        let angle = position as f32 * ROPE_THETA.powf(exponent);
        let cosine = angle.cos();
        let sine = angle.sin();
        let first = values[pair];
        let second = values[pair + 32];
        values[pair] = first * cosine - second * sine;
        values[pair + 32] = first * sine + second * cosine;
    }
}

/// Scalar oracle with the frozen operation order: ascending head dimensions,
/// FP32 RMSNorm sum, epsilon, offset-one scale, BF16 boundary, then partial
/// NeoX rotation and a final BF16 boundary.
fn scalar_head_oracle(input: &[u16], raw_scale: &[u16], position: i32) -> Vec<u16> {
    assert_eq!(input.len(), HEAD_DIM);
    assert_eq!(raw_scale.len(), HEAD_DIM);
    let mut sum = 0.0_f32;
    for value in input {
        let value = bf16_to_float(*value);
        sum += value * value;
    }
    let inverse_rms = 1.0_f32 / (sum / HEAD_DIM as f32 + EPSILON).sqrt();
    let mut values = Vec::with_capacity(HEAD_DIM);
    for (value, raw) in input.iter().zip(raw_scale) {
        let normalized = bf16_to_float(*value) * inverse_rms;
        let effective_scale = 1.0_f32 + bf16_to_float(*raw);
        values.push(bf16_to_float(float_to_bf16_rne(
            normalized * effective_scale,
        )));
    }
    rotate_neox(&mut values, position);
    values.into_iter().map(float_to_bf16_rne).collect()
}

fn scalar_attention_oracle(
    case: Case,
    packed_q_gate: &[u16],
    k: &[u16],
    q_raw_scale: &[u16],
    k_raw_scale: &[u16],
    position_values: &[i32],
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut q_output = Vec::with_capacity(case.m * Q_HEADS * HEAD_DIM);
    let mut gate_output = Vec::with_capacity(case.m * Q_HEADS * HEAD_DIM);
    let mut k_output = Vec::with_capacity(case.m * K_HEADS * HEAD_DIM);
    for (row, &position) in position_values.iter().enumerate() {
        for head in 0..Q_HEADS {
            let packed_offset = (row * Q_HEADS + head) * QGATE_HEAD_WIDTH;
            let q_offset = head * HEAD_DIM;
            q_output.extend(scalar_head_oracle(
                &packed_q_gate[packed_offset..packed_offset + HEAD_DIM],
                &q_raw_scale[q_offset..q_offset + HEAD_DIM],
                position,
            ));
            gate_output.extend_from_slice(
                &packed_q_gate[packed_offset + HEAD_DIM..packed_offset + QGATE_HEAD_WIDTH],
            );
        }
        for head in 0..K_HEADS {
            let input_offset = (row * K_HEADS + head) * HEAD_DIM;
            let scale_offset = head * HEAD_DIM;
            k_output.extend(scalar_head_oracle(
                &k[input_offset..input_offset + HEAD_DIM],
                &k_raw_scale[scale_offset..scale_offset + HEAD_DIM],
                position,
            ));
        }
    }
    (q_output, gate_output, k_output)
}

fn ordered_bf16(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        (!bits) as i32
    } else {
        (bits | 0x8000) as i32
    }
}

fn finite_bf16_ulp_distance(expected: u16, actual: u16) -> u16 {
    (ordered_bf16(expected) - ordered_bf16(actual)).unsigned_abs() as u16
}

fn compare_bf16_words(
    expected: &[u16],
    actual: &[u16],
    label: &str,
    observation: &mut ComparisonObservation,
) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "{label} length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        ));
    }
    for (index, (&expected_bits, &actual_bits)) in expected.iter().zip(actual).enumerate() {
        let expected_value = bf16_to_float(expected_bits);
        let actual_value = bf16_to_float(actual_bits);
        if expected_value.is_nan() {
            if !actual_value.is_nan() || (expected_bits ^ actual_bits) & 0x8000 != 0 {
                return Err(format!(
                    "{label}[{index}] NaN classification/sign mismatch: expected 0x{expected_bits:04x}, got 0x{actual_bits:04x}"
                ));
            }
            continue;
        }
        if expected_value.is_infinite() {
            if actual_bits != expected_bits {
                return Err(format!(
                    "{label}[{index}] infinity classification/sign mismatch: expected 0x{expected_bits:04x}, got 0x{actual_bits:04x}"
                ));
            }
            continue;
        }
        if expected_value == 0.0 {
            if actual_value != 0.0 || (expected_bits ^ actual_bits) & 0x8000 != 0 {
                return Err(format!(
                    "{label}[{index}] zero classification/sign mismatch: expected 0x{expected_bits:04x}, got 0x{actual_bits:04x}"
                ));
            }
            continue;
        }
        if !actual_value.is_finite() {
            return Err(format!(
                "{label}[{index}] finite classification mismatch: expected 0x{expected_bits:04x}, got 0x{actual_bits:04x}"
            ));
        }
        let ulp = finite_bf16_ulp_distance(expected_bits, actual_bits);
        observation.max_finite_ulp = observation.max_finite_ulp.max(ulp);
        observation.max_finite_abs_error = observation
            .max_finite_abs_error
            .max(f64::from((actual_value - expected_value).abs()));
        if ulp > FINITE_BF16_ULP_BOUND {
            return Err(format!(
                "{label}[{index}] finite BF16 error is {ulp} ULPs, bound is {FINITE_BF16_ULP_BOUND}: expected 0x{expected_bits:04x}, got 0x{actual_bits:04x}"
            ));
        }
    }
    Ok(())
}

fn wait_success(
    state: Result<ExecutionState, sllm_core::ExecutionError>,
    label: &str,
) -> Result<(), String> {
    match state.map_err(|error| format!("{label} failed: {error}"))? {
        ExecutionState::Success => Ok(()),
        ExecutionState::Pending => Err(format!("{label} remained pending")),
        ExecutionState::Failure => Err(format!("{label} reported failure")),
    }
}

fn validate_dispatch(dispatch: &DispatchEvidence, case: Case, target: &str) -> Result<(), String> {
    let grid_size = case
        .m
        .checked_mul(Q_HEADS + K_HEADS)
        .ok_or_else(|| "attention preprocess grid size overflowed usize".to_owned())?;
    let grid_size = u32::try_from(grid_size)
        .map_err(|_| "attention preprocess grid size does not fit u32".to_owned())?;
    let wave32 = target == "gfx1030"
        && default_on_env(env::var_os("SLLM_ATTENTION_PREPROCESS_GFX1030_WAVE32").as_deref());
    let expected_kernel_id = if wave32 {
        WAVE32_KERNEL_ID
    } else {
        ATTENTION_KERNEL_ID
    };
    let expected_workgroup = if wave32 {
        WAVE32_WORKGROUP_SIZE
    } else {
        WORKGROUP_SIZE
    };
    let expected_kernel_symbol = if wave32 {
        WAVE32_KERNEL_SYMBOL
    } else {
        KERNEL_SYMBOL
    };
    let expected_device_symbol = if wave32 {
        WAVE32_DEVICE_SYMBOL
    } else {
        DEVICE_SYMBOL
    };
    if dispatch.abi_version != 1
        || dispatch.info_version != 1
        || dispatch.dispatch_id == 0
        || dispatch.dispatch_count != 1
        || dispatch.kernel_id != expected_kernel_id
        || dispatch.workgroup_size_x != expected_workgroup
        || dispatch.grid_size_x != grid_size
        || dispatch.row_count != case.m as u64
        || dispatch.normalized_size != HEAD_DIM as u64
        || dispatch.backend != HIP_BACKEND
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != expected_kernel_symbol
        || dispatch.device_symbol != expected_device_symbol
        || dispatch.target != target
    {
        return Err(format!(
            "attention preprocess dispatch metadata violated the exact contract for M={} start={} target={target}",
            case.m, case.start_position
        ));
    }
    Ok(())
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    case: Case,
    case_index: usize,
    target: &str,
) -> Result<(CaseEvidence, ComparisonObservation), String> {
    let packed_q_gate = make_packed_q_gate(case.m, case_index);
    let k = make_k(case.m, case_index);
    let q_raw_scale = make_scale(Q_HEADS, case_index);
    let k_raw_scale = make_scale(K_HEADS, case_index + 17);
    let position_values = positions(case);
    let (expected_q, expected_gate, expected_k) = scalar_attention_oracle(
        case,
        &packed_q_gate,
        &k,
        &q_raw_scale,
        &k_raw_scale,
        &position_values,
    );

    let packed_buffer = session
        .allocate((packed_q_gate.len() * 2) as u64)
        .map_err(|error| format!("packed Q/gate allocation failed: {error}"))?;
    let k_buffer = session
        .allocate((k.len() * 2) as u64)
        .map_err(|error| format!("K allocation failed: {error}"))?;
    let q_scale_buffer = session
        .allocate((q_raw_scale.len() * 2) as u64)
        .map_err(|error| format!("Q raw scale allocation failed: {error}"))?;
    let k_scale_buffer = session
        .allocate((k_raw_scale.len() * 2) as u64)
        .map_err(|error| format!("K raw scale allocation failed: {error}"))?;
    let positions_buffer = session
        .allocate((position_values.len() * 4) as u64)
        .map_err(|error| format!("positions allocation failed: {error}"))?;
    let q_output_buffer = session
        .allocate((expected_q.len() * 2) as u64)
        .map_err(|error| format!("Q output allocation failed: {error}"))?;
    let gate_output_buffer = session
        .allocate((expected_gate.len() * 2) as u64)
        .map_err(|error| format!("gate output allocation failed: {error}"))?;
    let k_output_buffer = session
        .allocate((expected_k.len() * 2) as u64)
        .map_err(|error| format!("K output allocation failed: {error}"))?;

    let upload_data = [
        (
            "packed Q/gate",
            &packed_buffer,
            words_to_bytes(&packed_q_gate),
        ),
        ("K", &k_buffer, words_to_bytes(&k)),
        ("Q raw scale", &q_scale_buffer, words_to_bytes(&q_raw_scale)),
        ("K raw scale", &k_scale_buffer, words_to_bytes(&k_raw_scale)),
    ];
    for (label, buffer, bytes) in upload_data {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|error| format!("{label} H2D failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), &format!("{label} H2D"))?;
    }
    let position_bytes = i32_to_bytes(&position_values);
    let mut position_upload = session
        .upload(
            queue,
            positions_buffer
                .range(0, position_bytes.len() as u64)
                .map_err(|error| error.to_string())?,
            Arc::<[u8]>::from(position_bytes),
        )
        .map_err(|error| format!("positions H2D failed: {error}"))?;
    wait_success(position_upload.wait(WAIT_TIMEOUT), "positions H2D")?;

    let packed_view = TensorView::contiguous(DType::Bf16, &[case.m, Q_HEADS, QGATE_HEAD_WIDTH])
        .map_err(|error| format!("packed Q/gate tensor view failed: {error}"))?;
    let k_view = TensorView::contiguous(DType::Bf16, &[case.m, K_HEADS, HEAD_DIM])
        .map_err(|error| format!("K tensor view failed: {error}"))?;
    let q_scale_view = TensorView::contiguous(DType::Bf16, &[Q_HEADS, HEAD_DIM])
        .map_err(|error| format!("Q scale tensor view failed: {error}"))?;
    let k_scale_view = TensorView::contiguous(DType::Bf16, &[K_HEADS, HEAD_DIM])
        .map_err(|error| format!("K scale tensor view failed: {error}"))?;
    let positions_view = TensorView::contiguous(DType::I32, &[case.m])
        .map_err(|error| format!("positions tensor view failed: {error}"))?;
    let q_output_view = TensorView::contiguous(DType::Bf16, &[case.m, Q_HEADS, HEAD_DIM])
        .map_err(|error| format!("Q output tensor view failed: {error}"))?;
    let gate_output_view = TensorView::contiguous(DType::Bf16, &[case.m, Q_HEADS, HEAD_DIM])
        .map_err(|error| format!("gate output tensor view failed: {error}"))?;
    let k_output_view = TensorView::contiguous(DType::Bf16, &[case.m, K_HEADS, HEAD_DIM])
        .map_err(|error| format!("K output tensor view failed: {error}"))?;
    let contract = AttentionPreprocessContract::new_qwen3_5(
        case.mode,
        i64::from(case.start_position),
        case.m as u64,
    )
    .map_err(|error| format!("attention preprocess contract failed: {error}"))?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new_attention_preprocess(
            vec![
                packed_view.clone(),
                k_view.clone(),
                q_scale_view.clone(),
                k_scale_view.clone(),
                positions_view.clone(),
            ],
            vec![
                q_output_view.clone(),
                gate_output_view.clone(),
                k_output_view.clone(),
            ],
            contract,
        )
        .map_err(|error| format!("attention preprocess semantic descriptor failed: {error}"))?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&packed_buffer, packed_view, AccessMode::Read)
                    .map_err(|error| format!("packed Q/gate binding failed: {error}"))?,
                session
                    .bind(&k_buffer, k_view, AccessMode::Read)
                    .map_err(|error| format!("K binding failed: {error}"))?,
                session
                    .bind(&q_scale_buffer, q_scale_view, AccessMode::Read)
                    .map_err(|error| format!("Q raw scale binding failed: {error}"))?,
                session
                    .bind(&k_scale_buffer, k_scale_view, AccessMode::Read)
                    .map_err(|error| format!("K raw scale binding failed: {error}"))?,
                session
                    .bind(&positions_buffer, positions_view, AccessMode::Read)
                    .map_err(|error| format!("positions binding failed: {error}"))?,
            ],
            vec![
                session
                    .bind(&q_output_buffer, q_output_view, AccessMode::Write)
                    .map_err(|error| format!("Q output binding failed: {error}"))?,
                session
                    .bind(&gate_output_buffer, gate_output_view, AccessMode::Write)
                    .map_err(|error| format!("gate output binding failed: {error}"))?,
                session
                    .bind(&k_output_buffer, k_output_view, AccessMode::Write)
                    .map_err(|error| format!("K output binding failed: {error}"))?,
            ],
        )
        .map_err(|error| format!("owned attention preprocess binding failed: {error}"))?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| format!("attention preprocess prepare failed: {error}"))?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| format!("attention preprocess submit failed: {error}"))?;
    validate_dispatch(submission.dispatch(), case, target)?;
    wait_success(
        submission.wait(WAIT_TIMEOUT),
        "attention preprocess completion",
    )?;
    let dispatch = submission.dispatch().clone();

    let mut readbacks = [
        submission
            .start_output_readback(0)
            .map_err(|error| format!("Q output readback start failed: {error}"))?,
        submission
            .start_output_readback(1)
            .map_err(|error| format!("gate output readback start failed: {error}"))?,
        submission
            .start_output_readback(2)
            .map_err(|error| format!("K output readback start failed: {error}"))?,
    ];
    let expected_bytes = [
        expected_q.len() * 2,
        expected_gate.len() * 2,
        expected_k.len() * 2,
    ];
    let mut actual_bytes = [Vec::new(), Vec::new(), Vec::new()];
    for (index, readback) in readbacks.iter_mut().enumerate() {
        wait_success(
            readback.wait(WAIT_TIMEOUT),
            match index {
                0 => "Q output D2H",
                1 => "gate output D2H",
                _ => "K output D2H",
            },
        )?;
        let mut actual = vec![0_u8; expected_bytes[index]];
        let written = readback
            .read_into(&mut actual)
            .map_err(|error| format!("output {index} read failed: {error}"))?;
        if written != expected_bytes[index] as u64 {
            return Err(format!(
                "output {index} byte count mismatch: expected {}, got {written}",
                expected_bytes[index]
            ));
        }
        actual_bytes[index] = actual;
    }
    let actual_q = bytes_to_words(&actual_bytes[0])?;
    let actual_k = bytes_to_words(&actual_bytes[2])?;
    if words_to_bytes(&expected_gate) != actual_bytes[1] {
        return Err(format!(
            "gate output byte mismatch for M={} start={}",
            case.m, case.start_position
        ));
    }
    let mut observation = ComparisonObservation::default();
    compare_bf16_words(&expected_q, &actual_q, "Q output", &mut observation)?;
    compare_bf16_words(&expected_k, &actual_k, "K output", &mut observation)?;
    Ok((
        CaseEvidence {
            m: case.m,
            start_position: case.start_position,
            last_position: case.start_position + case.m as u32 - 1,
            position_mode: match case.mode {
                AttentionPreprocessPositionMode::Prefill => "prefill",
                AttentionPreprocessPositionMode::DecodeContinuation => "decode_continuation",
            },
            dispatch_id: dispatch.dispatch_id,
            dispatch_count: dispatch.dispatch_count,
            kernel_id: dispatch.kernel_id,
            workgroup_size_x: dispatch.workgroup_size_x,
            grid_size_x: dispatch.grid_size_x,
            row_count: dispatch.row_count,
            normalized_size: dispatch.normalized_size,
            kernel_symbol: dispatch.kernel_symbol,
            device_symbol: dispatch.device_symbol,
            exact_gate_match: true,
            exact_qk_classification_sign: true,
            max_finite_ulp: observation.max_finite_ulp,
            max_finite_abs_error: observation.max_finite_abs_error,
        },
        observation,
    ))
}

fn run(config: &Config) -> Result<Report, String> {
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("invalid execution-session request: {error}"))?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| format!("owned HIP execution-session open failed: {error}"))?;
    let result: Result<(Vec<CaseEvidence>, ComparisonObservation), String> = (|| {
        let queue = session
            .create_queue()
            .map_err(|error| format!("queue creation failed: {error}"))?;
        let mut cases = Vec::with_capacity(CASES.len());
        let mut observation = ComparisonObservation::default();
        for (case_index, case) in CASES.iter().copied().enumerate() {
            let (evidence, case_observation) =
                run_case(&session, &queue, case, case_index, &config.target)?;
            cases.push(evidence);
            observation.max_finite_ulp = observation
                .max_finite_ulp
                .max(case_observation.max_finite_ulp);
            observation.max_finite_abs_error = observation
                .max_finite_abs_error
                .max(case_observation.max_finite_abs_error);
        }
        Ok((cases, observation))
    })();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("execution-session shutdown failed: {error}"))?;
    let (cases, observation) = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("attention preprocess cleanup did not return to zero owned work".to_owned());
    }
    let operation_count = cases.len();
    let rounding_order_changed = config.target == "gfx1030"
        && default_on_env(env::var_os("SLLM_ATTENTION_PREPROCESS_GFX1030_WAVE32").as_deref());
    Ok(Report {
        schema_version: "attention-preprocess-g1-report-v1",
        state: "PASS",
        pass: true,
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        cpu_fallback_used: false,
        fallback: false,
        fallback_allowed: false,
        fallback_used: false,
        rounding_order_changed,
        operations: operation_count,
        dispatch_count: operation_count,
        kernel_dispatches: operation_count,
        comparison: ComparisonReport {
            finite_policy: "finite BF16 outputs require same classification and <=2 ordered BF16 ULPs; this narrow bound covers CPU/device libm variation only",
            finite_bf16_ulp_bound: FINITE_BF16_ULP_BOUND,
            special_policy: "zero, NaN, and Inf classification/sign are exact; NaN payload bits are not required",
            gate_policy: "gate BF16 bytes are exact",
            max_observed_finite_ulp: observation.max_finite_ulp,
            max_observed_finite_abs_error: observation.max_finite_abs_error,
        },
        cases,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
    })
}

fn main() -> ExitCode {
    match parse_config().and_then(|config| run(&config)) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("attention-preprocess-g1 report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("attention-preprocess-g1 evidence failed: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave32_gate_defaults_on_and_accepts_only_explicit_disable() {
        assert!(default_on_env(None));
        assert!(default_on_env(Some(std::ffi::OsStr::new("1"))));
        assert!(!default_on_env(Some(std::ffi::OsStr::new("0"))));
        assert!(!default_on_env(Some(std::ffi::OsStr::new("unknown"))));
    }

    #[test]
    fn bf16_rne_ties_round_to_even_and_specials_are_stable() {
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(float_to_bf16_rne(0.0), 0x0000);
        assert_eq!(float_to_bf16_rne(-0.0), 0x8000);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x0001)), 0x0000);
        assert_eq!(float_to_bf16_rne(f32::INFINITY), 0x7f80);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x7fc1_2345)), 0x7fc1);
    }

    #[test]
    fn packed_q_gate_generation_is_head_wise_not_flat() {
        let packed = make_packed_q_gate(2, 0);
        let first_head = 0;
        let second_head = QGATE_HEAD_WIDTH;
        assert_eq!(packed[first_head + 7], generated_word(7, 1));
        assert_eq!(packed[second_head + 7], generated_word(HEAD_DIM + 7, 1));
        assert_ne!(
            packed[first_head + 7],
            packed[second_head + 7],
            "head 1 must not be read as a flat continuation of head 0"
        );
        assert_eq!(
            packed[second_head + HEAD_DIM + 3],
            generated_word(HEAD_DIM + 3, 101)
        );
        assert_eq!(packed.len(), 2 * Q_HEADS * QGATE_HEAD_WIDTH);
    }

    #[test]
    fn neo_x_pairing_uses_d_and_d_plus_32_for_the_first_64_dimensions() {
        let mut values = vec![0.0_f32; HEAD_DIM];
        values[0] = 1.0;
        values[32] = 2.0;
        rotate_neox(&mut values, 1);
        let angle = ROPE_THETA.powf(0.0);
        assert!((values[0] - (angle.cos() - 2.0 * angle.sin())).abs() < 1.0e-6);
        assert!((values[32] - (angle.sin() + 2.0 * angle.cos())).abs() < 1.0e-6);
        assert_eq!(values[1], 0.0);
        assert_eq!(values[33], 0.0);
        assert_eq!(values[64], 0.0);
    }

    #[test]
    fn coverage_includes_requested_sizes_modes_and_absolute_positions() {
        assert_eq!(CASES.len(), 8);
        for size in [1, 3, 17] {
            assert!(CASES.iter().any(|case| case.m == size));
        }
        assert!(
            CASES
                .iter()
                .any(|case| case.mode == AttentionPreprocessPositionMode::Prefill
                    && case.start_position == 0)
        );
        assert!(CASES.iter().any(|case| {
            case.mode == AttentionPreprocessPositionMode::DecodeContinuation
                && case.start_position > 0
        }));
        let mut absolute_positions = Vec::new();
        for case in CASES {
            absolute_positions.extend(positions(case));
        }
        for position in [0, 1, 3, 255, 256, 257] {
            assert!(absolute_positions.contains(&position));
        }
        assert!(CASES.iter().map(|case| case.m).sum::<usize>() < 64);
    }

    #[test]
    fn generated_inputs_cover_all_required_bf16_classes_without_large_buffers() {
        let packed = make_packed_q_gate(17, 0);
        let k = make_k(17, 1);
        let scales = make_scale(Q_HEADS, 2);
        for values in [&packed, &k, &scales] {
            for word in SPECIAL_VALUES {
                assert!(
                    values.contains(&word),
                    "missing special BF16 word 0x{word:04x}"
                );
            }
        }
        assert!(packed.len() < 200_000);
        assert!(k.len() < 40_000);
    }

    #[test]
    fn comparison_policy_is_narrow_and_fail_closed() {
        let expected = [0x3f80, 0x0000, 0x8000, 0x7fc1, 0x7f80];
        let actual = [0x3f82, 0x0000, 0x8000, 0x7fc0, 0x7f80];
        let mut observation = ComparisonObservation::default();
        compare_bf16_words(&expected, &actual, "test", &mut observation).unwrap();
        assert_eq!(observation.max_finite_ulp, 2);
        assert!(observation.max_finite_abs_error > 0.0);

        let mut observation = ComparisonObservation::default();
        let error = compare_bf16_words(&[0x3f80], &[0x3f84], "test", &mut observation)
            .expect_err("three BF16 ULPs must exceed the explicit bound");
        assert!(error.contains("bound is 2"));

        let mut observation = ComparisonObservation::default();
        let error = compare_bf16_words(&[0x0000], &[0x8000], "test", &mut observation)
            .expect_err("zero sign must be exact");
        assert!(error.contains("zero classification/sign"));
    }

    #[test]
    fn scalar_oracle_keeps_head_boundaries_and_rope_stage_order() {
        let input = vec![float_to_bf16_rne(1.0); HEAD_DIM];
        let scale = vec![float_to_bf16_rne(0.0); HEAD_DIM];
        let first = scalar_head_oracle(&input, &scale, 0);
        let second = scalar_head_oracle(&input, &scale, 1);
        assert_eq!(first[0], float_to_bf16_rne(1.0));
        assert_ne!(first, second);
        let mut pair_input = vec![0_u16; HEAD_DIM];
        pair_input[0] = float_to_bf16_rne(1.0);
        pair_input[32] = float_to_bf16_rne(2.0);
        let pair_values = scalar_head_oracle(&pair_input, &scale, 1);
        assert_eq!(pair_values.len(), HEAD_DIM);
    }
}
