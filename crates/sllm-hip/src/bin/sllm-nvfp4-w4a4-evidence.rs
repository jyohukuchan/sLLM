//! Numerical and dispatch evidence for the first-class NVFP4 W4A4 path.
//!
//! CPU code independently applies the documented two-level activation
//! quantization and is an oracle only; it is never an execution fallback.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, Encoding, ExecutionSessionRequest, ExecutionState,
    SemanticOpDescriptor, SemanticOpKind, TensorView, quantize_nvfp4_weights,
};
use sllm_hip::HipBackend;

const WAIT: Duration = Duration::from_secs(30);
const SHUTDOWN: Duration = Duration::from_secs(16);
const PREFILL_KERNEL: &str = "matmul.nvfp4.w4a4.block16.prefill.row8_tiled256.v1";
const PREFILL_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1";
const PREFILL_COL8_KERNEL: &str = "matmul.nvfp4.w4a4.block16.prefill.row8_col8_tiled256.v1";
const PREFILL_COL8_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_block16_prefill_row8_col8_tiled256_v1";
const PREFILL_DP4A_KERNEL: &str = "matmul.nvfp4.w4a4.block16.prefill.dp4a64x64.v1";
const PREFILL_DP4A_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_block16_prefill_dp4a_64x64_v1";
const PREFILL_GFX1201_WMMA_KERNEL: &str = "matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x64.v1";
const PREFILL_GFX1201_WMMA_DEVICE: &str = "sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_v1";
const PREFILL_GFX1201_WMMA_F16SCALE_KERNEL: &str =
    "matmul.nvfp4.w4a4.prefill.gfx1201.wmma_f16scale128x64.v1";
const PREFILL_GFX1201_WMMA_F16SCALE_DEVICE: &str =
    "sllm_nvfp4_w4a4_prefill_gfx1201_wmma_f16scale128x64_v1";
const DECODE_KERNEL: &str = "matmul.nvfp4.w4a4.block16.decode.v1";
const DECODE_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_block16_decode_v1";
const DECODE_COLUMNS128_KERNEL: &str = "matmul.nvfp4.w4a4.decode.columns128.v1";
const DECODE_COLUMNS128_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_decode_columns128_v1";
const DECODE_WAVE4_KERNEL: &str = "matmul.nvfp4.w4a4.decode.dp4a.wave4col32.v1";
const DECODE_WAVE4_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_v1";
const BASELINE_KERNEL: &str = "matmul.nvfp4.w4a4.block16.packed.v1";
const BASELINE_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_block16_packed_v1";

#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    k: usize,
    n: usize,
}

const CASES: [Shape; 21] = [
    Shape { m: 1, k: 15, n: 17 },
    Shape { m: 3, k: 16, n: 16 },
    Shape { m: 7, k: 17, n: 15 },
    Shape {
        m: 17,
        k: 31,
        n: 17,
    },
    Shape {
        m: 32,
        k: 32,
        n: 33,
    },
    Shape {
        m: 33,
        k: 33,
        n: 31,
    },
    Shape { m: 1, k: 31, n: 33 },
    Shape { m: 3, k: 32, n: 31 },
    Shape { m: 7, k: 33, n: 32 },
    Shape {
        m: 17,
        k: 15,
        n: 33,
    },
    Shape {
        m: 32,
        k: 16,
        n: 31,
    },
    Shape {
        m: 33,
        k: 17,
        n: 32,
    },
    Shape {
        m: 128,
        k: 17,
        n: 33,
    },
    Shape {
        m: 512,
        k: 31,
        n: 17,
    },
    Shape {
        m: 1024,
        k: 33,
        n: 15,
    },
    Shape {
        m: 127,
        k: 48,
        n: 63,
    },
    Shape {
        m: 128,
        k: 48,
        n: 64,
    },
    Shape {
        m: 129,
        k: 48,
        n: 65,
    },
    Shape {
        m: 1,
        k: 16,
        n: 127,
    },
    Shape {
        m: 1,
        k: 32,
        n: 128,
    },
    Shape {
        m: 1,
        k: 48,
        n: 129,
    },
];

#[derive(Serialize)]
struct CaseReport {
    m: usize,
    k: usize,
    n: usize,
    dispatch_count: u32,
    kernel_id: u32,
    kernel_elapsed_ns: u64,
    kernel_elapsed_samples_ns: Vec<u64>,
    warmup_count: usize,
    measured_count: usize,
    input_decode_global: f32,
    max_abs_error: f32,
    max_relative_error: f32,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    provider: &'static str,
    arithmetic: &'static str,
    fallback_allowed: bool,
    fallback_used: bool,
    cases: Vec<CaseReport>,
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

fn from_bf16(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn words_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        sign * f32::from(mantissa) * 2.0_f32.powi(-9)
    } else {
        sign * (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    }
}

fn encode_e4m3(value: f32) -> u8 {
    if value == 0.0 {
        return 0;
    }
    let bounded = value.abs().min(448.0);
    (0_u8..=0x7e)
        .min_by(|left, right| {
            let left_error = (e4m3(*left) - bounded).abs();
            let right_error = (e4m3(*right) - bounded).abs();
            left_error
                .total_cmp(&right_error)
                .then_with(|| (left & 1).cmp(&(right & 1)))
        })
        .unwrap_or(0)
}

fn e2m1(code: u8) -> f32 {
    const POSITIVE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let value = POSITIVE[usize::from(code & 7)];
    if code & 8 == 0 { value } else { -value }
}

fn encode_e2m1(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 8 } else { 0 };
    let magnitude = value.abs();
    let code = (0_u8..8)
        .min_by(|left, right| {
            let left_error = (e2m1(*left) - magnitude).abs();
            let right_error = (e2m1(*right) - magnitude).abs();
            left_error
                .total_cmp(&right_error)
                .then_with(|| (left & 1).cmp(&(right & 1)))
        })
        .unwrap_or(0);
    sign | code
}

fn quantize_activation(values: &[u16], m: usize, k: usize, global: f32) -> Vec<f32> {
    let mut decoded = vec![0.0; m * k];
    for row in 0..m {
        for block in 0..k.div_ceil(16) {
            let start = block * 16;
            let end = (start + 16).min(k);
            let maximum = (start..end)
                .map(|column| from_bf16(values[row * k + column]).abs())
                .fold(0.0_f32, f32::max);
            let block_scale = e4m3(if maximum == 0.0 {
                0
            } else {
                encode_e4m3(maximum / (6.0 * global))
            });
            let scale = block_scale * global;
            for column in start..end {
                let value = from_bf16(values[row * k + column]);
                decoded[row * k + column] = if scale > 0.0 {
                    e2m1(encode_e2m1(value / scale)) * scale
                } else {
                    0.0
                };
            }
        }
    }
    decoded
}

fn wait_ok(
    state: Result<ExecutionState, sllm_core::ExecutionError>,
    label: &str,
) -> Result<(), String> {
    match state.map_err(|error| format!("{label}: {error}"))? {
        ExecutionState::Success => Ok(()),
        other => Err(format!("{label}: unexpected state {other:?}")),
    }
}

fn matrix(rows: usize, columns: usize, phase: usize) -> Vec<u16> {
    (0..rows * columns)
        .map(|index| {
            let block_scale = match (index % columns) / 16 % 4 {
                0 => 0.125,
                1 => 0.5,
                2 => 2.0,
                _ => 8.0,
            };
            bf16((((index * 37 + phase * 19) % 257) as i32 - 128) as f32 / 31.0 * block_scale)
        })
        .collect()
}

fn benchmark_iterations(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|count| *count > 0)
            .ok_or_else(|| format!("{name} must be a positive usize")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("cannot read {name}: {error}")),
    }
}

fn median_u64(values: &[u64]) -> u64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        let lower = sorted[middle - 1];
        let upper = sorted[middle];
        lower / 2 + upper / 2 + (lower % 2 + upper % 2) / 2
    } else {
        sorted[middle]
    }
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    shape: Shape,
    phase: usize,
    target: &str,
) -> Result<CaseReport, String> {
    let activation = matrix(shape.m, shape.k, phase);
    let weight = matrix(shape.n, shape.k, phase + 7);
    let weight_f32 = weight.iter().copied().map(from_bf16).collect::<Vec<_>>();
    let quantized_weight =
        quantize_nvfp4_weights(&weight_f32, shape.n, shape.k).map_err(|error| error.to_string())?;
    if shape.k >= 32
        && quantized_weight.block_scales.first() == quantized_weight.block_scales.get(1)
    {
        return Err(format!(
            "test fixture did not produce distinct adjacent K16 weight scales for k={}",
            shape.k
        ));
    }
    let weight_decoded = quantized_weight.dequantize();
    let activation_max = activation
        .iter()
        .copied()
        .map(from_bf16)
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    let input_decode_global = (activation_max / (448.0 * 6.0)).max(f32::MIN_POSITIVE);
    let activation_decoded =
        quantize_activation(&activation, shape.m, shape.k, input_decode_global);

    let mut resident = quantized_weight.packed_values.clone();
    resident.extend_from_slice(&quantized_weight.block_scales);
    while resident.len() & 3 != 0 {
        resident.push(0);
    }
    resident.extend_from_slice(&quantized_weight.tensor_scale.to_le_bytes());
    resident.extend_from_slice(&input_decode_global.to_le_bytes());
    let activation_bytes = words_bytes(&activation);
    let output_bytes = shape.m * shape.n * 2;
    let activation_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    let weight_buffer = session
        .allocate(resident.len() as u64)
        .map_err(|error| error.to_string())?;
    let output_buffer = session
        .allocate(output_bytes as u64)
        .map_err(|error| error.to_string())?;
    for (label, buffer, bytes) in [
        (
            "activation",
            &activation_buffer,
            activation_bytes.as_slice(),
        ),
        ("weight", &weight_buffer, resident.as_slice()),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|error| error.to_string())?;
        wait_ok(upload.wait(WAIT), label)?;
    }
    let activation_view = TensorView::contiguous(DType::Bf16, &[shape.m, shape.k])
        .map_err(|error| error.to_string())?;
    let weight_view = TensorView::with_encoding(
        DType::U8,
        Encoding::Nvfp4W4A4 {
            block_size: 16,
            scale_dtype: DType::F8E4M3Fn,
        },
        &[shape.n, shape.k],
    )
    .map_err(|error| error.to_string())?;
    let output_view = TensorView::contiguous(DType::Bf16, &[shape.m, shape.n])
        .map_err(|error| error.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation_view.clone(), weight_view.clone()],
            vec![output_view.clone()],
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&activation_buffer, activation_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&output_buffer, output_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let force_baseline = env::var("SLLM_NVFP4_W4A4_FORCE_BASELINE").as_deref() == Ok("1");
    let force_row8 = env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_ROW8").as_deref() == Ok("1");
    let force_col8 = env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_COL8").as_deref() == Ok("1");
    let force_dp4a = env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A").as_deref() == Ok("1");
    let force_gfx1201_wmma =
        env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA").as_deref() == Ok("1");
    let force_gfx1201_wmma_f16scale =
        env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA_F16SCALE").as_deref() == Ok("1");
    let force_decode_columns =
        env::var("SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_COLUMNS").as_deref() == Ok("1");
    let force_decode_wave4 =
        env::var("SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4").as_deref() == Ok("1");
    let (expected_kernel_id, expected_kernel, expected_device) = if force_baseline {
        (11, BASELINE_KERNEL, BASELINE_DEVICE)
    } else if shape.m == 1 {
        if force_decode_wave4
            && matches!(target, "gfx1030" | "gfx1201")
            && (shape.k % 16 == 0)
            && shape.k <= 17_408
        {
            (67, DECODE_WAVE4_KERNEL, DECODE_WAVE4_DEVICE)
        } else if force_decode_columns
            && matches!(target, "gfx1030" | "gfx1201")
            && (shape.k % 16 == 0)
            && shape.k <= 17_408
        {
            (65, DECODE_COLUMNS128_KERNEL, DECODE_COLUMNS128_DEVICE)
        } else {
            (58, DECODE_KERNEL, DECODE_DEVICE)
        }
    } else if force_row8 {
        (59, PREFILL_KERNEL, PREFILL_DEVICE)
    } else if force_col8 {
        (61, PREFILL_COL8_KERNEL, PREFILL_COL8_DEVICE)
    } else if force_gfx1201_wmma_f16scale && target == "gfx1201" && shape.m > 1 && shape.k % 16 == 0
    {
        (
            69,
            PREFILL_GFX1201_WMMA_F16SCALE_KERNEL,
            PREFILL_GFX1201_WMMA_F16SCALE_DEVICE,
        )
    } else if force_gfx1201_wmma && target == "gfx1201" && shape.k % 16 == 0 {
        (64, PREFILL_GFX1201_WMMA_KERNEL, PREFILL_GFX1201_WMMA_DEVICE)
    } else if force_dp4a && shape.k % 16 == 0 {
        (62, PREFILL_DP4A_KERNEL, PREFILL_DP4A_DEVICE)
    } else {
        (59, PREFILL_KERNEL, PREFILL_DEVICE)
    };
    let benchmark = env::var("SLLM_NVFP4_BENCHMARK").as_deref() == Ok("1");
    let warmup_count = if benchmark {
        benchmark_iterations("SLLM_NVFP4_BENCHMARK_WARMUPS", 3)?
    } else {
        0
    };
    let measured_count = if benchmark {
        benchmark_iterations("SLLM_NVFP4_BENCHMARK_MEASURED", 10)?
    } else {
        1
    };
    let total_count = warmup_count + measured_count;
    let mut kernel_elapsed_samples_ns = Vec::with_capacity(measured_count);
    let mut final_submission = None;
    for iteration in 0..total_count {
        let mut submission = session
            .submit(&prepared, queue)
            .map_err(|error| error.to_string())?;
        let dispatch = submission.dispatch().clone();
        wait_ok(submission.wait(WAIT), "W4A4 matmul")?;
        if dispatch.dispatch_count != 2
            || dispatch.kernel_id != expected_kernel_id
            || dispatch.kernel_symbol != expected_kernel
            || dispatch.device_symbol != expected_device
            || dispatch.target != target
            || dispatch.fallback_allowed
            || dispatch.fallback_used
        {
            return Err(format!("unexpected W4A4 dispatch: {dispatch:?}"));
        }
        if iteration >= warmup_count {
            kernel_elapsed_samples_ns.push(
                submission
                    .kernel_elapsed_ns()
                    .map_err(|error| error.to_string())?
                    .ok_or_else(|| "missing GPU timing".to_owned())?,
            );
        }
        if iteration + 1 == total_count {
            final_submission = Some(submission);
        }
    }
    let kernel_elapsed_ns = median_u64(&kernel_elapsed_samples_ns);
    let mut submission = final_submission.ok_or_else(|| "no measured submission".to_owned())?;
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| error.to_string())?;
    wait_ok(readback.wait(WAIT), "readback")?;
    let mut bytes = vec![0_u8; output_bytes];
    readback
        .read_into(&mut bytes)
        .map_err(|error| error.to_string())?;
    let mut max_abs_error = 0.0_f32;
    let mut max_relative_error = 0.0_f32;
    for row in 0..shape.m {
        for column in 0..shape.n {
            if benchmark && !matches!(row, 0 | 1) && row != shape.m / 2 && row + 1 != shape.m {
                continue;
            }
            if benchmark
                && !matches!(column, 0 | 1)
                && column != shape.n / 3
                && column != (shape.n * 2) / 3
                && column + 1 != shape.n
            {
                continue;
            }
            let expected = (0..shape.k)
                .map(|inner| {
                    activation_decoded[row * shape.k + inner]
                        * weight_decoded[column * shape.k + inner]
                })
                .sum::<f32>();
            let index = (row * shape.n + column) * 2;
            let actual = from_bf16(u16::from_le_bytes([bytes[index], bytes[index + 1]]));
            let absolute = (actual - expected).abs();
            let relative = absolute / expected.abs().max(1.0);
            max_abs_error = max_abs_error.max(absolute);
            max_relative_error = max_relative_error.max(relative);
            if !actual.is_finite() || relative > 0.02 {
                return Err(format!(
                    "numerical mismatch m={} k={} n={} row={row} column={column} expected={expected} actual={actual} relative={relative}",
                    shape.m, shape.k, shape.n
                ));
            }
        }
    }
    Ok(CaseReport {
        m: shape.m,
        k: shape.k,
        n: shape.n,
        dispatch_count: 2,
        kernel_id: expected_kernel_id,
        kernel_elapsed_ns,
        kernel_elapsed_samples_ns,
        warmup_count,
        measured_count,
        input_decode_global,
        max_abs_error,
        max_relative_error,
    })
}

fn run(device_index: u32, target: String) -> Result<Report, String> {
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("target must be gfx1030 or gfx1201".to_owned());
    }
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let result = (|| {
        let queue = session.create_queue().map_err(|error| error.to_string())?;
        let cases = if env::var("SLLM_NVFP4_BENCHMARK").as_deref() == Ok("1") {
            let scope = env::var("SLLM_NVFP4_BENCHMARK_SCOPE").unwrap_or_else(|_| "all".to_owned());
            let shapes = vec![
                Shape {
                    m: 1,
                    k: 5120,
                    n: 17408,
                },
                Shape {
                    m: 1,
                    k: 17408,
                    n: 5120,
                },
                Shape {
                    m: 128,
                    k: 5120,
                    n: 17408,
                },
                Shape {
                    m: 128,
                    k: 17408,
                    n: 5120,
                },
                Shape {
                    m: 512,
                    k: 5120,
                    n: 17408,
                },
                Shape {
                    m: 512,
                    k: 17408,
                    n: 5120,
                },
                Shape {
                    m: 1024,
                    k: 5120,
                    n: 17408,
                },
                Shape {
                    m: 1024,
                    k: 17408,
                    n: 5120,
                },
            ];
            let mut selected = match scope.as_str() {
                "all" => shapes,
                "decode" => shapes.into_iter().filter(|shape| shape.m == 1).collect(),
                "prefill" => shapes.into_iter().filter(|shape| shape.m > 1).collect(),
                _ => {
                    return Err(
                        "SLLM_NVFP4_BENCHMARK_SCOPE must be all, decode, or prefill".to_owned()
                    );
                }
            };
            if let Ok(rows) = env::var("SLLM_NVFP4_BENCHMARK_M") {
                let rows = rows
                    .parse::<usize>()
                    .map_err(|_| "SLLM_NVFP4_BENCHMARK_M must be an integer".to_owned())?;
                selected.retain(|shape| shape.m == rows);
                if selected.is_empty() {
                    return Err(format!(
                        "SLLM_NVFP4_BENCHMARK_M={rows} selected no benchmark shapes"
                    ));
                }
            }
            selected
        } else {
            CASES.to_vec()
        };
        cases
            .into_iter()
            .enumerate()
            .map(|(index, shape)| run_case(&session, &queue, shape, index, &target))
            .collect::<Result<Vec<_>, _>>()
    })();
    let cleanup = session
        .shutdown(SHUTDOWN)
        .map_err(|error| error.to_string())?;
    let cases = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("nonzero cleanup state".to_owned());
    }
    Ok(Report {
        schema_version: "phase16f-nvfp4-w4a4-v1",
        state: "PASS",
        target,
        device_index,
        provider: "dynamic-block16-w4a4-decode-row8-prefill",
        arithmetic: "E2M1xE2M1/FP32-accumulate/BF16-output",
        fallback_allowed: false,
        fallback_used: false,
        cases,
    })
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let result = match arguments.as_slice() {
        [device, target] => device
            .parse::<u32>()
            .map_err(|_| "device index must be u32".to_owned())
            .and_then(|device| run(device, target.clone())),
        _ => Err("usage: sllm-nvfp4-w4a4-evidence DEVICE_INDEX gfx1030|gfx1201".to_owned()),
    };
    match result {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string(&report).expect("report serialization")
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("NVFP4 W4A4 evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
