//! Numerical and dispatch evidence for the first-class NVFP4 W4A4 path.
//!
//! CPU code independently applies the documented two-level activation
//! quantization and is an oracle only; it is never an execution fallback.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
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
const DECODE_KERNEL: &str = "matmul.nvfp4.w4a4.block16.decode.v1";
const DECODE_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_block16_decode_v1";
const DECODE_WAVE4_KERNEL: &str = "matmul.nvfp4.w4a4.decode.dp4a.wave4col32.v1";
const DECODE_WAVE4_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_decode_dp4a_wave4col32_v1";
const BASELINE_KERNEL: &str = "matmul.nvfp4.w4a4.block16.packed.v1";
const BASELINE_DEVICE: &str = "sllm_matmul_nvfp4_w4a4_block16_packed_v1";
const BASELINE_ORACLE_ENV: &str = "SLLM_FORCE_BASELINE_ORACLE";
const BASELINE_ORACLE_SHAPE_ENV: &str = "SLLM_FORCE_BASELINE_ORACLE_SHAPE";
const BASELINE_ORACLE_CONTROL_ENV: &str = "SLLM_FORCE_BASELINE_ORACLE_CONTROL";
const BASELINE_ORACLE_MAX_M: usize = 4096;
const BASELINE_ORACLE_MAX_K: usize = 17_408;
const BASELINE_ORACLE_MAX_N: usize = 65_536;
const BASELINE_ORACLE_MAX_WEIGHT_ELEMENTS: usize = 200_000_000;
const BASELINE_ORACLE_MAX_OUTPUT_ELEMENTS: usize = 80_000_000;

#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    k: usize,
    n: usize,
}

// Keep the default oracle run bounded while covering the real Qwen3.8
// projections, decode, and the baseline grid boundary.  The last three cases
// deliberately use K=17/N=8192 so that only the M*N launch size crosses the
// 2^24-1 element boundary; they do not allocate a Qwen-sized weight plane.
const BASELINE_ORACLE_CASES: [Shape; 7] = [
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
        m: 2048,
        k: 5120,
        n: 17408,
    },
    Shape {
        m: 2048,
        k: 17408,
        n: 5120,
    },
    Shape {
        m: 2047,
        k: 17,
        n: 8192,
    },
    Shape {
        m: 2048,
        k: 17,
        n: 8192,
    },
    Shape {
        m: 2049,
        k: 17,
        n: 8192,
    },
];

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
    kernel_symbol: String,
    device_symbol: String,
    kernel_elapsed_ns: u64,
    kernel_elapsed_samples_ns: Vec<u64>,
    warmup_count: usize,
    measured_count: usize,
    input_decode_global: f32,
    max_abs_error: f32,
    max_relative_error: f32,
    output_bf16_sha256: String,
    sampled_coordinates: Option<Vec<[usize; 2]>>,
    sampled_output_bf16_sha256: Option<String>,
    sampled_oracle_fp32_sha256: Option<String>,
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
    baseline_oracle: bool,
    baseline_oracle_control: bool,
    shape_override: Option<String>,
    cases: Vec<CaseReport>,
    current_bytes_before_shutdown: u64,
    poisoned_before_shutdown: bool,
    cleanup_retryable: usize,
    cleanup_durable: usize,
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

fn quantize_activation_scales(values: &[u16], m: usize, k: usize, global: f32) -> Vec<f32> {
    let blocks_per_row = k.div_ceil(16);
    let mut scales = vec![0.0_f32; m * blocks_per_row];
    for row in 0..m {
        for block in 0..blocks_per_row {
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
            scales[row * blocks_per_row + block] = block_scale * global;
        }
    }
    scales
}

fn quantized_activation_value(
    values: &[u16],
    scales: &[f32],
    k: usize,
    row: usize,
    column: usize,
) -> f32 {
    let scale = scales[row * k.div_ceil(16) + column / 16];
    if scale > 0.0 {
        e2m1(encode_e2m1(from_bf16(values[row * k + column]) / scale)) * scale
    } else {
        0.0
    }
}

/// Decode one resident NVFP4 weight value using only scalar host arithmetic.
/// This intentionally does not call `QuantizedNvfp4::dequantize`, so the
/// sampled check remains independent of the production dequantizer.
fn quantized_weight_value(
    packed_values: &[u8],
    block_scales: &[u8],
    tensor_scale: f32,
    k: usize,
    row: usize,
    column: usize,
) -> f32 {
    let index = row * k + column;
    let packed = packed_values[index / 2];
    let code = if index & 1 == 0 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let scale = e4m3(block_scales[row * k.div_ceil(16) + column / 16]);
    e2m1(code) * scale * tensor_scale
}

fn independent_fp32_dot(
    activation: &[u16],
    activation_scales: &[f32],
    quantized_weight: &sllm_core::QuantizedNvfp4,
    row: usize,
    column: usize,
) -> f32 {
    (0..quantized_weight.columns)
        .map(|inner| {
            quantized_activation_value(
                activation,
                activation_scales,
                quantized_weight.columns,
                row,
                inner,
            ) * quantized_weight_value(
                &quantized_weight.packed_values,
                &quantized_weight.block_scales,
                quantized_weight.tensor_scale,
                quantized_weight.columns,
                column,
                inner,
            )
        })
        .sum::<f32>()
}

fn push_unique(values: &mut Vec<usize>, value: usize, limit: usize) {
    if value < limit && !values.contains(&value) {
        values.push(value);
    }
}

/// Return a deterministic coordinate set.  It is shared by baseline and
/// default runs so an output digest can be compared before and after a source
/// change without changing the oracle sample itself.
fn sampled_coordinates(shape: Shape) -> Vec<[usize; 2]> {
    let mut rows = Vec::new();
    for row in [
        0,
        1,
        15,
        16,
        17,
        127,
        128,
        129,
        shape.m / 2,
        shape.m.saturating_sub(2),
        shape.m.saturating_sub(1),
    ] {
        push_unique(&mut rows, row, shape.m);
    }
    let mut columns = Vec::new();
    for column in [
        0,
        1,
        15,
        16,
        17,
        shape.n / 2,
        shape.n.saturating_sub(2),
        shape.n.saturating_sub(1),
    ] {
        push_unique(&mut columns, column, shape.n);
    }
    let mut coordinates: Vec<[usize; 2]> = rows
        .into_iter()
        .flat_map(|row| columns.iter().copied().map(move |column| [row, column]))
        .collect();
    // Independently enumerate both sides of each launch boundary. The
    // reference must exercise the offset into a later launch, including a
    // boundary in the middle of an output row.
    let outputs = shape.m * shape.n;
    let launch_outputs = (u32::MAX as usize) / 256;
    for boundary in (launch_outputs..outputs).step_by(launch_outputs) {
        for index in [boundary - 1, boundary, boundary + 1] {
            if index < outputs {
                let point = [index / shape.n, index % shape.n];
                if !coordinates.contains(&point) {
                    coordinates.push(point);
                }
            }
        }
    }
    coordinates
}

fn sampled_digest(
    shape: Shape,
    coordinates: &[[usize; 2]],
    bytes: &[u8],
    oracle: Option<&[f32]>,
) -> String {
    let mut digest = Sha256::new();
    digest.update(if oracle.is_some() {
        b"sllm-nvfp4-w4a4-sampled-oracle-v1\0".as_slice()
    } else {
        b"sllm-nvfp4-w4a4-sampled-output-v1\0".as_slice()
    });
    digest.update((shape.m as u64).to_le_bytes());
    digest.update((shape.k as u64).to_le_bytes());
    digest.update((shape.n as u64).to_le_bytes());
    for (index, [row, column]) in coordinates.iter().copied().enumerate() {
        digest.update((row as u64).to_le_bytes());
        digest.update((column as u64).to_le_bytes());
        if oracle.is_none() {
            let output_index = (row * shape.n + column) * 2;
            digest.update(&bytes[output_index..output_index + 2]);
        }
        if let Some(oracle) = oracle {
            digest.update(oracle[index].to_bits().to_le_bytes());
        }
    }
    format!("sha256:{:x}", digest.finalize())
}

fn wait_ok(
    mut poll: impl FnMut(bool) -> Result<ExecutionState, sllm_core::ExecutionError>,
    label: &str,
) -> Result<(), String> {
    let long_oracle = env::var(BASELINE_ORACLE_ENV).as_deref() == Ok("1");
    let mut last_progress = std::time::Instant::now();
    loop {
        match poll(long_oracle).map_err(|error| format!("{label}: {error}"))? {
            ExecutionState::Success => return Ok(()),
            ExecutionState::Pending if long_oracle => {
                // Query is non-destructive. In contrast, the public wait API
                // treats an expired deadline as a terminal backend failure.
                // Never retry that error or relabel it as Pending/PASS.
                if last_progress.elapsed() >= WAIT {
                    eprintln!("[baseline oracle] {label} still pending; querying live operation");
                    last_progress = std::time::Instant::now();
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            other => return Err(format!("{label}: unexpected state {other:?}")),
        }
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

fn validate_baseline_oracle_shape(shape: Shape) -> Result<(), String> {
    let weight_elements = shape
        .k
        .checked_mul(shape.n)
        .ok_or_else(|| "baseline oracle weight shape overflowed usize".to_owned())?;
    let output_elements = shape
        .m
        .checked_mul(shape.n)
        .ok_or_else(|| "baseline oracle output shape overflowed usize".to_owned())?;
    if shape.m == 0
        || shape.k == 0
        || shape.n == 0
        || shape.m > BASELINE_ORACLE_MAX_M
        || shape.k > BASELINE_ORACLE_MAX_K
        || shape.n > BASELINE_ORACLE_MAX_N
        || weight_elements > BASELINE_ORACLE_MAX_WEIGHT_ELEMENTS
        || output_elements > BASELINE_ORACLE_MAX_OUTPUT_ELEMENTS
    {
        return Err(format!(
            "baseline oracle shape m={} k={} n={} is outside the supported diagnostic range (M<= {}, K<= {}, N<= {}, K*N<= {}, M*N<= {})",
            shape.m,
            shape.k,
            shape.n,
            BASELINE_ORACLE_MAX_M,
            BASELINE_ORACLE_MAX_K,
            BASELINE_ORACLE_MAX_N,
            BASELINE_ORACLE_MAX_WEIGHT_ELEMENTS,
            BASELINE_ORACLE_MAX_OUTPUT_ELEMENTS
        ));
    }
    Ok(())
}

fn parse_baseline_oracle_shape(value: &str) -> Result<Shape, String> {
    let dimensions = value
        .split(|character: char| matches!(character, 'x' | 'X' | ',' | ':' | ' '))
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>().map_err(|_| {
                format!("{BASELINE_ORACLE_SHAPE_ENV} must be MxKxN with decimal dimensions")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let [m, k, n] = dimensions.as_slice() else {
        return Err(format!(
            "{BASELINE_ORACLE_SHAPE_ENV} must be MxKxN with decimal dimensions"
        ));
    };
    let shape = Shape {
        m: *m,
        k: *k,
        n: *n,
    };
    validate_baseline_oracle_shape(shape)?;
    Ok(shape)
}

fn baseline_oracle_cases() -> Result<(Vec<Shape>, Option<String>), String> {
    match env::var(BASELINE_ORACLE_SHAPE_ENV) {
        Ok(value) => Ok((vec![parse_baseline_oracle_shape(&value)?], Some(value))),
        Err(env::VarError::NotPresent) => {
            for shape in BASELINE_ORACLE_CASES {
                validate_baseline_oracle_shape(shape)?;
            }
            Ok((BASELINE_ORACLE_CASES.to_vec(), None))
        }
        Err(error) => Err(format!("cannot read {BASELINE_ORACLE_SHAPE_ENV}: {error}")),
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
    sampled_oracle: bool,
    baseline_oracle_control: bool,
) -> Result<CaseReport, String> {
    if sampled_oracle {
        eprintln!(
            "[baseline oracle] start {}x{}x{}",
            shape.m, shape.k, shape.n
        );
    }
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
    let activation_max = activation
        .iter()
        .copied()
        .map(from_bf16)
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    let input_decode_global = (activation_max / (448.0 * 6.0)).max(f32::MIN_POSITIVE);
    let activation_scales =
        quantize_activation_scales(&activation, shape.m, shape.k, input_decode_global);

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
        wait_ok(
            |query| {
                if query {
                    upload.query()
                } else {
                    upload.wait(WAIT)
                }
            },
            label,
        )?;
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
    let force_decode_wave4 =
        env::var("SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4").as_deref() == Ok("1");
    let (expected_kernel_id, expected_kernel, expected_device) = if force_baseline {
        (11, BASELINE_KERNEL, BASELINE_DEVICE)
    } else if shape.m == 1 {
        let wave4_default = env::var_os("SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4").is_none()
            && shape.k >= 1024
            && shape.n >= 1024;
        if (force_decode_wave4 || wave4_default)
            && matches!(target, "gfx1030" | "gfx1201")
            && (shape.k % 16 == 0)
            && shape.k <= 17_408
        {
            (67, DECODE_WAVE4_KERNEL, DECODE_WAVE4_DEVICE)
        } else {
            (58, DECODE_KERNEL, DECODE_DEVICE)
        }
    } else if force_row8 {
        (59, PREFILL_KERNEL, PREFILL_DEVICE)
    } else if force_col8 {
        (61, PREFILL_COL8_KERNEL, PREFILL_COL8_DEVICE)
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
    let mut observed_dispatch_count = 0_u32;
    let mut observed_kernel_id = 0_u32;
    let mut observed_kernel_symbol = String::new();
    let mut observed_device_symbol = String::new();
    for iteration in 0..total_count {
        let mut submission = session
            .submit(&prepared, queue)
            .map_err(|error| error.to_string())?;
        let dispatch = submission.dispatch().clone();
        observed_dispatch_count = dispatch.dispatch_count;
        observed_kernel_id = dispatch.kernel_id;
        observed_kernel_symbol = dispatch.kernel_symbol.clone();
        observed_device_symbol = dispatch.device_symbol.clone();
        wait_ok(
            |query| {
                if query {
                    submission.query()
                } else {
                    submission.wait(WAIT)
                }
            },
            "W4A4 matmul",
        )?;
        let dispatch_valid = (if baseline_oracle_control {
            // The control run intentionally accepts whichever production
            // selector is active, while retaining the same dispatch and
            // fail-closed checks as the baseline run.
            dispatch.dispatch_count > 0
        } else if force_baseline {
            dispatch.dispatch_count
                == 1 + u32::try_from((shape.m * shape.n).div_ceil(16_777_215))
                    .map_err(|_| "baseline dispatch count overflowed".to_owned())?
                && dispatch.kernel_id == expected_kernel_id
                && dispatch.kernel_symbol == expected_kernel
                && dispatch.device_symbol == expected_device
        } else {
            dispatch.dispatch_count == 2
                && dispatch.kernel_id == expected_kernel_id
                && dispatch.kernel_symbol == expected_kernel
                && dispatch.device_symbol == expected_device
        }) && dispatch.target == target
            && !dispatch.fallback_allowed
            && !dispatch.fallback_used;
        if !dispatch_valid {
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
    wait_ok(
        |query| {
            if query {
                readback.query()
            } else {
                readback.wait(WAIT)
            }
        },
        "readback",
    )?;
    let mut bytes = vec![0_u8; output_bytes];
    readback
        .read_into(&mut bytes)
        .map_err(|error| error.to_string())?;

    // The complete output is read for the regression digest.  Scan it before
    // sampling so a large oracle run still fails closed on any non-finite GPU
    // value outside the sampled coordinates.
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        if !from_bf16(u16::from_le_bytes([pair[0], pair[1]])).is_finite() {
            return Err(format!(
                "non-finite GPU output m={} k={} n={} output_index={index}",
                shape.m, shape.k, shape.n
            ));
        }
    }

    let coordinates = if sampled_oracle {
        sampled_coordinates(shape)
    } else if benchmark {
        (0..shape.m)
            .flat_map(|row| {
                (0..shape.n).filter_map(move |column| {
                    let row_selected =
                        matches!(row, 0 | 1) || row == shape.m / 2 || row + 1 == shape.m;
                    let column_selected = matches!(column, 0 | 1)
                        || column == shape.n / 3
                        || column == (shape.n * 2) / 3
                        || column + 1 == shape.n;
                    (row_selected && column_selected).then_some([row, column])
                })
            })
            .collect::<Vec<_>>()
    } else {
        (0..shape.m)
            .flat_map(|row| (0..shape.n).map(move |column| [row, column]))
            .collect::<Vec<_>>()
    };
    let mut max_abs_error = 0.0_f32;
    let mut max_relative_error = 0.0_f32;
    let mut oracle_values = Vec::with_capacity(coordinates.len());
    for [row, column] in coordinates.iter().copied() {
        let expected = independent_fp32_dot(
            &activation,
            &activation_scales,
            &quantized_weight,
            row,
            column,
        );
        let index = (row * shape.n + column) * 2;
        let actual = from_bf16(u16::from_le_bytes([bytes[index], bytes[index + 1]]));
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(1.0);
        max_abs_error = max_abs_error.max(absolute);
        max_relative_error = max_relative_error.max(relative);
        oracle_values.push(expected);
        if !expected.is_finite() {
            return Err(format!(
                "non-finite FP32 oracle m={} k={} n={} row={row} column={column}",
                shape.m, shape.k, shape.n
            ));
        }
        if relative > 0.02 {
            return Err(format!(
                "numerical mismatch m={} k={} n={} row={row} column={column} expected={expected} actual={actual} relative={relative}",
                shape.m, shape.k, shape.n
            ));
        }
    }
    let output_bf16_sha256 = format!("sha256:{:x}", Sha256::digest(&bytes));
    let regression_coordinates = sampled_coordinates(shape);
    let regression_oracle_values = if sampled_oracle {
        oracle_values.clone()
    } else {
        regression_coordinates
            .iter()
            .map(|[row, column]| {
                independent_fp32_dot(
                    &activation,
                    &activation_scales,
                    &quantized_weight,
                    *row,
                    *column,
                )
            })
            .collect::<Vec<_>>()
    };
    let sampled_output_bf16_sha256 =
        Some(sampled_digest(shape, &regression_coordinates, &bytes, None));
    let sampled_oracle_fp32_sha256 = Some(sampled_digest(
        shape,
        &regression_coordinates,
        &bytes,
        Some(&regression_oracle_values),
    ));
    if sampled_oracle {
        eprintln!(
            "[baseline oracle] checked {}x{}x{} kernel={} samples={}",
            shape.m,
            shape.k,
            shape.n,
            observed_kernel_id,
            coordinates.len()
        );
    }
    Ok(CaseReport {
        m: shape.m,
        k: shape.k,
        n: shape.n,
        dispatch_count: observed_dispatch_count,
        kernel_id: observed_kernel_id,
        kernel_symbol: observed_kernel_symbol,
        device_symbol: observed_device_symbol,
        kernel_elapsed_ns,
        kernel_elapsed_samples_ns,
        warmup_count,
        measured_count,
        input_decode_global,
        max_abs_error,
        max_relative_error,
        output_bf16_sha256,
        sampled_coordinates: Some(regression_coordinates),
        sampled_output_bf16_sha256,
        sampled_oracle_fp32_sha256,
    })
}

fn run(device_index: u32, target: String) -> Result<Report, String> {
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("target must be gfx1030 or gfx1201".to_owned());
    }
    let baseline_oracle = env::var(BASELINE_ORACLE_ENV).as_deref() == Ok("1");
    let baseline_oracle_control = env::var(BASELINE_ORACLE_CONTROL_ENV).as_deref() == Ok("default");
    if baseline_oracle_control && !baseline_oracle {
        return Err(format!(
            "{BASELINE_ORACLE_CONTROL_ENV}=default requires {BASELINE_ORACLE_ENV}=1"
        ));
    }
    if env::var_os(BASELINE_ORACLE_CONTROL_ENV).is_some() && !baseline_oracle_control {
        return Err(format!(
            "{BASELINE_ORACLE_CONTROL_ENV} must be default when set"
        ));
    }
    let (oracle_cases, shape_override) = if baseline_oracle {
        baseline_oracle_cases()?
    } else {
        if env::var_os(BASELINE_ORACLE_SHAPE_ENV).is_some() {
            return Err(format!(
                "{BASELINE_ORACLE_SHAPE_ENV} requires {BASELINE_ORACLE_ENV}=1"
            ));
        }
        (Vec::new(), None)
    };
    if baseline_oracle && !baseline_oracle_control {
        // SAFETY: this process configures the selector before connecting the
        // HIP backend, which is the first operation that can create runtime
        // worker threads.  The oracle flag is intentionally process-scoped.
        unsafe {
            env::set_var("SLLM_NVFP4_W4A4_FORCE_BASELINE", "1");
        }
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
        let cases = if baseline_oracle {
            oracle_cases.clone()
        } else if env::var("SLLM_NVFP4_BENCHMARK").as_deref() == Ok("1") {
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
            .map(|(index, shape)| {
                run_case(
                    &session,
                    &queue,
                    shape,
                    index,
                    &target,
                    baseline_oracle,
                    baseline_oracle_control,
                )
            })
            .collect::<Result<Vec<_>, _>>()
    })();
    let allocation_before_shutdown = session.memory_snapshot();
    let cleanup = session.shutdown(SHUTDOWN);
    let cases = result.map_err(|error| format!("{error}; post-error cleanup: {cleanup:?}"))?;
    let cleanup = cleanup.map_err(|error| error.to_string())?;
    if cleanup.retryable_cleanup != 0
        || cleanup.durable_quarantine != 0
        || allocation_before_shutdown.current_bytes() != 0
        || allocation_before_shutdown.poisoned()
    {
        return Err("nonzero cleanup state".to_owned());
    }
    Ok(Report {
        schema_version: "phase16f-nvfp4-w4a4-v2",
        state: "PASS",
        target,
        device_index,
        provider: "dynamic-block16-w4a4-decode-row8-prefill",
        arithmetic: "E2M1xE2M1/FP32-accumulate/BF16-output",
        fallback_allowed: false,
        fallback_used: false,
        baseline_oracle,
        baseline_oracle_control,
        shape_override,
        cases,
        current_bytes_before_shutdown: allocation_before_shutdown.current_bytes(),
        poisoned_before_shutdown: allocation_before_shutdown.poisoned(),
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
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
