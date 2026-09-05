//! Focused numerical evidence for the public FP8 matmul path.
//!
//! gfx1201 consumes OCP E4M3FN, gfx942 consumes numerically converted CDNA3
//! E4M3FNUZ, and gfx1030 uses software providers. Phase 78 candidate ID70
//! expands exact OCP E4M3FN values into transient FP16, runs rocBLAS with FP32
//! accumulation, then applies the outer-vector scales in a BF16 epilogue.
//! ID71 is the direct gfx1030 64x64x32 half2 prefill candidate. CPU work is an
//! oracle only.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, Encoding, ExecutionSessionRequest, ExecutionState,
    Fp8ResidentRepresentation, Fp8ScaleGranularity, SemanticOpDescriptor, SemanticOpKind,
    TensorView, convert_e4m3fn_to_e4m3fnuz, decode_e4m3fnuz, encode_e4m3fnuz,
    quantize_e4m3fn_outer_rows,
};
use sllm_hip::HipBackend;

const WAIT: Duration = Duration::from_secs(120);
const SHUTDOWN: Duration = Duration::from_secs(16);

#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    k: usize,
    n: usize,
}

const CASES: [Shape; 10] = [
    Shape { m: 1, k: 64, n: 31 },
    Shape { m: 1, k: 64, n: 32 },
    Shape { m: 1, k: 64, n: 33 },
    Shape {
        m: 1,
        k: 128,
        n: 256,
    },
    Shape {
        m: 3,
        k: 128,
        n: 256,
    },
    Shape {
        m: 17,
        k: 31,
        n: 32,
    },
    Shape {
        m: 128,
        k: 33,
        n: 16,
    },
    Shape {
        m: 128,
        k: 16,
        n: 16,
    },
    Shape {
        m: 512,
        k: 31,
        n: 32,
    },
    Shape {
        m: 1024,
        k: 33,
        n: 16,
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
    max_abs_error: f32,
    max_relative_error: f32,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    resident_dtype: &'static str,
    provider: &'static str,
    fallback_allowed: bool,
    fallback_used: bool,
    cases: Vec<CaseReport>,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

fn quantize_e4m3fnuz_outer_rows(input: &[f32], rows: usize, columns: usize) -> (Vec<u8>, Vec<f32>) {
    let mut values = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(rows);
    for row in input.chunks_exact(columns).take(rows) {
        let maximum = row.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 240.0 };
        scales.push(scale);
        values.extend(row.iter().map(|value| encode_e4m3fnuz(*value / scale)));
    }
    (values, scales)
}

fn dequantize_e4m3fnuz(values: &[u8], scales: &[f32], columns: usize) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| decode_e4m3fnuz(*value) * scales[index / columns])
        .collect()
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

fn wait_ok(
    state: Result<ExecutionState, sllm_core::ExecutionError>,
    label: &str,
) -> Result<(), String> {
    match state.map_err(|error| format!("{label}: {error}"))? {
        ExecutionState::Success => Ok(()),
        other => Err(format!("{label}: unexpected state {other:?}")),
    }
}

fn make_matrix(rows: usize, columns: usize, phase: usize) -> Vec<u16> {
    (0..rows * columns)
        .map(|index| {
            let signed = ((index * 37 + phase * 19) % 257) as i32 - 128;
            bf16(signed as f32 / 31.0)
        })
        .collect()
}

fn median_u64(values: &[u64]) -> Result<u64, String> {
    if values.is_empty() {
        return Err("cannot compute median of empty timing samples".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    Ok(sorted[sorted.len() / 2])
}

fn filter_benchmark_shape(cases: Vec<Shape>) -> Result<Vec<Shape>, String> {
    let Ok(value) = env::var("SLLM_FP8_BENCHMARK_SHAPE") else {
        return Ok(cases);
    };
    let dimensions = value
        .split('x')
        .map(|part| {
            part.parse::<usize>().map_err(|_| {
                "SLLM_FP8_BENCHMARK_SHAPE must be KxN or MxKxN with decimal dimensions".to_owned()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !matches!(dimensions.len(), 2 | 3) {
        return Err(
            "SLLM_FP8_BENCHMARK_SHAPE must be KxN or MxKxN with decimal dimensions".to_owned(),
        );
    }
    let filtered = cases
        .iter()
        .filter(|shape| match dimensions.as_slice() {
            [k, n] => shape.k == *k && shape.n == *n,
            [m, k, n] => shape.m == *m && shape.k == *k && shape.n == *n,
            _ => false,
        })
        .copied()
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        return Err(format!(
            "SLLM_FP8_BENCHMARK_SHAPE={value:?} does not match a benchmark case"
        ));
    }
    Ok(filtered)
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    shape: Shape,
    case_index: usize,
    target: &str,
) -> Result<CaseReport, String> {
    let activation_words = make_matrix(shape.m, shape.k, case_index);
    let weight_words = make_matrix(shape.n, shape.k, case_index + 7);
    let activation_f32 = activation_words
        .iter()
        .copied()
        .map(from_bf16)
        .collect::<Vec<_>>();
    let weight_f32 = weight_words
        .iter()
        .copied()
        .map(from_bf16)
        .collect::<Vec<_>>();
    let activation_fp8 = quantize_e4m3fn_outer_rows(&activation_f32, shape.m, shape.k)
        .map_err(|error| format!("activation oracle quantization: {error}"))?;
    let weight_fp8 = quantize_e4m3fn_outer_rows(&weight_f32, shape.n, shape.k)
        .map_err(|error| format!("weight quantization: {error}"))?;
    let fnuz = target == "gfx942";

    let mut weight_storage = if fnuz {
        convert_e4m3fn_to_e4m3fnuz(&weight_fp8.values)
    } else {
        weight_fp8.values.clone()
    };
    for scale in &weight_fp8.scales {
        weight_storage.extend_from_slice(&scale.to_le_bytes());
    }
    let activation_bytes = words_bytes(&activation_words);
    let output_len = shape.m * shape.n * 2;
    let activation_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|error| format!("activation allocation: {error}"))?;
    let weight_buffer = session
        .allocate(weight_storage.len() as u64)
        .map_err(|error| format!("weight allocation: {error}"))?;
    let output_buffer = session
        .allocate(output_len as u64)
        .map_err(|error| format!("output allocation: {error}"))?;
    for (label, buffer, bytes) in [
        (
            "activation",
            &activation_buffer,
            activation_bytes.as_slice(),
        ),
        ("weight", &weight_buffer, weight_storage.as_slice()),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|error| format!("{label} upload: {error}"))?;
        wait_ok(upload.wait(WAIT), label)?;
    }

    let activation_view = TensorView::contiguous(DType::Bf16, &[shape.m, shape.k])
        .map_err(|error| error.to_string())?;
    let resident_dtype = if fnuz {
        DType::F8E4M3FnuZ
    } else {
        DType::F8E4M3Fn
    };
    let weight_view = TensorView::with_encoding(
        resident_dtype,
        Encoding::Fp8Scaled {
            granularity: Fp8ScaleGranularity::OuterDimension,
            scale_dtype: DType::F32,
            resident: Fp8ResidentRepresentation::PackedBytes,
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
        .map_err(|error| format!("prepare: {error}"))?;
    let benchmark = env::var("SLLM_FP8_BENCHMARK").as_deref() == Ok("1");
    let force_baseline = env::var("SLLM_FP8_OUTER_PREFILL_FORCE_BASELINE").as_deref() == Ok("1");
    let force_f16_staging =
        env::var("SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_F16_STAGING").as_deref() == Ok("1");
    let f16_staging_candidate = target == "gfx1030"
        && !force_baseline
        && force_f16_staging
        && shape.m >= 128
        && (shape.m % 128 == 0)
        && (1..=17_408).contains(&shape.k)
        && (shape.k % 16 == 0)
        && (1..=17_408).contains(&shape.n)
        && (shape.n % 16 == 0);
    let force_half2_64x64 =
        env::var("SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2_64X64").as_deref() == Ok("1");
    let force_half2_128x64 =
        env::var("SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2").as_deref() == Ok("1");
    let half2_64x64_candidate = target == "gfx1030"
        && !force_baseline
        && !f16_staging_candidate
        && shape.m > 1
        && (force_half2_64x64 || !(force_half2_128x64 && (shape.k % 2 == 0)));
    let focused_prefill_candidate = f16_staging_candidate || half2_64x64_candidate;
    let warmups = if benchmark {
        if focused_prefill_candidate { 5 } else { 3 }
    } else {
        0
    };
    let measured = if benchmark {
        if focused_prefill_candidate { 21 } else { 10 }
    } else {
        1
    };
    for _ in 0..warmups {
        let mut submission = session
            .submit(&prepared, queue)
            .map_err(|error| format!("warmup submit: {error}"))?;
        wait_ok(submission.wait(WAIT), "matmul warmup")?;
    }
    let mut dispatch = None;
    let mut last_submission = None;
    let mut kernel_elapsed_samples_ns = Vec::with_capacity(measured);
    for iteration in 0..measured {
        let mut submission = session
            .submit(&prepared, queue)
            .map_err(|error| format!("submit: {error}"))?;
        if dispatch.is_none() {
            dispatch = Some(submission.dispatch().clone());
        }
        wait_ok(submission.wait(WAIT), "matmul")?;
        kernel_elapsed_samples_ns.push(
            submission
                .kernel_elapsed_ns()
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "missing GPU timing".to_owned())?,
        );
        if iteration + 1 == measured {
            last_submission = Some(submission);
        }
    }
    let dispatch = dispatch.ok_or_else(|| "no measured dispatch".to_owned())?;
    let force_decode_baseline =
        env::var("SLLM_FP8_OUTER_DECODE_FORCE_BASELINE").as_deref() == Ok("1");
    let force_decode_half2 =
        env::var("SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_HALF2").as_deref() == Ok("1");
    let force_decode_dword8 =
        env::var("SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_DWORD8").as_deref() == Ok("1");
    let expected_kernel = if matches!(target, "gfx1201" | "gfx942") {
        5
    } else if shape.m == 1 {
        if !force_decode_baseline
            && force_decode_dword8
            && (64..=17_408).contains(&shape.k)
            && (shape.k % 64 == 0)
        {
            68
        } else if !force_decode_baseline
            && force_decode_half2
            && (64..=17_408).contains(&shape.k)
            && (shape.k % 64 == 0)
        {
            66
        } else {
            6
        }
    } else if force_baseline {
        6
    } else if f16_staging_candidate {
        70
    } else if half2_64x64_candidate {
        71
    } else if force_half2_128x64 && shape.k % 2 == 0 {
        63
    } else {
        60
    };
    let expected_dispatch_count = if expected_kernel == 70 { 4 } else { 2 };
    if dispatch.dispatch_count != expected_dispatch_count
        || dispatch.kernel_id != expected_kernel
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.target != target
    {
        return Err(format!("unexpected FP8 dispatch metadata: {dispatch:?}"));
    }
    let kernel_elapsed_ns = median_u64(&kernel_elapsed_samples_ns)?;
    let mut readback = last_submission
        .ok_or_else(|| "missing final submission".to_owned())?
        .start_output_readback(0)
        .map_err(|error| error.to_string())?;
    wait_ok(readback.wait(WAIT), "readback")?;
    let mut actual_bytes = vec![0_u8; output_len];
    readback
        .read_into(&mut actual_bytes)
        .map_err(|error| error.to_string())?;

    let (activation_dequantized, weight_dequantized) = if fnuz {
        let (activation_values, activation_scales) =
            quantize_e4m3fnuz_outer_rows(&activation_f32, shape.m, shape.k);
        let weight_values = convert_e4m3fn_to_e4m3fnuz(&weight_fp8.values);
        (
            dequantize_e4m3fnuz(&activation_values, &activation_scales, shape.k),
            dequantize_e4m3fnuz(&weight_values, &weight_fp8.scales, shape.k),
        )
    } else {
        (activation_fp8.dequantize(), weight_fp8.dequantize())
    };
    let mut max_abs_error = 0.0_f32;
    let mut max_relative_error = 0.0_f32;
    for row in 0..shape.m {
        if benchmark && !matches!(row, 0 | 1) && row != shape.m / 2 && row + 1 != shape.m {
            continue;
        }
        for column in 0..shape.n {
            if benchmark
                && !matches!(column, 0 | 1)
                && column != shape.n / 3
                && column != (shape.n * 2) / 3
                && column + 1 != shape.n
            {
                continue;
            }
            let expected = (0..shape.k).fold(0.0_f32, |sum, reduction| {
                sum + activation_dequantized[row * shape.k + reduction]
                    * weight_dequantized[column * shape.k + reduction]
            });
            let index = (row * shape.n + column) * 2;
            let actual = from_bf16(u16::from_le_bytes([
                actual_bytes[index],
                actual_bytes[index + 1],
            ]));
            let absolute = (actual - expected).abs();
            let relative = absolute / expected.abs().max(1.0);
            max_abs_error = max_abs_error.max(absolute);
            max_relative_error = max_relative_error.max(relative);
            if !actual.is_finite() || relative > 0.012 {
                return Err(format!(
                    "numerical mismatch row={row} column={column} expected={expected} actual={actual} relative={relative}"
                ));
            }
        }
    }
    Ok(CaseReport {
        m: shape.m,
        k: shape.k,
        n: shape.n,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        kernel_elapsed_ns,
        kernel_elapsed_samples_ns,
        max_abs_error,
        max_relative_error,
    })
}

fn run(device_index: u32, target: String) -> Result<Report, String> {
    if !matches!(target.as_str(), "gfx1030" | "gfx1201" | "gfx942") {
        return Err("target must be gfx1030, gfx1201, or gfx942".to_owned());
    }
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let request = ExecutionSessionRequest::new(device_index, target.clone())
        .map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| error.to_string())?;
    let cases_result = (|| {
        let queue = session.create_queue().map_err(|error| error.to_string())?;
        let benchmark = env::var("SLLM_FP8_BENCHMARK").as_deref() == Ok("1");
        let cases = if benchmark {
            vec![
                // Exact non-lm_head FP8 decode shapes in the locked
                // Qwen3.8-27B-NVFP4 artifact.  Keep the two historical
                // synthetic shapes below for continuity with earlier data.
                Shape {
                    m: 1,
                    k: 5120,
                    n: 1024,
                },
                Shape {
                    m: 1,
                    k: 6144,
                    n: 5120,
                },
                Shape {
                    m: 1,
                    k: 5120,
                    n: 6144,
                },
                Shape {
                    m: 1,
                    k: 5120,
                    n: 10240,
                },
                Shape {
                    m: 1,
                    k: 5120,
                    n: 12288,
                },
                Shape {
                    m: 1,
                    k: 2560,
                    n: 9216,
                },
                Shape {
                    m: 1,
                    k: 9216,
                    n: 2560,
                },
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
                    m: 32,
                    k: 2560,
                    n: 9216,
                },
                Shape {
                    m: 32,
                    k: 9216,
                    n: 2560,
                },
                // Representative exact prefill shapes from the locked model.
                // ID70 deliberately excludes the 248320-column vocabulary
                // projection and any non-128-aligned prompt tail.
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
            ]
        } else {
            CASES.to_vec()
        };
        let scope = env::var("SLLM_FP8_BENCHMARK_SCOPE").unwrap_or_else(|_| "all".to_owned());
        let cases = match scope.as_str() {
            "all" => cases,
            "decode" => cases.into_iter().filter(|shape| shape.m == 1).collect(),
            "prefill" => cases.into_iter().filter(|shape| shape.m > 1).collect(),
            _ => {
                return Err("SLLM_FP8_BENCHMARK_SCOPE must be all, decode, or prefill".to_owned());
            }
        };
        let cases = filter_benchmark_shape(cases)?;
        cases
            .iter()
            .copied()
            .enumerate()
            .map(|(index, shape)| run_case(&session, &queue, shape, index, &target))
            .collect::<Result<Vec<_>, _>>()
    })();
    let cleanup = session
        .shutdown(SHUTDOWN)
        .map_err(|error| error.to_string())?;
    let cases = cases_result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("nonzero cleanup state".to_owned());
    }
    let resident_dtype = if target == "gfx942" {
        "e4m3fnuz"
    } else {
        "e4m3fn"
    };
    Ok(Report {
        schema_version: "sllm-fp8-matmul-v2",
        state: "PASS",
        target,
        device_index,
        resident_dtype,
        provider: provider_name(&cases),
        fallback_allowed: false,
        fallback_used: false,
        cases,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
    })
}

fn expected_native(cases: &[CaseReport]) -> bool {
    !cases.is_empty() && cases.iter().all(|case| case.kernel_id == 5)
}

fn provider_name(cases: &[CaseReport]) -> &'static str {
    if expected_native(cases) {
        "hipblaslt-native"
    } else if cases.iter().any(|case| case.kernel_id == 70) {
        "f16-staging-rocblas"
    } else if cases.iter().any(|case| case.kernel_id == 71) {
        "gfx1030-half2-64x64-k32"
    } else {
        "gfx1030-software"
    }
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let result = match arguments.as_slice() {
        [device, target] => device
            .parse::<u32>()
            .map_err(|_| "device index must be u32".to_owned())
            .and_then(|device| run(device, target.clone())),
        _ => Err("usage: sllm-fp8-matmul-evidence DEVICE_INDEX gfx1030|gfx1201|gfx942".to_owned()),
    };
    match result {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("FP8 evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
