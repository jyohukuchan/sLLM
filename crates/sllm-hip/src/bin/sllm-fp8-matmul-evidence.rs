//! Focused numerical evidence for the Phase 10 public FP8 matmul path.
//!
//! Both providers consume the same OCP E4M3FN values and outer FP32 scales.
//! gfx1201 must report native hipBLASLt execution; gfx1030 must report the
//! explicit byte-decode emulation provider. CPU work is an oracle only.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, Encoding, ExecutionSessionRequest, ExecutionState,
    Fp8ResidentRepresentation, Fp8ScaleGranularity, SemanticOpDescriptor, SemanticOpKind,
    TensorView, quantize_e4m3fn_outer_rows,
};
use sllm_hip::HipBackend;

const WAIT: Duration = Duration::from_secs(30);
const SHUTDOWN: Duration = Duration::from_secs(16);

#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    k: usize,
    n: usize,
}

const CASES: [Shape; 2] = [
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
    fallback_allowed: bool,
    fallback_used: bool,
    cases: Vec<CaseReport>,
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

    let mut weight_storage = weight_fp8.values.clone();
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
    let weight_view = TensorView::with_encoding(
        DType::F8E4M3Fn,
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
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| format!("submit: {error}"))?;
    let dispatch = submission.dispatch().clone();
    let expected_kernel = if target == "gfx1201" { 5 } else { 6 };
    if dispatch.dispatch_count != 2
        || dispatch.kernel_id != expected_kernel
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.target != target
    {
        return Err(format!("unexpected FP8 dispatch metadata: {dispatch:?}"));
    }
    wait_ok(submission.wait(WAIT), "matmul")?;
    let kernel_elapsed_ns = submission
        .kernel_elapsed_ns()
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "missing GPU timing".to_owned())?;
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| error.to_string())?;
    wait_ok(readback.wait(WAIT), "readback")?;
    let mut actual_bytes = vec![0_u8; output_len];
    readback
        .read_into(&mut actual_bytes)
        .map_err(|error| error.to_string())?;

    let activation_dequantized = activation_fp8.dequantize();
    let weight_dequantized = weight_fp8.dequantize();
    let mut max_abs_error = 0.0_f32;
    let mut max_relative_error = 0.0_f32;
    for row in 0..shape.m {
        for column in 0..shape.n {
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
        max_abs_error,
        max_relative_error,
    })
}

fn run(device_index: u32, target: String) -> Result<Report, String> {
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("target must be gfx1030 or gfx1201".to_owned());
    }
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let request = ExecutionSessionRequest::new(device_index, target.clone())
        .map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| error.to_string())?;
    let cases_result = (|| {
        let queue = session.create_queue().map_err(|error| error.to_string())?;
        CASES
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
    Ok(Report {
        schema_version: "phase10-fp8-matmul-v1",
        state: "PASS",
        target,
        device_index,
        provider: if expected_native(&cases) {
            "hipblaslt-native"
        } else {
            "byte-decode-emulation"
        },
        fallback_allowed: false,
        fallback_used: false,
        cases,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
    })
}

fn expected_native(cases: &[CaseReport]) -> bool {
    cases.first().is_some_and(|case| case.kernel_id == 5)
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let result = match arguments.as_slice() {
        [device, target] => device
            .parse::<u32>()
            .map_err(|_| "device index must be u32".to_owned())
            .and_then(|device| run(device, target.clone())),
        _ => Err("usage: sllm-fp8-matmul-evidence DEVICE_INDEX gfx1030|gfx1201".to_owned()),
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
