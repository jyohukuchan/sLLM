//! Model-free OCP MXFP8 E4M3 W8A8 and MXFP6 E3M2 W6A6 GPU oracle.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, Encoding, ExecutionSessionRequest, ExecutionState,
    MxElementFormat, QuantizedMx, SemanticOpDescriptor, SemanticOpKind, TensorView,
    quantize_mxfp6_e3m2, quantize_mxfp8_e4m3,
};
use sllm_hip::{Context, HipBackend};

const WAIT: Duration = Duration::from_secs(60);
const SHUTDOWN: Duration = Duration::from_secs(16);

#[derive(Clone, Copy, Eq, PartialEq)]
enum Format {
    Mxfp8,
    Mxfp6,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Self::Mxfp8 => "mxfp8-e4m3-w8a8",
            Self::Mxfp6 => "mxfp6-e3m2-w6a6",
        }
    }

    fn quantize(self, values: &[f32], rows: usize, columns: usize) -> Result<QuantizedMx, String> {
        match self {
            Self::Mxfp8 => quantize_mxfp8_e4m3(values, rows, columns),
            Self::Mxfp6 => quantize_mxfp6_e3m2(values, rows, columns),
        }
        .map_err(|error| error.to_string())
    }

    fn view(self, n: usize, k: usize) -> Result<TensorView, String> {
        let (dtype, encoding) = match self {
            Self::Mxfp8 => (
                DType::F8E4M3Fn,
                Encoding::Mxfp8W8A8 {
                    block_size: 32,
                    scale_dtype: DType::U8,
                },
            ),
            Self::Mxfp6 => (
                DType::U8,
                Encoding::Mxfp6W6A6 {
                    block_size: 32,
                    scale_dtype: DType::U8,
                },
            ),
        };
        TensorView::with_encoding(dtype, encoding, &[n, k]).map_err(|error| error.to_string())
    }

    fn dispatch(self, m: usize) -> (u32, &'static str, &'static str) {
        match (self, m == 1) {
            (Self::Mxfp8, true) => (
                18,
                "matmul.mxfp8.w8a8.e4m3.block32.decode.v1",
                "sllm_matmul_mxfp8_w8a8_e4m3_block32_decode_v1",
            ),
            (Self::Mxfp8, false) => (
                22,
                "matmul.mxfp8.w8a8.e4m3.block32.prefill.row8.v2",
                "sllm_matmul_mxfp8_w8a8_e4m3_block32_prefill_row8_v2",
            ),
            (Self::Mxfp6, true) => (
                20,
                "matmul.mxfp6.w6a6.e3m2.block32.decode.v1",
                "sllm_matmul_mxfp6_w6a6_e3m2_block32_decode_v1",
            ),
            (Self::Mxfp6, false) => (
                25,
                "matmul.mxfp6.w6a6.e3m2.block32.prefill.tiled16.v3",
                "sllm_matmul_mxfp6_w6a6_e3m2_block32_prefill_tiled16_v3",
            ),
        }
    }
}

#[derive(Serialize)]
struct CaseReport {
    format: &'static str,
    m: usize,
    k: usize,
    n: usize,
    kernel_id: u32,
    kernel_symbol: String,
    device_symbol: String,
    kernel_elapsed_ns: u64,
    weight_value_sha256: String,
    weight_scale_sha256: String,
    output_bf16_sha256: String,
    max_abs_error: f32,
    max_relative_error: f32,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    block_size: usize,
    scale: &'static str,
    rounding: &'static str,
    accumulation: &'static str,
    fallback_allowed: bool,
    fallback_used: bool,
    cases: Vec<CaseReport>,
    retryable_cleanup: usize,
    durable_quarantine: usize,
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

fn matrix(rows: usize, columns: usize, phase: usize) -> Vec<u16> {
    (0..rows * columns)
        .map(|index| {
            let base = (((index * 37 + phase * 19) % 257) as i32 - 128) as f32 / 17.0;
            bf16(if index % 29 == 0 { base * 17.0 } else { base })
        })
        .collect()
}

fn words_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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

#[derive(Clone, Copy)]
struct CaseSpec {
    format: Format,
    m: usize,
    k: usize,
    n: usize,
    phase: usize,
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    target: &str,
    spec: CaseSpec,
) -> Result<CaseReport, String> {
    let CaseSpec {
        format,
        m,
        k,
        n,
        phase,
    } = spec;
    let activation_words = matrix(m, k, phase);
    let weight_words = matrix(n, k, phase + 11);
    let activation_source: Vec<_> = activation_words.iter().copied().map(from_bf16).collect();
    let weight_source: Vec<_> = weight_words.iter().copied().map(from_bf16).collect();
    let activation_quantized = format.quantize(&activation_source, m, k)?;
    let weight_quantized = format.quantize(&weight_source, n, k)?;
    if (format == Format::Mxfp8 && weight_quantized.format() != MxElementFormat::E4M3Fn)
        || (format == Format::Mxfp6 && weight_quantized.format() != MxElementFormat::E3M2)
    {
        return Err("host MX format identity differs".to_owned());
    }
    let activation_decoded = activation_quantized
        .dequantize()
        .map_err(|e| e.to_string())?;
    let weight_decoded = weight_quantized.dequantize().map_err(|e| e.to_string())?;
    let mut resident = weight_quantized.values().to_vec();
    resident.extend_from_slice(weight_quantized.scales());
    let activation_bytes = words_bytes(&activation_words);
    let output_len = m * n * 2;
    let activation_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    let weight_buffer = session
        .allocate(resident.len() as u64)
        .map_err(|error| error.to_string())?;
    let output_buffer = session
        .allocate(output_len as u64)
        .map_err(|error| error.to_string())?;
    for (label, buffer, bytes) in [
        (
            "activation upload",
            &activation_buffer,
            activation_bytes.as_slice(),
        ),
        ("weight upload", &weight_buffer, resident.as_slice()),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|e| e.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|error| error.to_string())?;
        wait_ok(upload.wait(WAIT), label)?;
    }
    let activation_view =
        TensorView::contiguous(DType::Bf16, &[m, k]).map_err(|e| e.to_string())?;
    let weight_view = format.view(n, k)?;
    let output_view = TensorView::contiguous(DType::Bf16, &[m, n]).map_err(|e| e.to_string())?;
    let semantic = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation_view.clone(), weight_view.clone()],
            vec![output_view.clone()],
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            semantic,
            vec![
                session
                    .bind(&activation_buffer, activation_view, AccessMode::Read)
                    .map_err(|e| e.to_string())?,
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .map_err(|e| e.to_string())?,
            ],
            vec![
                session
                    .bind(&output_buffer, output_view, AccessMode::Write)
                    .map_err(|e| e.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| error.to_string())?;
    let dispatch = submission.dispatch().clone();
    wait_ok(submission.wait(WAIT), format.name())?;
    let (kernel_id, kernel_symbol, device_symbol) = format.dispatch(m);
    if dispatch.dispatch_count != 2
        || dispatch.kernel_id != kernel_id
        || dispatch.kernel_symbol != kernel_symbol
        || dispatch.device_symbol != device_symbol
        || dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
    {
        return Err(format!(
            "unexpected {} dispatch: {dispatch:?}",
            format.name()
        ));
    }
    let elapsed = submission
        .kernel_elapsed_ns()
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "GPU timing is absent".to_owned())?;
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|e| e.to_string())?;
    wait_ok(readback.wait(WAIT), "output readback")?;
    let mut output = vec![0_u8; output_len];
    readback.read_into(&mut output).map_err(|e| e.to_string())?;
    let mut max_abs_error = 0.0_f32;
    let mut max_relative_error = 0.0_f32;
    for row in 0..m {
        for column in 0..n {
            let expected = (0..k)
                .map(|inner| {
                    activation_decoded[row * k + inner] * weight_decoded[column * k + inner]
                })
                .sum::<f32>();
            let index = (row * n + column) * 2;
            let actual = from_bf16(u16::from_le_bytes([output[index], output[index + 1]]));
            let absolute = (actual - expected).abs();
            let relative = absolute / expected.abs().max(1.0);
            max_abs_error = max_abs_error.max(absolute);
            max_relative_error = max_relative_error.max(relative);
            if !actual.is_finite() || relative > 0.02 {
                return Err(format!(
                    "{} mismatch row={row} column={column}: expected={expected} actual={actual} relative={relative}",
                    format.name()
                ));
            }
        }
    }
    Ok(CaseReport {
        format: format.name(),
        m,
        k,
        n,
        kernel_id,
        kernel_symbol: kernel_symbol.to_owned(),
        device_symbol: device_symbol.to_owned(),
        kernel_elapsed_ns: elapsed,
        weight_value_sha256: digest(weight_quantized.values()),
        weight_scale_sha256: digest(weight_quantized.scales()),
        output_bf16_sha256: digest(&output),
        max_abs_error,
        max_relative_error,
    })
}

fn run(device_index: u32, target: String) -> Result<Report, String> {
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("target must be exactly gfx1030 or gfx1201".to_owned());
    }
    let device = Context::query_device(device_index).map_err(|error| error.to_string())?;
    if device.gcn_arch_name != target {
        return Err(format!(
            "device {device_index} is {}, requested {target}",
            device.gcn_arch_name
        ));
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
        let mut cases = Vec::new();
        for (format, phase) in [(Format::Mxfp8, 0), (Format::Mxfp6, 7)] {
            cases.push(run_case(
                &session,
                &queue,
                &target,
                CaseSpec {
                    format,
                    m: 1,
                    k: 32,
                    n: 7,
                    phase,
                },
            )?);
            cases.push(run_case(
                &session,
                &queue,
                &target,
                CaseSpec {
                    format,
                    m: 3,
                    k: 64,
                    n: 5,
                    phase: phase + 1,
                },
            )?);
            // Qwen3.5-4B's GDN in_proj_b shape overlaps the gfx1030 BF16
            // short-mixed selector.  Keep this non-aligned M regression so a
            // quantized plan can never borrow the incompatible BF16
            // workspace sizing again.
            cases.push(run_case(
                &session,
                &queue,
                &target,
                CaseSpec {
                    format,
                    m: 17,
                    k: 2560,
                    n: 32,
                    phase: phase + 2,
                },
            )?);
        }
        Ok::<_, String>(cases)
    })();
    let cleanup = session
        .shutdown(SHUTDOWN)
        .map_err(|error| error.to_string())?;
    let cases = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("nonzero cleanup state".to_owned());
    }
    Ok(Report {
        schema_version: "sllm-ocp-mxfp8-mxfp6-wa-gpu-v1",
        state: "PASS",
        target,
        device_index,
        block_size: 32,
        scale: "E8M0",
        rounding: "roundTiesToEven-saturate",
        accumulation: "FP32",
        fallback_allowed: false,
        fallback_used: false,
        cases,
        retryable_cleanup: cleanup.retryable_cleanup,
        durable_quarantine: cleanup.durable_quarantine,
    })
}

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let device_index = match arguments.next().as_deref().unwrap_or("0").parse::<u32>() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid device index: {error}");
            return ExitCode::FAILURE;
        }
    };
    let target = arguments.next().unwrap_or_else(|| "gfx1030".to_owned());
    match run(device_index, target) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("serialize evidence: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("MX W/A evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
