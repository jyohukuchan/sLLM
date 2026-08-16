//! Fail-closed OCP MXFP4 W4A4 numerical evidence for synthetic boundaries and
//! exact Qwen3.5-35B-A3B routed-expert planes.
//!
//! Host arithmetic is an independent oracle only. Production execution always
//! uses the two-dispatch HIP path: dynamic BF16->MXFP4 followed by packed W4A4.

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, Encoding, ExecutionSessionRequest, ExecutionState,
    Qwen35MoeExpertProjection, Qwen35MoeTensorPlane, SemanticOpDescriptor, SemanticOpKind,
    TensorView, verify_qwen35_moe_artifact,
};
use sllm_hip::HipBackend;

const WAIT: Duration = Duration::from_secs(60);
const SHUTDOWN: Duration = Duration::from_secs(16);
const DECODE_KERNEL: &str = "matmul.mxfp4.w4a4.block32.decode.v1";
const DECODE_DEVICE: &str = "sllm_matmul_mxfp4_w4a4_block32_decode_v1";
const PREFILL_KERNEL: &str = "matmul.mxfp4.w4a4.block32.prefill.v1";
const PREFILL_DEVICE: &str = "sllm_matmul_mxfp4_w4a4_block32_prefill_v1";

#[derive(Clone)]
struct Case {
    label: String,
    m: usize,
    k: usize,
    n: usize,
    activation: Vec<u16>,
    packed_weight: Vec<u8>,
    weight_scales: Vec<u8>,
}

#[derive(Serialize)]
struct CaseReport {
    label: String,
    m: usize,
    k: usize,
    n: usize,
    kernel_id: u32,
    kernel_symbol: String,
    kernel_elapsed_ns: u64,
    activation_packed_sha256: String,
    activation_scale_sha256: String,
    weight_packed_sha256: String,
    weight_scale_sha256: String,
    output_bf16_sha256: String,
    max_abs_error: f32,
    max_relative_error: f32,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    repository: &'static str,
    revision: &'static str,
    target: String,
    device_index: u32,
    provider: &'static str,
    arithmetic: &'static str,
    fallback_allowed: bool,
    fallback_used: bool,
    cases: Vec<CaseReport>,
    cleanup: CleanupReport,
}

#[derive(Serialize)]
struct CleanupReport {
    retryable_cleanup: usize,
    durable_quarantine: usize,
}

#[derive(Clone)]
struct QuantizedRows {
    packed: Vec<u8>,
    scales: Vec<u8>,
    decoded: Vec<f32>,
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

fn e8m0(code: u8) -> Result<f32, String> {
    match code {
        255 => Err("reserved E8M0 NaN scale".to_owned()),
        0 => Ok(f32::from_bits(0x0040_0000)),
        _ => Ok(f32::from_bits(u32::from(code) << 23)),
    }
}

fn even_scale_code(maximum: f32) -> Result<u8, String> {
    if !maximum.is_finite() || maximum < 0.0 {
        return Err("nonfinite MXFP4 block maximum".to_owned());
    }
    if maximum == 0.0 {
        return Ok(0);
    }
    let rounded_exponent = (maximum.to_bits() + 0x0020_0000) & 0x7f80_0000;
    Ok(((rounded_exponent >> 23) as i32 - 2).clamp(0, 254) as u8)
}

fn quantize_rows(words: &[u16], rows: usize, columns: usize) -> Result<QuantizedRows, String> {
    if words.len() != rows * columns || rows == 0 || columns == 0 {
        return Err("MXFP4 oracle shape differs".to_owned());
    }
    let packed_row = columns.div_ceil(2);
    let blocks = columns.div_ceil(32);
    let mut packed = vec![0_u8; rows * packed_row];
    let mut scales = vec![0_u8; rows * blocks];
    let mut decoded = vec![0.0_f32; words.len()];
    for row in 0..rows {
        for block in 0..blocks {
            let start = block * 32;
            let end = (start + 32).min(columns);
            let maximum = (start..end)
                .map(|column| from_bf16(words[row * columns + column]).abs())
                .fold(0.0_f32, f32::max);
            let code = even_scale_code(maximum)?;
            let scale = e8m0(code)?;
            scales[row * blocks + block] = code;
            for column in start..end {
                let value = from_bf16(words[row * columns + column]);
                let element = encode_e2m1(value / scale);
                let packed_index = row * packed_row + column / 2;
                if column & 1 == 0 {
                    packed[packed_index] = element;
                } else {
                    packed[packed_index] |= element << 4;
                }
                decoded[row * columns + column] = e2m1(element) * scale;
            }
        }
    }
    Ok(QuantizedRows {
        packed,
        scales,
        decoded,
    })
}

fn decode_rows(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>, String> {
    let packed_row = columns.div_ceil(2);
    let blocks = columns.div_ceil(32);
    if packed.len() != rows * packed_row || scales.len() != rows * blocks {
        return Err("MXFP4 source plane length differs".to_owned());
    }
    let mut decoded = vec![0.0_f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let pair = packed[row * packed_row + column / 2];
            let code = if column & 1 == 0 {
                pair & 0x0f
            } else {
                pair >> 4
            };
            decoded[row * columns + column] =
                e2m1(code) * e8m0(scales[row * blocks + column / 32])?;
        }
    }
    Ok(decoded)
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

fn matrix(rows: usize, columns: usize, phase: usize) -> Vec<u16> {
    (0..rows * columns)
        .map(|index| bf16((((index * 37 + phase * 19) % 257) as i32 - 128) as f32 / 31.0))
        .collect()
}

fn read_plane(root: &Path, plane: &Qwen35MoeTensorPlane) -> Result<Vec<u8>, String> {
    let [start, end] = plane.absolute_byte_range;
    let mut file = File::open(root.join(&plane.source_file)).map_err(|error| error.to_string())?;
    file.seek(SeekFrom::Start(start))
        .map_err(|error| error.to_string())?;
    let mut bytes = vec![0_u8; usize::try_from(end - start).map_err(|_| "plane too large")?];
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
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

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    case: Case,
    target: &str,
) -> Result<CaseReport, String> {
    let activation_q = quantize_rows(&case.activation, case.m, case.k)?;
    let weight_decoded = decode_rows(&case.packed_weight, &case.weight_scales, case.n, case.k)?;
    let mut resident = case.packed_weight.clone();
    resident.extend_from_slice(&case.weight_scales);
    let activation_bytes = words_bytes(&case.activation);
    let output_bytes = case.m * case.n * 2;
    let activation_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|e| e.to_string())?;
    let weight_buffer = session
        .allocate(resident.len() as u64)
        .map_err(|e| e.to_string())?;
    let output_buffer = session
        .allocate(output_bytes as u64)
        .map_err(|e| e.to_string())?;
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
                    .map_err(|e| e.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|e| e.to_string())?;
        wait_ok(upload.wait(WAIT), label)?;
    }
    let activation_view =
        TensorView::contiguous(DType::Bf16, &[case.m, case.k]).map_err(|e| e.to_string())?;
    let weight_view = TensorView::with_encoding(
        DType::U8,
        Encoding::Mxfp4W4A4 {
            block_size: 32,
            scale_dtype: DType::U8,
        },
        &[case.n, case.k],
    )
    .map_err(|e| e.to_string())?;
    let output_view =
        TensorView::contiguous(DType::Bf16, &[case.m, case.n]).map_err(|e| e.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation_view.clone(), weight_view.clone()],
            vec![output_view.clone()],
        )
        .map_err(|e| e.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
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
        .map_err(|e| e.to_string())?,
    );
    let prepared = session.prepare(operation).map_err(|e| e.to_string())?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|e| e.to_string())?;
    let dispatch = submission.dispatch().clone();
    wait_ok(submission.wait(WAIT), "MXFP4 matmul")?;
    let (kernel_id, kernel, device) = if case.m == 1 {
        (14, DECODE_KERNEL, DECODE_DEVICE)
    } else {
        (15, PREFILL_KERNEL, PREFILL_DEVICE)
    };
    if dispatch.dispatch_count != 2
        || dispatch.kernel_id != kernel_id
        || dispatch.kernel_symbol != kernel
        || dispatch.device_symbol != device
        || dispatch.target != target
        || dispatch.fallback_allowed
        || dispatch.fallback_used
    {
        return Err(format!("unexpected MXFP4 dispatch: {dispatch:?}"));
    }
    let kernel_elapsed_ns = submission
        .kernel_elapsed_ns()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "missing GPU timing".to_owned())?;
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|e| e.to_string())?;
    wait_ok(readback.wait(WAIT), "readback")?;
    let mut bytes = vec![0_u8; output_bytes];
    readback.read_into(&mut bytes).map_err(|e| e.to_string())?;
    let mut max_abs_error = 0.0_f32;
    let mut max_relative_error = 0.0_f32;
    for row in 0..case.m {
        for column in 0..case.n {
            let expected = (0..case.k)
                .map(|inner| {
                    activation_q.decoded[row * case.k + inner]
                        * weight_decoded[column * case.k + inner]
                })
                .sum::<f32>();
            let index = (row * case.n + column) * 2;
            let actual = from_bf16(u16::from_le_bytes([bytes[index], bytes[index + 1]]));
            let absolute = (actual - expected).abs();
            let relative = absolute / expected.abs().max(1.0);
            max_abs_error = max_abs_error.max(absolute);
            max_relative_error = max_relative_error.max(relative);
            if !actual.is_finite() || relative > 0.02 {
                return Err(format!(
                    "{} numerical mismatch row={row} column={column} expected={expected} actual={actual} relative={relative}",
                    case.label
                ));
            }
        }
    }
    Ok(CaseReport {
        label: case.label,
        m: case.m,
        k: case.k,
        n: case.n,
        kernel_id,
        kernel_symbol: kernel.to_owned(),
        kernel_elapsed_ns,
        activation_packed_sha256: digest(&activation_q.packed),
        activation_scale_sha256: digest(&activation_q.scales),
        weight_packed_sha256: digest(&case.packed_weight),
        weight_scale_sha256: digest(&case.weight_scales),
        output_bf16_sha256: digest(&bytes),
        max_abs_error,
        max_relative_error,
    })
}

fn synthetic_case(m: usize, k: usize, n: usize, phase: usize) -> Result<Case, String> {
    let weight = matrix(n, k, phase + 7);
    let quantized = quantize_rows(&weight, n, k)?;
    Ok(Case {
        label: format!("synthetic-m{m}-k{k}-n{n}"),
        m,
        k,
        n,
        activation: matrix(m, k, phase),
        packed_weight: quantized.packed,
        weight_scales: quantized.scales,
    })
}

fn run(device_index: u32, target: String, root: &Path) -> Result<Report, String> {
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("target must be gfx1030 or gfx1201".to_owned());
    }
    let model = verify_qwen35_moe_artifact(root).map_err(|e| e.to_string())?;
    let mut cases = vec![
        synthetic_case(1, 31, 17, 0)?,
        synthetic_case(3, 32, 16, 1)?,
        synthetic_case(7, 33, 15, 2)?,
    ];
    for (layer, expert, projection, m) in [
        (0, 0, Qwen35MoeExpertProjection::Gate, 1),
        (20, 128, Qwen35MoeExpertProjection::Up, 3),
        (39, 255, Qwen35MoeExpertProjection::Down, 7),
    ] {
        let tensor = model
            .expert(layer, expert, projection)
            .ok_or("reviewed expert absent")?;
        let [n, k] = tensor.logical_shape.map(|value| value as usize);
        cases.push(Case {
            label: format!("artifact-layer{layer}-expert{expert}-{projection:?}"),
            m,
            k,
            n,
            activation: matrix(m, k, usize::from(layer) + usize::from(expert)),
            packed_weight: read_plane(model.root(), &tensor.value)?,
            weight_scales: read_plane(model.root(), &tensor.scale)?,
        });
    }
    let backend = HipBackend::connect().map_err(|e| e.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    let result = (|| {
        let queue = session.create_queue().map_err(|e| e.to_string())?;
        cases
            .into_iter()
            .map(|case| run_case(&session, &queue, case, &target))
            .collect::<Result<Vec<_>, _>>()
    })();
    let cleanup = session.shutdown(SHUTDOWN).map_err(|e| e.to_string())?;
    let cases = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("nonzero cleanup state".to_owned());
    }
    Ok(Report {
        schema_version: "sllm-qwen35-moe-mxfp4-w4a4-v1",
        state: "PASS",
        repository: sllm_core::QWEN35_MOE_REPOSITORY,
        revision: sllm_core::QWEN35_MOE_REVISION,
        target,
        device_index,
        provider: "dynamic-even-block32-packed-decode-prefill",
        arithmetic: "E2M1xE2M1/E8M0-block32/FP32-accumulate/BF16-output",
        fallback_allowed: false,
        fallback_used: false,
        cases,
        cleanup: CleanupReport {
            retryable_cleanup: cleanup.retryable_cleanup,
            durable_quarantine: cleanup.durable_quarantine,
        },
    })
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let device_index = match args.next().as_deref().unwrap_or("0").parse::<u32>() {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid device index: {error}");
            return ExitCode::FAILURE;
        }
    };
    let target = args.next().unwrap_or_else(|| "gfx1201".to_owned());
    let Some(root) = env::var_os("SLLM_QWEN35_MOE_CACHE") else {
        eprintln!("SLLM_QWEN35_MOE_CACHE is required");
        return ExitCode::FAILURE;
    };
    match run(device_index, target, Path::new(&root)) {
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
            eprintln!("MXFP4 evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
