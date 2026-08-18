//! Phase 22 BF16 M=1 matvec shape profile.
//!
//! Each reviewed Qwen3.5-4B decode shape reuses one prepared operation and
//! device-resident operands for three warmups plus ten measured submissions.
//! Transfer and preparation time are deliberately outside the HIP event
//! samples.  The deterministic all-one input has the exact BF16 result `K`,
//! so the final device output is still checked without a CPU O(K*N) oracle.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, ExecutionSessionRequest, ExecutionState,
    SemanticOpDescriptor, SemanticOpKind, TensorView,
};
use sllm_hip::HipBackend;

const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);
const WARMUPS: usize = 3;
const MEASURED: usize = 10;
const BF16_ONE: u16 = 0x3f80;

#[derive(Clone, Copy)]
struct Shape {
    role: &'static str,
    k: usize,
    n: usize,
    calls_per_token: u32,
}

const SHAPES: [Shape; 8] = [
    Shape {
        role: "mlp_gate_up",
        k: 2_560,
        n: 9_216,
        calls_per_token: 64,
    },
    Shape {
        role: "mlp_down",
        k: 9_216,
        n: 2_560,
        calls_per_token: 32,
    },
    Shape {
        role: "gdn_full_q",
        k: 2_560,
        n: 8_192,
        calls_per_token: 32,
    },
    Shape {
        role: "gdn_z",
        k: 2_560,
        n: 4_096,
        calls_per_token: 24,
    },
    Shape {
        role: "gdn_full_out",
        k: 4_096,
        n: 2_560,
        calls_per_token: 32,
    },
    Shape {
        role: "full_k_v",
        k: 2_560,
        n: 1_024,
        calls_per_token: 16,
    },
    Shape {
        role: "gdn_b_a",
        k: 2_560,
        n: 32,
        calls_per_token: 48,
    },
    Shape {
        role: "tied_vocabulary",
        k: 2_560,
        n: 248_320,
        calls_per_token: 1,
    },
];

struct Config {
    device_index: u32,
    target: String,
}

#[derive(Serialize)]
struct ShapeEvidence {
    role: &'static str,
    m: usize,
    k: usize,
    n: usize,
    calls_per_token: u32,
    effective_bytes_per_call: u64,
    kernel_id: u32,
    kernel_symbol: String,
    device_symbol: String,
    workgroup_size_x: u32,
    grid_size_x: u32,
    warmups: usize,
    measured: usize,
    kernel_elapsed_ns: Vec<u64>,
    output_exact: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    cpu_fallback_used: bool,
    fallback_allowed: bool,
    fallback_used: bool,
    production_completion_mode: &'static str,
    warmups: usize,
    measured: usize,
    shapes: Vec<ShapeEvidence>,
    allocation_current_bytes: u64,
    allocation_peak_bytes: u64,
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
                device_index = Some(
                    arguments
                        .next()
                        .ok_or_else(|| "--device-index requires a value".to_owned())?
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
            }
            "--target" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "--target requires a value".to_owned())?;
                if !matches!(value.as_str(), "gfx1030" | "gfx1201") {
                    return Err("--target must be gfx1030 or gfx1201".to_owned());
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

fn repeated_bf16_bytes(word: u16, elements: usize) -> Result<Vec<u8>, String> {
    let byte_count = elements
        .checked_mul(2)
        .ok_or_else(|| "BF16 byte count overflowed usize".to_owned())?;
    let bytes = word.to_le_bytes();
    let mut output = vec![0_u8; byte_count];
    for chunk in output.chunks_exact_mut(2) {
        chunk.copy_from_slice(&bytes);
    }
    Ok(output)
}

fn upload_repeated_bf16(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    buffer: &sllm_core::ExecutionBuffer,
    elements: usize,
    word: u16,
    label: &str,
) -> Result<(), String> {
    let total = elements
        .checked_mul(2)
        .ok_or_else(|| format!("{label} byte count overflowed usize"))?;
    let transfer_limit = usize::try_from(
        session
            .max_transfer_bytes()
            .map_err(|error| format!("query transfer limit failed: {error}"))?,
    )
    .map_err(|_| "transfer limit does not fit usize".to_owned())?;
    let chunk_bytes = transfer_limit - transfer_limit % 2;
    if chunk_bytes == 0 {
        return Err("transfer limit cannot hold one BF16 value".to_owned());
    }
    let full_chunk = Arc::<[u8]>::from(repeated_bf16_bytes(word, chunk_bytes / 2)?);
    let mut offset = 0_usize;
    while offset < total {
        let length = (total - offset).min(chunk_bytes);
        let payload = if length == chunk_bytes {
            Arc::clone(&full_chunk)
        } else {
            Arc::<[u8]>::from(repeated_bf16_bytes(word, length / 2)?)
        };
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(offset as u64, length as u64)
                    .map_err(|error| format!("{label} range failed: {error}"))?,
                payload,
            )
            .map_err(|error| format!("{label} upload failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), label)?;
        offset += length;
    }
    Ok(())
}

fn expected_sum_word(k: usize) -> Result<u16, String> {
    let value = k as f32;
    if !value.is_finite() || value > 65_280.0 {
        return Err("expected sum is outside finite BF16 range".to_owned());
    }
    let bits = value.to_bits();
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    Ok((upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16)
}

fn run_shape(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    shape: Shape,
    target: &str,
) -> Result<ShapeEvidence, String> {
    let activation_elements = shape.k;
    let weight_elements = shape
        .k
        .checked_mul(shape.n)
        .ok_or_else(|| "weight element count overflowed usize".to_owned())?;
    let activation_bytes = activation_elements
        .checked_mul(2)
        .ok_or_else(|| "activation byte count overflowed usize".to_owned())?;
    let weight_bytes = weight_elements
        .checked_mul(2)
        .ok_or_else(|| "weight byte count overflowed usize".to_owned())?;
    let output_bytes = shape
        .n
        .checked_mul(2)
        .ok_or_else(|| "output byte count overflowed usize".to_owned())?;

    let activation_buffer = session
        .allocate(activation_bytes as u64)
        .map_err(|error| format!("activation allocation failed: {error}"))?;
    let weight_buffer = session
        .allocate(weight_bytes as u64)
        .map_err(|error| format!("weight allocation failed: {error}"))?;
    let output_buffer = session
        .allocate(output_bytes as u64)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    upload_repeated_bf16(
        session,
        queue,
        &activation_buffer,
        activation_elements,
        BF16_ONE,
        "activation H2D",
    )?;
    upload_repeated_bf16(
        session,
        queue,
        &weight_buffer,
        weight_elements,
        BF16_ONE,
        "weight H2D",
    )?;

    let activation_view = TensorView::contiguous(DType::Bf16, &[1, shape.k])
        .map_err(|error| format!("activation view failed: {error}"))?;
    let weight_view = TensorView::contiguous(DType::Bf16, &[shape.n, shape.k])
        .map_err(|error| format!("weight view failed: {error}"))?;
    let output_view = TensorView::contiguous(DType::Bf16, &[1, shape.n])
        .map_err(|error| format!("output view failed: {error}"))?;
    let operation = Arc::new(
        BoundSemanticOp::new(
            Arc::new(
                SemanticOpDescriptor::new(
                    SemanticOpKind::Matmul,
                    vec![activation_view.clone(), weight_view.clone()],
                    vec![output_view.clone()],
                )
                .map_err(|error| format!("descriptor failed: {error}"))?,
            ),
            vec![
                session
                    .bind(&activation_buffer, activation_view, AccessMode::Read)
                    .map_err(|error| format!("activation bind failed: {error}"))?,
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .map_err(|error| format!("weight bind failed: {error}"))?,
            ],
            vec![
                session
                    .bind(&output_buffer, output_view, AccessMode::Write)
                    .map_err(|error| format!("output bind failed: {error}"))?,
            ],
        )
        .map_err(|error| format!("bound operation failed: {error}"))?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| format!("prepare failed: {error}"))?;

    let mut samples = Vec::with_capacity(MEASURED);
    let mut last_submission = None;
    let mut identity = None;
    for iteration in 0..WARMUPS + MEASURED {
        let mut submission = session
            .submit(&prepared, queue)
            .map_err(|error| format!("submit failed: {error}"))?;
        let dispatch = submission.dispatch();
        if dispatch.backend != 1
            || dispatch.target != target
            || dispatch.fallback_allowed
            || dispatch.fallback_used
            || dispatch.dispatch_count != 1
            || dispatch.kernel_symbol.is_empty()
            || dispatch.device_symbol.is_empty()
        {
            return Err(format!(
                "{} dispatch violated HIP/no-fallback identity",
                shape.role
            ));
        }
        let current_identity = (
            dispatch.kernel_id,
            dispatch.kernel_symbol.clone(),
            dispatch.device_symbol.clone(),
            dispatch.workgroup_size_x,
            dispatch.grid_size_x,
        );
        if let Some(expected) = &identity {
            if expected != &current_identity {
                return Err(format!(
                    "{} dispatch identity changed between samples",
                    shape.role
                ));
            }
        } else {
            identity = Some(current_identity);
        }
        wait_success(submission.wait(WAIT_TIMEOUT), "matvec completion")?;
        let elapsed = submission
            .kernel_elapsed_ns()
            .map_err(|error| format!("kernel timing failed: {error}"))?
            .ok_or_else(|| "HIP matvec did not publish event timing".to_owned())?;
        if elapsed == 0 {
            return Err("HIP matvec reported zero kernel time".to_owned());
        }
        if iteration >= WARMUPS {
            samples.push(elapsed);
        }
        last_submission = Some(submission);
    }

    let mut submission = last_submission.ok_or_else(|| "no matvec submission ran".to_owned())?;
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| format!("output readback start failed: {error}"))?;
    wait_success(readback.wait(WAIT_TIMEOUT), "output D2H")?;
    let mut actual = vec![0_u8; output_bytes];
    let written = readback
        .read_into(&mut actual)
        .map_err(|error| format!("output read failed: {error}"))?;
    if written != output_bytes as u64 {
        return Err("output readback length mismatch".to_owned());
    }
    let expected_word = expected_sum_word(shape.k)?.to_le_bytes();
    let output_exact = actual
        .chunks_exact(2)
        .all(|word| word == expected_word.as_slice());
    if !output_exact {
        return Err(format!("{} all-one numerical oracle mismatch", shape.role));
    }

    let (kernel_id, kernel_symbol, device_symbol, workgroup_size_x, grid_size_x) =
        identity.ok_or_else(|| "missing dispatch identity".to_owned())?;
    Ok(ShapeEvidence {
        role: shape.role,
        m: 1,
        k: shape.k,
        n: shape.n,
        calls_per_token: shape.calls_per_token,
        effective_bytes_per_call: (activation_bytes + weight_bytes + output_bytes) as u64,
        kernel_id,
        kernel_symbol,
        device_symbol,
        workgroup_size_x,
        grid_size_x,
        warmups: WARMUPS,
        measured: MEASURED,
        kernel_elapsed_ns: samples,
        output_exact,
    })
}

fn run(config: &Config) -> Result<Report, String> {
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(config.device_index, config.target.clone())
                .map_err(|error| format!("invalid session request: {error}"))?,
        )
        .map_err(|error| format!("HIP session open failed: {error}"))?;
    let result = (|| {
        let queue = session
            .create_queue()
            .map_err(|error| format!("queue creation failed: {error}"))?;
        SHAPES
            .iter()
            .copied()
            .map(|shape| run_shape(&session, &queue, shape, &config.target))
            .collect::<Result<Vec<_>, _>>()
    })();
    let before_shutdown = session.allocation_snapshot();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("session shutdown failed: {error}"))?;
    let shapes = result?;
    let after_shutdown = session.allocation_snapshot();
    if after_shutdown.current_bytes() != 0
        || cleanup.retryable_cleanup != 0
        || cleanup.durable_quarantine != 0
    {
        return Err("Phase 22 matvec cleanup did not return to zero".to_owned());
    }
    Ok(Report {
        schema_version: "phase22-matvec-profile-v1",
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        cpu_fallback_used: false,
        fallback_allowed: false,
        fallback_used: false,
        production_completion_mode: "profiled",
        warmups: WARMUPS,
        measured: MEASURED,
        shapes,
        allocation_current_bytes: after_shutdown.current_bytes(),
        allocation_peak_bytes: before_shutdown.high_water_bytes(),
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
                eprintln!("Phase 22 report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("Phase 22 matvec evidence failed: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_is_complete_and_bounded() {
        assert_eq!(SHAPES.len(), 8);
        assert_eq!(
            SHAPES
                .iter()
                .map(|shape| shape.calls_per_token)
                .sum::<u32>(),
            249
        );
        assert!(
            SHAPES
                .iter()
                .any(|shape| shape.k == 2_560 && shape.n == 9_216)
        );
        assert!(
            SHAPES
                .iter()
                .any(|shape| shape.k == 9_216 && shape.n == 2_560)
        );
        assert!(SHAPES.iter().any(|shape| shape.n == 248_320));
    }

    #[test]
    fn all_one_oracle_shapes_are_exact_bf16_integers() {
        for shape in SHAPES {
            let word = expected_sum_word(shape.k).unwrap();
            let reconstructed = f32::from_bits(u32::from(word) << 16);
            assert_eq!(reconstructed, shape.k as f32);
        }
    }
}
