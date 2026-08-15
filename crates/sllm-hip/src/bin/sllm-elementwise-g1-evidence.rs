//! Focused semantic G1 evidence for BF16 copy, residual add, and SiLU multiply.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, DispatchEvidence, ExecutionSessionRequest,
    ExecutionState, SemanticOpDescriptor, SemanticOpKind, TensorView,
};
use sllm_hip::HipBackend;

const CASE_SIZES: [usize; 7] = [1, 3, 17, 255, 256, 257, 2560];
const GEMMA_CASE_SIZES: [usize; 10] = [1, 3, 17, 255, 256, 257, 3839, 3840, 3841, 262_144];
const SILU_BOUNDARY_INDEX: usize = 3;
const SILU_BOUNDARY_GATE: u16 = 0xc100;
const SILU_BOUNDARY_UP: u16 = 0xc0fe;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
    gemma_ops: bool,
}

#[derive(Serialize)]
struct CaseEvidence {
    operation: &'static str,
    element_count: usize,
    kernel_id: u32,
    kernel_symbol: String,
    device_symbol: String,
    grid_size_x: u32,
    exact_match: bool,
    intermediate_bf16_boundary_distinct: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    fallback_allowed: bool,
    fallback_used: bool,
    cpu_fallback_used: bool,
    operations: usize,
    kernel_dispatches: u32,
    cases: Vec<CaseEvidence>,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

fn parse_config() -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut gemma_ops = false;
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
                    return Err("--target must be gfx1030, gfx1201, or gfx942".to_owned());
                }
                target = Some(value);
            }
            "--gemma-ops" => {
                if gemma_ops {
                    return Err("duplicate --gemma-ops".to_owned());
                }
                gemma_ops = true;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
        gemma_ops,
    })
}

fn float_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        if bits & 0x007f_ffff != 0 {
            return ((bits >> 16) as u16 & 0x803f) | 0x7fc0;
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

fn silu_mul_bf16_intermediate(gate_bits: u16, up_bits: u16) -> u16 {
    let gate = bf16_to_float(gate_bits);
    let silu_bf16 = float_to_bf16_rne(gate / (1.0 + (-gate).exp()));
    float_to_bf16_rne(bf16_to_float(silu_bf16) * bf16_to_float(up_bits))
}

fn silu_mul_f32_fused(gate_bits: u16, up_bits: u16) -> u16 {
    let gate = bf16_to_float(gate_bits);
    let silu = gate / (1.0 + (-gate).exp());
    float_to_bf16_rne(silu * bf16_to_float(up_bits))
}

fn scalar_mul(input_bits: u16, scalar_bits: u16) -> u16 {
    float_to_bf16_rne(bf16_to_float(input_bits) * bf16_to_float(scalar_bits))
}

fn gelu_tanh_mul_bf16_intermediate(gate_bits: u16, up_bits: u16) -> u16 {
    let gate = bf16_to_float(gate_bits);
    let inner = 0.797_884_6_f32 * (gate + 0.044_715_f32 * gate * gate * gate);
    let gelu_bf16 = float_to_bf16_rne(0.5_f32 * gate * (1.0_f32 + inner.tanh()));
    float_to_bf16_rne(bf16_to_float(gelu_bf16) * bf16_to_float(up_bits))
}

fn tanh_softcap(input_bits: u16, cap_bits: u16) -> u16 {
    let cap = bf16_to_float(cap_bits);
    float_to_bf16_rne((bf16_to_float(input_bits) / cap).tanh() * cap)
}

fn words_to_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn make_inputs(element_count: usize) -> (Vec<u16>, Vec<u16>) {
    let mut input0 = Vec::with_capacity(element_count);
    let mut input1 = Vec::with_capacity(element_count);
    for index in 0..element_count {
        let left = ((index * 37 + 11) % 503) as f32 / 31.0 - 8.0;
        let right = ((index * 19 + 7) % 257) as f32 / 47.0 - 2.0;
        input0.push(float_to_bf16_rne(left));
        input1.push(float_to_bf16_rne(right));
    }
    if element_count >= 3 {
        input0[0] = 0x7f80;
        input1[0] = 0xff80;
        input0[1] = 0x7fc1;
        input1[1] = float_to_bf16_rne(1.0);
        input0[2] = 0x8000;
        input1[2] = 0x0000;
    }
    (input0, input1)
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

fn validate_dispatch(
    dispatch: &DispatchEvidence,
    kind: SemanticOpKind,
    element_count: usize,
    target: &str,
) -> Result<(), String> {
    let (kernel_id, kernel_symbol, device_symbol) = match kind {
        SemanticOpKind::Copy => (
            1,
            "elementwise.copy.bf16.v1",
            "sllm_elementwise_copy_bf16_v1",
        ),
        SemanticOpKind::Add => (
            2,
            "elementwise.add.bf16_fp32.v1",
            "sllm_elementwise_add_bf16_fp32_v1",
        ),
        SemanticOpKind::SiluMul => (
            3,
            "elementwise.silu_mul.bf16_fp32.v1",
            "sllm_elementwise_silu_mul_bf16_fp32_v1",
        ),
        SemanticOpKind::ScalarMul => (
            5,
            "elementwise.scalar_mul.bf16_fp32.v1",
            "sllm_elementwise_scalar_mul_bf16_fp32_v1",
        ),
        SemanticOpKind::GeluTanhMul => (
            6,
            "elementwise.gelu_tanh_mul.bf16_fp32.v1",
            "sllm_elementwise_gelu_tanh_mul_bf16_fp32_v1",
        ),
        SemanticOpKind::TanhSoftcap => (
            7,
            "elementwise.tanh_softcap.bf16_fp32.v1",
            "sllm_elementwise_tanh_softcap_bf16_fp32_v1",
        ),
        _ => return Err("evidence received a non-elementwise operation".to_owned()),
    };
    let expected_grid = dispatch_grid_size(element_count)?;
    if dispatch.abi_version != 1
        || dispatch.info_version != 1
        || dispatch.dispatch_id == 0
        || dispatch.dispatch_count != 1
        || dispatch.kernel_id != kernel_id
        || dispatch.workgroup_size_x != 256
        || dispatch.grid_size_x != expected_grid
        || dispatch.row_count != 1
        || dispatch.normalized_size != element_count as u64
        || dispatch.backend != 1
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != kernel_symbol
        || dispatch.device_symbol != device_symbol
        || dispatch.target != target
    {
        return Err(format!(
            "{} dispatch metadata violated the exact contract",
            kind.name()
        ));
    }
    Ok(())
}

fn dispatch_grid_size(element_count: usize) -> Result<u32, String> {
    let workgroup = 256_usize;
    let blocks = element_count / workgroup + usize::from(element_count % workgroup != 0);
    u32::try_from(blocks).map_err(|_| "grid size does not fit u32".to_owned())
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    kind: SemanticOpKind,
    element_count: usize,
    target: &str,
) -> Result<CaseEvidence, String> {
    let (mut input0_words, mut input1_words) = make_inputs(element_count);
    let scalar_input = matches!(
        kind,
        SemanticOpKind::ScalarMul | SemanticOpKind::TanhSoftcap
    );
    if kind == SemanticOpKind::ScalarMul {
        input1_words = vec![float_to_bf16_rne((3840.0_f32).sqrt())];
    } else if kind == SemanticOpKind::TanhSoftcap {
        input1_words = vec![float_to_bf16_rne(30.0)];
    }
    if kind == SemanticOpKind::SiluMul && element_count > SILU_BOUNDARY_INDEX {
        input0_words[SILU_BOUNDARY_INDEX] = SILU_BOUNDARY_GATE;
        input1_words[SILU_BOUNDARY_INDEX] = SILU_BOUNDARY_UP;
    }
    if kind == SemanticOpKind::Add && element_count >= 3 {
        input0_words[0] = 0x7f80;
        input1_words[0] = float_to_bf16_rne(1.0);
        input0_words[1] = 0xff80;
        input1_words[1] = float_to_bf16_rne(-1.0);
        input0_words[2] = 0x8000;
        input1_words[2] = 0x0000;
    }
    let input0_bytes = words_to_bytes(&input0_words);
    let input1_bytes = words_to_bytes(&input1_words);
    let output_bytes = input0_bytes.len() as u64;
    let input0_buffer = session
        .allocate(output_bytes)
        .map_err(|error| format!("input0 allocation failed: {error}"))?;
    let input1_buffer = if kind != SemanticOpKind::Copy {
        Some(
            session
                .allocate(input1_bytes.len() as u64)
                .map_err(|error| format!("input1 allocation failed: {error}"))?,
        )
    } else {
        None
    };
    let output_buffer = session
        .allocate(output_bytes)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    let mut upload0 = session
        .upload(
            queue,
            input0_buffer
                .range(0, output_bytes)
                .map_err(|error| error.to_string())?,
            Arc::<[u8]>::from(input0_bytes.clone()),
        )
        .map_err(|error| format!("input0 H2D failed: {error}"))?;
    wait_success(upload0.wait(WAIT_TIMEOUT), "input0 H2D")?;
    if let Some(buffer) = &input1_buffer {
        let mut upload1 = session
            .upload(
                queue,
                buffer
                    .range(0, input1_bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                Arc::<[u8]>::from(input1_bytes.clone()),
            )
            .map_err(|error| format!("input1 H2D failed: {error}"))?;
        wait_success(upload1.wait(WAIT_TIMEOUT), "input1 H2D")?;
    }

    let view = TensorView::contiguous(DType::Bf16, &[element_count])
        .map_err(|error| format!("tensor view failed: {error}"))?;
    let mut input_bindings = vec![
        session
            .bind(&input0_buffer, view.clone(), AccessMode::Read)
            .map_err(|error| format!("input0 binding failed: {error}"))?,
    ];
    if let Some(buffer) = &input1_buffer {
        let input1_view = if scalar_input {
            TensorView::contiguous(DType::Bf16, &[1])
                .map_err(|error| format!("scalar tensor view failed: {error}"))?
        } else {
            view.clone()
        };
        input_bindings.push(
            session
                .bind(buffer, input1_view, AccessMode::Read)
                .map_err(|error| format!("input1 binding failed: {error}"))?,
        );
    }
    let descriptor_inputs = input_bindings
        .iter()
        .map(|binding| binding.view().clone())
        .collect();
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(kind, descriptor_inputs, vec![view.clone()])
            .map_err(|error| format!("semantic descriptor failed: {error}"))?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            input_bindings,
            vec![
                session
                    .bind(&output_buffer, view, AccessMode::Write)
                    .map_err(|error| format!("output binding failed: {error}"))?,
            ],
        )
        .map_err(|error| format!("owned operation binding failed: {error}"))?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| format!("elementwise prepare failed: {error}"))?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| format!("elementwise submit failed: {error}"))?;
    validate_dispatch(submission.dispatch(), kind, element_count, target)?;
    wait_success(submission.wait(WAIT_TIMEOUT), "elementwise completion")?;
    let dispatch = submission.dispatch().clone();
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| format!("output D2H failed: {error}"))?;
    wait_success(readback.wait(WAIT_TIMEOUT), "output D2H")?;
    let mut actual = vec![0_u8; input0_bytes.len()];
    let written = readback
        .read_into(&mut actual)
        .map_err(|error| format!("output read failed: {error}"))?;
    if written != actual.len() as u64 {
        return Err("output byte count mismatch".to_owned());
    }
    let expected_words = match kind {
        SemanticOpKind::Copy => input0_words.clone(),
        SemanticOpKind::Add => input0_words
            .iter()
            .zip(&input1_words)
            .map(|(&left, &right)| float_to_bf16_rne(bf16_to_float(left) + bf16_to_float(right)))
            .collect::<Vec<_>>(),
        SemanticOpKind::SiluMul => input0_words
            .iter()
            .zip(&input1_words)
            .map(|(&gate, &up)| silu_mul_bf16_intermediate(gate, up))
            .collect::<Vec<_>>(),
        SemanticOpKind::ScalarMul => input0_words
            .iter()
            .map(|&input| scalar_mul(input, input1_words[0]))
            .collect::<Vec<_>>(),
        SemanticOpKind::GeluTanhMul => input0_words
            .iter()
            .zip(&input1_words)
            .map(|(&gate, &up)| gelu_tanh_mul_bf16_intermediate(gate, up))
            .collect::<Vec<_>>(),
        SemanticOpKind::TanhSoftcap => input0_words
            .iter()
            .map(|&input| tanh_softcap(input, input1_words[0]))
            .collect::<Vec<_>>(),
        _ => unreachable!(),
    };
    let intermediate_bf16_boundary_distinct =
        kind == SemanticOpKind::SiluMul && element_count > SILU_BOUNDARY_INDEX;
    if intermediate_bf16_boundary_distinct {
        let fused_words = input0_words
            .iter()
            .zip(&input1_words)
            .map(|(&gate, &up)| silu_mul_f32_fused(gate, up))
            .collect::<Vec<_>>();
        if expected_words == fused_words {
            return Err(format!(
                "silu_mul oracle did not exercise the BF16 intermediate boundary at {element_count} elements"
            ));
        }
    }
    let expected = words_to_bytes(&expected_words);
    if actual != expected {
        return Err(format!(
            "{} numerical oracle mismatch at {element_count} elements",
            kind.name()
        ));
    }
    Ok(CaseEvidence {
        operation: kind.name(),
        element_count,
        kernel_id: dispatch.kernel_id,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        grid_size_x: dispatch.grid_size_x,
        exact_match: true,
        intermediate_bf16_boundary_distinct,
    })
}

fn run(config: &Config) -> Result<Report, String> {
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("invalid execution-session request: {error}"))?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| format!("execution-session open failed: {error}"))?;
    let result: Result<Vec<CaseEvidence>, String> = (|| {
        let queue = session
            .create_queue()
            .map_err(|error| format!("queue creation failed: {error}"))?;
        let (sizes, kinds): (&[usize], &[SemanticOpKind]) = if config.gemma_ops {
            (
                &GEMMA_CASE_SIZES,
                &[
                    SemanticOpKind::ScalarMul,
                    SemanticOpKind::GeluTanhMul,
                    SemanticOpKind::TanhSoftcap,
                ],
            )
        } else {
            (
                &CASE_SIZES,
                &[
                    SemanticOpKind::Copy,
                    SemanticOpKind::Add,
                    SemanticOpKind::SiluMul,
                ],
            )
        };
        let mut cases = Vec::with_capacity(sizes.len() * kinds.len());
        for &element_count in sizes {
            for &kind in kinds {
                cases.push(run_case(
                    &session,
                    &queue,
                    kind,
                    element_count,
                    &config.target,
                )?);
            }
        }
        Ok(cases)
    })();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("execution-session shutdown failed: {error}"))?;
    let cases = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("execution cleanup did not return to zero owned work".to_owned());
    }
    Ok(Report {
        schema_version: if config.gemma_ops {
            "gemma-elementwise-a3-report-v1"
        } else {
            "elementwise-g1-report-v1"
        },
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        fallback_allowed: false,
        fallback_used: false,
        cpu_fallback_used: false,
        operations: cases.len(),
        kernel_dispatches: cases.len() as u32,
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
                eprintln!("elementwise-g1 report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("elementwise-g1: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_grid_is_exact_at_workgroup_boundaries() {
        assert_eq!(dispatch_grid_size(1), Ok(1));
        assert_eq!(dispatch_grid_size(255), Ok(1));
        assert_eq!(dispatch_grid_size(256), Ok(1));
        assert_eq!(dispatch_grid_size(257), Ok(2));
    }

    #[test]
    fn silu_oracle_requires_the_bf16_intermediate_boundary() {
        let fused = silu_mul_f32_fused(SILU_BOUNDARY_GATE, SILU_BOUNDARY_UP);
        let corrected = silu_mul_bf16_intermediate(SILU_BOUNDARY_GATE, SILU_BOUNDARY_UP);

        assert_ne!(fused, corrected);
        assert_eq!(corrected, silu_mul_bf16_intermediate(0xc100, 0xc0fe));
    }
}
