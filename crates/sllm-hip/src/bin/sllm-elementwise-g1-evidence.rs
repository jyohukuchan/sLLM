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
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
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
        _ => return Err("evidence received a non-elementwise operation".to_owned()),
    };
    let expected_grid = u32::try_from(element_count.div_ceil(256))
        .map_err(|_| "grid size does not fit u32".to_owned())?;
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

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    kind: SemanticOpKind,
    element_count: usize,
    target: &str,
) -> Result<CaseEvidence, String> {
    let (mut input0_words, mut input1_words) = make_inputs(element_count);
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
                .allocate(output_bytes)
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
                    .range(0, output_bytes)
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
        input_bindings.push(
            session
                .bind(buffer, view.clone(), AccessMode::Read)
                .map_err(|error| format!("input1 binding failed: {error}"))?,
        );
    }
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            kind,
            vec![view.clone(); input_bindings.len()],
            vec![view.clone()],
        )
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
    let expected = match kind {
        SemanticOpKind::Copy => input0_bytes,
        SemanticOpKind::Add => words_to_bytes(
            &input0_words
                .iter()
                .zip(input1_words)
                .map(|(&left, right)| float_to_bf16_rne(bf16_to_float(left) + bf16_to_float(right)))
                .collect::<Vec<_>>(),
        ),
        SemanticOpKind::SiluMul => words_to_bytes(
            &input0_words
                .iter()
                .zip(input1_words)
                .map(|(&gate, up)| {
                    let gate = bf16_to_float(gate);
                    let silu = gate / (1.0 + (-gate).exp());
                    float_to_bf16_rne(silu * bf16_to_float(up))
                })
                .collect::<Vec<_>>(),
        ),
        _ => unreachable!(),
    };
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
        let mut cases = Vec::with_capacity(CASE_SIZES.len() * 3);
        for element_count in CASE_SIZES {
            cases.push(run_case(
                &session,
                &queue,
                SemanticOpKind::Copy,
                element_count,
                &config.target,
            )?);
            cases.push(run_case(
                &session,
                &queue,
                SemanticOpKind::Add,
                element_count,
                &config.target,
            )?);
            cases.push(run_case(
                &session,
                &queue,
                SemanticOpKind::SiluMul,
                element_count,
                &config.target,
            )?);
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
        schema_version: "elementwise-g1-report-v1",
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
