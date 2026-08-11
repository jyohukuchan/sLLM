//! Focused C3c semantic G1 evidence for the distinct sigmoid attention output gate.

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

const CASE_M: [usize; 6] = [1, 3, 17, 255, 256, 257];
const QUERY_HEADS: usize = 16;
const HEAD_DIM: usize = 256;
const O_PROJ_INPUT_WIDTH: usize = QUERY_HEADS * HEAD_DIM;
const DISTINCT_INDEX: usize = 8;
const INTERMEDIATE_BOUNDARY_INDEX: usize = 12;
const SIGMOID_BOUNDARY_GATE: u16 = 0xc100;
const SIGMOID_BOUNDARY_VALUE: u16 = 0xc0fe;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
}

#[derive(Serialize)]
struct ContractEvidence {
    semantic_op: &'static str,
    forbidden_semantic_reuse: &'static str,
    formula: &'static str,
    input_dtype: &'static str,
    operation_dtype: &'static str,
    output_dtype: &'static str,
    output_rounding: &'static str,
    shape: &'static str,
    gqa_query_heads: usize,
    head_dim: usize,
    o_proj_handoff: &'static str,
    broadcasting: bool,
    strides: bool,
    aliasing: bool,
    cpu_fallback: bool,
}

#[derive(Serialize)]
struct CaseEvidence {
    m: usize,
    shape: [usize; 3],
    element_count: usize,
    kernel_id: u32,
    kernel_symbol: String,
    device_symbol: String,
    grid_size_x: u32,
    finite_and_infinite_bit_match: bool,
    nan_classification_match: bool,
    signed_zero_match: bool,
    distinct_from_silu_mul: bool,
    intermediate_bf16_boundary_distinct: bool,
    rounded_down_values: usize,
    rounded_up_values: usize,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    contract: ContractEvidence,
    fallback_allowed: bool,
    fallback_used: bool,
    cpu_fallback_used: bool,
    operations: usize,
    kernel_dispatches: u32,
    cases: Vec<CaseEvidence>,
    cleanup_retryable: usize,
    cleanup_durable: usize,
    cleanup_terminal_zero: bool,
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

fn sigmoid_mul_bf16_intermediate(gate_bits: u16, value_bits: u16) -> u16 {
    let gate = bf16_to_float(gate_bits);
    let sigmoid_bf16 = float_to_bf16_rne(1.0_f32 / (1.0_f32 + (-gate).exp()));
    float_to_bf16_rne(bf16_to_float(sigmoid_bf16) * bf16_to_float(value_bits))
}

fn sigmoid_mul_f32_fused(gate_bits: u16, value_bits: u16) -> u16 {
    let gate = bf16_to_float(gate_bits);
    let sigmoid = 1.0_f32 / (1.0_f32 + (-gate).exp());
    float_to_bf16_rne(sigmoid * bf16_to_float(value_bits))
}

fn silu_mul_bf16_intermediate(gate_bits: u16, value_bits: u16) -> u16 {
    let gate = bf16_to_float(gate_bits);
    let silu_bf16 = float_to_bf16_rne(gate / (1.0_f32 + (-gate).exp()));
    float_to_bf16_rne(bf16_to_float(silu_bf16) * bf16_to_float(value_bits))
}

fn is_bf16_nan(value: u16) -> bool {
    value & 0x7f80 == 0x7f80 && value & 0x007f != 0
}

fn words_to_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u16>, String> {
    if bytes.len() % 2 != 0 {
        return Err("BF16 byte payload has odd length".to_owned());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn make_inputs(element_count: usize) -> (Vec<u16>, Vec<u16>) {
    let mut gate = Vec::with_capacity(element_count);
    let mut value = Vec::with_capacity(element_count);
    for index in 0..element_count {
        let gate_value = ((index * 37 + 11) % 503) as f32 / 31.0 - 8.0;
        let attention_value = ((index * 19 + 7) % 257) as f32 / 47.0 - 2.0;
        gate.push(float_to_bf16_rne(gate_value));
        value.push(float_to_bf16_rne(attention_value));
    }
    let special = [
        (0x0000, 0x0000),
        (0x8000, 0x8000),
        (0x7f80, 0x3f80),
        (0xff80, 0xbf80),
        (0x7fc1, 0x3f80),
        (0x3f80, 0x7f80),
        (0xbf80, 0xff80),
        (0x0001, 0x0080),
        (0x4000, 0x3f80),
        (0xc000, 0x7f7f),
        (0x7f7f, 0x3f81),
        (0x0080, 0x8080),
        (SIGMOID_BOUNDARY_GATE, SIGMOID_BOUNDARY_VALUE),
    ];
    for (index, (gate_bits, value_bits)) in special.into_iter().enumerate() {
        gate[index] = gate_bits;
        value[index] = value_bits;
    }
    (gate, value)
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
    element_count: usize,
    target: &str,
) -> Result<(), String> {
    let expected_grid = dispatch_grid_size(element_count)?;
    if dispatch.abi_version != 1
        || dispatch.info_version != 1
        || dispatch.dispatch_id == 0
        || dispatch.dispatch_count != 1
        || dispatch.kernel_id != 4
        || dispatch.workgroup_size_x != 256
        || dispatch.grid_size_x != expected_grid
        || dispatch.row_count != 1
        || dispatch.normalized_size != element_count as u64
        || dispatch.backend != 1
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != "elementwise.sigmoid_mul.bf16_fp32.v1"
        || dispatch.device_symbol != "sllm_elementwise_sigmoid_mul_bf16_fp32_v1"
        || dispatch.target != target
    {
        return Err("sigmoid_mul dispatch metadata violated the exact contract".to_owned());
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
    m: usize,
    target: &str,
) -> Result<CaseEvidence, String> {
    let element_count = m
        .checked_mul(O_PROJ_INPUT_WIDTH)
        .ok_or_else(|| "output-gate element count overflow".to_owned())?;
    let (gate_words, value_words) = make_inputs(element_count);
    let gate_bytes = words_to_bytes(&gate_words);
    let value_bytes = words_to_bytes(&value_words);
    let buffer_bytes = gate_bytes.len() as u64;
    let gate_buffer = session
        .allocate(buffer_bytes)
        .map_err(|error| format!("gate allocation failed: {error}"))?;
    let value_buffer = session
        .allocate(buffer_bytes)
        .map_err(|error| format!("attention-value allocation failed: {error}"))?;
    let output_buffer = session
        .allocate(buffer_bytes)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    for (label, buffer, bytes) in [
        ("gate", &gate_buffer, gate_bytes),
        ("attention value", &value_buffer, value_bytes),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, buffer_bytes)
                    .map_err(|error| error.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|error| format!("{label} H2D failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), &format!("{label} H2D"))?;
    }

    let view = TensorView::contiguous(DType::Bf16, &[m, QUERY_HEADS, HEAD_DIM])
        .map_err(|error| format!("tensor view failed: {error}"))?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::SigmoidMul,
            vec![view.clone(), view.clone()],
            vec![view.clone()],
        )
        .map_err(|error| format!("semantic descriptor failed: {error}"))?,
    );
    let handoff = descriptor
        .sigmoid_mul_o_proj_input_view()
        .ok_or_else(|| "core omitted the sigmoid_mul o_proj handoff".to_owned())?;
    if handoff.shape() != [m, O_PROJ_INPUT_WIDTH]
        || !handoff.is_contiguous()
        || handoff.payload_bytes() != view.payload_bytes()
    {
        return Err("core o_proj handoff is not contiguous [M,4096] storage".to_owned());
    }
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&gate_buffer, view.clone(), AccessMode::Read)
                    .map_err(|error| format!("gate binding failed: {error}"))?,
                session
                    .bind(&value_buffer, view.clone(), AccessMode::Read)
                    .map_err(|error| format!("attention-value binding failed: {error}"))?,
            ],
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
        .map_err(|error| format!("sigmoid_mul prepare failed: {error}"))?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| format!("sigmoid_mul submit failed: {error}"))?;
    validate_dispatch(submission.dispatch(), element_count, target)?;
    wait_success(submission.wait(WAIT_TIMEOUT), "sigmoid_mul completion")?;
    let dispatch = submission.dispatch().clone();
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| format!("output D2H failed: {error}"))?;
    wait_success(readback.wait(WAIT_TIMEOUT), "output D2H")?;
    let mut actual_bytes = vec![0_u8; buffer_bytes as usize];
    let written = readback
        .read_into(&mut actual_bytes)
        .map_err(|error| format!("output read failed: {error}"))?;
    if written != buffer_bytes {
        return Err("output byte count mismatch".to_owned());
    }
    let actual_words = bytes_to_words(&actual_bytes)?;

    let mut nan_classification_match = true;
    let mut finite_and_infinite_bit_match = true;
    let mut signed_zero_match = true;
    let mut rounded_down_values = 0_usize;
    let mut rounded_up_values = 0_usize;
    let mut distinct_from_silu_mul = false;
    let mut intermediate_bf16_boundary_distinct = false;
    for (index, ((&gate_bits, &value_bits), &actual)) in gate_words
        .iter()
        .zip(&value_words)
        .zip(&actual_words)
        .enumerate()
    {
        let gate = bf16_to_float(gate_bits);
        let value = bf16_to_float(value_bits);
        let sigmoid = 1.0_f32 / (1.0_f32 + (-gate).exp());
        let sigmoid_bf16 = float_to_bf16_rne(sigmoid);
        let fp32_product = bf16_to_float(sigmoid_bf16) * value;
        let expected = sigmoid_mul_bf16_intermediate(gate_bits, value_bits);
        if is_bf16_nan(expected) {
            nan_classification_match &= is_bf16_nan(actual);
        } else {
            finite_and_infinite_bit_match &= actual == expected;
            if expected & 0x7fff == 0 {
                signed_zero_match &= actual == expected;
            }
        }
        if fp32_product.is_finite() {
            let lower = fp32_product.to_bits() & 0xffff;
            if lower != 0 {
                if u32::from(expected) == fp32_product.to_bits() >> 16 {
                    rounded_down_values += 1;
                } else {
                    rounded_up_values += 1;
                }
            }
        }
        if index == DISTINCT_INDEX {
            let forbidden_silu = silu_mul_bf16_intermediate(gate_bits, value_bits);
            distinct_from_silu_mul = expected != forbidden_silu && actual == expected;
        }
        if index == INTERMEDIATE_BOUNDARY_INDEX {
            intermediate_bf16_boundary_distinct =
                expected != sigmoid_mul_f32_fused(gate_bits, value_bits) && actual == expected;
        }
    }
    if !finite_and_infinite_bit_match
        || !nan_classification_match
        || !signed_zero_match
        || !distinct_from_silu_mul
        || !intermediate_bf16_boundary_distinct
        || rounded_down_values == 0
        || rounded_up_values == 0
    {
        return Err(format!("sigmoid_mul numerical contract mismatch for M={m}"));
    }

    Ok(CaseEvidence {
        m,
        shape: [m, QUERY_HEADS, HEAD_DIM],
        element_count,
        kernel_id: dispatch.kernel_id,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        grid_size_x: dispatch.grid_size_x,
        finite_and_infinite_bit_match,
        nan_classification_match,
        signed_zero_match,
        distinct_from_silu_mul,
        intermediate_bf16_boundary_distinct,
        rounded_down_values,
        rounded_up_values,
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
        CASE_M
            .into_iter()
            .map(|m| run_case(&session, &queue, m, &config.target))
            .collect()
    })();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("execution-session shutdown failed: {error}"))?;
    let cases = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("execution cleanup did not return to zero owned work".to_owned());
    }
    Ok(Report {
        schema_version: "output-gate-g1-report-v1",
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        contract: ContractEvidence {
            semantic_op: "sigmoid_mul",
            forbidden_semantic_reuse: "silu_mul",
            formula: "bf16_rne(sigmoid(f32(bf16_gate))) -> f32 * f32(bf16_attention_value) -> bf16_rne",
            input_dtype: "BF16",
            operation_dtype: "FP32",
            output_dtype: "BF16",
            output_rounding: "round-to-nearest-even after sigmoid and once after FP32 multiply",
            shape: "[M,16,256] row-major contiguous",
            gqa_query_heads: QUERY_HEADS,
            head_dim: HEAD_DIM,
            o_proj_handoff: "zero-copy contiguous [M,4096] activation",
            broadcasting: false,
            strides: false,
            aliasing: false,
            cpu_fallback: false,
        },
        fallback_allowed: false,
        fallback_used: false,
        cpu_fallback_used: false,
        operations: cases.len(),
        kernel_dispatches: cases.len() as u32,
        cases,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
        cleanup_terminal_zero: true,
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
                eprintln!("output-gate-g1 report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("output-gate-g1: {error}");
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
    fn oracle_covers_required_m_special_values_and_bf16_rne_directions() {
        assert_eq!(
            float_to_bf16_rne(f32::from_bits(0x3f80_8000)),
            0x3f80,
            "halfway value with an even retained bit must round down"
        );
        assert_eq!(
            float_to_bf16_rne(f32::from_bits(0x3f81_8000)),
            0x3f82,
            "halfway value with an odd retained bit must round up"
        );

        for m in CASE_M {
            let element_count = m * O_PROJ_INPUT_WIDTH;
            let (gate, value) = make_inputs(element_count);
            assert_eq!(gate.len(), element_count);
            assert_eq!(value.len(), element_count);

            let mut rounded_down = 0_usize;
            let mut rounded_up = 0_usize;
            for (&gate_bits, &value_bits) in gate.iter().zip(&value) {
                let gate_value = bf16_to_float(gate_bits);
                let sigmoid_bf16 = float_to_bf16_rne(1.0_f32 / (1.0_f32 + (-gate_value).exp()));
                let product = bf16_to_float(sigmoid_bf16) * bf16_to_float(value_bits);
                if product.is_finite() && product.to_bits() & 0xffff != 0 {
                    if u32::from(float_to_bf16_rne(product)) == product.to_bits() >> 16 {
                        rounded_down += 1;
                    } else {
                        rounded_up += 1;
                    }
                }
            }
            assert!(rounded_down > 0, "M={m} must exercise RNE rounding down");
            assert!(rounded_up > 0, "M={m} must exercise RNE rounding up");

            assert_eq!(gate[0], 0x0000);
            assert_eq!(gate[1], 0x8000);
            assert!(is_bf16_nan(gate[4]));
            assert_eq!(gate[7], 0x0001);
            assert_eq!(gate[DISTINCT_INDEX], 0x4000);
            assert_ne!(
                sigmoid_mul_bf16_intermediate(gate[DISTINCT_INDEX], value[DISTINCT_INDEX]),
                silu_mul_bf16_intermediate(gate[DISTINCT_INDEX], value[DISTINCT_INDEX]),
                "sigmoid_mul oracle must remain distinct from silu_mul"
            );
            assert_eq!(gate[INTERMEDIATE_BOUNDARY_INDEX], SIGMOID_BOUNDARY_GATE);
            assert_eq!(value[INTERMEDIATE_BOUNDARY_INDEX], SIGMOID_BOUNDARY_VALUE);
            assert_ne!(
                sigmoid_mul_f32_fused(SIGMOID_BOUNDARY_GATE, SIGMOID_BOUNDARY_VALUE),
                sigmoid_mul_bf16_intermediate(SIGMOID_BOUNDARY_GATE, SIGMOID_BOUNDARY_VALUE),
                "sigmoid_mul oracle must require the BF16 sigmoid boundary"
            );
        }
    }
}
