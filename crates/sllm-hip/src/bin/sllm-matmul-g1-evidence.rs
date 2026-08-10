//! Focused semantic G1 evidence for the public BF16 matmul execution path.
//!
//! The cases are deliberately bounded and use checkpoint-oriented `[N, K]`
//! weights directly.  The only CPU computation here is an independent scalar
//! oracle used after the owned HIP output readback; it is never an execution
//! fallback.

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

const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);
const HIP_BACKEND: u32 = 1;
const MATMUL_KERNEL_ID: u32 = 1;
const WORKGROUP_SIZE: u32 = 256;
const KERNEL_SYMBOL: &str = "matmul.bf16_fp32.v1";
const DEVICE_SYMBOL: &str = "sllm_matmul_bf16_fp32_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CaseShape {
    m: usize,
    k: usize,
    n: usize,
}

// Keep the boundary coverage broad without taking the Cartesian product of
// all M/K/N values.  K=5 is retained as a small non-boundary reduction case.
const CASES: [CaseShape; 13] = [
    CaseShape { m: 1, k: 1, n: 1 },
    CaseShape { m: 1, k: 3, n: 17 },
    CaseShape { m: 3, k: 17, n: 3 },
    CaseShape { m: 17, k: 3, n: 1 },
    CaseShape { m: 1, k: 255, n: 3 },
    CaseShape { m: 1, k: 256, n: 3 },
    CaseShape { m: 1, k: 257, n: 3 },
    CaseShape { m: 3, k: 5, n: 255 },
    CaseShape { m: 3, k: 5, n: 256 },
    CaseShape { m: 3, k: 5, n: 257 },
    CaseShape { m: 255, k: 3, n: 1 },
    CaseShape { m: 256, k: 3, n: 1 },
    CaseShape { m: 257, k: 3, n: 1 },
];

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
}

#[derive(Serialize)]
struct CaseEvidence {
    m: usize,
    k: usize,
    n: usize,
    output_elements: usize,
    row_count: u64,
    normalized_size: u64,
    dispatch_id: u64,
    dispatch_count: u32,
    kernel_id: u32,
    workgroup_size_x: u32,
    grid_size_x: u32,
    kernel_symbol: String,
    device_symbol: String,
    exact_match: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    cpu_fallback_used: bool,
    fallback: bool,
    fallback_allowed: bool,
    fallback_used: bool,
    operations: usize,
    dispatch_count: usize,
    kernel_dispatches: usize,
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

/// Convert binary32 to BF16 using round-to-nearest-even.  Non-finite values
/// follow the native contract: infinities retain their sign and NaNs become
/// quiet BF16 NaNs retaining the sign and representable high payload bits.
fn float_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        if bits & 0x007f_ffff != 0 {
            let sign = ((bits >> 16) as u16) & 0x8000;
            let payload = ((bits >> 16) as u16) & 0x003f;
            return sign | 0x7fc0 | payload;
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

fn shape_element_count(shape: CaseShape) -> Result<(usize, usize, usize), String> {
    let activation = shape
        .m
        .checked_mul(shape.k)
        .ok_or_else(|| "activation element count overflowed usize".to_owned())?;
    let weight = shape
        .n
        .checked_mul(shape.k)
        .ok_or_else(|| "weight element count overflowed usize".to_owned())?;
    let output = shape
        .m
        .checked_mul(shape.n)
        .ok_or_else(|| "output element count overflowed usize".to_owned())?;
    Ok((activation, weight, output))
}

const ORDINARY_FINITE: [u16; 6] = [0x3f80, 0xbf80, 0x3fc0, 0xc020, 0x4000, 0x7f00];
const SPECIAL_VALUES: [u16; 8] = [
    0x0000, // +0
    0x8000, // -0
    0x0001, // smallest positive BF16 subnormal
    0x7fc1, // positive NaN with payload
    0x7f80, // +Inf
    0xff80, // -Inf
    0x8001, // negative BF16 subnormal
    0x7fc2, // another positive NaN payload
];

fn make_operands(shape: CaseShape, case_index: usize) -> Result<(Vec<u16>, Vec<u16>), String> {
    let (activation_count, weight_count, _) = shape_element_count(shape)?;
    let mut activation = (0..activation_count)
        .map(|index| ORDINARY_FINITE[(index * 17 + case_index) % ORDINARY_FINITE.len()])
        .collect::<Vec<_>>();
    let mut weight = (0..weight_count)
        .map(|index| ORDINARY_FINITE[(index * 29 + case_index + 1) % ORDINARY_FINITE.len()])
        .collect::<Vec<_>>();

    if shape.k == 1 || case_index % 2 == 0 {
        // Keep the weight finite while covering all special activation values,
        // avoiding an indeterminate zero-times-infinity product.
        for row in 0..shape.m {
            activation[row] = SPECIAL_VALUES[(row + case_index) % SPECIAL_VALUES.len()];
        }
        if shape.k == 1 {
            weight.fill(0x3f80);
        } else {
            for column in 0..shape.n {
                weight[column * shape.k] = 0x3f80;
                weight[column * shape.k + 1] = 0x3f80;
            }
        }
    } else {
        // This branch covers special checkpoint weights.  Activations remain
        // finite, so each special result has one deterministic special source.
        for column in 0..shape.n {
            weight[column * shape.k] = 0x3f80;
            weight[column * shape.k + 1] =
                SPECIAL_VALUES[(column + case_index + 2) % SPECIAL_VALUES.len()];
        }
    }
    Ok((activation, weight))
}

/// Independent scalar oracle.  The product is materialized before the add,
/// and reductions always visit k in ascending order.
#[inline(never)]
fn scalar_matmul_oracle(shape: CaseShape, activation: &[u16], weight: &[u16]) -> Vec<u16> {
    let mut output = Vec::with_capacity(shape.m * shape.n);
    for row in 0..shape.m {
        for column in 0..shape.n {
            let mut accumulator = 0.0_f32;
            for reduction in 0..shape.k {
                let product = bf16_to_float(activation[row * shape.k + reduction])
                    * bf16_to_float(weight[column * shape.k + reduction]);
                accumulator += product;
            }
            output.push(float_to_bf16_rne(accumulator));
        }
    }
    output
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
    shape: CaseShape,
    target: &str,
) -> Result<(), String> {
    let output_elements = shape
        .m
        .checked_mul(shape.n)
        .ok_or_else(|| "output element count overflowed usize".to_owned())?;
    let expected_grid = u32::try_from(output_elements.div_ceil(WORKGROUP_SIZE as usize))
        .map_err(|_| "matmul grid size does not fit u32".to_owned())?;
    if dispatch.abi_version != 1
        || dispatch.info_version != 1
        || dispatch.dispatch_id == 0
        || dispatch.dispatch_count != 1
        || dispatch.kernel_id != MATMUL_KERNEL_ID
        || dispatch.workgroup_size_x != WORKGROUP_SIZE
        || dispatch.grid_size_x != expected_grid
        || dispatch.row_count != shape.m as u64
        || dispatch.normalized_size != output_elements as u64
        || dispatch.backend != HIP_BACKEND
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != KERNEL_SYMBOL
        || dispatch.device_symbol != DEVICE_SYMBOL
        || dispatch.target != target
    {
        return Err(format!(
            "matmul dispatch metadata violated the exact contract for M={} K={} N={}",
            shape.m, shape.k, shape.n
        ));
    }
    Ok(())
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    shape: CaseShape,
    case_index: usize,
    target: &str,
) -> Result<CaseEvidence, String> {
    let (activation_words, weight_words) = make_operands(shape, case_index)?;
    let expected_words = scalar_matmul_oracle(shape, &activation_words, &weight_words);
    let activation_bytes = words_to_bytes(&activation_words);
    let weight_bytes = words_to_bytes(&weight_words);
    let output_bytes = words_to_bytes(&expected_words);

    let activation_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|error| format!("activation allocation failed: {error}"))?;
    let weight_buffer = session
        .allocate(weight_bytes.len() as u64)
        .map_err(|error| format!("weight allocation failed: {error}"))?;
    let output_buffer = session
        .allocate(output_bytes.len() as u64)
        .map_err(|error| format!("output allocation failed: {error}"))?;

    for (label, buffer, bytes) in [
        (
            "activation",
            &activation_buffer,
            activation_bytes.as_slice(),
        ),
        ("weight", &weight_buffer, weight_bytes.as_slice()),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|error| error.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|error| format!("{label} H2D failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), &format!("{label} H2D"))?;
    }

    let activation_view = TensorView::contiguous(DType::Bf16, &[shape.m, shape.k])
        .map_err(|error| format!("activation tensor view failed: {error}"))?;
    let weight_view = TensorView::contiguous(DType::Bf16, &[shape.n, shape.k])
        .map_err(|error| format!("weight tensor view failed: {error}"))?;
    let output_view = TensorView::contiguous(DType::Bf16, &[shape.m, shape.n])
        .map_err(|error| format!("output tensor view failed: {error}"))?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation_view.clone(), weight_view.clone()],
            vec![output_view.clone()],
        )
        .map_err(|error| format!("matmul semantic descriptor failed: {error}"))?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&activation_buffer, activation_view, AccessMode::Read)
                    .map_err(|error| format!("activation binding failed: {error}"))?,
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .map_err(|error| format!("weight binding failed: {error}"))?,
            ],
            vec![
                session
                    .bind(&output_buffer, output_view, AccessMode::Write)
                    .map_err(|error| format!("output binding failed: {error}"))?,
            ],
        )
        .map_err(|error| format!("owned matmul binding failed: {error}"))?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| format!("matmul prepare failed: {error}"))?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| format!("matmul submit failed: {error}"))?;
    validate_dispatch(submission.dispatch(), shape, target)?;
    wait_success(submission.wait(WAIT_TIMEOUT), "matmul completion")?;
    let dispatch = submission.dispatch().clone();
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| format!("owned matmul output readback failed: {error}"))?;
    wait_success(readback.wait(WAIT_TIMEOUT), "matmul D2H")?;
    let mut actual = vec![0_u8; output_bytes.len()];
    let written = readback
        .read_into(&mut actual)
        .map_err(|error| format!("matmul output read failed: {error}"))?;
    if written != output_bytes.len() as u64 {
        return Err(format!(
            "matmul output byte count mismatch for M={} K={} N={}",
            shape.m, shape.k, shape.n
        ));
    }
    if actual != output_bytes {
        return Err(format!(
            "matmul BF16 oracle mismatch for M={} K={} N={}",
            shape.m, shape.k, shape.n
        ));
    }

    Ok(CaseEvidence {
        m: shape.m,
        k: shape.k,
        n: shape.n,
        output_elements: shape.m * shape.n,
        row_count: dispatch.row_count,
        normalized_size: dispatch.normalized_size,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        exact_match: true,
    })
}

fn run(config: &Config) -> Result<Report, String> {
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("invalid execution-session request: {error}"))?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| format!("owned HIP execution-session open failed: {error}"))?;
    let result: Result<Vec<CaseEvidence>, String> = (|| {
        let queue = session
            .create_queue()
            .map_err(|error| format!("queue creation failed: {error}"))?;
        let mut cases = Vec::with_capacity(CASES.len());
        for (case_index, shape) in CASES.iter().copied().enumerate() {
            cases.push(run_case(
                &session,
                &queue,
                shape,
                case_index,
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
        return Err("matmul cleanup did not return to zero owned work".to_owned());
    }
    let operation_count = cases.len();
    Ok(Report {
        schema_version: "matmul-g1-report-v1",
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        cpu_fallback_used: false,
        fallback: false,
        fallback_allowed: false,
        fallback_used: false,
        operations: operation_count,
        dispatch_count: operation_count,
        kernel_dispatches: operation_count,
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
                eprintln!("matmul-g1 report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("matmul-g1 evidence failed: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rne_ties_round_to_even() {
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x3f80_8001)), 0x3f81);
    }

    #[test]
    fn bf16_rne_preserves_specials_and_canonicalizes_nan() {
        assert_eq!(float_to_bf16_rne(0.0), 0x0000);
        assert_eq!(float_to_bf16_rne(-0.0), 0x8000);
        assert_eq!(float_to_bf16_rne(f32::INFINITY), 0x7f80);
        assert_eq!(float_to_bf16_rne(f32::NEG_INFINITY), 0xff80);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0x7fc1_2345)), 0x7fc1);
        assert_eq!(float_to_bf16_rne(f32::from_bits(0xffc2_2345)), 0xffc2);
        assert_eq!(bf16_to_float(0x0001).to_bits(), 0x0001_0000);
    }

    #[test]
    fn scalar_oracle_uses_ascending_k_and_separate_product_add() {
        let shape = CaseShape { m: 1, k: 3, n: 1 };
        let one = float_to_bf16_rne(1.0);
        let large = float_to_bf16_rne(16_777_216.0);
        let negative_large = float_to_bf16_rne(-16_777_216.0);
        let ascending_sensitive =
            scalar_matmul_oracle(shape, &[large, one, negative_large], &[one, one, one]);
        let cancellation_first =
            scalar_matmul_oracle(shape, &[large, negative_large, one], &[one, one, one]);
        assert_eq!(ascending_sensitive, vec![0x0000]);
        assert_eq!(cancellation_first, vec![0x3f80]);
    }

    #[test]
    fn required_case_coverage_is_bounded_and_non_cartesian() {
        assert_eq!(CASES.len(), 13);
        assert!(CASES.iter().any(|case| case.m == 1));
        assert!(CASES.iter().any(|case| case.m == 3));
        assert!(CASES.iter().any(|case| case.m == 17));
        assert!(CASES.iter().any(|case| case.k == 1));
        assert!(CASES.iter().any(|case| case.k == 3));
        assert!(CASES.iter().any(|case| case.k == 17));
        assert!(CASES.iter().any(|case| case.n == 1));
        assert!(CASES.iter().any(|case| case.n == 3));
        assert!(CASES.iter().any(|case| case.n == 17));
        for boundary in [255, 256, 257] {
            assert!(CASES.iter().any(|case| case.m == boundary));
            assert!(CASES.iter().any(|case| case.k == boundary));
            assert!(CASES.iter().any(|case| case.n == boundary));
        }
        for shape in CASES {
            assert!(shape.m > 0 && shape.k > 0 && shape.n > 0);
            let (_, _, output) = shape_element_count(shape).unwrap();
            assert_eq!(output, shape.m * shape.n);
        }
    }

    #[test]
    fn deterministic_operands_cover_required_special_values() {
        let mut activation_values = Vec::new();
        let mut weight_values = Vec::new();
        for (index, shape) in CASES.iter().copied().enumerate() {
            let (activation, weight) = make_operands(shape, index).unwrap();
            activation_values.extend(activation);
            weight_values.extend(weight);
        }
        for word in SPECIAL_VALUES {
            assert!(activation_values.contains(&word) || weight_values.contains(&word));
        }
        assert!(activation_values.contains(&0x7f00));
        assert!(weight_values.contains(&0x7f00));
    }
}
