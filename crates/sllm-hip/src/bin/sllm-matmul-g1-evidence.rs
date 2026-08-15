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
const WORKGROUP_SIZE: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CaseShape {
    m: usize,
    k: usize,
    n: usize,
}

// Keep the boundary coverage broad without taking the Cartesian product of
// all M/K/N values.  K=5 is retained as a small non-boundary reduction case.
const CASES: [CaseShape; 17] = [
    CaseShape { m: 1, k: 1, n: 1 },
    CaseShape { m: 1, k: 3, n: 17 },
    CaseShape { m: 3, k: 17, n: 3 },
    CaseShape { m: 17, k: 3, n: 1 },
    CaseShape {
        m: 1,
        k: 255,
        n: 17,
    },
    CaseShape { m: 1, k: 256, n: 3 },
    CaseShape { m: 1, k: 257, n: 3 },
    CaseShape {
        m: 3,
        k: 256,
        n: 31,
    },
    CaseShape { m: 3, k: 5, n: 255 },
    CaseShape { m: 3, k: 5, n: 256 },
    CaseShape { m: 3, k: 5, n: 257 },
    CaseShape { m: 255, k: 3, n: 1 },
    CaseShape { m: 256, k: 3, n: 1 },
    CaseShape { m: 257, k: 3, n: 1 },
    CaseShape {
        m: 17,
        k: 257,
        n: 33,
    },
    CaseShape {
        m: 37,
        k: 1025,
        n: 65,
    },
    CaseShape {
        m: 1,
        k: 2560,
        n: 9216,
    },
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
    kernel_elapsed_ns: u64,
    exact_match: bool,
    numerical_match: bool,
    max_abs_error: f64,
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

// 0x5c00 is deliberately large but leaves enough FP32 exponent headroom even
// when both operands use it and all 1025 required products have the same
// sign. A near-maximum BF16 would turn this finite tolerance case into an
// FP32 product or reduction overflow case.
const ORDINARY_FINITE: [u16; 6] = [0x3f80, 0xbf80, 0x3fc0, 0xc020, 0x4000, 0x5c00];
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
    if shape.k > 1025 || shape.n > 1024 {
        let activation = (0..activation_count)
            .map(|index| ORDINARY_FINITE[index % 5])
            .collect();
        let weight = (0..weight_count)
            .map(|index| ORDINARY_FINITE[(index * 3 + 1) % 5])
            .collect();
        return Ok((activation, weight));
    }
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
            activation[row * shape.k] = SPECIAL_VALUES[(row + case_index) % SPECIAL_VALUES.len()];
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
    if shape == (CaseShape { m: 1, k: 256, n: 3 }) {
        // One explicit opposite-infinity reduction fixes the NaN output
        // classification promised by the frozen numerical manifest.
        activation[0] = 0x7f80;
        activation[1] = 0xff80;
        weight[0] = 0x3f80;
        weight[1] = 0x3f80;
    }
    Ok((activation, weight))
}

fn f64_to_bf16_rne(value: f64) -> u16 {
    if value.is_nan() {
        return if value.is_sign_negative() {
            0xffc0
        } else {
            0x7fc0
        };
    }
    if value == f64::INFINITY {
        return 0x7f80;
    }
    if value == f64::NEG_INFINITY {
        return 0xff80;
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            0x8000
        } else {
            0x0000
        };
    }

    let magnitude = value.abs();
    let exponent = magnitude.log2().floor() as i32;
    let quantum_exponent = if exponent < -126 { -133 } else { exponent - 7 };
    let quantum = 2.0_f64.powi(quantum_exponent);
    let rounded = (magnitude / quantum).round_ties_even() * quantum;
    let signed = if value.is_sign_negative() {
        -rounded
    } else {
        rounded
    };
    float_to_bf16_rne(signed as f32)
}

/// Independent exact-input oracle. BF16 operands and their products are
/// exactly representable in f64; reductions visit k in ascending order and
/// are rounded once to BF16 at the output boundary.
#[inline(never)]
fn scalar_matmul_oracle(
    shape: CaseShape,
    activation: &[u16],
    weight: &[u16],
) -> (Vec<f64>, Vec<u16>) {
    let mut exact = Vec::with_capacity(shape.m * shape.n);
    let mut rounded = Vec::with_capacity(shape.m * shape.n);
    for row in 0..shape.m {
        for column in 0..shape.n {
            let mut accumulator = 0.0_f64;
            for reduction in 0..shape.k {
                let product = f64::from(bf16_to_float(activation[row * shape.k + reduction]))
                    * f64::from(bf16_to_float(weight[column * shape.k + reduction]));
                accumulator += product;
            }
            exact.push(accumulator);
            rounded.push(f64_to_bf16_rne(accumulator));
        }
    }
    (exact, rounded)
}

fn compare_phase8_numerics(
    shape: CaseShape,
    activation: &[u16],
    weight: &[u16],
    exact_reference: &[f64],
    actual: &[u8],
) -> Result<f64, String> {
    let actual_words = actual
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    if actual_words.len() != exact_reference.len() {
        return Err("matmul numerical comparison length mismatch".to_owned());
    }
    let unit_roundoff = 2.0_f64.powi(-24);
    let gamma = shape.k as f64 * unit_roundoff / (1.0 - shape.k as f64 * unit_roundoff);
    let mut max_abs_error = 0.0_f64;
    for row in 0..shape.m {
        for column in 0..shape.n {
            let index = row * shape.n + column;
            let reference = exact_reference[index];
            let actual_word = actual_words[index];
            let observed = f64::from(bf16_to_float(actual_word));
            if reference.is_nan() {
                if !observed.is_nan() {
                    return Err("matmul NaN classification mismatch".to_owned());
                }
                continue;
            }
            if reference.is_infinite() {
                if observed != reference {
                    return Err(format!(
                        "matmul infinity classification mismatch at row {row} column {column}: reference={reference} observed={observed}"
                    ));
                }
                continue;
            }
            let rounded_reference = bf16_to_float(f64_to_bf16_rne(reference));
            if rounded_reference.is_infinite() {
                if observed != f64::from(rounded_reference) {
                    return Err("matmul finite-overflow classification mismatch".to_owned());
                }
                continue;
            }
            if !observed.is_finite() {
                return Err("matmul finite result became non-finite".to_owned());
            }
            let mut sum_abs_products = 0.0_f64;
            for reduction in 0..shape.k {
                sum_abs_products += f64::from(bf16_to_float(activation[row * shape.k + reduction]))
                    .abs()
                    * f64::from(bf16_to_float(weight[column * shape.k + reduction])).abs();
            }
            let half_ulp = if reference == 0.0 || reference.abs() < 2.0_f64.powi(-126) {
                2.0_f64.powi(-134)
            } else {
                let exponent = reference.abs().log2().floor() as i32;
                2.0_f64.powi(exponent - 8)
            };
            let error = (observed - reference).abs();
            max_abs_error = max_abs_error.max(error);
            if error > gamma * sum_abs_products + half_ulp {
                return Err(format!(
                    "matmul output exceeds frozen Phase 8 bound at row {row} column {column}"
                ));
            }
        }
    }
    Ok(max_abs_error)
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
    let forced_baseline = env::var("SLLM_MATMUL_FORCE_BASELINE").as_deref() == Ok("1");
    let (expected_kernel, expected_grid, expected_symbol, expected_device_symbol) =
        if forced_baseline {
            (
                1,
                output_elements.div_ceil(WORKGROUP_SIZE as usize) as u32,
                "matmul.bf16_fp32.v1",
                "sllm_matmul_bf16_fp32_v1",
            )
        } else if shape.m > 1 && target == "gfx1201" {
            (
                4,
                shape.n as u32,
                "matmul.hipblas.gemm_ex.v2",
                "hipblasGemmEx",
            )
        } else if shape.m == 1 {
            (
                3,
                shape.n as u32,
                "matmul.bf16_fp32.decode.v4",
                "sllm_matmul_bf16_fp32_decode_v4",
            )
        } else {
            (
                2,
                shape.n.div_ceil(16) as u32,
                "matmul.bf16_fp32.tiled16.v2",
                "sllm_matmul_bf16_fp32_tiled16_v2",
            )
        };
    if dispatch.abi_version != 1
        || dispatch.info_version != 1
        || dispatch.dispatch_id == 0
        || dispatch.dispatch_count != 1
        || dispatch.kernel_id != expected_kernel
        || dispatch.workgroup_size_x != WORKGROUP_SIZE
        || dispatch.grid_size_x != expected_grid
        || dispatch.row_count != shape.m as u64
        || dispatch.normalized_size != output_elements as u64
        || dispatch.backend != HIP_BACKEND
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != expected_symbol
        || dispatch.device_symbol != expected_device_symbol
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
    let (exact_reference, expected_words) =
        scalar_matmul_oracle(shape, &activation_words, &weight_words);
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
    let kernel_elapsed_ns = submission
        .kernel_elapsed_ns()
        .map_err(|error| format!("matmul kernel timing failed: {error}"))?
        .ok_or_else(|| "HIP matmul did not publish GPU kernel timing".to_owned())?;
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
    let exact_match = actual == output_bytes;
    let max_abs_error = compare_phase8_numerics(
        shape,
        &activation_words,
        &weight_words,
        &exact_reference,
        &actual,
    )?;

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
        kernel_elapsed_ns,
        exact_match,
        numerical_match: true,
        max_abs_error,
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
    fn scalar_oracle_accumulates_exact_bf16_products_in_f64() {
        let shape = CaseShape { m: 1, k: 3, n: 1 };
        let one = float_to_bf16_rne(1.0);
        let large = float_to_bf16_rne(16_777_216.0);
        let negative_large = float_to_bf16_rne(-16_777_216.0);
        let (ascending_exact, ascending_rounded) =
            scalar_matmul_oracle(shape, &[large, one, negative_large], &[one, one, one]);
        let (cancellation_exact, cancellation_rounded) =
            scalar_matmul_oracle(shape, &[large, negative_large, one], &[one, one, one]);
        assert_eq!(ascending_exact, vec![1.0]);
        assert_eq!(cancellation_exact, vec![1.0]);
        assert_eq!(ascending_rounded, vec![0x3f80]);
        assert_eq!(cancellation_rounded, vec![0x3f80]);
    }

    #[test]
    fn f64_to_bf16_rne_handles_ties_subnormals_and_overflow() {
        assert_eq!(f64_to_bf16_rne(1.0 + 2.0_f64.powi(-8)), 0x3f80);
        assert_eq!(f64_to_bf16_rne(1.0 + 3.0 * 2.0_f64.powi(-8)), 0x3f82);
        assert_eq!(f64_to_bf16_rne(2.0_f64.powi(-134)), 0x0000);
        assert_eq!(f64_to_bf16_rne(3.0 * 2.0_f64.powi(-134)), 0x0002);
        assert_eq!(
            f64_to_bf16_rne((2.0 - 2.0_f64.powi(-8)) * 2.0_f64.powi(127)),
            0x7f80
        );
    }

    #[test]
    fn required_case_coverage_is_bounded_and_non_cartesian() {
        assert_eq!(CASES.len(), 17);
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
        for required in [
            CaseShape { m: 1, k: 1, n: 1 },
            CaseShape {
                m: 1,
                k: 255,
                n: 17,
            },
            CaseShape {
                m: 3,
                k: 256,
                n: 31,
            },
            CaseShape {
                m: 17,
                k: 257,
                n: 33,
            },
            CaseShape {
                m: 37,
                k: 1025,
                n: 65,
            },
        ] {
            assert!(CASES.contains(&required));
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
        assert!(activation_values.contains(&0x5c00));
        assert!(weight_values.contains(&0x5c00));

        let row_shape = CaseShape { m: 3, k: 17, n: 3 };
        let (row_activation, _) = make_operands(row_shape, 2).unwrap();
        for row in 0..row_shape.m {
            assert!(SPECIAL_VALUES.contains(&row_activation[row * row_shape.k]));
        }

        let shape = CaseShape { m: 1, k: 256, n: 3 };
        let (activation, weight) = make_operands(shape, 5).unwrap();
        let (exact, _) = scalar_matmul_oracle(shape, &activation, &weight);
        assert!(exact[0].is_nan());
    }
}
