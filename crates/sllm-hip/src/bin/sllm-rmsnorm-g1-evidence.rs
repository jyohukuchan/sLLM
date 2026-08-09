//! Semantic RMSNorm G1 evidence executable.
//!
//! The stdin/stdout protocol is deliberately a bounded, ephemeral binary
//! protocol.  It carries BF16 input bytes into this process and returns the
//! actual BF16 output bytes only on stdout; no input/output array is written
//! to a file or embedded in metadata.  The executable uses only the public
//! Rust/C/native RMSNorm path.  A stub build fails at `HipBackend::connect`
//! and never emits a success response.

use std::env;
use std::io::{self, Read, Write};
use std::sync::Arc;
use std::time::Duration;

use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, DispatchEvidence, ExecutionSessionRequest,
    ExecutionState, RmsNormScaleMode, SemanticOpDescriptor, TensorView,
};
use sllm_hip::HipBackend;

const INPUT_MAGIC: [u8; 8] = *b"SLLMG1IN";
const OUTPUT_MAGIC: [u8; 8] = *b"SLLMG1OT";
const INPUT_PROTOCOL_VERSION: u32 = 1;
// v2 adds event counts observed by this executable after each successful
// public-runtime operation.  The controller consumes these raw words rather
// than accepting Python-side accounting constants.
const OUTPUT_PROTOCOL_VERSION: u32 = 2;
const INPUT_HEADER_BYTES: u32 = 112;
const OUTPUT_HEADER_BYTES: u32 = 428;
const MAX_ELEMENTS: u64 = 262_144;
const MAX_N: u64 = 4_096;
const BF16_BYTES: u64 = 2;
const MAX_INPUT_BYTES: u64 =
    INPUT_HEADER_BYTES as u64 + MAX_ELEMENTS * BF16_BYTES + MAX_N * BF16_BYTES;
const MAX_OUTPUT_BYTES: u64 = OUTPUT_HEADER_BYTES as u64 + MAX_ELEMENTS * BF16_BYTES;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const CLEANUP_ATTEMPTS: usize = 16;
const HIP_BACKEND: u32 = 1;
const RMSNORM_KERNEL_ID: u32 = 1;
const RMSNORM_WORKGROUP: u32 = 256;
const KERNEL_SYMBOL: &str = "rmsnorm.baseline.wave32.v1";
const DEVICE_SYMBOL: &str = "sllm_rmsnorm_baseline_wave32_v1";
const RMSNORM_SEMANTIC_OP_WIRE: u32 = 1;
const RMSNORM_CONTRACT_VERSION: u32 = 1;
const RMSNORM_ACCUMULATION_F32_WIRE: u32 = 3;
const RMSNORM_INPUT_COUNT: u32 = 2;
const RMSNORM_OUTPUT_COUNT: u32 = 1;
const RMSNORM_BINDING_COUNT: u32 = RMSNORM_INPUT_COUNT + RMSNORM_OUTPUT_COUNT;
const PROTOCOL_RESERVED_ZERO: u32 = 0;

#[derive(Clone, Debug)]
struct Config {
    device_index: u32,
    target: String,
}

#[derive(Debug)]
struct Request {
    shape: [u64; 8],
    rank: usize,
    epsilon_bits: u32,
    activation: Vec<u8>,
    raw_scale: Vec<u8>,
    element_count: u64,
    row_count: u64,
    normalized_size: u64,
}

#[derive(Debug)]
struct Evidence {
    output: Vec<u8>,
    shape: [u64; 8],
    rank: u32,
    element_count: u64,
    row_count: u64,
    normalized_size: u64,
    epsilon_bits: u32,
    device_index: u32,
    dispatch: DispatchEvidence,
    allocation_count: u32,
    copy_count: u32,
    kernel_count: u32,
    cleanup_pending: u64,
    cleanup_durable: u64,
    cleanup_accounting_errors: u64,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn parse_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| invalid("input header offset overflow"))?;
    let value = u32::from_le_bytes(
        bytes
            .get(*offset..end)
            .ok_or_else(|| invalid("input header is truncated"))?
            .try_into()
            .map_err(|_| invalid("input header field has the wrong size"))?,
    );
    *offset = end;
    Ok(value)
}

fn parse_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| invalid("input header offset overflow"))?;
    let value = u64::from_le_bytes(
        bytes
            .get(*offset..end)
            .ok_or_else(|| invalid("input header is truncated"))?
            .try_into()
            .map_err(|_| invalid("input header field has the wrong size"))?,
    );
    *offset = end;
    Ok(value)
}

fn validate_shape_and_lengths(
    rank: u32,
    shape: &[u64; 8],
    epsilon_bits: u32,
    activation_bytes: u64,
    scale_bytes: u64,
) -> Result<(usize, u64, u64, u64), String> {
    if !(1..=8).contains(&rank) {
        return Err(invalid("rank must be in 1..=8"));
    }
    let rank = usize::try_from(rank).map_err(|_| invalid("rank does not fit host usize"))?;
    if shape[rank..].iter().any(|value| *value != 0) {
        return Err(invalid("unused shape fields must be zero"));
    }
    let normalized_size = shape[rank - 1];
    if !(1..=MAX_N).contains(&normalized_size) {
        return Err(invalid("normalized size must be in 1..=4096"));
    }
    let mut element_count = 1_u64;
    for extent in &shape[..rank] {
        if *extent == 0 {
            return Err(invalid("tensor extents must be non-zero"));
        }
        element_count = element_count
            .checked_mul(*extent)
            .ok_or_else(|| invalid("tensor element count overflowed u64"))?;
        if element_count > MAX_ELEMENTS {
            return Err(invalid("semantic-G1 resource cap R*N<=262144 was exceeded"));
        }
    }
    let row_count = element_count / normalized_size;
    if row_count == 0 || element_count != row_count * normalized_size {
        return Err(invalid(
            "shape does not produce a positive integral row count",
        ));
    }
    let expected_activation_bytes = element_count
        .checked_mul(BF16_BYTES)
        .ok_or_else(|| invalid("activation byte length overflowed u64"))?;
    let expected_scale_bytes = normalized_size
        .checked_mul(BF16_BYTES)
        .ok_or_else(|| invalid("scale byte length overflowed u64"))?;
    if activation_bytes != expected_activation_bytes || scale_bytes != expected_scale_bytes {
        return Err(invalid("payload lengths do not match the BF16 shape"));
    }
    let epsilon = f32::from_bits(epsilon_bits);
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(invalid("epsilon must be finite and positive"));
    }
    Ok((rank, element_count, row_count, normalized_size))
}

fn read_request(reader: &mut impl Read) -> Result<Request, String> {
    let mut header = [0_u8; INPUT_HEADER_BYTES as usize];
    reader
        .read_exact(&mut header)
        .map_err(|error| format!("cannot read bounded input header: {error}"))?;
    if header[..INPUT_MAGIC.len()] != INPUT_MAGIC {
        return Err(invalid("input magic is invalid"));
    }
    let mut offset = INPUT_MAGIC.len();
    if parse_u32(&header, &mut offset)? != INPUT_PROTOCOL_VERSION {
        return Err(invalid("input protocol version is unsupported"));
    }
    if parse_u32(&header, &mut offset)? != INPUT_HEADER_BYTES {
        return Err(invalid("input header size is unsupported"));
    }
    let rank = parse_u32(&header, &mut offset)?;
    if parse_u32(&header, &mut offset)? != 0 {
        return Err(invalid("input reserved field is non-zero"));
    }
    let mut shape = [0_u64; 8];
    for extent in &mut shape {
        *extent = parse_u64(&header, &mut offset)?;
    }
    let epsilon_bits = parse_u32(&header, &mut offset)?;
    if parse_u32(&header, &mut offset)? != 0 {
        return Err(invalid("input reserved field is non-zero"));
    }
    let activation_bytes = parse_u64(&header, &mut offset)?;
    let scale_bytes = parse_u64(&header, &mut offset)?;
    if offset != header.len() {
        return Err(invalid(
            "input header parser did not consume the fixed header",
        ));
    }
    let (rank, element_count, row_count, normalized_size) =
        validate_shape_and_lengths(rank, &shape, epsilon_bits, activation_bytes, scale_bytes)?;
    let total_input_bytes = u64::from(INPUT_HEADER_BYTES)
        .checked_add(activation_bytes)
        .and_then(|value| value.checked_add(scale_bytes))
        .ok_or_else(|| invalid("bounded input size overflowed u64"))?;
    if total_input_bytes > MAX_INPUT_BYTES {
        return Err(invalid("bounded input protocol size was exceeded"));
    }
    let activation_len = usize::try_from(activation_bytes)
        .map_err(|_| invalid("activation payload does not fit host usize"))?;
    let scale_len = usize::try_from(scale_bytes)
        .map_err(|_| invalid("scale payload does not fit host usize"))?;
    let mut activation = vec![0_u8; activation_len];
    let mut raw_scale = vec![0_u8; scale_len];
    reader
        .read_exact(&mut activation)
        .map_err(|error| format!("cannot read activation payload: {error}"))?;
    reader
        .read_exact(&mut raw_scale)
        .map_err(|error| format!("cannot read raw-scale payload: {error}"))?;
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| format!("cannot check input protocol termination: {error}"))?
        != 0
    {
        return Err(invalid("input protocol has trailing bytes"));
    }
    Ok(Request {
        shape,
        rank,
        epsilon_bits,
        activation,
        raw_scale,
        element_count,
        row_count,
        normalized_size,
    })
}

fn parse_config() -> Result<Config, String> {
    let mut args = env::args().skip(1);
    let mut device_index = None;
    let mut target = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--device-index" => {
                if device_index.is_some() {
                    return Err(invalid("--device-index was provided more than once"));
                }
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--device-index requires a value"))?;
                device_index =
                    Some(value.parse::<u32>().map_err(|_| {
                        invalid("--device-index must be an unsigned 32-bit integer")
                    })?);
            }
            "--target" => {
                if target.is_some() {
                    return Err(invalid("--target was provided more than once"));
                }
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--target requires a value"))?;
                if value != "gfx1030" && value != "gfx1201" {
                    return Err(invalid("--target must be exactly gfx1030 or gfx1201"));
                }
                target = Some(value);
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| invalid("--device-index is required"))?,
        target: target.ok_or_else(|| invalid("--target is required"))?,
    })
}

fn wait_transfer(completion: &mut sllm_core::Transfer, label: &str) -> Result<(), String> {
    match completion
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("{label} completion failed: {error}"))?
    {
        ExecutionState::Success => Ok(()),
        ExecutionState::Pending => Err(format!("{label} completion remained pending")),
        ExecutionState::Failure => Err(format!("{label} completion reported failure")),
    }
}

fn wait_submission(completion: &mut sllm_core::Submission, label: &str) -> Result<(), String> {
    match completion
        .wait(WAIT_TIMEOUT)
        .map_err(|error| format!("{label} completion failed: {error}"))?
    {
        ExecutionState::Success => Ok(()),
        ExecutionState::Pending => Err(format!("{label} completion remained pending")),
        ExecutionState::Failure => Err(format!("{label} completion reported failure")),
    }
}

fn validate_dispatch(
    dispatch: &DispatchEvidence,
    request: &Request,
    target: &str,
) -> Result<(), String> {
    if dispatch.abi_version != 1
        || dispatch.info_version != 1
        || dispatch.dispatch_id == 0
        || dispatch.dispatch_count != 1
        || dispatch.kernel_id != RMSNORM_KERNEL_ID
        || dispatch.workgroup_size_x != RMSNORM_WORKGROUP
        || dispatch.grid_size_x != request.row_count as u32
        || dispatch.row_count != request.row_count
        || dispatch.normalized_size != request.normalized_size
        || dispatch.backend != HIP_BACKEND
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != KERNEL_SYMBOL
        || dispatch.device_symbol != DEVICE_SYMBOL
        || dispatch.target != target
    {
        return Err(invalid(
            "RMSNorm dispatch metadata violated the exact public contract",
        ));
    }
    Ok(())
}

fn run_request(config: &Config, request: Request) -> Result<Evidence, String> {
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let session_request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("execution-session request is invalid: {error}"))?;
    let session = backend
        .open_execution_session(session_request)
        .map_err(|error| format!("HIP execution-session open failed: {error}"))?;
    let result = (|| {
        let queue = session
            .create_queue()
            .map_err(|error| format!("queue creation failed: {error}"))?;
        let mut allocation_count = 0_u32;
        let activation_buffer = session
            .allocate(request.activation.len() as u64)
            .map_err(|error| format!("activation allocation failed: {error}"))?;
        allocation_count = allocation_count
            .checked_add(1)
            .ok_or_else(|| invalid("allocation event count overflow"))?;
        let scale_buffer = session
            .allocate(request.raw_scale.len() as u64)
            .map_err(|error| format!("raw-scale allocation failed: {error}"))?;
        allocation_count = allocation_count
            .checked_add(1)
            .ok_or_else(|| invalid("allocation event count overflow"))?;
        let output_buffer = session
            .allocate(request.activation.len() as u64)
            .map_err(|error| format!("output allocation failed: {error}"))?;
        allocation_count = allocation_count
            .checked_add(1)
            .ok_or_else(|| invalid("allocation event count overflow"))?;

        let activation_range = activation_buffer
            .range(0, request.activation.len() as u64)
            .map_err(|error| format!("activation upload range failed: {error}"))?;
        let mut copy_count = 0_u32;
        let mut activation_copy = session
            .upload(
                &queue,
                activation_range,
                Arc::<[u8]>::from(request.activation.clone()),
            )
            .map_err(|error| format!("activation H2D submission failed: {error}"))?;
        wait_transfer(&mut activation_copy, "activation H2D")?;
        copy_count = copy_count
            .checked_add(1)
            .ok_or_else(|| invalid("copy event count overflow"))?;
        let scale_range = scale_buffer
            .range(0, request.raw_scale.len() as u64)
            .map_err(|error| format!("raw-scale upload range failed: {error}"))?;
        let mut scale_copy = session
            .upload(
                &queue,
                scale_range,
                Arc::<[u8]>::from(request.raw_scale.clone()),
            )
            .map_err(|error| format!("raw-scale H2D submission failed: {error}"))?;
        wait_transfer(&mut scale_copy, "raw-scale H2D")?;
        copy_count = copy_count
            .checked_add(1)
            .ok_or_else(|| invalid("copy event count overflow"))?;

        let activation_view = TensorView::contiguous(
            DType::Bf16,
            &request.shape[..request.rank]
                .iter()
                .map(|extent| usize::try_from(*extent))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| invalid("activation shape does not fit host usize"))?,
        )
        .map_err(|error| format!("activation view creation failed: {error}"))?;
        let scale_view = TensorView::contiguous(
            DType::Bf16,
            &[usize::try_from(request.normalized_size)
                .map_err(|_| invalid("normalized size does not fit host usize"))?],
        )
        .map_err(|error| format!("scale view creation failed: {error}"))?;
        let output_view = activation_view.clone();
        let descriptor = Arc::new(
            SemanticOpDescriptor::new_rms_norm(
                vec![activation_view.clone(), scale_view.clone()],
                vec![output_view.clone()],
                f32::from_bits(request.epsilon_bits),
                RmsNormScaleMode::OffsetOne,
            )
            .map_err(|error| format!("RMSNorm semantic descriptor creation failed: {error}"))?,
        );
        let operation = Arc::new(
            BoundSemanticOp::new(
                descriptor,
                vec![
                    session
                        .bind(&activation_buffer, activation_view, AccessMode::Read)
                        .map_err(|error| format!("activation binding failed: {error}"))?,
                    session
                        .bind(&scale_buffer, scale_view, AccessMode::Read)
                        .map_err(|error| format!("raw-scale binding failed: {error}"))?,
                ],
                vec![
                    session
                        .bind(&output_buffer, output_view, AccessMode::Write)
                        .map_err(|error| format!("output binding failed: {error}"))?,
                ],
            )
            .map_err(|error| format!("RMSNorm binding validation failed: {error}"))?,
        );
        let prepared = session
            .prepare(operation)
            .map_err(|error| format!("public RMSNorm prepare failed: {error}"))?;
        let mut submission = session
            .submit(&prepared, &queue)
            .map_err(|error| format!("public RMSNorm execute failed: {error}"))?;
        validate_dispatch(submission.dispatch(), &request, &config.target)?;
        wait_submission(&mut submission, "RMSNorm")?;
        let kernel_count = submission.dispatch().dispatch_count;

        let mut output_copy = submission
            .start_output_readback(0)
            .map_err(|error| format!("output D2H submission failed: {error}"))?;
        match output_copy
            .wait(WAIT_TIMEOUT)
            .map_err(|error| format!("output D2H completion failed: {error}"))?
        {
            ExecutionState::Success => {}
            ExecutionState::Pending => {
                return Err(invalid("output D2H completion remained pending"));
            }
            ExecutionState::Failure => {
                return Err(invalid("output D2H completion reported failure"));
            }
        }
        copy_count = copy_count
            .checked_add(1)
            .ok_or_else(|| invalid("copy event count overflow"))?;
        let mut output = vec![0_u8; request.activation.len()];
        let bytes_written = output_copy
            .read_into(&mut output)
            .map_err(|error| format!("output D2H read failed: {error}"))?;
        if bytes_written != output.len() as u64 {
            return Err(invalid("output D2H length did not match the request"));
        }
        Ok((
            output,
            submission.dispatch().clone(),
            allocation_count,
            copy_count,
            kernel_count,
        ))
    })();

    let cleanup = session
        .shutdown(Duration::from_secs(CLEANUP_ATTEMPTS as u64))
        .map_err(|error| format!("public runtime cleanup failed: {error}"))?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err(invalid("public runtime cleanup was not empty and healthy"));
    }
    let (output, dispatch, allocation_count, copy_count, kernel_count) = result?;
    Ok(Evidence {
        output,
        shape: request.shape,
        rank: u32::try_from(request.rank).map_err(|_| invalid("rank does not fit u32"))?,
        element_count: request.element_count,
        row_count: request.row_count,
        normalized_size: request.normalized_size,
        epsilon_bits: request.epsilon_bits,
        device_index: config.device_index,
        dispatch,
        allocation_count,
        copy_count,
        kernel_count,
        cleanup_pending: cleanup.retryable_cleanup as u64,
        cleanup_durable: cleanup.durable_quarantine as u64,
        cleanup_accounting_errors: 0,
    })
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_fixed_string(writer: &mut impl Write, value: &str, width: usize) -> io::Result<()> {
    if width == 0 || !value.is_ascii() || value.as_bytes().contains(&0) || value.len() >= width {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "fixed protocol strings must be ASCII, NUL-free, and fit with a terminator",
        ));
    }
    let mut bytes = vec![0_u8; width];
    bytes[..value.len()].copy_from_slice(value.as_bytes());
    writer.write_all(&bytes)
}

fn write_response(writer: &mut impl Write, evidence: &Evidence) -> io::Result<()> {
    if u64::try_from(evidence.output.len()).unwrap_or(u64::MAX) > MAX_ELEMENTS * BF16_BYTES
        || MAX_OUTPUT_BYTES < u64::from(OUTPUT_HEADER_BYTES) + evidence.output.len() as u64
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "output exceeds the bounded response protocol",
        ));
    }
    if !(1..=8).contains(&evidence.rank)
        || evidence.shape[evidence.rank as usize..]
            .iter()
            .any(|value| *value != 0)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "response rank and trailing shape fields are inconsistent",
        ));
    }
    if evidence.dispatch.dispatch_count != 1
        || evidence.dispatch.dispatch_id == 0
        || evidence.allocation_count == 0
        || evidence.copy_count == 0
        || evidence.kernel_count != evidence.dispatch.dispatch_count
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "response dispatch metadata must describe one non-zero dispatch",
        ));
    }
    writer.write_all(&OUTPUT_MAGIC)?;
    write_u32(writer, OUTPUT_PROTOCOL_VERSION)?;
    write_u32(writer, OUTPUT_HEADER_BYTES)?;
    write_u32(writer, evidence.rank)?;
    write_u32(writer, PROTOCOL_RESERVED_ZERO)?;
    for extent in evidence.shape {
        write_u64(writer, extent)?;
    }
    write_u64(writer, evidence.element_count)?;
    write_u64(writer, evidence.normalized_size)?;
    write_u64(writer, evidence.row_count)?;
    write_u32(writer, evidence.epsilon_bits)?;
    write_u32(writer, PROTOCOL_RESERVED_ZERO)?;
    write_u32(writer, evidence.device_index)?;
    write_u32(writer, evidence.dispatch.backend)?;
    write_u64(writer, evidence.dispatch.dispatch_id)?;
    write_u32(writer, evidence.dispatch.dispatch_count)?;
    write_u32(writer, evidence.dispatch.kernel_id)?;
    write_u32(writer, evidence.dispatch.workgroup_size_x)?;
    write_u32(writer, evidence.dispatch.grid_size_x)?;
    write_u32(writer, u32::from(evidence.dispatch.fallback_allowed))?;
    write_u32(writer, u32::from(evidence.dispatch.fallback_used))?;
    // The seven words below are a fixed v1 RMSNorm response extension:
    // semantic operation, contract version, accumulation type, input/output
    // arity, total binding count, then an explicitly reserved zero word.
    write_u32(writer, RMSNORM_SEMANTIC_OP_WIRE)?;
    write_u32(writer, RMSNORM_CONTRACT_VERSION)?;
    write_u32(writer, RMSNORM_ACCUMULATION_F32_WIRE)?;
    write_u32(writer, RMSNORM_INPUT_COUNT)?;
    write_u32(writer, RMSNORM_OUTPUT_COUNT)?;
    write_u32(writer, RMSNORM_BINDING_COUNT)?;
    write_u32(writer, PROTOCOL_RESERVED_ZERO)?;
    write_u32(writer, evidence.allocation_count)?;
    write_u32(writer, evidence.copy_count)?;
    write_u32(writer, evidence.kernel_count)?;
    write_u32(writer, PROTOCOL_RESERVED_ZERO)?;
    write_u64(writer, evidence.cleanup_pending)?;
    write_u64(writer, evidence.cleanup_durable)?;
    write_u64(writer, evidence.cleanup_accounting_errors)?;
    write_fixed_string(writer, &evidence.dispatch.kernel_symbol, 64)?;
    write_fixed_string(writer, &evidence.dispatch.device_symbol, 64)?;
    write_fixed_string(writer, &evidence.dispatch.target, 64)?;
    write_u64(writer, evidence.output.len() as u64)?;
    writer.write_all(&evidence.output)
}

fn main() {
    let result = (|| {
        let config = parse_config()?;
        let mut stdin = io::stdin().lock();
        let request = read_request(&mut stdin)?;
        let evidence = run_request(&config, request)?;
        let mut stdout = io::stdout().lock();
        write_response(&mut stdout, &evidence)
            .map_err(|error| format!("cannot write bounded evidence response: {error}"))?;
        stdout
            .flush()
            .map_err(|error| format!("cannot flush bounded evidence response: {error}"))
    })();
    if let Err(error) = result {
        let _ = writeln!(io::stderr().lock(), "sllm-rmsnorm-g1-evidence: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(rank: u32, extents: &[u64]) -> [u64; 8] {
        let mut result = [0_u64; 8];
        result[..extents.len()].copy_from_slice(extents);
        assert_eq!(rank as usize, extents.len());
        result
    }

    #[test]
    fn semantic_g1_cap_accepts_exact_product_and_rejects_one_over() {
        let exact = shape(2, &[64, 4096]);
        assert!(
            validate_shape_and_lengths(
                2,
                &exact,
                1.0e-6_f32.to_bits(),
                MAX_ELEMENTS * BF16_BYTES,
                MAX_N * BF16_BYTES,
            )
            .is_ok()
        );
        let over = shape(2, &[65, 4096]);
        let error = validate_shape_and_lengths(
            2,
            &over,
            1.0e-6_f32.to_bits(),
            65 * 4096 * BF16_BYTES,
            MAX_N * BF16_BYTES,
        )
        .expect_err("one element over the evidence cap must fail");
        assert!(error.contains("262144"));
    }

    #[test]
    fn public_shape_boundaries_and_epsilon_fail_closed_before_payload_allocation() {
        let too_wide = shape(1, &[4097]);
        assert!(
            validate_shape_and_lengths(
                1,
                &too_wide,
                1.0e-6_f32.to_bits(),
                4097 * BF16_BYTES,
                4097 * BF16_BYTES,
            )
            .is_err()
        );
        let invalid_epsilon = shape(1, &[1]);
        let error = validate_shape_and_lengths(1, &invalid_epsilon, 0, 2, 2)
            .expect_err("zero epsilon must fail closed");
        assert!(error.contains("epsilon"));
    }

    #[test]
    fn exact_dispatch_metadata_is_required_before_output_is_accepted() {
        let request = Request {
            shape: shape(2, &[2, 17]),
            rank: 2,
            epsilon_bits: 1.0e-6_f32.to_bits(),
            activation: vec![0; 2 * 17 * 2],
            raw_scale: vec![0; 17 * 2],
            element_count: 34,
            row_count: 2,
            normalized_size: 17,
        };
        let dispatch = DispatchEvidence {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 1,
            dispatch_count: 1,
            kernel_id: RMSNORM_KERNEL_ID,
            workgroup_size_x: RMSNORM_WORKGROUP,
            grid_size_x: 2,
            row_count: 2,
            normalized_size: 17,
            backend: HIP_BACKEND,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: KERNEL_SYMBOL.to_owned(),
            device_symbol: DEVICE_SYMBOL.to_owned(),
            target: "gfx1030".to_owned(),
        };
        assert!(validate_dispatch(&dispatch, &request, "gfx1030").is_ok());
        let mut fallback = dispatch.clone();
        fallback.fallback_used = true;
        assert!(validate_dispatch(&fallback, &request, "gfx1030").is_err());
        let mut wrong_grid = dispatch;
        wrong_grid.grid_size_x = 1;
        assert!(validate_dispatch(&wrong_grid, &request, "gfx1030").is_err());
    }

    #[test]
    fn protocol_limits_are_explicit() {
        assert_eq!(INPUT_HEADER_BYTES, 112);
        assert_eq!(OUTPUT_HEADER_BYTES, 428);
        const { assert!(MAX_INPUT_BYTES < 600_000) };
        const { assert!(MAX_OUTPUT_BYTES < 600_000) };
    }

    fn response_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    fn response_u64(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    fn response_evidence(rank: u32) -> Evidence {
        let mut shape = [0_u64; 8];
        shape[..rank as usize].copy_from_slice(&[2, 3, 5][..rank as usize]);
        Evidence {
            output: vec![0; 60],
            shape,
            rank,
            element_count: 30,
            row_count: 6,
            normalized_size: 5,
            epsilon_bits: 1.0e-6_f32.to_bits(),
            device_index: 0,
            dispatch: DispatchEvidence {
                abi_version: 1,
                info_version: 1,
                dispatch_id: 9,
                dispatch_count: 1,
                kernel_id: RMSNORM_KERNEL_ID,
                workgroup_size_x: RMSNORM_WORKGROUP,
                grid_size_x: 6,
                row_count: 6,
                normalized_size: 5,
                backend: HIP_BACKEND,
                fallback_allowed: false,
                fallback_used: false,
                kernel_symbol: KERNEL_SYMBOL.to_owned(),
                device_symbol: DEVICE_SYMBOL.to_owned(),
                target: "gfx1030".to_owned(),
            },
            allocation_count: 3,
            copy_count: 3,
            kernel_count: 1,
            cleanup_pending: 0,
            cleanup_durable: 0,
            cleanup_accounting_errors: 0,
        }
    }

    #[test]
    fn response_protocol_pins_v2_rmsnorm_wire_literals_and_relationships() {
        let evidence = response_evidence(3);
        let mut bytes = Vec::new();
        write_response(&mut bytes, &evidence).expect("response must serialize");
        assert_eq!(bytes.len(), 428 + 60);

        // These offsets and values are an independent v2 wire contract. Keep
        // them literal so changing the serializer constants cannot make this
        // test agree with a changed protocol by construction.
        assert_eq!(&bytes[0..8], b"SLLMG1OT");
        assert_eq!(response_u32(&bytes, 8), 2);
        assert_eq!(response_u32(&bytes, 12), 428);
        assert_eq!(response_u32(&bytes, 16), 3);
        assert_eq!(response_u32(&bytes, 20), 0);
        assert_eq!(response_u64(&bytes, 24), 2);
        assert_eq!(response_u64(&bytes, 32), 3);
        assert_eq!(response_u64(&bytes, 40), 5);
        assert_eq!(response_u64(&bytes, 48), 0);
        assert_eq!(response_u64(&bytes, 56), 0);
        assert_eq!(response_u64(&bytes, 64), 0);
        assert_eq!(response_u64(&bytes, 72), 0);
        assert_eq!(response_u64(&bytes, 80), 0);
        assert_eq!(response_u64(&bytes, 88), 30);
        assert_eq!(response_u64(&bytes, 96), 5);
        assert_eq!(response_u64(&bytes, 104), 6);
        assert_eq!(response_u32(&bytes, 116), 0);

        assert_eq!(response_u32(&bytes, 160), 1);
        assert_eq!(response_u32(&bytes, 164), 1);
        assert_eq!(response_u32(&bytes, 168), 3);
        let input_count = response_u32(&bytes, 172);
        let output_count = response_u32(&bytes, 176);
        let binding_count = response_u32(&bytes, 180);
        assert_eq!(input_count, 2);
        assert_eq!(output_count, 1);
        assert_eq!(binding_count, 3);
        assert_eq!(input_count + output_count, binding_count);
        assert_eq!(response_u32(&bytes, 184), 0);
        assert_eq!(response_u32(&bytes, 188), 3);
        assert_eq!(response_u32(&bytes, 192), 3);
        assert_eq!(response_u32(&bytes, 196), 1);
        assert_eq!(response_u32(&bytes, 200), 0);
        assert_eq!(response_u64(&bytes, 420), 60);
        assert_eq!(&bytes[428..], &[0; 60]);
    }

    #[test]
    fn response_protocol_rejects_zero_or_inconsistent_rank() {
        let mut bytes = Vec::new();
        assert!(write_response(&mut bytes, &response_evidence(0)).is_err());
        let mut invalid = response_evidence(2);
        invalid.shape[2] = 1;
        assert!(write_response(&mut bytes, &invalid).is_err());
    }

    #[test]
    fn fixed_protocol_strings_reject_lossy_values_and_accept_the_boundary() {
        let mut bytes = Vec::new();
        write_fixed_string(&mut bytes, "abcdefg", 8).expect("width minus one must fit");
        assert_eq!(&bytes, b"abcdefg\0");
        assert!(write_fixed_string(&mut Vec::new(), "abcdefgh", 8).is_err());
        assert!(write_fixed_string(&mut Vec::new(), "abc\0def", 8).is_err());
        assert!(write_fixed_string(&mut Vec::new(), "café", 8).is_err());
    }
}
