//! Focused G1 evidence for the backend-neutral bounded transfer path.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{Backend, ExecutionSessionRequest, ExecutionState};
use sllm_hip::HipBackend;

const CASE_SIZES: [usize; 6] = [1, 3, 17, 255, 256, 257];
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
}

#[derive(Serialize)]
struct CaseEvidence {
    id: String,
    order: usize,
    offset_bytes: u64,
    size_bytes: u64,
    h2d_state: &'static str,
    d2h_state: &'static str,
    exact_match: bool,
}

#[derive(Serialize)]
struct Scope {
    selected_backend: &'static str,
    fallback_allowed: bool,
    fallback_used: bool,
    cpu_fallback_used: bool,
    gpu_execution: bool,
    model_used: bool,
    semantic_op_used: bool,
    kernel_dispatch_count: u32,
}

#[derive(Serialize)]
struct Counts {
    cases: usize,
    allocations: usize,
    h2d_transfers: usize,
    d2h_transfers: usize,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    max_transfer_bytes: u64,
    scope: Scope,
    counts: Counts,
    cases: Vec<CaseEvidence>,
    cleanup: Cleanup,
}

#[derive(Serialize)]
struct Cleanup {
    retryable_cleanup: usize,
    durable_quarantine: usize,
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
                let value = arguments
                    .next()
                    .ok_or_else(|| "--device-index requires a value".to_owned())?;
                device_index = Some(
                    value
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

fn run(config: &Config) -> Result<Report, String> {
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("invalid execution-session request: {error}"))?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| format!("execution-session open failed: {error}"))?;

    let operation = (|| {
        let queue = session
            .create_queue()
            .map_err(|error| format!("queue creation failed: {error}"))?;
        let max_transfer_bytes = session
            .max_transfer_bytes()
            .map_err(|error| format!("transfer capability query failed: {error}"))?;
        if max_transfer_bytes != 1_073_741_824 {
            return Err("HIP transfer capability differs from the public 1 GiB ABI".to_owned());
        }

        let mut cases = Vec::with_capacity(CASE_SIZES.len());
        for (order, size) in CASE_SIZES.into_iter().enumerate() {
            let offset = u64::try_from(order * 3 + 1)
                .map_err(|_| "case offset does not fit u64".to_owned())?;
            let size_u64 =
                u64::try_from(size).map_err(|_| "case size does not fit u64".to_owned())?;
            let allocation_size = offset
                .checked_add(size_u64)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| "case allocation size overflow".to_owned())?;
            let buffer = session
                .allocate(allocation_size)
                .map_err(|error| format!("case {order} allocation failed: {error}"))?;
            let range = buffer
                .range(offset, size_u64)
                .map_err(|error| format!("case {order} range failed: {error}"))?;
            let input: Vec<u8> = (0..size)
                .map(|index| ((index * 37 + order * 19 + 11) % 251) as u8)
                .collect();

            let mut upload = session
                .upload(&queue, range.clone(), Arc::<[u8]>::from(input.clone()))
                .map_err(|error| format!("case {order} H2D submit failed: {error}"))?;
            wait_success(upload.wait(WAIT_TIMEOUT), "H2D completion")?;

            let mut readback = session
                .readback(&queue, range)
                .map_err(|error| format!("case {order} D2H submit failed: {error}"))?;
            wait_success(readback.wait(WAIT_TIMEOUT), "D2H completion")?;
            let mut output = vec![0_u8; size];
            let copied = readback
                .read_into(&mut output)
                .map_err(|error| format!("case {order} D2H read failed: {error}"))?;
            if copied != size_u64 || output != input {
                return Err(format!("case {order} byte-exact comparison failed"));
            }
            cases.push(CaseEvidence {
                id: format!("bytes-{size}"),
                order,
                offset_bytes: offset,
                size_bytes: size_u64,
                h2d_state: "success",
                d2h_state: "success",
                exact_match: true,
            });
        }
        Ok((max_transfer_bytes, cases))
    })();

    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("execution-session shutdown failed: {error}"))?;
    let (max_transfer_bytes, cases) = operation?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("execution cleanup did not return to zero owned work".to_owned());
    }
    Ok(Report {
        schema_version: "execution-transfer-g1-report-v1",
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        max_transfer_bytes,
        scope: Scope {
            selected_backend: "hip",
            fallback_allowed: false,
            fallback_used: false,
            cpu_fallback_used: false,
            gpu_execution: true,
            model_used: false,
            semantic_op_used: false,
            kernel_dispatch_count: 0,
        },
        counts: Counts {
            cases: cases.len(),
            allocations: cases.len(),
            h2d_transfers: cases.len(),
            d2h_transfers: cases.len(),
        },
        cases,
        cleanup: Cleanup {
            retryable_cleanup: cleanup.retryable_cleanup,
            durable_quarantine: cleanup.durable_quarantine,
        },
    })
}

fn main() -> ExitCode {
    let result = parse_config().and_then(|config| run(&config));
    match result {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(output) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("execution-transfer-g1: report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("execution-transfer-g1: {error}");
            ExitCode::from(2)
        }
    }
}
