//! Distinctive-row GPU evidence for the Phase 24 terminal-row contract.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, Encoding, ExecutionSessionRequest, ExecutionState,
    SemanticOpDescriptor, SemanticOpKind, TensorView,
};
use sllm_hip::HipBackend;

const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);
const WIDTH: usize = 257;
const ROWS: [usize; 6] = [2, 3, 17, 255, 256, 257];

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
}

#[derive(Serialize)]
struct CaseEvidence {
    source_rows: usize,
    selected_row: usize,
    selected_byte_offset: u64,
    projected_rows: u64,
    argmax_rows: u64,
    expected_argmax: i32,
    actual_argmax: i32,
    max_abs_error: f64,
    fallback_used: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    selected_backend: &'static str,
    fallback_used: bool,
    cases: Vec<CaseEvidence>,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

fn parse_config() -> Result<Config, String> {
    let mut arguments = env::args().skip(1);
    let device_index = arguments
        .next()
        .ok_or_else(|| "device index is required".to_owned())?
        .parse::<u32>()
        .map_err(|_| "device index must be u32".to_owned())?;
    let target = arguments
        .next()
        .ok_or_else(|| "target is required".to_owned())?;
    if !matches!(target.as_str(), "gfx1030" | "gfx1201" | "gfx942") {
        return Err("target must be gfx1030, gfx1201, or gfx942".to_owned());
    }
    if arguments.next().is_some() {
        return Err("usage: DEVICE TARGET".to_owned());
    }
    Ok(Config {
        device_index,
        target,
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

fn words_to_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    rows: usize,
) -> Result<CaseEvidence, String> {
    let selected_row = rows - 1;
    let selected_byte_offset = u64::try_from(selected_row * WIDTH * 2)
        .map_err(|_| "selected-row offset overflowed".to_owned())?;
    let mut activation = vec![0_u16; rows * WIDTH];
    for row in 0..rows {
        activation[row * WIDTH + row] = 0x3f80;
    }
    let mut weight = vec![0_u16; WIDTH * WIDTH];
    for diagonal in 0..WIDTH {
        weight[diagonal * WIDTH + diagonal] = 0x3f80;
    }
    let activation_bytes = words_to_bytes(&activation);
    let weight_bytes = words_to_bytes(&weight);
    let activation_buffer = session
        .allocate(activation_bytes.len() as u64)
        .map_err(|error| format!("activation allocation failed: {error}"))?;
    let weight_buffer = session
        .allocate(weight_bytes.len() as u64)
        .map_err(|error| format!("weight allocation failed: {error}"))?;
    let logits_buffer = session
        .allocate((WIDTH * 2) as u64)
        .map_err(|error| format!("logits allocation failed: {error}"))?;
    let argmax_buffer = session
        .allocate(4)
        .map_err(|error| format!("Argmax allocation failed: {error}"))?;

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
            .map_err(|error| format!("{label} upload failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), &format!("{label} upload"))?;
    }

    let activation_view = TensorView::new(
        DType::Bf16,
        Encoding::Unquantized,
        &[1, WIDTH],
        &[WIDTH, 1],
        selected_byte_offset,
    )
    .map_err(|error| format!("terminal activation view failed: {error}"))?;
    let weight_view = TensorView::contiguous(DType::Bf16, &[WIDTH, WIDTH])
        .map_err(|error| format!("weight view failed: {error}"))?;
    let logits_view = TensorView::contiguous(DType::Bf16, &[1, WIDTH])
        .map_err(|error| format!("logits view failed: {error}"))?;
    let projection = Arc::new(
        BoundSemanticOp::new(
            Arc::new(
                SemanticOpDescriptor::new(
                    SemanticOpKind::Matmul,
                    vec![activation_view.clone(), weight_view.clone()],
                    vec![logits_view.clone()],
                )
                .map_err(|error| format!("projection descriptor failed: {error}"))?,
            ),
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
                    .bind(&logits_buffer, logits_view.clone(), AccessMode::Write)
                    .map_err(|error| format!("logits binding failed: {error}"))?,
            ],
        )
        .map_err(|error| format!("projection binding failed: {error}"))?,
    );
    let prepared_projection = session
        .prepare(projection)
        .map_err(|error| format!("projection prepare failed: {error}"))?;
    let mut projection_submission = session
        .submit(&prepared_projection, queue)
        .map_err(|error| format!("projection submit failed: {error}"))?;
    wait_success(
        projection_submission.wait(WAIT_TIMEOUT),
        "projection completion",
    )?;
    let projection_dispatch = projection_submission.dispatch().clone();
    if projection_dispatch.row_count != 1
        || projection_dispatch.normalized_size != WIDTH as u64
        || projection_dispatch.fallback_allowed
        || projection_dispatch.fallback_used
    {
        return Err("terminal projection dispatch violated the one-row contract".to_owned());
    }
    let mut logits_readback = projection_submission
        .start_output_readback(0)
        .map_err(|error| format!("logits readback failed: {error}"))?;
    wait_success(logits_readback.wait(WAIT_TIMEOUT), "logits readback")?;
    let mut logits = vec![0_u8; WIDTH * 2];
    if logits_readback
        .read_into(&mut logits)
        .map_err(|error| format!("logits read failed: {error}"))?
        != logits.len() as u64
    {
        return Err("logits readback byte count differed".to_owned());
    }
    let mut max_abs_error = 0.0_f64;
    for (column, word) in logits.chunks_exact(2).enumerate() {
        let actual = f64::from(f32::from_bits(
            u32::from(u16::from_le_bytes([word[0], word[1]])) << 16,
        ));
        let reference = if column == selected_row { 1.0 } else { 0.0 };
        let error = (actual - reference).abs();
        max_abs_error = max_abs_error.max(error);
        if error > 0.015625 + 0.015625 * reference.abs() {
            return Err(format!(
                "terminal projection exceeded tolerance at row {selected_row}, column {column}"
            ));
        }
    }

    let argmax_view = TensorView::contiguous(DType::I32, &[1])
        .map_err(|error| format!("Argmax view failed: {error}"))?;
    let argmax = Arc::new(
        BoundSemanticOp::new(
            Arc::new(
                SemanticOpDescriptor::new(
                    SemanticOpKind::Argmax,
                    vec![logits_view.clone()],
                    vec![argmax_view.clone()],
                )
                .map_err(|error| format!("Argmax descriptor failed: {error}"))?,
            ),
            vec![
                session
                    .bind(&logits_buffer, logits_view, AccessMode::Read)
                    .map_err(|error| format!("Argmax input binding failed: {error}"))?,
            ],
            vec![
                session
                    .bind(&argmax_buffer, argmax_view, AccessMode::Write)
                    .map_err(|error| format!("Argmax output binding failed: {error}"))?,
            ],
        )
        .map_err(|error| format!("Argmax binding failed: {error}"))?,
    );
    let prepared_argmax = session
        .prepare(argmax)
        .map_err(|error| format!("Argmax prepare failed: {error}"))?;
    let mut argmax_submission = session
        .submit(&prepared_argmax, queue)
        .map_err(|error| format!("Argmax submit failed: {error}"))?;
    wait_success(argmax_submission.wait(WAIT_TIMEOUT), "Argmax completion")?;
    let argmax_dispatch = argmax_submission.dispatch().clone();
    if argmax_dispatch.row_count != 1
        || argmax_dispatch.fallback_allowed
        || argmax_dispatch.fallback_used
    {
        return Err("terminal Argmax dispatch violated the one-row contract".to_owned());
    }
    let mut argmax_readback = argmax_submission
        .start_output_readback(0)
        .map_err(|error| format!("Argmax readback failed: {error}"))?;
    wait_success(argmax_readback.wait(WAIT_TIMEOUT), "Argmax readback")?;
    let mut argmax_bytes = [0_u8; 4];
    if argmax_readback
        .read_into(&mut argmax_bytes)
        .map_err(|error| format!("Argmax read failed: {error}"))?
        != 4
    {
        return Err("Argmax readback byte count differed".to_owned());
    }
    let actual_argmax = i32::from_le_bytes(argmax_bytes);
    let expected_argmax = i32::try_from(selected_row).expect("selected row fits i32");
    if actual_argmax != expected_argmax {
        return Err(format!(
            "terminal Argmax selected {actual_argmax}, expected {expected_argmax}"
        ));
    }

    Ok(CaseEvidence {
        source_rows: rows,
        selected_row,
        selected_byte_offset,
        projected_rows: projection_dispatch.row_count,
        argmax_rows: argmax_dispatch.row_count,
        expected_argmax,
        actual_argmax,
        max_abs_error,
        fallback_used: false,
    })
}

fn run(config: &Config) -> Result<Report, String> {
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("invalid execution-session request: {error}"))?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| format!("HIP session open failed: {error}"))?;
    let result = (|| {
        let queue = session
            .create_queue()
            .map_err(|error| format!("queue creation failed: {error}"))?;
        ROWS.iter()
            .copied()
            .map(|rows| run_case(&session, &queue, rows))
            .collect::<Result<Vec<_>, _>>()
    })();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("HIP session shutdown failed: {error}"))?;
    let cases = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("HIP cleanup did not return to zero owned work".to_owned());
    }
    Ok(Report {
        schema_version: "phase24-terminal-row-gpu-v1",
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        fallback_used: false,
        cases,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
    })
}

fn main() -> ExitCode {
    let result = parse_config().and_then(|config| run(&config));
    match result {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("phase24 terminal-row evidence serialization failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("phase24 terminal-row evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
