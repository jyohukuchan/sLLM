//! Focused semantic G1/G2 evidence for single-GPU BF16 embedding gather.

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    AccessMode, Backend, BoundSemanticOp, DType, DispatchEvidence, ExecutionSessionRequest,
    ExecutionState, SemanticOpDescriptor, SemanticOpKind, TensorView,
};
use sllm_hip::HipBackend;

const HIDDEN_SIZES: [usize; 7] = [1, 3, 17, 255, 256, 257, 2560];
const TOKEN_IDS: [i32; 3] = [4, 0, 4];
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);

struct Config {
    device_index: u32,
    target: String,
    weight_shard: PathBuf,
    weight_offset: u64,
}

#[derive(Serialize)]
struct CaseEvidence {
    source: &'static str,
    vocab_size: usize,
    token_count: usize,
    hidden_size: usize,
    kernel_symbol: String,
    device_symbol: String,
    grid_size_x: u32,
    exact_match: bool,
}

struct CaseInput {
    weight_bytes: Vec<u8>,
    vocab: usize,
    hidden: usize,
    ids: Vec<i32>,
    source: &'static str,
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
    kernel_dispatches: usize,
    real_weight_slice_sha256: String,
    negative_token_range_rejected_before_dispatch: bool,
    cases: Vec<CaseEvidence>,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

fn parse_config() -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut weight_shard = None;
    let mut weight_offset = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{argument} requires a value"))?;
        match argument.as_str() {
            "--device-index" => {
                device_index = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
            }
            "--target" if matches!(value.as_str(), "gfx1030" | "gfx1201") => {
                target = Some(value);
            }
            "--weight-shard" => weight_shard = Some(PathBuf::from(value)),
            "--weight-offset" => {
                weight_offset = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_| "--weight-offset must be a u64".to_owned())?,
                );
            }
            "--target" => return Err("--target must be gfx1030 or gfx1201".to_owned()),
            _ => return Err(format!("unexpected argument `{argument}`")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
        weight_shard: weight_shard.ok_or_else(|| "missing --weight-shard".to_owned())?,
        weight_offset: weight_offset.ok_or_else(|| "missing --weight-offset".to_owned())?,
    })
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
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
    tokens: usize,
    hidden: usize,
    target: &str,
) -> Result<(), String> {
    let expected_grid = u32::try_from((tokens * hidden).div_ceil(256))
        .map_err(|_| "embedding grid does not fit u32".to_owned())?;
    if dispatch.abi_version != 1
        || dispatch.info_version != 1
        || dispatch.dispatch_id == 0
        || dispatch.dispatch_count != 1
        || dispatch.kernel_id != 1
        || dispatch.workgroup_size_x != 256
        || dispatch.grid_size_x != expected_grid
        || dispatch.row_count != tokens as u64
        || dispatch.normalized_size != hidden as u64
        || dispatch.backend != 1
        || dispatch.fallback_allowed
        || dispatch.fallback_used
        || dispatch.kernel_symbol != "embedding.gather.bf16_i32.v1"
        || dispatch.device_symbol != "sllm_embedding_gather_bf16_i32_v1"
        || dispatch.target != target
    {
        return Err("embedding dispatch metadata violated the exact contract".to_owned());
    }
    Ok(())
}

fn run_case(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
    case: CaseInput,
    target: &str,
) -> Result<CaseEvidence, String> {
    let CaseInput {
        weight_bytes,
        vocab,
        hidden,
        ids,
        source,
    } = case;
    let expected_weight_bytes = vocab
        .checked_mul(hidden)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| "embedding weight size overflow".to_owned())?;
    if weight_bytes.len() != expected_weight_bytes {
        return Err("embedding weight bytes do not match shape".to_owned());
    }
    let id_bytes = i32_bytes(&ids);
    let output_bytes = ids.len() * hidden * 2;
    let weight_buffer = session
        .allocate(weight_bytes.len() as u64)
        .map_err(|error| format!("weight allocation failed: {error}"))?;
    let id_buffer = session
        .allocate(id_bytes.len() as u64)
        .map_err(|error| format!("token allocation failed: {error}"))?;
    let output_buffer = session
        .allocate(output_bytes as u64)
        .map_err(|error| format!("output allocation failed: {error}"))?;
    for (label, buffer, bytes) in [
        ("weight", &weight_buffer, weight_bytes.as_slice()),
        ("token", &id_buffer, id_bytes.as_slice()),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|e| e.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|error| format!("{label} H2D failed: {error}"))?;
        wait_success(upload.wait(WAIT_TIMEOUT), &format!("{label} H2D"))?;
    }
    let weight_view =
        TensorView::contiguous(DType::Bf16, &[vocab, hidden]).map_err(|error| error.to_string())?;
    let id_view =
        TensorView::contiguous(DType::I32, &[ids.len()]).map_err(|error| error.to_string())?;
    let output_view = TensorView::contiguous(DType::Bf16, &[ids.len(), hidden])
        .map_err(|error| error.to_string())?;
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::Embedding,
            vec![weight_view.clone(), id_view.clone()],
            vec![output_view.clone()],
        )
        .map_err(|error| error.to_string())?,
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&id_buffer, id_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&output_buffer, output_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    let mut submission = session
        .submit(&prepared, queue)
        .map_err(|error| format!("embedding submit failed: {error}"))?;
    validate_dispatch(submission.dispatch(), ids.len(), hidden, target)?;
    wait_success(submission.wait(WAIT_TIMEOUT), "embedding completion")?;
    let dispatch = submission.dispatch().clone();
    let mut readback = submission
        .start_output_readback(0)
        .map_err(|error| error.to_string())?;
    wait_success(readback.wait(WAIT_TIMEOUT), "embedding D2H")?;
    let mut actual = vec![0_u8; output_bytes];
    if readback
        .read_into(&mut actual)
        .map_err(|error| error.to_string())?
        != output_bytes as u64
    {
        return Err("embedding output byte count mismatch".to_owned());
    }
    let mut expected = Vec::with_capacity(output_bytes);
    for &id in &ids {
        let row = usize::try_from(id).map_err(|_| "negative oracle ID".to_owned())?;
        let start = row * hidden * 2;
        expected.extend_from_slice(&weight_bytes[start..start + hidden * 2]);
    }
    if actual != expected {
        return Err(format!(
            "embedding byte oracle mismatch for hidden={hidden}"
        ));
    }
    Ok(CaseEvidence {
        source,
        vocab_size: vocab,
        token_count: ids.len(),
        hidden_size: hidden,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        grid_size_x: dispatch.grid_size_x,
        exact_match: true,
    })
}

fn synthetic_weight(vocab: usize, hidden: usize) -> Vec<u8> {
    (0..vocab * hidden)
        .flat_map(|index| ((index * 37 + 11) as u16).to_le_bytes())
        .collect()
}

fn reject_negative_token_before_dispatch(
    session: &sllm_core::ExecutionSession,
    queue: &sllm_core::ExecutionQueue,
) -> Result<bool, String> {
    let weight_bytes = synthetic_weight(3, 3);
    let id_bytes = i32_bytes(&[-1]);
    let weight_buffer = session.allocate(18).map_err(|error| error.to_string())?;
    let id_buffer = session.allocate(4).map_err(|error| error.to_string())?;
    let output_buffer = session.allocate(6).map_err(|error| error.to_string())?;
    for (buffer, bytes) in [
        (&weight_buffer, weight_bytes.as_slice()),
        (&id_buffer, id_bytes.as_slice()),
    ] {
        let mut upload = session
            .upload(
                queue,
                buffer
                    .range(0, bytes.len() as u64)
                    .map_err(|e| e.to_string())?,
                Arc::<[u8]>::from(bytes),
            )
            .map_err(|error| error.to_string())?;
        wait_success(upload.wait(WAIT_TIMEOUT), "negative-case input H2D")?;
    }
    let weight_view = TensorView::contiguous(DType::Bf16, &[3, 3]).unwrap();
    let id_view = TensorView::contiguous(DType::I32, &[1]).unwrap();
    let output_view = TensorView::contiguous(DType::Bf16, &[1, 3]).unwrap();
    let descriptor = Arc::new(
        SemanticOpDescriptor::new(
            SemanticOpKind::Embedding,
            vec![weight_view.clone(), id_view.clone()],
            vec![output_view.clone()],
        )
        .unwrap(),
    );
    let operation = Arc::new(
        BoundSemanticOp::new(
            descriptor,
            vec![
                session
                    .bind(&weight_buffer, weight_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
                session
                    .bind(&id_buffer, id_view, AccessMode::Read)
                    .map_err(|error| error.to_string())?,
            ],
            vec![
                session
                    .bind(&output_buffer, output_view, AccessMode::Write)
                    .map_err(|error| error.to_string())?,
            ],
        )
        .map_err(|error| error.to_string())?,
    );
    let prepared = session
        .prepare(operation)
        .map_err(|error| error.to_string())?;
    match session.submit(&prepared, queue) {
        Err(sllm_core::ExecutionError::BackendStatus { status: 0x117, .. }) => Ok(true),
        Err(error) => Err(format!("negative token returned the wrong error: {error}")),
        Ok(_) => Err("negative token unexpectedly dispatched".to_owned()),
    }
}

fn run(config: &Config) -> Result<Report, String> {
    let mut shard = File::open(&config.weight_shard)
        .map_err(|error| format!("real weight shard open failed: {error}"))?;
    shard
        .seek(SeekFrom::Start(config.weight_offset))
        .map_err(|error| format!("real weight seek failed: {error}"))?;
    let mut real_weight = vec![0_u8; 3 * 2560 * 2];
    shard
        .read_exact(&mut real_weight)
        .map_err(|error| format!("bounded real weight read failed: {error}"))?;
    let real_weight_slice_sha256 = format!("{:x}", Sha256::digest(&real_weight));

    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(config.device_index, config.target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let result = (|| {
        let queue = session.create_queue().map_err(|error| error.to_string())?;
        let mut cases = Vec::new();
        for hidden in HIDDEN_SIZES {
            cases.push(run_case(
                &session,
                &queue,
                CaseInput {
                    weight_bytes: synthetic_weight(5, hidden),
                    vocab: 5,
                    hidden,
                    ids: TOKEN_IDS.to_vec(),
                    source: "synthetic",
                },
                &config.target,
            )?);
        }
        cases.push(run_case(
            &session,
            &queue,
            CaseInput {
                weight_bytes: real_weight,
                vocab: 3,
                hidden: 2560,
                ids: vec![2, 0, 2],
                source: "locked-real-first-three-rows",
            },
            &config.target,
        )?);
        let rejected = reject_negative_token_before_dispatch(&session, &queue)?;
        Ok::<_, String>((cases, rejected))
    })();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("execution-session shutdown failed: {error}"))?;
    let (cases, negative_token_range_rejected_before_dispatch) = result?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("embedding cleanup did not return to zero".to_owned());
    }
    Ok(Report {
        schema_version: "embedding-g1-report-v1",
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        selected_backend: "hip",
        fallback_allowed: false,
        fallback_used: false,
        cpu_fallback_used: false,
        operations: cases.len(),
        kernel_dispatches: cases.len(),
        real_weight_slice_sha256,
        negative_token_range_rejected_before_dispatch,
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
                eprintln!("embedding report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("embedding evidence failed: {error}");
            ExitCode::from(1)
        }
    }
}
