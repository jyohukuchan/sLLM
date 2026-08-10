//! Focused G1 evidence for verified, chunked model-weight upload.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    Backend, ExecutionSessionRequest, ExecutionState, TensorDType, WeightUploadRequest,
    build_verified_weight_load_plan, read_model_lock, upload_verified_weight,
};
use sllm_hip::HipBackend;

const TENSOR_NAME: &str = "model.language_model.layers.0.linear_attn.in_proj_z.weight";
const TENSOR_BYTES: u64 = 20 * 1024 * 1024;
const LOCK_FINGERPRINT: &str =
    "sha256:32265444b7cdd2a00e4e4e3e6aa8375a05acf6cddfcb9ffc348f54f67a7cd935";
const PLAN_DIGEST: &str = "sha256:0820227fdc4129e5ff100e0aa87db7663d75703c9ba723bc4adc950a3af6ab66";
const SOURCE_FILE: &str = "model.safetensors-00002-of-00002.safetensors";
const SOURCE_FILE_SHA256: &str = "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188";
const SOURCE_RANGE: [u64; 2] = [42_435_872, 63_407_392];
const DESTINATION_OFFSET: u64 = 7;
const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(16);

#[derive(Debug)]
struct Config {
    device_index: u32,
    target: String,
    lock: PathBuf,
    cache: PathBuf,
}

#[derive(Serialize)]
struct Scope {
    selected_backend: &'static str,
    fallback_allowed: bool,
    fallback_used: bool,
    cpu_fallback_used: bool,
    gpu_execution: bool,
    model_cache_used: bool,
    weight_payload_used: bool,
    model_execution: bool,
    semantic_op_used: bool,
    kernel_dispatch_count: u32,
    network_used: bool,
}

#[derive(Serialize)]
struct Counts {
    allocations: usize,
    chunks: usize,
    h2d_transfers: usize,
    d2h_transfers: usize,
}

#[derive(Serialize)]
struct ChunkEvidence {
    order: usize,
    tensor_offset: u64,
    source_offset: u64,
    destination_offset: u64,
    size_bytes: u64,
    h2d_state: &'static str,
    d2h_state: &'static str,
    exact_match: bool,
}

#[derive(Serialize)]
struct Cleanup {
    retryable_cleanup: usize,
    durable_quarantine: usize,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    lock_fingerprint: String,
    plan_digest: String,
    tensor_name: &'static str,
    tensor_dtype: &'static str,
    tensor_size_bytes: u64,
    source_file: String,
    source_file_sha256: String,
    source_range: [u64; 2],
    destination_offset: u64,
    peak_host_staging_bytes: u64,
    scope: Scope,
    counts: Counts,
    chunks: Vec<ChunkEvidence>,
    cleanup: Cleanup,
}

fn parse_config() -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut lock = None;
    let mut cache = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let slot = match argument.as_str() {
            "--device-index" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "--device-index requires a value".to_owned())?;
                if device_index.is_some() {
                    return Err("duplicate --device-index".to_owned());
                }
                device_index = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
                continue;
            }
            "--target" => &mut target,
            "--lock" => &mut lock,
            "--cache" => &mut cache,
            other => return Err(format!("unexpected argument `{other}`")),
        };
        if slot.is_some() {
            return Err(format!("duplicate {argument}"));
        }
        *slot = Some(
            arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))?,
        );
    }
    let target = target.ok_or_else(|| "missing --target".to_owned())?;
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("--target must be gfx1030 or gfx1201".to_owned());
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target,
        lock: PathBuf::from(lock.ok_or_else(|| "missing --lock".to_owned())?),
        cache: PathBuf::from(cache.ok_or_else(|| "missing --cache".to_owned())?),
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
    // Connect first so a host-stub build cannot be mistaken for model evidence.
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let lock =
        read_model_lock(&config.lock).map_err(|error| format!("lock read failed: {error}"))?;
    let cache = lock
        .verify_cache(&config.cache)
        .map_err(|error| format!("cache verification failed: {error}"))?;
    let plan = build_verified_weight_load_plan(&lock, &cache)
        .map_err(|error| format!("weight plan failed: {error}"))?;
    if plan.lock_fingerprint != LOCK_FINGERPRINT || plan.digest_hex() != PLAN_DIGEST {
        return Err("weight plan identity differs from the reviewed evidence contract".to_owned());
    }
    let entry = plan
        .entries
        .iter()
        .find(|entry| entry.tensor_name == TENSOR_NAME)
        .ok_or_else(|| "fixed evidence tensor is absent from the load plan".to_owned())?;
    if entry.dtype != TensorDType::Bf16
        || entry.source_range[1].checked_sub(entry.source_range[0]) != Some(TENSOR_BYTES)
        || entry.chunks.len() != 2
        || entry.chunks[0].byte_length != 16 * 1024 * 1024
        || entry.chunks[1].byte_length != 4 * 1024 * 1024
        || entry.source_file != SOURCE_FILE
        || entry.locked_file_sha256 != SOURCE_FILE_SHA256
        || entry.source_range != SOURCE_RANGE
    {
        return Err(
            "fixed evidence tensor no longer has the reviewed source/chunk plan".to_owned(),
        );
    }
    let source_file = entry.source_file.clone();
    let source_file_sha256 = entry.locked_file_sha256.clone();
    let source_range = entry.source_range;
    let planned_chunks = entry.chunks.clone();

    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("invalid execution-session request: {error}"))?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| format!("execution-session open failed: {error}"))?;
    let operation = (|| {
        let queue = session
            .create_queue()
            .map_err(|error| format!("queue creation failed: {error}"))?;
        let allocation_size = DESTINATION_OFFSET
            .checked_add(TENSOR_BYTES)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "evidence allocation size overflow".to_owned())?;
        let buffer = session
            .allocate(allocation_size)
            .map_err(|error| format!("allocation failed: {error}"))?;
        let destination = buffer
            .range(DESTINATION_OFFSET, TENSOR_BYTES)
            .map_err(|error| format!("destination range failed: {error}"))?;
        let receipt = upload_verified_weight(WeightUploadRequest {
            plan: &plan,
            expected_plan_digest: *plan.digest(),
            cache: &cache,
            tensor_name: TENSOR_NAME,
            expected_dtype: TensorDType::Bf16,
            session: &session,
            queue: &queue,
            destination,
            completion_timeout: WAIT_TIMEOUT,
        })
        .map_err(|error| format!("verified upload failed: {error}"))?;
        if receipt.byte_length != TENSOR_BYTES || receipt.chunks_uploaded != planned_chunks.len() {
            return Err("verified upload receipt differs from the reviewed plan".to_owned());
        }

        let plan_destination = entry
            .destination_start
            .ok_or_else(|| "fixed evidence tensor is not loadable".to_owned())?;
        let mut chunks = Vec::with_capacity(planned_chunks.len());
        for (order, chunk) in planned_chunks.iter().enumerate() {
            let tensor_offset = chunk
                .source_offset
                .checked_sub(source_range[0])
                .ok_or_else(|| "chunk source offset underflow".to_owned())?;
            let destination_relative = chunk
                .destination_offset
                .checked_sub(plan_destination)
                .ok_or_else(|| "chunk destination offset underflow".to_owned())?;
            if tensor_offset != destination_relative {
                return Err("chunk source/destination offsets differ".to_owned());
            }
            let length = usize::try_from(chunk.byte_length)
                .map_err(|_| "chunk length does not fit usize".to_owned())?;
            let expected = cache
                .read_tensor_range(TENSOR_NAME, tensor_offset, length)
                .map_err(|error| format!("verified chunk reread failed: {error}"))?;
            let absolute_destination = DESTINATION_OFFSET
                .checked_add(destination_relative)
                .ok_or_else(|| "readback destination offset overflow".to_owned())?;
            let range = buffer
                .range(absolute_destination, chunk.byte_length)
                .map_err(|error| format!("readback range failed: {error}"))?;
            let mut readback = session
                .readback(&queue, range)
                .map_err(|error| format!("D2H submit failed: {error}"))?;
            wait_success(readback.wait(WAIT_TIMEOUT), "D2H completion")?;
            let mut output = vec![0_u8; length];
            let copied = readback
                .read_into(&mut output)
                .map_err(|error| format!("D2H read failed: {error}"))?;
            if copied != chunk.byte_length || output != expected {
                return Err(format!("chunk {order} byte-exact comparison failed"));
            }
            chunks.push(ChunkEvidence {
                order,
                tensor_offset,
                source_offset: chunk.source_offset,
                destination_offset: absolute_destination,
                size_bytes: chunk.byte_length,
                h2d_state: "success",
                d2h_state: "success",
                exact_match: true,
            });
        }
        Ok((receipt, chunks))
    })();

    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("execution-session shutdown failed: {error}"))?;
    let (receipt, chunks) = operation?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("execution cleanup did not return to zero owned work".to_owned());
    }
    Ok(Report {
        schema_version: "weight-upload-g1-report-v1",
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        lock_fingerprint: plan.lock_fingerprint.clone(),
        plan_digest: plan.digest_hex(),
        tensor_name: TENSOR_NAME,
        tensor_dtype: "BF16",
        tensor_size_bytes: TENSOR_BYTES,
        source_file,
        source_file_sha256,
        source_range,
        destination_offset: DESTINATION_OFFSET,
        peak_host_staging_bytes: receipt.peak_host_staging_bytes,
        scope: Scope {
            selected_backend: "hip",
            fallback_allowed: false,
            fallback_used: false,
            cpu_fallback_used: false,
            gpu_execution: true,
            model_cache_used: true,
            weight_payload_used: true,
            model_execution: false,
            semantic_op_used: false,
            kernel_dispatch_count: 0,
            network_used: false,
        },
        counts: Counts {
            allocations: 1,
            chunks: chunks.len(),
            h2d_transfers: chunks.len(),
            d2h_transfers: chunks.len(),
        },
        chunks,
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
                eprintln!("weight-upload-g1: report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("weight-upload-g1: {error}");
            ExitCode::from(2)
        }
    }
}
