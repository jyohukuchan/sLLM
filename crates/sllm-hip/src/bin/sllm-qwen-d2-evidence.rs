//! D2 evidence for full Qwen weight upload and request-local provisioning.
//!
//! This binary does not execute a forward pass. It verifies the fixed cache,
//! builds the canonical graph/load plan, uploads all required text weights,
//! allocates every graph buffer and request-local state through the public HIP
//! execution owner, then holds those resources so an external controller can
//! observe peak VRAM. Numerical real-weight G2 remains the existing dedicated
//! RMSNorm evidence path; end-to-end model execution remains Stage E G3.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use serde::Serialize;
use sllm_core::{
    Backend, ExecutionSessionRequest, QwenExecutionRequest, QwenGraphTensorBacking,
    WeightClassification, build_qwen35_graph, build_verified_weight_load_plan, read_model_lock,
    reviewed_qwen35_spec,
};
use sllm_hip::HipBackend;

const TOKEN_COUNT: u64 = 3;
const STATE_CAPACITY: u64 = 17;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_HOLD_SECONDS: u64 = 60;
const EXECUTION_GUARD: &str = "SLLM_QWEN_D2_GPU_EXECUTION";

#[derive(Clone, Debug, Eq, PartialEq)]
struct Config {
    device_index: u32,
    target: String,
    lock: PathBuf,
    cache: PathBuf,
    hold_seconds: u64,
}

#[derive(Serialize)]
struct Counts {
    plan_entries: usize,
    required_weights: usize,
    known_unconsumed_weights: usize,
    graph_tensors: usize,
    graph_nodes: usize,
    logical_state_rows: usize,
    layers: usize,
}

#[derive(Serialize)]
struct Allocation {
    required_weight_bytes: u64,
    graph_owned_buffer_bytes: u64,
    logical_state_bytes: u64,
    upload_chunk_limit_bytes: u64,
    token_count: u64,
    state_capacity: u64,
    hold_seconds: u64,
}

#[derive(Serialize)]
struct Scope {
    selected_backend: &'static str,
    fallback_allowed: bool,
    fallback_used: bool,
    cpu_fallback_used: bool,
    verified_full_cache: bool,
    full_text_weight_payload_uploaded: bool,
    model_forward_executed: bool,
    kernel_dispatch_count: u32,
    external_vram_observation_required: bool,
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
    counts: Counts,
    allocation: Allocation,
    scope: Scope,
    cleanup: Cleanup,
}

fn parse_config_from(arguments: impl IntoIterator<Item = String>) -> Result<Config, String> {
    let mut device_index = None;
    let mut target = None;
    let mut lock = None;
    let mut cache = None;
    let mut hold_seconds = None;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{argument} requires a value"))?;
        match argument.as_str() {
            "--device-index" => set_once(
                &mut device_index,
                value
                    .parse::<u32>()
                    .map_err(|_| "--device-index must be a U32".to_owned())?,
                "--device-index",
            )?,
            "--target" => {
                if value != "gfx1030" && value != "gfx1201" {
                    return Err("--target must be gfx1030 or gfx1201".to_owned());
                }
                set_once(&mut target, value, "--target")?;
            }
            "--lock" => set_once(&mut lock, PathBuf::from(value), "--lock")?,
            "--cache" => set_once(&mut cache, PathBuf::from(value), "--cache")?,
            "--hold-seconds" => {
                let parsed = value
                    .parse::<u64>()
                    .map_err(|_| "--hold-seconds must be a U64".to_owned())?;
                if parsed == 0 || parsed > MAX_HOLD_SECONDS {
                    return Err(format!("--hold-seconds must be in [1,{MAX_HOLD_SECONDS}]"));
                }
                set_once(&mut hold_seconds, parsed, "--hold-seconds")?;
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    Ok(Config {
        device_index: device_index.ok_or_else(|| "--device-index is required".to_owned())?,
        target: target.ok_or_else(|| "--target is required".to_owned())?,
        lock: lock.ok_or_else(|| "--lock is required".to_owned())?,
        cache: cache.ok_or_else(|| "--cache is required".to_owned())?,
        hold_seconds: hold_seconds.ok_or_else(|| "--hold-seconds is required".to_owned())?,
    })
}

fn set_once<T>(slot: &mut Option<T>, value: T, name: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("{name} may be supplied only once"));
    }
    Ok(())
}

fn checked_owned_bytes(graph: &sllm_core::QwenGraph) -> Result<u64, String> {
    graph
        .tensor_metadata()
        .iter()
        .filter(|tensor| tensor.backing() == QwenGraphTensorBacking::Owned)
        .try_fold(0_u64, |total, tensor| {
            total
                .checked_add(tensor.view().end_offset())
                .ok_or_else(|| {
                    format!(
                        "owned graph buffer byte count overflowed at {}",
                        tensor.name()
                    )
                })
        })
}

fn run(config: &Config) -> Result<Report, String> {
    if env::var(EXECUTION_GUARD).as_deref() != Ok("1") {
        return Err(format!("{EXECUTION_GUARD}=1 is required"));
    }

    let lock = read_model_lock(&config.lock).map_err(|error| error.to_string())?;
    let spec = reviewed_qwen35_spec(&lock)
        .ok_or_else(|| "model is not a reviewed Qwen3.5 dense identity".to_owned())?;
    let cache = lock
        .verify_cache(&config.cache)
        .map_err(|error| error.to_string())?;
    let plan = build_verified_weight_load_plan(&lock, &cache).map_err(|error| error.to_string())?;
    let graph = build_qwen35_graph(&lock, &plan, TOKEN_COUNT, STATE_CAPACITY)
        .map_err(|error| error.to_string())?;

    let required_weights = plan
        .entries
        .iter()
        .filter(|entry| entry.classification == WeightClassification::Required)
        .count();
    let known_unconsumed_weights = plan
        .entries
        .iter()
        .filter(|entry| entry.classification == WeightClassification::KnownUnconsumed)
        .count();
    if plan.entries.len() != spec.indexed_tensor_count as usize
        || required_weights + known_unconsumed_weights != plan.entries.len()
        || graph.layer_types().len() != spec.layer_count as usize
        || graph.states().len() != spec.layer_count as usize * 2
    {
        return Err("canonical Qwen coverage counts differ".to_owned());
    }

    let counts = Counts {
        plan_entries: plan.entries.len(),
        required_weights,
        known_unconsumed_weights,
        graph_tensors: graph.tensor_metadata().len(),
        graph_nodes: graph.nodes().len(),
        logical_state_rows: graph.states().len(),
        layers: graph.layer_types().len(),
    };
    let allocation = Allocation {
        required_weight_bytes: plan.total_destination_bytes,
        graph_owned_buffer_bytes: checked_owned_bytes(&graph)?,
        logical_state_bytes: graph.total_state_bytes(),
        upload_chunk_limit_bytes: plan.chunk_size,
        token_count: TOKEN_COUNT,
        state_capacity: STATE_CAPACITY,
        hold_seconds: config.hold_seconds,
    };
    let lock_fingerprint = lock.fingerprint().to_owned();
    let plan_digest = plan.digest_hex();

    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(request)
        .map_err(|error| error.to_string())?;
    let owner = QwenExecutionRequest::new(
        Arc::clone(&session),
        graph,
        plan,
        Arc::new(cache),
        COMPLETION_TIMEOUT,
    )
    .map_err(|error| error.to_string())?;

    eprintln!(
        "SLLM_QWEN_D2_READY target={} hold_seconds={}",
        config.target, config.hold_seconds
    );
    thread::sleep(Duration::from_secs(config.hold_seconds));
    drop(owner);
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| error.to_string())?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("HIP session cleanup was not empty".to_owned());
    }

    Ok(Report {
        schema_version: "qwen-d2-provisioning-v1",
        state: "PASS",
        target: config.target.clone(),
        device_index: config.device_index,
        lock_fingerprint,
        plan_digest,
        counts,
        allocation,
        scope: Scope {
            selected_backend: "hip",
            fallback_allowed: false,
            fallback_used: false,
            cpu_fallback_used: false,
            verified_full_cache: true,
            full_text_weight_payload_uploaded: true,
            model_forward_executed: false,
            kernel_dispatch_count: 0,
            external_vram_observation_required: true,
        },
        cleanup: Cleanup {
            retryable_cleanup: cleanup.retryable_cleanup,
            durable_quarantine: cleanup.durable_quarantine,
        },
    })
}

fn main() -> ExitCode {
    let result = parse_config_from(env::args().skip(1)).and_then(|config| run(&config));
    match result {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(output) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("qwen-d2: report serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("qwen-d2: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_arguments() -> Vec<String> {
        [
            "--device-index",
            "0",
            "--target",
            "gfx1030",
            "--lock",
            "lock.json",
            "--cache",
            "cache",
            "--hold-seconds",
            "3",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn parses_exact_non_aligned_provisioning_arguments() {
        assert_eq!(
            parse_config_from(valid_arguments()).unwrap(),
            Config {
                device_index: 0,
                target: "gfx1030".to_owned(),
                lock: PathBuf::from("lock.json"),
                cache: PathBuf::from("cache"),
                hold_seconds: 3,
            }
        );
    }

    #[test]
    fn rejects_unknown_duplicate_missing_and_hold_boundaries() {
        let mut duplicate = valid_arguments();
        duplicate.extend(["--target".to_owned(), "gfx1201".to_owned()]);
        assert!(parse_config_from(duplicate).is_err());

        let mut unknown = valid_arguments();
        unknown.extend(["--other".to_owned(), "1".to_owned()]);
        assert!(parse_config_from(unknown).is_err());

        assert!(parse_config_from(Vec::<String>::new()).is_err());
        for value in ["0", "61"] {
            let mut arguments = valid_arguments();
            *arguments.last_mut().unwrap() = value.to_owned();
            assert!(parse_config_from(arguments).is_err());
        }
    }
}
