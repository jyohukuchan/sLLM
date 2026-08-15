//! Exact-target full-weight Gemma 4 graph bring-up evidence.
//!
//! This runner verifies the immutable cache, uploads the complete text model,
//! and executes one full 48-layer transition through the shared owned
//! execution path. It does not label the resulting token as a correctness
//! oracle; reference-logit comparison is a separate closeout requirement.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use sllm_core::{
    Backend, ExecutionSessionRequest, Gemma4ExecutionOptions, Gemma4RequestState,
    build_gemma4_execution_layout, build_gemma4_graph, build_verified_gemma4_weight_load_plan,
    parse_gemma4_model_lock, provision_gemma4_execution_buffers,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const REFERENCE_INPUT_TOKEN: i32 = 2;
const REFERENCE_GENERATED_TOKENS: [i32; 8] = [258_882; 8];

struct Config {
    lock: PathBuf,
    cache: PathBuf,
    device_index: u32,
    target: String,
    token_id: i32,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    model: &'static str,
    resolved_revision: String,
    lock_fingerprint: String,
    target: String,
    device_index: u32,
    input_token_id: i32,
    generated_token_ids: Vec<i32>,
    transitions: usize,
    graph_nodes: usize,
    model_weight_bytes: u64,
    workspace_bytes: u64,
    request_state_bytes: u64,
    available_memory_bytes: Option<u64>,
    peak_accounted_bytes: u64,
    model_resident_peak_bytes: u64,
    workspace_peak_bytes: u64,
    request_state_peak_bytes: u64,
    upload_milliseconds: u128,
    prefill_milliseconds: u128,
    decode_milliseconds: Vec<u128>,
    committed_length: u64,
    state_generation: u64,
    backend: u32,
    submission_count: u64,
    kernel_dispatch_count: u64,
    segment_count: u64,
    boundary_count: u64,
    fallback_used: bool,
    reference_oracle_verified: bool,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

fn parse_config() -> Result<Config, String> {
    let mut lock = None;
    let mut cache = None;
    let mut device_index = None;
    let mut target = None;
    let mut token_id = 2_i32;
    let mut token_id_seen = false;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{argument} requires a value"))?;
        match argument.as_str() {
            "--lock" if lock.is_none() => lock = Some(PathBuf::from(value)),
            "--cache" if cache.is_none() => cache = Some(PathBuf::from(value)),
            "--device-index" if device_index.is_none() => {
                device_index = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| "--device-index must be a u32".to_owned())?,
                );
            }
            "--target" if target.is_none() && matches!(value.as_str(), "gfx1030" | "gfx1201") => {
                target = Some(value);
            }
            "--target" => return Err("--target must be exactly gfx1030 or gfx1201".to_owned()),
            "--token-id" if !token_id_seen => {
                token_id = value
                    .parse::<i32>()
                    .map_err(|_| "--token-id must be an i32".to_owned())?;
                token_id_seen = true;
            }
            _ => return Err(format!("duplicate or unknown argument: {argument}")),
        }
    }
    if token_id != REFERENCE_INPUT_TOKEN {
        return Err("--token-id must match the fixed CPU reference input 2".to_owned());
    }
    Ok(Config {
        lock: lock.ok_or_else(|| "missing --lock".to_owned())?,
        cache: cache.ok_or_else(|| "missing --cache".to_owned())?,
        device_index: device_index.ok_or_else(|| "missing --device-index".to_owned())?,
        target: target.ok_or_else(|| "missing --target".to_owned())?,
        token_id,
    })
}

fn run(config: Config) -> Result<Report, String> {
    let lock_bytes =
        std::fs::read(&config.lock).map_err(|error| format!("cannot read Gemma lock: {error}"))?;
    let lock = parse_gemma4_model_lock(&lock_bytes)
        .map_err(|error| format!("invalid Gemma lock: {error}"))?;
    let cache = lock
        .verify_cache(&config.cache)
        .map_err(|error| format!("Gemma cache verification failed: {error}"))?;
    let plan = build_verified_gemma4_weight_load_plan(&lock, &cache)
        .map_err(|error| format!("Gemma weight plan failed: {error}"))?;
    let graph = build_gemma4_graph(&lock, &plan, 1, 0, 17)
        .map_err(|error| format!("Gemma graph failed: {error}"))?;
    let layout = build_gemma4_execution_layout(&graph, &plan)
        .map_err(|error| format!("Gemma execution layout failed: {error}"))?;

    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let request = ExecutionSessionRequest::new(config.device_index, config.target.clone())
        .map_err(|error| format!("invalid execution request: {error}"))?;
    let session = Arc::new(
        backend
            .open_execution_session(request)
            .map_err(|error| format!("cannot open HIP execution session: {error}"))?,
    );
    let available_memory_bytes = session
        .available_memory_bytes()
        .map_err(|error| format!("cannot query available memory: {error}"))?;
    let required = layout
        .model_weight_bytes()
        .checked_add(layout.workspace_bytes())
        .and_then(|bytes| bytes.checked_add(layout.request_state_bytes()))
        .ok_or_else(|| "Gemma required memory overflowed".to_owned())?;
    if available_memory_bytes.is_some_and(|available| required > available) {
        return Err(format!(
            "Gemma exact layout requires {required} bytes but only {} are available",
            available_memory_bytes.unwrap_or_default()
        ));
    }
    let queue = session
        .create_queue()
        .map_err(|error| format!("queue creation failed: {error}"))?;
    let buffers = provision_gemma4_execution_buffers(Arc::clone(&session), &layout)
        .map_err(|error| format!("Gemma buffer provisioning failed: {error}"))?;

    let upload_started = Instant::now();
    buffers
        .upload_immutable(&layout, &plan, &cache, &queue, COMPLETION_TIMEOUT)
        .map_err(|error| format!("Gemma immutable upload failed: {error}"))?;
    buffers
        .upload_transition_inputs(&layout, &queue, &[config.token_id], COMPLETION_TIMEOUT)
        .map_err(|error| format!("Gemma transition upload failed: {error}"))?;
    let upload_milliseconds = upload_started.elapsed().as_millis();

    let state = Gemma4RequestState::new(17)
        .map_err(|error| format!("Gemma request state failed: {error}"))?;
    let prefill_started = Instant::now();
    let prefill = buffers
        .execute_transition(
            &graph,
            &layout,
            &queue,
            &state,
            Gemma4ExecutionOptions {
                binding_generation: 1,
                completion_timeout: COMPLETION_TIMEOUT,
                expected_backend: 1,
            },
        )
        .map_err(|error| format!("Gemma full transition failed: {error}"))?;
    let prefill_milliseconds = prefill_started.elapsed().as_millis();
    if prefill.audit().fallback_used()
        || prefill.audit().backend() != 1
        || prefill.audit().target() != config.target
        || prefill.state().poisoned
        || prefill.state().committed_length != 1
    {
        return Err("Gemma full prefill audit is not exact/no-fallback".to_owned());
    }
    let mut current_token = *prefill
        .token_ids()
        .last()
        .ok_or_else(|| "Gemma prefill returned no token".to_owned())?;
    let mut generated_token_ids = vec![current_token];
    let mut submission_count = prefill.audit().submission_count();
    let mut kernel_dispatch_count = prefill.audit().kernel_dispatch_count();
    let mut segment_count = prefill.audit().segment_count();
    let mut boundary_count = prefill.audit().boundary_count();
    let mut fallback_used = prefill.audit().fallback_used();
    let backend = prefill.audit().backend();
    drop(prefill);

    let decode_graph = build_gemma4_graph(&lock, &plan, 1, 1, 17)
        .map_err(|error| format!("Gemma decode graph failed: {error}"))?;
    let decode_layout = build_gemma4_execution_layout(&decode_graph, &plan)
        .map_err(|error| format!("Gemma decode layout failed: {error}"))?;
    let decode_buffers = buffers
        .rebind_transition(&layout, &decode_layout)
        .map_err(|error| format!("Gemma decode rebind failed: {error}"))?;
    drop(buffers);
    decode_buffers
        .upload_transition_inputs(&decode_layout, &queue, &[current_token], COMPLETION_TIMEOUT)
        .map_err(|error| format!("Gemma decode input upload failed: {error}"))?;
    let decode_started = Instant::now();
    let decode = decode_buffers
        .execute_transition(
            &decode_graph,
            &decode_layout,
            &queue,
            &state,
            Gemma4ExecutionOptions {
                binding_generation: 2,
                completion_timeout: COMPLETION_TIMEOUT,
                expected_backend: 1,
            },
        )
        .map_err(|error| format!("Gemma full decode failed: {error}"))?;
    let mut decode_milliseconds = vec![decode_started.elapsed().as_millis()];
    if decode.audit().fallback_used()
        || decode.audit().backend() != 1
        || decode.audit().target() != config.target
        || decode.state().poisoned
        || decode.state().committed_length != 2
        || decode.state().state_generation != 2
    {
        return Err("Gemma full decode audit is not exact/no-fallback".to_owned());
    }
    current_token = *decode
        .token_ids()
        .last()
        .ok_or_else(|| "Gemma decode returned no token".to_owned())?;
    generated_token_ids.push(current_token);
    submission_count = submission_count
        .checked_add(decode.audit().submission_count())
        .ok_or_else(|| "submission count overflowed".to_owned())?;
    kernel_dispatch_count = kernel_dispatch_count
        .checked_add(decode.audit().kernel_dispatch_count())
        .ok_or_else(|| "kernel dispatch count overflowed".to_owned())?;
    segment_count = segment_count
        .checked_add(decode.audit().segment_count())
        .ok_or_else(|| "segment count overflowed".to_owned())?;
    boundary_count = boundary_count
        .checked_add(decode.audit().boundary_count())
        .ok_or_else(|| "boundary count overflowed".to_owned())?;
    fallback_used |= decode.audit().fallback_used();
    drop(decode);

    let mut current_layout = decode_layout;
    let mut current_buffers = decode_buffers;
    for start_position in 2_u64..8 {
        let next_graph = build_gemma4_graph(&lock, &plan, 1, start_position, 17)
            .map_err(|error| format!("Gemma decode graph failed: {error}"))?;
        let next_layout = build_gemma4_execution_layout(&next_graph, &plan)
            .map_err(|error| format!("Gemma decode layout failed: {error}"))?;
        let next_buffers = current_buffers
            .rebind_transition(&current_layout, &next_layout)
            .map_err(|error| format!("Gemma decode rebind failed: {error}"))?;
        drop(current_buffers);
        next_buffers
            .upload_transition_inputs(&next_layout, &queue, &[current_token], COMPLETION_TIMEOUT)
            .map_err(|error| format!("Gemma decode input upload failed: {error}"))?;
        let started = Instant::now();
        let output = next_buffers
            .execute_transition(
                &next_graph,
                &next_layout,
                &queue,
                &state,
                Gemma4ExecutionOptions {
                    binding_generation: start_position + 1,
                    completion_timeout: COMPLETION_TIMEOUT,
                    expected_backend: 1,
                },
            )
            .map_err(|error| format!("Gemma full decode failed: {error}"))?;
        decode_milliseconds.push(started.elapsed().as_millis());
        if output.audit().fallback_used()
            || output.audit().backend() != 1
            || output.audit().target() != config.target
            || output.state().poisoned
            || output.state().committed_length != start_position + 1
            || output.state().state_generation != start_position + 1
        {
            return Err("Gemma repeated decode audit is not exact/no-fallback".to_owned());
        }
        current_token = *output
            .token_ids()
            .last()
            .ok_or_else(|| "Gemma repeated decode returned no token".to_owned())?;
        generated_token_ids.push(current_token);
        submission_count = submission_count
            .checked_add(output.audit().submission_count())
            .ok_or_else(|| "submission count overflowed".to_owned())?;
        kernel_dispatch_count = kernel_dispatch_count
            .checked_add(output.audit().kernel_dispatch_count())
            .ok_or_else(|| "kernel dispatch count overflowed".to_owned())?;
        segment_count = segment_count
            .checked_add(output.audit().segment_count())
            .ok_or_else(|| "segment count overflowed".to_owned())?;
        boundary_count = boundary_count
            .checked_add(output.audit().boundary_count())
            .ok_or_else(|| "boundary count overflowed".to_owned())?;
        fallback_used |= output.audit().fallback_used();
        drop(output);
        current_layout = next_layout;
        current_buffers = next_buffers;
    }
    if generated_token_ids != REFERENCE_GENERATED_TOKENS {
        return Err(format!(
            "Gemma generated tokens differ from the independent CPU reference: {generated_token_ids:?}"
        ));
    }
    let snapshot = state
        .snapshot()
        .map_err(|error| format!("Gemma final state snapshot failed: {error}"))?;
    let committed_length = snapshot.committed_length;
    let state_generation = snapshot.state_generation;
    let memory = session.memory_snapshot();
    drop(state);
    drop(current_buffers);
    drop(queue);
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("session cleanup failed: {error}"))?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("session cleanup retained resources".to_owned());
    }

    Ok(Report {
        schema_version: "gemma4-full-generation-evidence-v1",
        state: "PASS",
        model: "google/gemma-4-12B",
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        target: config.target,
        device_index: config.device_index,
        input_token_id: config.token_id,
        generated_token_ids,
        transitions: 8,
        graph_nodes: graph.nodes().len(),
        model_weight_bytes: layout.model_weight_bytes(),
        workspace_bytes: layout.workspace_bytes(),
        request_state_bytes: layout.request_state_bytes(),
        available_memory_bytes,
        peak_accounted_bytes: memory.high_water_bytes(),
        model_resident_peak_bytes: memory.model_resident().high_water_bytes(),
        workspace_peak_bytes: memory.workspace().high_water_bytes(),
        request_state_peak_bytes: memory.request_state().high_water_bytes(),
        upload_milliseconds,
        prefill_milliseconds,
        decode_milliseconds,
        committed_length,
        state_generation,
        backend,
        submission_count,
        kernel_dispatch_count,
        segment_count,
        boundary_count,
        fallback_used,
        reference_oracle_verified: true,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
    })
}

fn main() -> ExitCode {
    match parse_config().and_then(run) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("cannot serialize evidence: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Gemma 4 full evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
