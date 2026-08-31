//! Phase 55 actual full-resident Gemma 4 MoE generation smoke.
//!
//! These tests are deliberately ignored and fail closed unless the matching
//! target gate is present. They verify the fixed source artifact (or a strict
//! derived GGUF), upload the canonical 17.6 GB resident plan, execute a
//! 17-token prefill followed by 17 committed decode transitions on the same
//! 30 opaque KV states, rewind/replay the last transition, and prove cleanup.

use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, GEMMA4_MOE_TEXT_RESIDENT_BYTES, Gemma4MoeExecutionLayout,
    Gemma4MoeExecutionOutput, Gemma4MoeGraph, Gemma4MoeResidentModel, Gemma4MoeWeightSource,
    VerifiedGemma4Moe, VerifiedGgufGemma4Moe, WeightLoadPlan, build_gemma4_moe_execution_layout,
    build_gemma4_moe_gguf_graph, build_gemma4_moe_graph,
    build_gemma4_moe_resident_weight_load_plan, read_derived_gguf_lock, verify_derived_gguf,
    verify_gemma4_moe_artifact, verify_gguf_gemma4_moe,
};
use sllm_hip::HipBackend;

const FIXED_SOURCE_CACHE: &str = "/home/homelab1/.cache/huggingface/hub/\
models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/\
a19cfe00be84568a6867111c9a68c9c44fdcffe6";
const PREFILL_TOKEN_COUNT: u64 = 17;
const DECODE_TOKEN_COUNT: u64 = 17;
const FINAL_COMMITTED_LENGTH: u64 = PREFILL_TOKEN_COUNT + DECODE_TOKEN_COUNT;
// The model's retained sliding window is 1,024 even though this smoke only
// commits 34 tokens. Static-FP8 sliding state rejects a smaller capacity.
const STATE_CAPACITY: u64 = 1_024;
const LAYER_COUNT: usize = 30;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(300);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);

// Fixed, in-vocabulary, deliberately non-power-of-two prompt coverage. The
// terminal boundary token also covers the highest valid vocabulary ID.
const PREFILL_TOKENS: [i32; PREFILL_TOKEN_COUNT as usize] = [
    2, 106, 1_645, 108, 9_259, 236_776, 563, 107, 17, 23, 42, 255, 256, 257, 4_097, 65_537, 262_143,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TargetContract {
    target: &'static str,
    gate_env: &'static str,
    device_env: &'static str,
}

const GFX1201: TargetContract = TargetContract {
    target: "gfx1201",
    gate_env: "SLLM_PHASE55_GEMMA4_MOE_GFX1201",
    device_env: "SLLM_PHASE55_GEMMA4_MOE_GFX1201_DEVICE",
};

const GFX1030: TargetContract = TargetContract {
    target: "gfx1030",
    gate_env: "SLLM_PHASE55_GEMMA4_MOE_GFX1030",
    device_env: "SLLM_PHASE55_GEMMA4_MOE_GFX1030_DEVICE",
};

#[derive(Debug)]
struct SmokeReport {
    source_identity: String,
    target: &'static str,
    device_index: u32,
    resident_bytes: u64,
    peak_accounted_bytes: u64,
    available_memory_bytes: u64,
    upload_milliseconds: u128,
    prefill_milliseconds: u128,
    decode_milliseconds: u128,
    prefill_tokens: u64,
    committed_decode_tokens: u64,
    output_tokens_checked: usize,
    output_token_sha256: String,
    layer_count: usize,
    state_capacity: u64,
    submission_count: u64,
    kernel_dispatch_count: u64,
    sparse_moe_submission_count: u64,
    sparse_moe_active_pair_count: u64,
    fallback_used: bool,
    nonfinite_free: bool,
    cancel_recovery_verified: bool,
    final_committed_length: u64,
    cleanup_retryable: usize,
    cleanup_durable: usize,
}

impl SmokeReport {
    fn assert_pass(&self, contract: TargetContract, device_index: u32) {
        assert!(!self.source_identity.is_empty());
        assert_eq!(self.target, contract.target);
        assert_eq!(self.device_index, device_index);
        assert_eq!(self.resident_bytes, GEMMA4_MOE_TEXT_RESIDENT_BYTES);
        assert!(self.peak_accounted_bytes >= self.resident_bytes);
        assert!(self.available_memory_bytes >= self.resident_bytes);
        assert!(self.upload_milliseconds > 0);
        assert!(self.prefill_milliseconds > 0);
        assert!(self.decode_milliseconds > 0);
        assert_eq!(self.prefill_tokens, PREFILL_TOKEN_COUNT);
        assert_eq!(self.committed_decode_tokens, DECODE_TOKEN_COUNT);
        assert_eq!(
            self.output_tokens_checked,
            PREFILL_TOKEN_COUNT as usize + DECODE_TOKEN_COUNT as usize + 1
        );
        assert_eq!(self.output_token_sha256.len(), 64);
        assert!(self.submission_count > 0);
        assert!(self.kernel_dispatch_count > 0);
        assert_eq!(self.layer_count, LAYER_COUNT);
        assert_eq!(self.state_capacity, STATE_CAPACITY);
        assert!(self.sparse_moe_submission_count > 0);
        assert!(self.sparse_moe_active_pair_count > 0);
        assert!(!self.fallback_used);
        assert!(self.nonfinite_free);
        assert!(self.cancel_recovery_verified);
        assert_eq!(self.final_committed_length, FINAL_COMMITTED_LENGTH);
        assert_eq!(self.cleanup_retryable, 0);
        assert_eq!(self.cleanup_durable, 0);
    }
}

#[derive(Default)]
struct DispatchTotals {
    output_tokens_checked: usize,
    output_token_ids: Vec<i32>,
    submission_count: u64,
    kernel_dispatch_count: u64,
    sparse_moe_submission_count: u64,
    sparse_moe_active_pair_count: u64,
}

impl DispatchTotals {
    fn record(
        &mut self,
        output: &Gemma4MoeExecutionOutput,
        expected_rows: usize,
        vocab_size: i32,
        target: &str,
    ) -> Result<(), String> {
        if output.token_ids().len() != expected_rows {
            return Err(format!(
                "terminal output row count differs: expected {expected_rows}, got {}",
                output.token_ids().len()
            ));
        }
        // The native Argmax contract returns -1 for every non-finite class
        // (NaN and +/-Inf). Therefore an in-vocabulary ID proves the terminal
        // row contained no non-finite value and also proves vocabulary range.
        if let Some(token) = output
            .token_ids()
            .iter()
            .find(|token| **token < 0 || **token >= vocab_size)
        {
            return Err(format!(
                "terminal token {token} is outside [0,{vocab_size}); non-finite Argmax sentinel or vocabulary violation"
            ));
        }
        let audit = output.audit();
        if audit.backend() != 1
            || audit.target() != target
            || audit.fallback_used()
            || audit.submission_count() == 0
            || audit.kernel_dispatch_count() == 0
            || audit.segment_count() == 0
            || audit.boundary_count() == 0
            || audit.sparse_moe_submission_count() == 0
            || audit.sparse_moe_active_pair_count() == 0
        {
            return Err(format!(
                "dispatch audit is not exact HIP/no-fallback: {audit:?}"
            ));
        }
        self.output_tokens_checked = self
            .output_tokens_checked
            .checked_add(output.token_ids().len())
            .ok_or_else(|| "output-token audit overflowed".to_owned())?;
        self.output_token_ids.extend_from_slice(output.token_ids());
        self.submission_count = self
            .submission_count
            .checked_add(audit.submission_count())
            .ok_or_else(|| "submission audit overflowed".to_owned())?;
        self.kernel_dispatch_count = self
            .kernel_dispatch_count
            .checked_add(audit.kernel_dispatch_count())
            .ok_or_else(|| "kernel-dispatch audit overflowed".to_owned())?;
        self.sparse_moe_submission_count = self
            .sparse_moe_submission_count
            .checked_add(audit.sparse_moe_submission_count())
            .ok_or_else(|| "sparse-MoE submission audit overflowed".to_owned())?;
        self.sparse_moe_active_pair_count = self
            .sparse_moe_active_pair_count
            .checked_add(audit.sparse_moe_active_pair_count())
            .ok_or_else(|| "sparse-MoE active-pair audit overflowed".to_owned())?;
        Ok(())
    }
}

enum ActualSource {
    Safetensors(VerifiedGemma4Moe),
    Gguf(VerifiedGgufGemma4Moe),
}

fn env_path(name: &str) -> Result<Option<PathBuf>, String> {
    env::var_os(name)
        .map(PathBuf::from)
        .map(|path| {
            if path.as_os_str().is_empty() {
                Err(format!("{name} must not be empty"))
            } else {
                Ok(path)
            }
        })
        .transpose()
}

fn require_gate(contract: TargetContract) -> Result<u32, String> {
    match env::var(contract.gate_env).as_deref() {
        Ok("1") => {}
        _ => {
            return Err(format!(
                "{}=1 is required for the ignored {} actual-GPU smoke",
                contract.gate_env, contract.target
            ));
        }
    }
    let device = env::var(contract.device_env)
        .map_err(|_| format!("{} is required", contract.device_env))?;
    device
        .parse::<u32>()
        .map_err(|_| format!("{} must be a u32 HIP device index", contract.device_env))
}

fn load_source() -> Result<ActualSource, String> {
    let source_dir = env_path("SLLM_PHASE55_GEMMA4_MOE_SOURCE_DIR")?;
    let gguf = env_path("SLLM_PHASE55_GEMMA4_MOE_GGUF")?;
    let gguf_lock = env_path("SLLM_PHASE55_GEMMA4_MOE_GGUF_LOCK")?;
    match (source_dir, gguf, gguf_lock) {
        (Some(_), Some(_), Some(_)) => Err(
            "source-dir and derived-GGUF lanes are mutually exclusive for exact identity"
                .to_owned(),
        ),
        (Some(_), Some(_), None) | (Some(_), None, Some(_)) => Err(
            "partial derived-GGUF variables cannot be combined with a source directory".to_owned(),
        ),
        (None, Some(path), Some(lock_path)) => {
            let lock = read_derived_gguf_lock(&lock_path)
                .map_err(|error| format!("derived GGUF lock verification failed: {error}"))?;
            let derived = verify_derived_gguf(lock, &path)
                .map_err(|error| format!("derived GGUF identity verification failed: {error}"))?;
            verify_gguf_gemma4_moe(derived)
                .map(ActualSource::Gguf)
                .map_err(|error| format!("canonical gemma4moe GGUF verification failed: {error}"))
        }
        (_, Some(_), None) | (_, None, Some(_)) => {
            Err("both SLLM_PHASE55_GEMMA4_MOE_GGUF and _GGUF_LOCK are required".to_owned())
        }
        (source_dir, None, None) => {
            let root = source_dir.unwrap_or_else(|| PathBuf::from(FIXED_SOURCE_CACHE));
            verify_source_dir(&root).map(ActualSource::Safetensors)
        }
    }
}

fn verify_source_dir(root: &Path) -> Result<VerifiedGemma4Moe, String> {
    verify_gemma4_moe_artifact(root).map_err(|error| {
        format!(
            "fixed Gemma 4 MoE source verification failed at {}: {error}",
            root.display()
        )
    })
}

fn assert_layer_lengths(
    request: &sllm_core::Gemma4MoeExecutionRequest,
    expected: u64,
) -> Result<(), String> {
    let state = request.state();
    if state.layers().len() != LAYER_COUNT || state.expected_length() < expected {
        return Err(format!(
            "request-state topology differs: layers={}, expected_length={}, required={expected}",
            state.layers().len(),
            state.expected_length()
        ));
    }
    if let Some((index, layer)) = state
        .layers()
        .iter()
        .enumerate()
        .find(|(_, layer)| layer.committed_length() != expected)
    {
        return Err(format!(
            "layer {index} committed length differs: expected {expected}, got {}",
            layer.committed_length()
        ));
    }
    Ok(())
}

fn execute_resident<S: Gemma4MoeWeightSource + 'static>(
    contract: TargetContract,
    device_index: u32,
    source_identity: String,
    source: Arc<S>,
    graph: Gemma4MoeGraph,
    plan: WeightLoadPlan,
    layout: Gemma4MoeExecutionLayout,
) -> Result<SmokeReport, String> {
    if graph.token_count() != PREFILL_TOKEN_COUNT
        || graph.start_position() != 0
        || graph.expected_length() != PREFILL_TOKEN_COUNT
        || graph.state_capacity() != STATE_CAPACITY
        || source.config().layer_count as usize != LAYER_COUNT
        || u64::from(source.config().sliding_window) > STATE_CAPACITY
    {
        return Err("fixed prefill graph/source topology differs".to_owned());
    }
    if plan.total_destination_bytes != GEMMA4_MOE_TEXT_RESIDENT_BYTES
        || layout.resident_weight_bytes() != GEMMA4_MOE_TEXT_RESIDENT_BYTES
        || layout.token_count() != PREFILL_TOKEN_COUNT
    {
        return Err("canonical resident plan/layout accounting differs".to_owned());
    }

    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, contract.target)
                .map_err(|error| format!("exact session request is invalid: {error}"))?,
        )
        .map_err(|error| format!("exact {} session failed: {error}", contract.target))?;
    let operation = (|| -> Result<_, String> {
        let available = session
            .available_memory_bytes()
            .map_err(|error| format!("available-memory query failed: {error}"))?
            .ok_or_else(|| "HIP session did not report available memory".to_owned())?;
        let required_before_native_state = plan
            .total_destination_bytes
            .checked_add(layout.workspace_bytes())
            .ok_or_else(|| "resident/workspace memory requirement overflowed".to_owned())?;
        if required_before_native_state > available {
            // This is an actual test failure, including on the secondary
            // gfx1030; insufficient capacity and OOM are never PASS/skip.
            return Err(format!(
                "{} has {available} available bytes but resident+workspace alone require {required_before_native_state}",
                contract.target
            ));
        }
        let upload_started = Instant::now();
        let resident = Gemma4MoeResidentModel::provision(
            Arc::clone(&session),
            Arc::clone(&source),
            plan.clone(),
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("full-resident provisioning failed: {error}"))?;
        let upload_milliseconds = upload_started.elapsed().as_millis();
        let resident_audit = resident.audit();
        if resident_audit.resident_allocations() != plan.entries.len()
            || resident_audit.direct_weight_allocations() + LAYER_COUNT != plan.entries.len()
            || resident_audit.expert_blob_allocations() != LAYER_COUNT
            || resident_audit.individual_expert_allocations() != 0
            || resident_audit.resident_bytes() != GEMMA4_MOE_TEXT_RESIDENT_BYTES
        {
            return Err(format!(
                "full-resident allocation audit differs: {resident_audit:?}"
            ));
        }

        let mut request = resident
            .new_request(graph)
            .map_err(|error| format!("request provisioning failed: {error}"))?;
        request
            .ensure_dispatchable()
            .map_err(|error| format!("prefill request is not dispatchable: {error}"))?;
        if request.prepared_audit().fallback_used() || request.is_poisoned() {
            return Err("prepared request is fallback-enabled or poisoned".to_owned());
        }
        assert_layer_lengths(&request, 0)?;

        let vocab_size = i32::try_from(source.config().vocab_size)
            .map_err(|_| "vocabulary does not fit i32".to_owned())?;
        let mut totals = DispatchTotals::default();
        let prefill_started = Instant::now();
        let prefill = request
            .execute(&PREFILL_TOKENS)
            .map_err(|error| format!("17-token prefill failed: {error}"))?;
        let prefill_milliseconds = prefill_started.elapsed().as_millis();
        totals.record(
            &prefill,
            PREFILL_TOKEN_COUNT as usize,
            vocab_size,
            contract.target,
        )?;
        assert_layer_lengths(&request, PREFILL_TOKEN_COUNT)?;
        if !request.transition_committed() || request.is_poisoned() {
            return Err("prefill did not commit cleanly".to_owned());
        }
        let mut current_token = *prefill
            .token_ids()
            .last()
            .ok_or_else(|| "prefill terminal output is empty".to_owned())?;
        drop(prefill);

        let decode_started = Instant::now();
        let mut final_decode_input = None;
        let mut final_decode_output = None;
        for decode_index in 0..DECODE_TOKEN_COUNT {
            let decode_input = current_token;
            let output = request
                .execute_next(&[decode_input])
                .map_err(|error| format!("decode {decode_index} failed: {error}"))?;
            totals.record(&output, 1, vocab_size, contract.target)?;
            let expected_length = PREFILL_TOKEN_COUNT + decode_index + 1;
            assert_layer_lengths(&request, expected_length)?;
            current_token = output.token_ids()[0];
            if decode_index + 1 == DECODE_TOKEN_COUNT {
                final_decode_input = Some(decode_input);
                final_decode_output = Some(current_token);
            }
        }
        let decode_milliseconds = decode_started.elapsed().as_millis();
        assert_layer_lengths(&request, FINAL_COMMITTED_LENGTH)?;

        // Rewind the last speculative transition on all 30 opaque KV states,
        // then replay it on the same request. A deterministic exact token is
        // required after recovery.
        request
            .cancel_last_transition()
            .map_err(|error| format!("last-transition cancellation failed: {error}"))?;
        if request.transition_committed() || request.is_poisoned() {
            return Err("successful cancellation left the request committed/poisoned".to_owned());
        }
        assert_layer_lengths(&request, FINAL_COMMITTED_LENGTH - 1)?;
        let replay = request
            .execute(&[final_decode_input.expect("17 decodes always set replay input")])
            .map_err(|error| format!("cancelled transition replay failed: {error}"))?;
        totals.record(&replay, 1, vocab_size, contract.target)?;
        if replay.token_ids() != [final_decode_output.expect("17 decodes always set output")]
            || request.is_poisoned()
            || !request.transition_committed()
        {
            return Err("cancel/recovery replay was not deterministic and committed".to_owned());
        }
        assert_layer_lengths(&request, FINAL_COMMITTED_LENGTH)?;

        drop(replay);
        drop(request);
        drop(resident);
        Ok((
            upload_milliseconds,
            prefill_milliseconds,
            decode_milliseconds,
            totals,
            available,
        ))
    })();

    // Always inspect and close the session, even when the operation failed.
    // Locals owned by the closure are dropped on every Result path first.
    let memory = session.memory_snapshot();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("session shutdown failed: {error}"))?;
    if memory.current_bytes() != 0
        || memory.model_resident().current_bytes() != 0
        || memory.request_state().current_bytes() != 0
        || memory.workspace().current_bytes() != 0
        || memory.poisoned()
        || cleanup.retryable_cleanup != 0
        || cleanup.durable_quarantine != 0
    {
        return Err(format!(
            "resource cleanup differs: memory={memory:?}, shutdown={cleanup:?}"
        ));
    }
    let (upload_milliseconds, prefill_milliseconds, decode_milliseconds, totals, available) =
        operation?;
    let mut output_digest = Sha256::new();
    for token in &totals.output_token_ids {
        output_digest.update(token.to_le_bytes());
    }
    Ok(SmokeReport {
        source_identity,
        target: contract.target,
        device_index,
        resident_bytes: GEMMA4_MOE_TEXT_RESIDENT_BYTES,
        peak_accounted_bytes: memory.high_water_bytes(),
        available_memory_bytes: available,
        upload_milliseconds,
        prefill_milliseconds,
        decode_milliseconds,
        prefill_tokens: PREFILL_TOKEN_COUNT,
        committed_decode_tokens: DECODE_TOKEN_COUNT,
        output_tokens_checked: totals.output_tokens_checked,
        output_token_sha256: format!("{:x}", output_digest.finalize()),
        layer_count: LAYER_COUNT,
        state_capacity: STATE_CAPACITY,
        submission_count: totals.submission_count,
        kernel_dispatch_count: totals.kernel_dispatch_count,
        sparse_moe_submission_count: totals.sparse_moe_submission_count,
        sparse_moe_active_pair_count: totals.sparse_moe_active_pair_count,
        fallback_used: false,
        nonfinite_free: true,
        cancel_recovery_verified: true,
        final_committed_length: FINAL_COMMITTED_LENGTH,
        cleanup_retryable: cleanup.retryable_cleanup,
        cleanup_durable: cleanup.durable_quarantine,
    })
}

fn run_actual(contract: TargetContract) -> Result<SmokeReport, String> {
    let device_index = require_gate(contract)?;
    match load_source()? {
        ActualSource::Safetensors(source) => {
            let source = Arc::new(source);
            let plan = build_gemma4_moe_resident_weight_load_plan(source.as_ref())
                .map_err(|error| format!("canonical source load plan failed: {error}"))?;
            let graph =
                build_gemma4_moe_graph(source.as_ref(), PREFILL_TOKEN_COUNT, 0, STATE_CAPACITY)
                    .map_err(|error| format!("canonical source graph failed: {error}"))?;
            let layout = build_gemma4_moe_execution_layout(&graph, &plan)
                .map_err(|error| format!("canonical source execution layout failed: {error}"))?;
            execute_resident(
                contract,
                device_index,
                sllm_core::GEMMA4_MOE_MODEL_FINGERPRINT.to_owned(),
                source,
                graph,
                plan,
                layout,
            )
        }
        ActualSource::Gguf(source) => {
            let source = Arc::new(source);
            let plan = build_gemma4_moe_resident_weight_load_plan(source.as_ref())
                .map_err(|error| format!("canonical GGUF load plan failed: {error}"))?;
            let graph = build_gemma4_moe_gguf_graph(
                source.as_ref(),
                PREFILL_TOKEN_COUNT,
                0,
                STATE_CAPACITY,
            )
            .map_err(|error| format!("canonical GGUF graph failed: {error}"))?;
            let layout = build_gemma4_moe_execution_layout(&graph, &plan)
                .map_err(|error| format!("canonical GGUF execution layout failed: {error}"))?;
            let source_identity = source.file_sha256().to_owned();
            execute_resident(
                contract,
                device_index,
                source_identity,
                source,
                graph,
                plan,
                layout,
            )
        }
    }
}

#[test]
#[ignore = "requires fixed 26B source/GGUF and the canonical R9700 gfx1201"]
fn phase55_gemma4_moe_full_resident_generation_gfx1201() {
    let device_index = require_gate(GFX1201).expect("gfx1201 actual GPU gate must be exact");
    let report = run_actual(GFX1201).expect("gfx1201 actual full-resident smoke must pass");
    report.assert_pass(GFX1201, device_index);
    eprintln!("phase55 Gemma 4 MoE actual GPU PASS: {report:#?}");
}

#[test]
#[ignore = "requires fixed 26B source/GGUF and a full-resident-capable V620 gfx1030"]
fn phase55_gemma4_moe_full_resident_generation_gfx1030() {
    let device_index = require_gate(GFX1030).expect("gfx1030 actual GPU gate must be exact");
    let report = run_actual(GFX1030).expect("gfx1030 actual full-resident smoke must pass");
    report.assert_pass(GFX1030, device_index);
    eprintln!("phase55 Gemma 4 MoE actual GPU PASS: {report:#?}");
}

#[test]
fn phase55_actual_gpu_contract_is_fixed_and_fail_closed() {
    assert_eq!(PREFILL_TOKENS.len(), 17);
    assert_eq!(PREFILL_TOKENS.last(), Some(&262_143));
    assert_eq!(FINAL_COMMITTED_LENGTH, 34);
    assert_eq!(STATE_CAPACITY, 1_024);
    assert_eq!(LAYER_COUNT, 30);
    assert_eq!(GFX1201.target, "gfx1201");
    assert_eq!(GFX1030.target, "gfx1030");
    assert_ne!(GFX1201.gate_env, GFX1030.gate_env);
}
