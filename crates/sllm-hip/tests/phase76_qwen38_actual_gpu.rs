//! Phase 76/78 actual-GPU smoke for the exact Unsloth Qwen3.8-27B NVFP4
//! artifact.  The tests are ignored by default and are deliberately gated by
//! an explicit environment variable so a host-only test run cannot claim GPU
//! evidence.

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use sllm_core::{
    Backend, ExecutionSessionRequest, KvCacheEncoding, QWEN35_VOCAB_SIZE, QwenResidentModel,
    build_qwen35_unsloth_qwen38_nvfp4_graph, build_qwen38_nvfp4_weight_load_plan, read_model_lock,
    verify_unsloth_qwen38_nvfp4,
};
use sllm_hip::HipBackend;

const PREFILL_TOKENS: [i32; 17] = [
    2, 106, 1_645, 108, 9_259, 236_776, 563, 107, 17, 23, 42, 255, 256, 257, 4_097, 65_537, 248_319,
];
const DECODE_TOKENS: usize = 4;
const STATE_CAPACITY: u64 = 34;
const PROFILE_LENGTHS: [usize; 3] = [512, 2_048, 9_435];
const PROFILE_LENGTHS_ENV: &str = "SLLM_PHASE78_QWEN38_PROFILE_LENGTHS";
const PROFILE_CHUNK_CAPACITY: u64 = 1_024;
const PROFILE_STATE_CAPACITY: u64 = 9_435;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(600);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Copy)]
struct TargetContract {
    target: &'static str,
    gate: &'static str,
    device: &'static str,
}

const GFX1201: TargetContract = TargetContract {
    target: "gfx1201",
    gate: "SLLM_PHASE76_QWEN38_GFX1201",
    device: "SLLM_PHASE76_QWEN38_GFX1201_DEVICE",
};
const GFX1030: TargetContract = TargetContract {
    target: "gfx1030",
    gate: "SLLM_PHASE76_QWEN38_GFX1030",
    device: "SLLM_PHASE76_QWEN38_GFX1030_DEVICE",
};

fn run_actual(contract: TargetContract) -> Result<(), String> {
    if env::var(contract.gate).as_deref() != Ok("1") {
        return Err(format!("{}=1 is required", contract.gate));
    }
    let device_index = env::var(contract.device)
        .map_err(|_| format!("{} is required", contract.device))?
        .parse::<u32>()
        .map_err(|_| format!("{} must be a u32 HIP device index", contract.device))?;
    let root = PathBuf::from(
        env::var_os("SLLM_QWEN38_NVFP4_CACHE")
            .ok_or_else(|| "SLLM_QWEN38_NVFP4_CACHE is required".to_owned())?,
    );
    let verify_started = Instant::now();
    let artifact = Arc::new(verify_unsloth_qwen38_nvfp4(&root).map_err(|e| e.to_string())?);
    let verify_ms = verify_started.elapsed().as_millis();
    let lock_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/models/locks/qwen3.5-27b-bf16.json");
    let lock = read_model_lock(&lock_path).map_err(|e| e.to_string())?;
    let plan_started = Instant::now();
    let plan = build_qwen38_nvfp4_weight_load_plan(&lock, &artifact).map_err(|e| e.to_string())?;
    let graph = build_qwen35_unsloth_qwen38_nvfp4_graph(
        &lock,
        &plan,
        &artifact,
        PREFILL_TOKENS.len() as u64,
        STATE_CAPACITY,
        KvCacheEncoding::Fp16,
    )
    .map_err(|e| e.to_string())?;
    let plan_graph_ms = plan_started.elapsed().as_millis();

    let backend = HipBackend::connect().map_err(|e| e.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, contract.target.to_owned())
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    let operation = (|| -> Result<(), String> {
        let available = session
            .available_memory_bytes()
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "HIP session did not report available memory".to_owned())?;
        let load_started = Instant::now();
        let resident = QwenResidentModel::new_unsloth_qwen38_nvfp4(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Arc::clone(&artifact),
            COMPLETION_TIMEOUT,
        )
        .map_err(|e| format!("model provisioning failed: {e}"))?;
        let load_ms = load_started.elapsed().as_millis();
        let ready = resident.memory_snapshot();
        if ready.poisoned() || ready.model_resident().current_bytes() == 0 {
            return Err(format!(
                "invalid model-ready allocation snapshot: {ready:?}"
            ));
        }
        let mut request = resident
            .new_request(graph.clone())
            .map_err(|e| format!("request creation failed: {e}"))?;
        let prefill_started = Instant::now();
        let prefill = request
            .prefill(&PREFILL_TOKENS)
            .map_err(|e| format!("prefill failed: {e}"))?;
        let prefill_ms = prefill_started.elapsed().as_millis();
        if prefill.token_ids().len() != PREFILL_TOKENS.len() {
            return Err("prefill output row count differs".to_owned());
        }
        let mut generated = Vec::with_capacity(DECODE_TOKENS);
        let mut current = *prefill
            .token_ids()
            .last()
            .ok_or_else(|| "prefill produced no terminal token".to_owned())?;
        if !(0..QWEN35_VOCAB_SIZE as i32).contains(&current) {
            return Err(format!(
                "prefill terminal token is outside vocabulary: {current}"
            ));
        }
        let decode_started = Instant::now();
        for _ in 0..DECODE_TOKENS {
            let output = request
                .decode(current)
                .map_err(|e| format!("decode failed: {e}"))?;
            if output.token_ids().len() != 1 {
                return Err("decode output row count differs".to_owned());
            }
            current = output.token_ids()[0];
            if !(0..QWEN35_VOCAB_SIZE as i32).contains(&current) {
                return Err(format!("decode token is outside vocabulary: {current}"));
            }
            generated.push(current);
        }
        let decode_ms = decode_started.elapsed().as_millis();
        let audit = request.audit_snapshot().map_err(|e| e.to_string())?;
        let nvfp4_decode_dispatches =
            audit.kernel_dispatch_count_for(58, "matmul.nvfp4.w4a4.block16.decode.v1");
        let nvfp4_decode_columns_dispatches =
            audit.kernel_dispatch_count_for(65, "matmul.nvfp4.w4a4.decode.columns128.v1");
        let nvfp4_decode_wave4_dispatches =
            audit.kernel_dispatch_count_for(67, "matmul.nvfp4.w4a4.decode.dp4a.wave4col32.v1");
        let nvfp4_prefill_dispatches = audit
            .kernel_dispatch_count_for(59, "matmul.nvfp4.w4a4.block16.prefill.row8_tiled256.v1");
        let nvfp4_prefill_col8_dispatches = audit.kernel_dispatch_count_for(
            61,
            "matmul.nvfp4.w4a4.block16.prefill.row8_col8_tiled256.v1",
        );
        let nvfp4_prefill_dp4a_dispatches =
            audit.kernel_dispatch_count_for(62, "matmul.nvfp4.w4a4.block16.prefill.dp4a64x64.v1");
        let nvfp4_prefill_wmma_dispatches =
            audit.kernel_dispatch_count_for(64, "matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x64.v1");
        let nvfp4_prefill_f16scale_dispatches = audit.kernel_dispatch_count_for(
            69,
            "matmul.nvfp4.w4a4.prefill.gfx1201.wmma_f16scale128x64.v1",
        );
        let fp8_outer_prefill_dispatches =
            audit.kernel_dispatch_count_for(60, "matmul.fp8.outer.prefill.tiled16.v1");
        let fp8_outer_half2_dispatches =
            audit.kernel_dispatch_count_for(63, "matmul.fp8.outer.prefill.gfx1030.half2.128x64.v1");
        let fp8_emulation_dispatches =
            audit.kernel_dispatch_count_for(6, "matmul.fp8.outer.emulation.v1");
        let fp8_decode_half2_dispatches = audit
            .kernel_dispatch_count_for(66, "matmul.fp8.outer.decode.gfx1030.half2.wave4col32.v1");
        let fp8_decode_dword8_dispatches = audit
            .kernel_dispatch_count_for(68, "matmul.fp8.outer.decode.gfx1030.dword8.wave4col32.v1");
        let fp8_native_dispatches =
            audit.kernel_dispatch_count_for(5, "matmul.fp8.outer.hipblaslt.v1");
        let nvfp4_decode_ok =
            if env::var("SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4").as_deref() == Ok("1") {
                nvfp4_decode_wave4_dispatches != 0
                    && nvfp4_decode_dispatches == 0
                    && nvfp4_decode_columns_dispatches == 0
            } else if env::var("SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_COLUMNS").as_deref() == Ok("1") {
                nvfp4_decode_columns_dispatches != 0 && nvfp4_decode_dispatches == 0
            } else {
                nvfp4_decode_dispatches != 0
                    && nvfp4_decode_columns_dispatches == 0
                    && nvfp4_decode_wave4_dispatches == 0
            };
        let nvfp4_prefill_ok =
            if env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_COL8").as_deref() == Ok("1") {
                nvfp4_prefill_col8_dispatches != 0
                    && nvfp4_prefill_dispatches == 0
                    && nvfp4_prefill_dp4a_dispatches == 0
                    && nvfp4_prefill_wmma_dispatches == 0
                    && nvfp4_prefill_f16scale_dispatches == 0
            } else if env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA_F16SCALE").as_deref()
                == Ok("1")
                && contract.target == "gfx1201"
            {
                nvfp4_prefill_f16scale_dispatches != 0
                    && nvfp4_prefill_dispatches == 0
                    && nvfp4_prefill_col8_dispatches == 0
                    && nvfp4_prefill_dp4a_dispatches == 0
                    && nvfp4_prefill_wmma_dispatches == 0
            } else if env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA").as_deref() == Ok("1")
                && contract.target == "gfx1201"
            {
                nvfp4_prefill_wmma_dispatches != 0
                    && nvfp4_prefill_dispatches == 0
                    && nvfp4_prefill_col8_dispatches == 0
                    && nvfp4_prefill_dp4a_dispatches == 0
                    && nvfp4_prefill_f16scale_dispatches == 0
            } else if env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A").as_deref() == Ok("1") {
                nvfp4_prefill_dp4a_dispatches != 0
                    && nvfp4_prefill_dispatches == 0
                    && nvfp4_prefill_col8_dispatches == 0
                    && nvfp4_prefill_wmma_dispatches == 0
                    && nvfp4_prefill_f16scale_dispatches == 0
            } else {
                nvfp4_prefill_dispatches != 0
                    && nvfp4_prefill_col8_dispatches == 0
                    && nvfp4_prefill_dp4a_dispatches == 0
                    && nvfp4_prefill_wmma_dispatches == 0
                    && nvfp4_prefill_f16scale_dispatches == 0
            };
        let target_prefill_ok = nvfp4_prefill_ok
            && if contract.target == "gfx1030" {
                if env::var("SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2").as_deref() == Ok("1") {
                    fp8_outer_half2_dispatches != 0
                        && fp8_outer_prefill_dispatches == 0
                        && fp8_native_dispatches == 0
                } else {
                    fp8_outer_prefill_dispatches != 0
                        && fp8_outer_half2_dispatches == 0
                        && fp8_native_dispatches == 0
                }
            } else {
                fp8_outer_prefill_dispatches == 0
                    && fp8_outer_half2_dispatches == 0
                    && fp8_native_dispatches != 0
            };
        let fp8_decode_ok = if contract.target == "gfx1030" {
            if env::var("SLLM_FP8_OUTER_DECODE_FORCE_BASELINE").as_deref() == Ok("1") {
                fp8_emulation_dispatches != 0
                    && fp8_decode_half2_dispatches == 0
                    && fp8_decode_dword8_dispatches == 0
            } else if env::var("SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_DWORD8").as_deref() == Ok("1") {
                fp8_decode_dword8_dispatches != 0
                    && fp8_emulation_dispatches == 0
                    && fp8_decode_half2_dispatches == 0
            } else if env::var("SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_HALF2").as_deref() == Ok("1") {
                fp8_decode_half2_dispatches != 0
                    && fp8_emulation_dispatches == 0
                    && fp8_decode_dword8_dispatches == 0
            } else {
                fp8_emulation_dispatches != 0
                    && fp8_decode_half2_dispatches == 0
                    && fp8_decode_dword8_dispatches == 0
            }
        } else {
            fp8_decode_half2_dispatches == 0
                && fp8_decode_dword8_dispatches == 0
                && fp8_emulation_dispatches == 0
        };
        if audit.fallback_used()
            || !audit.all_dispatches_hip()
            || audit.kernel_dispatch_count() == 0
            || audit.submission_count() == 0
            || !nvfp4_decode_ok
            || !target_prefill_ok
            || !fp8_decode_ok
        {
            return Err(format!(
                "dispatch audit is not HIP-only/fallback-free: {audit:?}"
            ));
        }

        // Replaying the same prefill on a fresh request is a deterministic
        // numerical oracle for the mixed graph and exercises full model load
        // independently of host-side tokenization.
        let mut replay = resident
            .new_request(graph)
            .map_err(|e| format!("replay request creation failed: {e}"))?;
        let replay_output = replay
            .prefill(&PREFILL_TOKENS)
            .map_err(|e| format!("replay prefill failed: {e}"))?;
        if replay_output.token_ids() != prefill.token_ids() {
            return Err("replayed prefill changed the token oracle".to_owned());
        }
        let replay_audit = replay.audit_snapshot().map_err(|e| e.to_string())?;
        if replay_audit.fallback_used() || !replay_audit.all_dispatches_hip() {
            return Err(format!(
                "replay dispatch audit is not HIP-only: {replay_audit:?}"
            ));
        }
        let memory = session.memory_snapshot();
        eprintln!(
            "phase76/78 Qwen3.8 actual GPU PASS target={} device={} artifact={} verify_ms={} plan_graph_ms={} load_ms={} prefill_ms={} decode_ms={} decode_tokens={:?} resident_bytes={} available_bytes={} dispatches={} submissions={} nvfp4_decode_dispatches={} nvfp4_decode_columns_dispatches={} nvfp4_decode_wave4_dispatches={} nvfp4_prefill_dispatches={} nvfp4_prefill_col8_dispatches={} nvfp4_prefill_dp4a_dispatches={} nvfp4_prefill_wmma_dispatches={} nvfp4_prefill_f16scale_dispatches={} fp8_outer_prefill_dispatches={} fp8_outer_half2_dispatches={} fp8_emulation_dispatches={} fp8_decode_half2_dispatches={} fp8_decode_dword8_dispatches={} fp8_native_dispatches={} high_water_bytes={}",
            contract.target,
            device_index,
            root.display(),
            verify_ms,
            plan_graph_ms,
            load_ms,
            prefill_ms,
            decode_ms,
            generated,
            memory.model_resident().current_bytes(),
            available,
            audit.kernel_dispatch_count(),
            audit.submission_count(),
            nvfp4_decode_dispatches,
            nvfp4_decode_columns_dispatches,
            nvfp4_decode_wave4_dispatches,
            nvfp4_prefill_dispatches,
            nvfp4_prefill_col8_dispatches,
            nvfp4_prefill_dp4a_dispatches,
            nvfp4_prefill_wmma_dispatches,
            nvfp4_prefill_f16scale_dispatches,
            fp8_outer_prefill_dispatches,
            fp8_outer_half2_dispatches,
            fp8_emulation_dispatches,
            fp8_decode_half2_dispatches,
            fp8_decode_dword8_dispatches,
            fp8_native_dispatches,
            memory.high_water_bytes(),
        );
        drop(replay);
        drop(request);
        drop(resident);
        Ok(())
    })();
    let memory = session.memory_snapshot();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|e| format!("session shutdown failed: {e}"))?;
    if memory.current_bytes() != 0
        || cleanup.retryable_cleanup != 0
        || cleanup.durable_quarantine != 0
    {
        return Err(format!(
            "resource cleanup differs: memory={memory:?}, shutdown={cleanup:?}"
        ));
    }
    operation
}

fn run_prefill_profile(contract: TargetContract) -> Result<(), String> {
    let gate = format!(
        "SLLM_PHASE78_QWEN38_PROFILE_{}",
        contract.target.to_ascii_uppercase()
    );
    if env::var(&gate).as_deref() != Ok("1") {
        return Err(format!("{gate}=1 is required"));
    }
    let device_index = env::var(contract.device)
        .map_err(|_| format!("{} is required", contract.device))?
        .parse::<u32>()
        .map_err(|_| format!("{} must be a u32 HIP device index", contract.device))?;
    let root = PathBuf::from(
        env::var_os("SLLM_QWEN38_NVFP4_CACHE")
            .ok_or_else(|| "SLLM_QWEN38_NVFP4_CACHE is required".to_owned())?,
    );
    let artifact = Arc::new(verify_unsloth_qwen38_nvfp4(&root).map_err(|e| e.to_string())?);
    let lock_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/models/locks/qwen3.5-27b-bf16.json");
    let lock = read_model_lock(&lock_path).map_err(|e| e.to_string())?;
    let plan = build_qwen38_nvfp4_weight_load_plan(&lock, &artifact).map_err(|e| e.to_string())?;
    let graph = build_qwen35_unsloth_qwen38_nvfp4_graph(
        &lock,
        &plan,
        &artifact,
        PROFILE_CHUNK_CAPACITY,
        PROFILE_STATE_CAPACITY,
        KvCacheEncoding::Fp16,
    )
    .map_err(|e| e.to_string())?;

    let backend = HipBackend::connect().map_err(|e| e.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, contract.target.to_owned())
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    let profile_lengths = requested_profile_lengths()?;
    let operation = (|| -> Result<Vec<u128>, String> {
        let resident = QwenResidentModel::new_unsloth_qwen38_nvfp4(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Arc::clone(&artifact),
            COMPLETION_TIMEOUT,
        )
        .map_err(|e| format!("model provisioning failed: {e}"))?;
        let mut elapsed = Vec::with_capacity(profile_lengths.len());
        for (case_index, length) in profile_lengths.iter().copied().enumerate() {
            let tokens = (0..length)
                .map(|index| {
                    ((index.wrapping_mul(7919) + case_index * 17) % QWEN35_VOCAB_SIZE) as i32
                })
                .collect::<Vec<_>>();
            let mut request = resident
                .new_request(graph.clone())
                .map_err(|e| format!("profile request creation failed: {e}"))?;
            let started = Instant::now();
            let output = request
                .prefill(&tokens)
                .map_err(|e| format!("profile prefill length {length} failed: {e}"))?;
            let elapsed_ms = started.elapsed().as_millis();
            if output.token_ids().is_empty()
                || output
                    .token_ids()
                    .iter()
                    .any(|token| !(0..QWEN35_VOCAB_SIZE as i32).contains(token))
            {
                return Err(format!(
                    "profile prefill length {length} returned invalid tokens"
                ));
            }
            let audit = request.audit_snapshot().map_err(|e| e.to_string())?;
            let nvfp4_prefill = audit.kernel_dispatch_count_for(
                59,
                "matmul.nvfp4.w4a4.block16.prefill.row8_tiled256.v1",
            );
            let nvfp4_prefill_col8 = audit.kernel_dispatch_count_for(
                61,
                "matmul.nvfp4.w4a4.block16.prefill.row8_col8_tiled256.v1",
            );
            let nvfp4_prefill_dp4a = audit
                .kernel_dispatch_count_for(62, "matmul.nvfp4.w4a4.block16.prefill.dp4a64x64.v1");
            let nvfp4_prefill_wmma = audit
                .kernel_dispatch_count_for(64, "matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x64.v1");
            let nvfp4_prefill_f16scale = audit.kernel_dispatch_count_for(
                69,
                "matmul.nvfp4.w4a4.prefill.gfx1201.wmma_f16scale128x64.v1",
            );
            let fp8_prefill =
                audit.kernel_dispatch_count_for(60, "matmul.fp8.outer.prefill.tiled16.v1");
            let fp8_half2 = audit
                .kernel_dispatch_count_for(63, "matmul.fp8.outer.prefill.gfx1030.half2.128x64.v1");
            let fp8_native = audit.kernel_dispatch_count_for(5, "matmul.fp8.outer.hipblaslt.v1");
            let fp8_ok = if contract.target == "gfx1030" {
                if env::var("SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2").as_deref() == Ok("1") {
                    fp8_half2 != 0 && fp8_prefill == 0 && fp8_native == 0
                } else {
                    fp8_prefill != 0 && fp8_half2 == 0 && fp8_native == 0
                }
            } else {
                fp8_prefill == 0 && fp8_half2 == 0 && fp8_native != 0
            };
            let nvfp4_ok = if env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_COL8").as_deref() == Ok("1") {
                nvfp4_prefill_col8 != 0
                    && nvfp4_prefill == 0
                    && nvfp4_prefill_dp4a == 0
                    && nvfp4_prefill_wmma == 0
                    && nvfp4_prefill_f16scale == 0
            } else if env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA_F16SCALE").as_deref()
                == Ok("1")
                && contract.target == "gfx1201"
            {
                nvfp4_prefill_f16scale != 0
                    && nvfp4_prefill == 0
                    && nvfp4_prefill_col8 == 0
                    && nvfp4_prefill_dp4a == 0
                    && nvfp4_prefill_wmma == 0
            } else if env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA").as_deref() == Ok("1")
                && contract.target == "gfx1201"
            {
                nvfp4_prefill_wmma != 0
                    && nvfp4_prefill == 0
                    && nvfp4_prefill_col8 == 0
                    && nvfp4_prefill_dp4a == 0
                    && nvfp4_prefill_f16scale == 0
            } else if env::var("SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A").as_deref() == Ok("1") {
                nvfp4_prefill_dp4a != 0
                    && nvfp4_prefill == 0
                    && nvfp4_prefill_col8 == 0
                    && nvfp4_prefill_wmma == 0
                    && nvfp4_prefill_f16scale == 0
            } else {
                nvfp4_prefill != 0
                    && nvfp4_prefill_col8 == 0
                    && nvfp4_prefill_dp4a == 0
                    && nvfp4_prefill_wmma == 0
                    && nvfp4_prefill_f16scale == 0
            };
            if audit.fallback_used() || !audit.all_dispatches_hip() || !nvfp4_ok || !fp8_ok {
                return Err(format!(
                    "profile dispatch audit failed length={length}: {audit:?}"
                ));
            }
            eprintln!(
                "phase78 profile target={} device={} tokens={} prefill_ms={} dispatches={} nvfp4_prefill={} nvfp4_prefill_col8={} nvfp4_prefill_dp4a={} nvfp4_prefill_wmma={} nvfp4_prefill_f16scale={} fp8_prefill={} fp8_half2={} fp8_native={}",
                contract.target,
                device_index,
                length,
                elapsed_ms,
                audit.kernel_dispatch_count(),
                nvfp4_prefill,
                nvfp4_prefill_col8,
                nvfp4_prefill_dp4a,
                nvfp4_prefill_wmma,
                nvfp4_prefill_f16scale,
                fp8_prefill,
                fp8_half2,
                fp8_native,
            );
            elapsed.push(elapsed_ms);
            drop(request);
        }
        drop(resident);
        Ok(elapsed)
    })();
    let memory = session.memory_snapshot();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|e| format!("session shutdown failed: {e}"))?;
    if memory.current_bytes() != 0
        || cleanup.retryable_cleanup != 0
        || cleanup.durable_quarantine != 0
    {
        return Err(format!(
            "profile resource cleanup differs: memory={memory:?}, shutdown={cleanup:?}"
        ));
    }
    let elapsed = operation?;
    eprintln!(
        "phase78 Qwen3.8 prefill profile PASS target={} device={} lengths={:?} elapsed_ms={:?}",
        contract.target, device_index, profile_lengths, elapsed
    );
    Ok(())
}

fn requested_profile_lengths() -> Result<Vec<usize>, String> {
    let text = match env::var(PROFILE_LENGTHS_ENV) {
        Ok(text) => text,
        Err(env::VarError::NotPresent) => return Ok(PROFILE_LENGTHS.to_vec()),
        Err(error) => return Err(format!("cannot read {PROFILE_LENGTHS_ENV}: {error}")),
    };
    let lengths = text
        .split(',')
        .map(str::trim)
        .map(|item| {
            item.parse::<usize>()
                .map_err(|_| format!("{PROFILE_LENGTHS_ENV} contains a non-usize value: {item}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if lengths.is_empty()
        || lengths
            .iter()
            .any(|length| !PROFILE_LENGTHS.contains(length))
    {
        return Err(format!(
            "{PROFILE_LENGTHS_ENV} must be a non-empty comma-separated subset of {PROFILE_LENGTHS:?}"
        ));
    }
    Ok(lengths)
}

#[test]
#[ignore = "requires the exact Unsloth Qwen3.8-27B-NVFP4 artifact and R9700 gfx1201"]
fn phase76_qwen38_nvfp4_full_resident_generation_gfx1201() {
    run_actual(GFX1201).expect("gfx1201 Qwen3.8 actual-GPU smoke must pass");
}

#[test]
#[ignore = "requires the exact Unsloth Qwen3.8-27B-NVFP4 artifact and V620 gfx1030"]
fn phase77_qwen38_nvfp4_full_resident_generation_gfx1030() {
    run_actual(GFX1030).expect("gfx1030 Qwen3.8 actual-GPU smoke must pass");
}

#[test]
#[ignore = "requires the exact Unsloth Qwen3.8-27B-NVFP4 artifact and R9700 gfx1201"]
fn phase78_qwen38_prefill_profile_gfx1201() {
    run_prefill_profile(GFX1201).expect("gfx1201 Qwen3.8 Phase78 profile must pass");
}

#[test]
#[ignore = "requires the exact Unsloth Qwen3.8-27B-NVFP4 artifact and V620 gfx1030"]
fn phase78_qwen38_prefill_profile_gfx1030() {
    run_prefill_profile(GFX1030).expect("gfx1030 Qwen3.8 Phase78 profile must pass");
}

#[test]
fn phase76_qwen38_gpu_contract_is_fixed_and_fail_closed() {
    assert_eq!(PREFILL_TOKENS.len(), 17);
    assert_eq!(DECODE_TOKENS, 4);
    assert_eq!(STATE_CAPACITY, 34);
    assert_eq!(PROFILE_LENGTHS, [512, 2_048, 9_435]);
    assert_eq!(PROFILE_LENGTHS_ENV, "SLLM_PHASE78_QWEN38_PROFILE_LENGTHS");
    assert_eq!(PROFILE_CHUNK_CAPACITY, 1_024);
    assert_eq!(PROFILE_STATE_CAPACITY, 9_435);
    assert_eq!(GFX1201.target, "gfx1201");
    assert_eq!(GFX1030.target, "gfx1030");
    assert_ne!(GFX1201.gate, GFX1030.gate);
}
