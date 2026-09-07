//! Phase 81 actual-GPU fixed-sampling integration harness.
//!
//! These tests are ignored by default.  They require an explicit target gate,
//! device index, and verified model cache, so a host-only test run cannot be
//! mistaken for GPU evidence.  The Qwen lane accepts `fp16` (the rollback
//! recipe) and `mxfp8` (the reviewed OCP MXFP8 E4 recipe) through
//! `SLLM_PHASE81_QWEN35_KV`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use sllm_core::{
    Backend, DeviceTokenSelectorRequestV1, ExecutionSessionRequest, GEMMA4_MOE_MODEL_FINGERPRINT,
    GEMMA4_MOE_TEXT_RESIDENT_BYTES, GEMMA4_VOCAB_SIZE, Gemma4MoeExecutionOutput, Gemma4MoeGraph,
    Gemma4MoeResidentModel, Gemma4MoeWeightSource, Gemma4ResidentModel, KvCacheEncoding,
    KvCacheSelectionRequest, MINISTRAL3_GRAPH_VOCAB_SIZE, MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256,
    MINISTRAL3_OFFICIAL_GGUF_REPOSITORY, MINISTRAL3_OFFICIAL_GGUF_REVISION,
    MINISTRAL3_WEIGHT_LOCK_FINGERPRINT, MINISTRAL3_WEIGHT_RESIDENT_BYTES, Ministral3DispatchAudit,
    Ministral3ResidentModel, ModelLock, QWEN35_4B_FINGERPRINT, QWEN35_VOCAB_SIZE,
    QwenResidentModel, SamplerChainConfigV1, SamplerChainV1, SamplingParametersV1,
    VerifiedGemma4Moe, VerifiedGgufGemma4Moe, VerifiedMinistral3WeightSource, WeightLoadPlan,
    build_gemma4_moe_execution_layout, build_gemma4_moe_gguf_graph, build_gemma4_moe_graph,
    build_gemma4_moe_resident_weight_load_plan, build_ministral3_weight_load_plan,
    build_qwen35_graph_with_kv_cache_selection, build_unsloth_gemma4_nvfp4_weight_load_plan,
    build_verified_gemma4_weight_load_plan, build_verified_gguf_gemma_weight_load_plan,
    build_verified_weight_load_plan, open_and_verify_official_ministral3_gguf,
    parse_gemma4_model_lock, read_derived_gguf_lock, read_model_lock, resolve_kv_cache_selection,
    verify_derived_gguf, verify_gemma4_moe_artifact, verify_gguf_gemma4_moe,
    verify_unsloth_gemma4_nvfp4,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(600);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);
const PREFILL_TOKENS: [i32; 17] = [
    2, 106, 1_645, 108, 9_259, 236_776, 563, 107, 17, 23, 42, 255, 256, 257, 4_097, 65_537, 248_319,
];
const STATE_CAPACITY: u64 = 34;
const SEED: u64 = 0x81_20_64_u64;
const FORCED_TOKEN: usize = 42;
const GEMMA4_MOE_VOCAB_SIZE: usize = 262_144;
const GEMMA4_MOE_STATE_CAPACITY: u64 = 1_024;
const GEMMA4_MOE_PREFILL_TOKENS: [i32; 17] = [
    2, 106, 1_645, 108, 9_259, 236_776, 563, 107, 17, 23, 42, 255, 256, 257, 4_097, 65_537, 262_143,
];
const GEMMA4_MOE_SOURCE_CACHE: &str = "/home/homelab1/.cache/huggingface/hub/\
models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/\
a19cfe00be84568a6867111c9a68c9c44fdcffe6";
const MINISTRAL3_GGUF_CACHE: &str = "/home/homelab1/.cache/sllm/phase81-ministral3-official-bf16/\
Ministral-3-3B-Instruct-2512-BF16.gguf";
const MINISTRAL3_LOCK_RELATIVE_PATH: &str =
    "docs/models/locks/ministral3-3b-instruct-2512-official-bf16-gguf.json";
const MINISTRAL3_PREFILL_TOKENS: [i32; 3] = [22_177, 1_307, 1_278];
const MINISTRAL3_STATE_CAPACITY: u64 = 4;

#[derive(Clone, Copy, Debug)]
struct TargetContract {
    target: &'static str,
    gate: &'static str,
    device: &'static str,
}

const GFX1201: TargetContract = TargetContract {
    target: "gfx1201",
    gate: "SLLM_PHASE81_FIXED_SAMPLING_GFX1201",
    device: "SLLM_PHASE81_FIXED_SAMPLING_GFX1201_DEVICE",
};
const GFX1030: TargetContract = TargetContract {
    target: "gfx1030",
    gate: "SLLM_PHASE81_FIXED_SAMPLING_GFX1030",
    device: "SLLM_PHASE81_FIXED_SAMPLING_GFX1030_DEVICE",
};

#[derive(Clone, Copy, Debug)]
struct GemmaMoeTargetContract {
    target: &'static str,
    gate: &'static str,
    device: &'static str,
}

const GEMMA_MOE_GFX1201: GemmaMoeTargetContract = GemmaMoeTargetContract {
    target: "gfx1201",
    gate: "SLLM_PHASE81_GEMMA4_MOE_GFX1201",
    device: "SLLM_PHASE81_GEMMA4_MOE_GFX1201_DEVICE",
};
const GEMMA_MOE_GFX1030: GemmaMoeTargetContract = GemmaMoeTargetContract {
    target: "gfx1030",
    gate: "SLLM_PHASE81_GEMMA4_MOE_GFX1030",
    device: "SLLM_PHASE81_GEMMA4_MOE_GFX1030_DEVICE",
};

#[derive(Clone, Copy, Debug)]
struct MinistralTargetContract {
    target: &'static str,
    gate: &'static str,
    device: &'static str,
}

const MINISTRAL_GFX1201: MinistralTargetContract = MinistralTargetContract {
    target: "gfx1201",
    gate: "SLLM_PHASE81_MINISTRAL3_GFX1201",
    device: "SLLM_PHASE81_MINISTRAL3_GFX1201_DEVICE",
};
const MINISTRAL_GFX1030: MinistralTargetContract = MinistralTargetContract {
    target: "gfx1030",
    gate: "SLLM_PHASE81_MINISTRAL3_GFX1030",
    device: "SLLM_PHASE81_MINISTRAL3_GFX1030_DEVICE",
};

fn repository_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(relative)
}

fn required_path(name: &str) -> Result<PathBuf, String> {
    env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{name} is required"))
}

fn device_index(contract: TargetContract) -> Result<u32, String> {
    env::var(contract.device)
        .map_err(|_| format!("{} is required", contract.device))?
        .parse::<u32>()
        .map_err(|_| format!("{} must be a u32 HIP device index", contract.device))
}

fn gemma_moe_device_index(contract: GemmaMoeTargetContract) -> Result<u32, String> {
    if env::var(contract.gate).as_deref() != Ok("1") {
        return Err(format!("{}=1 is required", contract.gate));
    }
    env::var(contract.device)
        .map_err(|_| format!("{} is required", contract.device))?
        .parse::<u32>()
        .map_err(|_| format!("{} must be a u32 HIP device index", contract.device))
}

fn ministral_device_index(contract: MinistralTargetContract) -> Result<u32, String> {
    if env::var(contract.gate).as_deref() != Ok("1") {
        return Err(format!("{}=1 is required", contract.gate));
    }
    env::var(contract.device)
        .map_err(|_| format!("{} is required", contract.device))?
        .parse::<u32>()
        .map_err(|_| format!("{} must be a u32 HIP device index", contract.device))
}

fn require_gate(contract: TargetContract) -> Result<u32, String> {
    if env::var(contract.gate).as_deref() != Ok("1") {
        return Err(format!("{}=1 is required", contract.gate));
    }
    device_index(contract)
}

fn selector(
    vocab_size: usize,
    top_k: usize,
    seed: u64,
    counter: u64,
    forced_token: Option<usize>,
) -> Result<DeviceTokenSelectorRequestV1, String> {
    let parameters = SamplingParametersV1::new(1.0, 0.95, 0.0, 0.0)
        .map_err(|error| format!("sampling parameters are invalid: {error}"))?;
    let config = SamplerChainConfigV1::new(parameters)
        .with_top_k(top_k)
        .map_err(|error| format!("fixed top_k is invalid: {error}"))?;
    let chain = SamplerChainV1::new(config, &[])
        .map_err(|error| format!("sampler chain construction failed: {error}"))?;
    let mask = forced_token.map(|token| {
        let mut mask = vec![false; vocab_size];
        mask[token] = true;
        mask
    });
    chain
        .prepare_device_selector(vocab_size, mask.as_deref(), seed, counter)
        .map_err(|error| format!("device selector preparation failed: {error}"))
}

fn check_qwen_output(
    output: &sllm_core::QwenExecutionOutput,
    forced_token: Option<usize>,
) -> Result<i32, String> {
    if output.last_logits().is_some() {
        return Err("device selector unexpectedly returned full logits".to_owned());
    }
    let selection = output
        .selection()
        .ok_or_else(|| "device selector returned no selection record".to_owned())?;
    let token = *output
        .token_ids()
        .last()
        .ok_or_else(|| "device selector returned no token".to_owned())?;
    if selection.token_id
        != u32::try_from(token).map_err(|_| "negative selected token".to_owned())?
    {
        return Err("selection record and output token differ".to_owned());
    }
    if !(0..QWEN35_VOCAB_SIZE as i32).contains(&token) {
        return Err(format!(
            "selected Qwen token is outside vocabulary: {token}"
        ));
    }
    if forced_token.is_some_and(|forced| token != forced as i32) {
        return Err(format!(
            "Qwen single-token mask selected {token}, expected {forced_token:?}"
        ));
    }
    Ok(token)
}

fn check_qwen_audit(
    audit: &sllm_core::QwenExecutionAudit,
    contract: TargetContract,
) -> Result<(), String> {
    if audit.selected_backend() != "hip"
        || audit.target() != contract.target
        || audit.fallback_used()
        || !audit.all_dispatches_hip()
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
    {
        return Err(format!(
            "Qwen dispatch audit is not exact HIP/no-fallback: {audit:?}"
        ));
    }
    Ok(())
}

fn check_gemma_output(
    output: &sllm_core::Gemma4ExecutionOutput,
    contract: TargetContract,
    forced_token: Option<usize>,
) -> Result<i32, String> {
    if output.last_logits().is_some() {
        return Err("device selector unexpectedly returned full logits".to_owned());
    }
    let selection = output
        .selection()
        .ok_or_else(|| "device selector returned no selection record".to_owned())?;
    let token = *output
        .token_ids()
        .last()
        .ok_or_else(|| "device selector returned no token".to_owned())?;
    if selection.token_id
        != u32::try_from(token).map_err(|_| "negative selected token".to_owned())?
    {
        return Err("selection record and output token differ".to_owned());
    }
    if !(0..GEMMA4_VOCAB_SIZE as i32).contains(&token) {
        return Err(format!(
            "selected Gemma token is outside vocabulary: {token}"
        ));
    }
    if forced_token.is_some_and(|forced| token != forced as i32) {
        return Err(format!(
            "Gemma single-token mask selected {token}, expected {forced_token:?}"
        ));
    }
    let audit = output.audit();
    if audit.backend() != 1
        || audit.target() != contract.target
        || audit.fallback_used()
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
    {
        return Err(format!(
            "Gemma dispatch audit is not exact HIP/no-fallback: {audit:?}"
        ));
    }
    Ok(token)
}

fn check_gemma_moe_output(
    output: &Gemma4MoeExecutionOutput,
    contract: GemmaMoeTargetContract,
    forced_token: Option<usize>,
) -> Result<i32, String> {
    // The MoE selector API exposes only the selected token and its fixed-size
    // selection record. A full logits vector is therefore impossible to
    // return through this path; require the record and token to agree.
    if output.token_ids().len() != 1 {
        return Err(format!(
            "Gemma MoE selector returned {} token rows, expected one",
            output.token_ids().len()
        ));
    }
    let selection = output
        .selection()
        .ok_or_else(|| "Gemma MoE device selector returned no selection record".to_owned())?;
    let token = output.token_ids()[0];
    if selection.token_id
        != u32::try_from(token).map_err(|_| "negative selected token".to_owned())?
    {
        return Err("Gemma MoE selection record and output token differ".to_owned());
    }
    if !(0..GEMMA4_MOE_VOCAB_SIZE as i32).contains(&token) {
        return Err(format!(
            "selected Gemma MoE token is outside vocabulary: {token}"
        ));
    }
    if forced_token.is_some_and(|forced| token != forced as i32) {
        return Err(format!(
            "Gemma MoE single-token mask selected {token}, expected {forced_token:?}"
        ));
    }
    let audit = output.audit();
    if audit.backend() != 1
        || audit.target() != contract.target
        || audit.fallback_used()
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
        || audit.segment_count() == 0
        || audit.boundary_count() == 0
        || audit.sparse_moe_submission_count() == 0
        || audit.sparse_moe_active_pair_count() == 0
    {
        return Err(format!(
            "Gemma MoE dispatch audit is not exact HIP/no-fallback: {audit:?}"
        ));
    }
    Ok(token)
}

fn check_ministral_audit(
    audit: &Ministral3DispatchAudit,
    contract: MinistralTargetContract,
) -> Result<(), String> {
    if audit.backend() != 1
        || audit.target() != contract.target
        || audit.fallback_used()
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
        || audit.dispatches().iter().any(|dispatch| {
            dispatch.backend != 1 || dispatch.target != contract.target || dispatch.fallback_used
        })
    {
        return Err(format!(
            "Ministral3 dispatch audit is not exact HIP/no-fallback: {audit:?}"
        ));
    }
    Ok(())
}

fn check_ministral_output(
    output: &sllm_core::Ministral3ExecutionOutput,
    selector: &DeviceTokenSelectorRequestV1,
    contract: MinistralTargetContract,
    forced_token: Option<usize>,
) -> Result<i32, String> {
    if output.last_logits_bf16().is_some() {
        return Err("Ministral3 device selector unexpectedly returned full logits".to_owned());
    }
    if selector.vocab_size() != MINISTRAL3_GRAPH_VOCAB_SIZE
        || selector.top_k() != 0
        || selector.top_p() != 0.95
        || selector.temperature() != 1.0
        || selector.return_logprob()
    {
        return Err(format!(
            "Ministral3 selector is not fixed K0/T1/p.95/no-logprob: {selector:?}"
        ));
    }
    let selection = output
        .selection()
        .ok_or_else(|| "Ministral3 device selector returned no selection record".to_owned())?;
    let token = *output
        .token_ids()
        .last()
        .ok_or_else(|| "Ministral3 device selector returned no token".to_owned())?;
    if selection.token_id
        != u32::try_from(token).map_err(|_| "negative selected token".to_owned())?
    {
        return Err("Ministral3 selection record and output token differ".to_owned());
    }
    if !(0..MINISTRAL3_GRAPH_VOCAB_SIZE as i32).contains(&token) {
        return Err(format!(
            "selected Ministral3 token is outside vocabulary: {token}"
        ));
    }
    if forced_token.is_some_and(|forced| token != forced as i32) {
        return Err(format!(
            "Ministral3 single-token mask selected {token}, expected {forced_token:?}"
        ));
    }
    if let Some(forced) = forced_token {
        if selector.valid_mask().len() != MINISTRAL3_GRAPH_VOCAB_SIZE
            || !selector.is_token_valid(forced)
            || selector
                .valid_mask()
                .iter()
                .enumerate()
                .any(|(token_id, valid)| (token_id != forced) && *valid != 0)
        {
            return Err("Ministral3 single-token selector mask is not exact".to_owned());
        }
    } else if !selector.valid_mask().is_empty() || !selector.additive_logits().is_empty() {
        return Err("Ministral3 neutral selector unexpectedly carries host vectors".to_owned());
    }
    check_ministral_audit(output.audit(), contract)?;
    Ok(token)
}

fn qwen_selection(
    contract: TargetContract,
    lock: &ModelLock,
) -> Result<sllm_core::KvCacheSelection, String> {
    let requested = match env::var("SLLM_PHASE81_QWEN35_KV")
        .unwrap_or_else(|_| "fp16".to_owned())
        .to_ascii_lowercase()
        .as_str()
    {
        "fp16" => KvCacheEncoding::Fp16,
        "mxfp8" | "mxfp8-e4" | "kv-mxfp8-e4" => KvCacheEncoding::Mxfp8E4,
        value => {
            return Err(format!(
                "SLLM_PHASE81_QWEN35_KV must be fp16 or mxfp8, got {value}"
            ));
        }
    };
    resolve_kv_cache_selection(KvCacheSelectionRequest::new(
        Some(requested),
        contract.target,
        lock.fingerprint(),
        true,
        true,
        true,
        256,
    ))
    .map_err(|error| format!("Qwen KV selection failed: {error}"))
}

fn run_qwen(contract: TargetContract) -> Result<(), String> {
    let device = require_gate(contract)?;
    let cache_root = env::var_os("SLLM_PHASE81_QWEN35_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/home/homelab1/.cache/sllm/models/Qwen--Qwen3.5-4B/snapshots/\
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            )
        });
    let lock = read_model_lock(repository_path("docs/models/locks/qwen3.5-4b-bf16.json"))
        .map_err(|error| format!("Qwen lock failed: {error}"))?;
    if lock.fingerprint() != QWEN35_4B_FINGERPRINT {
        return Err("Qwen lock fingerprint differs from the reviewed 4B identity".to_owned());
    }
    let cache = Arc::new(
        lock.verify_cache(&cache_root)
            .map_err(|error| format!("Qwen cache verification failed: {error}"))?,
    );
    let plan = build_verified_weight_load_plan(&lock, &cache)
        .map_err(|error| format!("Qwen weight plan failed: {error}"))?;
    let selection = qwen_selection(contract, &lock)?;
    let graph = build_qwen35_graph_with_kv_cache_selection(
        &lock,
        &plan,
        PREFILL_TOKENS.len() as u64,
        STATE_CAPACITY,
        selection,
    )
    .map_err(|error| format!("Qwen graph failed: {error}"))?;
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let session = Arc::new(
        backend
            .open_execution_session(
                ExecutionSessionRequest::new(device, contract.target.to_owned())
                    .map_err(|error| format!("invalid execution request: {error}"))?,
            )
            .map_err(|error| format!("HIP session open failed: {error}"))?,
    );
    let result = (|| -> Result<(), String> {
        let resident = QwenResidentModel::new(
            Arc::clone(&session),
            graph.clone(),
            plan,
            cache,
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("Qwen resident load failed: {error}"))?;

        let run_pair = |resident: &QwenResidentModel| -> Result<(Vec<i32>, Vec<i32>), String> {
            let mut request = resident
                .new_request(graph.clone())
                .map_err(|error| format!("Qwen request failed: {error}"))?;
            let prefill_selector = selector(QWEN35_VOCAB_SIZE, 20, SEED, 0, None)?;
            let prefill = request
                .prefill_with_device_selector(&PREFILL_TOKENS, &prefill_selector)
                .map_err(|error| format!("Qwen selector prefill failed: {error}"))?;
            let first = check_qwen_output(&prefill, None)?;
            let decode_selector = selector(QWEN35_VOCAB_SIZE, 20, SEED, 1, None)?;
            let decode = request
                .decode_with_device_selector(first, &decode_selector)
                .map_err(|error| format!("Qwen selector decode failed: {error}"))?;
            let second = check_qwen_output(&decode, None)?;
            let audit = request
                .audit_snapshot()
                .map_err(|error| format!("Qwen audit unavailable: {error}"))?;
            check_qwen_audit(&audit, contract)?;
            Ok((vec![first], vec![second]))
        };
        let first = run_pair(&resident)?;
        let second = run_pair(&resident)?;
        if first != second {
            return Err(format!(
                "Qwen fixed seed is not reproducible: {first:?} != {second:?}"
            ));
        }

        let mut masked = resident
            .new_request(graph.clone())
            .map_err(|error| format!("Qwen masked request failed: {error}"))?;
        let prefill_selector = selector(QWEN35_VOCAB_SIZE, 20, SEED, 0, Some(FORCED_TOKEN))?;
        let prefill = masked
            .prefill_with_device_selector(&PREFILL_TOKENS, &prefill_selector)
            .map_err(|error| format!("Qwen masked prefill failed: {error}"))?;
        let first = check_qwen_output(&prefill, Some(FORCED_TOKEN))?;
        let decode_selector = selector(QWEN35_VOCAB_SIZE, 20, SEED, 1, Some(FORCED_TOKEN))?;
        let decode = masked
            .decode_with_device_selector(first, &decode_selector)
            .map_err(|error| format!("Qwen masked decode failed: {error}"))?;
        check_qwen_output(&decode, Some(FORCED_TOKEN))?;
        let audit = masked
            .audit_snapshot()
            .map_err(|error| format!("Qwen masked audit unavailable: {error}"))?;
        check_qwen_audit(&audit, contract)?;
        Ok(())
    })();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("Qwen session shutdown failed: {error}"))?;
    let remaining = session.memory_snapshot().current_bytes();
    result.and_then(|()| {
        if remaining != 0 || cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err(format!(
                "Qwen cleanup is incomplete: remaining={remaining}, report={cleanup:?}"
            ));
        }
        Ok(())
    })
}

enum GemmaMoeActualSource {
    Safetensors(VerifiedGemma4Moe),
    Gguf(VerifiedGgufGemma4Moe),
}

fn optional_nonempty_path(name: &str) -> Result<Option<PathBuf>, String> {
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

fn load_gemma4_moe_source() -> Result<GemmaMoeActualSource, String> {
    let source_dir = optional_nonempty_path("SLLM_PHASE81_GEMMA4_MOE_SOURCE_DIR")?;
    let gguf = optional_nonempty_path("SLLM_PHASE81_GEMMA4_MOE_GGUF")?;
    let gguf_lock = optional_nonempty_path("SLLM_PHASE81_GEMMA4_MOE_GGUF_LOCK")?;
    match (source_dir, gguf, gguf_lock) {
        (Some(_), Some(_), Some(_)) => {
            Err("MoE source-dir and derived-GGUF lanes are mutually exclusive".to_owned())
        }
        (Some(_), Some(_), None) | (Some(_), None, Some(_)) => Err(
            "partial MoE derived-GGUF variables cannot be combined with source directory"
                .to_owned(),
        ),
        (None, Some(path), Some(lock_path)) => {
            let lock = read_derived_gguf_lock(&lock_path)
                .map_err(|error| format!("MoE derived GGUF lock failed: {error}"))?;
            let verified = verify_derived_gguf(lock, &path)
                .map_err(|error| format!("MoE derived GGUF identity failed: {error}"))?;
            verify_gguf_gemma4_moe(verified)
                .map(GemmaMoeActualSource::Gguf)
                .map_err(|error| format!("canonical Gemma MoE GGUF verification failed: {error}"))
        }
        (_, Some(_), None) | (_, None, Some(_)) => {
            Err("both SLLM_PHASE81_GEMMA4_MOE_GGUF and _GGUF_LOCK are required".to_owned())
        }
        (source_dir, None, None) => {
            let root = source_dir.unwrap_or_else(|| PathBuf::from(GEMMA4_MOE_SOURCE_CACHE));
            verify_gemma4_moe_artifact(&root)
                .map(GemmaMoeActualSource::Safetensors)
                .map_err(|error| {
                    format!(
                        "Gemma MoE NVFP4 source verification failed at {}: {error}",
                        root.display()
                    )
                })
        }
    }
}

fn run_ministral3(contract: MinistralTargetContract) -> Result<(), String> {
    let device = ministral_device_index(contract)?;
    let lock_path = repository_path(MINISTRAL3_LOCK_RELATIVE_PATH);
    if !lock_path.is_file() {
        return Err(format!(
            "reviewed Ministral3 lock is missing: {}",
            lock_path.display()
        ));
    }
    let gguf_path = env::var_os("SLLM_PHASE81_MINISTRAL3_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(MINISTRAL3_GGUF_CACHE));
    let verified = open_and_verify_official_ministral3_gguf(&gguf_path)
        .map_err(|error| format!("verify official Ministral3 GGUF: {error}"))?;
    let source = Arc::new(
        VerifiedMinistral3WeightSource::from_verified_gguf(verified)
            .map_err(|error| format!("bind Ministral3 weight source: {error}"))?,
    );
    if source.repository() != MINISTRAL3_OFFICIAL_GGUF_REPOSITORY
        || source.revision() != MINISTRAL3_OFFICIAL_GGUF_REVISION
        || source.file_sha256() != MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256
        || source.lock_fingerprint() != MINISTRAL3_WEIGHT_LOCK_FINGERPRINT
        || source.resident_bytes() != MINISTRAL3_WEIGHT_RESIDENT_BYTES
    {
        return Err("reviewed Ministral3 artifact identity differs".to_owned());
    }
    let plan = build_ministral3_weight_load_plan(source.as_ref())
        .map_err(|error| format!("build Ministral3 weight plan: {error}"))?;
    if plan.total_destination_bytes != MINISTRAL3_WEIGHT_RESIDENT_BYTES {
        return Err(format!(
            "Ministral3 resident plan bytes differ: {}",
            plan.total_destination_bytes
        ));
    }

    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device, contract.target.to_owned())
                .map_err(|error| format!("invalid execution request: {error}"))?,
        )
        .map_err(|error| format!("Ministral3 HIP session open failed: {error}"))?;
    let result = (|| -> Result<(), String> {
        let resident = Ministral3ResidentModel::new_gguf(
            Arc::clone(&session),
            plan,
            Arc::clone(&source),
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("Ministral3 resident load failed: {error}"))?;
        if resident.resident_bytes() != MINISTRAL3_WEIGHT_RESIDENT_BYTES {
            return Err(format!(
                "Ministral3 resident bytes differ: {}",
                resident.resident_bytes()
            ));
        }

        let run_pair = |resident: &Ministral3ResidentModel| -> Result<(i32, i32), String> {
            let mut request = resident
                .new_request(
                    MINISTRAL3_PREFILL_TOKENS.len() as u64,
                    MINISTRAL3_STATE_CAPACITY,
                )
                .map_err(|error| format!("Ministral3 request failed: {error}"))?;
            let prefill_selector = selector(MINISTRAL3_GRAPH_VOCAB_SIZE, 0, SEED, 0, None)?;
            let prefill = request
                .prefill_with_device_selector(&MINISTRAL3_PREFILL_TOKENS, &prefill_selector)
                .map_err(|error| format!("Ministral3 selector prefill failed: {error}"))?;
            let first = check_ministral_output(&prefill, &prefill_selector, contract, None)?;
            if prefill.committed_length() != MINISTRAL3_PREFILL_TOKENS.len() as u64 {
                return Err("Ministral3 selector prefill committed length differs".to_owned());
            }
            let decode_selector = selector(MINISTRAL3_GRAPH_VOCAB_SIZE, 0, SEED, 1, None)?;
            let decode = request
                .decode_with_device_selector(first, &decode_selector)
                .map_err(|error| format!("Ministral3 selector decode failed: {error}"))?;
            let second = check_ministral_output(&decode, &decode_selector, contract, None)?;
            if decode.committed_length() != MINISTRAL3_STATE_CAPACITY {
                return Err("Ministral3 selector decode committed length differs".to_owned());
            }
            Ok((first, second))
        };
        let first = run_pair(&resident)?;
        let second = run_pair(&resident)?;
        if first != second {
            return Err(format!(
                "Ministral3 fixed K0 seed is not reproducible: {first:?} != {second:?}"
            ));
        }

        let mut masked = resident
            .new_request(
                MINISTRAL3_PREFILL_TOKENS.len() as u64,
                MINISTRAL3_STATE_CAPACITY,
            )
            .map_err(|error| format!("Ministral3 masked request failed: {error}"))?;
        let prefill_selector =
            selector(MINISTRAL3_GRAPH_VOCAB_SIZE, 0, SEED, 0, Some(FORCED_TOKEN))?;
        let prefill = masked
            .prefill_with_device_selector(&MINISTRAL3_PREFILL_TOKENS, &prefill_selector)
            .map_err(|error| format!("Ministral3 masked prefill failed: {error}"))?;
        let first =
            check_ministral_output(&prefill, &prefill_selector, contract, Some(FORCED_TOKEN))?;
        let decode_selector =
            selector(MINISTRAL3_GRAPH_VOCAB_SIZE, 0, SEED, 1, Some(FORCED_TOKEN))?;
        let decode = masked
            .decode_with_device_selector(first, &decode_selector)
            .map_err(|error| format!("Ministral3 masked decode failed: {error}"))?;
        check_ministral_output(&decode, &decode_selector, contract, Some(FORCED_TOKEN))?;
        Ok(())
    })();
    let memory = session.memory_snapshot();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("Ministral3 session shutdown failed: {error}"))?;
    result.and_then(|()| {
        if memory.current_bytes() != 0
            || memory.model_resident().current_bytes() != 0
            || memory.request_state().current_bytes() != 0
            || memory.workspace().current_bytes() != 0
            || memory.poisoned()
            || cleanup.retryable_cleanup != 0
            || cleanup.durable_quarantine != 0
        {
            return Err(format!(
                "Ministral3 cleanup is incomplete: memory={memory:?}, report={cleanup:?}"
            ));
        }
        Ok(())
    })
}

fn execute_gemma4_moe_selector<S: Gemma4MoeWeightSource + 'static>(
    contract: GemmaMoeTargetContract,
    device: u32,
    source: Arc<S>,
    plan: WeightLoadPlan,
    graph: Gemma4MoeGraph,
    layout: sllm_core::Gemma4MoeExecutionLayout,
) -> Result<(), String> {
    if graph.model_fingerprint() != GEMMA4_MOE_MODEL_FINGERPRINT
        || graph.token_count() != GEMMA4_MOE_PREFILL_TOKENS.len() as u64
        || graph.start_position() != 0
        || graph.expected_length() != GEMMA4_MOE_PREFILL_TOKENS.len() as u64
        || graph.state_capacity() != GEMMA4_MOE_STATE_CAPACITY
        || plan.total_destination_bytes != GEMMA4_MOE_TEXT_RESIDENT_BYTES
        || layout.resident_weight_bytes() != GEMMA4_MOE_TEXT_RESIDENT_BYTES
        || layout.token_count() != GEMMA4_MOE_PREFILL_TOKENS.len() as u64
    {
        return Err("Gemma MoE selector graph/layout/resident contract differs".to_owned());
    }

    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let session = Arc::new(
        backend
            .open_execution_session(
                ExecutionSessionRequest::new(device, contract.target.to_owned())
                    .map_err(|error| format!("invalid execution request: {error}"))?,
            )
            .map_err(|error| format!("Gemma MoE HIP session open failed: {error}"))?,
    );
    let result = (|| -> Result<(), String> {
        let available = session
            .available_memory_bytes()
            .map_err(|error| format!("available-memory query failed: {error}"))?
            .ok_or_else(|| "HIP session did not report available memory".to_owned())?;
        let required = plan
            .total_destination_bytes
            .checked_add(layout.workspace_bytes())
            .ok_or_else(|| "Gemma MoE resident/workspace requirement overflowed".to_owned())?;
        if required > available {
            return Err(format!(
                "{} has {available} available bytes but Gemma MoE resident+workspace require {required}",
                contract.target
            ));
        }
        let resident = Gemma4MoeResidentModel::provision(
            Arc::clone(&session),
            source,
            plan,
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("Gemma MoE resident load failed: {error}"))?;
        if resident.audit().resident_bytes() != GEMMA4_MOE_TEXT_RESIDENT_BYTES
            || resident.audit().expert_blob_allocations() == 0
            || resident.audit().individual_expert_allocations() != 0
        {
            return Err(format!(
                "Gemma MoE resident allocation audit differs: {:?}",
                resident.audit()
            ));
        }

        let run_pair = |resident: &Gemma4MoeResidentModel| -> Result<(i32, i32), String> {
            let mut request = resident
                .new_request(graph.clone())
                .map_err(|error| format!("Gemma MoE request failed: {error}"))?;
            let prefill_selector = selector(GEMMA4_MOE_VOCAB_SIZE, 64, SEED, 0, None)?;
            let prefill = request
                .prefill_with_device_selector(&GEMMA4_MOE_PREFILL_TOKENS, &prefill_selector)
                .map_err(|error| format!("Gemma MoE selector prefill failed: {error}"))?;
            let first = check_gemma_moe_output(&prefill, contract, None)?;
            let decode_selector = selector(GEMMA4_MOE_VOCAB_SIZE, 64, SEED, 1, None)?;
            let decode = request
                .decode_with_device_selector(first, &decode_selector)
                .map_err(|error| format!("Gemma MoE selector decode failed: {error}"))?;
            let second = check_gemma_moe_output(&decode, contract, None)?;
            if request.is_poisoned() || !request.transition_committed() {
                return Err("Gemma MoE selector request did not commit cleanly".to_owned());
            }
            Ok((first, second))
        };
        let first = run_pair(&resident)?;
        let second = run_pair(&resident)?;
        if first != second {
            return Err(format!(
                "Gemma MoE fixed seed is not reproducible: {first:?} != {second:?}"
            ));
        }

        let mut masked = resident
            .new_request(graph)
            .map_err(|error| format!("Gemma MoE masked request failed: {error}"))?;
        let prefill_selector = selector(GEMMA4_MOE_VOCAB_SIZE, 64, SEED, 0, Some(FORCED_TOKEN))?;
        let prefill = masked
            .prefill_with_device_selector(&GEMMA4_MOE_PREFILL_TOKENS, &prefill_selector)
            .map_err(|error| format!("Gemma MoE masked prefill failed: {error}"))?;
        let first = check_gemma_moe_output(&prefill, contract, Some(FORCED_TOKEN))?;
        let decode_selector = selector(GEMMA4_MOE_VOCAB_SIZE, 64, SEED, 1, Some(FORCED_TOKEN))?;
        let decode = masked
            .decode_with_device_selector(first, &decode_selector)
            .map_err(|error| format!("Gemma MoE masked decode failed: {error}"))?;
        check_gemma_moe_output(&decode, contract, Some(FORCED_TOKEN))?;
        if masked.is_poisoned() || !masked.transition_committed() {
            return Err("Gemma MoE masked selector request did not commit cleanly".to_owned());
        }
        Ok(())
    })();
    let memory = session.memory_snapshot();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("Gemma MoE session shutdown failed: {error}"))?;
    result.and_then(|()| {
        if memory.current_bytes() != 0
            || memory.model_resident().current_bytes() != 0
            || memory.request_state().current_bytes() != 0
            || memory.workspace().current_bytes() != 0
            || memory.poisoned()
            || cleanup.retryable_cleanup != 0
            || cleanup.durable_quarantine != 0
        {
            return Err(format!(
                "Gemma MoE cleanup is incomplete: memory={memory:?}, report={cleanup:?}"
            ));
        }
        Ok(())
    })
}

fn run_gemma4_moe(contract: GemmaMoeTargetContract) -> Result<(), String> {
    let device = gemma_moe_device_index(contract)?;
    match load_gemma4_moe_source()? {
        GemmaMoeActualSource::Safetensors(source) => {
            let source = Arc::new(source);
            let plan = build_gemma4_moe_resident_weight_load_plan(source.as_ref())
                .map_err(|error| format!("Gemma MoE source load plan failed: {error}"))?;
            let graph = build_gemma4_moe_graph(
                source.as_ref(),
                GEMMA4_MOE_PREFILL_TOKENS.len() as u64,
                0,
                GEMMA4_MOE_STATE_CAPACITY,
            )
            .map_err(|error| format!("Gemma MoE source graph failed: {error}"))?;
            let layout = build_gemma4_moe_execution_layout(&graph, &plan)
                .map_err(|error| format!("Gemma MoE source layout failed: {error}"))?;
            execute_gemma4_moe_selector(contract, device, source, plan, graph, layout)
        }
        GemmaMoeActualSource::Gguf(source) => {
            let source = Arc::new(source);
            let plan = build_gemma4_moe_resident_weight_load_plan(source.as_ref())
                .map_err(|error| format!("Gemma MoE GGUF load plan failed: {error}"))?;
            let graph = build_gemma4_moe_gguf_graph(
                source.as_ref(),
                GEMMA4_MOE_PREFILL_TOKENS.len() as u64,
                0,
                GEMMA4_MOE_STATE_CAPACITY,
            )
            .map_err(|error| format!("Gemma MoE GGUF graph failed: {error}"))?;
            let layout = build_gemma4_moe_execution_layout(&graph, &plan)
                .map_err(|error| format!("Gemma MoE GGUF layout failed: {error}"))?;
            execute_gemma4_moe_selector(contract, device, source, plan, graph, layout)
        }
    }
}

fn run_gemma(contract: TargetContract) -> Result<(), String> {
    let device = require_gate(contract)?;
    let mode = env::var("SLLM_PHASE81_GEMMA4_MODE").unwrap_or_else(|_| "bf16".to_owned());
    if !matches!(mode.as_str(), "bf16" | "gguf" | "nvfp4") {
        return Err(format!(
            "SLLM_PHASE81_GEMMA4_MODE must be bf16, gguf, or nvfp4, got {mode}"
        ));
    }
    let lock_bytes = fs::read(repository_path("docs/models/locks/gemma4-12b-it-bf16.json"))
        .map_err(|error| format!("Gemma lock read failed: {error}"))?;
    let lock = parse_gemma4_model_lock(&lock_bytes)
        .map_err(|error| format!("Gemma lock failed: {error}"))?;
    let cache = if mode == "bf16" {
        let cache_root = required_path("SLLM_PHASE81_GEMMA4_CACHE")?;
        Some(
            lock.verify_cache(&cache_root)
                .map_err(|error| format!("Gemma cache verification failed: {error}"))?,
        )
    } else {
        None
    };
    let backend = HipBackend::connect().map_err(|error| format!("HIP connect failed: {error}"))?;
    let session = Arc::new(
        backend
            .open_execution_session(
                ExecutionSessionRequest::new(device, contract.target.to_owned())
                    .map_err(|error| format!("invalid execution request: {error}"))?,
            )
            .map_err(|error| format!("HIP session open failed: {error}"))?,
    );
    let result = (|| -> Result<(), String> {
        let resident = match mode.as_str() {
            "bf16" => {
                let cache = cache
                    .as_ref()
                    .ok_or_else(|| "Gemma BF16 cache was not verified".to_owned())?;
                let plan = build_verified_gemma4_weight_load_plan(&lock, cache)
                    .map_err(|error| format!("Gemma BF16 weight plan failed: {error}"))?;
                Gemma4ResidentModel::new(
                    Arc::clone(&session),
                    lock.clone(),
                    plan,
                    cache,
                    COMPLETION_TIMEOUT,
                )
                .map_err(|error| format!("Gemma BF16 resident load failed: {error}"))?
            }
            "nvfp4" => {
                let artifact_root = required_path("SLLM_PHASE81_GEMMA4_NVFP4")?;
                let artifact = Arc::new(
                    verify_unsloth_gemma4_nvfp4(&artifact_root)
                        .map_err(|error| format!("Gemma NVFP4 verification failed: {error}"))?,
                );
                let plan = build_unsloth_gemma4_nvfp4_weight_load_plan(&lock, &artifact)
                    .map_err(|error| format!("Gemma NVFP4 weight plan failed: {error}"))?;
                Gemma4ResidentModel::new_quantized(
                    Arc::clone(&session),
                    lock.clone(),
                    plan,
                    artifact,
                    COMPLETION_TIMEOUT,
                )
                .map_err(|error| format!("Gemma NVFP4 resident load failed: {error}"))?
            }
            "gguf" => {
                let gguf_root = required_path("SLLM_PHASE81_GEMMA4_GGUF")?;
                let derived_lock = required_path("SLLM_PHASE81_GEMMA4_DERIVED_LOCK")?;
                let verified = verify_derived_gguf(
                    read_derived_gguf_lock(&derived_lock)
                        .map_err(|error| format!("Gemma derived lock failed: {error}"))?,
                    &gguf_root,
                )
                .map_err(|error| format!("Gemma GGUF verification failed: {error}"))?;
                let (source, plan) = build_verified_gguf_gemma_weight_load_plan(&lock, verified)
                    .map_err(|error| format!("Gemma GGUF weight plan failed: {error}"))?;
                Gemma4ResidentModel::new_gguf_quantized(
                    Arc::clone(&session),
                    lock.clone(),
                    plan,
                    Arc::new(source),
                    COMPLETION_TIMEOUT,
                )
                .map_err(|error| format!("Gemma GGUF resident load failed: {error}"))?
            }
            _ => unreachable!("Gemma mode was validated above"),
        };

        let run_pair = |resident: &Gemma4ResidentModel| -> Result<(Vec<i32>, Vec<i32>), String> {
            let mut request = resident
                .new_request(PREFILL_TOKENS.len() as u64, STATE_CAPACITY)
                .map_err(|error| format!("Gemma request failed: {error}"))?;
            let prefill_selector = selector(GEMMA4_VOCAB_SIZE as usize, 64, SEED, 0, None)?;
            let prefill = request
                .prefill_with_device_selector(&PREFILL_TOKENS, &prefill_selector)
                .map_err(|error| format!("Gemma selector prefill failed: {error}"))?;
            let first = check_gemma_output(&prefill, contract, None)?;
            let decode_selector = selector(GEMMA4_VOCAB_SIZE as usize, 64, SEED, 1, None)?;
            let decode = request
                .decode_with_device_selector(first, &decode_selector)
                .map_err(|error| format!("Gemma selector decode failed: {error}"))?;
            let second = check_gemma_output(&decode, contract, None)?;
            Ok((vec![first], vec![second]))
        };
        let first = run_pair(&resident)?;
        let second = run_pair(&resident)?;
        if first != second {
            return Err(format!(
                "Gemma fixed seed is not reproducible: {first:?} != {second:?}"
            ));
        }

        let mut masked = resident
            .new_request(PREFILL_TOKENS.len() as u64, STATE_CAPACITY)
            .map_err(|error| format!("Gemma masked request failed: {error}"))?;
        let prefill_selector =
            selector(GEMMA4_VOCAB_SIZE as usize, 64, SEED, 0, Some(FORCED_TOKEN))?;
        let prefill = masked
            .prefill_with_device_selector(&PREFILL_TOKENS, &prefill_selector)
            .map_err(|error| format!("Gemma masked prefill failed: {error}"))?;
        let first = check_gemma_output(&prefill, contract, Some(FORCED_TOKEN))?;
        let decode_selector =
            selector(GEMMA4_VOCAB_SIZE as usize, 64, SEED, 1, Some(FORCED_TOKEN))?;
        let decode = masked
            .decode_with_device_selector(first, &decode_selector)
            .map_err(|error| format!("Gemma masked decode failed: {error}"))?;
        check_gemma_output(&decode, contract, Some(FORCED_TOKEN))?;
        Ok(())
    })();
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("Gemma session shutdown failed: {error}"))?;
    let remaining = session.memory_snapshot().current_bytes();
    result.and_then(|()| {
        if remaining != 0 || cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err(format!(
                "Gemma cleanup is incomplete: remaining={remaining}, report={cleanup:?}"
            ));
        }
        Ok(())
    })
}

fn run(contract: TargetContract, model: &'static str) {
    let result = match model {
        "qwen35" => run_qwen(contract),
        "gemma4" => run_gemma(contract),
        _ => Err("unknown Phase 81 model".to_owned()),
    };
    if let Err(error) = result {
        panic!("Phase 81 {model} {} failed: {error}", contract.target);
    }
}

fn run_moe(contract: GemmaMoeTargetContract) {
    if let Err(error) = run_gemma4_moe(contract) {
        panic!("Phase 81 Gemma MoE {} failed: {error}", contract.target);
    }
}

fn run_ministral(contract: MinistralTargetContract) {
    if let Err(error) = run_ministral3(contract) {
        panic!("Phase 81 Ministral3 {} failed: {error}", contract.target);
    }
}

#[test]
#[ignore = "requires SLLM_PHASE81_FIXED_SAMPLING_GFX1201=1 and a verified Qwen cache"]
fn phase81_qwen35_fixed_sampling_gfx1201() {
    run(GFX1201, "qwen35");
}

#[test]
#[ignore = "requires SLLM_PHASE81_FIXED_SAMPLING_GFX1030=1 and a verified Qwen cache"]
fn phase81_qwen35_fixed_sampling_gfx1030() {
    run(GFX1030, "qwen35");
}

#[test]
#[ignore = "requires SLLM_PHASE81_FIXED_SAMPLING_GFX1201=1 and a verified Gemma cache"]
fn phase81_gemma4_fixed_sampling_gfx1201() {
    run(GFX1201, "gemma4");
}

#[test]
#[ignore = "requires SLLM_PHASE81_FIXED_SAMPLING_GFX1030=1 and a verified Gemma cache"]
fn phase81_gemma4_fixed_sampling_gfx1030() {
    run(GFX1030, "gemma4");
}

#[test]
#[ignore = "requires SLLM_PHASE81_GEMMA4_MOE_GFX1201=1 and the verified Gemma MoE NVFP4 artifact"]
fn phase81_gemma4_moe_fixed_sampling_gfx1201() {
    run_moe(GEMMA_MOE_GFX1201);
}

#[test]
#[ignore = "requires SLLM_PHASE81_GEMMA4_MOE_GFX1030=1 and the verified Gemma MoE NVFP4 artifact"]
fn phase81_gemma4_moe_fixed_sampling_gfx1030() {
    run_moe(GEMMA_MOE_GFX1030);
}

#[test]
#[ignore = "requires SLLM_PHASE81_MINISTRAL3_GFX1201=1 and the verified official BF16 GGUF"]
fn phase81_ministral3_k0_fixed_sampling_gfx1201() {
    run_ministral(MINISTRAL_GFX1201);
}

#[test]
#[ignore = "requires SLLM_PHASE81_MINISTRAL3_GFX1030=1 and the verified official BF16 GGUF"]
fn phase81_ministral3_k0_fixed_sampling_gfx1030() {
    run_ministral(MINISTRAL_GFX1030);
}
