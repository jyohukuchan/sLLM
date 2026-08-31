//! Full-model Qwen3.5-4B BF16 adapter/control execution evidence.
//!
//! This runner deliberately builds the adapter payloads in memory.  The model
//! remains the already-reviewed derived GGUF supplied by the caller; no model
//! or synthetic payload is written to the repository or to an evidence file.

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    AdapterModelDimsV1, AdapterRequestSetV1, AllocationSnapshot, Backend, ControlVectorLockV1,
    ControlVectorSelectionV1, ExecutionSession, ExecutionSessionRequest, KvCacheEncoding,
    LoraAdapterLockV1, LoraAdapterSelectionV1, LoraTargetLockV1, QWEN35_VOCAB_SIZE, QwenGraph,
    QwenResidentModel, ReviewedModelLock, VerifiedControlVectorPayloadV1, VerifiedLoraPayloadV1,
    WeightLoadPlan, build_qwen35_graph_with_kv_cache_encoding,
    build_verified_gguf_qwen_weight_load_plan, builtin_reviewed_model_lock, read_derived_gguf_lock,
    verify_derived_gguf,
};
use sllm_hip::HipBackend;
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(180);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const EXECUTION_GUARD: &str = "SLLM_QWEN_ADAPTER_GPU_EXECUTION";
const PROMPT: [i32; 3] = [248_045, 9_707, 248_046];
const GRAPH_ROWS: u64 = PROMPT.len() as u64;
const STATE_CAPACITY: u64 = 17;
const HIDDEN_SIZE: u64 = 2_560;
const LAYER_COUNT: u64 = 32;
const ADAPTER_BF16: u16 = 0x3c80; // 0.015625, nonzero and deliberately small.
const LORA_TARGET: &str = "model.language_model.layers.0.mlp.gate_proj.weight";

#[derive(Serialize)]
struct CaseReport {
    identity: String,
    first_token_sha256: String,
    second_token_sha256: String,
    first_logits_sha256: String,
    second_logits_sha256: String,
    repeatable: bool,
    selected_backend: &'static str,
    target: String,
    fallback_used: bool,
    all_dispatches_hip: bool,
    first_dispatches: u64,
    second_dispatches: u64,
    differs_from_disabled: bool,
}

#[derive(Serialize)]
struct CleanupReport {
    model_ready_current_bytes: u64,
    pre_shutdown_current_bytes: u64,
    final_current_bytes: u64,
    final_request_state_bytes: u64,
    final_workspace_bytes: u64,
    retryable_cleanup: usize,
    durable_quarantine: usize,
    empty: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    model_lock_fingerprint: String,
    derived_gguf_sha256: String,
    weight_plan_digest: String,
    prompt_sha256: String,
    lora_identity: String,
    control_identity: String,
    cases: Vec<CaseReport>,
    enabled_differs_from_disabled: bool,
    cleanup: CleanupReport,
    elapsed_ms: u128,
}

#[derive(Clone)]
struct OneRun {
    token_sha256: String,
    logits_sha256: String,
    selected_backend: &'static str,
    target: String,
    fallback_used: bool,
    all_dispatches_hip: bool,
    dispatches: u64,
}

type SyntheticAdapterSets = (
    Arc<VerifiedLoraPayloadV1>,
    Arc<VerifiedControlVectorPayloadV1>,
    AdapterRequestSetV1,
    AdapterRequestSetV1,
    AdapterRequestSetV1,
);

struct CaseContext<'a> {
    resident: &'a QwenResidentModel,
    graph: &'a QwenGraph,
    session: &'a ExecutionSession,
    baseline: AllocationSnapshot,
    expected_target: &'a str,
}

fn sha256_bytes(bytes: impl AsRef<[u8]>) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes.as_ref()))
}

fn hash_tokens(tokens: &[i32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

fn hash_logits(logits: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in logits {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

fn bf16_payload(words: usize) -> Vec<u8> {
    let mut payload = Vec::with_capacity(words * 2);
    for _ in 0..words {
        payload.extend_from_slice(&ADAPTER_BF16.to_le_bytes());
    }
    payload
}

fn build_synthetic_adapters(
    lock_fingerprint: &str,
    plan: &WeightLoadPlan,
) -> Result<SyntheticAdapterSets, String> {
    let target_entry = plan
        .entries
        .iter()
        .find(|entry| entry.tensor_name == LORA_TARGET)
        .ok_or_else(|| format!("required LoRA target is absent: {LORA_TARGET}"))?;
    if target_entry.shape != [9_216, 2_560] || target_entry.dtype != sllm_core::TensorDType::Bf16 {
        return Err(format!(
            "LoRA target has unexpected dtype/shape: {:?} {:?}",
            target_entry.dtype, target_entry.shape
        ));
    }

    let rank = 1_u64;
    let a_size = 2_560_u64 * rank * 2;
    let b_size = 9_216_u64 * rank * 2;
    let lora_payload = bf16_payload(
        usize::try_from((a_size + b_size) / 2)
            .map_err(|_| "synthetic LoRA payload size exceeds host usize".to_owned())?,
    );
    let lora_lock = LoraAdapterLockV1 {
        schema_version: "sllm-adapter-lock-v1".to_owned(),
        kind: "lora".to_owned(),
        artifact_id: "phase45-synthetic-lora".to_owned(),
        alpha: 1.0,
        base_model_fingerprint: lock_fingerprint.to_owned(),
        base_weight_plan_digest: plan.digest_hex(),
        payload_sha256: sha256_bytes(&lora_payload),
        payload_size: u64::try_from(lora_payload.len())
            .map_err(|_| "synthetic LoRA payload size exceeds u64".to_owned())?,
        targets: vec![LoraTargetLockV1 {
            tensor_name: LORA_TARGET.to_owned(),
            dtype: "BF16".to_owned(),
            target_shape: vec![9_216, 2_560],
            rank,
            a_offset: 0,
            a_size,
            b_offset: a_size,
            b_size,
        }],
    };
    let lora_lock_json = serde_json::to_vec(&lora_lock)
        .map_err(|error| format!("serialize synthetic LoRA lock: {error}"))?;
    let lora = Arc::new(
        VerifiedLoraPayloadV1::from_bytes(&lora_lock_json, lora_payload, lock_fingerprint, plan)
            .map_err(|error| format!("verify synthetic LoRA: {error}"))?,
    );

    let control_payload = bf16_payload(
        usize::try_from(HIDDEN_SIZE * 2 / 2)
            .map_err(|_| "synthetic control payload size exceeds host usize".to_owned())?,
    );
    let control_lock = ControlVectorLockV1 {
        schema_version: "sllm-adapter-lock-v1".to_owned(),
        kind: "control-vector".to_owned(),
        artifact_id: "phase45-synthetic-control".to_owned(),
        dtype: "bf16".to_owned(),
        base_model_fingerprint: lock_fingerprint.to_owned(),
        base_weight_plan_digest: plan.digest_hex(),
        payload_sha256: sha256_bytes(&control_payload),
        payload_size: u64::try_from(control_payload.len())
            .map_err(|_| "synthetic control payload size exceeds u64".to_owned())?,
        hidden_size: HIDDEN_SIZE,
        layer_start: 0,
        layer_end: 1,
        vector_offset: 0,
        vector_size: HIDDEN_SIZE * 2,
    };
    let control_lock_json = serde_json::to_vec(&control_lock)
        .map_err(|error| format!("serialize synthetic control lock: {error}"))?;
    let dims = AdapterModelDimsV1::new(HIDDEN_SIZE, LAYER_COUNT)
        .map_err(|error| format!("construct adapter dimensions: {error}"))?;
    let control = Arc::new(
        VerifiedControlVectorPayloadV1::from_bytes(
            &control_lock_json,
            control_payload,
            lock_fingerprint,
            plan,
            dims,
        )
        .map_err(|error| format!("verify synthetic control vector: {error}"))?,
    );

    let lora_set = AdapterRequestSetV1::new(
        vec![LoraAdapterSelectionV1 {
            alias: "lora-a".to_owned(),
            artifact: Arc::clone(&lora),
            scale: 1.0,
        }],
        Vec::new(),
    )
    .map_err(|error| format!("construct LoRA request set: {error}"))?;
    let control_set = AdapterRequestSetV1::new(
        Vec::new(),
        vec![ControlVectorSelectionV1 {
            alias: "control-a".to_owned(),
            artifact: Arc::clone(&control),
            scale: 1.0,
        }],
    )
    .map_err(|error| format!("construct control request set: {error}"))?;
    let combined_set = AdapterRequestSetV1::new(
        vec![LoraAdapterSelectionV1 {
            alias: "lora-a".to_owned(),
            artifact: Arc::clone(&lora),
            scale: 1.0,
        }],
        vec![ControlVectorSelectionV1 {
            alias: "control-a".to_owned(),
            artifact: Arc::clone(&control),
            scale: 1.0,
        }],
    )
    .map_err(|error| format!("construct combined request set: {error}"))?;
    Ok((lora, control, lora_set, control_set, combined_set))
}

fn check_baseline(session: &ExecutionSession, baseline: AllocationSnapshot) -> Result<(), String> {
    let current = session.memory_snapshot();
    if current.poisoned()
        || current.model_resident().current_bytes() != baseline.model_resident().current_bytes()
        || current.request_state().current_bytes() != 0
        || current.workspace().current_bytes() != 0
        || current.current_bytes() != baseline.current_bytes()
    {
        return Err(format!(
            "request allocation baseline changed: baseline={baseline:?}, current={current:?}"
        ));
    }
    Ok(())
}

fn execute_once(
    resident: &QwenResidentModel,
    graph: &QwenGraph,
    adapters: &AdapterRequestSetV1,
    session: &ExecutionSession,
    baseline: AllocationSnapshot,
    expected_target: &str,
) -> Result<OneRun, String> {
    let mut request = resident
        .new_request_with_adapters(graph.clone(), adapters.clone())
        .map_err(|error| format!("create adapter request: {error}"))?;
    let output = request
        .prefill_with_last_logits(&PROMPT)
        .map_err(|error| format!("adapter prefill: {error}"))?;
    let logits = output
        .last_logits()
        .ok_or("prefill did not publish last-token logits")?;
    if logits.len() != QWEN35_VOCAB_SIZE || logits.iter().any(|value| !value.is_finite()) {
        return Err("last-token logits are not a finite full-vocabulary row".to_owned());
    }
    if output.token_ids().is_empty() {
        return Err("prefill did not publish a token result".to_owned());
    }
    let audit = request
        .audit_snapshot()
        .map_err(|error| format!("read adapter dispatch audit: {error}"))?;
    if audit.selected_backend() != "hip"
        || audit.target() != expected_target
        || audit.fallback_used()
        || !audit.all_dispatches_hip()
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
    {
        return Err(format!("adapter dispatch was not HIP-only: {audit:?}"));
    }
    let result = OneRun {
        token_sha256: hash_tokens(output.token_ids()),
        logits_sha256: hash_logits(logits),
        selected_backend: audit.selected_backend(),
        target: audit.target().to_owned(),
        fallback_used: audit.fallback_used(),
        all_dispatches_hip: audit.all_dispatches_hip(),
        dispatches: audit.kernel_dispatch_count(),
    };
    drop(request);
    check_baseline(session, baseline)?;
    Ok(result)
}

fn run_case(
    name: &str,
    adapters: &AdapterRequestSetV1,
    context: &CaseContext<'_>,
    disabled_logits_sha256: Option<&str>,
) -> Result<CaseReport, String> {
    let first = execute_once(
        context.resident,
        context.graph,
        adapters,
        context.session,
        context.baseline,
        context.expected_target,
    )?;
    let second = execute_once(
        context.resident,
        context.graph,
        adapters,
        context.session,
        context.baseline,
        context.expected_target,
    )?;
    let repeatable =
        first.token_sha256 == second.token_sha256 && first.logits_sha256 == second.logits_sha256;
    if !repeatable {
        return Err(format!("{name} replay changed token or logit hash"));
    }
    let differs_from_disabled =
        disabled_logits_sha256.is_none_or(|disabled| first.logits_sha256 != disabled);
    if disabled_logits_sha256.is_some() && !differs_from_disabled {
        return Err(format!("{name} logits are identical to disabled identity"));
    }
    Ok(CaseReport {
        identity: adapters.identity().to_owned(),
        first_token_sha256: first.token_sha256,
        second_token_sha256: second.token_sha256,
        first_logits_sha256: first.logits_sha256,
        second_logits_sha256: second.logits_sha256,
        repeatable,
        selected_backend: first.selected_backend,
        target: first.target,
        fallback_used: first.fallback_used || second.fallback_used,
        all_dispatches_hip: first.all_dispatches_hip && second.all_dispatches_hip,
        first_dispatches: first.dispatches,
        second_dispatches: second.dispatches,
        differs_from_disabled,
    })
}

fn required_path(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<PathBuf, String> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{name} path is required"))
}

fn run() -> Result<Report, String> {
    if env::var(EXECUTION_GUARD).as_deref() != Ok("1") {
        return Err(format!("{EXECUTION_GUARD}=1 is required"));
    }
    let mut arguments = env::args().skip(1);
    let device_index = arguments
        .next()
        .ok_or("device index is required")?
        .parse::<u32>()
        .map_err(|_| "device index must be U32".to_owned())?;
    let target = arguments.next().ok_or("target is required")?;
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("target must be gfx1030 or gfx1201".to_owned());
    }
    let gguf_path = required_path(&mut arguments, "GGUF")?;
    let derived_lock_path = required_path(&mut arguments, "derived lock")?;
    if arguments.next().is_some() {
        return Err("usage: DEVICE_INDEX TARGET DERIVED_GGUF DERIVED_LOCK".to_owned());
    }
    if !Path::new(&gguf_path).is_file() || !Path::new(&derived_lock_path).is_file() {
        return Err("GGUF and derived-lock paths must be regular files".to_owned());
    }
    let started = Instant::now();

    let derived = read_derived_gguf_lock(&derived_lock_path)
        .map_err(|error| format!("read derived GGUF lock: {error}"))?;
    let reviewed = builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
        .map_err(|error| format!("resolve reviewed model lock: {error}"))?;
    let lock = match reviewed {
        ReviewedModelLock::Qwen35(lock) => lock,
        ReviewedModelLock::Gemma4(_) | ReviewedModelLock::Ministral3(_) => {
            return Err("derived artifact is not reviewed Qwen3.5".to_owned());
        }
    };
    if lock.fingerprint() != sllm_core::QWEN35_4B_FINGERPRINT {
        return Err("adapter smoke requires the reviewed Qwen3.5-4B lock".to_owned());
    }
    let verified = verify_derived_gguf(derived, &gguf_path)
        .map_err(|error| format!("verify derived GGUF: {error}"))?;
    let derived_gguf_sha256 = verified.lock.output.sha256.clone();
    let (source, plan) = build_verified_gguf_qwen_weight_load_plan(
        &lock,
        verified,
        sllm_core::QwenComponentSelection::TEXT_ONLY,
    )
    .map_err(|error| format!("build verified Qwen GGUF load plan: {error}"))?;
    if source.has_fp8_recipe() {
        return Err("adapter smoke requires the unquantized BF16 GGUF".to_owned());
    }
    let (lora, control, lora_set, control_set, combined_set) =
        build_synthetic_adapters(lock.fingerprint(), &plan)?;
    let disabled_set = AdapterRequestSetV1::disabled();

    let backend = HipBackend::connect().map_err(|error| format!("connect HIP backend: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| format!("build execution session request: {error}"))?,
        )
        .map_err(|error| format!("open HIP execution session: {error}"))?;
    let resident_graph = build_qwen35_graph_with_kv_cache_encoding(
        &lock,
        &plan,
        GRAPH_ROWS,
        STATE_CAPACITY,
        KvCacheEncoding::Fp16,
    )
    .map_err(|error| format!("build Qwen BF16 graph: {error}"))?;
    let request_graph = resident_graph.clone();
    let resident = QwenResidentModel::new_gguf(
        Arc::clone(&session),
        resident_graph,
        plan.clone(),
        Arc::new(source),
        COMPLETION_TIMEOUT,
    )
    .map_err(|error| format!("provision Qwen BF16 resident model: {error}"))?;
    let baseline = session.memory_snapshot();
    if baseline.poisoned()
        || baseline.model_resident().current_bytes() == 0
        || baseline.request_state().current_bytes() != 0
        || baseline.workspace().current_bytes() != 0
        || baseline.current_bytes() != baseline.model_resident().current_bytes()
    {
        return Err(format!(
            "invalid model-ready allocation baseline: {baseline:?}"
        ));
    }

    let case_context = CaseContext {
        resident: &resident,
        graph: &request_graph,
        session: &session,
        baseline,
        expected_target: &target,
    };
    let disabled = run_case("disabled", &disabled_set, &case_context, None)?;
    let lora_case = run_case(
        "lora",
        &lora_set,
        &case_context,
        Some(&disabled.first_logits_sha256),
    )?;
    let control_case = run_case(
        "control",
        &control_set,
        &case_context,
        Some(&disabled.first_logits_sha256),
    )?;
    let combined_case = run_case(
        "combined",
        &combined_set,
        &case_context,
        Some(&disabled.first_logits_sha256),
    )?;
    let enabled_differs_from_disabled = lora_case.differs_from_disabled
        && control_case.differs_from_disabled
        && combined_case.differs_from_disabled;
    if !enabled_differs_from_disabled {
        return Err("one or more enabled adapter identities matched disabled logits".to_owned());
    }
    if lora_case.first_dispatches <= disabled.first_dispatches
        || control_case.first_dispatches <= disabled.first_dispatches
        || combined_case.first_dispatches <= disabled.first_dispatches
    {
        return Err("enabled adapter execution did not add HIP dispatches".to_owned());
    }
    check_baseline(&session, baseline)?;
    drop(resident);
    let pre_shutdown = session.memory_snapshot();
    if pre_shutdown.current_bytes() != 0 || pre_shutdown.poisoned() {
        return Err(format!(
            "resident model did not release before shutdown: {pre_shutdown:?}"
        ));
    }
    let shutdown = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("shutdown HIP session: {error}"))?;
    let final_snapshot = session.memory_snapshot();
    let cleanup_empty = shutdown.retryable_cleanup == 0
        && shutdown.durable_quarantine == 0
        && final_snapshot.current_bytes() == 0
        && final_snapshot.request_state().current_bytes() == 0
        && final_snapshot.workspace().current_bytes() == 0
        && !final_snapshot.poisoned();
    if !cleanup_empty {
        return Err(format!(
            "adapter evidence cleanup was not empty: shutdown={shutdown:?}, final={final_snapshot:?}"
        ));
    }

    Ok(Report {
        schema_version: "qwen35-adapter-gpu-evidence-v1",
        state: "PASS",
        target,
        device_index,
        model_lock_fingerprint: lock.fingerprint().to_owned(),
        derived_gguf_sha256,
        weight_plan_digest: plan.digest_hex(),
        prompt_sha256: hash_tokens(&PROMPT),
        lora_identity: lora.identity().canonical_string(),
        control_identity: control.identity().canonical_string(),
        cases: vec![disabled, lora_case, control_case, combined_case],
        enabled_differs_from_disabled,
        cleanup: CleanupReport {
            model_ready_current_bytes: baseline.model_resident().current_bytes(),
            pre_shutdown_current_bytes: pre_shutdown.current_bytes(),
            final_current_bytes: final_snapshot.current_bytes(),
            final_request_state_bytes: final_snapshot.request_state().current_bytes(),
            final_workspace_bytes: final_snapshot.workspace().current_bytes(),
            retryable_cleanup: shutdown.retryable_cleanup,
            durable_quarantine: shutdown.durable_quarantine,
            empty: cleanup_empty,
        },
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(value) => {
                println!("{value}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("adapter evidence serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("adapter evidence failed: {error}");
            ExitCode::from(2)
        }
    }
}
