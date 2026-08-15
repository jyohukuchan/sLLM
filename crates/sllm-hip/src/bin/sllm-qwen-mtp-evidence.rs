//! Full real-weight Qwen3.5 MTP execution evidence.

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, QwenComponentSelection, QwenExecutionRequest,
    QwenResidentModel, build_qwen35_graph, build_qwen35_mtp_graph,
    build_verified_qwen_component_weight_load_plan, build_verified_weight_load_plan,
    read_model_lock, verify_greedy,
};
use sllm_hip::HipBackend;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(180);
const GUARD: &str = "SLLM_QWEN_MTP_GPU_EXECUTION";
const PROMPT: [i32; 3] = [248_045, 9707, 248_046];

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    lock_fingerprint: String,
    text_plan_digest: String,
    mtp_plan_digest: String,
    prompt_tokens: Vec<i32>,
    target_first: i32,
    draft_tokens: Vec<u32>,
    target_verify: Vec<u32>,
    accepted_draft_tokens: usize,
    emitted_tokens: Vec<u32>,
    deterministic_replay: bool,
    target_hidden_sha256: String,
    mtp_hidden_sha256: String,
    target_dispatches: u64,
    mtp_dispatches: u64,
    fallback_used: bool,
    elapsed_ms: u128,
    cleanup_empty: bool,
}

fn hash_words(words: &[u16]) -> String {
    let mut hash = Sha256::new();
    for word in words {
        hash.update(word.to_le_bytes());
    }
    format!("sha256:{:x}", hash.finalize())
}

fn run_mtp_prefix(
    resident: &QwenResidentModel,
    lock: &sllm_core::ModelLock,
    plan: &sllm_core::WeightLoadPlan,
    hidden: &[u16],
    target_first: i32,
    capacity: u64,
) -> Result<(QwenExecutionRequest, Vec<u32>, Vec<u16>), String> {
    let graph = build_qwen35_mtp_graph(lock, plan, capacity).map_err(|error| error.to_string())?;
    let mut request = resident
        .new_request(graph)
        .map_err(|error| error.to_string())?;
    let zero = vec![0_u16; 2_560];
    request
        .prefill_mtp(PROMPT[0], &zero)
        .map_err(|error| error.to_string())?;
    request
        .decode_mtp(PROMPT[1], &hidden[..2_560])
        .map_err(|error| error.to_string())?;
    request
        .decode_mtp(PROMPT[2], &hidden[2_560..5_120])
        .map_err(|error| error.to_string())?;
    let first = request
        .decode_mtp(target_first, &hidden[5_120..7_680])
        .map_err(|error| error.to_string())?;
    let first_token = u32::try_from(first.token_ids()[0]).map_err(|_| "negative draft token")?;
    let first_hidden = first
        .hidden_states_bf16()
        .ok_or("MTP first hidden row is absent")?
        .to_vec();
    let second = request
        .decode_mtp(
            i32::try_from(first_token).map_err(|_| "draft token overflow")?,
            &first_hidden,
        )
        .map_err(|error| error.to_string())?;
    let second_token = u32::try_from(second.token_ids()[0]).map_err(|_| "negative draft token")?;
    let second_hidden = second
        .hidden_states_bf16()
        .ok_or("MTP second hidden row is absent")?
        .to_vec();
    Ok((request, vec![first_token, second_token], second_hidden))
}

fn run() -> Result<Report, String> {
    if env::var(GUARD).as_deref() != Ok("1") {
        return Err(format!("{GUARD}=1 is required"));
    }
    let mut args = env::args().skip(1);
    let device_index = args
        .next()
        .ok_or("device index is required")?
        .parse::<u32>()
        .map_err(|_| "device index must be U32")?;
    let target = args.next().ok_or("target is required")?;
    let lock_path = PathBuf::from(args.next().ok_or("lock path is required")?);
    let cache_path = PathBuf::from(args.next().ok_or("cache path is required")?);
    if args.next().is_some() || !matches!(target.as_str(), "gfx1030" | "gfx1201" | "gfx942") {
        return Err("usage: DEVICE TARGET LOCK CACHE".to_owned());
    }
    let started = Instant::now();
    let lock = read_model_lock(lock_path).map_err(|error| error.to_string())?;
    let cache = Arc::new(
        lock.verify_cache(cache_path)
            .map_err(|error| error.to_string())?,
    );
    let text_plan =
        build_verified_weight_load_plan(&lock, &cache).map_err(|error| error.to_string())?;
    let mtp_plan = build_verified_qwen_component_weight_load_plan(
        &lock,
        &cache,
        QwenComponentSelection::MTP_ONLY,
    )
    .map_err(|error| error.to_string())?;
    let capacity = 17;
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let text_resident_graph =
        build_qwen35_graph(&lock, &text_plan, 1, capacity).map_err(|error| error.to_string())?;
    let text_resident = QwenResidentModel::new(
        Arc::clone(&session),
        text_resident_graph,
        text_plan.clone(),
        Arc::clone(&cache),
        TIMEOUT,
    )
    .map_err(|error| error.to_string())?;
    let mtp_resident_graph =
        build_qwen35_mtp_graph(&lock, &mtp_plan, capacity).map_err(|error| error.to_string())?;
    let mtp_resident = QwenResidentModel::new(
        Arc::clone(&session),
        mtp_resident_graph,
        mtp_plan.clone(),
        Arc::clone(&cache),
        TIMEOUT,
    )
    .map_err(|error| error.to_string())?;

    let text_graph = build_qwen35_graph(&lock, &text_plan, PROMPT.len() as u64, capacity)
        .map_err(|error| error.to_string())?;
    let mut target_request = text_resident
        .new_request(text_graph)
        .map_err(|error| error.to_string())?;
    let target_prefill = target_request
        .prefill_with_mtp_state(&PROMPT)
        .map_err(|error| error.to_string())?;
    let target_first = target_prefill.token_ids()[PROMPT.len() - 1];
    let target_hidden = target_prefill
        .hidden_states_bf16()
        .ok_or("target hidden rows are absent")?
        .to_vec();
    if target_hidden.len() != PROMPT.len() * 2_560 {
        return Err("target hidden row count differs".to_owned());
    }

    let (mtp_request, drafts, last_mtp_hidden) = run_mtp_prefix(
        &mtp_resident,
        &lock,
        &mtp_plan,
        &target_hidden,
        target_first,
        capacity,
    )?;
    let (_, replay, _) = run_mtp_prefix(
        &mtp_resident,
        &lock,
        &mtp_plan,
        &target_hidden,
        target_first,
        capacity,
    )?;
    let deterministic_replay = replay == drafts;
    if !deterministic_replay {
        return Err("MTP deterministic replay changed draft tokens".to_owned());
    }

    let mut verify = Vec::with_capacity(3);
    let first_verify = target_request
        .decode_with_mtp_state(target_first)
        .map_err(|error| error.to_string())?;
    verify.push(u32::try_from(first_verify.token_ids()[0]).map_err(|_| "negative target token")?);
    for draft in &drafts {
        let output = target_request
            .decode_with_mtp_state(i32::try_from(*draft).map_err(|_| "draft token overflow")?)
            .map_err(|error| error.to_string())?;
        verify.push(u32::try_from(output.token_ids()[0]).map_err(|_| "negative target token")?);
    }
    let decision = verify_greedy(&drafts, &verify).map_err(|error| error.to_string())?;
    let target_audit = target_request
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    let mtp_audit = mtp_request
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    let fallback_used = target_audit.fallback_used() || mtp_audit.fallback_used();
    if fallback_used || !target_audit.all_dispatches_hip() || !mtp_audit.all_dispatches_hip() {
        return Err("MTP evidence observed fallback or a non-HIP dispatch".to_owned());
    }
    drop(target_request);
    drop(mtp_request);
    drop(mtp_resident);
    drop(text_resident);
    let cleanup = session
        .shutdown(Duration::from_secs(30))
        .map_err(|error| error.to_string())?;
    let cleanup_empty = cleanup.retryable_cleanup == 0 && cleanup.durable_quarantine == 0;
    if !cleanup_empty {
        return Err("MTP evidence cleanup was not empty".to_owned());
    }
    Ok(Report {
        schema_version: "qwen35-mtp-gpu-evidence-v1",
        state: "PASS",
        target,
        device_index,
        lock_fingerprint: lock.fingerprint().to_owned(),
        text_plan_digest: text_plan.digest_hex(),
        mtp_plan_digest: mtp_plan.digest_hex(),
        prompt_tokens: PROMPT.to_vec(),
        target_first,
        draft_tokens: drafts,
        target_verify: verify,
        accepted_draft_tokens: decision.accepted_draft_tokens(),
        emitted_tokens: decision.emitted_tokens().to_vec(),
        deterministic_replay,
        target_hidden_sha256: hash_words(&target_hidden),
        mtp_hidden_sha256: hash_words(&last_mtp_hidden),
        target_dispatches: target_audit.kernel_dispatch_count(),
        mtp_dispatches: mtp_audit.kernel_dispatch_count(),
        fallback_used,
        elapsed_ms: started.elapsed().as_millis(),
        cleanup_empty,
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
                eprintln!("MTP evidence serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("MTP evidence failed: {error}");
            ExitCode::from(2)
        }
    }
}
