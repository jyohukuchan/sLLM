//! Full real-weight Qwen3.5 MTP execution evidence.

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, QwenComponentSelection, QwenExecutionRequest,
    QwenResidentModel, build_qwen35_fp8_graph, build_qwen35_graph, build_qwen35_mtp_graph,
    build_verified_qwen_component_weight_load_plan, build_verified_weight_load_plan,
    read_model_lock, verify_fp8_sidecar, verify_greedy,
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
    weight_encoding: &'static str,
    kv_encoding: &'static str,
    draft_width: usize,
    target_block_rows: usize,
    target_first: i32,
    draft_tokens: Vec<u32>,
    target_verify: Vec<u32>,
    target_block_verify: Vec<u32>,
    target_block_matches_sequential: bool,
    target_block_logits_match_sequential: bool,
    target_block_hidden_matches_sequential: bool,
    partial_prefix_replay_matches_sequential: bool,
    committed_kv_matches_sequential: bool,
    accepted_draft_tokens: usize,
    emitted_tokens: Vec<u32>,
    deterministic_replay: bool,
    target_hidden_sha256: String,
    target_logits_sha256: String,
    committed_kv_sha256: String,
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

fn hash_kv(payloads: &[sllm_core::QwenKvPayloadEvidence]) -> String {
    let mut hash = Sha256::new();
    for (layer, key, value) in payloads {
        hash.update(layer.to_le_bytes());
        hash.update((key.len() as u64).to_le_bytes());
        hash.update(key);
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value);
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
    draft_width: usize,
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
    let mut proposal_token = target_first;
    let mut proposal_hidden = hidden[5_120..7_680].to_vec();
    let mut drafts = Vec::with_capacity(draft_width);
    for _ in 0..draft_width {
        let proposal = request
            .decode_mtp(proposal_token, &proposal_hidden)
            .map_err(|error| error.to_string())?;
        proposal_token = proposal.token_ids()[0];
        drafts.push(u32::try_from(proposal_token).map_err(|_| "negative draft token")?);
        proposal_hidden = proposal
            .hidden_states_bf16()
            .ok_or("MTP proposal hidden row is absent")?
            .to_vec();
    }
    Ok((request, drafts, proposal_hidden))
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
    let draft_width = args
        .next()
        .unwrap_or_else(|| "2".to_owned())
        .parse::<usize>()
        .map_err(|_| "draft width must be usize")?;
    let fp8_manifest = args.next().map(PathBuf::from);
    let fp8_artifact = args.next().map(PathBuf::from);
    if args.next().is_some()
        || !matches!(target.as_str(), "gfx1030" | "gfx1201" | "gfx942")
        || !(1..=7).contains(&draft_width)
        || fp8_manifest.is_some() != fp8_artifact.is_some()
    {
        return Err(
            "usage: DEVICE TARGET LOCK CACHE [DRAFT_WIDTH=1..7] [FP8_MANIFEST FP8_ARTIFACT]"
                .to_owned(),
        );
    }
    let started = Instant::now();
    let lock = read_model_lock(&lock_path).map_err(|error| error.to_string())?;
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
    let fp8_sidecar = fp8_manifest
        .zip(fp8_artifact)
        .map(|(manifest, artifact)| {
            verify_fp8_sidecar(&manifest, &artifact, &lock_path, &lock)
                .map(Arc::new)
                .map_err(|error| error.to_string())
        })
        .transpose()?;
    let capacity = 17;
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let graph_rows = (draft_width + 1).max(PROMPT.len()) as u64;
    let text_resident_graph = match &fp8_sidecar {
        Some(sidecar) => build_qwen35_fp8_graph(&lock, &text_plan, sidecar, graph_rows, capacity),
        None => build_qwen35_graph(&lock, &text_plan, graph_rows, capacity),
    }
    .map_err(|error| error.to_string())?;
    let text_resident = match &fp8_sidecar {
        Some(sidecar) => QwenResidentModel::new_fp8(
            Arc::clone(&session),
            text_resident_graph,
            text_plan.clone(),
            Arc::clone(&cache),
            Arc::clone(sidecar),
            TIMEOUT,
        ),
        None => QwenResidentModel::new(
            Arc::clone(&session),
            text_resident_graph,
            text_plan.clone(),
            Arc::clone(&cache),
            TIMEOUT,
        ),
    }
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

    let text_graph = match &fp8_sidecar {
        Some(sidecar) => build_qwen35_fp8_graph(&lock, &text_plan, sidecar, graph_rows, capacity),
        None => build_qwen35_graph(&lock, &text_plan, graph_rows, capacity),
    }
    .map_err(|error| error.to_string())?;
    let mut target_request = text_resident
        .new_request(text_graph.clone())
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
        draft_width,
    )?;
    let (_, replay, _) = run_mtp_prefix(
        &mtp_resident,
        &lock,
        &mtp_plan,
        &target_hidden,
        target_first,
        capacity,
        draft_width,
    )?;
    let deterministic_replay = replay == drafts;
    if !deterministic_replay {
        return Err("MTP deterministic replay changed draft tokens".to_owned());
    }

    let mut verify = Vec::with_capacity(3);
    let mut sequential_hidden = Vec::with_capacity(3 * 2_560);
    let mut sequential_logits = Vec::with_capacity(
        (draft_width + 1)
            .checked_mul(sllm_core::QWEN35_VOCAB_SIZE)
            .ok_or("target logits capacity overflow")?,
    );
    let first_verify = target_request
        .decode_with_mtp_state_and_logits(target_first)
        .map_err(|error| error.to_string())?;
    verify.push(u32::try_from(first_verify.token_ids()[0]).map_err(|_| "negative target token")?);
    sequential_hidden.extend_from_slice(
        first_verify
            .hidden_states_bf16()
            .ok_or("first target verify hidden row is absent")?,
    );
    sequential_logits.extend_from_slice(
        first_verify
            .logits_bf16()
            .ok_or("first target verify logits row is absent")?,
    );
    for draft in &drafts {
        let output = target_request
            .decode_with_mtp_state_and_logits(
                i32::try_from(*draft).map_err(|_| "draft token overflow")?,
            )
            .map_err(|error| error.to_string())?;
        verify.push(u32::try_from(output.token_ids()[0]).map_err(|_| "negative target token")?);
        sequential_hidden.extend_from_slice(
            output
                .hidden_states_bf16()
                .ok_or("target verify hidden row is absent")?,
        );
        sequential_logits.extend_from_slice(
            output
                .logits_bf16()
                .ok_or("target verify logits row is absent")?,
        );
    }
    let mut target_block_request = text_resident
        .new_request(text_graph.clone())
        .map_err(|error| error.to_string())?;
    target_block_request
        .prefill_with_mtp_state(&PROMPT)
        .map_err(|error| error.to_string())?;
    let mut block_inputs = Vec::with_capacity(drafts.len() + 1);
    block_inputs.push(target_first);
    for &token in &drafts {
        block_inputs.push(i32::try_from(token).map_err(|_| "draft token overflow")?);
    }
    let block = target_block_request
        .decode_block_with_mtp_state_and_logits(&block_inputs)
        .map_err(|error| error.to_string())?;
    let target_block_verify = block
        .token_ids()
        .iter()
        .copied()
        .map(|token| u32::try_from(token).map_err(|_| "negative block target token"))
        .collect::<Result<Vec<_>, _>>()?;
    let target_block_matches_sequential = target_block_verify == verify;
    let target_block_logits_match_sequential = block
        .logits_bf16()
        .is_some_and(|logits| logits == sequential_logits);
    let target_block_hidden_matches_sequential = block
        .hidden_states_bf16()
        .is_some_and(|hidden| hidden == sequential_hidden);
    if !target_block_matches_sequential
        || !target_block_logits_match_sequential
        || !target_block_hidden_matches_sequential
    {
        return Err(format!(
            "serial target block differs: tokens={target_block_matches_sequential}, logits={target_block_logits_match_sequential}, hidden={target_block_hidden_matches_sequential}"
        ));
    }
    let decision = verify_greedy(&drafts, &verify).map_err(|error| error.to_string())?;
    let committed_input_rows = 1 + decision.accepted_draft_tokens();
    let resolved = target_block_request
        .resolve_decode_block(committed_input_rows)
        .map_err(|error| error.to_string())?;
    let committed_hidden_words = committed_input_rows * 2_560;
    let partial_prefix_replay_matches_sequential = resolved
        .hidden_states_bf16()
        .is_some_and(|hidden| hidden == &sequential_hidden[..committed_hidden_words])
        && target_block_request.committed_length()
            == u64::try_from(PROMPT.len() + committed_input_rows)
                .map_err(|_| "resolved length overflow")?;
    if !partial_prefix_replay_matches_sequential {
        return Err("resolved speculative prefix differs from sequential target state".to_owned());
    }
    let mut prefix_oracle_request = text_resident
        .new_request(text_graph)
        .map_err(|error| error.to_string())?;
    prefix_oracle_request
        .prefill_with_mtp_state(&PROMPT)
        .map_err(|error| error.to_string())?;
    for &token in &block_inputs[..committed_input_rows] {
        prefix_oracle_request
            .decode_with_mtp_state(token)
            .map_err(|error| error.to_string())?;
    }
    let sequential_kv = prefix_oracle_request
        .kv_payload_bytes_for_evidence()
        .map_err(|error| error.to_string())?;
    let block_kv = target_block_request
        .kv_payload_bytes_for_evidence()
        .map_err(|error| error.to_string())?;
    let committed_kv_matches_sequential = block_kv == sequential_kv;
    if !committed_kv_matches_sequential {
        return Err("committed speculative KV payload differs from sequential target".to_owned());
    }
    let committed_kv_sha256 = hash_kv(&block_kv);
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
    drop(target_block_request);
    drop(prefix_oracle_request);
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
        weight_encoding: if fp8_sidecar.is_some() {
            "ocp-e4m3fn-w8a8"
        } else {
            "bf16"
        },
        kv_encoding: if fp8_sidecar.is_some() {
            "static-fp8"
        } else {
            "fp16"
        },
        draft_width,
        target_block_rows: draft_width + 1,
        target_first,
        draft_tokens: drafts,
        target_verify: verify,
        target_block_verify,
        target_block_matches_sequential,
        target_block_logits_match_sequential,
        target_block_hidden_matches_sequential,
        partial_prefix_replay_matches_sequential,
        committed_kv_matches_sequential,
        accepted_draft_tokens: decision.accepted_draft_tokens(),
        emitted_tokens: decision.emitted_tokens().to_vec(),
        deterministic_replay,
        target_hidden_sha256: hash_words(&target_hidden),
        target_logits_sha256: hash_words(&sequential_logits),
        committed_kv_sha256,
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
