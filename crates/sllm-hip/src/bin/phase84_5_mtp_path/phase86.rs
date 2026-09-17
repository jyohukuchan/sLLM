//! Phase86 Stage 2a MTP catch-up state check.
//!
//! This diagnostic deliberately keeps the existing Phase84.5 entry point and
//! model construction contract.  It only runs when
//! `SLLM_PHASE84_5_DIAGNOSTIC=1` and `SLLM_PHASE86_STATE_CHECK=1` are both
//! present.  The production executor is not changed by this module.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, KvCacheEncoding, QwenExecutionAudit, QwenResidentModel,
    build_qwen35_unsloth_qwen38_nvfp4_graph, build_qwen38_nvfp4_mtp_graph_with_companion,
    build_qwen38_nvfp4_mtp_weight_load_plan, build_qwen38_nvfp4_weight_load_plan, read_model_lock,
    verify_unsloth_qwen38_nvfp4,
};
use sllm_hip::HipBackend;

use super::{
    CHUNK_CAPACITY, COMPAT_MODEL_ENV, COMPLETION_TIMEOUT, DEVICE_ENV, MODEL_ENV,
    MTP_GRAPH_CAPACITY, PHASE83_KV_ENV, PHASE84_MTP_COMPANION_PATH_ENV,
    QWEN38_TOKENIZER_SIZE_BYTES, STATE_CAPACITY, STATE_CAPACITY_ENV, TARGET_ENV, hash_tokens,
    prime_mtp, prompt_cases, provision_qwen38_mtp_resident,
};
use super::{hash_words, kv_image, numeric_diff};

const SCHEMA: &str = "phase86-mtp-catch-up-state-check-v1";
const DRAFT_WIDTH: usize = 2;
const EXPECTED_V620_UUID: &str = "GPU-76a08c022586fed6";
const EXPECTED_R9700_UUID: &str = "GPU-a8e9ddefa2d60f55";

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    gpu_uuid: IdentityReport,
    mtp_encoding: &'static str,
    draft_width: usize,
    stop_after_kv_append: bool,
    bf16_max_ulp: u64,
    cases: Vec<CaseReport>,
    failures: Vec<String>,
    dispatch: DispatchReport,
    cleanup: CleanupReport,
}

#[derive(Serialize)]
struct IdentityReport {
    expected: String,
    observed_visibility: Option<String>,
    exact_match: bool,
}

#[derive(Serialize)]
struct CaseReport {
    name: &'static str,
    prompt_tokens: usize,
    prompt_sha256: String,
    target_first: i32,
    draft_tokens: Vec<i32>,
    target_hidden_sha256: String,
    base_length: u64,
    accepted_draft_tokens: usize,
    catch_up_rows: usize,
    rewound_rows: usize,
    expected_length: u64,
    batch_length: u64,
    scalar_length: u64,
    length_match: bool,
    kv_payload_match: bool,
    batch_kv_sha256: String,
    scalar_kv_sha256: String,
    next_input: i32,
    next_hidden_bf16_ulp: u64,
    next_hidden_max_abs: f32,
    next_logits_max_abs: f32,
    next_logits_max_relative: f32,
    next_logits_first_mismatch: Option<FloatDiffSample>,
    next_token_match: bool,
    next_batch_token: Option<i32>,
    next_scalar_token: Option<i32>,
    truncation: Vec<TruncationReport>,
    dispatch: CaseDispatchReport,
    failures: Vec<String>,
}

#[derive(Serialize)]
struct TruncationReport {
    from_rows: usize,
    to_rows: usize,
    rewind_rows: usize,
    expected_length: u64,
    observed_length: u64,
    length_match: bool,
    kv_payload_match: bool,
    next_hidden_bf16_ulp: u64,
    next_logits_max_abs: f32,
    next_token_match: bool,
    dispatch: TruncationDispatchReport,
    exact: bool,
}

#[derive(Serialize)]
struct TruncationDispatchReport {
    truncated_kernel_dispatches: u64,
    scalar_kernel_dispatches: u64,
    exact_target: bool,
    all_hip: bool,
    fallback_used: bool,
}

#[derive(Serialize)]
struct CaseDispatchReport {
    target_kernel_dispatches: u64,
    proposal_kernel_dispatches: u64,
    scalar_kernel_dispatches: u64,
    all_target_match: bool,
    all_hip: bool,
    fallback_used: bool,
}

#[derive(Serialize)]
struct DispatchReport {
    target_kernel_dispatches: u64,
    mtp_kernel_dispatches: u64,
    nonzero: bool,
    exact_target: bool,
    all_hip: bool,
    fallback_used: bool,
}

#[derive(Serialize)]
struct CleanupReport {
    current_bytes_before_shutdown: u64,
    poisoned_before_shutdown: bool,
    retryable_cleanup: usize,
    durable_quarantine: usize,
    zero: bool,
}

#[derive(Serialize)]
struct FloatDiffSample {
    index: usize,
    left: f32,
    right: f32,
    abs: f32,
    relative: f32,
}

struct FloatDiff {
    max_abs: f32,
    max_relative: f32,
    first_mismatch: Option<FloatDiffSample>,
}

struct CaseRun {
    reports: Vec<CaseReport>,
    audits: Vec<QwenExecutionAudit>,
    target_dispatches: u64,
    mtp_dispatches: u64,
}

struct TruncationRun {
    report: TruncationReport,
    audits: Vec<QwenExecutionAudit>,
}

fn expected_uuid(target: &str) -> Result<&'static str, String> {
    match target {
        "gfx1030" => Ok(EXPECTED_V620_UUID),
        "gfx1201" => Ok(EXPECTED_R9700_UUID),
        _ => Err(format!("unsupported Phase86 target: {target}")),
    }
}

fn visible_uuid() -> Option<String> {
    env::var("ROCR_VISIBLE_DEVICES")
        .ok()
        .or_else(|| env::var("HIP_VISIBLE_DEVICES").ok())
}

fn float_diff(left: &[f32], right: &[f32]) -> Result<FloatDiff, String> {
    if left.len() != right.len() {
        return Err(format!(
            "next proposal logits length differs: {} vs {}",
            left.len(),
            right.len()
        ));
    }
    let mut diff = FloatDiff {
        max_abs: 0.0,
        max_relative: 0.0,
        first_mismatch: None,
    };
    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err(format!(
                "non-finite next proposal logit at index {index}: left={a} right={b}"
            ));
        }
        let abs = (a - b).abs();
        let relative = abs / a.abs().max(b.abs()).max(1.0e-12);
        diff.max_abs = diff.max_abs.max(abs);
        diff.max_relative = diff.max_relative.max(relative);
        if a != b && diff.first_mismatch.is_none() {
            diff.first_mismatch = Some(FloatDiffSample {
                index,
                left: a,
                right: b,
                abs,
                relative,
            });
        }
    }
    Ok(diff)
}

fn hash_payload(payload: &[(u32, Vec<u8>, Vec<u8>)]) -> String {
    let mut hash = Sha256::new();
    for (layer, key, value) in payload {
        hash.update(layer.to_le_bytes());
        hash.update((key.len() as u64).to_le_bytes());
        hash.update(key);
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value);
    }
    format!("sha256:{:x}", hash.finalize())
}

fn audit_is_valid(audit: &QwenExecutionAudit, target: &str) -> bool {
    audit.target() == target
        && audit.selected_backend() == "hip"
        && audit.kernel_dispatch_count() > 0
        && !audit.fallback_used()
        && audit.all_dispatches_hip()
}

fn truncation_check(
    mtp_resident: &QwenResidentModel,
    mtp_graph: &sllm_core::QwenGraph,
    prompt: &[i32],
    target_hidden: &[u16],
    last_target_hidden: &[u16],
    hidden_width: usize,
    inputs: &[i32],
    draft_tokens: &[i32],
    target_hidden_rows: &[u16],
    block_tokens: &[i32],
    block_hidden: &[u16],
    to_rows: usize,
    target_name: &str,
) -> Result<TruncationRun, String> {
    let from_rows = DRAFT_WIDTH + 1;
    if !(1..from_rows).contains(&to_rows) {
        return Err(format!("invalid truncation row count: {to_rows}"));
    }
    let mut truncated = mtp_resident
        .new_request(mtp_graph.clone())
        .map_err(|error| error.to_string())?;
    prime_mtp(&mut truncated, prompt, target_hidden, hidden_width)?;
    let base_length = truncated.committed_length();
    let mut proposal_token = inputs[0];
    let mut proposal_hidden = last_target_hidden.to_vec();
    for (row, &expected_token) in draft_tokens.iter().enumerate() {
        let output = truncated
            .decode_mtp(proposal_token, &proposal_hidden)
            .map_err(|error| error.to_string())?;
        let observed = *output
            .token_ids()
            .last()
            .ok_or("truncation draft replay omitted token")?;
        if observed != expected_token {
            return Err(format!(
                "truncation draft replay row {row}: observed {observed}, expected {expected_token}"
            ));
        }
        proposal_token = observed;
        proposal_hidden = output
            .hidden_states_bf16()
            .ok_or("truncation draft replay omitted hidden row")?
            .to_vec();
    }
    for _ in 0..DRAFT_WIDTH {
        truncated
            .rewind_last_decode_transition()
            .map_err(|error| error.to_string())?;
    }
    truncated
        .decode_mtp_state_only_batch(inputs, target_hidden_rows)
        .map_err(|error| error.to_string())?;
    let rewind_rows = from_rows - to_rows;
    for _ in 0..rewind_rows {
        truncated
            .rewind_last_decode_transition()
            .map_err(|error| error.to_string())?;
    }
    let expected_length = base_length + to_rows as u64;
    let observed_length = truncated.committed_length();
    let truncated_payload = kv_image::kv_payload(&truncated)?;

    let mut scalar = mtp_resident
        .new_request(mtp_graph.clone())
        .map_err(|error| error.to_string())?;
    prime_mtp(&mut scalar, prompt, target_hidden, hidden_width)?;
    scalar
        .decode_mtp_state_only_batch(
            &inputs[..to_rows],
            &target_hidden_rows[..to_rows * hidden_width],
        )
        .map_err(|error| error.to_string())?;
    let scalar_payload = kv_image::kv_payload(&scalar)?;
    let kv_payload_match = truncated_payload == scalar_payload;
    let next_index = to_rows - 1;
    let next_hidden = &block_hidden[next_index * hidden_width..(next_index + 1) * hidden_width];
    let next_input = block_tokens[next_index];
    let truncated_next = truncated
        .decode_mtp(next_input, next_hidden)
        .map_err(|error| error.to_string())?;
    let scalar_next = scalar
        .decode_mtp(next_input, next_hidden)
        .map_err(|error| error.to_string())?;
    let hidden_diff = numeric_diff(
        truncated_next
            .hidden_states_bf16()
            .ok_or("truncation next hidden row missing")?,
        scalar_next
            .hidden_states_bf16()
            .ok_or("scalar truncation next hidden row missing")?,
    )?;
    let logits_diff = float_diff(
        truncated_next
            .last_logits()
            .ok_or("truncation next logits missing")?,
        scalar_next
            .last_logits()
            .ok_or("scalar truncation next logits missing")?,
    )?;
    let next_token_match = truncated_next.token_ids().last() == scalar_next.token_ids().last();
    let truncated_audit = truncated
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    let scalar_audit = scalar.audit_snapshot().map_err(|error| error.to_string())?;
    let exact_target =
        truncated_audit.target() == target_name && scalar_audit.target() == target_name;
    let all_hip = truncated_audit.selected_backend() == "hip"
        && scalar_audit.selected_backend() == "hip"
        && truncated_audit.all_dispatches_hip()
        && scalar_audit.all_dispatches_hip();
    let fallback_used = truncated_audit.fallback_used() || scalar_audit.fallback_used();
    let nonzero =
        truncated_audit.kernel_dispatch_count() > 0 && scalar_audit.kernel_dispatch_count() > 0;
    let length_match = observed_length == expected_length;
    let exact = length_match
        && kv_payload_match
        && hidden_diff.max_ulp == 0
        && logits_diff.max_abs == 0.0
        && next_token_match
        && exact_target
        && all_hip
        && !fallback_used
        && nonzero;
    Ok(TruncationRun {
        report: TruncationReport {
            from_rows,
            to_rows,
            rewind_rows,
            expected_length,
            observed_length,
            length_match,
            kv_payload_match,
            next_hidden_bf16_ulp: hidden_diff.max_ulp,
            next_logits_max_abs: logits_diff.max_abs,
            next_token_match,
            dispatch: TruncationDispatchReport {
                truncated_kernel_dispatches: truncated_audit.kernel_dispatch_count(),
                scalar_kernel_dispatches: scalar_audit.kernel_dispatch_count(),
                exact_target,
                all_hip,
                fallback_used,
            },
            exact,
        },
        audits: vec![truncated_audit, scalar_audit],
    })
}

fn run_case(
    target_resident: &QwenResidentModel,
    mtp_resident: &QwenResidentModel,
    graph: &sllm_core::QwenGraph,
    mtp_graph: &sllm_core::QwenGraph,
    name: &'static str,
    prompt: &[i32],
    target_name: &str,
) -> Result<CaseRun, String> {
    let mut target = target_resident
        .new_request(graph.clone())
        .map_err(|error| error.to_string())?;
    let prefill = target
        .prefill_with_mtp_state(prompt)
        .map_err(|error| error.to_string())?;
    let target_hidden = prefill
        .hidden_states_bf16()
        .ok_or("target prefill hidden rows missing")?
        .to_vec();
    let hidden_width = target
        .mtp_hidden_width()
        .map_err(|error| error.to_string())?;
    if target_hidden.len() != prompt.len() * hidden_width {
        return Err("target prefill hidden shape differs from prompt rows".to_owned());
    }
    let target_first = *prefill
        .token_ids()
        .last()
        .ok_or("target prefill token missing")?;
    let last_target_hidden = target_hidden[(prompt.len() - 1) * hidden_width..].to_vec();

    // Obtain one fixed width-2 draft chain.  The hidden for row 1 is the
    // draft output, matching the production path whose state is being tested.
    let mut proposal = mtp_resident
        .new_request(mtp_graph.clone())
        .map_err(|error| error.to_string())?;
    prime_mtp(&mut proposal, prompt, &target_hidden, hidden_width)?;
    let base_length = proposal.committed_length();
    let mut proposal_token = target_first;
    let mut proposal_hidden = last_target_hidden.clone();
    let mut draft_tokens = Vec::with_capacity(DRAFT_WIDTH);
    for _ in 0..DRAFT_WIDTH {
        let output = proposal
            .decode_mtp(proposal_token, &proposal_hidden)
            .map_err(|error| error.to_string())?;
        let token = *output
            .token_ids()
            .last()
            .ok_or("MTP proposal omitted token")?;
        proposal_hidden = output
            .hidden_states_bf16()
            .ok_or("MTP proposal omitted hidden row")?
            .to_vec();
        draft_tokens.push(token);
        proposal_token = token;
    }

    // The target block supplies the same target hidden row to both catch-up
    // owners.  No target M3/M1 comparison is performed here.
    let mut target_block = target_resident
        .new_request(graph.clone())
        .map_err(|error| error.to_string())?;
    target_block
        .prefill_with_mtp_state(prompt)
        .map_err(|error| error.to_string())?;
    let mut inputs = Vec::with_capacity(DRAFT_WIDTH + 1);
    inputs.push(target_first);
    inputs.extend(draft_tokens.iter().copied());
    let target_output = target_block
        .decode_block_with_mtp_state_and_logits(&inputs)
        .map_err(|error| error.to_string())?;
    let block_hidden = target_output
        .hidden_states_bf16()
        .ok_or("target block hidden rows missing")?
        .to_vec();
    if block_hidden.len() != inputs.len() * hidden_width {
        return Err("target block hidden row count differs from draft width".to_owned());
    }
    let block_tokens = target_output.token_ids().to_vec();
    if block_tokens.len() != inputs.len() {
        return Err("target block token rows differ from draft width".to_owned());
    }
    let mut target_hidden_rows = Vec::with_capacity(inputs.len() * hidden_width);
    target_hidden_rows.extend_from_slice(&last_target_hidden);
    target_hidden_rows.extend_from_slice(&block_hidden[..DRAFT_WIDTH * hidden_width]);
    let target_hidden_sha256 = hash_words(&target_hidden_rows);
    let target_audit = target_block
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    let target_dispatches = target_audit.kernel_dispatch_count();
    let target_audit_valid = audit_is_valid(&target_audit, target_name);

    // Rewind both draft transitions, then use one state-only batch for each
    // accepted prefix.  This is the Stage2a candidate under test.
    for _ in 0..DRAFT_WIDTH {
        proposal
            .rewind_last_decode_transition()
            .map_err(|error| error.to_string())?;
    }
    if proposal.committed_length() != base_length {
        return Err(format!(
            "rewind did not restore base length: {} vs {base_length}",
            proposal.committed_length()
        ));
    }

    let mut accepted_reports = Vec::with_capacity(DRAFT_WIDTH + 1);
    let mut audits = vec![target_audit];
    let mut rewound_proposal = Some(proposal);
    let mut mtp_dispatches = 0_u64;
    for accepted_draft_tokens in 0..=DRAFT_WIDTH {
        let catch_up_rows = accepted_draft_tokens + 1;
        let row_width = hidden_width;
        let target_rows = &target_hidden_rows[..catch_up_rows * row_width];
        let commit_tokens = &inputs[..catch_up_rows];
        let mut row_failures = Vec::new();

        // The batch owner is independent across accepted counts.  Keeping a
        // fresh request per row count also exercises row truncation directly.
        let mut batch = if accepted_draft_tokens == 0 {
            rewound_proposal
                .take()
                .ok_or("rewound proposal owner was already consumed")?
        } else {
            let mut request = mtp_resident
                .new_request(mtp_graph.clone())
                .map_err(|error| error.to_string())?;
            prime_mtp(&mut request, prompt, &target_hidden, hidden_width)?;
            request
        };
        let batch_base = batch.committed_length();
        if accepted_draft_tokens != 0 {
            let mut replay_token = target_first;
            let mut replay_hidden = last_target_hidden.clone();
            for (row, &expected_token) in draft_tokens.iter().enumerate() {
                let output = batch
                    .decode_mtp(replay_token, &replay_hidden)
                    .map_err(|error| error.to_string())?;
                let observed_token = *output
                    .token_ids()
                    .last()
                    .ok_or("MTP replay proposal omitted token")?;
                if observed_token != expected_token {
                    row_failures.push(format!(
                        "draft replay row {row} selected {observed_token}, expected {expected_token}"
                    ));
                }
                replay_token = observed_token;
                replay_hidden = output
                    .hidden_states_bf16()
                    .ok_or("MTP replay proposal omitted hidden row")?
                    .to_vec();
            }
            for _ in 0..DRAFT_WIDTH {
                batch
                    .rewind_last_decode_transition()
                    .map_err(|error| error.to_string())?;
            }
            if batch.committed_length() != batch_base {
                row_failures.push(format!(
                    "draft rewind restored {} rows, expected base {batch_base}",
                    batch.committed_length()
                ));
            }
        }
        batch
            .decode_mtp_state_only_batch(commit_tokens, target_rows)
            .map_err(|error| error.to_string())?;
        let expected_length = batch_base + catch_up_rows as u64;
        let batch_length = batch.committed_length();

        // The scalar owner uses exactly the same target hidden rows, one row
        // at a time.  It is the independent oracle for effective state.
        let mut scalar = mtp_resident
            .new_request(mtp_graph.clone())
            .map_err(|error| error.to_string())?;
        prime_mtp(&mut scalar, prompt, &target_hidden, hidden_width)?;
        for (token, hidden) in commit_tokens.iter().zip(target_rows.chunks(row_width)) {
            scalar
                .decode_mtp_state_only_batch(std::slice::from_ref(token), hidden)
                .map_err(|error| error.to_string())?;
        }
        let scalar_length = scalar.committed_length();
        let batch_payload = kv_image::kv_payload(&batch)?;
        let scalar_payload = kv_image::kv_payload(&scalar)?;
        let batch_kv_sha256 = hash_payload(&batch_payload);
        let scalar_kv_sha256 = hash_payload(&scalar_payload);
        let kv_payload_match = batch_payload == scalar_payload;

        let next_index = catch_up_rows - 1;
        let next_input = block_tokens[next_index];
        let next_hidden = &block_hidden[next_index * row_width..(next_index + 1) * row_width];
        let next_batch = batch
            .decode_mtp(next_input, next_hidden)
            .map_err(|error| error.to_string())?;
        let next_scalar = scalar
            .decode_mtp(next_input, next_hidden)
            .map_err(|error| error.to_string())?;
        let next_hidden_diff = numeric_diff(
            next_batch
                .hidden_states_bf16()
                .ok_or("batch next proposal hidden row missing")?,
            next_scalar
                .hidden_states_bf16()
                .ok_or("scalar next proposal hidden row missing")?,
        )?;
        let next_logits_diff = float_diff(
            next_batch
                .last_logits()
                .ok_or("batch next proposal logits missing")?,
            next_scalar
                .last_logits()
                .ok_or("scalar next proposal logits missing")?,
        )?;
        let next_batch_token = next_batch.token_ids().last().copied();
        let next_scalar_token = next_scalar.token_ids().last().copied();
        let next_token_match = next_batch_token == next_scalar_token;
        let batch_audit = batch.audit_snapshot().map_err(|error| error.to_string())?;
        let scalar_audit = scalar.audit_snapshot().map_err(|error| error.to_string())?;
        let audits_valid =
            audit_is_valid(&batch_audit, target_name) && audit_is_valid(&scalar_audit, target_name);
        let mut truncation = Vec::new();
        if accepted_draft_tokens == DRAFT_WIDTH {
            for to_rows in [1_usize, 2] {
                match truncation_check(
                    mtp_resident,
                    mtp_graph,
                    prompt,
                    &target_hidden,
                    &last_target_hidden,
                    hidden_width,
                    &inputs,
                    &draft_tokens,
                    &target_hidden_rows,
                    &block_tokens,
                    &block_hidden,
                    to_rows,
                    target_name,
                ) {
                    Ok(check) => {
                        if !check.report.exact {
                            row_failures.push(format!(
                                "truncation 3->{to_rows} mismatch: length={} kv={} hidden_ulp={} logits_abs={} token={}",
                                check.report.length_match,
                                check.report.kv_payload_match,
                                check.report.next_hidden_bf16_ulp,
                                check.report.next_logits_max_abs,
                                check.report.next_token_match,
                            ));
                        }
                        mtp_dispatches = mtp_dispatches.saturating_add(
                            check
                                .audits
                                .iter()
                                .map(QwenExecutionAudit::kernel_dispatch_count)
                                .sum::<u64>(),
                        );
                        audits.extend(check.audits);
                        truncation.push(check.report);
                    }
                    Err(error) => row_failures.push(format!(
                        "truncation 3->{to_rows} could not be measured: {error}"
                    )),
                }
            }
        }

        if !batch_length.eq(&expected_length)
            || !scalar_length.eq(&expected_length)
            || !kv_payload_match
            || next_hidden_diff.max_ulp != 0
            || next_logits_diff.max_abs != 0.0
            || !next_token_match
            || !target_audit_valid
            || !audits_valid
        {
            row_failures.push(format!(
                "accepted={accepted_draft_tokens} batch_length={batch_length} scalar_length={scalar_length} expected={expected_length} kv_match={kv_payload_match} next_hidden_ulp={} next_logits_abs={} next_token_match={next_token_match} audits_valid={audits_valid}",
                next_hidden_diff.max_ulp, next_logits_diff.max_abs
            ));
        }
        mtp_dispatches = mtp_dispatches
            .saturating_add(batch_audit.kernel_dispatch_count())
            .saturating_add(scalar_audit.kernel_dispatch_count());
        accepted_reports.push(CaseReport {
            name,
            prompt_tokens: prompt.len(),
            prompt_sha256: hash_tokens(prompt),
            target_first,
            draft_tokens: draft_tokens.clone(),
            target_hidden_sha256: target_hidden_sha256.clone(),
            base_length: batch_base,
            accepted_draft_tokens,
            catch_up_rows,
            rewound_rows: DRAFT_WIDTH,
            expected_length,
            batch_length,
            scalar_length,
            length_match: batch_length == expected_length && scalar_length == expected_length,
            kv_payload_match,
            batch_kv_sha256,
            scalar_kv_sha256,
            next_input,
            next_hidden_bf16_ulp: next_hidden_diff.max_ulp,
            next_hidden_max_abs: next_hidden_diff.max_abs,
            next_logits_max_abs: next_logits_diff.max_abs,
            next_logits_max_relative: next_logits_diff.max_relative,
            next_logits_first_mismatch: next_logits_diff.first_mismatch,
            next_token_match,
            next_batch_token,
            next_scalar_token,
            truncation,
            dispatch: CaseDispatchReport {
                target_kernel_dispatches: target_dispatches,
                proposal_kernel_dispatches: batch_audit.kernel_dispatch_count(),
                scalar_kernel_dispatches: scalar_audit.kernel_dispatch_count(),
                all_target_match: target_audit_valid,
                all_hip: audits_valid,
                fallback_used: batch_audit.fallback_used() || scalar_audit.fallback_used(),
            },
            failures: row_failures,
        });
        audits.push(batch_audit);
        audits.push(scalar_audit);
    }

    if accepted_reports.is_empty() {
        return Err("state check accepted-row report list is empty".to_owned());
    }
    Ok(CaseRun {
        reports: accepted_reports,
        audits,
        target_dispatches,
        mtp_dispatches,
    })
}

fn run_state_check() -> Result<Report, String> {
    let target = env::var(TARGET_ENV).map_err(|_| format!("{TARGET_ENV} is required"))?;
    let expected_gpu_uuid = expected_uuid(&target)?;
    let observed_gpu_uuid = visible_uuid();
    let uuid_match = observed_gpu_uuid.as_deref() == Some(expected_gpu_uuid);
    let device_index = env::var(DEVICE_ENV)
        .map_err(|_| format!("{DEVICE_ENV} is required"))?
        .parse::<u32>()
        .map_err(|_| format!("{DEVICE_ENV} must be a u32"))?;
    let model_root = env::var_os(MODEL_ENV)
        .or_else(|| env::var_os(COMPAT_MODEL_ENV))
        .map(PathBuf::from)
        .ok_or_else(|| format!("{MODEL_ENV} or {COMPAT_MODEL_ENV} is required"))?;
    if env::var_os(PHASE84_MTP_COMPANION_PATH_ENV).is_some() {
        return Err(format!(
            "Phase86 Stage2a requires BF16 MTP; {PHASE84_MTP_COMPANION_PATH_ENV} must be unset"
        ));
    }
    let state_capacity = env::var(STATE_CAPACITY_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(STATE_CAPACITY);
    if state_capacity < STATE_CAPACITY {
        return Err(format!(
            "{STATE_CAPACITY_ENV} must be at least {STATE_CAPACITY}"
        ));
    }
    let kv = match env::var(PHASE83_KV_ENV) {
        Ok(_) => super::parse_phase83_kv()?,
        Err(env::VarError::NotPresent) => KvCacheEncoding::Mxfp8E4,
        Err(error) => return Err(error.to_string()),
    };
    let artifact =
        Arc::new(verify_unsloth_qwen38_nvfp4(&model_root).map_err(|error| error.to_string())?);
    if artifact
        .root()
        .join("tokenizer.json")
        .metadata()
        .map_err(|error| error.to_string())?
        .len()
        != QWEN38_TOKENIZER_SIZE_BYTES
    {
        return Err("verified tokenizer size changed".to_owned());
    }
    let lock_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/models/locks/qwen3.5-27b-bf16.json");
    let lock = read_model_lock(&lock_path).map_err(|error| error.to_string())?;
    let plan =
        build_qwen38_nvfp4_weight_load_plan(&lock, &artifact).map_err(|error| error.to_string())?;
    let mtp_plan = build_qwen38_nvfp4_mtp_weight_load_plan(&lock, &artifact)
        .map_err(|error| error.to_string())?;
    let graph = build_qwen35_unsloth_qwen38_nvfp4_graph(
        &lock,
        &plan,
        &artifact,
        CHUNK_CAPACITY,
        state_capacity,
        kv,
    )
    .map_err(|error| error.to_string())?;
    let mtp_graph = build_qwen38_nvfp4_mtp_graph_with_companion(
        &lock,
        &mtp_plan,
        &artifact,
        state_capacity,
        kv,
        MTP_GRAPH_CAPACITY,
        None,
    )
    .map_err(|error| error.to_string())?;
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let target_resident = QwenResidentModel::new_unsloth_qwen38_nvfp4(
        Arc::clone(&session),
        graph.clone(),
        plan,
        Arc::clone(&artifact),
        COMPLETION_TIMEOUT,
    )
    .map_err(|error| error.to_string())?;
    let mtp_resident = provision_qwen38_mtp_resident(
        &target_resident,
        mtp_graph.clone(),
        mtp_plan,
        Arc::clone(&artifact),
        None,
    )?;

    let prompts = prompt_cases(&artifact)?;
    let mut cases = Vec::new();
    let mut failures = Vec::new();
    let mut audits = Vec::new();
    let mut target_dispatches = 0_u64;
    let mut mtp_dispatches = 0_u64;
    for (name, prompt) in prompts {
        match run_case(
            &target_resident,
            &mtp_resident,
            &graph,
            &mtp_graph,
            name,
            &prompt,
            &target,
        ) {
            Ok(case) => {
                for report in &case.reports {
                    for failure in &report.failures {
                        failures.push(format!(
                            "case={name} accepted={}: {failure}",
                            report.accepted_draft_tokens
                        ));
                    }
                }
                cases.extend(case.reports);
                target_dispatches = target_dispatches.saturating_add(case.target_dispatches);
                mtp_dispatches = mtp_dispatches.saturating_add(case.mtp_dispatches);
                audits.extend(case.audits);
            }
            Err(error) => failures.push(format!("case={name}: {error}")),
        }
    }
    if !uuid_match {
        failures.push(format!(
            "GPU UUID visibility mismatch: expected={expected_gpu_uuid} observed={observed_gpu_uuid:?}"
        ));
    }
    if cases.is_empty() {
        failures.push("all Phase86 state-check cases failed before producing evidence".to_owned());
    }

    let exact_target = audits.iter().all(|audit| audit.target() == target);
    let all_hip = audits
        .iter()
        .all(|audit| audit.selected_backend() == "hip" && audit.all_dispatches_hip());
    let fallback_used = audits.iter().any(|audit| audit.fallback_used());
    let nonzero = audits.iter().all(|audit| audit.kernel_dispatch_count() > 0);

    drop(mtp_resident);
    drop(target_resident);
    let allocation_before_shutdown = session.memory_snapshot();
    let cleanup = session
        .shutdown(Duration::from_secs(30))
        .map_err(|error| error.to_string())?;
    let cleanup_zero = allocation_before_shutdown.current_bytes() == 0
        && !allocation_before_shutdown.poisoned()
        && cleanup.retryable_cleanup == 0
        && cleanup.durable_quarantine == 0;
    if !cleanup_zero {
        failures.push(format!(
            "cleanup was not zero: current_bytes={} poisoned={} retryable={} durable={}",
            allocation_before_shutdown.current_bytes(),
            allocation_before_shutdown.poisoned(),
            cleanup.retryable_cleanup,
            cleanup.durable_quarantine
        ));
    }
    if !nonzero || !exact_target || !all_hip || fallback_used {
        failures.push(format!(
            "dispatch audit failed: nonzero={nonzero} exact_target={exact_target} all_hip={all_hip} fallback_used={fallback_used}"
        ));
    }
    Ok(Report {
        schema_version: SCHEMA,
        state: if failures.is_empty() { "PASS" } else { "FAIL" },
        target,
        device_index,
        gpu_uuid: IdentityReport {
            expected: expected_gpu_uuid.to_owned(),
            observed_visibility: observed_gpu_uuid,
            exact_match: uuid_match,
        },
        mtp_encoding: "bf16",
        draft_width: DRAFT_WIDTH,
        stop_after_kv_append: true,
        bf16_max_ulp: 0,
        cases,
        failures,
        dispatch: DispatchReport {
            target_kernel_dispatches: target_dispatches,
            mtp_kernel_dispatches: mtp_dispatches,
            nonzero,
            exact_target,
            all_hip,
            fallback_used,
        },
        cleanup: CleanupReport {
            current_bytes_before_shutdown: allocation_before_shutdown.current_bytes(),
            poisoned_before_shutdown: allocation_before_shutdown.poisoned(),
            retryable_cleanup: cleanup.retryable_cleanup,
            durable_quarantine: cleanup.durable_quarantine,
            zero: cleanup_zero,
        },
    })
}

pub(super) fn run() -> ExitCode {
    match run_state_check() {
        Ok(report) => {
            let failed = report.state == "FAIL";
            println!(
                "{}",
                serde_json::to_string(&report).unwrap_or_else(|error| {
                    format!(
                        "{{\"schema_version\":\"{SCHEMA}\",\"state\":\"FAIL\",\"error\":{error:?}}}"
                    )
                })
            );
            if failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            println!(
                "{{\"schema_version\":\"{SCHEMA}\",\"state\":\"FAIL\",\"error\":{}}}",
                serde_json::to_string(&error)
                    .unwrap_or_else(|_| "\"state check failed\"".to_owned())
            );
            ExitCode::FAILURE
        }
    }
}
