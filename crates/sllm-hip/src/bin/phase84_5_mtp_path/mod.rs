//! Bounded Phase84.5 Qwen3.8 MTP path diagnostic.
//!
//! This is intentionally a benchmark-local diagnostic.  It reuses the
//! production Qwen request and frontend executor seams, and only exposes
//! redacted hashes, discrete state checks, and bounded numerical differences.
//! It is not a second inference implementation and is enabled explicitly by
//! `SLLM_PHASE84_5_DIAGNOSTIC=1`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, KvCacheEncoding, OsSamplingRandom, QWEN35_VOCAB_SIZE,
    QwenExecutionRequest, QwenResidentModel, SamplerChainConfigV1, SamplerChainV1,
    SamplingParametersV1, build_qwen35_unsloth_qwen38_nvfp4_graph,
    build_qwen38_nvfp4_mtp_graph_with_companion, build_qwen38_nvfp4_mtp_weight_load_plan,
    build_qwen38_nvfp4_weight_load_plan, read_model_lock, verify_qwen38_mtp_quantized_sidecar,
    verify_unsloth_qwen38_nvfp4,
};
use sllm_frontend::{GenerationExecutorV1, QwenMtpGenerationExecutorV1, TokenizerFrontendV1};
use sllm_hip::HipBackend;

mod kv_image;

use super::{
    COMPAT_MODEL_ENV, COMPLETION_TIMEOUT, DEVICE_ENV, MODEL_ENV, PHASE83_KV_ENV,
    PHASE84_MTP_COMPANION_PATH_ENV, QWEN38_TOKENIZER_SIZE_BYTES, STATE_CAPACITY_ENV, TARGET_ENV,
    hash_tokens, parse_phase83_kv, provision_qwen38_mtp_resident,
};

const DIAGNOSTIC_SCHEMA: &str = "phase84-5-qwen38-mtp-path-diagnostic-v1";
const STATE_CAPACITY: u64 = 256;
const CHUNK_CAPACITY: u64 = 128;
const MTP_GRAPH_CAPACITY: u64 = 256;
const DRAFT_WIDTH: usize = 2;
const SAMPLER_SEED: u64 = 123;
// BF16 output readback is the comparison boundary. Three representable BF16
// steps is a provisional diagnostic screen borrowed from an existing GDN
// fixture; it is not a validated full-model Qwen3.8 bound. Discrete tokens,
// lengths, and KV bytes remain exact comparisons.
const NUMERIC_MAX_BF16_ULP: u64 = 3;
const LOGPROB_MAX_ABS_ERROR: f64 = 1.0e-4;

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    kv_encoding: &'static str,
    mtp_encoding: &'static str,
    sampling: &'static str,
    draft_width: usize,
    thresholds: ThresholdReport,
    cases: Vec<CaseReport>,
    failures: Vec<String>,
    raw_dump_directory: Option<String>,
    frontend_executor: FrontendExecutorReport,
    dispatch: DispatchReport,
}

#[derive(Serialize)]
struct ThresholdReport {
    bf16_max_ulp: u64,
    logprob_max_abs_error: f64,
    rationale: &'static str,
    discrete_comparisons: &'static str,
}

#[derive(Serialize)]
struct CaseReport {
    name: &'static str,
    prompt_tokens: usize,
    prompt_sha256: String,
    target_prefill_hidden_sha256: String,
    sequential_prefill_hidden_sha256: String,
    block_prefill_hidden_sha256: String,
    target_prefill_hidden_matches: bool,
    target_first: i32,
    draft: DraftReport,
    target_block: TargetBlockReport,
    resolve_rows: Vec<ResolveReport>,
    failures: Vec<String>,
}

#[derive(Serialize)]
struct DraftReport {
    optimized_tokens: Vec<u32>,
    optimized_selection_logprobs: Vec<f64>,
    sequential_logit_sha256: Vec<String>,
    optimized_hidden_matches_sequential: Vec<bool>,
    optimized_hidden_max_abs: Vec<f32>,
    optimized_hidden_max_relative: Vec<f32>,
    optimized_hidden_max_bf16_ulp: Vec<u64>,
    cpu_sampler_token_matches: Vec<bool>,
    cpu_sampler_logprobs: Vec<f64>,
    cpu_oracle_logprobs_for_gpu_token: Vec<f64>,
    gpu_logprob_abs_error: Vec<f64>,
    gpu_logprob_within_threshold: Vec<bool>,
}

#[derive(Serialize)]
struct TargetBlockReport {
    inputs: Vec<i32>,
    sequential_tokens: Vec<i32>,
    block_tokens: Vec<i32>,
    token_match: bool,
    hidden_max_abs: f32,
    hidden_max_relative: f32,
    hidden_max_bf16_ulp: u64,
    hidden_mismatch: Option<DiffSample>,
    logits_max_abs: f32,
    logits_max_relative: f32,
    logits_max_bf16_ulp: u64,
    logits_mismatch: Option<DiffSample>,
    hidden_within_threshold: bool,
    logits_within_threshold: bool,
    sequential_logits_sha256: String,
    block_logits_sha256: String,
}

#[derive(Serialize, Debug)]
struct ResolveReport {
    accepted_draft_tokens: usize,
    committed_input_rows: usize,
    committed_length: u64,
    sequential_length: u64,
    committed_length_match: bool,
    resolved_token_match: bool,
    block_prefill_hidden_sha256: String,
    sequential_prefill_hidden_sha256: String,
    prefill_hidden_match: bool,
    hidden_max_abs: f32,
    hidden_max_relative: f32,
    hidden_max_bf16_ulp: u64,
    hidden_within_threshold: bool,
    kv_match: bool,
    next_token_match: bool,
    next_hidden_max_abs: f32,
    next_hidden_max_relative: f32,
    next_hidden_max_bf16_ulp: u64,
    next_logits_max_abs: f32,
    next_logits_max_relative: f32,
    next_logits_max_bf16_ulp: u64,
    next_within_threshold: bool,
    companion_kv_match: bool,
    companion_next_token_match: bool,
    companion_next_logprob_abs_error: f64,
    companion_next_hidden_max_bf16_ulp: u64,
}

#[derive(Serialize)]
struct FrontendExecutorReport {
    prefill_token: u32,
    decode_token: u32,
    prefill_logprob: f64,
    decode_logprob: f64,
    target_committed_length: u64,
    mtp_committed_length: u64,
    target_hip_only: bool,
    mtp_hip_only: bool,
    finished: bool,
}

#[derive(Serialize)]
struct DispatchReport {
    target_dispatches: u64,
    mtp_dispatches: u64,
    fallback_used: bool,
    all_dispatches_hip: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
struct NumericDiff {
    max_abs: f32,
    max_relative: f32,
    max_ulp: u64,
    first_mismatch: Option<DiffSample>,
}

#[derive(Clone, Debug, Serialize)]
struct DiffSample {
    index: usize,
    left_bf16: u16,
    right_bf16: u16,
    left_f32: f32,
    right_f32: f32,
    abs: f32,
    relative: f32,
    ulp: u64,
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn ordered_bf16_bits(value: u16) -> u16 {
    if value & 0x7fff == 0 {
        0x8000
    } else if value & 0x8000 != 0 {
        !value
    } else {
        value | 0x8000
    }
}

fn numeric_diff(left: &[u16], right: &[u16]) -> Result<NumericDiff, String> {
    if left.len() != right.len() {
        return Err(format!(
            "numeric row length differs: {} vs {}",
            left.len(),
            right.len()
        ));
    }
    let mut diff = NumericDiff::default();
    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        let af = bf16_to_f32(a);
        let bf = bf16_to_f32(b);
        if !af.is_finite() || !bf.is_finite() {
            return Err(format!(
                "non-finite BF16 value observed at index {index}: left=0x{a:04x} right=0x{b:04x}"
            ));
        }
        let abs = (af - bf).abs();
        let relative = abs / af.abs().max(bf.abs()).max(1.0e-12);
        let ulp = u64::from(ordered_bf16_bits(a).abs_diff(ordered_bf16_bits(b)));
        if a != b && diff.first_mismatch.is_none() {
            diff.first_mismatch = Some(DiffSample {
                index,
                left_bf16: a,
                right_bf16: b,
                left_f32: af,
                right_f32: bf,
                abs,
                relative,
                ulp,
            });
        }
        diff.max_abs = diff.max_abs.max(abs);
        diff.max_relative = diff.max_relative.max(relative);
        diff.max_ulp = diff.max_ulp.max(ulp);
    }
    Ok(diff)
}

fn hash_words(words: &[u16]) -> String {
    let mut hash = Sha256::new();
    for word in words {
        hash.update(word.to_le_bytes());
    }
    format!("sha256:{:x}", hash.finalize())
}

fn dump_u16_words(path: &std::path::Path, words: &[u16]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(words.len() * 2);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

#[allow(clippy::too_many_arguments)] // Explicitly names paired diagnostic observations.
fn dump_block_failure(
    case_name: &str,
    sequential_hidden: &[u16],
    block_hidden: &[u16],
    sequential_logits: &[u16],
    block_logits: &[u16],
    draft_tokens: &[u32],
    draft_logprob_errors: &[f64],
    hidden_diff: Option<&NumericDiff>,
    logits_diff: Option<&NumericDiff>,
) -> Option<String> {
    let directory = env::var_os("SLLM_PHASE84_5_DUMP_DIR").map(PathBuf::from)?;
    if let Err(error) = fs::create_dir_all(&directory) {
        return Some(format!("dump directory {}: {error}", directory.display()));
    }
    let stem = case_name.replace(|character: char| !character.is_ascii_alphanumeric(), "_");
    let files = [
        ("sequential_hidden.bf16le", sequential_hidden),
        ("block_hidden.bf16le", block_hidden),
        ("sequential_logits.bf16le", sequential_logits),
        ("block_logits.bf16le", block_logits),
    ];
    for (name, words) in files {
        if let Err(error) =
            dump_u16_words(&directory.join(format!("phase84_5_{stem}_{name}")), words)
        {
            return Some(error);
        }
    }
    let manifest = serde_json::json!({
        "schema_version": DIAGNOSTIC_SCHEMA,
        "case": case_name,
        "draft_tokens": draft_tokens,
        "draft_logprob_abs_error": draft_logprob_errors,
        "sequential_hidden_sha256": hash_words(sequential_hidden),
        "block_hidden_sha256": hash_words(block_hidden),
        "sequential_logits_sha256": hash_words(sequential_logits),
        "block_logits_sha256": hash_words(block_logits),
        "hidden_diff": hidden_diff,
        "logits_diff": logits_diff,
    });
    let manifest_path = directory.join(format!("phase84_5_{stem}_manifest.json"));
    if let Err(error) = fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
    ) {
        return Some(format!("write {}: {error}", manifest_path.display()));
    }
    Some(directory.display().to_string())
}

fn fixed_selector(counter: u64) -> Result<sllm_core::DeviceTokenSelectorRequestV1, String> {
    let params =
        SamplingParametersV1::new(1.0, 0.95, 0.0, 0.0).map_err(|error| error.to_string())?;
    let chain = SamplerChainV1::new(
        SamplerChainConfigV1::new(params)
            .with_top_k(20)
            .map_err(|error| error.to_string())?,
        &[],
    )
    .map_err(|error| error.to_string())?;
    chain
        .prepare_device_selector(QWEN35_VOCAB_SIZE, None, SAMPLER_SEED, counter)
        .map_err(|error| error.to_string())
}

fn cpu_sampler(prompt: &[i32]) -> Result<(SamplerChainV1, OsSamplingRandom), String> {
    let params =
        SamplingParametersV1::new(1.0, 0.95, 0.0, 0.0).map_err(|error| error.to_string())?;
    let chain = SamplerChainV1::new(
        SamplerChainConfigV1::new(params)
            .with_top_k(20)
            .map_err(|error| error.to_string())?,
        &prompt
            .iter()
            .map(|&token| u32::try_from(token).map_err(|_| "negative prompt token"))
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(|error| error.to_string())?;
    let random = OsSamplingRandom::for_parameters_and_seed(params, Some(SAMPLER_SEED))
        .map_err(|error| error.to_string())?;
    Ok((chain, random))
}

/// Independent fixed-profile p oracle for one already observed token.  This
/// deliberately does not call `SamplerChain::select`; it only reconstructs
/// top-k -> softmax -> top-p -> renormalize and therefore can compare the GPU
/// selector's returned logprob for its own selected token.
fn fixed_profile_logprob(logits: &[f32], token: u32) -> Result<f64, String> {
    let mut candidates = logits
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some((index as u32, value)))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    candidates.truncate(20.min(candidates.len()));
    if candidates.is_empty() {
        return Err("fixed p oracle has no finite candidates".to_owned());
    }
    let max = f64::from(candidates[0].1);
    let mut weighted = candidates
        .into_iter()
        .map(|(id, value)| (id, (f64::from(value) - max).exp()))
        .collect::<Vec<_>>();
    let total = weighted.iter().map(|(_, value)| *value).sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err("fixed p oracle has invalid softmax mass".to_owned());
    }
    for (_, value) in &mut weighted {
        *value /= total;
    }
    weighted.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut cumulative = 0.0;
    let mut keep = 0usize;
    for (_, probability) in &weighted {
        cumulative += *probability;
        keep += 1;
        if cumulative >= 0.95 {
            break;
        }
    }
    weighted.truncate(keep.max(1));
    let retained = weighted.iter().map(|(_, value)| *value).sum::<f64>();
    let probability = weighted
        .into_iter()
        .find(|(id, _)| *id == token)
        .map(|(_, value)| value / retained)
        .ok_or_else(|| "GPU selected token was outside the fixed p support".to_owned())?;
    Ok(probability.ln())
}

fn prime_mtp(
    request: &mut QwenExecutionRequest,
    prompt: &[i32],
    hidden: &[u16],
    hidden_width: usize,
) -> Result<(), String> {
    if prompt.is_empty() || hidden.len() != prompt.len() * hidden_width {
        return Err("MTP prefix shape mismatch".to_owned());
    }
    request
        .prefill_mtp_state_only(prompt[0], &vec![0_u16; hidden_width])
        .map_err(|error| error.to_string())?;
    if prompt.len() > 1 {
        request
            .decode_mtp_state_only_batch(&prompt[1..], &hidden[..(prompt.len() - 1) * hidden_width])
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn prompt_cases(
    artifact: &sllm_core::VerifiedUnslothQwen38Nvfp4,
) -> Result<Vec<(&'static str, Vec<i32>)>, String> {
    let tokenizer = TokenizerFrontendV1::from_unsloth_qwen38_nvfp4(artifact)
        .map_err(|error| error.to_string())?;
    // These are raw source/question prefixes.  They deliberately avoid
    // slicing a chat-template control prefix, which can make the sampler
    // distribution degenerate while still looking like a valid prompt.
    const CODING_SOURCE: &str = r#"fn checked_sum(values: &[i32]) -> Result<i32, &'static str> {
    values.iter().try_fold(0_i32, |sum, value| {
        sum.checked_add(*value).ok_or("overflow")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_overflow() {
        assert!(checked_sum(&[i32::MAX, 1]).is_err());
    }
}

// Review the boundary and preserve the error contract for callers.
// boundary
"#;
    let coding = tokenizer
        .encode(CODING_SOURCE)
        .map_err(|error| error.to_string())?
        .as_slice()
        .iter()
        .copied()
        .map(|token| i32::try_from(token).map_err(|_| "token ID overflow".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    let japanese_source =
        "Rustの所有権とエラー処理を確認し、短い実装例とテストを示してください。".repeat(12);
    let japanese = tokenizer
        .encode(&japanese_source)
        .map_err(|error| error.to_string())?
        .as_slice()
        .iter()
        .copied()
        .map(|token| i32::try_from(token).map_err(|_| "token ID overflow".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    let japanese = japanese
        .get(..129)
        .ok_or("Japanese prompt did not reach the 129-token boundary")?
        .to_vec();
    if coding.len() != 127 {
        return Err(format!(
            "raw coding diagnostic fixture tokenized to {}, expected 127",
            coding.len()
        ));
    }
    Ok(vec![("coding", coding), ("japanese", japanese)])
}

#[allow(clippy::too_many_arguments)] // Fixed fixture inputs are kept separate from execution state.
fn resolve_case(
    resident: &QwenResidentModel,
    graph: &sllm_core::QwenGraph,
    mtp_resident: &QwenResidentModel,
    mtp_graph: &sllm_core::QwenGraph,
    prompt: &[i32],
    inputs: &[i32],
    target_first: i32,
    target_prefix_hidden: &[u16],
    last_target_hidden: &[u16],
    block_tokens: &[i32],
    draft_tokens: &[u32],
    block_hidden: &[u16],
    committed_input_rows: usize,
) -> Result<ResolveReport, String> {
    let mut block_request = resident
        .new_request(graph.clone())
        .map_err(|error| error.to_string())?;
    let block_prefill = block_request
        .prefill_with_mtp_state(prompt)
        .map_err(|error| error.to_string())?;
    let block_prefill_hidden = block_prefill
        .hidden_states_bf16()
        .ok_or("block prefill hidden rows missing")?;
    let block_prefill_hidden_sha256 = hash_words(block_prefill_hidden);
    block_request
        .decode_block_with_mtp_state_and_logits(inputs)
        .map_err(|error| error.to_string())?;
    let resolved = block_request
        .resolve_decode_block(committed_input_rows)
        .map_err(|error| error.to_string())?;

    let mut sequential = resident
        .new_request(graph.clone())
        .map_err(|error| error.to_string())?;
    let sequential_prefill = sequential
        .prefill_with_mtp_state(prompt)
        .map_err(|error| error.to_string())?;
    let sequential_prefill_hidden = sequential_prefill
        .hidden_states_bf16()
        .ok_or("sequential prefill hidden rows missing")?;
    let sequential_prefill_hidden_sha256 = hash_words(sequential_prefill_hidden);
    let prefill_hidden_match = block_prefill_hidden == sequential_prefill_hidden;
    let mut sequential_tokens = Vec::new();
    let mut sequential_hidden = Vec::new();
    for &token in &inputs[..committed_input_rows] {
        let output = sequential
            .decode_with_mtp_state_and_logits(token)
            .map_err(|error| error.to_string())?;
        sequential_hidden.extend_from_slice(
            output
                .hidden_states_bf16()
                .ok_or("sequential hidden row missing")?,
        );
        sequential_tokens.extend_from_slice(output.token_ids());
    }
    let resolved_hidden = resolved
        .hidden_states_bf16()
        .ok_or("resolved hidden rows missing")?;
    let resolved_token_match = resolved.token_ids() == sequential_tokens;
    let hidden_diff = numeric_diff(resolved_hidden, &sequential_hidden)?;
    let kv_a = kv_image::kv_payload(&block_request)?;
    let kv_b = kv_image::kv_payload(&sequential)?;
    let next_input = *block_tokens
        .get(committed_input_rows - 1)
        .ok_or("resolved block token row missing")?;
    let next_a = block_request
        .decode_with_mtp_state_and_logits(next_input)
        .map_err(|error| error.to_string())?;
    let next_b = sequential
        .decode_with_mtp_state_and_logits(next_input)
        .map_err(|error| error.to_string())?;
    let next_hidden_diff = numeric_diff(
        next_a
            .hidden_states_bf16()
            .ok_or("next resolved hidden missing")?,
        next_b
            .hidden_states_bf16()
            .ok_or("next sequential hidden missing")?,
    )?;
    let next_logits_diff = numeric_diff(
        next_a.logits_bf16().ok_or("next resolved logits missing")?,
        next_b
            .logits_bf16()
            .ok_or("next sequential logits missing")?,
    )?;
    let next_token_match = next_a.token_ids() == next_b.token_ids();
    let sequential_length = sequential.committed_length();
    let committed_length = block_request.committed_length();
    let hidden_within = hidden_diff.max_ulp <= NUMERIC_MAX_BF16_ULP;
    let next_within = next_hidden_diff.max_ulp <= NUMERIC_MAX_BF16_ULP
        && next_logits_diff.max_ulp <= NUMERIC_MAX_BF16_ULP;

    let mut mtp_optimized = mtp_resident
        .new_request(mtp_graph.clone())
        .map_err(|error| error.to_string())?;
    let mut mtp_sequential = mtp_resident
        .new_request(mtp_graph.clone())
        .map_err(|error| error.to_string())?;
    let hidden_width = last_target_hidden.len();
    if hidden_width == 0 || target_prefix_hidden.len() != prompt.len() * hidden_width {
        return Err("target MTP hidden row is empty".to_owned());
    }
    // Reproduce the proposal transitions on the optimized owner.  The
    // selected draft IDs are fixed from the earlier observation; only the
    // state transition and its hidden-row input are under test here.
    prime_mtp(
        &mut mtp_optimized,
        prompt,
        target_prefix_hidden,
        hidden_width,
    )?;
    prime_mtp(
        &mut mtp_sequential,
        prompt,
        target_prefix_hidden,
        hidden_width,
    )?;
    let mut proposal_token = target_first;
    let mut proposal_hidden = last_target_hidden.to_vec();
    let mut proposal_hidden_rows = Vec::with_capacity(DRAFT_WIDTH);
    for row in 0..DRAFT_WIDTH {
        let output = mtp_optimized
            .decode_mtp_with_device_selector(
                proposal_token,
                &proposal_hidden,
                &fixed_selector(row as u64)?,
            )
            .map_err(|error| error.to_string())?;
        let selected = output
            .selection()
            .ok_or("optimized rewind proposal selection missing")?;
        let expected = *draft_tokens
            .get(row)
            .ok_or("draft token row missing during selector check")?;
        if selected.token_id != expected {
            return Err(format!(
                "optimized rewind proposal token mismatch at row {row}: selected={} expected={expected}",
                selected.token_id
            ));
        }
        proposal_hidden = output
            .hidden_states_bf16()
            .ok_or("optimized MTP proposal hidden row missing")?
            .to_vec();
        proposal_hidden_rows.push(proposal_hidden.clone());
        proposal_token = i32::try_from(
            *draft_tokens
                .get(row)
                .ok_or("draft token row missing during rewind check")?,
        )
        .map_err(|_| "draft token overflow")?;
    }
    let accepted = committed_input_rows - 1;
    if accepted < DRAFT_WIDTH {
        for _ in 0..DRAFT_WIDTH - committed_input_rows {
            mtp_optimized
                .rewind_last_decode_transition()
                .map_err(|error| error.to_string())?;
        }
    } else {
        let final_hidden_start = (DRAFT_WIDTH - 1) * hidden_width;
        mtp_optimized
            .decode_mtp_state_only_batch(
                &[i32::try_from(draft_tokens[DRAFT_WIDTH - 1])
                    .map_err(|_| "draft token overflow")?],
                &block_hidden[final_hidden_start..final_hidden_start + hidden_width],
            )
            .map_err(|error| error.to_string())?;
    }
    let mut committed_ids = Vec::with_capacity(committed_input_rows);
    committed_ids.push(target_first);
    committed_ids.extend(
        draft_tokens
            .iter()
            .take(accepted)
            .copied()
            .map(|token| i32::try_from(token).map_err(|_| "draft token overflow".to_owned()))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let mut committed_hidden = Vec::with_capacity(committed_ids.len() * hidden_width);
    committed_hidden.extend_from_slice(last_target_hidden);
    if accepted >= 1 {
        committed_hidden.extend_from_slice(&proposal_hidden_rows[0]);
    }
    if accepted >= 2 {
        let final_hidden_start = (DRAFT_WIDTH - 1) * hidden_width;
        committed_hidden.extend_from_slice(
            &block_hidden[final_hidden_start..final_hidden_start + hidden_width],
        );
    }
    for (token, hidden) in committed_ids
        .iter()
        .zip(committed_hidden.chunks(hidden_width))
    {
        mtp_sequential
            .decode_mtp_state_only_batch(&[*token], hidden)
            .map_err(|error| error.to_string())?;
    }
    let optimized_kv = kv_image::kv_payload(&mtp_optimized)?;
    let sequential_kv = kv_image::kv_payload(&mtp_sequential)?;
    let companion_kv_match = optimized_kv == sequential_kv;
    let next_target_hidden_start = (committed_input_rows - 1) * hidden_width;
    let next_target_hidden =
        &block_hidden[next_target_hidden_start..next_target_hidden_start + hidden_width];
    let next_mtp_input = block_tokens[committed_input_rows - 1];
    let optimized_next = mtp_optimized
        .decode_mtp_with_device_selector(
            next_mtp_input,
            next_target_hidden,
            &fixed_selector(DRAFT_WIDTH as u64)?,
        )
        .map_err(|error| error.to_string())?;
    let sequential_next = mtp_sequential
        .decode_mtp_with_device_selector(
            next_mtp_input,
            next_target_hidden,
            &fixed_selector(DRAFT_WIDTH as u64)?,
        )
        .map_err(|error| error.to_string())?;
    let companion_next_hidden_diff = numeric_diff(
        optimized_next
            .hidden_states_bf16()
            .ok_or("optimized next MTP hidden row missing")?,
        sequential_next
            .hidden_states_bf16()
            .ok_or("sequential next MTP hidden row missing")?,
    )?;
    let optimized_next_selection = optimized_next
        .selection()
        .ok_or("optimized next MTP selection missing")?;
    let sequential_next_selection = sequential_next
        .selection()
        .ok_or("sequential next MTP selection missing")?;
    let companion_next_token_match =
        optimized_next_selection.token_id == sequential_next_selection.token_id;
    let companion_next_logprob_abs_error =
        (optimized_next_selection.logprob - sequential_next_selection.logprob).abs();
    let _ = mtp_optimized;
    Ok(ResolveReport {
        accepted_draft_tokens: committed_input_rows - 1,
        committed_input_rows,
        committed_length,
        sequential_length,
        committed_length_match: committed_length == sequential_length,
        resolved_token_match,
        block_prefill_hidden_sha256,
        sequential_prefill_hidden_sha256,
        prefill_hidden_match,
        hidden_max_abs: hidden_diff.max_abs,
        hidden_max_relative: hidden_diff.max_relative,
        hidden_max_bf16_ulp: hidden_diff.max_ulp,
        hidden_within_threshold: hidden_within,
        kv_match: kv_a == kv_b,
        next_token_match,
        next_hidden_max_abs: next_hidden_diff.max_abs,
        next_hidden_max_relative: next_hidden_diff.max_relative,
        next_hidden_max_bf16_ulp: next_hidden_diff.max_ulp,
        next_logits_max_abs: next_logits_diff.max_abs,
        next_logits_max_relative: next_logits_diff.max_relative,
        next_logits_max_bf16_ulp: next_logits_diff.max_ulp,
        next_within_threshold: next_within,
        companion_kv_match,
        companion_next_token_match,
        companion_next_logprob_abs_error,
        companion_next_hidden_max_bf16_ulp: companion_next_hidden_diff.max_ulp,
    })
}

fn run_case(
    resident: &QwenResidentModel,
    mtp_resident: &QwenResidentModel,
    graph: &sllm_core::QwenGraph,
    mtp_graph: &sllm_core::QwenGraph,
    name: &'static str,
    prompt: &[i32],
) -> Result<CaseReport, String> {
    if prompt.is_empty() || prompt.len() + 4 > STATE_CAPACITY as usize {
        return Err("diagnostic prompt does not fit state capacity".to_owned());
    }
    let mut failures = Vec::new();
    let mut target = resident
        .new_request(graph.clone())
        .map_err(|error| error.to_string())?;
    let prefill = target
        .prefill_with_mtp_state(prompt)
        .map_err(|error| error.to_string())?;
    let hidden_width = target
        .mtp_hidden_width()
        .map_err(|error| error.to_string())?;
    let hidden = prefill
        .hidden_states_bf16()
        .ok_or("target prefill hidden rows missing")?
        .to_vec();
    let target_first = *prefill
        .token_ids()
        .last()
        .ok_or("target prefill token missing")?;
    let last_hidden = hidden[(prompt.len() - 1) * hidden_width..].to_vec();
    let target_prefill_hidden_sha256 = hash_words(&hidden);

    let mut mtp_seq = mtp_resident
        .new_request(mtp_graph.clone())
        .map_err(|error| error.to_string())?;
    prime_mtp(&mut mtp_seq, prompt, &hidden, hidden_width)?;
    let mut mtp_opt = mtp_resident
        .new_request(mtp_graph.clone())
        .map_err(|error| error.to_string())?;
    prime_mtp(&mut mtp_opt, prompt, &hidden, hidden_width)?;
    let (mut cpu_chain, mut cpu_random) = cpu_sampler(prompt)?;
    let mut optimized_tokens = Vec::with_capacity(DRAFT_WIDTH);
    let mut optimized_logprobs = Vec::with_capacity(DRAFT_WIDTH);
    let mut sequential_logit_sha256 = Vec::with_capacity(DRAFT_WIDTH);
    let mut optimized_hidden_matches = Vec::with_capacity(DRAFT_WIDTH);
    let mut optimized_hidden_abs = Vec::with_capacity(DRAFT_WIDTH);
    let mut optimized_hidden_relative = Vec::with_capacity(DRAFT_WIDTH);
    let mut optimized_hidden_ulps = Vec::with_capacity(DRAFT_WIDTH);
    let mut cpu_matches = Vec::with_capacity(DRAFT_WIDTH);
    let mut cpu_logprobs = Vec::with_capacity(DRAFT_WIDTH);
    let mut cpu_oracle_logprobs = Vec::with_capacity(DRAFT_WIDTH);
    let mut gpu_logprob_errors = Vec::with_capacity(DRAFT_WIDTH);
    let mut gpu_logprob_within = Vec::with_capacity(DRAFT_WIDTH);
    let mut proposal_token = target_first;
    let mut proposal_hidden = last_hidden.clone();
    let mut replay_token = target_first;
    let mut replay_hidden = last_hidden.clone();
    for row in 0..DRAFT_WIDTH {
        let selector = fixed_selector(row as u64)?;
        let optimized = mtp_opt
            .decode_mtp_with_device_selector(proposal_token, &proposal_hidden, &selector)
            .map_err(|error| error.to_string())?;
        let selection = optimized
            .selection()
            .ok_or("MTP optimized route omitted selection")?;
        let optimized_token = selection.token_id;
        let optimized_hidden = optimized
            .hidden_states_bf16()
            .ok_or("MTP optimized route omitted hidden row")?;
        let sequential = mtp_seq
            .decode_mtp(replay_token, &replay_hidden)
            .map_err(|error| error.to_string())?;
        let sequential_hidden = sequential
            .hidden_states_bf16()
            .ok_or("MTP sequential route omitted hidden row")?;
        let sequential_logits = sequential
            .last_logits()
            .ok_or("MTP sequential route omitted logits")?;
        let diff = numeric_diff(optimized_hidden, sequential_hidden)?;
        let cpu_selection = cpu_chain
            .select(
                sequential.token_ids()[0] as u32,
                Some(sequential_logits),
                &mut cpu_random,
            )
            .map_err(|error| error.to_string())?;
        let cpu_oracle_logprob = fixed_profile_logprob(sequential_logits, optimized_token)?;
        let gpu_logprob_error = (selection.logprob - cpu_oracle_logprob).abs();
        let gpu_logprob_ok = gpu_logprob_error <= LOGPROB_MAX_ABS_ERROR;
        sequential_logit_sha256.push(hash_f32(sequential_logits));
        optimized_tokens.push(optimized_token);
        optimized_logprobs.push(selection.logprob);
        optimized_hidden_matches.push(diff.max_ulp <= NUMERIC_MAX_BF16_ULP);
        optimized_hidden_abs.push(diff.max_abs);
        optimized_hidden_relative.push(diff.max_relative);
        optimized_hidden_ulps.push(diff.max_ulp);
        cpu_matches.push(cpu_selection.token_id == optimized_token);
        cpu_logprobs.push(cpu_selection.logprob);
        cpu_oracle_logprobs.push(cpu_oracle_logprob);
        gpu_logprob_errors.push(gpu_logprob_error);
        gpu_logprob_within.push(gpu_logprob_ok);
        if diff.max_ulp > NUMERIC_MAX_BF16_ULP {
            failures.push(format!(
                "MTP optimized/sequential hidden mismatch at draft row {row}: {} BF16 ULP",
                diff.max_ulp
            ));
        }
        if !gpu_logprob_ok {
            failures.push(format!(
                "GPU selector logprob mismatch at draft row {row}: observed={} oracle={} abs_error={}",
                selection.logprob, cpu_oracle_logprob, gpu_logprob_error
            ));
        }
        proposal_token = i32::try_from(optimized_token).map_err(|_| "draft token overflow")?;
        proposal_hidden = optimized_hidden.to_vec();
        // The sequential comparison intentionally follows the same selected
        // history on the next row.  This isolates route behavior from model
        // prediction differences caused by an unrelated token history.
        replay_token = proposal_token;
        replay_hidden = sequential_hidden.to_vec();
    }
    let inputs = {
        let mut values = vec![target_first];
        values.extend(
            optimized_tokens
                .iter()
                .copied()
                .map(|token| i32::try_from(token).map_err(|_| "draft token overflow".to_owned()))
                .collect::<Result<Vec<_>, _>>()?,
        );
        values
    };
    let mut sequential_target = resident
        .new_request(graph.clone())
        .map_err(|error| error.to_string())?;
    let sequential_prefill = sequential_target
        .prefill_with_mtp_state(prompt)
        .map_err(|error| error.to_string())?;
    let sequential_prefill_hidden = sequential_prefill
        .hidden_states_bf16()
        .ok_or("sequential target prefill hidden missing")?;
    let sequential_prefill_hidden_sha256 = hash_words(sequential_prefill_hidden);
    let mut sequential_tokens = Vec::with_capacity(inputs.len());
    let mut sequential_hidden = Vec::new();
    let mut sequential_logits = Vec::new();
    for &input in &inputs {
        let output = sequential_target
            .decode_with_mtp_state_and_logits(input)
            .map_err(|error| error.to_string())?;
        sequential_tokens.extend_from_slice(output.token_ids());
        sequential_hidden
            .extend_from_slice(output.hidden_states_bf16().ok_or("target hidden missing")?);
        sequential_logits.extend_from_slice(output.logits_bf16().ok_or("target logits missing")?);
    }
    let mut target_block = resident
        .new_request(graph.clone())
        .map_err(|error| error.to_string())?;
    let block_prefill = target_block
        .prefill_with_mtp_state(prompt)
        .map_err(|error| error.to_string())?;
    let block_prefill_hidden = block_prefill
        .hidden_states_bf16()
        .ok_or("target block prefill hidden missing")?;
    let block_prefill_hidden_sha256 = hash_words(block_prefill_hidden);
    let target_prefill_hidden_matches =
        hidden == sequential_prefill_hidden && hidden == block_prefill_hidden;
    if !target_prefill_hidden_matches {
        failures.push(format!(
            "target prefill hidden differs across requests: target={}, sequential={}, block={}",
            target_prefill_hidden_sha256,
            sequential_prefill_hidden_sha256,
            block_prefill_hidden_sha256
        ));
    }
    let block = target_block
        .decode_block_with_mtp_state_and_logits(&inputs)
        .map_err(|error| error.to_string())?;
    let block_tokens = block.token_ids().to_vec();
    let block_hidden = block.hidden_states_bf16().ok_or("block hidden missing")?;
    let block_logits = block.logits_bf16().ok_or("block logits missing")?;
    let hidden_diff = match numeric_diff(block_hidden, &sequential_hidden) {
        Ok(diff) => diff,
        Err(error) => {
            let dump = dump_block_failure(
                name,
                &sequential_hidden,
                block_hidden,
                &sequential_logits,
                block_logits,
                &optimized_tokens,
                &gpu_logprob_errors,
                None,
                None,
            );
            return Err(format!(
                "target block hidden numeric failure: {error}; draft_tokens={optimized_tokens:?}; draft_logprob_errors={gpu_logprob_errors:?}; dump={dump:?}"
            ));
        }
    };
    let logits_diff = match numeric_diff(block_logits, &sequential_logits) {
        Ok(diff) => diff,
        Err(error) => {
            let dump = dump_block_failure(
                name,
                &sequential_hidden,
                block_hidden,
                &sequential_logits,
                block_logits,
                &optimized_tokens,
                &gpu_logprob_errors,
                Some(&hidden_diff),
                None,
            );
            return Err(format!(
                "target block logits numeric failure: {error}; draft_tokens={optimized_tokens:?}; draft_logprob_errors={gpu_logprob_errors:?}; hidden_diff={hidden_diff:?}; dump={dump:?}"
            ));
        }
    };
    let block_token_match = block_tokens == sequential_tokens;
    let hidden_within_threshold = hidden_diff.max_ulp <= NUMERIC_MAX_BF16_ULP;
    let logits_within_threshold = logits_diff.max_ulp <= NUMERIC_MAX_BF16_ULP;
    if !block_token_match || !hidden_within_threshold || !logits_within_threshold {
        let dump = dump_block_failure(
            name,
            &sequential_hidden,
            block_hidden,
            &sequential_logits,
            block_logits,
            &optimized_tokens,
            &gpu_logprob_errors,
            Some(&hidden_diff),
            Some(&logits_diff),
        );
        failures.push(format!(
            "target block mismatch: tokens={}, hidden_diff={hidden_diff:?}, logits_diff={logits_diff:?}, draft_tokens={optimized_tokens:?}, draft_logprob_errors={gpu_logprob_errors:?}, dump={dump:?}",
            block_token_match,
        ));
    }
    let mut resolves = Vec::with_capacity(3);
    for committed_input_rows in 1..=DRAFT_WIDTH + 1 {
        match resolve_case(
            resident,
            graph,
            mtp_resident,
            mtp_graph,
            prompt,
            &inputs,
            target_first,
            &hidden,
            &last_hidden,
            &block_tokens,
            &optimized_tokens,
            block_hidden,
            committed_input_rows,
        ) {
            Ok(resolve) => {
                if !resolve.committed_length_match
                    || !resolve.resolved_token_match
                    || !resolve.prefill_hidden_match
                    || !resolve.hidden_within_threshold
                    || !resolve.kv_match
                    || !resolve.next_token_match
                    || !resolve.next_within_threshold
                    || !resolve.companion_kv_match
                    || !resolve.companion_next_token_match
                    || resolve.companion_next_logprob_abs_error > LOGPROB_MAX_ABS_ERROR
                    || resolve.companion_next_hidden_max_bf16_ulp > NUMERIC_MAX_BF16_ULP
                {
                    failures.push(format!(
                        "resolve rows={committed_input_rows} mismatch: {resolve:?}"
                    ));
                }
                resolves.push(resolve);
            }
            Err(error) => failures.push(format!(
                "resolve rows={committed_input_rows} could not be measured: {error}"
            )),
        }
    }
    drop(target);
    Ok(CaseReport {
        name,
        prompt_tokens: prompt.len(),
        prompt_sha256: hash_tokens(prompt),
        target_prefill_hidden_sha256,
        sequential_prefill_hidden_sha256,
        block_prefill_hidden_sha256,
        target_prefill_hidden_matches,
        target_first,
        draft: DraftReport {
            optimized_tokens,
            optimized_selection_logprobs: optimized_logprobs,
            sequential_logit_sha256,
            optimized_hidden_matches_sequential: optimized_hidden_matches,
            optimized_hidden_max_abs: optimized_hidden_abs,
            optimized_hidden_max_relative: optimized_hidden_relative,
            optimized_hidden_max_bf16_ulp: optimized_hidden_ulps,
            cpu_sampler_token_matches: cpu_matches,
            cpu_sampler_logprobs: cpu_logprobs,
            cpu_oracle_logprobs_for_gpu_token: cpu_oracle_logprobs,
            gpu_logprob_abs_error: gpu_logprob_errors,
            gpu_logprob_within_threshold: gpu_logprob_within,
        },
        target_block: TargetBlockReport {
            inputs,
            sequential_tokens,
            block_tokens,
            token_match: block_token_match,
            hidden_max_abs: hidden_diff.max_abs,
            hidden_max_relative: hidden_diff.max_relative,
            hidden_max_bf16_ulp: hidden_diff.max_ulp,
            hidden_mismatch: hidden_diff.first_mismatch.clone(),
            logits_max_abs: logits_diff.max_abs,
            logits_max_relative: logits_diff.max_relative,
            logits_max_bf16_ulp: logits_diff.max_ulp,
            logits_mismatch: logits_diff.first_mismatch.clone(),
            hidden_within_threshold,
            logits_within_threshold,
            sequential_logits_sha256: hash_words(&sequential_logits),
            block_logits_sha256: hash_words(block_logits),
        },
        resolve_rows: resolves,
        failures,
    })
}

fn hash_f32(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("sha256:{:x}", hash.finalize())
}

fn run_diagnostic() -> Result<Report, String> {
    let target = env::var(TARGET_ENV).map_err(|_| format!("{TARGET_ENV} is required"))?;
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err(format!("{TARGET_ENV} must be gfx1030 or gfx1201"));
    }
    let device_index = env::var(DEVICE_ENV)
        .map_err(|_| format!("{DEVICE_ENV} is required"))?
        .parse::<u32>()
        .map_err(|_| format!("{DEVICE_ENV} must be a u32"))?;
    let model_root = env::var_os(MODEL_ENV)
        .or_else(|| env::var_os(COMPAT_MODEL_ENV))
        .map(PathBuf::from)
        .ok_or_else(|| format!("{MODEL_ENV} or {COMPAT_MODEL_ENV} is required"))?;
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
        Ok(_) => parse_phase83_kv()?,
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
    let companion = env::var_os(PHASE84_MTP_COMPANION_PATH_ENV)
        .map(PathBuf::from)
        .map(|directory| {
            verify_qwen38_mtp_quantized_sidecar(
                &lock,
                &artifact,
                &directory.join("manifest.json"),
                &directory.join("payload.safetensors"),
            )
            .map(Arc::new)
            .map_err(|error| error.to_string())
        })
        .transpose()?;
    if companion.is_some() {
        return Err("Phase84.5 diagnostic requires BF16 MTP companion; unset SLLM_PHASE84_MTP_COMPANION_PATH".to_owned());
    }
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
    let mut cases = Vec::with_capacity(prompts.len());
    let mut failures = Vec::new();
    for (name, prompt) in prompts {
        match run_case(
            &target_resident,
            &mtp_resident,
            &graph,
            &mtp_graph,
            name,
            &prompt,
        ) {
            Ok(case) => {
                failures.extend(
                    case.failures
                        .iter()
                        .map(|failure| format!("case={name}: {failure}")),
                );
                cases.push(case);
            }
            Err(error) => failures.push(format!("case={name} could not be measured: {error}")),
        }
    }
    if cases.is_empty() {
        return Err(format!(
            "all diagnostic cases failed before producing evidence: {failures:?}"
        ));
    }
    let frontend = run_frontend_executor(
        &target_resident,
        &mtp_resident,
        &graph,
        &mtp_graph,
        &cases[0],
        &prompts_for_frontend(&artifact)?[0],
    )?;
    let target_audit = frontend.0;
    let mtp_audit = frontend.1;
    let fallback_used = target_audit.fallback_used() || mtp_audit.fallback_used();
    let all_dispatches_hip = target_audit.all_dispatches_hip() && mtp_audit.all_dispatches_hip();
    if fallback_used || !all_dispatches_hip {
        failures.push("frontend executor observed fallback or non-HIP dispatch".to_owned());
    }
    drop(mtp_resident);
    drop(target_resident);
    let cleanup = session
        .shutdown(Duration::from_secs(30))
        .map_err(|error| error.to_string())?;
    if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("diagnostic cleanup was not empty".to_owned());
    }
    Ok(Report {
        schema_version: DIAGNOSTIC_SCHEMA,
        state: if failures.is_empty() { "PASS" } else { "FAIL" },
        target,
        device_index,
        kv_encoding: kv.canonical_name(),
        mtp_encoding: "bf16",
        sampling: "temperature=1.0 top_p=0.95 top_k=20; GPU selector plus CPU oracle observation",
        draft_width: DRAFT_WIDTH,
        thresholds: ThresholdReport {
            bf16_max_ulp: NUMERIC_MAX_BF16_ULP,
            logprob_max_abs_error: LOGPROB_MAX_ABS_ERROR,
            rationale: "BF16 readback permits at most three representable steps as a provisional diagnostic screen borrowed from an existing GDN fixture; this is not a validated full-model bound; tokens, lengths, and KV payloads remain exact",
            discrete_comparisons: "exact",
        },
        cases,
        failures,
        raw_dump_directory: env::var("SLLM_PHASE84_5_DUMP_DIR").ok(),
        frontend_executor: frontend.2,
        dispatch: DispatchReport {
            target_dispatches: target_audit.kernel_dispatch_count(),
            mtp_dispatches: mtp_audit.kernel_dispatch_count(),
            fallback_used,
            all_dispatches_hip,
        },
    })
}

fn prompts_for_frontend(
    artifact: &sllm_core::VerifiedUnslothQwen38Nvfp4,
) -> Result<Vec<Vec<i32>>, String> {
    Ok(vec![
        super::coding_prompt_fixture(artifact)?.tokens[..17].to_vec(),
    ])
}

fn run_frontend_executor(
    target_resident: &QwenResidentModel,
    mtp_resident: &QwenResidentModel,
    graph: &sllm_core::QwenGraph,
    mtp_graph: &sllm_core::QwenGraph,
    _case: &CaseReport,
    prompt: &[i32],
) -> Result<
    (
        sllm_core::QwenExecutionAudit,
        sllm_core::QwenExecutionAudit,
        FrontendExecutorReport,
    ),
    String,
> {
    let mut executor = QwenMtpGenerationExecutorV1::new_with_draft_width(
        target_resident
            .new_request(graph.clone())
            .map_err(|error| error.to_string())?,
        mtp_resident
            .new_request(mtp_graph.clone())
            .map_err(|error| error.to_string())?,
        DRAFT_WIDTH,
    )
    .map_err(|error| error.to_string())?;
    let prompt_u32 = prompt
        .iter()
        .map(|&token| u32::try_from(token).map_err(|_| "negative prompt token".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    let prefill_selector = fixed_selector(0)?;
    let prefill = executor
        .prefill_with_device_selector(&prompt_u32, &prefill_selector)
        .map_err(|error| error.to_string())?;
    let pending = prefill.device_argmax();
    let decode = executor
        .decode_with_device_selector(pending, &fixed_selector(1)?)
        .map_err(|error| error.to_string())?;
    executor.finish().map_err(|error| error.to_string())?;
    let target_length = executor.target().committed_length();
    let mtp_length = executor.mtp().committed_length();
    let target_audit = executor
        .target()
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    let mtp_audit = executor
        .mtp()
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    let prefill_logprob = prefill
        .device_selection()
        .ok_or("frontend prefill omitted selection")?
        .logprob;
    let decode_logprob = decode
        .device_selection()
        .ok_or("frontend decode omitted selection")?
        .logprob;
    Ok((
        target_audit,
        mtp_audit,
        FrontendExecutorReport {
            prefill_token: pending,
            decode_token: decode.device_argmax(),
            prefill_logprob,
            decode_logprob,
            target_committed_length: target_length,
            mtp_committed_length: mtp_length,
            target_hip_only: true,
            mtp_hip_only: true,
            finished: true,
        },
    ))
}

pub(super) fn run() -> ExitCode {
    match run_diagnostic() {
        Ok(report) => {
            let failed = report.state == "FAIL";
            println!("{}", serde_json::to_string(&report).unwrap_or_else(|error| {
                format!("{{\"schema_version\":\"{DIAGNOSTIC_SCHEMA}\",\"state\":\"FAIL\",\"error\":{error:?}}}")
            }));
            if failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            println!(
                "{{\"schema_version\":\"{DIAGNOSTIC_SCHEMA}\",\"state\":\"FAIL\",\"error\":{}}}",
                serde_json::to_string(&error)
                    .unwrap_or_else(|_| "\"diagnostic failed\"".to_owned())
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{fixed_profile_logprob, numeric_diff};

    #[test]
    fn bf16_ulp_metric_uses_bf16_steps() {
        assert_eq!(numeric_diff(&[0x3f80], &[0x3f81]).unwrap().max_ulp, 1);
        assert_eq!(numeric_diff(&[0xbf80], &[0xbf81]).unwrap().max_ulp, 1);
        assert_eq!(numeric_diff(&[0x0000], &[0x8000]).unwrap().max_ulp, 0);
        assert_eq!(numeric_diff(&[0x7f7d], &[0x7f7e]).unwrap().max_ulp, 1);
        assert!(numeric_diff(&[0x7f80], &[0x7f80]).is_err());
    }

    #[test]
    fn fixed_profile_oracle_returns_selected_support_logprob() {
        let logits = [4.0_f32, 2.0, 1.0, -1.0];
        let logprob = fixed_profile_logprob(&logits, 0).unwrap();
        assert!(logprob.is_finite());
        assert!(logprob < 0.0);
        assert!(fixed_profile_logprob(&logits, 9).is_err());
    }
}
