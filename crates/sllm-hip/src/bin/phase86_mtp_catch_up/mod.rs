//! Phase 86 forced-column MTP conditioning diagnostic.
//!
//! This is deliberately a diagnostic teacher-forced harness.  Its acceptance
//! count is the companion top-1 match against the frozen column, not the
//! production p/q acceptance decision.  The target block and the MTP state
//! retention/rewind rules are nevertheless the same width-two rules used by
//! the production executor.

use super::*;

pub const MODE_ENV: &str = "SLLM_PHASE86_MODE";
pub const PREPARE_ONLY_ENV: &str = "SLLM_PHASE86_PREPARE_PREFIX_ONLY";
const PREFIX_ENV: &str = "SLLM_PHASE86_PREFIX_FILE";
const MANIFEST_ENV: &str = "SLLM_PHASE86_BENCH_MANIFEST";
const PREFIX_OUTPUT_ENV: &str = "SLLM_PHASE86_PREFIX_OUTPUT";
const CATCH_UP_ENV: &str = "SLLM_QWEN_MTP_CATCH_UP";
const MODE_T: &str = "T";
const MODE_P: &str = "P";
const CATCH_UP_SEPARATE: &str = "separate";
const WIDTH: usize = 2;

fn digest_matches(expected: &str, actual_with_prefix: &str) -> bool {
    let expected = expected
        .strip_prefix("sha256:")
        .unwrap_or(expected)
        .to_ascii_lowercase();
    let actual = actual_with_prefix
        .strip_prefix("sha256:")
        .unwrap_or(actual_with_prefix)
        .to_ascii_lowercase();
    expected == actual
}

fn canonical_sequence_hash(tokens: &[i32]) -> String {
    let canonical = tokens
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("sha256:{:x}", Sha256::digest(canonical.as_bytes()))
}

fn measured_rows_present(blocks: usize, target_rows: usize, draft_rows: usize) -> bool {
    blocks > 0 && target_rows > 0 && draft_rows > 0
}

struct ForcedBlockDecision {
    accepted: usize,
    committed_rows: usize,
    rewind_rows: usize,
    needs_bonus_state: bool,
    next_input_start: usize,
    tail_tokens_omitted: usize,
}

fn forced_block_decision(
    input_start: usize,
    output_len: usize,
    row0_matches: bool,
    row1_matches: bool,
) -> Result<ForcedBlockDecision, String> {
    let accepted = if row0_matches {
        if row1_matches { WIDTH } else { 1 }
    } else {
        0
    };
    forced_block_decision_for_count(input_start, output_len, accepted)
}

fn forced_block_decision_for_count(
    input_start: usize,
    output_len: usize,
    accepted: usize,
) -> Result<ForcedBlockDecision, String> {
    if accepted > WIDTH {
        return Err("Phase86 accepted count exceeds forced width".to_owned());
    }
    let remaining = output_len
        .checked_sub(input_start)
        .ok_or_else(|| "Phase86 block input start exceeds output length".to_owned())?;
    if remaining < WIDTH + 1 {
        return Err("Phase86 forced width-two block has an incomplete tail".to_owned());
    }
    let committed_rows = accepted + 1;
    let rewind_rows = WIDTH.saturating_sub(committed_rows);
    let next_input_start = input_start + committed_rows;
    Ok(ForcedBlockDecision {
        accepted,
        committed_rows,
        rewind_rows,
        needs_bonus_state: accepted == WIDTH,
        next_input_start,
        tail_tokens_omitted: output_len - next_input_start,
    })
}

#[derive(Clone, Serialize)]
pub struct Phase86LogitSummary {
    pub sequence_index: usize,
    pub block_row: usize,
    pub input_token: i32,
    pub forced_token: Option<i32>,
    pub top1_token: usize,
    pub top1_value: f32,
    pub top2_value: f32,
    pub margin: f32,
    pub top1_matches_forced: Option<bool>,
    pub hidden_conditioning: &'static str,
}

#[derive(Serialize)]
pub struct Phase86BlockReport {
    pub input_start: usize,
    pub accepted_draft_tokens: usize,
    pub committed_input_rows: usize,
    pub row0_matches: bool,
    pub row1_matches: bool,
}

#[derive(Serialize)]
pub struct Phase86Audit {
    pub target: AuditReport,
    pub draft: AuditReport,
}

#[derive(Serialize)]
pub struct Phase86Entry {
    pub case_id: String,
    pub seed: u64,
    pub prompt_token_count: usize,
    pub output_prefix_token_count: usize,
    pub prompt_sha256: String,
    pub output_prefix_sha256: String,
    pub full_prefix_sha256: String,
    pub target_hidden_sha256: String,
    pub blocks: usize,
    pub tail_tokens_omitted: usize,
    pub proposed_draft_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub block_reports: Vec<Phase86BlockReport>,
    pub forced_acceptance_contract: &'static str,
    pub catch_up_contract: &'static str,
    pub target_logits_file: String,
    pub target_logits_sha256: String,
    pub draft_logits_file: String,
    pub draft_logits_sha256: String,
    pub target_logit_rows: Vec<Phase86LogitSummary>,
    pub draft_logit_rows: Vec<Phase86LogitSummary>,
    pub audit: Phase86Audit,
}

#[derive(Serialize)]
pub struct Phase86Report {
    pub schema_version: &'static str,
    pub state: &'static str,
    pub mode: &'static str,
    pub catch_up: &'static str,
    pub width: usize,
    pub fixed_column_contract: &'static str,
    pub p_q_relationship: &'static str,
    pub prefix_file: String,
    pub prefix_file_sha256: String,
    pub prefix_provenance: &'static str,
    pub entries: Vec<Phase86Entry>,
    pub all_valid: bool,
}

struct Primed {
    hidden_width: usize,
    last_target_hidden: Vec<u16>,
    target_hidden_digest: Sha256,
}

fn parse_mode() -> Result<&'static str, String> {
    match env::var(MODE_ENV).as_deref() {
        Ok(MODE_T) => Ok(MODE_T),
        Ok(MODE_P) => Ok(MODE_P),
        Ok(value) => Err(format!("{MODE_ENV} must be T or P, got {value}")),
        Err(_) => Err(format!("{MODE_ENV} is required for Phase86")),
    }
}

fn parse_catch_up() -> Result<bool, String> {
    match env::var(CATCH_UP_ENV) {
        Ok(value) if value == CATCH_UP_SEPARATE => Ok(true),
        Ok(value) => Err(format!(
            "{CATCH_UP_ENV} must be unset or {CATCH_UP_SEPARATE}, got {value}"
        )),
        Err(_) => Ok(false),
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf16_rows_to_f32(values: &[u16], row_width: usize) -> Result<Vec<Vec<f32>>, String> {
    if row_width == 0 || values.len() % row_width != 0 {
        return Err("BF16 logits do not contain whole rows".to_owned());
    }
    Ok(values
        .chunks(row_width)
        .map(|row| row.iter().copied().map(bf16_to_f32).collect())
        .collect())
}

fn top_summary(logits: &[f32]) -> Result<(usize, f32, f32, f32), String> {
    if logits.len() < 2 || logits.iter().any(|value| !value.is_finite()) {
        return Err("logit row is empty, too short, or non-finite".to_owned());
    }
    let mut first = 0_usize;
    let mut second = 1_usize;
    for index in 1..logits.len() {
        let value = logits[index];
        let first_is_better = value > logits[first] || (value == logits[first] && index < first);
        if first_is_better {
            second = first;
            first = index;
        } else if index != first
            && (value > logits[second] || (value == logits[second] && index < second))
        {
            second = index;
        }
    }
    Ok((
        first,
        logits[first],
        logits[second],
        logits[first] - logits[second],
    ))
}

fn stage1_prefix_path(output_dir: &Path) -> Result<PathBuf, String> {
    if let Some(manifest) = env::var_os(MANIFEST_ENV).map(PathBuf::from) {
        if !manifest.is_absolute() {
            return Err(format!("{MANIFEST_ENV} must be an absolute path"));
        }
        let output = env::var_os(PREFIX_OUTPUT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| output_dir.join("phase86").join("prefixes.json"));
        if !output.is_absolute() {
            return Err(format!("{PREFIX_OUTPUT_ENV} must be an absolute path"));
        }
        prepare_manifest_prefixes(&manifest, &output)?;
        return Ok(output);
    }
    if let Some(path) = env::var_os(PREFIX_ENV).map(PathBuf::from) {
        if !path.is_absolute() {
            return Err(format!("{PREFIX_ENV} must be an absolute path"));
        }
        return Ok(path);
    }
    super::stage0_required_path(super::PHASE85_A16_PREFIX_FILE_ENV)
}

fn prepare_manifest_prefixes(manifest_path: &Path, output_path: &Path) -> Result<(), String> {
    let manifest_bytes = fs::read(manifest_path)
        .map_err(|error| format!("read Phase86 manifest {}: {error}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("decode Phase86 manifest: {error}"))?;
    let fixture_root = manifest_path
        .parent()
        .and_then(Path::parent)
        .ok_or("Phase86 manifest has no fixture root")?
        .join("fixtures/mtp-bench-v1");
    let artifact_root = PathBuf::from(env::var_os(super::MODEL_ENV).ok_or_else(|| {
        format!(
            "{} is required when preparing Phase86 prefixes",
            super::MODEL_ENV
        )
    })?);
    let artifact = super::verify_unsloth_qwen38_nvfp4(&artifact_root)
        .map_err(|error| format!("verify tokenizer artifact for Phase86 prefixes: {error}"))?;
    let tokenizer = TokenizerFrontendV1::from_unsloth_qwen38_nvfp4(&artifact)
        .map_err(|error| format!("load Phase86 tokenizer: {error}"))?;
    let renderer = Qwen35ChatTemplateV1::from_unsloth_qwen38_nvfp4(&artifact)
        .map_err(|error| format!("load Phase86 renderer: {error}"))?;
    let seed = manifest
        .pointer("/reference_sequence_provenance/sampling/seed")
        .and_then(serde_json::Value::as_u64)
        .ok_or("Phase86 manifest has no reference seed")?;
    let conditions = manifest
        .get("conditions")
        .and_then(serde_json::Value::as_array)
        .ok_or("Phase86 manifest has no conditions")?;
    let tier_a_ids = conditions
        .iter()
        .filter(|condition| condition.get("tier").and_then(serde_json::Value::as_str) == Some("A"))
        .filter_map(|condition| {
            condition
                .get("cond_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .collect::<std::collections::BTreeSet<_>>();
    if tier_a_ids.len() != 26 {
        return Err(format!(
            "Phase86 manifest must provide 26 Tier A IDs, got {}",
            tier_a_ids.len()
        ));
    }
    let mut entries = std::collections::BTreeMap::<String, serde_json::Value>::new();
    let existing_path = env::var_os(PREFIX_ENV)
        .or_else(|| env::var_os(super::PHASE85_A16_PREFIX_FILE_ENV))
        .map(PathBuf::from);
    if let Some(existing_path) = existing_path {
        if existing_path.is_absolute() && existing_path.exists() {
            let existing = super::stage0_read_prefix_file(&existing_path)?;
            for entry in existing.entries {
                if !tier_a_ids.contains(&entry.case_id) {
                    return Err(format!(
                        "existing Phase86 prefix is outside Tier A: {}",
                        entry.case_id
                    ));
                }
                entries.insert(
                    entry.case_id.clone(),
                    serde_json::json!({
                        "case_id": entry.case_id,
                        "seed": entry.seed,
                        "prompt_tokens": entry.prompt_tokens,
                        "output_prefix_tokens": entry.output_prefix_tokens,
                    }),
                );
            }
        }
    }
    for condition in conditions {
        if condition.get("tier").and_then(serde_json::Value::as_str) != Some("A") {
            continue;
        }
        let case_id = condition
            .get("cond_id")
            .and_then(serde_json::Value::as_str)
            .ok_or("Phase86 condition has no cond_id")?;
        let prompt_file = condition
            .get("prompt_file")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("Phase86 condition {case_id} has no prompt_file"))?;
        let sequence_file = condition
            .pointer("/reference_sequences/bf16/sequence_file")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("Phase86 condition {case_id} has no BF16 sequence"))?;
        let prompt_path = fixture_root.join(prompt_file);
        let prompt = fs::read_to_string(&prompt_path)
            .map_err(|error| format!("read Phase86 prompt {case_id}: {error}"))?;
        let expected_prompt_hash = condition
            .get("prompt_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("Phase86 condition {case_id} has no prompt hash"))?;
        if !digest_matches(expected_prompt_hash, &super::hash_text(&prompt)) {
            return Err(format!("Phase86 prompt hash differs for {case_id}"));
        }
        let rendered = renderer
            .render(
                &[Qwen35ChatMessageV1::user(&prompt)],
                Qwen35RenderOptionsV1 {
                    add_generation_prompt: true,
                    thinking: ThinkingModeV1::Disabled,
                },
            )
            .map_err(|error| format!("render Phase86 prompt {case_id}: {error}"))?;
        let prompt_tokens = tokenizer
            .encode(&rendered)
            .map_err(|error| format!("tokenize Phase86 prompt {case_id}: {error}"))?
            .as_slice()
            .iter()
            .copied()
            .map(|token| i32::try_from(token).map_err(|_| format!("token overflows i32: {token}")))
            .collect::<Result<Vec<_>, _>>()?;
        let expected_prompt_tokens = condition
            .get("prompt_tokens_rendered")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("Phase86 condition {case_id} has no rendered length"))?;
        if prompt_tokens.len() != usize::try_from(expected_prompt_tokens).unwrap_or(usize::MAX) {
            return Err(format!(
                "Phase86 rendered prompt length differs for {case_id}"
            ));
        }
        let sequence_path = fixture_root.join(sequence_file);
        let sequence_bytes = fs::read(&sequence_path)
            .map_err(|error| format!("read Phase86 sequence {case_id}: {error}"))?;
        let expected_sequence_hash = condition
            .pointer("/reference_sequences/bf16/sequence_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("Phase86 condition {case_id} has no sequence hash"))?;
        let sequence_hash = format!("sha256:{:x}", Sha256::digest(&sequence_bytes));
        if !digest_matches(expected_sequence_hash, &sequence_hash) {
            return Err(format!("Phase86 sequence file hash differs for {case_id}"));
        }
        let sequence: serde_json::Value = serde_json::from_slice(&sequence_bytes)
            .map_err(|error| format!("decode Phase86 sequence {case_id}: {error}"))?;
        let output = sequence
            .get("committed_token_ids")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("Phase86 sequence {case_id} has no committed tokens"))?
            .iter()
            .map(|token| {
                token
                    .as_i64()
                    .and_then(|value| i32::try_from(value).ok())
                    .ok_or_else(|| format!("Phase86 sequence token is invalid for {case_id}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected_output_count = condition
            .pointer("/reference_sequences/bf16/committed_token_count")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("Phase86 condition {case_id} has no token count"))?;
        if output.len() != usize::try_from(expected_output_count).unwrap_or(usize::MAX) {
            return Err(format!("Phase86 token count differs for {case_id}"));
        }
        for token in prompt_tokens.iter().chain(output.iter()) {
            super::validate_token(*token)?;
        }
        let expected_output_hash = condition
            .pointer("/reference_sequences/bf16/committed_token_sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("Phase86 condition {case_id} has no output hash"))?;
        if !digest_matches(expected_output_hash, &canonical_sequence_hash(&output)) {
            return Err(format!("Phase86 output hash differs for {case_id}"));
        }
        if let Some(existing) = entries.get(case_id) {
            let existing_prompt = existing
                .get("prompt_tokens")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("existing Phase86 prompt is malformed for {case_id}"))?
                .iter()
                .map(|token| {
                    token
                        .as_i64()
                        .and_then(|value| i32::try_from(value).ok())
                        .ok_or_else(|| {
                            format!("existing Phase86 prompt token is invalid for {case_id}")
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let existing_output = existing
                .get("output_prefix_tokens")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("existing Phase86 output is malformed for {case_id}"))?
                .iter()
                .map(|token| {
                    token
                        .as_i64()
                        .and_then(|value| i32::try_from(value).ok())
                        .ok_or_else(|| {
                            format!("existing Phase86 output token is invalid for {case_id}")
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            // The eight gen2 columns are authoritative overrides, not a
            // second copy of the manifest sequences. In particular the
            // Python-review column differs; preserve it rather than replace
            // it or require equality to a different frozen reference.
            if existing_prompt != prompt_tokens || existing_output.is_empty() {
                return Err(format!("existing Phase86 prefix differs for {case_id}"));
            }
            for token in existing_output {
                super::validate_token(token)?;
            }
        }
        entries.entry(case_id.to_owned()).or_insert_with(|| {
            serde_json::json!({
                "case_id": case_id,
                "seed": seed,
                "prompt_tokens": prompt_tokens,
                "output_prefix_tokens": output,
            })
        });
    }
    if entries.len() != tier_a_ids.len() || entries.keys().any(|id| !tier_a_ids.contains(id)) {
        return Err(format!(
            "Phase86 manifest must provide 26 Tier A entries, got {}",
            entries.len()
        ));
    }
    let payload = serde_json::json!({
        "schema": "phase85-a16-mtp-prefix-v1",
        "source": manifest_path.display().to_string(),
        "generated_series": "bf16-frozen-sequence+existing-stage0",
        "entries": entries.into_values().collect::<Vec<_>>(),
    });
    let bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|error| format!("encode Phase86 prefixes: {error}"))?;
    let parent = output_path
        .parent()
        .ok_or("Phase86 prefix output has no parent")?;
    fs::create_dir_all(parent).map_err(|error| format!("create Phase86 prefix output: {error}"))?;
    fs::write(output_path, bytes).map_err(|error| format!("write Phase86 prefix output: {error}"))
}

#[derive(Serialize)]
pub struct Phase86PrepareReport {
    pub schema_version: &'static str,
    pub state: &'static str,
    pub prefix_file: String,
    pub prefix_file_sha256: String,
    pub entry_count: usize,
    pub prefix_provenance: &'static str,
}

pub fn prepare_prefixes_only() -> Result<Phase86PrepareReport, String> {
    let manifest = PathBuf::from(
        env::var_os(MANIFEST_ENV).ok_or_else(|| format!("{MANIFEST_ENV} is required"))?,
    );
    if !manifest.is_absolute() {
        return Err(format!("{MANIFEST_ENV} must be an absolute path"));
    }
    let output = env::var_os(PREFIX_OUTPUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            manifest
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("phase86-prefixes.json")
        });
    if !output.is_absolute() {
        return Err(format!("{PREFIX_OUTPUT_ENV} must be an absolute path"));
    }
    prepare_manifest_prefixes(&manifest, &output)?;
    let prefix = super::stage0_read_prefix_file(&output)?;
    Ok(Phase86PrepareReport {
        schema_version: "phase86-prefix-prep-v1",
        state: "PASS",
        prefix_file: output.display().to_string(),
        prefix_file_sha256: super::stage0_file_sha256(&output)?,
        entry_count: prefix.entries.len(),
        prefix_provenance: "existing Stage0 entries preserved; missing Tier A entries appended from frozen BF16 sequences",
    })
}

fn prime_requests(
    target_request: &mut QwenExecutionRequest,
    draft_request: &mut QwenExecutionRequest,
    prompt: &[i32],
) -> Result<Primed, String> {
    if prompt.is_empty() {
        return Err("Phase86 prompt is empty".to_owned());
    }
    let target_prefill = target_request
        .prefill_with_mtp_state(prompt)
        .map_err(|error| format!("Phase86 target prompt prefill failed: {error}"))?;
    let hidden = target_prefill
        .hidden_states_bf16()
        .ok_or("Phase86 target prompt omitted hidden rows")?;
    let hidden_width = hidden
        .len()
        .checked_div(prompt.len())
        .filter(|width| *width > 0)
        .ok_or("Phase86 target hidden width is invalid")?;
    if hidden.len() != prompt.len() * hidden_width {
        return Err("Phase86 target prompt hidden rows are invalid".to_owned());
    }
    let mut target_hidden_digest = Sha256::new();
    for value in hidden {
        target_hidden_digest.update(value.to_le_bytes());
    }
    draft_request
        .prefill_mtp_state_only(prompt[0], &vec![0_u16; hidden_width])
        .map_err(|error| format!("Phase86 draft prompt prime failed: {error}"))?;
    let capacity = usize::try_from(draft_request.prefill_chunk_capacity())
        .map_err(|_| "Phase86 draft capacity overflowed".to_owned())?
        .max(1);
    let mut index = 1;
    while index < prompt.len() {
        let end = (index + capacity).min(prompt.len());
        let start_hidden = (index - 1) * hidden_width;
        let end_hidden = (end - 1) * hidden_width;
        draft_request
            .decode_mtp_state_only_batch(&prompt[index..end], &hidden[start_hidden..end_hidden])
            .map_err(|error| format!("Phase86 draft prompt prime failed: {error}"))?;
        index = end;
    }
    Ok(Primed {
        hidden_width,
        last_target_hidden: hidden[(prompt.len() - 1) * hidden_width..].to_vec(),
        target_hidden_digest,
    })
}

fn retain_or_catch_up(
    draft_request: &mut QwenExecutionRequest,
    mode: &'static str,
    separate: bool,
    inputs: &[i32],
    target_hidden_before: &[u16],
    target_hidden: &[u16],
    hidden_width: usize,
    decision: &ForcedBlockDecision,
) -> Result<(), String> {
    let committed = decision.committed_rows;
    if mode == MODE_P && separate {
        for _ in 0..WIDTH {
            draft_request
                .rewind_last_decode_transition()
                .map_err(|error| format!("Phase86 catch-up rewind failed: {error}"))?;
        }
        let mut state_hidden = Vec::with_capacity(committed * hidden_width);
        state_hidden.extend_from_slice(target_hidden_before);
        if committed > 1 {
            state_hidden.extend_from_slice(&target_hidden[..(committed - 1) * hidden_width]);
        }
        draft_request
            .decode_mtp_state_only_batch(&inputs[..committed], &state_hidden)
            .map_err(|error| format!("Phase86 separate catch-up failed: {error}"))?;
        return Ok(());
    }
    for _ in 0..decision.rewind_rows {
        draft_request
            .rewind_last_decode_transition()
            .map_err(|error| format!("Phase86 draft rewind failed: {error}"))?;
    }
    if decision.needs_bonus_state {
        let start = (WIDTH - 1) * hidden_width;
        draft_request
            .decode_mtp_state_only_batch(
                &[inputs[WIDTH]],
                &target_hidden[start..start + hidden_width],
            )
            .map_err(|error| format!("Phase86 all-accept state alignment failed: {error}"))?;
    }
    Ok(())
}

fn audit_pair(
    target_request: &QwenExecutionRequest,
    draft_request: &QwenExecutionRequest,
    target: &str,
) -> Result<Phase86Audit, String> {
    let target_audit = target_request
        .audit_snapshot()
        .map_err(|error| format!("Phase86 target audit failed: {error}"))?;
    let draft_audit = draft_request
        .audit_snapshot()
        .map_err(|error| format!("Phase86 draft audit failed: {error}"))?;
    let valid = |audit: &QwenExecutionAudit| {
        audit.selected_backend() == "hip"
            && audit.target() == target
            && audit.submission_count() > 0
            && audit.kernel_dispatch_count() > 0
            && !audit.fallback_used()
            && audit.all_dispatches_hip()
    };
    if !valid(&target_audit) || !valid(&draft_audit) {
        return Err(format!(
            "Phase86 dispatch audit is not HIP-only: target={target_audit:?} draft={draft_audit:?}"
        ));
    }
    Ok(Phase86Audit {
        target: super::audit_report(&target_audit),
        draft: super::audit_report(&draft_audit),
    })
}

fn run_entry_t(
    target_resident: &QwenResidentModel,
    target_graph: &sllm_core::QwenGraph,
    draft_resident: &QwenResidentModel,
    draft_graph: &sllm_core::QwenGraph,
    prefix: &Stage0PrefixInput,
    output_dir: &Path,
    target: &str,
) -> Result<Phase86Entry, String> {
    let mut target_request = target_resident
        .new_request(target_graph.clone())
        .map_err(|error| format!("Phase86 T target request creation failed: {error}"))?;
    let mut draft_request = draft_resident
        .new_request(draft_graph.clone())
        .map_err(|error| format!("Phase86 T draft request creation failed: {error}"))?;
    let primed = prime_requests(
        &mut target_request,
        &mut draft_request,
        &prefix.prompt_tokens,
    )?;
    let hidden_width = primed.hidden_width;
    let mut target_hidden_before = primed.last_target_hidden;
    let mut target_hidden_digest = primed.target_hidden_digest;
    let mut target_rows = Vec::new();
    let mut draft_rows = Vec::new();
    let mut target_logits = Vec::new();
    let mut draft_logits = Vec::new();
    let mut block_reports = Vec::new();
    let output = &prefix.output_prefix_tokens;
    for index in 0..output.len() {
        let input = output[index];
        let target_output = target_request
            .decode_with_mtp_state_and_logits(input)
            .map_err(|error| format!("Phase86 T target row failed at {index}: {error}"))?;
        let target_hidden = target_output
            .hidden_states_bf16()
            .ok_or_else(|| format!("Phase86 T target hidden missing at {index}"))?;
        if target_hidden.len() != hidden_width {
            return Err(format!("Phase86 T target hidden width differs at {index}"));
        }
        for value in target_hidden {
            target_hidden_digest.update(value.to_le_bytes());
        }
        let target_bf16 = target_output
            .logits_bf16()
            .ok_or_else(|| format!("Phase86 T target logits missing at {index}"))?;
        let target_row = bf16_rows_to_f32(target_bf16, QWEN35_VOCAB_SIZE)?
            .into_iter()
            .next()
            .ok_or_else(|| format!("Phase86 T target logits row missing at {index}"))?;
        let forced = output.get(index + 1).copied();
        let (target_top1, target_top1_value, target_top2_value, target_margin) =
            top_summary(&target_row)?;
        target_rows.push(Phase86LogitSummary {
            sequence_index: index,
            block_row: 0,
            input_token: input,
            forced_token: forced,
            top1_token: target_top1,
            top1_value: target_top1_value,
            top2_value: target_top2_value,
            margin: target_margin,
            top1_matches_forced: forced
                .map(|token| target_top1 == usize::try_from(token).unwrap_or(usize::MAX)),
            hidden_conditioning: "target-hidden",
        });
        target_logits.push(target_row);

        let draft_output = draft_request
            .decode_mtp(input, &target_hidden_before)
            .map_err(|error| format!("Phase86 T draft row failed at {index}: {error}"))?;
        let draft_row = draft_output
            .last_logits()
            .ok_or_else(|| format!("Phase86 T draft logits missing at {index}"))?
            .to_vec();
        let (draft_top1, draft_top1_value, draft_top2_value, draft_margin) =
            top_summary(&draft_row)?;
        let matches =
            forced.map(|token| draft_top1 == usize::try_from(token).unwrap_or(usize::MAX));
        draft_rows.push(Phase86LogitSummary {
            sequence_index: index,
            block_row: 0,
            input_token: input,
            forced_token: forced,
            top1_token: draft_top1,
            top1_value: draft_top1_value,
            top2_value: draft_top2_value,
            margin: draft_margin,
            top1_matches_forced: matches,
            hidden_conditioning: "target-hidden",
        });
        draft_logits.push(draft_row);
        block_reports.push(Phase86BlockReport {
            input_start: index,
            accepted_draft_tokens: usize::from(matches == Some(true)),
            committed_input_rows: 1,
            row0_matches: matches == Some(true),
            row1_matches: false,
        });
        target_hidden_before = target_hidden.to_vec();
    }
    let logits_dir = output_dir.join("logits");
    fs::create_dir_all(&logits_dir)
        .map_err(|error| format!("create Phase86 logits directory: {error}"))?;
    let base = format!(
        "{}-seed-{}",
        super::stage0_name(&prefix.case_id),
        prefix.seed
    );
    let target_path = logits_dir.join(format!("{base}-target.f32"));
    let draft_path = logits_dir.join(format!("{base}-draft-T.f32"));
    let target_sha = super::stage0_write_logits(&target_path, &target_logits)?;
    let draft_sha = super::stage0_write_logits(&draft_path, &draft_logits)?;
    let accepted = block_reports
        .iter()
        .map(|block| block.accepted_draft_tokens)
        .sum();
    let audit = audit_pair(&target_request, &draft_request, target)?;
    Ok(Phase86Entry {
        case_id: prefix.case_id.clone(),
        seed: prefix.seed,
        prompt_token_count: prefix.prompt_tokens.len(),
        output_prefix_token_count: prefix.output_prefix_tokens.len(),
        prompt_sha256: super::hash_tokens(&prefix.prompt_tokens),
        output_prefix_sha256: super::hash_tokens(&prefix.output_prefix_tokens),
        full_prefix_sha256: super::hash_tokens(
            &prefix
                .prompt_tokens
                .iter()
                .chain(prefix.output_prefix_tokens.iter())
                .copied()
                .collect::<Vec<_>>(),
        ),
        target_hidden_sha256: format!("sha256:{:x}", target_hidden_digest.finalize()),
        blocks: block_reports.len(),
        tail_tokens_omitted: 0,
        proposed_draft_tokens: output.len().saturating_sub(1),
        accepted_draft_tokens: accepted,
        block_reports,
        forced_acceptance_contract: "diagnostic teacher-forcing: q top1 equals frozen forced token; p/q is not claimed",
        catch_up_contract: "T baseline: every forced position uses target hidden conditioning; no width-two retention experiment",
        target_logits_file: target_path.display().to_string(),
        target_logits_sha256: target_sha,
        draft_logits_file: draft_path.display().to_string(),
        draft_logits_sha256: draft_sha,
        target_logit_rows: target_rows,
        draft_logit_rows: draft_rows,
        audit,
    })
}

fn run_entry(
    target_resident: &QwenResidentModel,
    target_graph: &sllm_core::QwenGraph,
    draft_resident: &QwenResidentModel,
    draft_graph: &sllm_core::QwenGraph,
    prefix: &Stage0PrefixInput,
    output_dir: &Path,
    target: &str,
    mode: &'static str,
    separate: bool,
) -> Result<Phase86Entry, String> {
    let mut target_request = target_resident
        .new_request(target_graph.clone())
        .map_err(|error| format!("Phase86 target request creation failed: {error}"))?;
    let mut draft_request = draft_resident
        .new_request(draft_graph.clone())
        .map_err(|error| format!("Phase86 draft request creation failed: {error}"))?;
    let primed = prime_requests(
        &mut target_request,
        &mut draft_request,
        &prefix.prompt_tokens,
    )?;
    let hidden_width = primed.hidden_width;
    let mut target_hidden_digest = primed.target_hidden_digest;
    let mut target_hidden_before = primed.last_target_hidden;
    let mut cursor = 0_usize;
    let mut blocks = 0_usize;
    let mut accepted_total = 0_usize;
    let mut tail_tokens_omitted = prefix.output_prefix_tokens.len();
    let mut target_rows = Vec::new();
    let mut draft_rows = Vec::new();
    let mut target_logits = Vec::new();
    let mut draft_logits = Vec::new();
    let mut block_reports = Vec::new();
    while cursor + WIDTH < prefix.output_prefix_tokens.len() {
        let inputs = prefix.output_prefix_tokens[cursor..cursor + WIDTH + 1].to_vec();
        let target_output = target_request
            .decode_block_with_mtp_state_and_logits(&inputs)
            .map_err(|error| format!("Phase86 target block failed at {cursor}: {error}"))?;
        let target_hidden = target_output
            .hidden_states_bf16()
            .ok_or_else(|| format!("Phase86 target block omitted hidden rows at {cursor}"))?;
        if target_hidden.len() != (WIDTH + 1) * hidden_width {
            return Err(format!("Phase86 target hidden rows differ at {cursor}"));
        }
        for value in target_hidden {
            target_hidden_digest.update(value.to_le_bytes());
        }
        let target_bf16 = target_output
            .logits_bf16()
            .ok_or_else(|| format!("Phase86 target block omitted logits at {cursor}"))?;
        let target_block_logits = bf16_rows_to_f32(target_bf16, QWEN35_VOCAB_SIZE)?;
        if target_block_logits.len() != WIDTH + 1 {
            return Err(format!("Phase86 target logits rows differ at {cursor}"));
        }
        for row in 0..=WIDTH {
            let forced = (row < WIDTH).then(|| inputs[row + 1]);
            let (top1, top1_value, top2_value, margin) = top_summary(&target_block_logits[row])?;
            target_rows.push(Phase86LogitSummary {
                sequence_index: cursor + row,
                block_row: row,
                input_token: inputs[row],
                forced_token: forced,
                top1_token: top1,
                top1_value,
                top2_value,
                margin,
                top1_matches_forced: forced
                    .map(|token| top1 == usize::try_from(token).unwrap_or(usize::MAX)),
                hidden_conditioning: "target-hidden",
            });
        }
        target_logits.extend(target_block_logits);

        let q0 = draft_request
            .decode_mtp(inputs[0], &target_hidden_before)
            .map_err(|error| format!("Phase86 draft row 0 failed at {cursor}: {error}"))?;
        let q0_hidden = q0
            .hidden_states_bf16()
            .ok_or_else(|| format!("Phase86 draft row 0 omitted hidden at {cursor}"))?
            .to_vec();
        let q0_logits = q0
            .last_logits()
            .ok_or_else(|| format!("Phase86 draft row 0 omitted logits at {cursor}"))?
            .to_vec();
        let q1_hidden_input = if mode == MODE_P {
            &q0_hidden
        } else {
            &target_hidden[..hidden_width]
        };
        let q1 = draft_request
            .decode_mtp(inputs[1], q1_hidden_input)
            .map_err(|error| format!("Phase86 draft row 1 failed at {cursor}: {error}"))?;
        let q1_logits = q1
            .last_logits()
            .ok_or_else(|| format!("Phase86 draft row 1 omitted logits at {cursor}"))?
            .to_vec();
        let mut row_matches = [false; WIDTH];
        for (row, logits) in [q0_logits.as_slice(), q1_logits.as_slice()]
            .into_iter()
            .enumerate()
        {
            let (top1, top1_value, top2_value, margin) = top_summary(logits)?;
            let forced = inputs[row + 1];
            let matches = top1 == usize::try_from(forced).unwrap_or(usize::MAX);
            row_matches[row] = matches;
            draft_rows.push(Phase86LogitSummary {
                sequence_index: cursor + row,
                block_row: row,
                input_token: inputs[row],
                forced_token: Some(forced),
                top1_token: top1,
                top1_value,
                top2_value,
                margin,
                top1_matches_forced: Some(matches),
                hidden_conditioning: if mode == MODE_P && row == 1 {
                    "draft-hidden"
                } else {
                    "target-hidden"
                },
            });
        }
        draft_logits.push(q0_logits);
        draft_logits.push(q1_logits);
        let decision = forced_block_decision(
            cursor,
            prefix.output_prefix_tokens.len(),
            row_matches[0],
            row_matches[1],
        )?;
        tail_tokens_omitted = decision.tail_tokens_omitted;
        let row0_matches = draft_rows[draft_rows.len() - 2]
            .top1_matches_forced
            .unwrap_or(false);
        let row1_matches = draft_rows[draft_rows.len() - 1]
            .top1_matches_forced
            .unwrap_or(false);

        let committed = decision.committed_rows;
        let replay = target_request
            .resolve_decode_block(committed)
            .map_err(|error| format!("Phase86 target replay failed at {cursor}: {error}"))?;
        let replay_hidden = replay
            .hidden_states_bf16()
            .unwrap_or(&target_hidden[..committed * hidden_width]);
        if replay_hidden.len() < committed * hidden_width {
            return Err(format!("Phase86 replay hidden rows differ at {cursor}"));
        }
        let target_next_start = (committed - 1) * hidden_width;
        let next_target_hidden =
            replay_hidden[target_next_start..target_next_start + hidden_width].to_vec();
        let catch_up_hidden_before = target_hidden_before.clone();
        retain_or_catch_up(
            &mut draft_request,
            mode,
            separate,
            &inputs,
            &catch_up_hidden_before,
            target_hidden,
            hidden_width,
            &decision,
        )?;
        block_reports.push(Phase86BlockReport {
            input_start: cursor,
            accepted_draft_tokens: decision.accepted,
            committed_input_rows: committed,
            row0_matches,
            row1_matches,
        });
        target_hidden_before = next_target_hidden;
        cursor = decision.next_input_start;
        accepted_total += decision.accepted;
        blocks += 1;
    }
    let logits_dir = output_dir.join("logits");
    fs::create_dir_all(&logits_dir)
        .map_err(|error| format!("create Phase86 logits directory: {error}"))?;
    let base = format!(
        "{}-seed-{}",
        super::stage0_name(&prefix.case_id),
        prefix.seed
    );
    let target_path = logits_dir.join(format!("{base}-target.f32"));
    let draft_path = logits_dir.join(format!("{base}-draft-{mode}.f32"));
    let target_sha = super::stage0_write_logits(&target_path, &target_logits)?;
    let draft_sha = super::stage0_write_logits(&draft_path, &draft_logits)?;
    let audit = audit_pair(&target_request, &draft_request, target)?;
    Ok(Phase86Entry {
        case_id: prefix.case_id.clone(),
        seed: prefix.seed,
        prompt_token_count: prefix.prompt_tokens.len(),
        output_prefix_token_count: prefix.output_prefix_tokens.len(),
        prompt_sha256: super::hash_tokens(&prefix.prompt_tokens),
        output_prefix_sha256: super::hash_tokens(&prefix.output_prefix_tokens),
        full_prefix_sha256: super::hash_tokens(
            &prefix
                .prompt_tokens
                .iter()
                .chain(prefix.output_prefix_tokens.iter())
                .copied()
                .collect::<Vec<_>>(),
        ),
        target_hidden_sha256: format!("sha256:{:x}", target_hidden_digest.finalize()),
        blocks,
        tail_tokens_omitted,
        proposed_draft_tokens: blocks * WIDTH,
        accepted_draft_tokens: accepted_total,
        block_reports,
        forced_acceptance_contract: "diagnostic teacher-forcing: q top1 equals frozen forced token; p/q is not claimed",
        catch_up_contract: if mode == MODE_P && separate {
            "separate: rewind all proposal rows then target-hidden state-only replay"
        } else {
            "off: production retention/rewind shape, with forced q columns"
        },
        target_logits_file: target_path.display().to_string(),
        target_logits_sha256: target_sha,
        draft_logits_file: draft_path.display().to_string(),
        draft_logits_sha256: draft_sha,
        target_logit_rows: target_rows,
        draft_logit_rows: draft_rows,
        audit,
    })
}

pub fn run(
    target_resident: &QwenResidentModel,
    target_graph: &sllm_core::QwenGraph,
    draft_resident: &QwenResidentModel,
    draft_graph: &sllm_core::QwenGraph,
    output_dir: &Path,
    target: &str,
) -> Result<Phase86Report, String> {
    let mode = parse_mode()?;
    let separate = parse_catch_up()?;
    if mode == MODE_T && separate {
        return Err("Phase86 T baseline does not accept separate catch-up".to_owned());
    }
    let prefix_path = stage1_prefix_path(output_dir)?;
    let prefix_file = super::stage0_read_prefix_file(&prefix_path)?;
    let entries = prefix_file
        .entries
        .iter()
        .map(|prefix| {
            if mode == MODE_T {
                run_entry_t(
                    target_resident,
                    target_graph,
                    draft_resident,
                    draft_graph,
                    prefix,
                    &output_dir.join("phase86").join(mode),
                    target,
                )
            } else {
                run_entry(
                    target_resident,
                    target_graph,
                    draft_resident,
                    draft_graph,
                    prefix,
                    &output_dir.join("phase86").join(mode),
                    target,
                    mode,
                    separate,
                )
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let all_valid = !entries.is_empty()
        && entries.iter().all(|entry| {
            measured_rows_present(
                entry.blocks,
                entry.target_logit_rows.len(),
                entry.draft_logit_rows.len(),
            ) && entry
                .target_logit_rows
                .iter()
                .chain(entry.draft_logit_rows.iter())
                .all(|row| {
                    row.top1_value.is_finite()
                        && row.top2_value.is_finite()
                        && row.margin.is_finite()
                })
                && !entry.audit.target.fallback_used
                && !entry.audit.draft.fallback_used
        });
    Ok(Phase86Report {
        schema_version: "phase86-mtp-catch-up-v1",
        state: if all_valid { "PASS" } else { "FAIL" },
        mode,
        catch_up: if separate { CATCH_UP_SEPARATE } else { "off" },
        width: WIDTH,
        fixed_column_contract: "block=[c[i],c[i+1],c[i+2]], accepted advances i by accepted+1; incomplete tail is omitted",
        p_q_relationship: "forced-column top1 diagnostic; production p/q acceptance and residual replacement are not measured",
        prefix_file: prefix_path.display().to_string(),
        prefix_file_sha256: super::stage0_file_sha256(&prefix_path)?,
        prefix_provenance: if env::var_os(MANIFEST_ENV).is_some() {
            "existing Stage0 entries preserved; missing Tier A entries appended from frozen BF16 sequences"
        } else {
            "existing Stage0 frozen prefix file"
        },
        entries,
        all_valid,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn zero_measurement_rows_cannot_pass_on_prefill_dispatches_alone() {
        assert!(!super::measured_rows_present(0, 0, 0));
        assert!(!super::measured_rows_present(1, 0, 2));
        assert!(!super::measured_rows_present(1, 3, 0));
        assert!(super::measured_rows_present(1, 3, 2));
    }
    #[test]
    fn manifest_hash_accepts_bare_and_prefixed_forms() {
        assert!(super::digest_matches("abcd", "sha256:abcd"));
        assert!(super::digest_matches("sha256:abcd", "sha256:abcd"));
        assert!(!super::digest_matches("abce", "sha256:abcd"));
    }

    #[test]
    fn frozen_sequence_hash_uses_decimal_csv_contract() {
        let hash = super::canonical_sequence_hash(&[1, -2, 3]);
        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash, super::canonical_sequence_hash(&[1, -2, 3]));
        assert_ne!(hash, super::canonical_sequence_hash(&[1, 2, 3]));
    }

    #[test]
    fn forced_width_two_acceptance_is_prefix_ordered() {
        let decision = super::forced_block_decision(0, 3, true, true).unwrap();
        assert_eq!(decision.accepted, 2);
        assert_eq!(decision.committed_rows, 3);
        assert_eq!(decision.rewind_rows, 0);
        assert!(decision.needs_bonus_state);
        assert_eq!(decision.next_input_start, 3);
        assert_eq!(decision.tail_tokens_omitted, 0);
    }

    #[test]
    fn forced_width_two_rejection_does_not_accept_second_row() {
        let decision = super::forced_block_decision(0, 4, false, true).unwrap();
        assert_eq!(decision.accepted, 0);
        assert_eq!(decision.committed_rows, 1);
        assert_eq!(decision.rewind_rows, 1);
        assert!(!decision.needs_bonus_state);
        assert_eq!(decision.next_input_start, 1);
        assert_eq!(decision.tail_tokens_omitted, 3);
    }

    #[test]
    fn forced_width_two_boundary_lengths_preserve_tail_contract() {
        for length in [3_usize, 4, 7, 8] {
            for (row0, row1, expected) in [(false, false, 0), (true, false, 1), (true, true, 2)] {
                let decision = super::forced_block_decision(0, length, row0, row1).unwrap();
                assert_eq!(decision.accepted, expected);
                assert_eq!(
                    decision.tail_tokens_omitted,
                    length - decision.next_input_start
                );
            }
        }
        assert!(super::forced_block_decision(2, 4, true, true).is_err());
        assert!(super::forced_block_decision(0, 2, true, true).is_err());
        assert!(super::forced_block_decision_for_count(0, 3, 3).is_err());
    }
}
