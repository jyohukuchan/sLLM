// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Capture bounded Qwen3.5 AQ4 all-M=1 path-oracle rows.
//!
//! The binary intentionally uses the model runtime's read-only calibration observer. It
//! retains only five hidden values, the first 32 logit values, and a bounded top-k list for each
//! prepared token; a vocabulary or sequence-by-vocabulary matrix is never retained.

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use ullm_engine::inference_api::{CancellationToken, InferenceRequest, SamplingParams};
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_CONTEXT_LENGTH, QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4CalibrationObserver,
    Qwen35Aq4ModelLoadConfig,
};
use ullm_engine::qwen35_aq4_session::{
    QWEN35_AQ4_PREFILL_CHUNK_GRID, Qwen35Aq4CalibrationReplay, Qwen35Aq4InferenceSession,
    Qwen35Aq4SessionConfig, Qwen35Aq4SessionStatus,
};
use ullm_engine::worker_driver::{InferenceSession, SessionAdvance};

const TOP_K: usize = 10;
const LOGIT_SAMPLE_COUNT: usize = 32;
const HIDDEN_SAMPLE_INDICES: [usize; 5] = [0, 1, 1024, 2048, 4095];
const DEFAULT_DEVICE_INDEX: u32 = 0;
const DEFAULT_CHUNK_BYTES: usize = 1024 * 1024;
const DEFAULT_PREFILL_M: usize = 1;
const DEFAULT_ROTARY_DIM: usize = 64;
const DEFAULT_ROPE_BASE: f32 = 10_000_000.0;
const EOS_TOKEN_IDS: [usize; 2] = [248044, 248046];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CasesFile {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    case_id: String,
    prompt_token_ids: Vec<usize>,
    step_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayFile {
    cases: Vec<ReplayCase>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayCase {
    case_id: String,
    token_ids: Vec<usize>,
}

#[derive(Debug, Clone)]
struct TopEntry {
    token_id: usize,
    logit: f32,
}

struct RowCollector {
    hidden_len: usize,
    vocab_len: usize,
    hidden: Vec<Option<f32>>,
    first_logits: Vec<Option<f32>>,
    topk: Vec<TopEntry>,
}

impl RowCollector {
    fn new() -> Self {
        Self {
            hidden_len: 0,
            vocab_len: 0,
            hidden: vec![None; HIDDEN_SAMPLE_INDICES.len()],
            first_logits: vec![None; LOGIT_SAMPLE_COUNT],
            topk: Vec::with_capacity(TOP_K),
        }
    }

    fn finish_record(self, case_id: &str, step: usize) -> Result<Value, String> {
        if self.hidden_len <= *HIDDEN_SAMPLE_INDICES.last().unwrap() {
            return Err(format!(
                "hidden row is shorter than bounded sample: {}",
                self.hidden_len
            ));
        }
        if self.vocab_len < LOGIT_SAMPLE_COUNT || self.topk.len() != TOP_K {
            return Err(format!(
                "logit row is too short for bounded sample: vocab={} topk={}",
                self.vocab_len,
                self.topk.len()
            ));
        }
        let hidden_values = self
            .hidden
            .into_iter()
            .map(|value| value.ok_or_else(|| "hidden bounded sample was not observed".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut logit_indices: Vec<usize> = (0..LOGIT_SAMPLE_COUNT).collect();
        logit_indices.extend(self.topk.iter().map(|entry| entry.token_id));
        logit_indices.sort_unstable();
        logit_indices.dedup();
        let topk_values = self
            .topk
            .iter()
            .map(|entry| (entry.token_id, entry.logit))
            .collect::<std::collections::BTreeMap<_, _>>();
        let logit_values = logit_indices
            .iter()
            .copied()
            .map(|index| {
                if index < LOGIT_SAMPLE_COUNT {
                    self.first_logits[index]
                        .ok_or_else(|| format!("logit sample index {index} was not observed"))
                } else {
                    topk_values
                        .get(&index)
                        .copied()
                        .ok_or_else(|| format!("top-k logit index {index} was not observed"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let topk = self
            .topk
            .iter()
            .map(|entry| json!({"token_id": entry.token_id, "logit": entry.logit}))
            .collect::<Vec<_>>();
        Ok(json!({
            "case_id": case_id,
            "step": step,
            "greedy_token_id": self.topk[0].token_id,
            "hidden_sample": {
                "dtype": "f32",
                "indices": HIDDEN_SAMPLE_INDICES,
                "shape": [self.hidden_len],
                "values": hidden_values,
            },
            "logit_sample": {
                "dtype": "f32",
                "indices": logit_indices,
                "shape": [self.vocab_len],
                "values": logit_values,
            },
            "topk": topk,
        }))
    }
}

impl Qwen35Aq4CalibrationObserver for RowCollector {
    fn begin(&mut self, hidden_elements: usize, logit_elements: usize) -> Result<(), String> {
        self.hidden_len = hidden_elements;
        self.vocab_len = logit_elements;
        if hidden_elements <= *HIDDEN_SAMPLE_INDICES.last().unwrap()
            || logit_elements < LOGIT_SAMPLE_COUNT
        {
            return Err(format!(
                "bounded path oracle shapes are too small: hidden={} vocab={}",
                hidden_elements, logit_elements
            ));
        }
        Ok(())
    }

    fn observe_hidden_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        for (offset, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err("non-finite hidden value observed".to_string());
            }
            let index = start
                .checked_add(offset)
                .ok_or_else(|| "hidden observer index overflows".to_string())?;
            for (slot, sample_index) in HIDDEN_SAMPLE_INDICES.iter().copied().enumerate() {
                if index == sample_index {
                    self.hidden[slot] = Some(value);
                }
            }
        }
        Ok(())
    }

    fn observe_logit_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        for (offset, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err("non-finite logit value observed".to_string());
            }
            let token_id = start
                .checked_add(offset)
                .ok_or_else(|| "logit observer index overflows".to_string())?;
            if token_id < LOGIT_SAMPLE_COUNT {
                self.first_logits[token_id] = Some(value);
            }
            self.topk.push(TopEntry {
                token_id,
                logit: value,
            });
            self.topk.sort_by(|left, right| {
                right
                    .logit
                    .total_cmp(&left.logit)
                    .then_with(|| left.token_id.cmp(&right.token_id))
            });
            if self.topk.len() > TOP_K {
                self.topk.truncate(TOP_K);
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }
}

fn sha256_tokens(token_ids: &[usize]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"[");
    for (index, token_id) in token_ids.iter().copied().enumerate() {
        if index != 0 {
            digest.update(b",");
        }
        digest.update(token_id.to_string().as_bytes());
    }
    digest.update(b"]\n");
    format!("{:x}", digest.finalize())
}

fn parse_usize(raw: Option<&String>, default: usize, label: &str) -> Result<usize, String> {
    let value = raw.map_or(Ok(default), |value| {
        value
            .parse::<usize>()
            .map_err(|error| format!("invalid {label} {value:?}: {error}"))
    })?;
    if value == 0 {
        return Err(format!("{label} must be positive"));
    }
    Ok(value)
}

fn parse_u32(raw: Option<&String>, default: u32, label: &str) -> Result<u32, String> {
    raw.map_or(Ok(default), |value| {
        value
            .parse::<u32>()
            .map_err(|error| format!("invalid {label} {value:?}: {error}"))
    })
}

fn parse_f32(raw: Option<&String>, default: f32, label: &str) -> Result<f32, String> {
    let value = raw.map_or(Ok(default), |value| {
        value
            .parse::<f32>()
            .map_err(|error| format!("invalid {label} {value:?}: {error}"))
    })?;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{label} must be finite and positive"));
    }
    Ok(value)
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &PathBuf, label: &str) -> Result<T, String> {
    let raw =
        fs::read_to_string(path).map_err(|error| format!("failed to read {label}: {error}"))?;
    serde_json::from_str(&raw).map_err(|error| format!("failed to decode {label}: {error}"))
}

fn run(
    package_dir: PathBuf,
    cases_path: PathBuf,
    replay_path: PathBuf,
    device_index: u32,
    chunk_bytes: usize,
    prefill_m: usize,
    rotary_dim: usize,
    rope_base: f32,
) -> Result<(), String> {
    let cases: CasesFile = load_json(&cases_path, "cases")?;
    if cases.cases.is_empty() {
        return Err("cases must not be empty".to_string());
    }
    let replay: ReplayFile = load_json(&replay_path, "replay")?;
    let replay_by_id = replay
        .cases
        .into_iter()
        .map(|case| (case.case_id, case.token_ids))
        .collect::<std::collections::BTreeMap<_, _>>();
    let max_steps = cases
        .cases
        .iter()
        .map(|case| case.step_count)
        .max()
        .ok_or_else(|| "cases must contain one row".to_string())?;
    for case in &cases.cases {
        let replay_tokens = replay_by_id
            .get(&case.case_id)
            .ok_or_else(|| format!("replay is missing case {}", case.case_id))?;
        if replay_tokens.len() != case.step_count {
            return Err(format!(
                "replay case {} has {} tokens, expected {}",
                case.case_id,
                replay_tokens.len(),
                case.step_count
            ));
        }
    }
    if !QWEN35_AQ4_PREFILL_CHUNK_GRID.contains(&prefill_m) {
        return Err(format!(
            "prefill M must be one of {:?}, got {}",
            QWEN35_AQ4_PREFILL_CHUNK_GRID, prefill_m
        ));
    }
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        return Err("rotary dim must be a positive even number".to_string());
    }
    let model_config = Qwen35Aq4ModelLoadConfig {
        package_dir,
        device_index,
        expected_architecture: None,
        chunk_bytes,
        context_length: QWEN35_AQ4_CONTEXT_LENGTH,
        kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
        layer_indices: None,
        lm_head_mode: PackageLmHeadMode::GpuResidentF32,
        lm_head_chunk_rows: 8192,
    };
    let mut session_config = Qwen35Aq4SessionConfig::greedy(max_steps, EOS_TOKEN_IDS.to_vec())
        .with_prefill_chunk_tokens(prefill_m)?;
    session_config.rotary_dim = rotary_dim;
    session_config.rope_base = rope_base;
    let mut session = Qwen35Aq4InferenceSession::load(model_config, session_config)?;
    for case in cases.cases {
        let replay_tokens = replay_by_id
            .get(&case.case_id)
            .ok_or_else(|| format!("replay is missing case {}", case.case_id))?;
        let request = InferenceRequest::new_with_eos(
            format!("aq4-p2-path-oracle-{}", case.case_id),
            case.prompt_token_ids,
            case.step_count,
            EOS_TOKEN_IDS.to_vec(),
            SamplingParams::greedy_with_top_k(0, 1),
        );
        let replay =
            Qwen35Aq4CalibrationReplay::new(sha256_tokens(replay_tokens), replay_tokens.clone())?;
        session.start_calibration_request(request, CancellationToken::new(), replay)?;
        let mut step = 0usize;
        loop {
            match session.prepare_advance()? {
                SessionAdvance::PromptProgress { .. } => {}
                SessionAdvance::Token { prepared, .. } => {
                    let mut collector = RowCollector::new();
                    session.observe_prepared_calibration(&prepared, &mut collector)?;
                    let record = collector.finish_record(&case.case_id, step)?;
                    println!(
                        "{}",
                        serde_json::to_string(&record).map_err(|error| error.to_string())?
                    );
                    session.publish_calibration_prepared(prepared, |_| Ok(()))?;
                    step = step
                        .checked_add(1)
                        .ok_or_else(|| "oracle step counter overflowed".to_string())?;
                    if session.status() == Qwen35Aq4SessionStatus::Terminal {
                        session.finish_and_reset()?;
                        break;
                    }
                }
                SessionAdvance::CancellationObserved => {
                    return Err(
                        "calibration path oracle unexpectedly observed cancellation".to_string()
                    );
                }
            }
        }
        if step != case.step_count {
            return Err(format!(
                "case {} emitted {} rows, expected {}",
                case.case_id, step, case.step_count
            ));
        }
    }
    session.shutdown()
}

fn main() -> ExitCode {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 4 || args.len() > 10 {
        eprintln!(
            "usage: ullm-aq4-p2-path-oracle PACKAGE_DIR CASES_JSON REPLAY_JSON [DEVICE_INDEX] [CHUNK_BYTES] [PREFILL_M] [ROTARY_DIM] [ROPE_BASE]"
        );
        return ExitCode::from(2);
    }
    let result = (|| {
        let device_index = parse_u32(args.get(4), DEFAULT_DEVICE_INDEX, "device index")?;
        let chunk_bytes = parse_usize(args.get(5), DEFAULT_CHUNK_BYTES, "chunk bytes")?;
        let prefill_m = parse_usize(args.get(6), DEFAULT_PREFILL_M, "prefill M")?;
        let rotary_dim = parse_usize(args.get(7), DEFAULT_ROTARY_DIM, "rotary dim")?;
        let rope_base = parse_f32(args.get(8), DEFAULT_ROPE_BASE, "rope base")?;
        run(
            PathBuf::from(&args[1]),
            PathBuf::from(&args[2]),
            PathBuf::from(&args[3]),
            device_index,
            chunk_bytes,
            prefill_m,
            rotary_dim,
            rope_base,
        )
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Qwen3.5 AQ4 P2 path oracle failed: {error}");
            ExitCode::from(1)
        }
    }
}
