// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Opt-in bounded Qwen3.5 AQ4 intermediate differential trace.
//!
//! This binary is deliberately separate from the production worker and gateway.  It requires an
//! explicit `--enable-intermediate-trace` flag, reuses the existing hash-bound calibration replay,
//! and retains only fixed coordinate samples/statistics for embedding, every decoder layer, final
//! norm, and LM head.

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ullm_engine::inference_api::{CancellationToken, InferenceRequest, SamplingParams};
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_CONTEXT_LENGTH, QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4CalibrationObserver,
    Qwen35Aq4IntermediateTraceObserver, Qwen35Aq4ModelLoadConfig,
};
use ullm_engine::qwen35_aq4_session::{
    Qwen35Aq4CalibrationReplay, Qwen35Aq4InferenceSession, Qwen35Aq4SessionConfig,
    Qwen35Aq4SessionStatus,
};
use ullm_engine::worker_driver::{InferenceSession, SessionAdvance};

const SCHEMA: &str = "ullm.qwen35_aq4_differential_trace.v1";
const HIDDEN_COORDINATES: [usize; 5] = [0, 1, 1024, 2048, 4095];
const LOGIT_COORDINATES: [usize; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];
const EOS_TOKEN_IDS: [usize; 2] = [248044, 248046];
const MAX_ROW_BYTES: usize = 32 * 1024;

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

struct RowCollector {
    stages: Vec<Value>,
    hidden_len: usize,
    hidden_values: Vec<Option<f32>>,
    hidden_max_abs: f32,
    hidden_sum_sq: f64,
    logit_len: usize,
    logit_values: Vec<Option<f32>>,
    logit_max_abs: f32,
    logit_sum_sq: f64,
}

impl RowCollector {
    fn new() -> Self {
        Self {
            stages: Vec::with_capacity(35),
            hidden_len: 0,
            hidden_values: vec![None; HIDDEN_COORDINATES.len()],
            hidden_max_abs: 0.0,
            hidden_sum_sq: 0.0,
            logit_len: 0,
            logit_values: vec![None; LOGIT_COORDINATES.len()],
            logit_max_abs: 0.0,
            logit_sum_sq: 0.0,
        }
    }

    fn sample_stage(
        stage: &str,
        layer_index: Option<usize>,
        values: &[f32],
    ) -> Result<Value, String> {
        if values.len() <= *HIDDEN_COORDINATES.last().unwrap() {
            return Err(format!("{stage} row is shorter than hidden coordinates"));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{stage} row contains non-finite data"));
        }
        let coordinates = if stage == "lm_head" {
            LOGIT_COORDINATES.to_vec()
        } else {
            HIDDEN_COORDINATES.to_vec()
        };
        let sampled = coordinates
            .iter()
            .map(|index| values[*index])
            .collect::<Vec<_>>();
        let max_abs = values
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        let l2 = values
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        let mut object = json!({
            "stage": stage,
            "sample": {
                "coordinates": coordinates,
                "elements": values.len(),
                "values": sampled,
                "max_abs": max_abs,
                "l2": l2,
            }
        });
        if let Some(layer_index) = layer_index {
            object["layer_index"] = json!(layer_index);
        }
        Ok(object)
    }

    fn finish_record(
        self,
        case_id: &str,
        step: usize,
        context_token_ids_sha256: String,
        predicted_token_id: usize,
    ) -> Result<Value, String> {
        if self.hidden_len <= *HIDDEN_COORDINATES.last().unwrap()
            || self.hidden_values.iter().any(Option::is_none)
            || self.logit_len < LOGIT_COORDINATES.len()
            || self.logit_values.iter().any(Option::is_none)
            || self.stages.len() != 33
        {
            return Err("differential trace row has incomplete bounded stages".to_string());
        }
        let hidden_values = self
            .hidden_values
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();
        let logit_values = self
            .logit_values
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();
        let mut stages = self.stages;
        stages.push(json!({
            "stage": "final_norm",
            "sample": {
                "coordinates": HIDDEN_COORDINATES,
                "elements": self.hidden_len,
                "values": hidden_values,
                "max_abs": self.hidden_max_abs,
                "l2": self.hidden_sum_sq.sqrt(),
            }
        }));
        stages.push(json!({
            "stage": "lm_head",
            "sample": {
                "coordinates": LOGIT_COORDINATES,
                "elements": self.logit_len,
                "values": logit_values,
                "max_abs": self.logit_max_abs,
                "l2": self.logit_sum_sq.sqrt(),
            }
        }));
        let row = json!({
            "case_id": case_id,
            "step": step,
            "context_token_ids_sha256": context_token_ids_sha256,
            "stages": stages,
            "greedy_token_id": predicted_token_id,
        });
        let encoded = serde_json::to_vec(&row).map_err(|error| error.to_string())?;
        if encoded.len() > MAX_ROW_BYTES {
            return Err(format!(
                "differential trace row exceeds {MAX_ROW_BYTES} bytes"
            ));
        }
        Ok(row)
    }
}

impl Qwen35Aq4IntermediateTraceObserver for RowCollector {
    fn observe_embedding(&mut self, values: &[f32]) -> Result<(), String> {
        self.stages
            .push(Self::sample_stage("embedding", None, values)?);
        Ok(())
    }

    fn observe_decoder_layer(&mut self, layer_index: usize, values: &[f32]) -> Result<(), String> {
        self.stages.push(Self::sample_stage(
            "decoder_layer",
            Some(layer_index),
            values,
        )?);
        Ok(())
    }
}

impl Qwen35Aq4CalibrationObserver for RowCollector {
    fn begin(&mut self, hidden_elements: usize, logit_elements: usize) -> Result<(), String> {
        self.hidden_len = hidden_elements;
        self.logit_len = logit_elements;
        if hidden_elements <= *HIDDEN_COORDINATES.last().unwrap()
            || logit_elements < LOGIT_COORDINATES.len()
        {
            return Err("differential trace final row shape is too small".to_string());
        }
        Ok(())
    }

    fn observe_hidden_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        for (offset, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err("differential trace hidden row is non-finite".to_string());
            }
            let index = start + offset;
            self.hidden_max_abs = self.hidden_max_abs.max(value.abs());
            self.hidden_sum_sq += f64::from(value) * f64::from(value);
            for (slot, coordinate) in HIDDEN_COORDINATES.iter().copied().enumerate() {
                if index == coordinate {
                    self.hidden_values[slot] = Some(value);
                }
            }
        }
        Ok(())
    }

    fn observe_logit_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        for (offset, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err("differential trace logit row is non-finite".to_string());
            }
            let index = start + offset;
            self.logit_max_abs = self.logit_max_abs.max(value.abs());
            self.logit_sum_sq += f64::from(value) * f64::from(value);
            for (slot, coordinate) in LOGIT_COORDINATES.iter().copied().enumerate() {
                if index == coordinate {
                    self.logit_values[slot] = Some(value);
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }
}

fn canonical_token_hash(tokens: &[usize]) -> Result<String, String> {
    let mut bytes = serde_json::to_vec(tokens).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path, label: &str) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("failed to read {label}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("failed to decode {label}: {error}"))
}

fn run(
    package_dir: PathBuf,
    cases_path: PathBuf,
    replay_path: PathBuf,
    output: PathBuf,
    device_index: u32,
    enabled: bool,
) -> Result<(), String> {
    if !enabled {
        return Err(
            "intermediate trace is disabled; pass --enable-intermediate-trace explicitly"
                .to_string(),
        );
    }
    if output.exists() || fs::symlink_metadata(&output).is_ok() {
        return Err(format!("refusing to overwrite output {}", output.display()));
    }
    let cases: CasesFile = load_json(&cases_path, "cases")?;
    let replay: ReplayFile = load_json(&replay_path, "replay")?;
    let replay_by_id = replay
        .cases
        .into_iter()
        .map(|case| (case.case_id, case.token_ids))
        .collect::<BTreeMap<_, _>>();
    if cases.cases.is_empty()
        || cases
            .cases
            .iter()
            .any(|case| case.step_count == 0 || case.step_count > 128)
    {
        return Err("cases must contain bounded positive step counts".to_string());
    }
    for case in &cases.cases {
        let replay_tokens = replay_by_id
            .get(&case.case_id)
            .ok_or_else(|| format!("replay is missing {}", case.case_id))?;
        if replay_tokens.len() != case.step_count {
            return Err(format!("replay length differs for {}", case.case_id));
        }
    }
    let max_steps = cases
        .cases
        .iter()
        .map(|case| case.step_count)
        .max()
        .unwrap_or(1);
    let model_config = Qwen35Aq4ModelLoadConfig {
        package_dir: package_dir.clone(),
        device_index,
        expected_architecture: None,
        chunk_bytes: 1024 * 1024,
        context_length: QWEN35_AQ4_CONTEXT_LENGTH,
        kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
        layer_indices: None,
        lm_head_mode: PackageLmHeadMode::GpuResidentF32,
        lm_head_chunk_rows: 8192,
    };
    let mut session_config = Qwen35Aq4SessionConfig::greedy(max_steps, EOS_TOKEN_IDS.to_vec())
        .with_prefill_chunk_tokens(1)?;
    session_config.rotary_dim = 64;
    session_config.rope_base = 10_000_000.0;
    let mut session = Qwen35Aq4InferenceSession::load(model_config, session_config)?;
    let temporary = output.with_extension(format!("incomplete-{}", std::process::id()));
    fs::create_dir_all(&temporary)
        .map_err(|error| format!("failed to create trace temporary root: {error}"))?;
    let result = (|| {
        let payload_path = temporary.join("payload.jsonl");
        let mut payload = BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&payload_path)
                .map_err(|error| format!("failed to create payload: {error}"))?,
        );
        for case in &cases.cases {
            let replay_tokens = replay_by_id
                .get(&case.case_id)
                .ok_or_else(|| format!("replay is missing {}", case.case_id))?;
            let replay_sha =
                Qwen35Aq4CalibrationReplay::source_sequence_sha256_for_tokens(replay_tokens)?;
            let replay = Qwen35Aq4CalibrationReplay::new(replay_sha, replay_tokens.clone())?;
            let request = InferenceRequest::new_with_eos(
                format!("aq4-differential-trace-{}", case.case_id),
                case.prompt_token_ids.clone(),
                case.step_count,
                EOS_TOKEN_IDS.to_vec(),
                SamplingParams::greedy_with_top_k(0, 1),
            );
            session.start_calibration_request(request, CancellationToken::new(), replay)?;
            let mut step = 0usize;
            loop {
                match session.prepare_advance()? {
                    SessionAdvance::PromptProgress { .. } => {}
                    SessionAdvance::Token {
                        prepared, token_id, ..
                    } => {
                        let mut collector = RowCollector::new();
                        session
                            .model_mut()
                            .visit_intermediate_trace(&mut collector)?;
                        session.observe_prepared_calibration(&prepared, &mut collector)?;
                        let context_tokens = if step == 0 {
                            case.prompt_token_ids.clone()
                        } else {
                            let mut context = case.prompt_token_ids.clone();
                            context.extend_from_slice(&replay_tokens[..step]);
                            context
                        };
                        let record = collector.finish_record(
                            &case.case_id,
                            step,
                            canonical_token_hash(&context_tokens)?,
                            token_id,
                        )?;
                        serde_json::to_writer(&mut payload, &record)
                            .map_err(|error| error.to_string())?;
                        payload
                            .write_all(b"\n")
                            .map_err(|error| error.to_string())?;
                        session.publish_calibration_prepared(prepared, |_| Ok(()))?;
                        step += 1;
                        if session.status() == Qwen35Aq4SessionStatus::Terminal {
                            session.finish_and_reset()?;
                            break;
                        }
                    }
                    SessionAdvance::CancellationObserved => {
                        return Err("calibration trace observed cancellation".to_string());
                    }
                }
            }
            if step != case.step_count {
                return Err(format!(
                    "case {} emitted {step} rows expected {}",
                    case.case_id, case.step_count
                ));
            }
        }
        payload.flush().map_err(|error| error.to_string())?;
        let manifest = json!({
            "schema_version": SCHEMA,
            "mode": "aq4_gpu_intermediate_diagnostic",
            "package_dir": package_dir,
            "cases_path": cases_path,
            "replay_path": replay_path,
            "device_index": device_index,
            "rows": cases.cases.iter().map(|case| case.step_count).sum::<usize>(),
            "stage_contract": {"embedding": true, "decoder_layers": 32, "final_norm": true, "lm_head": true, "hidden_coordinates": HIDDEN_COORDINATES, "logit_coordinates": LOGIT_COORDINATES},
            "production_worker_unchanged": true,
        });
        let manifest_path = temporary.join("manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let runtime = json!({"device_index": device_index, "mode": "diagnostic_only", "model_loads": 1, "rows": manifest["rows"]});
        let runtime_path = temporary.join("runtime.json");
        fs::write(
            &runtime_path,
            serde_json::to_vec_pretty(&runtime).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let sums = ["manifest.json", "payload.jsonl", "runtime.json"]
            .iter()
            .map(|name| Ok(format!("{}  {name}\n", sha256_file(&temporary.join(name))?)))
            .collect::<Result<String, String>>()?;
        fs::write(temporary.join("SHA256SUMS"), sums).map_err(|error| error.to_string())?;
        fs::rename(&temporary, &output)
            .map_err(|error| format!("failed to publish trace root: {error}"))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}

fn main() -> ExitCode {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 6 || args.len() > 8 {
        eprintln!(
            "usage: ullm-aq4-differential-trace PACKAGE_DIR CASES_JSON REPLAY_JSON OUTPUT_DIR [DEVICE_INDEX] --enable-intermediate-trace"
        );
        return ExitCode::from(2);
    }
    let enabled = args.iter().any(|arg| arg == "--enable-intermediate-trace");
    let device_index = args
        .get(5)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(1);
    let result = run(
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
        PathBuf::from(&args[4]),
        device_index,
        enabled,
    );
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Qwen3.5 AQ4 differential trace failed: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intermediate_trace_is_explicitly_opt_in() {
        let result = run(
            PathBuf::from("missing-package"),
            PathBuf::from("missing-cases"),
            PathBuf::from("missing-replay"),
            PathBuf::from("missing-output"),
            0,
            false,
        );
        assert_eq!(
            result.expect_err("disabled trace must be rejected"),
            "intermediate trace is disabled; pass --enable-intermediate-trace explicitly"
        );
    }

    #[test]
    fn bounded_row_contract_and_context_hash_are_stable() {
        assert_eq!(
            canonical_token_hash(&[1, 2, 3]).expect("hash should succeed"),
            "9c6bc7ac937d2daffe5ecdbe7eb3a59aba4f43e96a58a99f08838d4ce48c92ba"
        );
        let hidden = vec![0.25_f32; 4096];
        let logits = vec![0.5_f32; 64];
        let mut collector = RowCollector::new();
        collector
            .observe_embedding(&hidden)
            .expect("embedding stage should fit contract");
        for layer in 0..32 {
            collector
                .observe_decoder_layer(layer, &hidden)
                .expect("decoder stage should fit contract");
        }
        collector
            .begin(hidden.len(), logits.len())
            .expect("final shape should fit contract");
        collector
            .observe_hidden_chunk(0, &hidden)
            .expect("hidden summary should fit contract");
        collector
            .observe_logit_chunk(0, &logits)
            .expect("logit summary should fit contract");
        collector.finish().expect("collector finish should succeed");
        let row = collector
            .finish_record("case", 0, canonical_token_hash(&[1, 2, 3]).unwrap(), 42)
            .expect("complete row should fit contract");
        let encoded = serde_json::to_vec(&row).expect("row should encode");
        assert!(encoded.len() <= MAX_ROW_BYTES);
        assert_eq!(row["stages"].as_array().unwrap().len(), 35);
        assert_eq!(row["stages"][0]["stage"], "embedding");
        assert_eq!(row["stages"][33]["stage"], "final_norm");
        assert_eq!(row["stages"][34]["stage"], "lm_head");
    }
}
