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
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ullm_engine::inference_api::{CancellationToken, InferenceRequest, SamplingParams};
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_layer_runtime::PackageLinearAttnIntermediateTraceStage;
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_CONTEXT_LENGTH, QWEN35_AQ4_KV_BLOCK_SIZE, QWEN35_AQ4_LINEAR_STAGE_TRACE_CHUNK_BYTES,
    Qwen35Aq4CalibrationObserver, Qwen35Aq4IntermediateTraceObserver, Qwen35Aq4ModelLoadConfig,
};
use ullm_engine::qwen35_aq4_session::{
    Qwen35Aq4CalibrationReplay, Qwen35Aq4InferenceSession, Qwen35Aq4SessionConfig,
    Qwen35Aq4SessionStatus,
};
use ullm_engine::worker_driver::{InferenceSession, SessionAdvance};

const SCHEMA: &str = "ullm.qwen35_aq4_differential_trace.v1";
const KERNEL_STAGE_SCHEMA: &str = "ullm.qwen35_aq4_kernel_stage_trace.v1";
const KERNEL_STAGE_LAYER_INDEX: usize = 0;
const KERNEL_STAGE_PAYLOAD_FILE: &str = "kernel-stages.jsonl";
const KERNEL_STAGE_STREAM_FILE: &str = "kernel-stages.f32le";
const HIDDEN_COORDINATES: [usize; 5] = [0, 1, 1024, 2048, 4095];
const LOGIT_COORDINATES: [usize; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];
const EOS_TOKEN_IDS: [usize; 2] = [248044, 248046];
const MAX_ROW_BYTES: usize = 32 * 1024;
const MAX_CASES: usize = 3;
const MAX_ROWS: usize = 3;
const MAX_INPUT_BYTES: u64 = 1024 * 1024;
const MAX_OUTPUT_BYTES: u64 = 96 * 1024;
const MAX_KERNEL_STAGE_STREAM_BYTES: u64 = 16 * 1024 * 1024;
const MAX_IDENTITY_FILE_BYTES: u64 = 256 * 1024 * 1024;
const EMBEDDED_BUILD_GIT_COMMIT: Option<&str> = option_env!("ULLM_BUILD_GIT_COMMIT");
const REQUIRED_LINEAR_STAGE_ENV: [&str; 7] = [
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
];
const DISALLOWED_LINEAR_STAGE_ENV: [&str; 2] = [
    "ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING",
    "ULLM_DISABLE_AQ4_MATVEC_QKV_Z_GATE_BETA",
];
const EXPECTED_CASES: [(&str, usize, &str, usize, &'static [usize]); 2] = [
    (
        "fixture-prompt-0",
        3,
        "42ea52c728680a54afafd1c1e1e45f13300c3ceb962f320f3900196a0c46215c",
        2,
        &[220, 16],
    ),
    (
        "fixture-prompt-1",
        2,
        "3bca9e21e3b6f741ed412f91d7696146c254ff68bd9be9ca41b1d172eb3549e6",
        1,
        &[15],
    ),
];

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

#[derive(Clone)]
struct LinearStageTraceContext {
    case_id: String,
    step: usize,
    context_length: usize,
    context_token_ids_sha256: String,
}

struct ActiveLinearStage {
    stage: PackageLinearAttnIntermediateTraceStage,
    elements: usize,
    next_start: usize,
    coordinates: Vec<usize>,
    values: Vec<Option<f32>>,
    max_abs: f32,
    sum_sq: f64,
}

struct LinearStageCollector {
    layer_index: usize,
    completed: Vec<Value>,
    active: Option<ActiveLinearStage>,
}

impl LinearStageCollector {
    fn new(layer_index: usize) -> Self {
        Self {
            layer_index,
            completed: Vec::with_capacity(PackageLinearAttnIntermediateTraceStage::ORDERED.len()),
            active: None,
        }
    }

    fn begin(
        &mut self,
        layer_index: usize,
        stage: PackageLinearAttnIntermediateTraceStage,
        elements: usize,
    ) -> Result<(), String> {
        if layer_index != self.layer_index {
            return Err(format!(
                "kernel-stage trace received layer {layer_index}, expected {}",
                self.layer_index
            ));
        }
        if elements == 0 {
            return Err(format!("kernel-stage {} has zero elements", stage.label()));
        }
        if self.active.is_some() {
            return Err(
                "kernel-stage trace began a stage before finishing the prior stage".to_string(),
            );
        }
        let expected = PackageLinearAttnIntermediateTraceStage::ORDERED
            .get(self.completed.len())
            .copied()
            .ok_or_else(|| "kernel-stage trace has more stages than its contract".to_string())?;
        if stage != expected {
            return Err(format!(
                "kernel-stage trace order differs: got {} expected {}",
                stage.label(),
                expected.label()
            ));
        }
        let coordinates = kernel_stage_coordinates(elements);
        self.active = Some(ActiveLinearStage {
            stage,
            elements,
            next_start: 0,
            values: vec![None; coordinates.len()],
            coordinates,
            max_abs: 0.0,
            sum_sq: 0.0,
        });
        Ok(())
    }

    fn observe_chunk(
        &mut self,
        layer_index: usize,
        stage: PackageLinearAttnIntermediateTraceStage,
        start: usize,
        values: &[f32],
    ) -> Result<(), String> {
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| "kernel-stage trace received a chunk without a stage".to_string())?;
        if layer_index != self.layer_index || active.stage != stage {
            return Err("kernel-stage trace chunk does not match its active stage".to_string());
        }
        if start != active.next_start {
            return Err(format!(
                "kernel-stage {} chunk begins at {start}, expected {}",
                stage.label(),
                active.next_start
            ));
        }
        let end = start
            .checked_add(values.len())
            .ok_or_else(|| "kernel-stage trace chunk end overflows".to_string())?;
        if end > active.elements {
            return Err(format!(
                "kernel-stage {} chunk exceeds its declared element count",
                stage.label()
            ));
        }
        for (offset, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "kernel-stage {} contains non-finite data",
                    stage.label()
                ));
            }
            let index = start + offset;
            active.max_abs = active.max_abs.max(value.abs());
            active.sum_sq += f64::from(value) * f64::from(value);
            for (slot, coordinate) in active.coordinates.iter().copied().enumerate() {
                if index == coordinate {
                    active.values[slot] = Some(value);
                }
            }
        }
        active.next_start = end;
        Ok(())
    }

    fn finish(
        &mut self,
        layer_index: usize,
        stage: PackageLinearAttnIntermediateTraceStage,
    ) -> Result<(), String> {
        let active = self
            .active
            .take()
            .ok_or_else(|| "kernel-stage trace finished a stage that was not active".to_string())?;
        if layer_index != self.layer_index || active.stage != stage {
            return Err("kernel-stage trace finish does not match its active stage".to_string());
        }
        if active.next_start != active.elements || active.values.iter().any(Option::is_none) {
            return Err(format!(
                "kernel-stage {} did not receive every expected value",
                stage.label()
            ));
        }
        self.completed.push(json!({
            "stage": stage.label(),
            "layer_index": layer_index,
            "sample": {
                "coordinates": active.coordinates,
                "elements": active.elements,
                "values": active.values.into_iter().map(Option::unwrap).collect::<Vec<_>>(),
                "max_abs": active.max_abs,
                "l2": active.sum_sq.sqrt(),
            },
        }));
        Ok(())
    }

    fn finish_record(self) -> Result<Vec<Value>, String> {
        if self.active.is_some()
            || self.completed.len() != PackageLinearAttnIntermediateTraceStage::ORDERED.len()
        {
            return Err("kernel-stage trace record has incomplete stages".to_string());
        }
        Ok(self.completed)
    }
}

fn kernel_stage_coordinates(elements: usize) -> Vec<usize> {
    let mut coordinates = Vec::with_capacity(8);
    for coordinate in [0, 1, 31, 127, 1024, 2048, 4095, elements.saturating_sub(1)] {
        if coordinate < elements && !coordinates.contains(&coordinate) {
            coordinates.push(coordinate);
        }
    }
    coordinates
}

struct KernelStageFrameWriter {
    output: BufWriter<File>,
    bytes: u64,
    active: Option<(PackageLinearAttnIntermediateTraceStage, usize, usize)>,
}

impl KernelStageFrameWriter {
    fn create(path: &Path) -> Result<Self, String> {
        Ok(Self {
            output: BufWriter::new(
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(path)
                    .map_err(|error| format!("failed to create kernel-stage stream: {error}"))?,
            ),
            bytes: 0,
            active: None,
        })
    }

    fn write_limited(&mut self, bytes: &[u8]) -> Result<(), String> {
        let next = self
            .bytes
            .checked_add(
                u64::try_from(bytes.len())
                    .map_err(|_| "kernel-stage stream byte count does not fit u64".to_string())?,
            )
            .ok_or_else(|| "kernel-stage stream byte count overflows".to_string())?;
        if next > MAX_KERNEL_STAGE_STREAM_BYTES {
            return Err(format!(
                "kernel-stage stream exceeds {MAX_KERNEL_STAGE_STREAM_BYTES} bytes"
            ));
        }
        self.output
            .write_all(bytes)
            .map_err(|error| error.to_string())?;
        self.bytes = next;
        Ok(())
    }

    fn begin(
        &mut self,
        context: &LinearStageTraceContext,
        stage: PackageLinearAttnIntermediateTraceStage,
        elements: usize,
    ) -> Result<(), String> {
        if self.active.is_some() {
            return Err(
                "kernel-stage stream began a stage before finishing the prior stage".to_string(),
            );
        }
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "kernel-stage stream stage byte length overflows".to_string())?;
        let timestep = context
            .context_length
            .checked_sub(1)
            .ok_or_else(|| "kernel-stage stream context length is zero".to_string())?;
        let header = json!({
            "kind": "stage",
            "case_id": context.case_id,
            "step": context.step,
            "context_token_ids_sha256": context.context_token_ids_sha256,
            "context_length": context.context_length,
            "timestep": timestep,
            "stage": stage.label(),
            "dtype": "f32le",
            "shape": [elements],
            "bytes": bytes,
        });
        let mut encoded = serde_json::to_vec(&header).map_err(|error| error.to_string())?;
        encoded.push(b'\n');
        self.write_limited(&encoded)?;
        self.active = Some((stage, elements, 0));
        Ok(())
    }

    fn observe_chunk(
        &mut self,
        stage: PackageLinearAttnIntermediateTraceStage,
        start: usize,
        values: &[f32],
    ) -> Result<(), String> {
        let (active_stage, elements, next_start) = *self.active.as_ref().ok_or_else(|| {
            "kernel-stage stream received a chunk without an active stage".to_string()
        })?;
        if active_stage != stage || next_start != start {
            return Err("kernel-stage stream chunk order differs from its header".to_string());
        }
        let end = start
            .checked_add(values.len())
            .ok_or_else(|| "kernel-stage stream chunk end overflows".to_string())?;
        if end > elements {
            return Err("kernel-stage stream chunk exceeds its declared shape".to_string());
        }
        let mut encoded = Vec::with_capacity(
            values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "kernel-stage stream chunk byte length overflows".to_string())?,
        );
        for value in values {
            encoded.extend_from_slice(&value.to_le_bytes());
        }
        self.write_limited(&encoded)?;
        self.active = Some((active_stage, elements, end));
        Ok(())
    }

    fn finish(&mut self, stage: PackageLinearAttnIntermediateTraceStage) -> Result<(), String> {
        let (active_stage, elements, next_start) = self.active.take().ok_or_else(|| {
            "kernel-stage stream finished a stage that was not active".to_string()
        })?;
        if active_stage != stage || next_start != elements {
            return Err(
                "kernel-stage stream ended before its declared tensor was complete".to_string(),
            );
        }
        Ok(())
    }

    fn require_idle(&self) -> Result<(), String> {
        if self.active.is_some() {
            return Err("kernel-stage stream still has an active tensor".to_string());
        }
        Ok(())
    }

    fn finish_stream(mut self) -> Result<u64, String> {
        self.require_idle()?;
        self.write_limited(b"{\"kind\":\"end\"}\n")?;
        self.output.flush().map_err(|error| error.to_string())?;
        self.output
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
        Ok(self.bytes)
    }
}

struct FinishedTraceRow {
    payload: Value,
    kernel_stage: Option<Value>,
}

struct RowCollector<'a> {
    stages: Vec<Value>,
    hidden_len: usize,
    hidden_values: Vec<Option<f32>>,
    hidden_max_abs: f32,
    hidden_sum_sq: f64,
    logit_len: usize,
    logit_values: Vec<Option<f32>>,
    logit_max_abs: f32,
    logit_sum_sq: f64,
    linear_stage_context: Option<LinearStageTraceContext>,
    linear_stage_collector: Option<LinearStageCollector>,
    linear_stage_writer: Option<&'a mut KernelStageFrameWriter>,
}

impl<'a> RowCollector<'a> {
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
            linear_stage_context: None,
            linear_stage_collector: None,
            linear_stage_writer: None,
        }
    }

    fn with_linear_stage_trace(
        context: LinearStageTraceContext,
        writer: &'a mut KernelStageFrameWriter,
    ) -> Self {
        let mut collector = Self::new();
        collector.linear_stage_collector =
            Some(LinearStageCollector::new(KERNEL_STAGE_LAYER_INDEX));
        collector.linear_stage_context = Some(context);
        collector.linear_stage_writer = Some(writer);
        collector
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
        mut self,
        case_id: &str,
        step: usize,
        context_length: usize,
        context_token_ids_sha256: String,
        predicted_token_id: usize,
    ) -> Result<FinishedTraceRow, String> {
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
            "context_length": context_length,
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
        let kernel_stage = match self.linear_stage_collector.take() {
            Some(stage_collector) => {
                let context = self.linear_stage_context.take().ok_or_else(|| {
                    "kernel-stage trace collector has no row identity context".to_string()
                })?;
                if context.case_id != case_id
                    || context.step != step
                    || context.context_length != context_length
                    || context.context_token_ids_sha256 != context_token_ids_sha256
                {
                    return Err(
                        "kernel-stage trace row identity changed during collection".to_string()
                    );
                }
                if let Some(writer) = self.linear_stage_writer.as_deref_mut() {
                    writer.require_idle()?;
                } else {
                    return Err("kernel-stage trace collector has no frame writer".to_string());
                }
                Some(json!({
                    "schema_version": KERNEL_STAGE_SCHEMA,
                    "case_id": case_id,
                    "step": step,
                    "context_length": context_length,
                    "context_token_ids_sha256": context_token_ids_sha256,
                    "layer_index": KERNEL_STAGE_LAYER_INDEX,
                    "stages": stage_collector.finish_record()?,
                    "greedy_token_id": predicted_token_id,
                }))
            }
            None => {
                if self.linear_stage_context.is_some() || self.linear_stage_writer.is_some() {
                    return Err("kernel-stage trace row has incomplete collector state".to_string());
                }
                None
            }
        };
        Ok(FinishedTraceRow {
            payload: row,
            kernel_stage,
        })
    }
}

impl Qwen35Aq4IntermediateTraceObserver for RowCollector<'_> {
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

    fn wants_linear_attention_stages(&self, layer_index: usize) -> bool {
        self.linear_stage_collector.is_some() && layer_index == KERNEL_STAGE_LAYER_INDEX
    }

    fn begin_linear_attention_stage(
        &mut self,
        layer_index: usize,
        stage: PackageLinearAttnIntermediateTraceStage,
        elements: usize,
    ) -> Result<(), String> {
        self.linear_stage_collector
            .as_mut()
            .ok_or_else(|| "kernel-stage trace was not enabled for this row".to_string())?
            .begin(layer_index, stage, elements)?;
        let context = self
            .linear_stage_context
            .as_ref()
            .ok_or_else(|| "kernel-stage trace has no row identity context".to_string())?;
        self.linear_stage_writer
            .as_deref_mut()
            .ok_or_else(|| "kernel-stage trace has no frame writer".to_string())?
            .begin(context, stage, elements)
    }

    fn observe_linear_attention_stage_chunk(
        &mut self,
        layer_index: usize,
        stage: PackageLinearAttnIntermediateTraceStage,
        start: usize,
        values: &[f32],
    ) -> Result<(), String> {
        self.linear_stage_collector
            .as_mut()
            .ok_or_else(|| "kernel-stage trace was not enabled for this row".to_string())?
            .observe_chunk(layer_index, stage, start, values)?;
        self.linear_stage_writer
            .as_deref_mut()
            .ok_or_else(|| "kernel-stage trace has no frame writer".to_string())?
            .observe_chunk(stage, start, values)
    }

    fn finish_linear_attention_stage(
        &mut self,
        layer_index: usize,
        stage: PackageLinearAttnIntermediateTraceStage,
    ) -> Result<(), String> {
        self.linear_stage_collector
            .as_mut()
            .ok_or_else(|| "kernel-stage trace was not enabled for this row".to_string())?
            .finish(layer_index, stage)?;
        self.linear_stage_writer
            .as_deref_mut()
            .ok_or_else(|| "kernel-stage trace has no frame writer".to_string())?
            .finish(stage)
    }
}

impl Qwen35Aq4CalibrationObserver for RowCollector<'_> {
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

fn read_bounded_file(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("failed to stat {label}: {error}"))?;
    if !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular file"));
    }
    if metadata.len() > MAX_INPUT_BYTES {
        return Err(format!(
            "{label} exceeds the {MAX_INPUT_BYTES}-byte input bound"
        ));
    }
    let file = File::open(path).map_err(|error| format!("failed to read {label}: {error}"))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("failed to stat opened {label}: {error}"))?;
    if file_identity(&metadata) != file_identity(&opened) {
        return Err(format!("{label} changed while opening"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read {label}: {error}"))?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err(format!(
            "{label} exceeds the {MAX_INPUT_BYTES}-byte input bound"
        ));
    }
    let after =
        fs::symlink_metadata(path).map_err(|error| format!("failed to restat {label}: {error}"))?;
    if file_identity(&metadata) != file_identity(&after) {
        return Err(format!("{label} changed while reading"));
    }
    Ok(bytes)
}

fn file_identity(metadata: &std::fs::Metadata) -> (u64, u64, u64, u32, i64, i64, u64) {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.size(),
        metadata.mode(),
        metadata.mtime_nsec(),
        metadata.ctime_nsec(),
        metadata.nlink(),
    )
}

fn load_json_with_sha<T: for<'de> Deserialize<'de>>(
    path: &Path,
    label: &str,
) -> Result<(T, String), String> {
    let bytes = read_bounded_file(path, label)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {label}: {error}"))?;
    Ok((value, digest))
}

fn validate_inputs(
    cases: &CasesFile,
    replay: &ReplayFile,
) -> Result<(BTreeMap<String, Vec<usize>>, usize), String> {
    if cases.cases.is_empty() || cases.cases.len() > MAX_CASES {
        return Err(format!("cases must contain 1..={MAX_CASES} entries"));
    }
    let mut case_by_id = BTreeMap::new();
    for case in &cases.cases {
        if case_by_id.insert(case.case_id.clone(), case).is_some() {
            return Err(format!("duplicate case_id {}", case.case_id));
        }
    }
    if case_by_id.len() != EXPECTED_CASES.len() {
        return Err(format!(
            "cases must contain exactly the {} hash-bound fixture IDs",
            EXPECTED_CASES.len()
        ));
    }
    let mut rows = 0usize;
    for (case_id, prompt_len, prompt_hash, step_count, _) in EXPECTED_CASES {
        let case = case_by_id
            .get(case_id)
            .ok_or_else(|| format!("cases is missing expected case_id {case_id}"))?;
        if case.prompt_token_ids.len() != prompt_len
            || canonical_token_hash(&case.prompt_token_ids)? != prompt_hash
        {
            return Err(format!("prompt hash/count differs for {case_id}"));
        }
        if case.step_count != step_count {
            return Err(format!(
                "step_count differs for {case_id}: got {} expected {step_count}",
                case.step_count
            ));
        }
        rows = rows
            .checked_add(case.step_count)
            .ok_or_else(|| "total trace row count overflows".to_string())?;
    }
    if rows != MAX_ROWS {
        return Err(format!(
            "hash-bound fixture must emit exactly {MAX_ROWS} rows, got {rows}"
        ));
    }

    let mut replay_by_id = BTreeMap::new();
    for replay_case in &replay.cases {
        if replay_by_id
            .insert(replay_case.case_id.clone(), replay_case.token_ids.clone())
            .is_some()
        {
            return Err(format!("duplicate replay case_id {}", replay_case.case_id));
        }
    }
    if replay_by_id.len() != EXPECTED_CASES.len() {
        return Err(format!(
            "replay must contain exactly the {} hash-bound fixture IDs",
            EXPECTED_CASES.len()
        ));
    }
    for (case_id, _, _, _, expected_tokens) in EXPECTED_CASES {
        let actual = replay_by_id
            .get(case_id)
            .ok_or_else(|| format!("replay is missing expected case_id {case_id}"))?;
        if actual.as_slice() != expected_tokens {
            return Err(format!("replay token coverage differs for {case_id}"));
        }
    }
    Ok((replay_by_id, rows))
}

fn expected_case_bindings() -> Result<Vec<Value>, String> {
    EXPECTED_CASES
        .iter()
        .map(
            |(case_id, prompt_len, prompt_hash, step_count, replay_tokens)| {
                Ok(json!({
                    "case_id": case_id,
                    "prompt_token_count": prompt_len,
                    "prompt_token_ids_sha256": prompt_hash,
                    "step_count": step_count,
                    "replay_token_ids_sha256": canonical_token_hash(replay_tokens)?,
                    "replay_source_sequence_sha256":
                        Qwen35Aq4CalibrationReplay::source_sequence_sha256_for_tokens(replay_tokens)?,
                }))
            },
        )
        .collect()
}

fn total_output_bytes(root: &Path) -> Result<u64, String> {
    let mut total = 0_u64;
    for entry in fs::read_dir(root).map_err(|error| format!("failed to list output: {error}"))? {
        let entry = entry.map_err(|error| format!("failed to inspect output: {error}"))?;
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|error| format!("failed to stat output entry: {error}"))?;
        if !metadata.file_type().is_file() {
            return Err("trace output contains a non-regular entry".to_string());
        }
        total = total
            .checked_add(metadata.len())
            .ok_or_else(|| "trace output byte count overflows".to_string())?;
    }
    Ok(total)
}

fn required_build_git_commit() -> Result<String, String> {
    let embedded = EMBEDDED_BUILD_GIT_COMMIT
        .ok_or_else(|| "ULLM_BUILD_GIT_COMMIT is missing from the binary build".to_string())?;
    let embedded = validate_build_git_commit(embedded)?;
    if let Ok(runtime) = env::var("ULLM_BUILD_GIT_COMMIT") {
        let runtime = validate_build_git_commit(&runtime)?;
        if runtime != embedded {
            return Err(
                "runtime ULLM_BUILD_GIT_COMMIT differs from embedded build commit".to_string(),
            );
        }
    }
    Ok(embedded)
}

fn validate_build_git_commit(value: &str) -> Result<String, String> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("ULLM_BUILD_GIT_COMMIT must be a 40-character hexadecimal commit".to_string());
    }
    Ok(value.to_string())
}

fn trace_env_flag_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn linear_stage_trace_guard_set(device_index: u32) -> Result<Value, String> {
    if device_index != 1 {
        return Err(format!(
            "linear-stage trace requires global runtime device index 1, got {device_index}"
        ));
    }
    let mut visibility = BTreeMap::new();
    for name in ["HIP_VISIBLE_DEVICES", "ULLM_HIP_VISIBLE_DEVICES"] {
        let value = env::var(name).map_err(|_| format!("linear-stage trace requires {name}=1"))?;
        if value != "1" {
            return Err(format!("linear-stage trace requires {name}=1, got {value}"));
        }
        visibility.insert(name.to_string(), value);
    }
    let mut required = BTreeMap::new();
    for name in REQUIRED_LINEAR_STAGE_ENV {
        let value = env::var(name).map_err(|_| format!("linear-stage trace requires {name}=1"))?;
        if value != "1" {
            return Err(format!("linear-stage trace requires {name}=1, got {value}"));
        }
        required.insert(name.to_string(), value);
    }
    let mut disabled = BTreeMap::new();
    for name in DISALLOWED_LINEAR_STAGE_ENV {
        let value = env::var(name).ok();
        if trace_env_flag_enabled(name) {
            return Err(format!("linear-stage trace rejects enabled {name}"));
        }
        disabled.insert(name.to_string(), value);
    }
    Ok(json!({
        "target_backend": "hip",
        "expected_architecture": "gfx1201",
        "global_runtime_device_index": device_index,
        "visibility": visibility,
        "required_environment": required,
        "disabled_environment": disabled,
        "qkv_z_gate_beta_fusion_required": true,
        "component_timing_sync_allowed": false,
    }))
}

fn required_regular_sha256(path: &Path, label: &str) -> Result<String, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("{label} is unavailable: {error}"))?;
    if !metadata.file_type().is_file() || metadata.nlink() != 1 {
        return Err(format!("{label} must be a regular file"));
    }
    if metadata.len() > MAX_IDENTITY_FILE_BYTES {
        return Err(format!(
            "{label} exceeds the {MAX_IDENTITY_FILE_BYTES}-byte identity bound"
        ));
    }
    let mut file = File::open(path).map_err(|error| format!("{label} open failed: {error}"))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("{label} opened metadata failed: {error}"))?;
    if file_identity(&metadata) != file_identity(&opened)
        || !opened.file_type().is_file()
        || opened.nlink() != 1
    {
        return Err(format!("{label} changed while opening"));
    }
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    let mut total = 0_u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("{label} read failed: {error}"))?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| format!("{label} byte count overflows"))?;
        if total > MAX_IDENTITY_FILE_BYTES {
            return Err(format!(
                "{label} exceeds the {MAX_IDENTITY_FILE_BYTES}-byte identity bound"
            ));
        }
        digest.update(&buffer[..count]);
    }
    let after = file
        .metadata()
        .map_err(|error| format!("{label} final metadata failed: {error}"))?;
    if file_identity(&opened) != file_identity(&after) {
        return Err(format!("{label} changed while reading"));
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn run(
    package_dir: PathBuf,
    cases_path: PathBuf,
    replay_path: PathBuf,
    output: PathBuf,
    device_index: u32,
    enabled: bool,
    linear_stage_trace: bool,
) -> Result<(), String> {
    if !enabled {
        return Err(
            "intermediate trace is disabled; pass --enable-intermediate-trace explicitly"
                .to_string(),
        );
    }
    if linear_stage_trace && !enabled {
        return Err(
            "linear-stage trace requires --enable-intermediate-trace explicitly".to_string(),
        );
    }
    let linear_stage_guard = if linear_stage_trace {
        Some(linear_stage_trace_guard_set(device_index)?)
    } else {
        None
    };
    if output.exists() || fs::symlink_metadata(&output).is_ok() {
        return Err(format!("refusing to overwrite output {}", output.display()));
    }
    let (cases, cases_sha256): (CasesFile, String) = load_json_with_sha(&cases_path, "cases")?;
    let (replay, replay_sha256): (ReplayFile, String) = load_json_with_sha(&replay_path, "replay")?;
    let (replay_by_id, total_rows) = validate_inputs(&cases, &replay)?;
    let build_git_commit = required_build_git_commit()?;
    let expected_bindings = expected_case_bindings()?;
    let actual_case_bindings = cases
        .cases
        .iter()
        .map(|case| {
            Ok(json!({
                "case_id": case.case_id,
                "prompt_token_count": case.prompt_token_ids.len(),
                "prompt_token_ids": case.prompt_token_ids,
                "prompt_token_ids_sha256": canonical_token_hash(&case.prompt_token_ids)?,
                "step_count": case.step_count,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let actual_replay_bindings = replay_by_id
        .iter()
        .map(|(case_id, token_ids)| {
            Ok(json!({
                "case_id": case_id,
                "token_ids_sha256": canonical_token_hash(token_ids)?,
                "source_sequence_sha256":
                    Qwen35Aq4CalibrationReplay::source_sequence_sha256_for_tokens(token_ids)?,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let tool_binary =
        env::current_exe().map_err(|error| format!("failed to resolve trace binary: {error}"))?;
    let tool_binary_sha256 = required_regular_sha256(&tool_binary, "trace binary")?;
    let active_manifest_path = env::var("ULLM_SERVED_MODEL_MANIFEST")
        .unwrap_or_else(|_| "/etc/ullm/served-models/active.json".to_string());
    let active_manifest = PathBuf::from(&active_manifest_path);
    let active_manifest_sha256 =
        required_regular_sha256(&active_manifest, "active served-model manifest")?;
    let package_manifest = package_dir.join("manifest.json");
    let package_manifest_sha256 = required_regular_sha256(&package_manifest, "package manifest")?;
    let guard_set = json!({
        "explicit_flag": "--enable-intermediate-trace",
        "linear_stage_trace": linear_stage_trace,
        "linear_stage_guard": linear_stage_guard,
        "max_cases": MAX_CASES,
        "max_rows": MAX_ROWS,
        "max_row_bytes": MAX_ROW_BYTES,
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "max_kernel_stage_stream_bytes": MAX_KERNEL_STAGE_STREAM_BYTES,
        "scratch_bytes": ullm_engine::qwen35_aq4_model_runtime::QWEN35_AQ4_INTERMEDIATE_TRACE_SCRATCH_BYTES,
        "linear_stage_readback_chunk_bytes": QWEN35_AQ4_LINEAR_STAGE_TRACE_CHUNK_BYTES,
    });
    let guard_set_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&guard_set).map_err(|error| error.to_string())?)
    );
    if guard_set_sha256.len() != 64 {
        return Err("guard-set identity is incomplete".to_string());
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
        expected_architecture: linear_stage_trace.then(|| "gfx1201".to_string()),
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
    let device_name = session.model().device_name().to_string();
    let backend = session.model().backend().to_string();
    let total_global_mem = session.model().device_total_global_mem();
    if device_name.is_empty() || backend.is_empty() || total_global_mem == 0 {
        return Err("runtime device identity is incomplete".to_string());
    }
    if linear_stage_trace && !backend.eq_ignore_ascii_case("hip") {
        return Err(format!(
            "linear-stage trace requires HIP backend, got {backend}"
        ));
    }
    let device_identity = json!({
        "index": device_index,
        "name": device_name,
        "backend": backend,
        "total_global_mem": total_global_mem,
    });
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
        let mut payload_bytes = 0_u64;
        let mut kernel_stage_payload = if linear_stage_trace {
            Some(BufWriter::new(
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(temporary.join(KERNEL_STAGE_PAYLOAD_FILE))
                    .map_err(|error| format!("failed to create kernel-stage payload: {error}"))?,
            ))
        } else {
            None
        };
        let mut kernel_stage_payload_bytes = 0_u64;
        let mut kernel_stage_stream = if linear_stage_trace {
            Some(KernelStageFrameWriter::create(
                &temporary.join(KERNEL_STAGE_STREAM_FILE),
            )?)
        } else {
            None
        };
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
                        let mut context_tokens = case.prompt_token_ids.clone();
                        context_tokens.extend_from_slice(&replay_tokens[..step]);
                        let context_token_ids_sha256 = canonical_token_hash(&context_tokens)?;
                        let mut collector = if linear_stage_trace {
                            RowCollector::with_linear_stage_trace(
                                LinearStageTraceContext {
                                    case_id: case.case_id.clone(),
                                    step,
                                    context_length: context_tokens.len(),
                                    context_token_ids_sha256: context_token_ids_sha256.clone(),
                                },
                                kernel_stage_stream.as_mut().ok_or_else(|| {
                                    "kernel-stage trace has no stream writer".to_string()
                                })?,
                            )
                        } else {
                            RowCollector::new()
                        };
                        session
                            .model_mut()
                            .visit_intermediate_trace(&mut collector)?;
                        session.observe_prepared_calibration(&prepared, &mut collector)?;
                        let record = collector.finish_record(
                            &case.case_id,
                            step,
                            context_tokens.len(),
                            context_token_ids_sha256,
                            token_id,
                        )?;
                        let encoded = serde_json::to_vec(&record.payload)
                            .map_err(|error| error.to_string())?;
                        let next_payload_bytes = payload_bytes
                            .checked_add(encoded.len() as u64 + 1)
                            .ok_or_else(|| "trace payload byte count overflows".to_string())?;
                        if next_payload_bytes > MAX_OUTPUT_BYTES {
                            return Err(format!(
                                "trace output exceeds {MAX_OUTPUT_BYTES} bytes before publication"
                            ));
                        }
                        payload
                            .write_all(&encoded)
                            .map_err(|error| error.to_string())?;
                        payload
                            .write_all(b"\n")
                            .map_err(|error| error.to_string())?;
                        payload_bytes = next_payload_bytes;
                        match (linear_stage_trace, record.kernel_stage) {
                            (true, Some(kernel_stage_record)) => {
                                let encoded = serde_json::to_vec(&kernel_stage_record)
                                    .map_err(|error| error.to_string())?;
                                let next_bytes = kernel_stage_payload_bytes
                                    .checked_add(encoded.len() as u64 + 1)
                                    .ok_or_else(|| {
                                        "kernel-stage payload byte count overflows".to_string()
                                    })?;
                                if next_bytes > MAX_OUTPUT_BYTES {
                                    return Err(format!(
                                        "kernel-stage payload exceeds {MAX_OUTPUT_BYTES} bytes before publication"
                                    ));
                                }
                                let writer = kernel_stage_payload.as_mut().ok_or_else(|| {
                                    "kernel-stage trace has no summary writer".to_string()
                                })?;
                                writer
                                    .write_all(&encoded)
                                    .and_then(|_| writer.write_all(b"\n"))
                                    .map_err(|error| error.to_string())?;
                                kernel_stage_payload_bytes = next_bytes;
                            }
                            (true, None) => {
                                return Err(
                                    "kernel-stage trace omitted its row summary".to_string()
                                );
                            }
                            (false, Some(_)) => {
                                return Err(
                                    "disabled kernel-stage trace emitted a row summary".to_string()
                                );
                            }
                            (false, None) => {}
                        }
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
        if let Some(payload) = kernel_stage_payload.as_mut() {
            payload.flush().map_err(|error| error.to_string())?;
        }
        let kernel_stage_stream_bytes = match kernel_stage_stream.take() {
            Some(writer) => Some(writer.finish_stream()?),
            None => None,
        };
        if linear_stage_trace && kernel_stage_stream_bytes.is_none() {
            return Err("kernel-stage trace did not produce its frame stream".to_string());
        }
        let kernel_stage_contract = if linear_stage_trace {
            json!({
                "enabled": true,
                "schema_version": KERNEL_STAGE_SCHEMA,
                "layer_index": KERNEL_STAGE_LAYER_INDEX,
                "stage_order": PackageLinearAttnIntermediateTraceStage::ORDERED
                    .iter()
                    .map(|stage| stage.label())
                    .collect::<Vec<_>>(),
                "summary_file": KERNEL_STAGE_PAYLOAD_FILE,
                "f32le_stream_file": KERNEL_STAGE_STREAM_FILE,
                "f32le_stream_bytes": kernel_stage_stream_bytes,
                "stream_contract": "one framed f32le tensor per stage; timestep is context_length - 1 and matches the CPU hybrid stage emitter",
                "readback_chunk_bytes": QWEN35_AQ4_LINEAR_STAGE_TRACE_CHUNK_BYTES,
            })
        } else {
            json!({"enabled": false})
        };
        let manifest = json!({
            "schema_version": SCHEMA,
            "mode": "aq4_gpu_intermediate_diagnostic",
            "package_dir": package_dir,
            "cases_path": cases_path,
            "replay_path": replay_path,
            "device_index": device_index,
            "rows": total_rows,
            "input_binding": {
                "cases_sha256": cases_sha256,
                "replay_sha256": replay_sha256,
                "expected_cases": expected_bindings,
                "actual_cases": actual_case_bindings,
                "actual_replay_sequences": actual_replay_bindings,
            },
            "identity": {
                "tool_binary": tool_binary,
                "tool_binary_sha256": tool_binary_sha256,
                "build_git_commit": build_git_commit,
                "active_manifest_path": active_manifest_path,
                "active_manifest_sha256": active_manifest_sha256,
                "package_manifest_sha256": package_manifest_sha256,
                "guard_set_sha256": guard_set_sha256,
                "device": device_identity,
            },
            "guard_set": guard_set,
            "stage_contract": {"embedding": true, "decoder_layers": 32, "final_norm": true, "lm_head": true, "hidden_coordinates": HIDDEN_COORDINATES, "logit_coordinates": LOGIT_COORDINATES, "kernel_stage_trace": kernel_stage_contract},
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
        let mut sum_names = vec!["manifest.json", "payload.jsonl", "runtime.json"];
        if linear_stage_trace {
            sum_names.push(KERNEL_STAGE_PAYLOAD_FILE);
            sum_names.push(KERNEL_STAGE_STREAM_FILE);
        }
        let sums = sum_names
            .iter()
            .map(|name| Ok(format!("{}  {name}\n", sha256_file(&temporary.join(name))?)))
            .collect::<Result<String, String>>()?;
        fs::write(temporary.join("SHA256SUMS"), sums).map_err(|error| error.to_string())?;
        let total_bytes = total_output_bytes(&temporary)?;
        let max_total_bytes = if linear_stage_trace {
            MAX_OUTPUT_BYTES
                .checked_add(MAX_KERNEL_STAGE_STREAM_BYTES)
                .ok_or_else(|| "trace output bound overflows".to_string())?
        } else {
            MAX_OUTPUT_BYTES
        };
        if total_bytes > max_total_bytes {
            return Err(format!(
                "trace output total {total_bytes} exceeds {max_total_bytes} bytes"
            ));
        }
        fs::rename(&temporary, &output)
            .map_err(|error| format!("failed to publish trace root: {error}"))
    })();
    match result {
        Ok(()) => Ok(()),
        Err(error) => Err(format!(
            "{error}; incomplete diagnostic evidence was retained at {} (do not compare or promote it)",
            temporary.display()
        )),
    }
}

const USAGE: &str = "usage: ullm-aq4-differential-trace PACKAGE_DIR CASES_JSON REPLAY_JSON OUTPUT_DIR [DEVICE_INDEX] --enable-intermediate-trace [--enable-linear-stage-trace]";

fn parse_cli(args: &[String]) -> Result<(u32, bool, bool), String> {
    if args.len() < 6 || args.len() > 8 {
        return Err(USAGE.to_string());
    }
    let mut device_index = 1_u32;
    let mut saw_device_index = false;
    let mut enabled = false;
    let mut linear_stage_trace = false;
    for argument in &args[5..] {
        match argument.as_str() {
            "--enable-intermediate-trace" => enabled = true,
            "--enable-linear-stage-trace" => linear_stage_trace = true,
            value if !value.starts_with('-') && !saw_device_index => {
                device_index = value
                    .parse::<u32>()
                    .map_err(|_| format!("invalid DEVICE_INDEX {value}; {USAGE}"))?;
                saw_device_index = true;
            }
            _ => return Err(USAGE.to_string()),
        }
    }
    if linear_stage_trace && !enabled {
        return Err("--enable-linear-stage-trace requires --enable-intermediate-trace".to_string());
    }
    Ok((device_index, enabled, linear_stage_trace))
}

fn main() -> ExitCode {
    let args = env::args().collect::<Vec<_>>();
    let (device_index, enabled, linear_stage_trace) = match parse_cli(&args) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let result = run(
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
        PathBuf::from(&args[4]),
        device_index,
        enabled,
        linear_stage_trace,
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

    fn valid_inputs() -> (CasesFile, ReplayFile) {
        (
            CasesFile {
                cases: vec![
                    Case {
                        case_id: "fixture-prompt-0".to_string(),
                        prompt_token_ids: vec![11, 12, 13],
                        step_count: 2,
                    },
                    Case {
                        case_id: "fixture-prompt-1".to_string(),
                        prompt_token_ids: vec![21, 22],
                        step_count: 1,
                    },
                ],
            },
            ReplayFile {
                cases: vec![
                    ReplayCase {
                        case_id: "fixture-prompt-0".to_string(),
                        token_ids: vec![220, 16],
                    },
                    ReplayCase {
                        case_id: "fixture-prompt-1".to_string(),
                        token_ids: vec![15],
                    },
                ],
            },
        )
    }

    #[test]
    fn intermediate_trace_is_explicitly_opt_in() {
        let result = run(
            PathBuf::from("missing-package"),
            PathBuf::from("missing-cases"),
            PathBuf::from("missing-replay"),
            PathBuf::from("missing-output"),
            0,
            false,
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
            .finish_record("case", 0, 3, canonical_token_hash(&[1, 2, 3]).unwrap(), 42)
            .expect("complete row should fit contract");
        let encoded = serde_json::to_vec(&row.payload).expect("row should encode");
        assert!(encoded.len() <= MAX_ROW_BYTES);
        assert_eq!(row.payload["stages"].as_array().unwrap().len(), 35);
        assert_eq!(row.payload["stages"][0]["stage"], "embedding");
        assert_eq!(row.payload["context_length"], 3);
        assert_eq!(row.payload["stages"][33]["stage"], "final_norm");
        assert_eq!(row.payload["stages"][34]["stage"], "lm_head");
        assert!(row.kernel_stage.is_none());
    }

    #[test]
    fn linear_stage_sidecar_preserves_v1_payload_and_cpu_frame_contract() {
        let stream_path = env::temp_dir().join(format!(
            "ullm-aq4-differential-trace-kernel-stages-{}",
            std::process::id()
        ));
        let mut writer = KernelStageFrameWriter::create(&stream_path)
            .expect("kernel-stage stream should create");
        let context_hash = canonical_token_hash(&[1, 2, 3]).unwrap();
        let row = {
            let mut collector = RowCollector::with_linear_stage_trace(
                LinearStageTraceContext {
                    case_id: "case".to_string(),
                    step: 0,
                    context_length: 3,
                    context_token_ids_sha256: context_hash.clone(),
                },
                &mut writer,
            );
            let hidden = vec![0.25_f32; 4096];
            let logits = vec![0.5_f32; 64];
            collector.observe_embedding(&hidden).unwrap();
            for layer in 0..32 {
                collector.observe_decoder_layer(layer, &hidden).unwrap();
            }
            for stage in PackageLinearAttnIntermediateTraceStage::ORDERED {
                let elements = match stage {
                    PackageLinearAttnIntermediateTraceStage::RecurrentGate
                    | PackageLinearAttnIntermediateTraceStage::RecurrentBeta => 32,
                    _ => 4096,
                };
                let values = vec![0.125_f32; elements];
                collector
                    .begin_linear_attention_stage(KERNEL_STAGE_LAYER_INDEX, stage, elements)
                    .unwrap();
                collector
                    .observe_linear_attention_stage_chunk(
                        KERNEL_STAGE_LAYER_INDEX,
                        stage,
                        0,
                        &values,
                    )
                    .unwrap();
                collector
                    .finish_linear_attention_stage(KERNEL_STAGE_LAYER_INDEX, stage)
                    .unwrap();
            }
            collector.begin(hidden.len(), logits.len()).unwrap();
            collector.observe_hidden_chunk(0, &hidden).unwrap();
            collector.observe_logit_chunk(0, &logits).unwrap();
            collector.finish().unwrap();
            collector
                .finish_record("case", 0, 3, context_hash, 42)
                .expect("complete kernel-stage row should fit contract")
        };
        assert_eq!(row.payload["stages"].as_array().unwrap().len(), 35);
        let kernel_stage = row.kernel_stage.expect("kernel-stage summary must exist");
        assert_eq!(kernel_stage["schema_version"], KERNEL_STAGE_SCHEMA);
        assert_eq!(
            kernel_stage["stages"].as_array().unwrap().len(),
            PackageLinearAttnIntermediateTraceStage::ORDERED.len()
        );
        assert_eq!(
            kernel_stage["stages"][0]["stage"],
            PackageLinearAttnIntermediateTraceStage::QkvDequantRowScale.label()
        );
        assert_eq!(kernel_stage["stages"][2]["sample"]["elements"], 32);
        let stream_bytes = writer.finish_stream().unwrap();
        assert!(stream_bytes > 0 && stream_bytes <= MAX_KERNEL_STAGE_STREAM_BYTES);
        let raw = fs::read(&stream_path).expect("kernel-stage stream should read");
        let header_end = raw
            .iter()
            .position(|byte| *byte == b'\n')
            .expect("kernel-stage frame must have a JSON header");
        let header: Value = serde_json::from_slice(&raw[..header_end]).unwrap();
        assert_eq!(header["kind"], "stage");
        assert_eq!(header["timestep"], 2);
        assert_eq!(
            header["stage"],
            PackageLinearAttnIntermediateTraceStage::QkvDequantRowScale.label()
        );
        assert!(raw.ends_with(b"{\"kind\":\"end\"}\n"));
        fs::remove_file(&stream_path).expect("kernel-stage stream should remove");
    }

    #[test]
    fn cli_accepts_only_the_explicit_linear_stage_opt_in() {
        let args = vec![
            "ullm-aq4-differential-trace".to_string(),
            "package".to_string(),
            "cases".to_string(),
            "replay".to_string(),
            "output".to_string(),
            "1".to_string(),
            "--enable-intermediate-trace".to_string(),
            "--enable-linear-stage-trace".to_string(),
        ];
        assert_eq!(parse_cli(&args).unwrap(), (1, true, true));
        let without_base_flag = vec![
            "ullm-aq4-differential-trace".to_string(),
            "package".to_string(),
            "cases".to_string(),
            "replay".to_string(),
            "output".to_string(),
            "--enable-linear-stage-trace".to_string(),
        ];
        assert!(parse_cli(&without_base_flag).is_err());
        let unknown_flag = vec![
            "ullm-aq4-differential-trace".to_string(),
            "package".to_string(),
            "cases".to_string(),
            "replay".to_string(),
            "output".to_string(),
            "--enable-intermediate-trace".to_string(),
            "--unexpected".to_string(),
        ];
        assert!(parse_cli(&unknown_flag).is_err());
    }

    #[test]
    fn three_row_fixture_and_full_context_hashes_are_bound() {
        let (cases, replay) = valid_inputs();
        let (replay_by_id, rows) = validate_inputs(&cases, &replay).expect("fixture is valid");
        assert_eq!(rows, MAX_ROWS);
        assert_eq!(
            canonical_token_hash(&cases.cases[0].prompt_token_ids).unwrap(),
            "42ea52c728680a54afafd1c1e1e45f13300c3ceb962f320f3900196a0c46215c"
        );
        let mut context = cases.cases[0].prompt_token_ids.clone();
        context.extend_from_slice(&replay_by_id["fixture-prompt-0"][..1]);
        assert_eq!(context.len(), 4);
        assert_eq!(
            canonical_token_hash(&context).unwrap(),
            "6af1601b9bf35d095b24c5bac3a95a01bf77d047b576441d0a5f9510eec66249"
        );
        assert_eq!(
            canonical_token_hash(&cases.cases[1].prompt_token_ids).unwrap(),
            "3bca9e21e3b6f741ed412f91d7696146c254ff68bd9be9ca41b1d172eb3549e6"
        );
        assert_eq!(expected_case_bindings().unwrap().len(), 2);
    }

    #[test]
    fn input_guard_rejects_overlimit_duplicate_missing_and_extra_cases() {
        let (mut cases, replay) = valid_inputs();
        cases.cases.push(Case {
            case_id: "overflow".to_string(),
            prompt_token_ids: vec![1],
            step_count: 1,
        });
        cases.cases.push(Case {
            case_id: "overflow-2".to_string(),
            prompt_token_ids: vec![2],
            step_count: 1,
        });
        assert!(validate_inputs(&cases, &replay).is_err());

        let (mut duplicate_cases, replay) = valid_inputs();
        duplicate_cases.cases[1].case_id = duplicate_cases.cases[0].case_id.clone();
        assert!(
            validate_inputs(&duplicate_cases, &replay)
                .expect_err("duplicate IDs must reject")
                .contains("duplicate case_id")
        );

        let (mut missing_cases, replay) = valid_inputs();
        missing_cases.cases.pop();
        assert!(
            validate_inputs(&missing_cases, &replay)
                .expect_err("missing ID must reject")
                .contains("exactly")
        );

        let (mut extra_cases, replay) = valid_inputs();
        extra_cases.cases[1].case_id = "unexpected".to_string();
        assert!(
            validate_inputs(&extra_cases, &replay)
                .expect_err("extra ID must reject")
                .contains("missing expected")
        );
    }

    #[test]
    fn input_guard_rejects_duplicate_replay_and_oversized_file() {
        let (cases, mut replay) = valid_inputs();
        replay.cases.push(ReplayCase {
            case_id: "fixture-prompt-0".to_string(),
            token_ids: vec![220, 16],
        });
        assert!(
            validate_inputs(&cases, &replay)
                .expect_err("duplicate replay IDs must reject")
                .contains("duplicate replay")
        );

        let path = env::temp_dir().join(format!(
            "ullm-aq4-differential-trace-oversized-{}",
            std::process::id()
        ));
        fs::write(&path, vec![b'x'; (MAX_INPUT_BYTES + 1) as usize])
            .expect("oversized test input should write");
        let result = read_bounded_file(&path, "oversized test");
        let run_result = run(
            PathBuf::from("missing-package"),
            path.clone(),
            PathBuf::from("missing-replay"),
            PathBuf::from(format!("missing-output-{}", std::process::id())),
            0,
            true,
            false,
        );
        fs::remove_file(&path).expect("oversized test input should remove");
        assert!(
            result
                .expect_err("oversized input must reject")
                .contains("input bound")
        );
        assert!(
            run_result
                .expect_err("run must reject oversized input before model load")
                .contains("input bound")
        );
    }

    #[test]
    fn scratch_and_output_limits_are_explicit() {
        assert_eq!(
            ullm_engine::qwen35_aq4_model_runtime::QWEN35_AQ4_INTERMEDIATE_TRACE_SCRATCH_BYTES,
            32 * 1024
        );
        assert!(
            4096 * std::mem::size_of::<f32>() * 2
                <= ullm_engine::qwen35_aq4_model_runtime::QWEN35_AQ4_INTERMEDIATE_TRACE_SCRATCH_BYTES
        );
        assert_eq!(MAX_ROWS, 3);
        assert_eq!(MAX_ROW_BYTES, 32 * 1024);
        assert_eq!(MAX_OUTPUT_BYTES, 96 * 1024);
        assert_eq!(QWEN35_AQ4_LINEAR_STAGE_TRACE_CHUNK_BYTES, 256 * 1024);
        assert_eq!(MAX_KERNEL_STAGE_STREAM_BYTES, 16 * 1024 * 1024);
    }

    #[test]
    fn identity_guards_reject_unknown_or_missing_values() {
        assert!(validate_build_git_commit("").is_err());
        assert!(validate_build_git_commit("unknown").is_err());
        assert!(validate_build_git_commit(&"g".repeat(40)).is_err());
        assert_eq!(
            validate_build_git_commit(&"a".repeat(40)).unwrap(),
            "a".repeat(40)
        );
        assert!(
            required_regular_sha256(
                Path::new("missing-active-manifest"),
                "active served-model manifest"
            )
            .is_err()
        );
        let path = env::temp_dir().join(format!(
            "ullm-aq4-differential-trace-identity-{}",
            std::process::id()
        ));
        fs::write(&path, b"identity").expect("identity fixture should write");
        let digest = required_regular_sha256(&path, "identity fixture")
            .expect("identity fixture should hash");
        fs::remove_file(&path).expect("identity fixture should remove");
        assert_eq!(digest, format!("{:x}", Sha256::digest(b"identity")));
    }

    #[test]
    fn bounded_json_hashes_and_parses_the_same_bytes() {
        let path = env::temp_dir().join(format!(
            "ullm-aq4-differential-trace-json-{}",
            std::process::id()
        ));
        let raw = b"{\"cases\":[]}\n";
        fs::write(&path, raw).expect("test JSON should write");
        let (decoded, digest) =
            load_json_with_sha::<CasesFile>(&path, "test cases").expect("test JSON should parse");
        fs::write(
            &path,
            b"{\"cases\":[{\"case_id\":\"replacement\",\"prompt_token_ids\":[],\"step_count\":1}]}\n",
        )
        .expect("replacement JSON should write");
        fs::remove_file(&path).expect("test JSON should remove");
        assert!(decoded.cases.is_empty());
        assert_eq!(digest, format!("{:x}", Sha256::digest(raw)));
    }

    #[cfg(unix)]
    #[test]
    fn bounded_reader_rejects_symlink_replacement() {
        use std::os::unix::fs::symlink;
        let target = env::temp_dir().join(format!(
            "ullm-aq4-differential-trace-target-{}",
            std::process::id()
        ));
        let link = env::temp_dir().join(format!(
            "ullm-aq4-differential-trace-link-{}",
            std::process::id()
        ));
        fs::write(&target, b"{}").expect("target should write");
        symlink(&target, &link).expect("symlink should create");
        let result = read_bounded_file(&link, "symlink test");
        fs::remove_file(&link).expect("symlink should remove");
        fs::remove_file(&target).expect("target should remove");
        assert!(result.is_err());
    }
}
