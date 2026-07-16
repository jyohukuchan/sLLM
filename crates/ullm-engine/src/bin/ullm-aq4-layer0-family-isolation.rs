// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU-only layer-0 AQ4 family isolation probe.
//!
//! This diagnostic intentionally calls the production standalone AQ4 matvec
//! once per weight family (QKV, Z, A, and B).  It does not call a fused
//! production operator and it does not change production defaults.  The raw
//! f32 rows are written for an independent BF16 source comparison tool.
//!
//! In addition to the original raw-matvec mode, `--hybrid-input` runs the
//! complete layer-0 linear-attention + MLP block with the production AQ4
//! decoder.  That mode is deliberately CPU-only and emits framed, transient
//! stage tensors on stdout only when requested.  Persistent reports retain
//! fixed-coordinate samples and streaming summaries, never full hidden-state
//! or vocabulary tensors.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use ullm_engine::aq4_package_runtime::PackageAq4ResidentMatvec;
use ullm_engine::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use ullm_engine::loader::{
    effective_rmsnorm_weight_values, read_named_passthrough_f32, WeightRegistry,
};
use ullm_engine::qwen35_aq4_layer_runtime::{
    runtime_host_linear_attn_gate_beta_f32, runtime_host_linear_attn_recurrent_f32,
};

const SCHEMA: &str = "ullm.aq4_layer0_family_isolation.aq4_cpu.v1";
const INPUT_SCHEMA: &str = "ullm.aq4_layer0_input_normed_jsonl.v1";
const INPUT_COLS: usize = 4096;
const MAX_CASES: usize = 4096;
const MAX_LINE_BYTES: usize = 2 * 1024 * 1024;
const MAX_CHUNK_BYTES: usize = 256 * 1024 * 1024;

const HYBRID_SCHEMA: &str = "ullm.aq4_layer0_hybrid_diagnostic.aq4_cpu.v1";
const HYBRID_INPUT_SCHEMA: &str = "ullm.aq4_layer0_hybrid_input_jsonl.v1";
const HYBRID_MAX_CASES: usize = 128;
const HYBRID_MAX_CONTEXT_LENGTH: usize = 512;
const HIDDEN: usize = 4096;
const QKV_ROWS: usize = 8192;
const VALUE_HEADS: usize = 32;
const KEY_HEADS: usize = 16;
const KEY_DIM: usize = 128;
const VALUE_DIM: usize = 128;
const CONV_KERNEL: usize = 4;
const INTERMEDIATE: usize = 12288;
const STATE_ELEMENTS: usize = VALUE_HEADS * KEY_DIM * VALUE_DIM;
const INPUT_RMS_EPSILON: f32 = 1e-6_f32;
const ATTENTION_RMS_EPSILON: f32 = 1e-6_f32;
// This is deliberately the standalone AQ4 runtime's current layer-0 post
// norm epsilon.  The BF16 comparator records the source value separately;
// this diagnostic must not silently substitute it.
const AQ4_POST_RMS_EPSILON: f32 = 1e-5_f32;
const Q_SCALE: f32 = 1.0_f32 / 11.313_708_f32;
const DIAGNOSTIC_LOGIT_ROWS: [usize; 34] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 220, 41330,
];

const FAMILY_NAMES: [&str; 4] = ["qkv", "z", "a", "b"];
const TENSOR_NAMES: [&str; 4] = [
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.linear_attn.in_proj_z.weight",
    "model.language_model.layers.0.linear_attn.in_proj_a.weight",
    "model.language_model.layers.0.linear_attn.in_proj_b.weight",
];
const EXPECTED_ROWS: [usize; 4] = [8192, 4096, 32, 32];

const OUT_TENSOR: &str = "model.language_model.layers.0.linear_attn.out_proj.weight";
const MLP_GATE_TENSOR: &str = "model.language_model.layers.0.mlp.gate_proj.weight";
const MLP_UP_TENSOR: &str = "model.language_model.layers.0.mlp.up_proj.weight";
const MLP_DOWN_TENSOR: &str = "model.language_model.layers.0.mlp.down_proj.weight";
const INPUT_NORM_TENSOR: &str = "model.language_model.layers.0.input_layernorm.weight";
const CONV_TENSOR: &str = "model.language_model.layers.0.linear_attn.conv1d.weight";
const A_LOG_TENSOR: &str = "model.language_model.layers.0.linear_attn.A_log";
const DT_BIAS_TENSOR: &str = "model.language_model.layers.0.linear_attn.dt_bias";
const ATTN_NORM_TENSOR: &str = "model.language_model.layers.0.linear_attn.norm.weight";
const POST_NORM_TENSOR: &str = "model.language_model.layers.0.post_attention_layernorm.weight";
const LM_HEAD_TENSOR: &str = "lm_head.weight";

type Result<T> = std::result::Result<T, String>;

#[derive(Debug)]
struct Args {
    package: PathBuf,
    input: Option<PathBuf>,
    hybrid_input: Option<PathBuf>,
    output: PathBuf,
    chunk_bytes: usize,
    stage_stream_stdout: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    kind: String,
    schema_version: String,
    tensor_name: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    kind: String,
    case_id: String,
    step: usize,
    context_token_ids_sha256: String,
    context_length: usize,
    input_sha256: String,
    values: Vec<f32>,
}

#[derive(Debug, Serialize, Clone)]
struct CaseReport {
    case_id: String,
    step: usize,
    context_token_ids_sha256: String,
    context_length: usize,
    input_sha256: String,
    output_offset_bytes: u64,
    output_elements: usize,
    output_sha256: String,
    finite: bool,
}

#[derive(Debug, Serialize)]
struct OutputReport {
    path: String,
    shape: Vec<usize>,
    dtype: String,
    row_order: String,
    bytes: u64,
    sha256: String,
    cases: Vec<CaseReport>,
}

#[derive(Debug, Serialize)]
struct TensorReport {
    family: String,
    tensor_name: String,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
    dtype: String,
    manifest_entry_sha256: String,
    index_path: String,
    index_sha256: String,
    scale_path: String,
    scale_sha256: String,
    codebook_path: String,
    codebook_sha256: String,
}

#[derive(Debug, Serialize)]
struct InputReport {
    path: String,
    schema: String,
    dtype: String,
    shape: Vec<usize>,
    rows: usize,
    consumed_sha256: String,
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    schema_version: String,
    status: String,
    classification: String,
    promotion: bool,
    holdout: String,
    policy_evaluation: String,
    device: String,
    chunk_bytes: usize,
    package_root: String,
    package_manifest_sha256: String,
    input: InputReport,
    tensors: Vec<TensorReport>,
    outputs: BTreeMap<String, OutputReport>,
    family_order: Vec<String>,
    one_at_a_time_hybrid: HybridReport,
}

#[derive(Debug, Serialize)]
struct HybridReport {
    attempted: bool,
    status: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HybridInputHeader {
    kind: String,
    schema_version: String,
    tensor_name: String,
    dtype: String,
    shape: Vec<usize>,
    residual_encoding: String,
    source_model_index_sha256: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct HybridInputCase {
    kind: String,
    case_id: String,
    step: usize,
    context_token_ids: Vec<u32>,
    context_token_ids_sha256: String,
    context_length: usize,
    residual_path: String,
    residual_sha256: String,
    residual_shape: Vec<usize>,
    residual_dtype: String,
}

#[derive(Debug, Serialize)]
struct HybridInputReport {
    path: String,
    schema: String,
    source_embedding_tensor: String,
    dtype: String,
    shape: Vec<usize>,
    residual_encoding: String,
    source_model_index_sha256: String,
    rows: usize,
    consumed_sha256: String,
    cases: Vec<HybridCaseBinding>,
}

#[derive(Debug, Serialize, Clone)]
struct HybridCaseBinding {
    case_id: String,
    step: usize,
    context_token_ids_sha256: String,
    context_length: usize,
    residual_path: String,
    residual_sha256: String,
}

#[derive(Debug, Serialize)]
struct HybridStageSample {
    case_id: String,
    step: usize,
    context_token_ids_sha256: String,
    timestep: usize,
    elements: usize,
    coordinates: Vec<usize>,
    values: Vec<f32>,
    max_abs: f64,
    l2: f64,
}

#[derive(Debug, Serialize)]
struct HybridStageSummary {
    records: usize,
    elements_per_record: usize,
    aggregate_l2: f64,
    max_abs: f64,
    samples: Vec<HybridStageSample>,
}

#[derive(Debug, Serialize)]
struct HybridExecutionReport {
    attempted: bool,
    status: String,
    formula: String,
    aq4_projection_tensors: Vec<String>,
    input_rms_epsilon: f32,
    attention_rms_epsilon: f32,
    post_rms_epsilon: f32,
    q_scale: f32,
    rope: HybridRopeReport,
    logits: HybridLogitReport,
    state_contract: String,
    persistence_contract: String,
}

#[derive(Debug, Serialize)]
struct HybridRopeReport {
    applicable: bool,
    status: String,
    reason: String,
}

#[derive(Debug, Serialize)]
struct HybridLogitReport {
    status: String,
    stage: String,
    token_rows: Vec<usize>,
    reason: String,
}

#[derive(Debug, Serialize)]
struct HybridProbeReport {
    schema_version: String,
    status: String,
    classification: String,
    promotion: bool,
    holdout: String,
    policy_evaluation: String,
    device: String,
    chunk_bytes: usize,
    package_root: String,
    package_manifest_sha256: String,
    input: HybridInputReport,
    one_at_a_time_hybrid: HybridExecutionReport,
    stage_summaries: BTreeMap<String, HybridStageSummary>,
}

#[derive(Debug, Clone)]
struct FinalLayerOutput {
    case: HybridInputCase,
    values: Vec<f32>,
}

#[derive(Debug)]
struct OutputSink {
    path: PathBuf,
    file: File,
    digest: Sha256,
    bytes: u64,
    shape: usize,
    cases: Vec<CaseReport>,
}

impl OutputSink {
    fn create(dir: &Path, name: &str, shape: usize) -> Result<Self> {
        let path = dir.join(format!("{name}.f32le"));
        let file = File::create(&path)
            .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
        Ok(Self {
            path,
            file,
            digest: Sha256::new(),
            bytes: 0,
            shape,
            cases: Vec::new(),
        })
    }

    fn row(&mut self, case: &Case, values: &[f32]) -> Result<()> {
        if values.len() != self.shape || values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{} output row is invalid", self.path.display()));
        }
        let bytes = encode_f32_to_bytes(values);
        let offset = self.bytes;
        self.file
            .write_all(&bytes)
            .map_err(|err| format!("failed writing {}: {err}", self.path.display()))?;
        self.digest.update(&bytes);
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| "output size overflow".to_string())?;
        self.cases.push(CaseReport {
            case_id: case.case_id.clone(),
            step: case.step,
            context_token_ids_sha256: case.context_token_ids_sha256.clone(),
            context_length: case.context_length,
            input_sha256: case.input_sha256.clone(),
            output_offset_bytes: offset,
            output_elements: self.shape,
            output_sha256: sha256_bytes(&bytes),
            finite: true,
        });
        Ok(())
    }

    fn finish(mut self) -> Result<OutputReport> {
        self.file
            .flush()
            .map_err(|err| format!("failed flushing {}: {err}", self.path.display()))?;
        self.file
            .sync_all()
            .map_err(|err| format!("failed syncing {}: {err}", self.path.display()))?;
        Ok(OutputReport {
            path: self.path.display().to_string(),
            shape: vec![self.shape],
            dtype: "f32".to_string(),
            row_order: "input_jsonl_order".to_string(),
            bytes: self.bytes,
            sha256: hex_digest(self.digest.finalize()),
            cases: self.cases,
        })
    }
}

#[derive(Debug, Default)]
struct StageAccumulator {
    records: usize,
    elements_per_record: Option<usize>,
    sum_sq: f64,
    max_abs: f64,
    samples: Vec<HybridStageSample>,
}

struct StageEmitter {
    stream_stdout: bool,
    stdout: io::BufWriter<io::Stdout>,
    stages: BTreeMap<String, StageAccumulator>,
}

impl StageEmitter {
    fn new(stream_stdout: bool) -> Self {
        Self {
            stream_stdout,
            stdout: io::BufWriter::new(io::stdout()),
            stages: BTreeMap::new(),
        }
    }

    fn emit(
        &mut self,
        case: &HybridInputCase,
        timestep: usize,
        stage: &str,
        values: &[f32],
    ) -> Result<()> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(format!("hybrid {stage} produced an invalid tensor"));
        }
        let accumulator = self.stages.entry(stage.to_string()).or_default();
        match accumulator.elements_per_record {
            Some(elements) if elements != values.len() => {
                return Err(format!(
                    "hybrid {stage} element count changed: expected {elements} got {}",
                    values.len()
                ));
            }
            Some(_) => {}
            None => accumulator.elements_per_record = Some(values.len()),
        }
        accumulator.records = accumulator
            .records
            .checked_add(1)
            .ok_or_else(|| "hybrid stage record count overflow".to_string())?;
        let mut l2_sq = 0.0_f64;
        let mut max_abs = 0.0_f64;
        for value in values {
            let value64 = f64::from(*value);
            l2_sq += value64 * value64;
            max_abs = max_abs.max(value64.abs());
        }
        accumulator.sum_sq += l2_sq;
        accumulator.max_abs = accumulator.max_abs.max(max_abs);
        let coordinates = fixed_coordinates(values.len());
        accumulator.samples.push(HybridStageSample {
            case_id: case.case_id.clone(),
            step: case.step,
            context_token_ids_sha256: case.context_token_ids_sha256.clone(),
            timestep,
            elements: values.len(),
            values: coordinates.iter().map(|index| values[*index]).collect(),
            coordinates,
            max_abs,
            l2: l2_sq.sqrt(),
        });

        if self.stream_stdout {
            let payload = encode_f32_to_bytes(values);
            let header = serde_json::json!({
                "kind": "stage",
                "case_id": case.case_id,
                "step": case.step,
                "context_token_ids_sha256": case.context_token_ids_sha256,
                "context_length": case.context_length,
                "timestep": timestep,
                "stage": stage,
                "dtype": "f32le",
                "shape": [values.len()],
                "bytes": payload.len(),
            });
            serde_json::to_writer(&mut self.stdout, &header)
                .map_err(|err| format!("failed serializing hybrid stage header: {err}"))?;
            self.stdout
                .write_all(b"\n")
                .and_then(|_| self.stdout.write_all(&payload))
                .and_then(|_| self.stdout.flush())
                .map_err(|err| format!("failed streaming hybrid {stage}: {err}"))?;
        }
        Ok(())
    }

    fn finish_stream(&mut self) -> Result<()> {
        if self.stream_stdout {
            self.stdout
                .write_all(b"{\"kind\":\"end\"}\n")
                .and_then(|_| self.stdout.flush())
                .map_err(|err| format!("failed finalizing hybrid stage stream: {err}"))?;
        }
        Ok(())
    }

    fn summaries(self) -> BTreeMap<String, HybridStageSummary> {
        self.stages
            .into_iter()
            .map(|(name, accumulator)| {
                (
                    name,
                    HybridStageSummary {
                        records: accumulator.records,
                        elements_per_record: accumulator.elements_per_record.unwrap_or(0),
                        aggregate_l2: accumulator.sum_sq.sqrt(),
                        max_abs: accumulator.max_abs,
                        samples: accumulator.samples,
                    },
                )
            })
            .collect()
    }
}

fn fixed_coordinates(elements: usize) -> Vec<usize> {
    let candidates = [
        0_usize,
        1,
        31,
        127,
        1024,
        2048,
        4095,
        elements.saturating_sub(1),
    ];
    let mut coordinates = Vec::new();
    for candidate in candidates {
        if candidate < elements && !coordinates.contains(&candidate) {
            coordinates.push(candidate);
        }
    }
    coordinates
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ullm-aq4-layer0-family-isolation: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = parse_args(env::args().skip(1))?;
    match (&args.input, &args.hybrid_input) {
        (Some(_), None) => run_raw_family_probe(args),
        (None, Some(_)) => run_hybrid_probe(args),
        (Some(_), Some(_)) => Err("--input and --hybrid-input are mutually exclusive".to_string()),
        (None, None) => Err("one of --input or --hybrid-input is required".to_string()),
    }
}

fn run_raw_family_probe(args: Args) -> Result<()> {
    let input_path = args
        .input
        .as_ref()
        .ok_or_else(|| "raw mode requires --input".to_string())?;
    if args.chunk_bytes == 0 || args.chunk_bytes > MAX_CHUNK_BYTES {
        return Err(format!("chunk bytes must be in 1..={MAX_CHUNK_BYTES}"));
    }
    if !args.package.is_dir() {
        return Err(format!(
            "package is not a directory: {}",
            args.package.display()
        ));
    }
    if !input_path.is_file() {
        return Err(format!("input is not a file: {}", input_path.display()));
    }
    if args.output.exists() {
        return Err(format!(
            "refusing to overwrite output: {}",
            args.output.display()
        ));
    }
    fs::create_dir_all(&args.output)
        .map_err(|err| format!("failed creating output directory: {err}"))?;

    let input_bytes = fs::read(input_path).map_err(|err| format!("failed reading input: {err}"))?;
    let input_digest = sha256_bytes(&input_bytes);
    let (header, cases) = parse_input(&input_bytes)?;
    if header.schema_version != INPUT_SCHEMA || header.tensor_name != TENSOR_NAMES[0] {
        return Err("input header does not identify layer0 QKV input".to_string());
    }
    if header.dtype != "f32" || header.shape != [INPUT_COLS] {
        return Err("input must be f32 with shape [4096]".to_string());
    }

    let manifest_path = args.package.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|err| format!("failed reading {}: {err}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|err| format!("failed parsing manifest: {err}"))?;
    let tensor_reports = package_tensor_reports(&args.package, &manifest)?;

    let mut context = ullm_runtime_sys::RuntimeContext::create(0)
        .map_err(|err| format!("failed creating CPU runtime context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed querying CPU runtime device: {err}"))?;
    if !info.backend.eq_ignore_ascii_case("cpu") {
        return Err(format!("device 0 is not CPU: {}", info.backend));
    }
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed creating CPU runtime stream: {err}"))?;
    let package_path = args
        .package
        .to_str()
        .ok_or_else(|| "package path is not UTF-8".to_string())?;
    let mut registry = WeightRegistry::new();
    let mut weights = Vec::new();
    for (family, tensor) in FAMILY_NAMES.iter().zip(TENSOR_NAMES.iter()) {
        let weight = PackageAq4ResidentMatvec::load_single_diagnostic(
            &mut context,
            &mut stream,
            &mut registry,
            package_path,
            tensor,
            args.chunk_bytes,
        )
        .map_err(|err| format!("failed loading {family} AQ4 tensor: {err}"))?;
        if weight.rows != EXPECTED_ROWS[weights.len()] || weight.cols != INPUT_COLS {
            return Err(format!(
                "{family} geometry is [{},{}]",
                weight.rows, weight.cols
            ));
        }
        weights.push(weight);
    }

    let mut input_buffer = context
        .alloc_buffer(INPUT_COLS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed allocating input buffer: {err}"))?;
    let mut output_buffers = Vec::new();
    for rows in EXPECTED_ROWS {
        output_buffers.push(
            context
                .alloc_buffer(rows * std::mem::size_of::<f32>())
                .map_err(|err| format!("failed allocating output buffer: {err}"))?,
        );
    }
    let mut sinks = FAMILY_NAMES
        .iter()
        .zip(EXPECTED_ROWS)
        .map(|(name, rows)| OutputSink::create(&args.output, name, rows))
        .collect::<Result<Vec<_>>>()?;

    for case in &cases {
        let input = encode_f32_to_bytes(&case.values);
        input_buffer
            .copy_from_host(0, &input, Some(&mut stream))
            .map_err(|err| format!("failed uploading input {}: {err}", case.case_id))?;
        for index in 0..weights.len() {
            weights[index]
                .matvec(
                    &input_buffer,
                    &mut output_buffers[index],
                    &mut stream,
                    &format!("aq4_layer0_family_{}", FAMILY_NAMES[index]),
                )
                .map_err(|err| format!("AQ4 {} matvec failed: {err}", FAMILY_NAMES[index]))?;
            let mut bytes = vec![0_u8; EXPECTED_ROWS[index] * std::mem::size_of::<f32>()];
            output_buffers[index]
                .copy_to_host(0, &mut bytes, Some(&mut stream))
                .map_err(|err| format!("failed reading {} output: {err}", FAMILY_NAMES[index]))?;
            stream.synchronize().map_err(|err| {
                format!("failed synchronizing {} output: {err}", FAMILY_NAMES[index])
            })?;
            sinks[index].row(case, &decode_f32_le_values(&bytes))?;
        }
    }
    if cases.is_empty() {
        return Err("input has no cases".to_string());
    }
    let current_input =
        fs::read(input_path).map_err(|err| format!("failed rereading input: {err}"))?;
    if sha256_bytes(&current_input) != input_digest {
        return Err("input changed during probe".to_string());
    }
    let current_manifest = fs::read(&manifest_path)
        .map_err(|err| format!("failed rereading {}: {err}", manifest_path.display()))?;
    if sha256_bytes(&current_manifest) != manifest_sha256 {
        return Err("package manifest changed during probe".to_string());
    }
    let current_tensor_reports = package_tensor_reports(&args.package, &manifest)?;
    if serde_json::to_vec(&current_tensor_reports)
        .map_err(|err| format!("failed serializing package identity: {err}"))?
        != serde_json::to_vec(&tensor_reports)
            .map_err(|err| format!("failed serializing package identity: {err}"))?
    {
        return Err("package tensor payload identity changed during probe".to_string());
    }
    let outputs = sinks
        .into_iter()
        .map(|sink| {
            let name = sink
                .path
                .file_stem()
                .and_then(|value| value.to_str())
                .ok_or_else(|| "output name is not UTF-8".to_string())?
                .to_string();
            Ok((name, sink.finish()?))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let report = ProbeReport {
        schema_version: SCHEMA.to_string(),
        status: "valid".to_string(),
        classification: "unclassified".to_string(),
        promotion: false,
        holdout: "not_run".to_string(),
        policy_evaluation: "policy_not_evaluated".to_string(),
        device: format!("cpu:{}", info.device_id),
        chunk_bytes: args.chunk_bytes,
        package_root: args.package.display().to_string(),
        package_manifest_sha256: manifest_sha256,
        input: InputReport {
            path: input_path.display().to_string(),
            schema: INPUT_SCHEMA.to_string(),
            dtype: header.dtype,
            shape: header.shape,
            rows: cases.len(),
            consumed_sha256: input_digest,
        },
        tensors: tensor_reports,
        outputs,
        family_order: FAMILY_NAMES.iter().map(|name| (*name).to_string()).collect(),
        one_at_a_time_hybrid: HybridReport {
            attempted: false,
            status: "not_implemented".to_string(),
            reason: "This probe has only the independently attributable raw matvec boundary; a full layer hidden-state hybrid would require production/source recurrent-state semantics and is intentionally not inferred here.".to_string(),
        },
    };
    let report_path = args.output.join("aq4-report.json");
    let report_json = serde_json::to_vec_pretty(&report)
        .map_err(|err| format!("failed serializing report: {err}"))?;
    fs::write(&report_path, report_json)
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;
    Ok(())
}

fn run_hybrid_probe(args: Args) -> Result<()> {
    if args.chunk_bytes == 0 || args.chunk_bytes > MAX_CHUNK_BYTES {
        return Err(format!("chunk bytes must be in 1..={MAX_CHUNK_BYTES}"));
    }
    if !args.package.is_dir() {
        return Err(format!(
            "package is not a directory: {}",
            args.package.display()
        ));
    }
    let input_path = args
        .hybrid_input
        .as_ref()
        .ok_or_else(|| "hybrid mode requires --hybrid-input".to_string())?;
    if !input_path.is_file() {
        return Err(format!(
            "hybrid input is not a file: {}",
            input_path.display()
        ));
    }
    if args.output.exists() {
        return Err(format!(
            "refusing to overwrite output: {}",
            args.output.display()
        ));
    }
    fs::create_dir_all(&args.output)
        .map_err(|err| format!("failed creating hybrid output directory: {err}"))?;

    let input_bytes = fs::read(input_path).map_err(|err| {
        format!(
            "failed reading hybrid input {}: {err}",
            input_path.display()
        )
    })?;
    let input_digest = sha256_bytes(&input_bytes);
    let (header, cases) = parse_hybrid_input(&input_bytes)?;
    let input_root = input_path
        .parent()
        .ok_or_else(|| "hybrid input has no parent directory".to_string())?;
    for case in &cases {
        validate_hybrid_case_sidecar(input_root, case)?;
    }

    let manifest_path = args.package.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|err| format!("failed reading {}: {err}", manifest_path.display()))?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);

    let mut emitter = StageEmitter::new(args.stage_stream_stdout);
    let finals = one_at_a_time_hybrid(&args, input_root, &cases, &mut emitter)?;
    emit_diagnostic_lm_head_readout(&args, &finals, &mut emitter)?;
    emitter.finish_stream()?;

    let current_input = fs::read(input_path).map_err(|err| {
        format!(
            "failed rereading hybrid input {}: {err}",
            input_path.display()
        )
    })?;
    if sha256_bytes(&current_input) != input_digest {
        return Err("hybrid input changed during probe".to_string());
    }
    let current_manifest = fs::read(&manifest_path)
        .map_err(|err| format!("failed rereading {}: {err}", manifest_path.display()))?;
    if sha256_bytes(&current_manifest) != manifest_sha256 {
        return Err("package manifest changed during hybrid probe".to_string());
    }

    let report = HybridProbeReport {
        schema_version: HYBRID_SCHEMA.to_string(),
        status: "valid".to_string(),
        classification: "unclassified".to_string(),
        promotion: false,
        holdout: "not_run".to_string(),
        policy_evaluation: "policy_not_evaluated".to_string(),
        device: "cpu:0".to_string(),
        chunk_bytes: args.chunk_bytes,
        package_root: args.package.display().to_string(),
        package_manifest_sha256: manifest_sha256,
        input: HybridInputReport {
            path: input_path.display().to_string(),
            schema: header.schema_version,
            source_embedding_tensor: header.tensor_name,
            dtype: header.dtype,
            shape: header.shape,
            residual_encoding: header.residual_encoding,
            source_model_index_sha256: header.source_model_index_sha256,
            rows: cases.len(),
            consumed_sha256: input_digest,
            cases: cases
                .iter()
                .map(|case| HybridCaseBinding {
                    case_id: case.case_id.clone(),
                    step: case.step,
                    context_token_ids_sha256: case.context_token_ids_sha256.clone(),
                    context_length: case.context_length,
                    residual_path: case.residual_path.clone(),
                    residual_sha256: case.residual_sha256.clone(),
                })
                .collect(),
        },
        one_at_a_time_hybrid: HybridExecutionReport {
            attempted: true,
            status: "valid".to_string(),
            formula: "production standalone layer0: input RMSNorm -> AQ4 QKV/Z/A/B matvec (decoder includes dequant + row-scale) -> causal depthwise Conv1d -> SiLU -> q/k L2 normalization -> gated delta recurrent update -> per-value-head RMSNorm(SiLU(Z)) -> AQ4 out projection -> residual -> post RMSNorm -> AQ4 SwiGLU MLP -> residual".to_string(),
            aq4_projection_tensors: vec![
                TENSOR_NAMES[0].to_string(),
                TENSOR_NAMES[1].to_string(),
                TENSOR_NAMES[2].to_string(),
                TENSOR_NAMES[3].to_string(),
                OUT_TENSOR.to_string(),
                MLP_GATE_TENSOR.to_string(),
                MLP_UP_TENSOR.to_string(),
                MLP_DOWN_TENSOR.to_string(),
            ],
            input_rms_epsilon: INPUT_RMS_EPSILON,
            attention_rms_epsilon: ATTENTION_RMS_EPSILON,
            post_rms_epsilon: AQ4_POST_RMS_EPSILON,
            q_scale: Q_SCALE,
            rope: HybridRopeReport {
                applicable: false,
                status: "not_applicable".to_string(),
                reason: "layer 0 is Qwen3.5 linear attention; it has no RoPE operation. The no-op is recorded explicitly so a self-attention RoPE path is not inferred.".to_string(),
            },
            logits: HybridLogitReport {
                status: "diagnostic_readout_only".to_string(),
                stage: "diagnostic_lm_head_readout_logits".to_string(),
                token_rows: DIAGNOSTIC_LOGIT_ROWS.to_vec(),
                reason: "These are fixed LM-head rows applied directly to the layer-0 output. They are bounded diagnostic readouts, not final-model vocabulary logits because layers 1..31 and final RMSNorm are intentionally outside Phase 1.".to_string(),
            },
            state_contract: "each context is reset to zero conv/recurrent state and replayed token-by-token; Conv1d history is [kernel, channel], recurrent state is [value_head, key_dim, value_dim].".to_string(),
            persistence_contract: "full tensors exist only for the current CPU step or stdout comparison frame; aq4-report.json persists fixed coordinate samples and aggregate L2/max-abs only.".to_string(),
        },
        stage_summaries: emitter.summaries(),
    };
    let report_path = args.output.join("aq4-report.json");
    let report_json = serde_json::to_vec_pretty(&report)
        .map_err(|err| format!("failed serializing hybrid report: {err}"))?;
    fs::write(&report_path, report_json)
        .map_err(|err| format!("failed writing {}: {err}", report_path.display()))?;
    Ok(())
}

fn parse_args<I>(mut args: I) -> Result<Args>
where
    I: Iterator<Item = String>,
{
    let mut package = None;
    let mut input = None;
    let mut hybrid_input = None;
    let mut output = None;
    let mut chunk_bytes = 16 * 1024 * 1024;
    let mut stage_stream_stdout = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--package" => {
                package = Some(PathBuf::from(
                    args.next().ok_or("--package requires a path")?,
                ))
            }
            "--input" => input = Some(PathBuf::from(args.next().ok_or("--input requires a path")?)),
            "--hybrid-input" => {
                hybrid_input = Some(PathBuf::from(
                    args.next().ok_or("--hybrid-input requires a path")?,
                ))
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires a path")?,
                ))
            }
            "--chunk-bytes" => {
                chunk_bytes = args
                    .next()
                    .ok_or("--chunk-bytes requires a value")?
                    .parse()
                    .map_err(|_| "invalid --chunk-bytes".to_string())?
            }
            "--stage-stream-stdout" => stage_stream_stdout = true,
            "--help" | "-h" => {
                return Err(
                    "usage: --package DIR (--input JSONL | --hybrid-input JSONL) --output DIR [--chunk-bytes N] [--stage-stream-stdout]".to_string(),
                )
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Args {
        package: package.ok_or("missing --package")?,
        input,
        hybrid_input,
        output: output.ok_or("missing --output")?,
        chunk_bytes,
        stage_stream_stdout,
    })
}

fn parse_input(bytes: &[u8]) -> Result<(Header, Vec<Case>)> {
    let mut lines = bytes.split(|byte| *byte == b'\n');
    let header_line = lines.next().ok_or("input is empty")?;
    let header: Header = serde_json::from_slice(header_line)
        .map_err(|err| format!("invalid input header: {err}"))?;
    if header.kind != "header" {
        return Err("input first record is not header".to_string());
    }
    let mut cases = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if line.len() > MAX_LINE_BYTES {
            return Err("input line exceeds bound".to_string());
        }
        let case: Case =
            serde_json::from_slice(line).map_err(|err| format!("invalid input case: {err}"))?;
        if case.kind != "case"
            || case.values.len() != INPUT_COLS
            || !case.values.iter().all(|value| value.is_finite())
        {
            return Err(format!("invalid input case {}", case.case_id));
        }
        if cases.len() >= MAX_CASES {
            return Err(format!("input exceeds {MAX_CASES} cases"));
        }
        if case.input_sha256 != sha256_bytes(&encode_f32_to_bytes(&case.values)) {
            return Err(format!("input hash mismatch for {}", case.case_id));
        }
        cases.push(case);
    }
    Ok((header, cases))
}

fn package_tensor_reports(root: &Path, manifest: &serde_json::Value) -> Result<Vec<TensorReport>> {
    let tensors = manifest
        .get("tensors")
        .and_then(serde_json::Value::as_array)
        .ok_or("manifest tensors is not an array")?;
    let mut reports = Vec::new();
    for ((family, name), expected_rows) in FAMILY_NAMES.iter().zip(TENSOR_NAMES).zip(EXPECTED_ROWS)
    {
        let entry = tensors
            .iter()
            .find(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some(name))
            .ok_or_else(|| format!("manifest is missing {name}"))?;
        let shape = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("manifest shape missing for {name}"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| format!("invalid shape for {name}"))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        if shape != [expected_rows, INPUT_COLS] {
            return Err(format!("unexpected manifest shape for {name}: {shape:?}"));
        }
        let index_path = entry
            .get("index_file")
            .and_then(serde_json::Value::as_str)
            .ok_or("manifest index_file missing")?;
        let scale_path = entry
            .get("scale_file")
            .and_then(serde_json::Value::as_str)
            .ok_or("manifest scale_file missing")?;
        let codebook_path = entry
            .get("codebook_file")
            .and_then(serde_json::Value::as_str)
            .ok_or("manifest codebook_file missing")?;
        let entry_bytes =
            serde_json::to_vec(entry).map_err(|err| format!("failed serializing {name}: {err}"))?;
        reports.push(TensorReport {
            family: (*family).to_string(),
            tensor_name: name.to_string(),
            shape: shape.clone(),
            rows: shape[0],
            cols: shape[1],
            dtype: entry
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            manifest_entry_sha256: sha256_bytes(&entry_bytes),
            index_path: index_path.to_string(),
            index_sha256: hash_relative(root, index_path)?,
            scale_path: scale_path.to_string(),
            scale_sha256: hash_relative(root, scale_path)?,
            codebook_path: codebook_path.to_string(),
            codebook_sha256: hash_relative(root, codebook_path)?,
        });
    }
    Ok(reports)
}

fn hash_relative(root: &Path, relative: &str) -> Result<String> {
    let path = root.join(relative);
    let bytes =
        fs::read(&path).map_err(|err| format!("failed reading {}: {err}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_digest(Sha256::digest(bytes))
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parse_hybrid_input(bytes: &[u8]) -> Result<(HybridInputHeader, Vec<HybridInputCase>)> {
    let mut lines = bytes.split(|byte| *byte == b'\n');
    let header_line = lines.next().ok_or("hybrid input is empty")?;
    let header: HybridInputHeader = serde_json::from_slice(header_line)
        .map_err(|err| format!("invalid hybrid input header: {err}"))?;
    if header.kind != "header"
        || header.schema_version != HYBRID_INPUT_SCHEMA
        || header.tensor_name != "model.language_model.embed_tokens.weight"
        || header.dtype != "f32"
        || header.shape != [HIDDEN]
        || header.residual_encoding != "f32le_row_major"
        || !is_sha256_hex(&header.source_model_index_sha256)
    {
        return Err("hybrid input header contract differs".to_string());
    }
    let mut cases = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if line.len() > MAX_LINE_BYTES {
            return Err("hybrid input line exceeds bound".to_string());
        }
        let case: HybridInputCase = serde_json::from_slice(line)
            .map_err(|err| format!("invalid hybrid input case: {err}"))?;
        if case.kind != "case"
            || case.case_id.is_empty()
            || case.context_length == 0
            || case.context_length > HYBRID_MAX_CONTEXT_LENGTH
            || case.context_token_ids.len() != case.context_length
            || case.residual_shape != [case.context_length, HIDDEN]
            || case.residual_dtype != "f32le"
            || case.residual_path.is_empty()
            || !is_sha256_hex(&case.context_token_ids_sha256)
            || !is_sha256_hex(&case.residual_sha256)
        {
            return Err(format!(
                "hybrid input case contract differs: {}",
                case.case_id
            ));
        }
        let context_hash = canonical_token_ids_hash(&case.context_token_ids);
        if context_hash != case.context_token_ids_sha256 {
            return Err(format!("hybrid context hash mismatch for {}", case.case_id));
        }
        if cases
            .iter()
            .any(|other: &HybridInputCase| other.case_id == case.case_id && other.step == case.step)
        {
            return Err(format!(
                "duplicate hybrid case/step: {}:{}",
                case.case_id, case.step
            ));
        }
        if cases.len() >= HYBRID_MAX_CASES {
            return Err(format!("hybrid input exceeds {HYBRID_MAX_CASES} cases"));
        }
        cases.push(case);
    }
    if cases.is_empty() {
        return Err("hybrid input has no cases".to_string());
    }
    Ok((header, cases))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn canonical_token_ids_hash(token_ids: &[u32]) -> String {
    let mut json = String::from("[");
    for (index, token_id) in token_ids.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        json.push_str(&token_id.to_string());
    }
    json.push_str("]\n");
    sha256_bytes(json.as_bytes())
}

fn hybrid_sidecar_path(input_root: &Path, case: &HybridInputCase) -> Result<PathBuf> {
    let path = Path::new(&case.residual_path);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(format!(
            "hybrid residual path must be a relative child path: {}",
            case.residual_path
        ));
    }
    Ok(input_root.join(path))
}

fn validate_hybrid_case_sidecar(input_root: &Path, case: &HybridInputCase) -> Result<()> {
    let path = hybrid_sidecar_path(input_root, case)?;
    let expected_bytes = case
        .context_length
        .checked_mul(HIDDEN)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "hybrid residual byte count overflows".to_string())?;
    let metadata = fs::metadata(&path)
        .map_err(|err| format!("failed statting hybrid residual {}: {err}", path.display()))?;
    if !metadata.is_file() || metadata.len() != expected_bytes as u64 {
        return Err(format!(
            "hybrid residual geometry differs for {}: expected {expected_bytes} bytes",
            case.case_id
        ));
    }
    let bytes = fs::read(&path)
        .map_err(|err| format!("failed reading hybrid residual {}: {err}", path.display()))?;
    if sha256_bytes(&bytes) != case.residual_sha256 {
        return Err(format!(
            "hybrid residual hash mismatch for {}",
            case.case_id
        ));
    }
    let values = decode_f32_le_values(&bytes);
    if values.len() != case.context_length * HIDDEN || values.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "hybrid residual values are invalid for {}",
            case.case_id
        ));
    }
    Ok(())
}

fn read_hybrid_residual(input_root: &Path, case: &HybridInputCase) -> Result<Vec<f32>> {
    let path = hybrid_sidecar_path(input_root, case)?;
    let bytes = fs::read(&path)
        .map_err(|err| format!("failed reading hybrid residual {}: {err}", path.display()))?;
    if sha256_bytes(&bytes) != case.residual_sha256 {
        return Err(format!(
            "hybrid residual changed during run: {}",
            case.case_id
        ));
    }
    let values = decode_f32_le_values(&bytes);
    if values.len() != case.context_length * HIDDEN || values.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "hybrid residual values changed/invalid: {}",
            case.case_id
        ));
    }
    Ok(values)
}

fn one_at_a_time_hybrid(
    args: &Args,
    input_root: &Path,
    cases: &[HybridInputCase],
    emitter: &mut StageEmitter,
) -> Result<Vec<FinalLayerOutput>> {
    let package_path = args
        .package
        .to_str()
        .ok_or_else(|| "package path is not UTF-8".to_string())?;
    let mut context = ullm_runtime_sys::RuntimeContext::create(0)
        .map_err(|err| format!("failed creating CPU runtime context: {err}"))?;
    let device = context
        .device_info()
        .map_err(|err| format!("failed querying CPU runtime device: {err}"))?;
    if !device.backend.eq_ignore_ascii_case("cpu") {
        return Err(format!(
            "hybrid probe requires CPU device zero, got {}",
            device.backend
        ));
    }
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed creating CPU runtime stream: {err}"))?;
    let mut registry = WeightRegistry::new();

    let qkv_weight = load_hybrid_aq4_weight(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        TENSOR_NAMES[0],
        QKV_ROWS,
        HIDDEN,
        args.chunk_bytes,
    )?;
    let z_weight = load_hybrid_aq4_weight(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        TENSOR_NAMES[1],
        HIDDEN,
        HIDDEN,
        args.chunk_bytes,
    )?;
    let a_weight = load_hybrid_aq4_weight(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        TENSOR_NAMES[2],
        VALUE_HEADS,
        HIDDEN,
        args.chunk_bytes,
    )?;
    let b_weight = load_hybrid_aq4_weight(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        TENSOR_NAMES[3],
        VALUE_HEADS,
        HIDDEN,
        args.chunk_bytes,
    )?;
    let out_weight = load_hybrid_aq4_weight(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        OUT_TENSOR,
        HIDDEN,
        HIDDEN,
        args.chunk_bytes,
    )?;
    let mlp_gate_weight = load_hybrid_aq4_weight(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        MLP_GATE_TENSOR,
        INTERMEDIATE,
        HIDDEN,
        args.chunk_bytes,
    )?;
    let mlp_up_weight = load_hybrid_aq4_weight(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        MLP_UP_TENSOR,
        INTERMEDIATE,
        HIDDEN,
        args.chunk_bytes,
    )?;
    let mlp_down_weight = load_hybrid_aq4_weight(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        MLP_DOWN_TENSOR,
        HIDDEN,
        INTERMEDIATE,
        args.chunk_bytes,
    )?;

    let input_norm =
        read_named_passthrough_f32(&args.package, INPUT_NORM_TENSOR, args.chunk_bytes)?;
    let conv = read_named_passthrough_f32(&args.package, CONV_TENSOR, args.chunk_bytes)?;
    let a_log = read_named_passthrough_f32(&args.package, A_LOG_TENSOR, args.chunk_bytes)?;
    let dt_bias = read_named_passthrough_f32(&args.package, DT_BIAS_TENSOR, args.chunk_bytes)?;
    let attn_norm = read_named_passthrough_f32(&args.package, ATTN_NORM_TENSOR, args.chunk_bytes)?;
    let post_norm = read_named_passthrough_f32(&args.package, POST_NORM_TENSOR, args.chunk_bytes)?;
    if input_norm.values.len() != HIDDEN
        || conv.values.len() != QKV_ROWS * CONV_KERNEL
        || a_log.values.len() != VALUE_HEADS
        || dt_bias.values.len() != VALUE_HEADS
        || attn_norm.values.len() != VALUE_DIM
        || post_norm.values.len() != HIDDEN
        || input_norm.values.iter().any(|value| !value.is_finite())
        || conv.values.iter().any(|value| !value.is_finite())
        || a_log.values.iter().any(|value| !value.is_finite())
        || dt_bias.values.iter().any(|value| !value.is_finite())
        || attn_norm.values.iter().any(|value| !value.is_finite())
        || post_norm.values.iter().any(|value| !value.is_finite())
    {
        return Err("layer0 hybrid passthrough geometry/value contract differs".to_string());
    }
    let input_norm_weight = effective_rmsnorm_weight_values(INPUT_NORM_TENSOR, &input_norm.values);
    let post_norm_weight = effective_rmsnorm_weight_values(POST_NORM_TENSOR, &post_norm.values);

    let mut hidden_input_buffer = context
        .alloc_buffer(HIDDEN * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed allocating hybrid hidden input buffer: {err}"))?;
    let mut intermediate_input_buffer = context
        .alloc_buffer(INTERMEDIATE * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed allocating hybrid intermediate input buffer: {err}"))?;
    let mut qkv_output_buffer = context
        .alloc_buffer(QKV_ROWS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed allocating hybrid QKV output buffer: {err}"))?;
    let mut hidden_output_buffer = context
        .alloc_buffer(HIDDEN * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed allocating hybrid hidden output buffer: {err}"))?;
    let mut small_output_buffer = context
        .alloc_buffer(VALUE_HEADS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed allocating hybrid gate output buffer: {err}"))?;
    let mut intermediate_output_buffer = context
        .alloc_buffer(INTERMEDIATE * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed allocating hybrid intermediate output buffer: {err}"))?;

    let mut finals = Vec::with_capacity(cases.len());
    for case in cases {
        let residual_sequence = read_hybrid_residual(input_root, case)?;
        let mut conv_state = vec![0.0_f32; QKV_ROWS * CONV_KERNEL];
        let mut recurrent_state = vec![0.0_f32; STATE_ELEMENTS];
        let mut attention_residuals = Vec::with_capacity(case.context_length * HIDDEN);
        let mut post_normed_sequence = Vec::with_capacity(case.context_length * HIDDEN);

        for timestep in 0..case.context_length {
            let residual = &residual_sequence[timestep * HIDDEN..(timestep + 1) * HIDDEN];
            let input_normed = rmsnorm_f32(residual, &input_norm_weight, INPUT_RMS_EPSILON)?;
            emitter.emit(case, timestep, "input_rmsnorm", &input_normed)?;

            let qkv = aq4_matvec_to_host(
                &qkv_weight,
                &input_normed,
                &mut hidden_input_buffer,
                &mut qkv_output_buffer,
                &mut stream,
                "hybrid_qkv",
            )?;
            emitter.emit(case, timestep, "qkv_dequant_row_scale", &qkv)?;
            let z = aq4_matvec_to_host(
                &z_weight,
                &input_normed,
                &mut hidden_input_buffer,
                &mut hidden_output_buffer,
                &mut stream,
                "hybrid_z",
            )?;
            emitter.emit(case, timestep, "z_dequant_row_scale", &z)?;
            let a = aq4_matvec_to_host(
                &a_weight,
                &input_normed,
                &mut hidden_input_buffer,
                &mut small_output_buffer,
                &mut stream,
                "hybrid_a",
            )?;
            emitter.emit(case, timestep, "a_dequant_row_scale", &a)?;
            let b = aq4_matvec_to_host(
                &b_weight,
                &input_normed,
                &mut hidden_input_buffer,
                &mut small_output_buffer,
                &mut stream,
                "hybrid_b",
            )?;
            emitter.emit(case, timestep, "b_dequant_row_scale", &b)?;

            let conv_pre_silu = conv1d_step_f32(&mut conv_state, &qkv, &conv.values)?;
            emitter.emit(case, timestep, "conv_state_after", &conv_state)?;
            emitter.emit(case, timestep, "conv_pre_silu", &conv_pre_silu)?;
            let conv_silu = silu_f32(&conv_pre_silu);
            emitter.emit(case, timestep, "conv_silu", &conv_silu)?;
            let (q, k, v) = split_qkv_for_recurrent_step(&conv_silu)?;
            emitter.emit(case, timestep, "q_after_l2norm", &q)?;
            emitter.emit(case, timestep, "k_after_l2norm", &k)?;
            emitter.emit(case, timestep, "v_after_split", &v)?;

            let (gate, beta) = runtime_host_linear_attn_gate_beta_f32(
                &a,
                &b,
                &a_log.values,
                &dt_bias.values,
                VALUE_HEADS,
                1,
            );
            if gate.len() != VALUE_HEADS || beta.len() != VALUE_HEADS {
                return Err("hybrid gate/beta helper rejected layer0 geometry".to_string());
            }
            emitter.emit(case, timestep, "recurrent_gate", &gate)?;
            emitter.emit(case, timestep, "recurrent_beta", &beta)?;
            let recurrent = runtime_host_linear_attn_recurrent_f32(
                &q,
                &k,
                &v,
                &gate,
                &beta,
                KEY_HEADS,
                VALUE_HEADS,
                1,
                KEY_DIM,
                VALUE_DIM,
                &mut recurrent_state,
            );
            if recurrent.len() != HIDDEN || recurrent.iter().any(|value| !value.is_finite()) {
                return Err("hybrid recurrent helper produced an invalid output".to_string());
            }
            emitter.emit(case, timestep, "recurrent_state_after", &recurrent_state)?;
            emitter.emit(case, timestep, "recurrent_output", &recurrent)?;

            let (attention_head_rmsnorm, z_silu, gate_composed) =
                attention_gated_norm_f32(&recurrent, &z, &attn_norm.values)?;
            emitter.emit(
                case,
                timestep,
                "attention_head_rmsnorm",
                &attention_head_rmsnorm,
            )?;
            emitter.emit(case, timestep, "z_silu", &z_silu)?;
            emitter.emit(case, timestep, "gate_composed", &gate_composed)?;
            let attention_projection = aq4_matvec_to_host(
                &out_weight,
                &gate_composed,
                &mut hidden_input_buffer,
                &mut hidden_output_buffer,
                &mut stream,
                "hybrid_out",
            )?;
            emitter.emit(
                case,
                timestep,
                "attention_projection",
                &attention_projection,
            )?;
            let attention_residual = add_f32(residual, &attention_projection)?;
            emitter.emit(case, timestep, "attention_residual", &attention_residual)?;
            let post_normed =
                rmsnorm_f32(&attention_residual, &post_norm_weight, AQ4_POST_RMS_EPSILON)?;
            emitter.emit(case, timestep, "post_norm", &post_normed)?;
            attention_residuals.extend_from_slice(&attention_residual);
            post_normed_sequence.extend_from_slice(&post_normed);
        }

        let mut last_layer_output = None;
        for timestep in 0..case.context_length {
            let post_normed = &post_normed_sequence[timestep * HIDDEN..(timestep + 1) * HIDDEN];
            let attention_residual =
                &attention_residuals[timestep * HIDDEN..(timestep + 1) * HIDDEN];
            let mlp_gate = aq4_matvec_to_host(
                &mlp_gate_weight,
                post_normed,
                &mut hidden_input_buffer,
                &mut intermediate_output_buffer,
                &mut stream,
                "hybrid_mlp_gate",
            )?;
            emitter.emit(case, timestep, "mlp_gate_projection", &mlp_gate)?;
            let mlp_up = aq4_matvec_to_host(
                &mlp_up_weight,
                post_normed,
                &mut hidden_input_buffer,
                &mut intermediate_output_buffer,
                &mut stream,
                "hybrid_mlp_up",
            )?;
            emitter.emit(case, timestep, "mlp_up_projection", &mlp_up)?;
            let mlp_gate_silu = silu_f32(&mlp_gate);
            emitter.emit(case, timestep, "mlp_gate_silu", &mlp_gate_silu)?;
            let mlp_activation = mul_f32(&mlp_gate_silu, &mlp_up)?;
            emitter.emit(case, timestep, "mlp_activation", &mlp_activation)?;
            let mlp_output = aq4_matvec_to_host(
                &mlp_down_weight,
                &mlp_activation,
                &mut intermediate_input_buffer,
                &mut hidden_output_buffer,
                &mut stream,
                "hybrid_mlp_down",
            )?;
            emitter.emit(case, timestep, "mlp_output", &mlp_output)?;
            let layer_output = add_f32(attention_residual, &mlp_output)?;
            emitter.emit(case, timestep, "layer_output", &layer_output)?;
            if timestep + 1 == case.context_length {
                last_layer_output = Some(layer_output);
            }
        }
        finals.push(FinalLayerOutput {
            case: case.clone(),
            values: last_layer_output
                .ok_or_else(|| "missing hybrid final layer output".to_string())?,
        });
    }
    Ok(finals)
}

fn load_hybrid_aq4_weight(
    context: &mut ullm_runtime_sys::RuntimeContext,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    registry: &mut WeightRegistry,
    package_path: &str,
    tensor_name: &str,
    expected_rows: usize,
    expected_cols: usize,
    chunk_bytes: usize,
) -> Result<PackageAq4ResidentMatvec> {
    let weight = PackageAq4ResidentMatvec::load_single_diagnostic(
        context,
        stream,
        registry,
        package_path,
        tensor_name,
        chunk_bytes,
    )
    .map_err(|err| format!("failed loading hybrid AQ4 {tensor_name}: {err}"))?;
    if weight.rows != expected_rows || weight.cols != expected_cols {
        return Err(format!(
            "hybrid AQ4 geometry differs for {tensor_name}: expected [{expected_rows},{expected_cols}] got [{},{}]",
            weight.rows, weight.cols
        ));
    }
    Ok(weight)
}

fn aq4_matvec_to_host(
    weight: &PackageAq4ResidentMatvec,
    input: &[f32],
    input_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    output_buffer: &mut ullm_runtime_sys::RuntimeBuffer,
    stream: &mut ullm_runtime_sys::RuntimeStream,
    label: &str,
) -> Result<Vec<f32>> {
    if input.len() != weight.cols || input.iter().any(|value| !value.is_finite()) {
        return Err(format!("invalid hybrid input for {label}"));
    }
    input_buffer
        .copy_from_host(0, &encode_f32_to_bytes(input), Some(stream))
        .map_err(|err| format!("failed uploading hybrid {label} input: {err}"))?;
    weight
        .matvec(input_buffer, output_buffer, stream, label)
        .map_err(|err| format!("hybrid {label} AQ4 matvec failed: {err}"))?;
    let byte_count = weight
        .rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("hybrid {label} output byte size overflows"))?;
    let mut bytes = vec![0_u8; byte_count];
    output_buffer
        .copy_to_host(0, &mut bytes, Some(stream))
        .map_err(|err| format!("failed reading hybrid {label} output: {err}"))?;
    stream
        .synchronize()
        .map_err(|err| format!("failed synchronizing hybrid {label}: {err}"))?;
    let values = decode_f32_le_values(&bytes);
    if values.len() != weight.rows || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("hybrid {label} AQ4 output is invalid"));
    }
    Ok(values)
}

fn rmsnorm_f32(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    if input.len() != weight.len()
        || input.is_empty()
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || input.iter().any(|value| !value.is_finite())
        || weight.iter().any(|value| !value.is_finite())
    {
        return Err("invalid RMSNorm inputs".to_string());
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = 1.0_f32 / (mean_square + epsilon).sqrt();
    let values = input
        .iter()
        .zip(weight)
        .map(|(input_value, weight_value)| input_value * inv_rms * weight_value)
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("RMSNorm output is non-finite".to_string());
    }
    Ok(values)
}

fn silu_f32(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| *value * (1.0_f32 / (1.0_f32 + (-*value).exp())))
        .collect()
}

fn mul_f32(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    if left.len() != right.len()
        || left.iter().any(|value| !value.is_finite())
        || right.iter().any(|value| !value.is_finite())
    {
        return Err("invalid elementwise multiply inputs".to_string());
    }
    let values = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("elementwise multiply output is non-finite".to_string());
    }
    Ok(values)
}

fn add_f32(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    if left.len() != right.len()
        || left.iter().any(|value| !value.is_finite())
        || right.iter().any(|value| !value.is_finite())
    {
        return Err("invalid residual add inputs".to_string());
    }
    let values = left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("residual add output is non-finite".to_string());
    }
    Ok(values)
}

/// Matches `LinearAttnConv1dStepState::step` in the AQ4 runtime: history is
/// `[kernel, channel]`, rotates once per token, then performs a depthwise
/// causal convolution.  SiLU is intentionally *not* applied here.
fn conv1d_step_f32(state: &mut [f32], current: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if state.len() != QKV_ROWS * CONV_KERNEL
        || current.len() != QKV_ROWS
        || weight.len() != QKV_ROWS * CONV_KERNEL
        || current.iter().any(|value| !value.is_finite())
        || weight.iter().any(|value| !value.is_finite())
    {
        return Err("invalid depthwise Conv1d step inputs".to_string());
    }
    state.rotate_left(QKV_ROWS);
    let latest_start = (CONV_KERNEL - 1) * QKV_ROWS;
    state[latest_start..latest_start + QKV_ROWS].copy_from_slice(current);
    let mut output = vec![0.0_f32; QKV_ROWS];
    for channel in 0..QKV_ROWS {
        let mut value = 0.0_f32;
        for kernel in 0..CONV_KERNEL {
            value += state[kernel * QKV_ROWS + channel] * weight[channel * CONV_KERNEL + kernel];
        }
        output[channel] = value;
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("depthwise Conv1d step output is non-finite".to_string());
    }
    Ok(output)
}

fn split_qkv_for_recurrent_step(values: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let qk_elements = KEY_HEADS * KEY_DIM;
    if values.len() != QKV_ROWS || qk_elements * 2 + HIDDEN != QKV_ROWS {
        return Err("invalid layer0 QKV split geometry".to_string());
    }
    let mut q = values[..qk_elements].to_vec();
    let mut k = values[qk_elements..qk_elements * 2].to_vec();
    let v = values[qk_elements * 2..].to_vec();
    for head in 0..KEY_HEADS {
        let start = head * KEY_DIM;
        let end = start + KEY_DIM;
        let q_norm =
            (q[start..end].iter().map(|value| value * value).sum::<f32>() + 1e-6_f32).sqrt();
        let k_norm =
            (k[start..end].iter().map(|value| value * value).sum::<f32>() + 1e-6_f32).sqrt();
        for value in &mut q[start..end] {
            *value = (*value / q_norm) * Q_SCALE;
        }
        for value in &mut k[start..end] {
            *value /= k_norm;
        }
    }
    if q.iter().chain(&k).chain(&v).any(|value| !value.is_finite()) {
        return Err("layer0 QKV split produced a non-finite value".to_string());
    }
    Ok((q, k, v))
}

fn attention_gated_norm_f32(
    recurrent: &[f32],
    z: &[f32],
    head_weight: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if recurrent.len() != HIDDEN
        || z.len() != HIDDEN
        || head_weight.len() != VALUE_DIM
        || recurrent.iter().any(|value| !value.is_finite())
        || z.iter().any(|value| !value.is_finite())
        || head_weight.iter().any(|value| !value.is_finite())
    {
        return Err("invalid attention gated RMSNorm inputs".to_string());
    }
    let mut normed = Vec::with_capacity(HIDDEN);
    for head in 0..VALUE_HEADS {
        let start = head * VALUE_DIM;
        normed.extend_from_slice(&rmsnorm_f32(
            &recurrent[start..start + VALUE_DIM],
            head_weight,
            ATTENTION_RMS_EPSILON,
        )?);
    }
    let z_silu = silu_f32(z);
    let composed = mul_f32(&normed, &z_silu)?;
    Ok((normed, z_silu, composed))
}

fn emit_diagnostic_lm_head_readout(
    args: &Args,
    finals: &[FinalLayerOutput],
    emitter: &mut StageEmitter,
) -> Result<()> {
    if finals.is_empty() {
        return Err("hybrid diagnostic has no final layer output for LM-head readout".to_string());
    }
    let package_path = args
        .package
        .to_str()
        .ok_or_else(|| "package path is not UTF-8".to_string())?;
    let mut context = ullm_runtime_sys::RuntimeContext::create(0)
        .map_err(|err| format!("failed creating CPU LM-head context: {err}"))?;
    let info = context
        .device_info()
        .map_err(|err| format!("failed querying CPU LM-head device: {err}"))?;
    if !info.backend.eq_ignore_ascii_case("cpu") {
        return Err(format!(
            "hybrid LM-head readout requires CPU, got {}",
            info.backend
        ));
    }
    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed creating CPU LM-head stream: {err}"))?;
    let mut registry = WeightRegistry::new();
    let lm_head = PackageAq4ResidentMatvec::load_single_diagnostic(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        LM_HEAD_TENSOR,
        args.chunk_bytes,
    )
    .map_err(|err| format!("failed loading diagnostic AQ4 LM head: {err}"))?;
    if lm_head.cols != HIDDEN || lm_head.rows <= *DIAGNOSTIC_LOGIT_ROWS.iter().max().unwrap_or(&0) {
        return Err(format!(
            "diagnostic LM-head geometry differs: [{},{}]",
            lm_head.rows, lm_head.cols
        ));
    }
    let mut row_buffer = context
        .alloc_buffer(HIDDEN * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed allocating diagnostic LM-head row buffer: {err}"))?;
    for final_output in finals {
        if final_output.values.len() != HIDDEN
            || final_output.values.iter().any(|value| !value.is_finite())
        {
            return Err("invalid final layer output before diagnostic LM-head readout".to_string());
        }
        let mut logits = Vec::with_capacity(DIAGNOSTIC_LOGIT_ROWS.len());
        for row_index in DIAGNOSTIC_LOGIT_ROWS {
            lm_head
                .row_f32(
                    row_index,
                    &mut row_buffer,
                    &mut stream,
                    "hybrid_diagnostic_lm_head_row",
                )
                .map_err(|err| {
                    format!("failed decoding diagnostic LM-head row {row_index}: {err}")
                })?;
            let mut bytes = vec![0_u8; HIDDEN * std::mem::size_of::<f32>()];
            row_buffer
                .copy_to_host(0, &mut bytes, Some(&mut stream))
                .map_err(|err| {
                    format!("failed reading diagnostic LM-head row {row_index}: {err}")
                })?;
            stream.synchronize().map_err(|err| {
                format!("failed synchronizing diagnostic LM-head row {row_index}: {err}")
            })?;
            let row = decode_f32_le_values(&bytes);
            let logit = row
                .iter()
                .zip(&final_output.values)
                .map(|(weight, hidden)| weight * hidden)
                .sum::<f32>();
            if !logit.is_finite() {
                return Err("diagnostic LM-head readout produced a non-finite logit".to_string());
            }
            logits.push(logit);
        }
        emitter.emit(
            &final_output.case,
            final_output.case.context_length - 1,
            "diagnostic_lm_head_readout_logits",
            &logits,
        )?;
    }
    Ok(())
}
