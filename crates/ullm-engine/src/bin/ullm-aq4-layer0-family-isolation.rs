// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU-only layer-0 AQ4 family isolation probe.
//!
//! This diagnostic intentionally calls the production standalone AQ4 matvec
//! once per weight family (QKV, Z, A, and B).  It does not call a fused
//! production operator and it does not change production defaults.  The raw
//! f32 rows are written for an independent BF16 source comparison tool.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use ullm_engine::aq4_package_runtime::PackageAq4ResidentMatvec;
use ullm_engine::host_bytes::{decode_f32_le_values, encode_f32_to_bytes};
use ullm_engine::loader::WeightRegistry;

const SCHEMA: &str = "ullm.aq4_layer0_family_isolation.aq4_cpu.v1";
const INPUT_SCHEMA: &str = "ullm.aq4_layer0_input_normed_jsonl.v1";
const INPUT_COLS: usize = 4096;
const MAX_CASES: usize = 4096;
const MAX_LINE_BYTES: usize = 2 * 1024 * 1024;
const MAX_CHUNK_BYTES: usize = 256 * 1024 * 1024;

const FAMILY_NAMES: [&str; 4] = ["qkv", "z", "a", "b"];
const TENSOR_NAMES: [&str; 4] = [
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.linear_attn.in_proj_z.weight",
    "model.language_model.layers.0.linear_attn.in_proj_a.weight",
    "model.language_model.layers.0.linear_attn.in_proj_b.weight",
];
const EXPECTED_ROWS: [usize; 4] = [8192, 4096, 32, 32];

type Result<T> = std::result::Result<T, String>;

#[derive(Debug)]
struct Args {
    package: PathBuf,
    input: PathBuf,
    output: PathBuf,
    chunk_bytes: usize,
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

fn main() {
    if let Err(error) = run() {
        eprintln!("ullm-aq4-layer0-family-isolation: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = parse_args(env::args().skip(1))?;
    if args.chunk_bytes == 0 || args.chunk_bytes > MAX_CHUNK_BYTES {
        return Err(format!("chunk bytes must be in 1..={MAX_CHUNK_BYTES}"));
    }
    if !args.package.is_dir() {
        return Err(format!(
            "package is not a directory: {}",
            args.package.display()
        ));
    }
    if !args.input.is_file() {
        return Err(format!("input is not a file: {}", args.input.display()));
    }
    if args.output.exists() {
        return Err(format!(
            "refusing to overwrite output: {}",
            args.output.display()
        ));
    }
    fs::create_dir_all(&args.output)
        .map_err(|err| format!("failed creating output directory: {err}"))?;

    let input_bytes =
        fs::read(&args.input).map_err(|err| format!("failed reading input: {err}"))?;
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
        fs::read(&args.input).map_err(|err| format!("failed rereading input: {err}"))?;
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
            path: args.input.display().to_string(),
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

fn parse_args<I>(mut args: I) -> Result<Args>
where
    I: Iterator<Item = String>,
{
    let mut package = None;
    let mut input = None;
    let mut output = None;
    let mut chunk_bytes = 16 * 1024 * 1024;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--package" => {
                package = Some(PathBuf::from(
                    args.next().ok_or("--package requires a path")?,
                ))
            }
            "--input" => input = Some(PathBuf::from(args.next().ok_or("--input requires a path")?)),
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
            "--help" | "-h" => {
                return Err(
                    "usage: --package DIR --input JSONL --output DIR [--chunk-bytes N]".to_string(),
                )
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Args {
        package: package.ok_or("missing --package")?,
        input: input.ok_or("missing --input")?,
        output: output.ok_or("missing --output")?,
        chunk_bytes,
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
