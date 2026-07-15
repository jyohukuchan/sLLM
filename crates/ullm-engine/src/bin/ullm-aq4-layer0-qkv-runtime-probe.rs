// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Standalone AQ4 layer-0 QKV runtime probe.
//!
//! The probe deliberately takes input_normed vectors from an external JSONL
//! sidecar.  It performs one direct `PackageAq4ResidentMatvec::matvec` call
//! per row, writes concatenated little-endian f32 output, and emits an
//! identity-bound report.  It does not run the fused QKV/Z/Gate/Beta wrapper.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ullm_engine::aq4_package_runtime::PackageAq4ResidentMatvec;
use ullm_engine::host_bytes::encode_f32_to_bytes;
use ullm_engine::loader::WeightRegistry;
use ullm_engine::package::{TensorPayloadBundle, select_tensor_payload_bundle};

const SCHEMA: &str = "ullm.aq4_layer0_qkv_runtime_probe.v1";
const INPUT_SCHEMA: &str = "ullm.aq4_layer0_input_normed_jsonl.v1";
const DEFAULT_TENSOR: &str = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
const EXPECTED_INPUT_COLS: usize = 4096;
const EXPECTED_OUTPUT_ROWS: usize = 8192;
const DEFAULT_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_INPUT_LINE_BYTES: usize = 2 * 1024 * 1024;
const MAX_CASES: usize = 4096;
const MAX_CONTEXT_LENGTH: usize = 1 << 20;
const MAX_OUTPUT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

type ProbeResult<T> = Result<T, String>;

#[derive(Debug, Clone)]
struct Args {
    package: PathBuf,
    input: PathBuf,
    output_dir: PathBuf,
    device_index: u32,
    chunk_bytes: usize,
    tensor_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputHeader {
    kind: String,
    schema_version: String,
    tensor_name: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputCase {
    kind: String,
    case_id: String,
    step: usize,
    context_token_ids_sha256: String,
    context_length: usize,
    input_sha256: String,
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct DeviceReport {
    device_index: u32,
    device_id: i32,
    backend: String,
    name: String,
    total_global_mem: u64,
    compute_major: i32,
    compute_minor: i32,
    gcn_arch_name: String,
    flags: u32,
}

#[derive(Debug, Serialize)]
struct GuardReport {
    hip_aq4_matvec_kernel_required: bool,
    fallback_allowed: bool,
    relevant_environment: std::collections::BTreeMap<String, Option<String>>,
}

#[derive(Debug, Serialize)]
struct PackageFileReport {
    relative_path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct PackageReport {
    root: String,
    manifest_sha256: String,
    manifest_schema_version: String,
    tensor_name: String,
    tensor_dtype: Option<String>,
    tensor_shape: Vec<u64>,
    tensor_family: Option<String>,
    tensor_candidate_id: Option<String>,
    group_size: Option<usize>,
    tensor_scale_f32: Option<f32>,
    index_encoding: Option<String>,
    scale_encoding: Option<String>,
    scale_format: Option<String>,
    row_scale_count: usize,
    payload_sha256: String,
    index: PackageFileReport,
    scale: PackageFileReport,
    codebook: PackageFileReport,
}

#[derive(Debug, Serialize)]
struct InputReport {
    path: String,
    sidecar_sha256: String,
    schema: String,
    dtype: String,
    shape: Vec<usize>,
    rows: usize,
    identity: InputSidecarIdentity,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
struct FileStat {
    device: u64,
    inode: u64,
    size_bytes: u64,
    mtime_ns: i64,
    nlink: u64,
}

#[derive(Debug, Serialize, Clone)]
struct InputSidecarIdentity {
    canonical_path: String,
    pre_stat: FileStat,
    post_stat: FileStat,
    consumed_sha256: String,
}

#[derive(Debug, Serialize)]
struct OutputCaseReport {
    case_id: String,
    step: usize,
    context_token_ids_sha256: String,
    context_length: usize,
    input_sha256: String,
    output_offset_bytes: u64,
    output_elements: usize,
    output_sha256: String,
}

#[derive(Debug, Serialize)]
struct OutputReport {
    path: String,
    format: String,
    dtype: String,
    row_shape: Vec<usize>,
    row_order: String,
    bytes: u64,
    sha256: String,
    cases: Vec<OutputCaseReport>,
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    schema_version: String,
    status: String,
    classification: String,
    promotion_eligible: bool,
    operation: String,
    fused: bool,
    device: DeviceReport,
    guard: GuardReport,
    package: PackageReport,
    input: InputReport,
    output: OutputReport,
}

#[derive(Debug)]
struct PackageIdentity {
    root: PathBuf,
    manifest_sha256: String,
    manifest_schema_version: String,
    bundle: TensorPayloadBundle,
    index: PackageFileReport,
    scale: PackageFileReport,
    codebook: PackageFileReport,
    payload_sha256: String,
}

#[derive(Debug)]
struct AtomicFile {
    temp: PathBuf,
    final_path: PathBuf,
    file: File,
}

impl AtomicFile {
    fn create(final_path: PathBuf) -> ProbeResult<Self> {
        if fs::symlink_metadata(&final_path).is_ok() {
            return Err(format!(
                "refusing to overwrite existing output {}",
                final_path.display()
            ));
        }
        let parent = final_path
            .parent()
            .ok_or_else(|| format!("output path has no parent: {}", final_path.display()))?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| format!("system clock before UNIX epoch: {err}"))?
            .as_nanos();
        let temp = parent.join(format!(
            ".{}.tmp-{}-{stamp}",
            final_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("output"),
            std::process::id()
        ));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|err| {
                format!(
                    "failed to create atomic temporary output {}: {err}",
                    temp.display()
                )
            })?;
        Ok(Self {
            temp,
            final_path,
            file,
        })
    }

    fn publish(mut self) -> ProbeResult<()> {
        self.file
            .flush()
            .map_err(|err| format!("failed to flush {}: {err}", self.temp.display()))?;
        self.file
            .sync_all()
            .map_err(|err| format!("failed to sync {}: {err}", self.temp.display()))?;
        fs::hard_link(&self.temp, &self.final_path).map_err(|err| {
            let _ = fs::remove_file(&self.temp);
            format!(
                "failed to atomically publish {} (target must not already exist): {err}",
                self.final_path.display()
            )
        })?;
        fs::remove_file(&self.temp).map_err(|err| {
            format!(
                "published {} but could not remove temporary {}: {err}",
                self.final_path.display(),
                self.temp.display()
            )
        })
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temp);
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ullm-aq4-layer0-qkv-runtime-probe: {err}");
        std::process::exit(1);
    }
}

fn run() -> ProbeResult<()> {
    let args = parse_args(env::args().skip(1))?;
    validate_args(&args)?;
    let package = load_package_identity(&args.package, &args.tensor_name)?;
    let mut input = InputReader::open(&args.input, &args.tensor_name)?;

    let mut context =
        ullm_runtime_sys::RuntimeContext::create(args.device_index).map_err(|err| {
            format!(
                "failed to create runtime context {}: {err}",
                args.device_index
            )
        })?;
    let device_info = context
        .device_info()
        .map_err(|err| format!("failed to query runtime context device: {err}"))?;
    let hip_guard_required = !device_info.backend.eq_ignore_ascii_case("cpu");
    if hip_guard_required && !env_flag_enabled("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL") {
        return Err(
            "non-CPU probe requires ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1 to fail closed"
                .to_string(),
        );
    }
    let device = DeviceReport {
        device_index: args.device_index,
        device_id: device_info.device_id,
        backend: device_info.backend.clone(),
        name: device_info.name.clone(),
        total_global_mem: device_info.total_global_mem,
        compute_major: device_info.compute_major,
        compute_minor: device_info.compute_minor,
        gcn_arch_name: device_info.gcn_arch_name.clone(),
        flags: device_info.flags,
    };
    let guard = GuardReport {
        hip_aq4_matvec_kernel_required: hip_guard_required,
        fallback_allowed: false,
        relevant_environment: relevant_environment(),
    };

    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;
    let mut registry = WeightRegistry::new();
    let matvec = PackageAq4ResidentMatvec::load_single_diagnostic(
        &mut context,
        &mut stream,
        &mut registry,
        args.package
            .to_str()
            .ok_or_else(|| "package path is not valid UTF-8".to_string())?,
        &args.tensor_name,
        args.chunk_bytes,
    )
    .map_err(|err| format!("failed to load AQ4 package tensor: {err}"))?;
    assert_package_unchanged(&args.package, &args.tensor_name, &package)?;
    validate_loaded_geometry(&matvec, &package.bundle)?;

    let mut input_buffer = context
        .alloc_buffer(EXPECTED_INPUT_COLS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate input buffer: {err}"))?;
    let mut output_buffer = context
        .alloc_buffer(EXPECTED_OUTPUT_ROWS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate output buffer: {err}"))?;

    fs::create_dir_all(&args.output_dir).map_err(|err| {
        format!(
            "failed to create output directory {}: {err}",
            args.output_dir.display()
        )
    })?;
    let output_dir = fs::canonicalize(&args.output_dir)
        .map_err(|err| format!("failed to canonicalize output directory: {err}"))?;
    if !fs::symlink_metadata(&output_dir)
        .map_err(|err| format!("failed to inspect output directory: {err}"))?
        .is_dir()
    {
        return Err(format!(
            "output directory is not a directory: {}",
            output_dir.display()
        ));
    }
    let output_path = output_dir.join("output.f32le");
    let report_path = output_dir.join("report.json");
    if fs::symlink_metadata(&output_path).is_ok() || fs::symlink_metadata(&report_path).is_ok() {
        return Err("refusing to overwrite an existing output sidecar".to_string());
    }
    let output_file = AtomicFile::create(output_path)?;
    let report_file = AtomicFile::create(report_path)?;
    let mut output_file = output_file;
    let mut output_digest = Sha256::new();
    let mut output_bytes = 0_u64;
    let mut output_cases = Vec::new();
    let mut seen_cases = HashSet::new();

    while let Some(case) = input.next_case()? {
        if output_cases.len() >= MAX_CASES {
            return Err(format!("input case count exceeds {MAX_CASES}"));
        }
        let key = (case.case_id.clone(), case.step);
        if !seen_cases.insert(key) {
            return Err(format!(
                "duplicate input case: {} step {}",
                case.case_id, case.step
            ));
        }
        if case.values.len() != EXPECTED_INPUT_COLS {
            return Err(format!(
                "input case {} step {} has {} values; expected {EXPECTED_INPUT_COLS}",
                case.case_id,
                case.step,
                case.values.len()
            ));
        }
        if case.context_length > MAX_CONTEXT_LENGTH {
            return Err(format!(
                "input case {} context length exceeds bound",
                case.case_id
            ));
        }
        validate_sha256_hex(&case.context_token_ids_sha256, "context_token_ids_sha256")?;
        validate_sha256_hex(&case.input_sha256, "input_sha256")?;
        let input_bytes = encode_f32_to_bytes(&case.values);
        let actual_input_sha = sha256_bytes(&input_bytes);
        if case.input_sha256 != actual_input_sha {
            return Err(format!(
                "input hash differs for case {} step {}: declared {} actual {}",
                case.case_id, case.step, case.input_sha256, actual_input_sha
            ));
        }
        input_buffer
            .copy_from_host(0, &input_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to upload input case {}: {err}", case.case_id))?;
        matvec
            .matvec(
                &input_buffer,
                &mut output_buffer,
                &mut stream,
                "aq4_layer0_qkv_runtime_probe",
            )
            .map_err(|err| format!("AQ4 runtime matvec failed for case {}: {err}", case.case_id))?;
        let mut row_bytes = vec![0_u8; EXPECTED_OUTPUT_ROWS * std::mem::size_of::<f32>()];
        output_buffer
            .copy_to_host(0, &mut row_bytes, Some(&mut stream))
            .map_err(|err| format!("failed to copy output case {}: {err}", case.case_id))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize output case {}: {err}", case.case_id))?;
        for chunk in row_bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !value.is_finite() {
                return Err(format!("non-finite AQ4 output for case {}", case.case_id));
            }
        }
        let row_sha = sha256_bytes(&row_bytes);
        output_file
            .file
            .write_all(&row_bytes)
            .map_err(|err| format!("failed to write output case {}: {err}", case.case_id))?;
        output_digest.update(&row_bytes);
        let row_bytes_len = u64::try_from(row_bytes.len())
            .map_err(|_| "output row byte count does not fit u64".to_string())?;
        output_bytes = output_bytes
            .checked_add(row_bytes_len)
            .ok_or_else(|| "output byte count overflow".to_string())?;
        if output_bytes > MAX_OUTPUT_BYTES {
            return Err("output exceeds bounded sidecar size".to_string());
        }
        output_cases.push(OutputCaseReport {
            case_id: case.case_id,
            step: case.step,
            context_token_ids_sha256: case.context_token_ids_sha256,
            context_length: case.context_length,
            input_sha256: case.input_sha256,
            output_offset_bytes: output_bytes - row_bytes_len,
            output_elements: EXPECTED_OUTPUT_ROWS,
            output_sha256: row_sha,
        });
    }
    if output_cases.is_empty() {
        return Err("input sidecar contains no cases".to_string());
    }
    assert_package_unchanged(&args.package, &args.tensor_name, &package)?;

    let input_dtype = input.header.dtype.clone();
    let input_shape = input.header.shape.clone();
    let input_sidecar_identity = input.finish_identity()?;
    let input_sidecar_sha256 = input_sidecar_identity.consumed_sha256.clone();
    let package_report = PackageReport {
        root: package.root.display().to_string(),
        manifest_sha256: package.manifest_sha256,
        manifest_schema_version: package.manifest_schema_version,
        tensor_name: package.bundle.tensor_name.clone(),
        tensor_dtype: package.bundle.dtype.clone(),
        tensor_shape: package.bundle.shape.clone(),
        tensor_family: package.bundle.family.clone(),
        tensor_candidate_id: package.bundle.candidate_id.clone(),
        group_size: package.bundle.group_size,
        tensor_scale_f32: package.bundle.tensor_scale,
        index_encoding: package.bundle.index_encoding.clone(),
        scale_encoding: package.bundle.scale_encoding.clone(),
        scale_format: package.bundle.scale_format.clone(),
        row_scale_count: package.bundle.row_scale_overrides.len(),
        payload_sha256: package.payload_sha256,
        index: package.index,
        scale: package.scale,
        codebook: package.codebook,
    };
    let output_report = OutputReport {
        path: output_dir.join("output.f32le").display().to_string(),
        format: "concatenated_little_endian_f32_rows".to_string(),
        dtype: "f32".to_string(),
        row_shape: vec![EXPECTED_OUTPUT_ROWS],
        row_order: "input_jsonl_order".to_string(),
        bytes: output_bytes,
        sha256: hex_digest(output_digest.finalize()),
        cases: output_cases,
    };
    let report = ProbeReport {
        schema_version: SCHEMA.to_string(),
        status: "valid".to_string(),
        classification: "unclassified".to_string(),
        promotion_eligible: false,
        operation: "standalone_aq4_matvec_f32".to_string(),
        fused: false,
        device,
        guard,
        package: package_report,
        input: InputReport {
            path: args.input.display().to_string(),
            sidecar_sha256: input_sidecar_sha256,
            schema: INPUT_SCHEMA.to_string(),
            dtype: input_dtype,
            shape: input_shape,
            rows: output_report.cases.len(),
            identity: input_sidecar_identity,
        },
        output: output_report,
    };
    let report_json = serde_json::to_vec_pretty(&report)
        .map_err(|err| format!("failed to serialize report: {err}"))?;
    let mut report_file = report_file;
    report_file
        .file
        .write_all(&report_json)
        .map_err(|err| format!("failed to write report: {err}"))?;
    // Publish the report and f32 sidecar independently; each target is a
    // complete, no-overwrite hard-link publication.  A consumer accepts the
    // pair only when both files are present and the report hash matches.
    report_file.publish()?;
    output_file.publish()?;
    Ok(())
}

struct InputReader {
    reader: BufReader<File>,
    header: InputHeader,
    digest: Sha256,
    canonical_path: PathBuf,
    pre_stat: FileStat,
}

impl InputReader {
    fn open(path: &Path, expected_tensor: &str) -> ProbeResult<Self> {
        ensure_regular_nlink_one(path, "input sidecar")?;
        let canonical_path = fs::canonicalize(path)
            .map_err(|err| format!("failed to canonicalize input sidecar: {err}"))?;
        let pre_stat = file_stat(&canonical_path, "input sidecar")?;
        if pre_stat.nlink != 1 {
            return Err("input sidecar must have nlink=1".to_string());
        }
        let file = File::open(&canonical_path)
            .map_err(|err| format!("failed to open input sidecar: {err}"))?;
        let mut reader = BufReader::new(file);
        let mut digest = Sha256::new();
        let line = read_capped_line(&mut reader, MAX_INPUT_LINE_BYTES)
            .map_err(|err| format!("failed to read input header: {err}"))?
            .ok_or_else(|| "input sidecar is empty".to_string())?;
        digest.update(&line);
        let header: InputHeader = serde_json::from_slice(trim_newline(&line))
            .map_err(|err| format!("invalid input header JSON: {err}"))?;
        if header.kind != "header"
            || header.schema_version != INPUT_SCHEMA
            || header.tensor_name != expected_tensor
            || header.dtype != "f32"
            || header.shape != [EXPECTED_INPUT_COLS]
        {
            return Err("input sidecar header identity/shape differs".to_string());
        }
        Ok(Self {
            reader,
            header,
            digest,
            canonical_path,
            pre_stat,
        })
    }

    fn next_case(&mut self) -> ProbeResult<Option<InputCase>> {
        let Some(line) = read_capped_line(&mut self.reader, MAX_INPUT_LINE_BYTES)
            .map_err(|err| format!("failed to read input case: {err}"))?
        else {
            return Ok(None);
        };
        self.digest.update(&line);
        let trimmed = trim_newline(&line);
        if trimmed.iter().all(u8::is_ascii_whitespace) {
            return Err("blank input case line is not allowed".to_string());
        }
        let case: InputCase = serde_json::from_slice(trimmed)
            .map_err(|err| format!("invalid input case JSON: {err}"))?;
        if case.kind != "case"
            || case.case_id.is_empty()
            || case.case_id.len() > 256
            || case.values.len() != EXPECTED_INPUT_COLS
            || case.values.iter().any(|value| !value.is_finite())
        {
            return Err(
                "input case identity, geometry, or finite-value contract differs".to_string(),
            );
        }
        Ok(Some(case))
    }

    fn finish_identity(self) -> ProbeResult<InputSidecarIdentity> {
        let InputReader {
            reader,
            header: _,
            digest,
            canonical_path,
            pre_stat,
        } = self;
        drop(reader);
        let post_stat = file_stat(&canonical_path, "input sidecar")?;
        if pre_stat != post_stat {
            return Err("input sidecar changed during probe".to_string());
        }
        let consumed_sha256 = hex_digest(digest.finalize());
        let (whole_sha256, whole_bytes) = hash_file(&canonical_path)?;
        if whole_sha256 != consumed_sha256 || whole_bytes != pre_stat.size_bytes {
            return Err("input sidecar consumed SHA differs from final file".to_string());
        }
        Ok(InputSidecarIdentity {
            canonical_path: canonical_path.display().to_string(),
            pre_stat,
            post_stat,
            consumed_sha256,
        })
    }
}

fn parse_args<I>(mut args: I) -> ProbeResult<Args>
where
    I: Iterator<Item = String>,
{
    let mut package = None;
    let mut input = None;
    let mut output_dir = None;
    let mut device_index = 0_u32;
    let mut chunk_bytes = DEFAULT_CHUNK_BYTES;
    let mut tensor_name = DEFAULT_TENSOR.to_string();
    while let Some(flag) = args.next() {
        let value = |label: &str, args: &mut I| {
            args.next()
                .ok_or_else(|| format!("missing value for {label}"))
        };
        match flag.as_str() {
            "--package" => package = Some(PathBuf::from(value("--package", &mut args)?)),
            "--input" => input = Some(PathBuf::from(value("--input", &mut args)?)),
            "--output-dir" => output_dir = Some(PathBuf::from(value("--output-dir", &mut args)?)),
            "--device-index" => {
                device_index = value("--device-index", &mut args)?
                    .parse()
                    .map_err(|err| format!("invalid --device-index: {err}"))?;
            }
            "--chunk-bytes" => {
                chunk_bytes = value("--chunk-bytes", &mut args)?
                    .parse()
                    .map_err(|err| format!("invalid --chunk-bytes: {err}"))?;
            }
            "--tensor" => tensor_name = value("--tensor", &mut args)?,
            "--help" | "-h" => {
                println!(
                    "usage: ullm-aq4-layer0-qkv-runtime-probe --package DIR --input FILE --output-dir DIR [--device-index N] [--chunk-bytes N] [--tensor NAME]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Args {
        package: package.ok_or_else(|| "--package is required".to_string())?,
        input: input.ok_or_else(|| "--input is required".to_string())?,
        output_dir: output_dir.ok_or_else(|| "--output-dir is required".to_string())?,
        device_index,
        chunk_bytes,
        tensor_name,
    })
}

fn validate_args(args: &Args) -> ProbeResult<()> {
    if args.tensor_name != DEFAULT_TENSOR {
        return Err(format!("probe tensor is fixed to {DEFAULT_TENSOR}"));
    }
    if args.chunk_bytes < 4096 || args.chunk_bytes > 256 * 1024 * 1024 {
        return Err("--chunk-bytes must be between 4096 and 268435456".to_string());
    }
    Ok(())
}

fn load_package_identity(package_path: &Path, tensor_name: &str) -> ProbeResult<PackageIdentity> {
    ensure_directory_not_symlink(package_path, "package directory")?;
    let root = fs::canonicalize(package_path)
        .map_err(|err| format!("failed to canonicalize package directory: {err}"))?;
    let manifest_path = secure_relative_file(&root, "manifest.json", "manifest")?;
    let (manifest_sha256, manifest_bytes) = hash_file(&manifest_path)?;
    if manifest_bytes > u64::try_from(MAX_MANIFEST_BYTES).unwrap_or(u64::MAX) {
        return Err("package manifest exceeds bounded size".to_string());
    }
    let manifest_data = fs::read(&manifest_path)
        .map_err(|err| format!("failed to read package manifest: {err}"))?;
    let manifest_value: serde_json::Value = serde_json::from_slice(&manifest_data)
        .map_err(|err| format!("invalid package manifest JSON: {err}"))?;
    let manifest_schema_version = manifest_value
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "package manifest schema_version is missing".to_string())?
        .to_string();
    if manifest_schema_version != "ullm-prototype-manifest-v0.1" {
        return Err(format!(
            "unsupported package manifest schema_version {manifest_schema_version}"
        ));
    }
    let bundle = select_tensor_payload_bundle(
        &root,
        &ullm_engine::package::TensorSelector::Name(tensor_name.to_string()),
    )
    .map_err(|err| format!("failed to select package tensor: {err}"))?;
    if bundle.tensor_name != tensor_name
        || bundle.dtype.as_deref() != Some("BF16")
        || bundle.shape != [8192, 4096]
        || bundle.family.as_deref() != Some("linear_attn_qkv")
        || bundle.candidate_id.as_deref() != Some("aq4_e4m3_g16_ts_flloyd16")
        || bundle.group_size != Some(16)
        || bundle.index_encoding.as_deref() != Some("idx4_low_nibble_first")
        || bundle.scale_encoding.as_deref() != Some("u8_scale_table_index")
        || bundle.scale_format.as_deref() != Some("e4m3")
        || bundle.elements != 33_554_432
        || bundle.groups != 2_097_152
        || bundle.row_scale_overrides.len() != 0
        || !bundle
            .tensor_scale
            .is_some_and(|value| value.is_finite() && value > 0.0)
    {
        return Err(
            "package QKV tensor identity/geometry differs from the probe contract".to_string(),
        );
    }
    let index_path = secure_bundle_file(&root, &bundle.index_file.relative_path, "qkv index")?;
    let scale_path = secure_bundle_file(&root, &bundle.scale_file.relative_path, "qkv scale")?;
    let codebook_path =
        secure_bundle_file(&root, &bundle.codebook_file.relative_path, "qkv codebook")?;
    let index = package_file_report(&root, &bundle.index_file.relative_path, &index_path)?;
    let scale = package_file_report(&root, &bundle.scale_file.relative_path, &scale_path)?;
    let codebook = package_file_report(&root, &bundle.codebook_file.relative_path, &codebook_path)?;
    let payload_sha256 = hash_files_concat([
        index_path.as_path(),
        scale_path.as_path(),
        codebook_path.as_path(),
    ])?;
    Ok(PackageIdentity {
        root,
        manifest_sha256,
        manifest_schema_version,
        bundle,
        index,
        scale,
        codebook,
        payload_sha256,
    })
}

fn assert_package_unchanged(
    package_path: &Path,
    tensor_name: &str,
    expected: &PackageIdentity,
) -> ProbeResult<()> {
    let current = load_package_identity(package_path, tensor_name)?;
    if current.manifest_sha256 != expected.manifest_sha256
        || current.payload_sha256 != expected.payload_sha256
        || current.index.sha256 != expected.index.sha256
        || current.scale.sha256 != expected.scale.sha256
        || current.codebook.sha256 != expected.codebook.sha256
    {
        return Err("package manifest or QKV payload changed during probe".to_string());
    }
    Ok(())
}

fn validate_loaded_geometry(
    matvec: &PackageAq4ResidentMatvec,
    bundle: &TensorPayloadBundle,
) -> ProbeResult<()> {
    if matvec.rows != EXPECTED_OUTPUT_ROWS
        || matvec.cols != EXPECTED_INPUT_COLS
        || matvec.group_size != bundle.group_size.unwrap_or_default()
        || matvec.scale_count == 0
        || !matvec.tensor_scale.is_finite()
        || matvec.tensor_scale <= 0.0
    {
        return Err("loaded runtime geometry differs from package identity".to_string());
    }
    Ok(())
}

fn package_file_report(root: &Path, relative: &str, path: &Path) -> ProbeResult<PackageFileReport> {
    let canonical = secure_bundle_file(root, relative, "package payload")?;
    if canonical != path {
        return Err(format!(
            "package payload path changed during validation: {relative}"
        ));
    }
    let (sha256, bytes) = hash_file(path)?;
    Ok(PackageFileReport {
        relative_path: relative.to_string(),
        bytes,
        sha256,
    })
}

fn secure_bundle_file(root: &Path, relative: &str, label: &str) -> ProbeResult<PathBuf> {
    secure_relative_file(root, relative, label)
}

fn secure_relative_file(root: &Path, relative: &str, label: &str) -> ProbeResult<PathBuf> {
    let relative_path = Path::new(relative);
    if relative.is_empty()
        || relative_path.is_absolute()
        || relative_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!("{label} path must be a safe relative path"));
    }
    let mut cursor = root.to_path_buf();
    for component in relative_path.components() {
        let Component::Normal(name) = component else {
            continue;
        };
        cursor.push(name);
        let metadata = fs::symlink_metadata(&cursor).map_err(|err| {
            format!(
                "{label} component {} is unavailable: {err}",
                cursor.display()
            )
        })?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "{label} path contains a symlink: {}",
                cursor.display()
            ));
        }
    }
    let canonical =
        fs::canonicalize(&cursor).map_err(|err| format!("{label} path is unavailable: {err}"))?;
    if !canonical.starts_with(root) {
        return Err(format!("{label} path escapes package root"));
    }
    ensure_regular_nlink_one(&canonical, label)?;
    Ok(canonical)
}

fn ensure_directory_not_symlink(path: &Path, label: &str) -> ProbeResult<()> {
    let metadata =
        fs::symlink_metadata(path).map_err(|err| format!("{label} is unavailable: {err}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("{label} must be a non-symlink directory"));
    }
    Ok(())
}

fn file_stat(path: &Path, label: &str) -> ProbeResult<FileStat> {
    let metadata =
        fs::symlink_metadata(path).map_err(|err| format!("{label} is unavailable: {err}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let mtime_ns = i128::from(metadata.mtime())
            .checked_mul(1_000_000_000)
            .and_then(|seconds| seconds.checked_add(i128::from(metadata.mtime_nsec())))
            .ok_or_else(|| format!("{label} mtime overflows nanoseconds"))?;
        let mtime_ns = i64::try_from(mtime_ns)
            .map_err(|_| format!("{label} mtime does not fit i64 nanoseconds"))?;
        return Ok(FileStat {
            device: metadata.dev(),
            inode: metadata.ino(),
            size_bytes: metadata.size(),
            mtime_ns,
            nlink: metadata.nlink(),
        });
    }
    #[cfg(not(unix))]
    {
        let mtime_ns = metadata
            .modified()
            .ok()
            .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
            .and_then(|value| i64::try_from(value.as_nanos()).ok())
            .unwrap_or(0);
        Ok(FileStat {
            device: 0,
            inode: 0,
            size_bytes: metadata.len(),
            mtime_ns,
            nlink: 1,
        })
    }
}

fn ensure_regular_nlink_one(path: &Path, label: &str) -> ProbeResult<()> {
    let stat = file_stat(path, label)?;
    if stat.nlink != 1 {
        return Err(format!("{label} must have nlink=1"));
    }
    Ok(())
}

fn hash_file(path: &Path) -> ProbeResult<(String, u64)> {
    ensure_regular_nlink_one(path, "hashed file")?;
    let mut file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut chunk)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&chunk[..count]);
        bytes = bytes
            .checked_add(u64::try_from(count).map_err(|_| "file byte count overflow".to_string())?)
            .ok_or_else(|| "file byte count overflow".to_string())?;
    }
    Ok((hex_digest(digest.finalize()), bytes))
}

fn hash_files_concat<'a, I>(paths: I) -> ProbeResult<String>
where
    I: IntoIterator<Item = &'a Path>,
{
    let mut digest = Sha256::new();
    for path in paths {
        ensure_regular_nlink_one(path, "hashed package payload")?;
        let mut file =
            File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
        let mut chunk = [0_u8; 1024 * 1024];
        loop {
            let count = file
                .read(&mut chunk)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            if count == 0 {
                break;
            }
            digest.update(&chunk[..count]);
        }
    }
    Ok(hex_digest(digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex_digest(digest.finalize())
}

fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn validate_sha256_hex(value: &str, label: &str) -> ProbeResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{label} must be a lowercase SHA-256 hex digest"));
    }
    Ok(())
}

fn env_flag_enabled(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn relevant_environment() -> std::collections::BTreeMap<String, Option<String>> {
    [
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
        "ULLM_AQ4_MATVEC_RPB",
        "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB",
        "ULLM_AQ4_FUSED_RPB",
    ]
    .into_iter()
    .map(|name| (name.to_string(), env::var(name).ok()))
    .collect()
}

fn trim_newline(line: &[u8]) -> &[u8] {
    line.strip_suffix(b"\n")
        .unwrap_or(line)
        .strip_suffix(b"\r")
        .unwrap_or_else(|| line.strip_suffix(b"\n").unwrap_or(line))
}

fn read_capped_line<R: BufRead>(reader: &mut R, max_bytes: usize) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let take = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if line.len().saturating_add(take) > max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "input line exceeds bounded size",
            ));
        }
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if line.last() == Some(&b'\n') {
            return Ok(Some(line));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn f32_hash_is_little_endian_and_stable() {
        assert_eq!(
            sha256_bytes(&encode_f32_to_bytes(&[1.0, -2.5])),
            "48943f7a0ea247f8e3c9386d0c5822fe181d323a9289980426638cc4e72a43e1"
        );
    }

    #[test]
    fn capped_line_preserves_multiple_lines() {
        let mut reader = BufReader::new(Cursor::new(b"a\nb\n"));
        assert_eq!(
            read_capped_line(&mut reader, 8).unwrap(),
            Some(b"a\n".to_vec())
        );
        assert_eq!(
            read_capped_line(&mut reader, 8).unwrap(),
            Some(b"b\n".to_vec())
        );
        assert_eq!(read_capped_line(&mut reader, 8).unwrap(), None);
    }

    #[test]
    fn safe_relative_path_rejects_parent_and_absolute() {
        let root = env::temp_dir();
        assert!(secure_relative_file(&root, "../escape", "test").is_err());
        assert!(secure_relative_file(&root, "/escape", "test").is_err());
    }

    #[test]
    fn input_header_contract_is_strict() {
        let header = format!(
            "{{\"kind\":\"header\",\"schema_version\":\"{INPUT_SCHEMA}\",\"tensor_name\":\"{DEFAULT_TENSOR}\",\"dtype\":\"f32\",\"shape\":[4096]}}\n"
        );
        let path = env::temp_dir().join(format!("aq4-probe-input-header-{}", std::process::id()));
        fs::write(&path, header).unwrap();
        let mut reader = InputReader::open(&path, DEFAULT_TENSOR).unwrap();
        assert!(reader.next_case().unwrap().is_none());
        let identity = reader.finish_identity().unwrap();
        assert_eq!(identity.pre_stat, identity.post_stat);
        assert_eq!(identity.pre_stat.nlink, 1);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn input_sidecar_replacement_is_fail_closed() {
        let path = env::temp_dir().join(format!(
            "aq4-probe-input-replacement-{}",
            std::process::id()
        ));
        let header = format!(
            "{{\"kind\":\"header\",\"schema_version\":\"{INPUT_SCHEMA}\",\"tensor_name\":\"{DEFAULT_TENSOR}\",\"dtype\":\"f32\",\"shape\":[4096]}}\n"
        );
        fs::write(&path, header).unwrap();
        let reader = InputReader::open(&path, DEFAULT_TENSOR).unwrap();
        fs::write(&path, b"replaced\n").unwrap();
        assert!(
            reader
                .finish_identity()
                .expect_err("replacement must be rejected")
                .contains("changed during probe")
        );
        let _ = fs::remove_file(path);
    }
}
