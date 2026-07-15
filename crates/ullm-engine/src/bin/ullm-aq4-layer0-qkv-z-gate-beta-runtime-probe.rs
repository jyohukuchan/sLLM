// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic AQ4 layer-0 fused QKV/Z/Gate/Beta runtime probe.
//!
//! The probe deliberately takes input_normed vectors from an external JSONL
//! sidecar. It performs one fused
//! `PackageAq4ResidentMatvec::matvec_qkv_z_gate_beta_with` call per row on
//! CPU device zero, compares the fused QKV output with the existing standalone
//! `matvec` path bit-for-bit, writes four independent little-endian f32
//! sidecars, and emits an identity-bound report. This binary is diagnostic
//! only; it does not alter the production layer or its defaults.

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
use ullm_engine::loader::{read_named_passthrough_f32, WeightRegistry};
use ullm_engine::package::{
    select_passthrough_payload_bundle, select_tensor_payload_bundle, PassthroughPayloadBundle,
    TensorPayloadBundle, TensorSelector,
};

const SCHEMA: &str = "ullm.aq4_layer0_qkv_z_gate_beta_runtime_probe.v1";
const INPUT_SCHEMA: &str = "ullm.aq4_layer0_input_normed_jsonl.v1";
const QKV_TENSOR: &str = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
const Z_TENSOR: &str = "model.language_model.layers.0.linear_attn.in_proj_z.weight";
const A_TENSOR: &str = "model.language_model.layers.0.linear_attn.in_proj_a.weight";
const B_TENSOR: &str = "model.language_model.layers.0.linear_attn.in_proj_b.weight";
const A_LOG_TENSOR: &str = "model.language_model.layers.0.linear_attn.A_log";
const DT_BIAS_TENSOR: &str = "model.language_model.layers.0.linear_attn.dt_bias";
const EXPECTED_INPUT_COLS: usize = 4096;
const EXPECTED_QKV_ROWS: usize = 8192;
const EXPECTED_QKV_Q_ROWS: usize = 2048;
const EXPECTED_QKV_K_ROWS: usize = 2048;
const EXPECTED_QKV_V_ROWS: usize = 4096;
const EXPECTED_Z_ROWS: usize = 4096;
const EXPECTED_HEADS: usize = 32;
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
    fused_kernel_required: bool,
    fallback_allowed: bool,
    relevant_environment: std::collections::BTreeMap<String, Option<String>>,
    effective_rpb_raw: std::collections::BTreeMap<String, Option<String>>,
    fused_rpb_raw: Option<String>,
    fused_rpb_effective: Option<u32>,
    fused_rpb_source: String,
}

#[derive(Debug, Serialize)]
struct VisibilityReport {
    hip_visible_devices: Option<String>,
    ullm_hip_visible_devices: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct PackageFileReport {
    relative_path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct TensorReport {
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
struct PassthroughReport {
    tensor_name: String,
    dtype: Option<String>,
    shape: Vec<u64>,
    elements: u64,
    payload_bytes: u64,
    payload_encoding: Option<String>,
    declared_payload_sha256: String,
    payload: PackageFileReport,
}

#[derive(Debug, Serialize)]
struct PackageReport {
    root: String,
    manifest_sha256: String,
    manifest_schema_version: String,
    tensors: Vec<TensorReport>,
    passthrough_inputs: Vec<PassthroughReport>,
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

#[derive(Debug, Serialize, Clone)]
struct OutputCaseReport {
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
    format: String,
    dtype: String,
    row_shape: Vec<usize>,
    row_order: String,
    bytes: u64,
    sha256: String,
    cases: Vec<OutputCaseReport>,
}

#[derive(Debug, Serialize)]
struct QkvRowSegment {
    name: String,
    start_row: usize,
    end_row_exclusive: usize,
}

#[derive(Debug, Serialize)]
struct OutputLayoutReport {
    format: String,
    dtype: String,
    row_order: String,
    qkv_shape: Vec<usize>,
    z_shape: Vec<usize>,
    gate_shape: Vec<usize>,
    beta_shape: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct QkvReferenceReport {
    operation: String,
    bit_exact: bool,
    bit_mismatch_count: usize,
    byte_mismatch_count: usize,
    max_abs: f32,
    relative_l2: f64,
    fused_rows_sha256: Vec<String>,
    standalone_rows_sha256: Vec<String>,
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
    visibility: VisibilityReport,
    guard: GuardReport,
    package: PackageReport,
    input: InputReport,
    outputs: std::collections::BTreeMap<String, OutputReport>,
    qkv_row_segments: Vec<QkvRowSegment>,
    output_layout: OutputLayoutReport,
    qkv_component_reference: QkvReferenceReport,
}

#[derive(Debug)]
struct PackageIdentity {
    root: PathBuf,
    manifest_sha256: String,
    manifest_schema_version: String,
    tensors: Vec<TensorIdentity>,
    passthrough_inputs: Vec<PassthroughIdentity>,
}

#[derive(Debug)]
struct TensorIdentity {
    bundle: TensorPayloadBundle,
    index: PackageFileReport,
    scale: PackageFileReport,
    codebook: PackageFileReport,
    payload_sha256: String,
}

#[derive(Debug)]
struct PassthroughIdentity {
    bundle: PassthroughPayloadBundle,
    payload: PackageFileReport,
    declared_payload_sha256: String,
}

#[derive(Debug)]
struct AtomicFile {
    temp: PathBuf,
    final_path: PathBuf,
    file: File,
}

struct RollbackGuard {
    published: Vec<PublishedPath>,
    committed: bool,
}

struct PublishedPath {
    path: PathBuf,
    identity: FileStat,
}

impl RollbackGuard {
    fn new() -> Self {
        Self {
            published: Vec::new(),
            committed: false,
        }
    }

    fn register_published(&mut self, path: PathBuf) -> ProbeResult<()> {
        let identity = file_stat(&path, "published output")?;
        if identity.nlink != 1 {
            return Err(format!(
                "published output must have nlink=1: {}",
                path.display()
            ));
        }
        self.published.push(PublishedPath { path, identity });
        Ok(())
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for RollbackGuard {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for published in &self.published {
            let Ok(current) = file_stat(&published.path, "rollback output") else {
                continue;
            };
            if current != published.identity {
                continue;
            }
            let _ = fs::remove_file(&published.path);
        }
    }
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

struct OutputSink {
    file: AtomicFile,
    path: PathBuf,
    shape: usize,
    digest: Sha256,
    bytes: u64,
    cases: Vec<OutputCaseReport>,
}

impl OutputSink {
    fn create(output_dir: &Path, name: &str, shape: usize) -> ProbeResult<Self> {
        let path = output_dir.join(format!("{name}.f32le"));
        Ok(Self {
            file: AtomicFile::create(path.clone())?,
            path,
            shape,
            digest: Sha256::new(),
            bytes: 0,
            cases: Vec::new(),
        })
    }

    fn write_row(&mut self, case: &InputCase, row: &[u8]) -> ProbeResult<()> {
        let expected_bytes = self
            .shape
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "output row byte count overflows".to_string())?;
        if row.len() != expected_bytes {
            return Err(format!(
                "output row has {} bytes; expected {expected_bytes}",
                row.len()
            ));
        }
        let row_bytes = u64::try_from(row.len())
            .map_err(|_| "output row byte count does not fit u64".to_string())?;
        let offset = self.bytes;
        self.file
            .file
            .write_all(row)
            .map_err(|err| format!("failed to write {} output row: {err}", self.path.display()))?;
        self.digest.update(row);
        self.bytes = self
            .bytes
            .checked_add(row_bytes)
            .ok_or_else(|| "output byte count overflow".to_string())?;
        if self.bytes > MAX_OUTPUT_BYTES {
            return Err(format!(
                "{} output exceeds bounded sidecar size",
                self.path.display()
            ));
        }
        self.cases.push(OutputCaseReport {
            case_id: case.case_id.clone(),
            step: case.step,
            context_token_ids_sha256: case.context_token_ids_sha256.clone(),
            context_length: case.context_length,
            input_sha256: case.input_sha256.clone(),
            output_offset_bytes: offset,
            output_elements: self.shape,
            output_sha256: sha256_bytes(row),
            finite: true,
        });
        Ok(())
    }

    fn report(&self) -> OutputReport {
        let digest = hex_digest(self.digest.clone().finalize());
        OutputReport {
            path: self.path.display().to_string(),
            format: "concatenated_little_endian_f32_rows".to_string(),
            dtype: "f32".to_string(),
            row_shape: vec![self.shape],
            row_order: "input_jsonl_order".to_string(),
            bytes: self.bytes,
            sha256: digest,
            cases: self.cases.clone(),
        }
    }

    fn publish(self) -> ProbeResult<()> {
        self.file.publish()?;
        Ok(())
    }
}

#[derive(Default)]
struct QkvCompare {
    bit_mismatch_count: usize,
    byte_mismatch_count: usize,
    max_abs: f32,
    sum_diff_sq: f64,
    sum_reference_sq: f64,
    fused_rows_sha256: Vec<String>,
    standalone_rows_sha256: Vec<String>,
}

impl QkvCompare {
    fn row(&mut self, fused: &[u8], standalone: &[u8]) -> ProbeResult<()> {
        if fused.len() != standalone.len() || !fused.len().is_multiple_of(4) {
            return Err("QKV comparison row byte geometry differs".to_string());
        }
        let mut row_bits = 0_usize;
        let mut row_bytes = 0_usize;
        for (fused_chunk, standalone_chunk) in fused.chunks_exact(4).zip(standalone.chunks_exact(4))
        {
            let fused_bits = u32::from_le_bytes(fused_chunk.try_into().unwrap());
            let standalone_bits = u32::from_le_bytes(standalone_chunk.try_into().unwrap());
            if fused_bits != standalone_bits {
                row_bits += (fused_bits ^ standalone_bits).count_ones() as usize;
            }
            row_bytes += fused_chunk
                .iter()
                .zip(standalone_chunk)
                .filter(|(a, b)| a != b)
                .count();
            let fused_value = f32::from_bits(fused_bits);
            let standalone_value = f32::from_bits(standalone_bits);
            if !fused_value.is_finite() || !standalone_value.is_finite() {
                return Err("non-finite QKV comparison value".to_string());
            }
            let diff = f64::from(fused_value) - f64::from(standalone_value);
            self.max_abs = self.max_abs.max(diff.abs() as f32);
            self.sum_diff_sq += diff * diff;
            self.sum_reference_sq += f64::from(standalone_value) * f64::from(standalone_value);
        }
        self.bit_mismatch_count += row_bits;
        self.byte_mismatch_count += row_bytes;
        self.fused_rows_sha256.push(sha256_bytes(fused));
        self.standalone_rows_sha256.push(sha256_bytes(standalone));
        Ok(())
    }

    fn finish(self) -> QkvReferenceReport {
        let relative_l2 = if self.sum_reference_sq == 0.0 {
            if self.sum_diff_sq == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            self.sum_diff_sq.sqrt() / self.sum_reference_sq.sqrt()
        };
        QkvReferenceReport {
            operation: "standalone_aq4_matvec_f32".to_string(),
            bit_exact: self.bit_mismatch_count == 0 && self.byte_mismatch_count == 0,
            bit_mismatch_count: self.bit_mismatch_count,
            byte_mismatch_count: self.byte_mismatch_count,
            max_abs: self.max_abs,
            relative_l2,
            fused_rows_sha256: self.fused_rows_sha256,
            standalone_rows_sha256: self.standalone_rows_sha256,
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe: {err}");
        std::process::exit(1);
    }
}

fn run() -> ProbeResult<()> {
    let args = parse_args(env::args().skip(1))?;
    validate_args(&args)?;
    if !env_flag_enabled("ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL") {
        return Err(
            "fused diagnostic requires ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1, including on CPU"
                .to_string(),
        );
    }
    if args.device_index == 1 {
        validate_hip_environment()?;
    }
    let package = load_package_identity(&args.package)?;
    let mut input = InputReader::open(&args.input, QKV_TENSOR)?;

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
    let is_cpu = args.device_index == 0;
    let is_hip = args.device_index == 1;
    if is_cpu && !device_info.backend.eq_ignore_ascii_case("cpu") {
        return Err(format!(
            "fused diagnostic CPU device 0 requires CPU backend; got {}",
            device_info.backend
        ));
    }
    if is_hip
        && (!device_info.backend.eq_ignore_ascii_case("hip")
            || !device_info.gcn_arch_name.eq_ignore_ascii_case("gfx1201"))
    {
        return Err(format!(
            "fused diagnostic HIP device 1 requires HIP gfx1201; got backend {} arch {}",
            device_info.backend, device_info.gcn_arch_name
        ));
    }
    if !is_cpu && !is_hip {
        return Err(format!(
            "fused diagnostic supports only device 0 CPU or device 1 HIP; got index {} backend {}",
            args.device_index, device_info.backend
        ));
    }
    let hip_guard_required = is_hip;
    let (fused_rpb_raw, fused_rpb_effective, fused_rpb_source) = if is_hip {
        resolve_fused_rpb_environment()?
    } else {
        (None, None, "not_applicable_cpu".to_string())
    };
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
        fused_kernel_required: true,
        fallback_allowed: false,
        relevant_environment: relevant_environment(),
        effective_rpb_raw: effective_rpb_raw_environment(),
        fused_rpb_raw,
        fused_rpb_effective,
        fused_rpb_source,
    };
    let visibility = VisibilityReport {
        hip_visible_devices: env::var("HIP_VISIBLE_DEVICES").ok(),
        ullm_hip_visible_devices: env::var("ULLM_HIP_VISIBLE_DEVICES").ok(),
    };

    let mut stream = context
        .create_stream()
        .map_err(|err| format!("failed to create runtime stream: {err}"))?;
    let mut registry = WeightRegistry::new();
    let package_path = args
        .package
        .to_str()
        .ok_or_else(|| "package path is not valid UTF-8".to_string())?;
    let qkv = PackageAq4ResidentMatvec::load_single_diagnostic(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        QKV_TENSOR,
        args.chunk_bytes,
    )
    .map_err(|err| format!("failed to load AQ4 QKV tensor: {err}"))?;
    let z = PackageAq4ResidentMatvec::load_single_diagnostic(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        Z_TENSOR,
        args.chunk_bytes,
    )
    .map_err(|err| format!("failed to load AQ4 Z tensor: {err}"))?;
    let a = PackageAq4ResidentMatvec::load_single_diagnostic(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        A_TENSOR,
        args.chunk_bytes,
    )
    .map_err(|err| format!("failed to load AQ4 A tensor: {err}"))?;
    let b = PackageAq4ResidentMatvec::load_single_diagnostic(
        &mut context,
        &mut stream,
        &mut registry,
        package_path,
        B_TENSOR,
        args.chunk_bytes,
    )
    .map_err(|err| format!("failed to load AQ4 B tensor: {err}"))?;
    for (label, matvec, name) in [
        ("qkv", &qkv, QKV_TENSOR),
        ("z", &z, Z_TENSOR),
        ("a", &a, A_TENSOR),
        ("b", &b, B_TENSOR),
    ] {
        let identity = tensor_identity(&package, name)?;
        validate_loaded_geometry(matvec, &identity.bundle, label)?;
    }
    let a_log = read_named_passthrough_f32(&args.package, A_LOG_TENSOR, args.chunk_bytes)?;
    let dt_bias = read_named_passthrough_f32(&args.package, DT_BIAS_TENSOR, args.chunk_bytes)?;
    validate_passthrough_values(&a_log, A_LOG_TENSOR, "F32")?;
    validate_passthrough_values(&dt_bias, DT_BIAS_TENSOR, "BF16")?;
    if a_log.values.len() != EXPECTED_HEADS || dt_bias.values.len() != EXPECTED_HEADS {
        return Err("A_log/dt_bias must contain 32 values".to_string());
    }

    let mut input_buffer = context
        .alloc_buffer(EXPECTED_INPUT_COLS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate input buffer: {err}"))?;
    let mut qkv_output_buffer = context
        .alloc_buffer(EXPECTED_QKV_ROWS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate QKV output buffer: {err}"))?;
    let mut standalone_qkv_buffer = context
        .alloc_buffer(EXPECTED_QKV_ROWS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate standalone QKV buffer: {err}"))?;
    let mut z_output_buffer = context
        .alloc_buffer(EXPECTED_Z_ROWS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate Z output buffer: {err}"))?;
    let mut gate_output_buffer = context
        .alloc_buffer(EXPECTED_HEADS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate gate output buffer: {err}"))?;
    let mut beta_output_buffer = context
        .alloc_buffer(EXPECTED_HEADS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate beta output buffer: {err}"))?;
    let mut a_log_buffer = context
        .alloc_buffer(EXPECTED_HEADS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate A_log buffer: {err}"))?;
    let mut dt_bias_buffer = context
        .alloc_buffer(EXPECTED_HEADS * std::mem::size_of::<f32>())
        .map_err(|err| format!("failed to allocate dt_bias buffer: {err}"))?;
    a_log_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&a_log.values), Some(&mut stream))
        .map_err(|err| format!("failed to upload A_log: {err}"))?;
    dt_bias_buffer
        .copy_from_host(0, &encode_f32_to_bytes(&dt_bias.values), Some(&mut stream))
        .map_err(|err| format!("failed to upload dt_bias: {err}"))?;

    let output_dir = prepare_output_directory(&args.output_dir)?;
    let report_path = output_dir.join("report.json");
    if fs::symlink_metadata(&report_path).is_ok()
        || ["qkv", "z", "gate", "beta"]
            .iter()
            .any(|name| fs::symlink_metadata(output_dir.join(format!("{name}.f32le"))).is_ok())
    {
        return Err("refusing to overwrite an existing output sidecar".to_string());
    }
    let report_file = AtomicFile::create(report_path)?;
    let mut rollback = RollbackGuard::new();
    let mut qkv_sink = OutputSink::create(&output_dir, "qkv", EXPECTED_QKV_ROWS)?;
    let mut z_sink = OutputSink::create(&output_dir, "z", EXPECTED_Z_ROWS)?;
    let mut gate_sink = OutputSink::create(&output_dir, "gate", EXPECTED_HEADS)?;
    let mut beta_sink = OutputSink::create(&output_dir, "beta", EXPECTED_HEADS)?;
    let mut qkv_compare = QkvCompare::default();
    let mut seen_cases = HashSet::new();

    while let Some(case) = input.next_case()? {
        if qkv_sink.cases.len() >= MAX_CASES {
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
        qkv.matvec_qkv_z_gate_beta_with(
            &z,
            &a,
            &b,
            &input_buffer,
            &a_log_buffer,
            &dt_bias_buffer,
            &mut qkv_output_buffer,
            &mut z_output_buffer,
            &mut gate_output_buffer,
            &mut beta_output_buffer,
            &mut stream,
            "aq4_layer0_qkv_z_gate_beta_runtime_probe",
        )
        .map_err(|err| format!("AQ4 fused call failed for case {}: {err}", case.case_id))?;
        qkv.matvec(
            &input_buffer,
            &mut standalone_qkv_buffer,
            &mut stream,
            "aq4_layer0_qkv_z_gate_beta_standalone_reference",
        )
        .map_err(|err| format!("AQ4 standalone QKV failed for case {}: {err}", case.case_id))?;
        let mut qkv_row = vec![0_u8; EXPECTED_QKV_ROWS * std::mem::size_of::<f32>()];
        let mut standalone_qkv_row = vec![0_u8; EXPECTED_QKV_ROWS * std::mem::size_of::<f32>()];
        let mut z_row = vec![0_u8; EXPECTED_Z_ROWS * std::mem::size_of::<f32>()];
        let mut gate_row = vec![0_u8; EXPECTED_HEADS * std::mem::size_of::<f32>()];
        let mut beta_row = vec![0_u8; EXPECTED_HEADS * std::mem::size_of::<f32>()];
        qkv_output_buffer
            .copy_to_host(0, &mut qkv_row, Some(&mut stream))
            .map_err(|err| format!("failed to copy fused QKV case {}: {err}", case.case_id))?;
        standalone_qkv_buffer
            .copy_to_host(0, &mut standalone_qkv_row, Some(&mut stream))
            .map_err(|err| format!("failed to copy standalone QKV case {}: {err}", case.case_id))?;
        z_output_buffer
            .copy_to_host(0, &mut z_row, Some(&mut stream))
            .map_err(|err| format!("failed to copy fused Z case {}: {err}", case.case_id))?;
        gate_output_buffer
            .copy_to_host(0, &mut gate_row, Some(&mut stream))
            .map_err(|err| format!("failed to copy fused gate case {}: {err}", case.case_id))?;
        beta_output_buffer
            .copy_to_host(0, &mut beta_row, Some(&mut stream))
            .map_err(|err| format!("failed to copy fused beta case {}: {err}", case.case_id))?;
        stream
            .synchronize()
            .map_err(|err| format!("failed to synchronize output case {}: {err}", case.case_id))?;
        ensure_finite_row(&qkv_row, "qkv", &case.case_id)?;
        ensure_finite_row(&z_row, "z", &case.case_id)?;
        ensure_finite_row(&gate_row, "gate", &case.case_id)?;
        ensure_finite_row(&beta_row, "beta", &case.case_id)?;
        qkv_compare.row(&qkv_row, &standalone_qkv_row)?;
        qkv_sink.write_row(&case, &qkv_row)?;
        z_sink.write_row(&case, &z_row)?;
        gate_sink.write_row(&case, &gate_row)?;
        beta_sink.write_row(&case, &beta_row)?;
    }
    if qkv_sink.cases.is_empty() {
        return Err("input sidecar contains no cases".to_string());
    }
    assert_package_unchanged(&args.package, &package)?;

    let input_dtype = input.header.dtype.clone();
    let input_shape = input.header.shape.clone();
    let input_sidecar_identity = input.finish_identity()?;
    let input_sidecar_sha256 = input_sidecar_identity.consumed_sha256.clone();
    let package_report = PackageReport {
        root: package.root.display().to_string(),
        manifest_sha256: package.manifest_sha256,
        manifest_schema_version: package.manifest_schema_version,
        tensors: package.tensors.iter().map(tensor_report).collect(),
        passthrough_inputs: package
            .passthrough_inputs
            .iter()
            .map(passthrough_report)
            .collect(),
    };
    let qkv_component_reference = qkv_compare.finish();
    if !qkv_component_reference.bit_exact {
        return Err("fused QKV component differs from standalone CPU output".to_string());
    }
    let mut outputs = std::collections::BTreeMap::new();
    outputs.insert("qkv".to_string(), qkv_sink.report());
    outputs.insert("z".to_string(), z_sink.report());
    outputs.insert("gate".to_string(), gate_sink.report());
    outputs.insert("beta".to_string(), beta_sink.report());
    let report = ProbeReport {
        schema_version: SCHEMA.to_string(),
        status: "valid".to_string(),
        classification: "unclassified".to_string(),
        promotion_eligible: false,
        operation: "aq4_matvec_qkv_z_gate_beta_f32".to_string(),
        fused: true,
        device,
        visibility,
        guard,
        package: package_report,
        input: InputReport {
            path: args.input.display().to_string(),
            sidecar_sha256: input_sidecar_sha256,
            schema: INPUT_SCHEMA.to_string(),
            dtype: input_dtype,
            shape: input_shape,
            rows: outputs.get("qkv").map_or(0, |output| output.cases.len()),
            identity: input_sidecar_identity,
        },
        outputs,
        qkv_row_segments: vec![
            QkvRowSegment {
                name: "Q".to_string(),
                start_row: 0,
                end_row_exclusive: EXPECTED_QKV_Q_ROWS,
            },
            QkvRowSegment {
                name: "K".to_string(),
                start_row: EXPECTED_QKV_Q_ROWS,
                end_row_exclusive: EXPECTED_QKV_Q_ROWS + EXPECTED_QKV_K_ROWS,
            },
            QkvRowSegment {
                name: "V".to_string(),
                start_row: EXPECTED_QKV_Q_ROWS + EXPECTED_QKV_K_ROWS,
                end_row_exclusive: EXPECTED_QKV_Q_ROWS + EXPECTED_QKV_K_ROWS + EXPECTED_QKV_V_ROWS,
            },
        ],
        output_layout: OutputLayoutReport {
            format: "concatenated_little_endian_f32_rows".to_string(),
            dtype: "f32".to_string(),
            row_order: "input_jsonl_order".to_string(),
            qkv_shape: vec![EXPECTED_QKV_ROWS],
            z_shape: vec![EXPECTED_Z_ROWS],
            gate_shape: vec![EXPECTED_HEADS],
            beta_shape: vec![EXPECTED_HEADS],
        },
        qkv_component_reference,
    };
    let report_json = serde_json::to_vec_pretty(&report)
        .map_err(|err| format!("failed to serialize report: {err}"))?;
    let mut report_file = report_file;
    report_file
        .file
        .write_all(&report_json)
        .map_err(|err| format!("failed to write report: {err}"))?;
    // Sidecars publish before the report commit marker. RollbackGuard removes
    // every already-published sidecar if any later publication fails.
    let qkv_path = qkv_sink.path.clone();
    qkv_sink.publish()?;
    rollback.register_published(qkv_path)?;
    let z_path = z_sink.path.clone();
    z_sink.publish()?;
    rollback.register_published(z_path)?;
    let gate_path = gate_sink.path.clone();
    gate_sink.publish()?;
    rollback.register_published(gate_path)?;
    let beta_path = beta_sink.path.clone();
    beta_sink.publish()?;
    rollback.register_published(beta_path)?;
    let report_path = report_file.final_path.clone();
    report_file.publish()?;
    rollback.register_published(report_path)?;
    rollback.commit();
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
            "--help" | "-h" => {
                println!(
                    "usage: ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe --package DIR --input FILE --output-dir DIR [--device-index 0] [--chunk-bytes N]"
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
    })
}

fn validate_args(args: &Args) -> ProbeResult<()> {
    if args.device_index > 1 {
        return Err("--device-index must be 0 (CPU) or 1 (HIP gfx1201)".to_string());
    }
    if args.chunk_bytes < 4096 || args.chunk_bytes > 256 * 1024 * 1024 {
        return Err("--chunk-bytes must be between 4096 and 268435456".to_string());
    }
    Ok(())
}

fn prepare_output_directory(path: &Path) -> ProbeResult<PathBuf> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(format!(
                    "output directory must not be a symlink: {}",
                    path.display()
                ));
            }
            if !metadata.is_dir() {
                return Err(format!(
                    "output directory is not a directory: {}",
                    path.display()
                ));
            }
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(path).map_err(|create_err| {
                format!(
                    "failed to create output directory {}: {create_err}",
                    path.display()
                )
            })?;
        }
        Err(err) => {
            return Err(format!(
                "failed to inspect output directory {}: {err}",
                path.display()
            ));
        }
    }
    let canonical = fs::canonicalize(path)
        .map_err(|err| format!("failed to canonicalize output directory: {err}"))?;
    let metadata = fs::symlink_metadata(&canonical)
        .map_err(|err| format!("failed to inspect output directory: {err}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "output directory is not a non-symlink directory: {}",
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn validate_hip_environment_values(
    hip_visible_devices: Option<&str>,
    ullm_hip_visible_devices: Option<&str>,
    single_guard: Option<&str>,
    fused_guard: Option<&str>,
    fused_rpb: Option<&str>,
    generic_fused_rpb: Option<&str>,
) -> ProbeResult<()> {
    if hip_visible_devices != Some("1") {
        return Err("HIP_VISIBLE_DEVICES must be exactly 1 for HIP device 1".to_string());
    }
    if ullm_hip_visible_devices != Some("1") {
        return Err("ULLM_HIP_VISIBLE_DEVICES must be exactly 1 for HIP device 1".to_string());
    }
    if single_guard != Some("1") {
        return Err("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL must be exactly 1 for HIP".to_string());
    }
    if fused_guard != Some("1") {
        return Err(
            "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL must be exactly 1 for HIP"
                .to_string(),
        );
    }
    resolve_fused_rpb_values(fused_rpb, generic_fused_rpb)?;
    Ok(())
}

fn validate_hip_environment() -> ProbeResult<()> {
    validate_hip_environment_values(
        env::var("HIP_VISIBLE_DEVICES").ok().as_deref(),
        env::var("ULLM_HIP_VISIBLE_DEVICES").ok().as_deref(),
        env::var("ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL")
            .ok()
            .as_deref(),
        env::var("ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL")
            .ok()
            .as_deref(),
        env::var("ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB")
            .ok()
            .as_deref(),
        env::var("ULLM_AQ4_FUSED_RPB").ok().as_deref(),
    )
}

fn resolve_fused_rpb_environment() -> ProbeResult<(Option<String>, Option<u32>, String)> {
    let dedicated = env::var("ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB").ok();
    let generic = env::var("ULLM_AQ4_FUSED_RPB").ok();
    let (effective, source) = resolve_fused_rpb_values(dedicated.as_deref(), generic.as_deref())?;
    if dedicated.as_deref().and_then(parse_rpb_value).is_some() {
        return Ok((
            dedicated,
            Some(effective),
            "environment:ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB".to_string(),
        ));
    }
    Ok((generic, Some(effective), source))
}

fn parse_rpb_value(raw: &str) -> Option<u32> {
    let value = raw.parse::<u32>().ok()?;
    if (1..=32).contains(&value) && 256 % value == 0 {
        Some(value)
    } else {
        None
    }
}

fn resolve_fused_rpb_values(
    dedicated: Option<&str>,
    generic: Option<&str>,
) -> ProbeResult<(u32, String)> {
    if let Some(value) = dedicated.and_then(parse_rpb_value) {
        if value != 4 {
            return Err(format!(
                "dedicated fused RPB must be exactly 4, got {value}"
            ));
        }
        return Ok((
            value,
            "environment:ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB".to_string(),
        ));
    }
    if let Some(value) = generic.and_then(parse_rpb_value) {
        if value != 4 {
            return Err(format!("generic fused RPB must resolve to 4, got {value}"));
        }
        return Ok((value, "environment:ULLM_AQ4_FUSED_RPB".to_string()));
    }
    Err("fused RPB must resolve to exactly 4 from a valid environment value".to_string())
}

fn load_package_identity(package_path: &Path) -> ProbeResult<PackageIdentity> {
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
    validate_manifest_exact_name_counts(&manifest_value)?;
    let contracts = [
        (
            QKV_TENSOR,
            [8192_u64, 4096],
            "linear_attn_qkv",
            33_554_432_u64,
            2_097_152_u64,
        ),
        (
            Z_TENSOR,
            [4096, 4096],
            "linear_attn_z",
            16_777_216,
            1_048_576,
        ),
        (A_TENSOR, [32, 4096], "linear_attn_a", 131_072, 8_192),
        (B_TENSOR, [32, 4096], "linear_attn_b", 131_072, 8_192),
    ];
    let mut tensors = Vec::with_capacity(contracts.len());
    for (tensor_name, shape, family, elements, groups) in contracts {
        let bundle =
            select_tensor_payload_bundle(&root, &TensorSelector::Name(tensor_name.to_string()))
                .map_err(|err| format!("failed to select package tensor {tensor_name}: {err}"))?;
        if bundle.tensor_name != tensor_name
            || bundle.dtype.as_deref() != Some("BF16")
            || bundle.shape != shape
            || bundle.family.as_deref() != Some(family)
            || bundle.candidate_id.as_deref() != Some("aq4_e4m3_g16_ts_flloyd16")
            || bundle.group_size != Some(16)
            || bundle.index_encoding.as_deref() != Some("idx4_low_nibble_first")
            || bundle.scale_encoding.as_deref() != Some("u8_scale_table_index")
            || bundle.scale_format.as_deref() != Some("e4m3")
            || bundle.elements != elements
            || bundle.groups != groups
            || bundle.row_scale_overrides.len() != 0
            || !bundle
                .tensor_scale
                .is_some_and(|value| value.is_finite() && value > 0.0)
        {
            return Err(format!(
                "package {tensor_name} identity/geometry differs from the fused probe contract"
            ));
        }
        let index_path = secure_bundle_file(&root, &bundle.index_file.relative_path, "AQ4 index")?;
        let scale_path = secure_bundle_file(&root, &bundle.scale_file.relative_path, "AQ4 scale")?;
        let codebook_path =
            secure_bundle_file(&root, &bundle.codebook_file.relative_path, "AQ4 codebook")?;
        let index = package_file_report(&root, &bundle.index_file.relative_path, &index_path)?;
        let scale = package_file_report(&root, &bundle.scale_file.relative_path, &scale_path)?;
        let codebook =
            package_file_report(&root, &bundle.codebook_file.relative_path, &codebook_path)?;
        let payload_sha256 = hash_files_concat([
            index_path.as_path(),
            scale_path.as_path(),
            codebook_path.as_path(),
        ])?;
        tensors.push(TensorIdentity {
            bundle,
            index,
            scale,
            codebook,
            payload_sha256,
        });
    }
    let mut passthrough_inputs = Vec::with_capacity(2);
    for (name, dtype, shape) in [
        (A_LOG_TENSOR, "F32", vec![32_u64]),
        (DT_BIAS_TENSOR, "BF16", vec![32_u64]),
    ] {
        let bundle =
            select_passthrough_payload_bundle(&root, &TensorSelector::Name(name.to_string()))
                .map_err(|err| format!("failed to select passthrough tensor {name}: {err}"))?;
        let declared = bundle
            .payload_sha256
            .clone()
            .ok_or_else(|| format!("passthrough tensor {name} lacks payload SHA"))?;
        validate_sha256_hex(&declared, "passthrough payload_sha256")?;
        if bundle.tensor_name != name
            || bundle.dtype.as_deref() != Some(dtype)
            || bundle.shape != shape
            || bundle.elements != 32
        {
            return Err(format!("passthrough tensor {name} identity/shape differs"));
        }
        let payload_path = secure_bundle_file(
            &root,
            &bundle.payload_file.relative_path,
            "passthrough payload",
        )?;
        let payload =
            package_file_report(&root, &bundle.payload_file.relative_path, &payload_path)?;
        if payload.sha256 != declared || payload.bytes != bundle.payload_bytes {
            return Err(format!(
                "passthrough tensor {name} payload SHA/size differs"
            ));
        }
        passthrough_inputs.push(PassthroughIdentity {
            bundle,
            payload,
            declared_payload_sha256: declared,
        });
    }
    Ok(PackageIdentity {
        root,
        manifest_sha256,
        manifest_schema_version,
        tensors,
        passthrough_inputs,
    })
}

fn validate_manifest_exact_name_counts(manifest: &serde_json::Value) -> ProbeResult<()> {
    let names = [
        QKV_TENSOR,
        Z_TENSOR,
        A_TENSOR,
        B_TENSOR,
        A_LOG_TENSOR,
        DT_BIAS_TENSOR,
    ];
    let quantized = manifest
        .get("tensors")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "package manifest tensors array is missing".to_string())?;
    let passthrough = manifest
        .get("passthrough_tensors")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "package manifest passthrough_tensors array is missing".to_string())?;
    for name in names {
        let count = quantized
            .iter()
            .chain(passthrough)
            .filter(|entry| entry.get("name").and_then(serde_json::Value::as_str) == Some(name))
            .count();
        if count != 1 {
            return Err(format!(
                "package manifest exact tensor name {name} must occur once, got {count}"
            ));
        }
    }
    Ok(())
}

fn assert_package_unchanged(package_path: &Path, expected: &PackageIdentity) -> ProbeResult<()> {
    let current = load_package_identity(package_path)?;
    if current.manifest_sha256 != expected.manifest_sha256 {
        return Err("package manifest changed during probe".to_string());
    }
    for expected_tensor in &expected.tensors {
        let current_tensor = tensor_identity(&current, &expected_tensor.bundle.tensor_name)?;
        if current_tensor.payload_sha256 != expected_tensor.payload_sha256
            || current_tensor.index.sha256 != expected_tensor.index.sha256
            || current_tensor.scale.sha256 != expected_tensor.scale.sha256
            || current_tensor.codebook.sha256 != expected_tensor.codebook.sha256
        {
            return Err(format!(
                "package {} payload changed during probe",
                expected_tensor.bundle.tensor_name
            ));
        }
    }
    for expected_passthrough in &expected.passthrough_inputs {
        let current_passthrough =
            passthrough_identity(&current, &expected_passthrough.bundle.tensor_name)?;
        if current_passthrough.declared_payload_sha256
            != expected_passthrough.declared_payload_sha256
            || current_passthrough.payload.sha256 != expected_passthrough.payload.sha256
        {
            return Err(format!(
                "passthrough {} payload changed during probe",
                expected_passthrough.bundle.tensor_name
            ));
        }
    }
    Ok(())
}

fn tensor_identity<'a>(
    package: &'a PackageIdentity,
    name: &str,
) -> ProbeResult<&'a TensorIdentity> {
    package
        .tensors
        .iter()
        .find(|tensor| tensor.bundle.tensor_name == name)
        .ok_or_else(|| format!("package tensor identity is missing: {name}"))
}

fn passthrough_identity<'a>(
    package: &'a PackageIdentity,
    name: &str,
) -> ProbeResult<&'a PassthroughIdentity> {
    package
        .passthrough_inputs
        .iter()
        .find(|tensor| tensor.bundle.tensor_name == name)
        .ok_or_else(|| format!("package passthrough identity is missing: {name}"))
}

fn validate_loaded_geometry(
    matvec: &PackageAq4ResidentMatvec,
    bundle: &TensorPayloadBundle,
    label: &str,
) -> ProbeResult<()> {
    let expected_rows = match label {
        "qkv" => EXPECTED_QKV_ROWS,
        "z" => EXPECTED_Z_ROWS,
        "a" | "b" => EXPECTED_HEADS,
        _ => return Err(format!("unknown tensor geometry label {label}")),
    };
    if matvec.rows != expected_rows
        || matvec.cols != EXPECTED_INPUT_COLS
        || matvec.group_size != bundle.group_size.unwrap_or_default()
        || matvec.scale_count == 0
        || !matvec.tensor_scale.is_finite()
        || matvec.tensor_scale <= 0.0
    {
        return Err(format!(
            "loaded {label} geometry differs from package identity"
        ));
    }
    Ok(())
}

fn tensor_report(identity: &TensorIdentity) -> TensorReport {
    let bundle = &identity.bundle;
    TensorReport {
        tensor_name: bundle.tensor_name.clone(),
        tensor_dtype: bundle.dtype.clone(),
        tensor_shape: bundle.shape.clone(),
        tensor_family: bundle.family.clone(),
        tensor_candidate_id: bundle.candidate_id.clone(),
        group_size: bundle.group_size,
        tensor_scale_f32: bundle.tensor_scale,
        index_encoding: bundle.index_encoding.clone(),
        scale_encoding: bundle.scale_encoding.clone(),
        scale_format: bundle.scale_format.clone(),
        row_scale_count: bundle.row_scale_overrides.len(),
        payload_sha256: identity.payload_sha256.clone(),
        index: identity.index.clone(),
        scale: identity.scale.clone(),
        codebook: identity.codebook.clone(),
    }
}

fn passthrough_report(identity: &PassthroughIdentity) -> PassthroughReport {
    let bundle = &identity.bundle;
    PassthroughReport {
        tensor_name: bundle.tensor_name.clone(),
        dtype: bundle.dtype.clone(),
        shape: bundle.shape.clone(),
        elements: bundle.elements,
        payload_bytes: bundle.payload_bytes,
        payload_encoding: bundle.payload_encoding.clone(),
        declared_payload_sha256: identity.declared_payload_sha256.clone(),
        payload: identity.payload.clone(),
    }
}

fn validate_passthrough_values(
    values: &ullm_engine::loader::PassthroughF32Data,
    name: &str,
    expected_dtype: &str,
) -> ProbeResult<()> {
    if values.dtype != expected_dtype || values.shape != [EXPECTED_HEADS as u64] {
        return Err(format!("passthrough {name} dtype/shape differs"));
    }
    if values.values.iter().any(|value| !value.is_finite()) {
        return Err(format!("passthrough {name} contains non-finite values"));
    }
    Ok(())
}

fn ensure_finite_row(row: &[u8], label: &str, case_id: &str) -> ProbeResult<()> {
    if !row.len().is_multiple_of(4) {
        return Err(format!(
            "{label} output has invalid byte length for {case_id}"
        ));
    }
    for chunk in row.chunks_exact(4) {
        if !f32::from_le_bytes(chunk.try_into().unwrap()).is_finite() {
            return Err(format!(
                "non-finite fused {label} output for case {case_id}"
            ));
        }
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

fn effective_rpb_raw_environment() -> std::collections::BTreeMap<String, Option<String>> {
    [
        "ULLM_AQ4_FUSED_RPB",
        "ULLM_AQ4_MATVEC_QKV_Z_GATE_BETA_RPB",
        "ULLM_AQ4_MATVEC_RPB",
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

    fn temp_path(label: &str) -> PathBuf {
        env::temp_dir().join(format!("aq4-fused-{label}-{}", std::process::id()))
    }

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
            "{{\"kind\":\"header\",\"schema_version\":\"{INPUT_SCHEMA}\",\"tensor_name\":\"{QKV_TENSOR}\",\"dtype\":\"f32\",\"shape\":[4096]}}\n"
        );
        let path = env::temp_dir().join(format!("aq4-probe-input-header-{}", std::process::id()));
        fs::write(&path, header).unwrap();
        let mut reader = InputReader::open(&path, QKV_TENSOR).unwrap();
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
            "{{\"kind\":\"header\",\"schema_version\":\"{INPUT_SCHEMA}\",\"tensor_name\":\"{QKV_TENSOR}\",\"dtype\":\"f32\",\"shape\":[4096]}}\n"
        );
        fs::write(&path, header).unwrap();
        let reader = InputReader::open(&path, QKV_TENSOR).unwrap();
        fs::write(&path, b"replaced\n").unwrap();
        assert!(reader
            .finish_identity()
            .expect_err("replacement must be rejected")
            .contains("changed during probe"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn device_index_contract_accepts_cpu_and_hip_only() {
        let base = vec![
            "--package".to_string(),
            "package".to_string(),
            "--input".to_string(),
            "input".to_string(),
            "--output-dir".to_string(),
            "output".to_string(),
        ];
        for index in ["0", "1"] {
            let mut args = base.clone();
            args.extend(["--device-index".to_string(), index.to_string()]);
            assert!(validate_args(&parse_args(args.into_iter()).unwrap()).is_ok());
        }
        let mut args = base;
        args.extend(["--device-index".to_string(), "2".to_string()]);
        assert!(validate_args(&parse_args(args.into_iter()).unwrap()).is_err());
    }

    #[test]
    fn hip_environment_contract_requires_exact_values() {
        assert!(validate_hip_environment_values(
            Some("1"),
            Some("1"),
            Some("1"),
            Some("1"),
            Some("4"),
            None,
        )
        .is_ok());
        assert!(validate_hip_environment_values(
            Some("1"),
            Some("1"),
            Some("1"),
            Some("1"),
            None,
            Some("4"),
        )
        .is_ok());
        assert!(validate_hip_environment_values(
            Some("0"),
            Some("1"),
            Some("1"),
            Some("1"),
            Some("4"),
            None,
        )
        .is_err());
        assert!(validate_hip_environment_values(
            Some("1"),
            Some("1"),
            Some("1"),
            Some("1"),
            Some("32"),
            None,
        )
        .is_err());
        assert!(validate_hip_environment_values(
            Some("1"),
            Some("1"),
            Some("1"),
            Some("1"),
            Some("2"),
            Some("4"),
        )
        .is_err());
        assert!(validate_hip_environment_values(
            Some("1"),
            Some("1"),
            Some("1"),
            Some("1"),
            Some("4"),
            Some("2"),
        )
        .is_ok());
        assert!(validate_hip_environment_values(
            Some("1"),
            Some("1"),
            Some("1"),
            Some("1"),
            Some("invalid"),
            Some("4"),
        )
        .is_ok());
        assert!(resolve_fused_rpb_values(None, None).is_err());
    }

    #[test]
    fn manifest_exact_name_count_rejects_duplicates() {
        let mut tensors = vec![QKV_TENSOR, Z_TENSOR, A_TENSOR, B_TENSOR]
            .into_iter()
            .map(|name| serde_json::json!({"name": name}))
            .collect::<Vec<_>>();
        let passthrough = vec![A_LOG_TENSOR, DT_BIAS_TENSOR]
            .into_iter()
            .map(|name| serde_json::json!({"name": name}))
            .collect::<Vec<_>>();
        let manifest = serde_json::json!({
            "tensors": tensors,
            "passthrough_tensors": passthrough,
        });
        assert!(validate_manifest_exact_name_counts(&manifest).is_ok());
        tensors.push(serde_json::json!({"name": QKV_TENSOR}));
        let duplicate = serde_json::json!({
            "tensors": tensors,
            "passthrough_tensors": [
                {"name": A_LOG_TENSOR},
                {"name": DT_BIAS_TENSOR}
            ],
        });
        assert!(validate_manifest_exact_name_counts(&duplicate).is_err());
    }

    #[test]
    fn rollback_guard_removes_published_files_on_error() {
        let root = temp_path("rollback");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let qkv = root.join("qkv.f32le");
        let report = root.join("report.json");
        fs::write(&qkv, b"qkv").unwrap();
        fs::write(&report, b"report").unwrap();
        {
            let mut guard = RollbackGuard::new();
            guard.register_published(qkv.clone()).unwrap();
            guard.register_published(report.clone()).unwrap();
        }
        assert!(!qkv.exists());
        assert!(!report.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rollback_guard_preserves_foreign_replacement() {
        let root = temp_path("rollback-race");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let path = root.join("qkv.f32le");
        fs::write(&path, b"probe").unwrap();
        {
            let mut guard = RollbackGuard::new();
            guard.register_published(path.clone()).unwrap();
            fs::remove_file(&path).unwrap();
            fs::write(&path, b"foreign").unwrap();
        }
        assert_eq!(fs::read(&path).unwrap(), b"foreign");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn output_directory_symlink_is_rejected() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let root = temp_path("output-symlink");
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).unwrap();
            let target = root.join("target");
            let link = root.join("link");
            fs::create_dir_all(&target).unwrap();
            symlink(&target, &link).unwrap();
            assert!(prepare_output_directory(&link)
                .expect_err("output symlink must be rejected")
                .contains("symlink"));
            fs::remove_dir_all(root).unwrap();
        }
    }
}
