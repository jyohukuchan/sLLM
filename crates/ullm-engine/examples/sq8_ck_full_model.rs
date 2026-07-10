// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;
use ullm_engine::loader::{
    PassthroughPayloadVerification, read_named_passthrough_f32, read_named_passthrough_f32_rows,
    verify_named_passthrough_payload,
};
use ullm_engine::package::select_exact_passthrough_payload_bundle;
use ullm_engine::sq_canonical::{Sq8CanonicalArtifact, read_sq8_canonical_artifact};
use ullm_engine::sq_reference::{
    Sq8CorrectnessMetrics, compare_sq8_correctness, sq8_f32_le_sha256,
};
use ullm_engine::sq8_layer_oracle::{
    QWEN3_14B_HEAD_DIM, QWEN3_14B_HIDDEN_SIZE, QWEN3_14B_RMS_NORM_EPSILON, Sq8LayerNormWeights,
    Sq8LayerProjectionNames, run_qwen3_14b_sq8_layer_oracle,
};
use ullm_engine::sq8_layer_runtime::{
    QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV, Qwen3Sq8LayerNormValues, Sq8LayerExecutionReport,
    Sq8LayerProjectionExecution, qwen3_sq8_layer_tensor_names,
};
use ullm_engine::sq8_model_head_runtime::{
    QWEN3_14B_FINAL_NORM_TENSOR, QWEN3_14B_LM_HEAD_TENSOR,
    QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_VOCAB_SIZE,
    Qwen3Sq8ModelHeadRuntime, Sq8ModelHeadExecutionReport, Sq8ModelHeadRuntimeStatus,
    validate_qwen3_14b_sq8_r9700_device_info,
};
use ullm_engine::sq8_stack_runtime::{
    QWEN3_14B_SQ8_STACK_ACTIVATION_QUANTIZATIONS, QWEN3_14B_SQ8_STACK_LAYERS,
    QWEN3_14B_SQ8_STACK_PROJECTIONS, Qwen3Sq8StackRuntime, Sq8StackExecutionMode,
    Sq8StackExecutionReport, Sq8StackInputOrigin, Sq8StackRuntimeStatus,
};
use ullm_runtime_sys::{
    DeviceInfo, RuntimeContext, RuntimeStream, Sq8CkImplementation, device_count, device_info,
    sq8_ck_activation_buffer_bytes, sq8_ck_projection_buffer_bytes,
};

const SCHEMA_VERSION: &str = "ullm.sq8.full_model.v1";
const ORACLE_SCHEMA_VERSION: &str = "ullm.qwen3_full_model_oracle.v1";
const SEQUENCE_LEN: usize = 8;
const TOKEN_IDS: [usize; SEQUENCE_LEN] = [1, 2, 3, 4, 5, 6, 7, 8];
const POSITION_IDS: [usize; SEQUENCE_LEN] = [0, 1, 2, 3, 4, 5, 6, 7];
const EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";
const UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const CPU_HEAD_ROWS_PER_CHUNK: usize = 128;
const CPU_HEAD_MAX_WORKERS: usize = 8;
const WARMUPS: usize = 3;
const REPEATS: usize = 10;
const TOP_K: usize = 10;

const EXPECTED_ARTIFACT_CONTENT_SHA256: &str =
    "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147";
const EXPECTED_PACKAGE_MANIFEST_SHA256: &str =
    "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb";
const EXPECTED_ORACLE_METADATA_SHA256: &str =
    "5caafcd2c976482dd01e51b537593d8924d381a8a9ab076b2082325e22fea39e";
const EXPECTED_ORACLE_METADATA_BYTES: u64 = 49_967;
const EXPECTED_MODEL_REVISION: &str = "9a283b4a5efbc09ce247e0ae5b02b744739e525a";
const EXPECTED_CONFIG_SHA256: &str =
    "c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793";
const EXPECTED_INDEX_SHA256: &str =
    "6a9c8e17744118347080916d8f673b881941cf42989ee77266b14dc2062a7151";
const EXPECTED_FINAL_HIDDEN_SHA256: &str =
    "a6772963cee66d8429eaa7b4e72e2594345b1a6613a06a1bf67660b4f02aa9a7";
const EXPECTED_LOGITS_SHA256: &str =
    "24c93f3fbe0fc3d2a101c782f0e181be1206cabd56e900814a608d2a09fd268e";
const EXPECTED_VLLM_TOP_10: [usize; TOP_K] = [353, 3764, 25010, 220, 5572, 671, 3014, 374, 262, 16];
const EXPECTED_TOP_1: usize = 353;

#[derive(Debug)]
struct Options {
    artifact: PathBuf,
    package: PathBuf,
    oracle: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct OracleMetadata {
    schema_version: String,
    model: OracleModel,
    input: OracleInput,
    oracle: OracleOutputs,
}

#[derive(Debug, Deserialize)]
struct OracleModel {
    revision: OracleRevision,
}

#[derive(Debug, Deserialize)]
struct OracleRevision {
    revision: String,
    revision_consistent: bool,
}

#[derive(Debug, Deserialize)]
struct OracleInput {
    token_ids: Vec<usize>,
    position_ids: Vec<usize>,
    attention: String,
    bos_inserted: bool,
    chat_template_applied: bool,
}

#[derive(Debug, Deserialize)]
struct OracleOutputs {
    layers: Vec<OracleLayerDescriptor>,
    final_hidden: OracleTensorDescriptor,
    logits: OracleTensorDescriptor,
}

#[derive(Debug, Deserialize)]
struct OracleLayerDescriptor {
    layer_index: usize,
    #[serde(flatten)]
    tensor: OracleTensorDescriptor,
}

#[derive(Debug, Deserialize)]
struct OracleTensorDescriptor {
    file: String,
    bytes: u64,
    sha256: String,
    shape: Vec<usize>,
    storage_dtype: String,
}

#[derive(Debug)]
struct OracleFixture {
    metadata_sha256: String,
    revision: String,
    layer_outputs: Vec<Vec<f32>>,
    layer_sha256: Vec<String>,
    final_hidden_last_row: Vec<f32>,
    final_hidden_sha256: String,
    logits_last_row: Vec<f32>,
    logits_sha256: String,
    top_10: Vec<TopKEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct PayloadIdentity {
    tensor_name: String,
    dtype: String,
    shape: Vec<u64>,
    elements: u64,
    payload_bytes: u64,
    payload_sha256: String,
    verified_chunks: u64,
}

impl From<&PassthroughPayloadVerification> for PayloadIdentity {
    fn from(value: &PassthroughPayloadVerification) -> Self {
        Self {
            tensor_name: value.tensor_name.clone(),
            dtype: value.dtype.clone(),
            shape: value.shape.clone(),
            elements: value.elements,
            payload_bytes: value.payload_bytes,
            payload_sha256: value.payload_sha256.clone(),
            verified_chunks: value.verified_chunks,
        }
    }
}

#[derive(Debug, Serialize)]
struct FullModelResult {
    schema_version: &'static str,
    passed: bool,
    source: SourceIdentity,
    input: InputIdentity,
    payloads: PayloadEvidence,
    device: DeviceRecord,
    cpu_oracle: CpuOracleRecord,
    layer_boundaries: Vec<LayerBoundaryCheck>,
    final_head: FinalHeadCheck,
    execution: ExecutionContract,
    timing: TimingRecord,
    vram: VramContract,
}

#[derive(Debug, Serialize)]
struct SourceIdentity {
    artifact_content_sha256: String,
    artifact_config_sha256: String,
    artifact_index_sha256: String,
    package_manifest_sha256: String,
    vllm_oracle_metadata_sha256: String,
    model_revision: String,
    vllm_layer_output_sha256: Vec<String>,
    vllm_final_hidden_sha256: String,
    vllm_logits_sha256: String,
}

#[derive(Debug, Serialize)]
struct InputIdentity {
    sequence_len: usize,
    token_ids: [usize; SEQUENCE_LEN],
    position_ids: [usize; SEQUENCE_LEN],
    embedding_tensor: String,
    selected_embedding_f32_le_sha256: String,
}

#[derive(Debug, Serialize)]
struct PayloadEvidence {
    embedding: PayloadIdentity,
    layer_norms: Vec<PayloadIdentity>,
    final_norm: PayloadIdentity,
    lm_head: PayloadIdentity,
}

#[derive(Debug, Serialize)]
struct DeviceRecord {
    runtime_index: u32,
    backend_device_id: i32,
    backend: String,
    name: String,
    gcn_arch_name: String,
    compute_major: i32,
    compute_minor: i32,
    total_global_mem: u64,
}

#[derive(Debug, Serialize)]
struct CpuOracleRecord {
    elapsed_ms: f64,
    layer_output_f32_le_sha256: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LayerBoundaryCheck {
    layer_index: usize,
    optimized_health: TensorHealth,
    cpu_sq8_f32_le_sha256: String,
    vllm_f32_le_sha256: String,
    optimized_vs_cpu_sq8: TensorGate,
    optimized_vs_vllm: Sq8CorrectnessMetrics,
}

#[derive(Debug, Serialize)]
struct TensorGate {
    metrics: Sq8CorrectnessMetrics,
    max_relative_l2: f64,
    min_cosine: f64,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct TensorHealth {
    elements: usize,
    nonfinite: usize,
    minimum: f32,
    maximum: f32,
    max_abs: f32,
    f32_le_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct TopKEntry {
    token_id: usize,
    logit: f32,
}

#[derive(Debug, Serialize)]
struct FinalHeadCheck {
    resident_validation_layer39_matches_audit_bits: bool,
    device_final_hidden_health: TensorHealth,
    device_logits_health: TensorHealth,
    cpu_final_hidden_f32_le_sha256: String,
    cpu_logits_f32_le_sha256: String,
    device_vs_cpu_final_hidden: TensorGate,
    device_vs_cpu_logits: TensorGate,
    device_vs_vllm_final_hidden: TensorGate,
    device_vs_vllm_logits: TensorGate,
    device_top_10: Vec<TopKEntry>,
    cpu_top_10: Vec<TopKEntry>,
    vllm_top_10: Vec<TopKEntry>,
    device_top_1: usize,
    cpu_top_1: usize,
    vllm_top_1: usize,
    device_vllm_top_10_overlap: usize,
    top_1_contract_passed: bool,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ExecutionContract {
    sequence_len: usize,
    stack_invocations_per_timed_sample: usize,
    layers: usize,
    projections: usize,
    activation_quantizations: usize,
    layer_d2d_copies: usize,
    stack_execution_synchronizations: usize,
    head_d2d_copies: usize,
    head_rmsnorm_calls: usize,
    head_bf16_matvec_calls: usize,
    head_result_readbacks: usize,
    head_execution_synchronizations: usize,
    timed_path_fallback_used: bool,
    timed_path_host_staging_used: bool,
    layerwise_audit_is_non_timed: bool,
    layerwise_audit_host_staging_used: bool,
    layerwise_audit_readbacks: usize,
    fresh_input_uploads_for_validation_and_timing: usize,
    input_ready_state_checks: usize,
    output_ready_state_checks: usize,
    validated_stack_reports: usize,
    validated_head_reports: usize,
    timed_output_hash_stability_checks: usize,
    dispatch_implementation_counts: BTreeMap<String, usize>,
    required_hip_kernel_env: Vec<String>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct TimingRecord {
    warmups: usize,
    repeats: usize,
    input_upload_excluded: bool,
    inter_stage_host_validation_excluded: bool,
    stack_final_synchronization_included: bool,
    head_readback_decode_and_validation_included: bool,
    full_stack_and_head: TimingSummary,
    stack: TimingSummary,
    head: TimingSummary,
}

#[derive(Debug, Serialize)]
struct TimingSummary {
    samples_ms: Vec<f64>,
    p50_ms: f64,
    p95_ms: f64,
}

#[derive(Debug, Serialize)]
struct VramContract {
    device_total_global_mem: u64,
    artifact_weight_and_scale_bytes: u64,
    layer_norm_f32_bytes: u64,
    shared_stack_workspace_bytes: u64,
    resident_stack_hidden_bytes: u64,
    model_head_resident_bytes: u64,
    minimum_accounted_resident_bytes: u64,
    unaccounted_device_bytes: u64,
    excludes_allocator_and_backend_overhead: bool,
    fits_device: bool,
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    require_hip_kernel_guards()?;

    let artifact = read_sq8_canonical_artifact(&options.artifact)?;
    validate_artifact_identity(&artifact)?;
    let package_manifest_sha256 = sha256_regular_file(&options.package.join("manifest.json"))?;
    if package_manifest_sha256 != EXPECTED_PACKAGE_MANIFEST_SHA256 {
        return Err(format!(
            "thin-package manifest SHA-256 mismatch: expected={EXPECTED_PACKAGE_MANIFEST_SHA256} actual={package_manifest_sha256}"
        ));
    }

    let oracle = load_oracle_fixture(&options.oracle)?;
    let (embedding, embedding_identity) = load_verified_embeddings(&options.package)?;
    let selected_embedding_f32_le_sha256 = sq8_f32_le_sha256(&embedding)?;
    let (norms, norm_identities) = load_all_verified_layer_norms(&options.package)?;
    let (final_norm, final_norm_identity) = read_verified_bf16_tensor(
        &options.package,
        QWEN3_14B_FINAL_NORM_TENSOR,
        &[QWEN3_14B_HIDDEN_SIZE as u64],
    )?;
    let lm_head_verification = verify_named_passthrough_payload(
        &options.package,
        QWEN3_14B_LM_HEAD_TENSOR,
        "BF16",
        &[QWEN3_14B_VOCAB_SIZE as u64, QWEN3_14B_HIDDEN_SIZE as u64],
        UPLOAD_CHUNK_BYTES,
    )?;
    let lm_head_identity = PayloadIdentity::from(&lm_head_verification);

    let cpu_started = Instant::now();
    let cpu_boundaries = run_cpu_sq8_stack_oracle(&artifact, &norms, &embedding)?;
    let cpu_elapsed_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
    let cpu_boundary_hashes = cpu_boundaries
        .iter()
        .map(|values| sq8_f32_le_sha256(values))
        .collect::<Result<Vec<_>, _>>()?;

    let (runtime_index, device) = isolated_gfx1201_device()?;
    let vram = vram_contract(&artifact, &device)?;
    if !vram.fits_device {
        return Err(format!(
            "accounted SQ8 model residency {} exceeds device memory {}",
            vram.minimum_accounted_resident_bytes, vram.device_total_global_mem
        ));
    }

    let mut context = RuntimeContext::create(runtime_index)?;
    let mut stream = context.create_stream()?;
    let mut stack = Qwen3Sq8StackRuntime::load(
        &mut context,
        &mut stream,
        &artifact,
        SEQUENCE_LEN,
        norms.clone(),
        UPLOAD_CHUNK_BYTES,
    )?;
    let mut head = Qwen3Sq8ModelHeadRuntime::load(
        &mut context,
        &mut stream,
        &options.package,
        UPLOAD_CHUNK_BYTES,
    )?;

    let audit = stack.run_host_input_layerwise_audit(
        &embedding,
        ullm_engine::sq8_layer_runtime::Sq8LayerExecutionProfile::Rdna4W8a8BlockCk,
        &mut stream,
    )?;
    audit.report.validate_contract()?;
    validate_audit_report(&audit.report)?;
    let layer_boundaries = compare_layer_boundaries(&audit.layers, &cpu_boundaries, &oracle)?;

    stack.upload_host_input_synchronized(&embedding, &mut stream)?;
    require_stack_status(
        &stack,
        Sq8StackRuntimeStatus::InputReady,
        "validation upload",
    )?;
    let validation_report = stack.run_uploaded_optimized_synchronized(&mut stream)?;
    validation_report.validate_optimized_promotion()?;
    require_stack_status(
        &stack,
        Sq8StackRuntimeStatus::OutputReady,
        "validation stack",
    )?;
    let resident_validation_output = stack.read_output_synchronized(&mut stream)?;
    let audit_layer39 = &audit.layers[QWEN3_14B_SQ8_STACK_LAYERS - 1].output;
    let resident_validation_layer39_matches_audit_bits =
        f32_bits_equal(&resident_validation_output, audit_layer39);
    let device_head = head.run_synchronized(&stack, &mut stream)?;
    device_head.validate_contract()?;

    let last_hidden_row = last_row(&resident_validation_output, QWEN3_14B_HIDDEN_SIZE)?;
    let cpu_final_hidden = cpu_rmsnorm(last_hidden_row, &final_norm)?;
    let lm_head_bundle =
        select_exact_passthrough_payload_bundle(&options.package, QWEN3_14B_LM_HEAD_TENSOR)?;
    validate_lm_head_bundle(&lm_head_bundle, &lm_head_verification)?;
    let cpu_logits = cpu_bf16_lm_head_parallel(
        &lm_head_bundle.payload_file.absolute_path,
        &cpu_final_hidden,
    )?;
    let final_head = build_final_head_check(
        resident_validation_layer39_matches_audit_bits,
        &device_head.final_hidden,
        &device_head.logits,
        &cpu_final_hidden,
        &cpu_logits,
        &oracle,
    )?;

    let mut total_samples = Vec::with_capacity(REPEATS);
    let mut stack_samples = Vec::with_capacity(REPEATS);
    let mut head_samples = Vec::with_capacity(REPEATS);
    let mut input_ready_state_checks = 1_usize;
    let mut output_ready_state_checks = 1_usize;
    let mut validated_stack_reports = 1_usize;
    let mut validated_head_reports = 1_usize;
    let mut timed_output_hash_stability_checks = 0_usize;

    for _ in 0..WARMUPS {
        let (_, _, _, stack_report, head_report) = run_full_model_once(
            &mut stack,
            &mut head,
            &embedding,
            &validation_report,
            &device_head.report,
            &mut stream,
        )?;
        stack_report.validate_optimized_promotion()?;
        head_report.validate_contract()?;
        input_ready_state_checks += 1;
        output_ready_state_checks += 1;
        validated_stack_reports += 1;
        validated_head_reports += 1;
        timed_output_hash_stability_checks += 1;
    }
    for _ in 0..REPEATS {
        let (total_ms, stack_ms, head_ms, stack_report, head_report) = run_full_model_once(
            &mut stack,
            &mut head,
            &embedding,
            &validation_report,
            &device_head.report,
            &mut stream,
        )?;
        stack_report.validate_optimized_promotion()?;
        head_report.validate_contract()?;
        total_samples.push(total_ms);
        stack_samples.push(stack_ms);
        head_samples.push(head_ms);
        input_ready_state_checks += 1;
        output_ready_state_checks += 1;
        validated_stack_reports += 1;
        validated_head_reports += 1;
        timed_output_hash_stability_checks += 1;
    }

    let execution = build_execution_contract(
        &audit.report,
        &validation_report,
        &device_head.report,
        input_ready_state_checks,
        output_ready_state_checks,
        validated_stack_reports,
        validated_head_reports,
        timed_output_hash_stability_checks,
    )?;
    let timing = TimingRecord {
        warmups: WARMUPS,
        repeats: REPEATS,
        input_upload_excluded: true,
        inter_stage_host_validation_excluded: true,
        stack_final_synchronization_included: true,
        head_readback_decode_and_validation_included: true,
        full_stack_and_head: timing_summary(total_samples)?,
        stack: timing_summary(stack_samples)?,
        head: timing_summary(head_samples)?,
    };
    let boundaries_passed = layer_boundaries
        .iter()
        .all(|check| check.optimized_vs_cpu_sq8.passed && check.optimized_health.nonfinite == 0);
    let passed = boundaries_passed && final_head.passed && execution.passed && vram.fits_device;

    let result = FullModelResult {
        schema_version: SCHEMA_VERSION,
        passed,
        source: SourceIdentity {
            artifact_content_sha256: artifact.manifest().integrity.content_sha256.clone(),
            artifact_config_sha256: artifact.manifest().source.config_sha256.clone(),
            artifact_index_sha256: artifact
                .manifest()
                .source
                .index_sha256
                .clone()
                .ok_or_else(|| "SQ8 artifact has no source index SHA-256".to_string())?,
            package_manifest_sha256,
            vllm_oracle_metadata_sha256: oracle.metadata_sha256.clone(),
            model_revision: oracle.revision.clone(),
            vllm_layer_output_sha256: oracle.layer_sha256.clone(),
            vllm_final_hidden_sha256: oracle.final_hidden_sha256.clone(),
            vllm_logits_sha256: oracle.logits_sha256.clone(),
        },
        input: InputIdentity {
            sequence_len: SEQUENCE_LEN,
            token_ids: TOKEN_IDS,
            position_ids: POSITION_IDS,
            embedding_tensor: EMBEDDING_TENSOR.to_string(),
            selected_embedding_f32_le_sha256,
        },
        payloads: PayloadEvidence {
            embedding: embedding_identity,
            layer_norms: norm_identities,
            final_norm: final_norm_identity,
            lm_head: lm_head_identity,
        },
        device: DeviceRecord {
            runtime_index,
            backend_device_id: device.device_id,
            backend: device.backend,
            name: device.name,
            gcn_arch_name: device.gcn_arch_name,
            compute_major: device.compute_major,
            compute_minor: device.compute_minor,
            total_global_mem: device.total_global_mem,
        },
        cpu_oracle: CpuOracleRecord {
            elapsed_ms: cpu_elapsed_ms,
            layer_output_f32_le_sha256: cpu_boundary_hashes,
        },
        layer_boundaries,
        final_head,
        execution,
        timing,
        vram,
    };
    write_json_no_clobber(&options.output, &result)?;
    if !result.passed {
        return Err(format!(
            "SQ8 full-model gate failed; evidence was written to {}",
            options.output.display()
        ));
    }
    println!(
        "passed=true output={} full_p50_ms={:.6} stack_p50_ms={:.6} head_p50_ms={:.6} top1={}",
        options.output.display(),
        result.timing.full_stack_and_head.p50_ms,
        result.timing.stack.p50_ms,
        result.timing.head.p50_ms,
        result.final_head.device_top_1,
    );
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut args = std::env::args_os().skip(1);
    let artifact = PathBuf::from(args.next().ok_or_else(usage)?);
    let package = PathBuf::from(args.next().ok_or_else(usage)?);
    let oracle = PathBuf::from(args.next().ok_or_else(usage)?);
    let output = PathBuf::from(args.next().ok_or_else(usage)?);
    if args.next().is_some() {
        return Err(usage());
    }
    for (label, path) in [
        ("artifact", &artifact),
        ("package", &package),
        ("vLLM oracle", &oracle),
    ] {
        if !path.is_dir() {
            return Err(format!("{label} is not a directory: {}", path.display()));
        }
    }
    match std::fs::symlink_metadata(&output) {
        Ok(_) => return Err(format!("output already exists: {}", output.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(format!("failed to inspect {}: {err}", output.display())),
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("output has no parent: {}", output.display()))?;
    if !parent.is_dir() {
        return Err(format!(
            "output parent is not a directory: {}",
            parent.display()
        ));
    }
    Ok(Options {
        artifact,
        package,
        oracle,
        output,
    })
}

fn usage() -> String {
    "usage: sq8_ck_full_model ARTIFACT_DIR THIN_PACKAGE VLLM_ORACLE_DIR OUTPUT_JSON".to_string()
}

fn validate_artifact_identity(artifact: &Sq8CanonicalArtifact) -> Result<(), String> {
    let manifest = artifact.manifest();
    if manifest.integrity.content_sha256 != EXPECTED_ARTIFACT_CONTENT_SHA256 {
        return Err(format!(
            "SQ8 artifact content SHA-256 mismatch: expected={EXPECTED_ARTIFACT_CONTENT_SHA256} actual={}",
            manifest.integrity.content_sha256
        ));
    }
    if manifest.source.config_sha256 != EXPECTED_CONFIG_SHA256
        || manifest.source.index_sha256.as_deref() != Some(EXPECTED_INDEX_SHA256)
    {
        return Err("SQ8 artifact does not identify the fixed Qwen3-14B-FP8 checkpoint".into());
    }
    if manifest.coverage.selected_pair_count != QWEN3_14B_SQ8_STACK_PROJECTIONS as u64
        || manifest.coverage.unpaired_tensor_count != 0
        || manifest.quantized_tensors.len() != QWEN3_14B_SQ8_STACK_PROJECTIONS
    {
        return Err(format!(
            "SQ8 artifact coverage mismatch: selected={} manifest_pairs={} unpaired={}",
            manifest.coverage.selected_pair_count,
            manifest.quantized_tensors.len(),
            manifest.coverage.unpaired_tensor_count
        ));
    }
    Ok(())
}

fn load_oracle_fixture(root: &Path) -> Result<OracleFixture, String> {
    let metadata_path = root.join("metadata.json");
    let metadata_bytes = read_regular_file_exact(&metadata_path, EXPECTED_ORACLE_METADATA_BYTES)?;
    let metadata_sha256 = sha256_bytes(&metadata_bytes);
    if metadata_sha256 != EXPECTED_ORACLE_METADATA_SHA256 {
        return Err(format!(
            "vLLM oracle metadata SHA-256 mismatch: expected={EXPECTED_ORACLE_METADATA_SHA256} actual={metadata_sha256}"
        ));
    }
    let metadata: OracleMetadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|err| format!("failed to parse {}: {err}", metadata_path.display()))?;
    if metadata.schema_version != ORACLE_SCHEMA_VERSION
        || metadata.model.revision.revision != EXPECTED_MODEL_REVISION
        || !metadata.model.revision.revision_consistent
    {
        return Err("vLLM oracle schema or model revision mismatch".into());
    }
    if metadata.input.token_ids != TOKEN_IDS
        || metadata.input.position_ids != POSITION_IDS
        || metadata.input.attention != "causal"
        || metadata.input.bos_inserted
        || metadata.input.chat_template_applied
    {
        return Err("vLLM oracle input semantics do not match fixed M=8 tokens".into());
    }
    if metadata.oracle.layers.len() != QWEN3_14B_SQ8_STACK_LAYERS {
        return Err(format!(
            "vLLM oracle layer count mismatch: expected={} actual={}",
            QWEN3_14B_SQ8_STACK_LAYERS,
            metadata.oracle.layers.len()
        ));
    }

    let mut layer_outputs = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
    let mut layer_sha256 = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
    for (layer_index, descriptor) in metadata.oracle.layers.iter().enumerate() {
        let expected_file = format!("layers/layer-{layer_index:02}-output.f32");
        if descriptor.layer_index != layer_index
            || descriptor.tensor.file != expected_file
            || descriptor.tensor.shape != [SEQUENCE_LEN, QWEN3_14B_HIDDEN_SIZE]
        {
            return Err(format!(
                "vLLM oracle layer {layer_index} descriptor mismatch"
            ));
        }
        let values = read_oracle_tensor(root, &descriptor.tensor)?;
        layer_sha256.push(descriptor.tensor.sha256.clone());
        layer_outputs.push(values);
    }
    validate_oracle_descriptor(
        &metadata.oracle.final_hidden,
        "final-hidden.f32",
        &[SEQUENCE_LEN, QWEN3_14B_HIDDEN_SIZE],
        EXPECTED_FINAL_HIDDEN_SHA256,
    )?;
    let final_hidden = read_oracle_tensor(root, &metadata.oracle.final_hidden)?;
    let final_hidden_last_row = last_row(&final_hidden, QWEN3_14B_HIDDEN_SIZE)?.to_vec();
    drop(final_hidden);

    validate_oracle_descriptor(
        &metadata.oracle.logits,
        "logits.f32",
        &[SEQUENCE_LEN, QWEN3_14B_VOCAB_SIZE],
        EXPECTED_LOGITS_SHA256,
    )?;
    let logits = read_oracle_tensor(root, &metadata.oracle.logits)?;
    let logits_last_row = last_row(&logits, QWEN3_14B_VOCAB_SIZE)?.to_vec();
    drop(logits);
    let top_10 = top_k(&logits_last_row, TOP_K)?;
    let ids = top_10
        .iter()
        .map(|entry| entry.token_id)
        .collect::<Vec<_>>();
    if ids != EXPECTED_VLLM_TOP_10 {
        return Err(format!(
            "vLLM oracle top-10 mismatch: expected={EXPECTED_VLLM_TOP_10:?} actual={ids:?}"
        ));
    }
    Ok(OracleFixture {
        metadata_sha256,
        revision: metadata.model.revision.revision,
        layer_outputs,
        layer_sha256,
        final_hidden_last_row,
        final_hidden_sha256: metadata.oracle.final_hidden.sha256,
        logits_last_row,
        logits_sha256: metadata.oracle.logits.sha256,
        top_10,
    })
}

fn validate_oracle_descriptor(
    descriptor: &OracleTensorDescriptor,
    expected_file: &str,
    expected_shape: &[usize],
    expected_sha256: &str,
) -> Result<(), String> {
    if descriptor.file != expected_file
        || descriptor.shape != expected_shape
        || descriptor.sha256 != expected_sha256
        || descriptor.storage_dtype != "float32_le"
    {
        return Err(format!(
            "vLLM oracle descriptor mismatch for {expected_file}"
        ));
    }
    let expected_bytes = expected_shape
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| format!("vLLM oracle descriptor {expected_file} byte size overflows"))?;
    if descriptor.bytes != expected_bytes as u64 {
        return Err(format!(
            "vLLM oracle descriptor {expected_file} byte mismatch: expected={expected_bytes} actual={}",
            descriptor.bytes
        ));
    }
    Ok(())
}

fn read_oracle_tensor(
    root: &Path,
    descriptor: &OracleTensorDescriptor,
) -> Result<Vec<f32>, String> {
    if descriptor.storage_dtype != "float32_le" {
        return Err(format!(
            "vLLM oracle {} storage dtype is not float32_le",
            descriptor.file
        ));
    }
    let path = safe_relative_regular_file(root, &descriptor.file)?;
    let bytes = read_regular_file_exact(&path, descriptor.bytes)?;
    let actual_sha256 = sha256_bytes(&bytes);
    if actual_sha256 != descriptor.sha256 {
        return Err(format!(
            "vLLM oracle {} checksum mismatch: metadata={} file={actual_sha256}",
            descriptor.file, descriptor.sha256
        ));
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "vLLM oracle {} is not F32 aligned",
            descriptor.file
        ));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "vLLM oracle {} contains a non-finite value",
            descriptor.file
        ));
    }
    Ok(values)
}

fn load_verified_embeddings(package: &Path) -> Result<(Vec<f32>, PayloadIdentity), String> {
    let verification = verify_named_passthrough_payload(
        package,
        EMBEDDING_TENSOR,
        "BF16",
        &[QWEN3_14B_VOCAB_SIZE as u64, QWEN3_14B_HIDDEN_SIZE as u64],
        UPLOAD_CHUNK_BYTES,
    )?;
    let rows = read_named_passthrough_f32_rows(package, EMBEDDING_TENSOR, &TOKEN_IDS)?;
    if rows.dtype != "BF16"
        || rows.shape != [QWEN3_14B_VOCAB_SIZE as u64, QWEN3_14B_HIDDEN_SIZE as u64]
        || rows.row_indices != TOKEN_IDS
        || rows.columns != QWEN3_14B_HIDDEN_SIZE
        || rows.values.len() != SEQUENCE_LEN * QWEN3_14B_HIDDEN_SIZE
    {
        return Err("selected embedding rows do not match the fixed M=8 input contract".into());
    }
    validate_finite(&rows.values, "selected embeddings")?;
    Ok((rows.values, PayloadIdentity::from(&verification)))
}

fn load_all_verified_layer_norms(
    package: &Path,
) -> Result<(Vec<Qwen3Sq8LayerNormValues>, Vec<PayloadIdentity>), String> {
    let mut norms = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
    let mut identities = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS * 4);
    for layer_index in 0..QWEN3_14B_SQ8_STACK_LAYERS {
        let prefix = format!("model.layers.{layer_index}");
        let (input, input_identity) = read_verified_bf16_tensor(
            package,
            &format!("{prefix}.input_layernorm.weight"),
            &[QWEN3_14B_HIDDEN_SIZE as u64],
        )?;
        let (post_attention, post_identity) = read_verified_bf16_tensor(
            package,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[QWEN3_14B_HIDDEN_SIZE as u64],
        )?;
        let (q, q_identity) = read_verified_bf16_tensor(
            package,
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[QWEN3_14B_HEAD_DIM as u64],
        )?;
        let (k, k_identity) = read_verified_bf16_tensor(
            package,
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[QWEN3_14B_HEAD_DIM as u64],
        )?;
        identities.extend([input_identity, post_identity, q_identity, k_identity]);
        norms.push(Qwen3Sq8LayerNormValues {
            input,
            post_attention,
            q,
            k,
        });
    }
    if norms.len() != QWEN3_14B_SQ8_STACK_LAYERS
        || identities.len() != QWEN3_14B_SQ8_STACK_LAYERS * 4
    {
        return Err("verified SQ8 layer norm count mismatch".into());
    }
    Ok((norms, identities))
}

fn read_verified_bf16_tensor(
    package: &Path,
    tensor_name: &str,
    expected_shape: &[u64],
) -> Result<(Vec<f32>, PayloadIdentity), String> {
    let verification = verify_named_passthrough_payload(
        package,
        tensor_name,
        "BF16",
        expected_shape,
        UPLOAD_CHUNK_BYTES,
    )?;
    let data = read_named_passthrough_f32(package, tensor_name, UPLOAD_CHUNK_BYTES)?;
    if data.dtype != "BF16" || data.shape != expected_shape {
        return Err(format!(
            "verified BF16 tensor {tensor_name} changed before decode"
        ));
    }
    validate_finite(&data.values, tensor_name)?;
    let decoded_payload_sha256 = bf16_values_sha256(&data.values);
    if decoded_payload_sha256 != verification.payload_sha256 {
        return Err(format!(
            "verified BF16 tensor {tensor_name} checksum changed before decode: verified={} decoded={decoded_payload_sha256}",
            verification.payload_sha256
        ));
    }
    Ok((data.values, PayloadIdentity::from(&verification)))
}

fn run_cpu_sq8_stack_oracle(
    artifact: &Sq8CanonicalArtifact,
    norms: &[Qwen3Sq8LayerNormValues],
    input: &[f32],
) -> Result<Vec<Vec<f32>>, String> {
    if norms.len() != QWEN3_14B_SQ8_STACK_LAYERS {
        return Err("CPU SQ8 stack oracle requires exactly 40 norm sets".into());
    }
    let mut hidden = input.to_vec();
    let mut boundaries = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
    for (layer_index, norm) in norms.iter().enumerate() {
        let names = qwen3_sq8_layer_tensor_names(layer_index);
        let output = run_qwen3_14b_sq8_layer_oracle(
            artifact,
            projection_names(&names),
            norm_refs(norm),
            &hidden,
            SEQUENCE_LEN,
        )?;
        hidden = output.output;
        validate_finite(&hidden, &format!("CPU SQ8 layer {layer_index}"))?;
        boundaries.push(hidden.clone());
        if layer_index == 0 || (layer_index + 1).is_multiple_of(5) {
            eprintln!(
                "CPU SQ8 oracle completed layer {}/{}",
                layer_index + 1,
                QWEN3_14B_SQ8_STACK_LAYERS
            );
        }
    }
    Ok(boundaries)
}

fn projection_names<'a>(names: &'a [String; 7]) -> Sq8LayerProjectionNames<'a> {
    Sq8LayerProjectionNames {
        q_proj: &names[0],
        k_proj: &names[1],
        v_proj: &names[2],
        o_proj: &names[3],
        gate_proj: &names[4],
        up_proj: &names[5],
        down_proj: &names[6],
    }
}

fn norm_refs(norms: &Qwen3Sq8LayerNormValues) -> Sq8LayerNormWeights<'_> {
    Sq8LayerNormWeights {
        input: &norms.input,
        post_attention: &norms.post_attention,
        q: &norms.q,
        k: &norms.k,
    }
}

fn isolated_gfx1201_device() -> Result<(u32, DeviceInfo), String> {
    let mut devices = Vec::new();
    for index in 1..device_count()? {
        let info = device_info(index)
            .map_err(|err| format!("failed to inspect runtime device {index}: {err}"))?;
        if info.backend == "hip" {
            devices.push((index, info));
        }
    }
    if devices.len() != 1 {
        return Err(format!(
            "SQ8 full-model validation requires exactly one visible runtime HIP device, found {}",
            devices.len()
        ));
    }
    let (runtime_index, device) = devices.into_iter().next().expect("one HIP device");
    validate_qwen3_14b_sq8_r9700_device_info(&device)?;
    Ok((runtime_index, device))
}

fn compare_layer_boundaries(
    optimized: &[ullm_engine::sq8_stack_runtime::Sq8StackLayerAudit; QWEN3_14B_SQ8_STACK_LAYERS],
    cpu: &[Vec<f32>],
    vllm: &OracleFixture,
) -> Result<Vec<LayerBoundaryCheck>, String> {
    if cpu.len() != QWEN3_14B_SQ8_STACK_LAYERS
        || vllm.layer_outputs.len() != QWEN3_14B_SQ8_STACK_LAYERS
    {
        return Err("SQ8 boundary source count mismatch".into());
    }
    let mut checks = Vec::with_capacity(QWEN3_14B_SQ8_STACK_LAYERS);
    for layer_index in 0..QWEN3_14B_SQ8_STACK_LAYERS {
        if optimized[layer_index].layer_index != layer_index {
            return Err(format!(
                "optimized audit layer order mismatch at {layer_index}"
            ));
        }
        let max_relative_l2 = if layer_index == QWEN3_14B_SQ8_STACK_LAYERS - 1 {
            0.08
        } else {
            0.10
        };
        let min_cosine = if layer_index == QWEN3_14B_SQ8_STACK_LAYERS - 1 {
            0.997
        } else {
            0.995
        };
        checks.push(LayerBoundaryCheck {
            layer_index,
            optimized_health: tensor_health(&optimized[layer_index].output)?,
            cpu_sq8_f32_le_sha256: sq8_f32_le_sha256(&cpu[layer_index])?,
            vllm_f32_le_sha256: vllm.layer_sha256[layer_index].clone(),
            optimized_vs_cpu_sq8: tensor_gate(
                &cpu[layer_index],
                &optimized[layer_index].output,
                max_relative_l2,
                min_cosine,
            )?,
            optimized_vs_vllm: compare_sq8_correctness(
                &vllm.layer_outputs[layer_index],
                &optimized[layer_index].output,
            )?,
        });
    }
    Ok(checks)
}

fn run_full_model_once(
    stack: &mut Qwen3Sq8StackRuntime,
    head: &mut Qwen3Sq8ModelHeadRuntime,
    input: &[f32],
    expected_report: &Sq8StackExecutionReport,
    expected_head_report: &Sq8ModelHeadExecutionReport,
    stream: &mut RuntimeStream,
) -> Result<
    (
        f64,
        f64,
        f64,
        Sq8StackExecutionReport,
        Sq8ModelHeadExecutionReport,
    ),
    String,
> {
    stack.upload_host_input_synchronized(input, stream)?;
    require_stack_status(stack, Sq8StackRuntimeStatus::InputReady, "timed upload")?;
    let total_started = Instant::now();
    let stack_started = Instant::now();
    let stack_report = stack.run_uploaded_optimized_synchronized(stream)?;
    let stack_ms = stack_started.elapsed().as_secs_f64() * 1000.0;
    let head_started = Instant::now();
    let head_result = head.run_synchronized(stack, stream)?;
    let head_ms = head_started.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_started.elapsed().as_secs_f64() * 1000.0;
    require_stack_status(stack, Sq8StackRuntimeStatus::OutputReady, "timed stack")?;
    if stack_report.layer_reports != expected_report.layer_reports {
        return Err("timed SQ8 stack dispatch report changed from validation run".into());
    }
    if head.status() != Sq8ModelHeadRuntimeStatus::OutputReady {
        return Err("timed SQ8 model head did not reach OutputReady".into());
    }
    if head_result.report != *expected_head_report {
        return Err(
            "timed SQ8 model-head report or output hashes changed from validation run".into(),
        );
    }
    Ok((
        total_ms,
        stack_ms,
        head_ms,
        stack_report,
        head_result.report,
    ))
}

fn build_final_head_check(
    resident_validation_layer39_matches_audit_bits: bool,
    device_final_hidden: &[f32],
    device_logits: &[f32],
    cpu_final_hidden: &[f32],
    cpu_logits: &[f32],
    oracle: &OracleFixture,
) -> Result<FinalHeadCheck, String> {
    let device_vs_cpu_final_hidden =
        tensor_gate(cpu_final_hidden, device_final_hidden, 2.0e-3, 0.999_999)?;
    let device_vs_cpu_logits = tensor_gate(cpu_logits, device_logits, 2.0e-3, 0.999_999)?;
    let device_vs_vllm_final_hidden = tensor_gate(
        &oracle.final_hidden_last_row,
        device_final_hidden,
        0.15,
        0.99,
    )?;
    let device_vs_vllm_logits = tensor_gate(&oracle.logits_last_row, device_logits, 0.15, 0.99)?;
    let device_top_10 = top_k(device_logits, TOP_K)?;
    let cpu_top_10 = top_k(cpu_logits, TOP_K)?;
    let device_top_1 = device_top_10[0].token_id;
    let cpu_top_1 = cpu_top_10[0].token_id;
    let vllm_top_1 = oracle.top_10[0].token_id;
    let device_vllm_top_10_overlap = top_k_overlap(&device_top_10, &oracle.top_10);
    let top_1_contract_passed =
        device_top_1 == cpu_top_1 && device_top_1 == EXPECTED_TOP_1 && vllm_top_1 == EXPECTED_TOP_1;
    let passed = resident_validation_layer39_matches_audit_bits
        && device_vs_cpu_final_hidden.passed
        && device_vs_cpu_logits.passed
        && device_vs_vllm_final_hidden.passed
        && device_vs_vllm_logits.passed
        && top_1_contract_passed
        && device_vllm_top_10_overlap >= 5;
    Ok(FinalHeadCheck {
        resident_validation_layer39_matches_audit_bits,
        device_final_hidden_health: tensor_health(device_final_hidden)?,
        device_logits_health: tensor_health(device_logits)?,
        cpu_final_hidden_f32_le_sha256: sq8_f32_le_sha256(cpu_final_hidden)?,
        cpu_logits_f32_le_sha256: sq8_f32_le_sha256(cpu_logits)?,
        device_vs_cpu_final_hidden,
        device_vs_cpu_logits,
        device_vs_vllm_final_hidden,
        device_vs_vllm_logits,
        device_top_10,
        cpu_top_10,
        vllm_top_10: oracle.top_10.clone(),
        device_top_1,
        cpu_top_1,
        vllm_top_1,
        device_vllm_top_10_overlap,
        top_1_contract_passed,
        passed,
    })
}

fn build_execution_contract(
    audit: &Sq8StackExecutionReport,
    stack: &Sq8StackExecutionReport,
    head: &Sq8ModelHeadExecutionReport,
    input_ready_state_checks: usize,
    output_ready_state_checks: usize,
    validated_stack_reports: usize,
    validated_head_reports: usize,
    timed_output_hash_stability_checks: usize,
) -> Result<ExecutionContract, String> {
    audit.validate_contract()?;
    stack.validate_optimized_promotion()?;
    head.validate_contract()?;
    let dispatch_implementation_counts = dispatch_counts(&stack.layer_reports)?;
    let guards = required_guard_names();
    let expected_runs = 1 + WARMUPS + REPEATS;
    let passed = stack.sequence_len == SEQUENCE_LEN
        && stack.mode == Sq8StackExecutionMode::SynchronizedResident
        && stack.input_origin == Sq8StackInputOrigin::PreviouslyUploadedResident
        && stack.projection_calls == QWEN3_14B_SQ8_STACK_PROJECTIONS
        && stack.activation_quantizations == QWEN3_14B_SQ8_STACK_ACTIVATION_QUANTIZATIONS
        && stack.d2d_copy_count == QWEN3_14B_SQ8_STACK_LAYERS
        && stack.execution_synchronization_count == 1
        && !stack.fallback_used
        && !stack.host_staging_used
        && audit.mode == Sq8StackExecutionMode::LayerwiseAuditNonTimed
        && audit.host_staging_used
        && audit.host_readback_count == QWEN3_14B_SQ8_STACK_LAYERS
        && head.d2d_copy_count == 1
        && head.rmsnorm_call_count == 1
        && head.bf16_matvec_call_count == 1
        && head.result_readback_count == 2
        && head.execution_synchronization_count == 1
        && !head.fallback_used
        && !head.host_staging_used
        && input_ready_state_checks == expected_runs
        && output_ready_state_checks == expected_runs
        && validated_stack_reports == expected_runs
        && validated_head_reports == expected_runs
        && timed_output_hash_stability_checks == WARMUPS + REPEATS
        && dispatch_implementation_counts.values().sum::<usize>()
            == QWEN3_14B_SQ8_STACK_PROJECTIONS;
    Ok(ExecutionContract {
        sequence_len: SEQUENCE_LEN,
        stack_invocations_per_timed_sample: 1,
        layers: QWEN3_14B_SQ8_STACK_LAYERS,
        projections: stack.projection_calls,
        activation_quantizations: stack.activation_quantizations,
        layer_d2d_copies: stack.d2d_copy_count,
        stack_execution_synchronizations: stack.execution_synchronization_count,
        head_d2d_copies: head.d2d_copy_count,
        head_rmsnorm_calls: head.rmsnorm_call_count,
        head_bf16_matvec_calls: head.bf16_matvec_call_count,
        head_result_readbacks: head.result_readback_count,
        head_execution_synchronizations: head.execution_synchronization_count,
        timed_path_fallback_used: stack.fallback_used || head.fallback_used,
        timed_path_host_staging_used: stack.host_staging_used || head.host_staging_used,
        layerwise_audit_is_non_timed: audit.mode == Sq8StackExecutionMode::LayerwiseAuditNonTimed,
        layerwise_audit_host_staging_used: audit.host_staging_used,
        layerwise_audit_readbacks: audit.host_readback_count,
        fresh_input_uploads_for_validation_and_timing: expected_runs,
        input_ready_state_checks,
        output_ready_state_checks,
        validated_stack_reports,
        validated_head_reports,
        timed_output_hash_stability_checks,
        dispatch_implementation_counts,
        required_hip_kernel_env: guards,
        passed,
    })
}

fn validate_audit_report(report: &Sq8StackExecutionReport) -> Result<(), String> {
    if report.mode != Sq8StackExecutionMode::LayerwiseAuditNonTimed
        || report.input_origin != Sq8StackInputOrigin::SynchronizedHostUploadBeforeExecution
        || !report.host_staging_used
        || report.host_readback_count != QWEN3_14B_SQ8_STACK_LAYERS
        || report.execution_synchronization_count != QWEN3_14B_SQ8_STACK_LAYERS
    {
        return Err("SQ8 layerwise audit did not remain on the explicit non-timed path".into());
    }
    if report.fallback_used || !report.all_ck() {
        return Err("SQ8 layerwise audit used a fallback or non-CK projection".into());
    }
    Ok(())
}

fn dispatch_counts(
    reports: &[Sq8LayerExecutionReport; QWEN3_14B_SQ8_STACK_LAYERS],
) -> Result<BTreeMap<String, usize>, String> {
    let mut counts = BTreeMap::new();
    for report in reports {
        for execution in [
            report.q,
            report.k,
            report.v,
            report.o,
            report.gate,
            report.up,
            report.down,
        ] {
            let implementation = match execution {
                Sq8LayerProjectionExecution::Ck(value) => value,
                Sq8LayerProjectionExecution::Reference(_) => {
                    return Err("optimized SQ8 stack report contains a reference projection".into());
                }
            };
            *counts
                .entry(ck_implementation_name(implementation).to_string())
                .or_insert(0) += 1;
        }
    }
    Ok(counts)
}

fn ck_implementation_name(value: Sq8CkImplementation) -> &'static str {
    match value {
        Sq8CkImplementation::MemV1DefaultTile16x128x128 => "mem_v1_default_tile_16x128x128",
        Sq8CkImplementation::MemV1KPaddingTile16x128x256 => "mem_v1_kpadding_tile_16x128x256",
        Sq8CkImplementation::MemV1DefaultTile16x256x128 => "mem_v1_default_tile_16x256x128",
        Sq8CkImplementation::MemV1DefaultTile16x128x256 => "mem_v1_default_tile_16x128x256",
    }
}

fn require_stack_status(
    stack: &Qwen3Sq8StackRuntime,
    expected: Sq8StackRuntimeStatus,
    label: &str,
) -> Result<(), String> {
    let actual = stack.status();
    if actual != expected {
        return Err(format!(
            "SQ8 stack status mismatch after {label}: expected={expected:?} actual={actual:?}"
        ));
    }
    Ok(())
}

fn cpu_rmsnorm(input: &[f32], weight: &[f32]) -> Result<Vec<f32>, String> {
    if input.len() != QWEN3_14B_HIDDEN_SIZE || weight.len() != QWEN3_14B_HIDDEN_SIZE {
        return Err(format!(
            "CPU final RMSNorm shape mismatch: input={} weight={} expected={QWEN3_14B_HIDDEN_SIZE}",
            input.len(),
            weight.len()
        ));
    }
    validate_finite(input, "CPU final RMSNorm input")?;
    validate_finite(weight, "CPU final RMSNorm weight")?;
    let mean_square = input
        .iter()
        .copied()
        .map(f64::from)
        .map(|value| value * value)
        .sum::<f64>()
        / QWEN3_14B_HIDDEN_SIZE as f64;
    let inverse_rms = 1.0_f64 / (mean_square + f64::from(QWEN3_14B_RMS_NORM_EPSILON)).sqrt();
    let output = input
        .iter()
        .zip(weight)
        .map(|(value, weight)| (f64::from(*value) * inverse_rms * f64::from(*weight)) as f32)
        .collect::<Vec<_>>();
    validate_finite(&output, "CPU final RMSNorm output")?;
    Ok(output)
}

fn validate_lm_head_bundle(
    bundle: &ullm_engine::package::PassthroughPayloadBundle,
    verification: &PassthroughPayloadVerification,
) -> Result<(), String> {
    if bundle.tensor_name != QWEN3_14B_LM_HEAD_TENSOR
        || bundle.dtype.as_deref() != Some("BF16")
        || bundle.shape != [QWEN3_14B_VOCAB_SIZE as u64, QWEN3_14B_HIDDEN_SIZE as u64]
        || bundle.elements != verification.elements
        || bundle.payload_bytes != verification.payload_bytes
        || bundle.payload_sha256.as_deref() != Some(&verification.payload_sha256)
    {
        return Err("CPU BF16 LM head bundle does not match verified payload identity".into());
    }
    Ok(())
}

fn cpu_bf16_lm_head_parallel(path: &Path, hidden: &[f32]) -> Result<Vec<f32>, String> {
    if hidden.len() != QWEN3_14B_HIDDEN_SIZE {
        return Err(format!(
            "CPU BF16 LM head hidden size mismatch: expected={QWEN3_14B_HIDDEN_SIZE} actual={}",
            hidden.len()
        ));
    }
    validate_finite(hidden, "CPU BF16 LM head hidden")?;
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(CPU_HEAD_MAX_WORKERS)
        .min(QWEN3_14B_VOCAB_SIZE)
        .max(1);
    let rows_per_worker = QWEN3_14B_VOCAB_SIZE.div_ceil(workers);
    let mut logits = vec![0.0_f32; QWEN3_14B_VOCAB_SIZE];
    std::thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::new();
        for (worker_index, output) in logits.chunks_mut(rows_per_worker).enumerate() {
            let start_row = worker_index * rows_per_worker;
            let path = path.to_path_buf();
            handles.push(
                scope.spawn(move || cpu_bf16_lm_head_partition(&path, hidden, start_row, output)),
            );
        }
        for handle in handles {
            match handle.join() {
                Ok(result) => result?,
                Err(_) => return Err("CPU BF16 LM head worker panicked".into()),
            }
        }
        Ok(())
    })?;
    validate_finite(&logits, "CPU BF16 LM head logits")?;
    Ok(logits)
}

fn cpu_bf16_lm_head_partition(
    path: &Path,
    hidden: &[f32],
    start_row: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let row_bytes = QWEN3_14B_HIDDEN_SIZE
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| "CPU BF16 LM head row byte size overflows".to_string())?;
    let mut local_row = 0_usize;
    while local_row < output.len() {
        let chunk_rows = (output.len() - local_row).min(CPU_HEAD_ROWS_PER_CHUNK);
        let chunk_bytes = chunk_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| "CPU BF16 LM head chunk size overflows".to_string())?;
        let mut bytes = vec![0_u8; chunk_bytes];
        let global_row = start_row
            .checked_add(local_row)
            .ok_or_else(|| "CPU BF16 LM head row offset overflows".to_string())?;
        let byte_offset = global_row
            .checked_mul(row_bytes)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| "CPU BF16 LM head byte offset overflows".to_string())?;
        file.read_exact_at(&mut bytes, byte_offset).map_err(|err| {
            format!(
                "failed to read CPU BF16 LM head rows {global_row}..{}: {err}",
                global_row + chunk_rows
            )
        })?;
        for chunk_row in 0..chunk_rows {
            let row = &bytes[chunk_row * row_bytes..(chunk_row + 1) * row_bytes];
            let mut sum = 0.0_f64;
            for (column, encoded) in row.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([encoded[0], encoded[1]]);
                let weight = f32::from_bits(u32::from(bits) << 16);
                sum += f64::from(weight) * f64::from(hidden[column]);
            }
            let value = sum as f32;
            if !value.is_finite() {
                return Err(format!(
                    "CPU BF16 LM head produced non-finite logit at row {}",
                    global_row + chunk_row
                ));
            }
            output[local_row + chunk_row] = value;
        }
        local_row += chunk_rows;
    }
    Ok(())
}

fn tensor_gate(
    reference: &[f32],
    actual: &[f32],
    max_relative_l2: f64,
    min_cosine: f64,
) -> Result<TensorGate, String> {
    let metrics = compare_sq8_correctness(reference, actual)?;
    let passed = metrics.nonfinite_count == 0
        && metrics.relative_l2.is_finite()
        && metrics.relative_l2 <= max_relative_l2
        && metrics.cosine_similarity.is_finite()
        && metrics.cosine_similarity >= min_cosine;
    Ok(TensorGate {
        metrics,
        max_relative_l2,
        min_cosine,
        passed,
    })
}

fn tensor_health(values: &[f32]) -> Result<TensorHealth, String> {
    if values.is_empty() {
        return Err("tensor health requires non-empty values".into());
    }
    let nonfinite = values.iter().filter(|value| !value.is_finite()).count();
    let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let f32_le_sha256 = sq8_f32_le_sha256(values)?;
    Ok(TensorHealth {
        elements: values.len(),
        nonfinite,
        minimum,
        maximum,
        max_abs,
        f32_le_sha256,
    })
}

fn top_k(values: &[f32], k: usize) -> Result<Vec<TopKEntry>, String> {
    if k == 0 || values.len() < k {
        return Err(format!(
            "top-k requires 0 < k <= values, got k={k} values={}",
            values.len()
        ));
    }
    validate_finite(values, "top-k input")?;
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|lhs, rhs| {
        values[*rhs]
            .total_cmp(&values[*lhs])
            .then_with(|| lhs.cmp(rhs))
    });
    Ok(indices
        .into_iter()
        .take(k)
        .map(|token_id| TopKEntry {
            token_id,
            logit: values[token_id],
        })
        .collect())
}

fn top_k_overlap(lhs: &[TopKEntry], rhs: &[TopKEntry]) -> usize {
    let rhs_ids = rhs
        .iter()
        .map(|entry| entry.token_id)
        .collect::<BTreeSet<_>>();
    lhs.iter()
        .filter(|entry| rhs_ids.contains(&entry.token_id))
        .count()
}

fn timing_summary(samples_ms: Vec<f64>) -> Result<TimingSummary, String> {
    if samples_ms.is_empty()
        || samples_ms
            .iter()
            .any(|sample| !sample.is_finite() || *sample <= 0.0)
    {
        return Err("timing samples must be finite, positive, and non-empty".into());
    }
    let mut sorted = samples_ms.clone();
    sorted.sort_by(f64::total_cmp);
    Ok(TimingSummary {
        samples_ms,
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
    })
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let position = fraction * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn vram_contract(
    artifact: &Sq8CanonicalArtifact,
    device: &DeviceInfo,
) -> Result<VramContract, String> {
    let artifact_weight_and_scale_bytes = artifact.manifest().storage.total_payload_bytes;
    let layer_norm_elements = QWEN3_14B_SQ8_STACK_LAYERS
        .checked_mul(2 * QWEN3_14B_HIDDEN_SIZE + 2 * QWEN3_14B_HEAD_DIM)
        .ok_or_else(|| "layer norm residency overflows".to_string())?;
    let layer_norm_f32_bytes = byte_count(layer_norm_elements, 4)?;
    let hidden_elements = checked_mul(SEQUENCE_LEN, QWEN3_14B_HIDDEN_SIZE, "hidden")?;
    let kv_elements = checked_mul(SEQUENCE_LEN, 8 * QWEN3_14B_HEAD_DIM, "KV")?;
    let intermediate_elements = checked_mul(SEQUENCE_LEN, 17_408, "intermediate")?;
    let f32_workspace_elements = checked_add(
        checked_add(
            checked_mul(10, hidden_elements, "hidden workspace")?,
            checked_mul(4, kv_elements, "KV workspace")?,
            "workspace",
        )?,
        checked_mul(3, intermediate_elements, "intermediate workspace")?,
        "workspace",
    )?;
    let f32_workspace_bytes = byte_count(f32_workspace_elements, 4)?;
    let (projection_workspace, _) = sq8_ck_projection_buffer_bytes(SEQUENCE_LEN, 17_408)?;
    let (hidden_activation, hidden_scales) =
        sq8_ck_activation_buffer_bytes(SEQUENCE_LEN, QWEN3_14B_HIDDEN_SIZE)?;
    let (down_activation, down_scales) = sq8_ck_activation_buffer_bytes(SEQUENCE_LEN, 17_408)?;
    let activation_bytes = checked_add(
        checked_mul(
            3,
            checked_add(hidden_activation, hidden_scales, "hidden activation")?,
            "hidden activations",
        )?,
        checked_add(down_activation, down_scales, "down activation")?,
        "activation workspace",
    )?;
    let shared_stack_workspace_bytes = checked_add(
        checked_add(f32_workspace_bytes, projection_workspace, "stack workspace")?,
        activation_bytes,
        "stack workspace",
    )? as u64;
    let resident_stack_hidden_bytes = byte_count(hidden_elements, 4)? as u64;
    let lm_head_bf16 = byte_count(
        checked_mul(QWEN3_14B_VOCAB_SIZE, QWEN3_14B_HIDDEN_SIZE, "LM head")?,
        2,
    )?;
    let model_head_resident_bytes = checked_add(
        lm_head_bf16,
        checked_add(
            byte_count(3 * QWEN3_14B_HIDDEN_SIZE, 4)?,
            byte_count(QWEN3_14B_VOCAB_SIZE, 4)?,
            "model head",
        )?,
        "model head",
    )? as u64;
    let minimum_accounted_resident_bytes = [
        artifact_weight_and_scale_bytes,
        layer_norm_f32_bytes as u64,
        shared_stack_workspace_bytes,
        resident_stack_hidden_bytes,
        model_head_resident_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| total.checked_add(value))
    .ok_or_else(|| "total VRAM residency overflows".to_string())?;
    let fits_device = minimum_accounted_resident_bytes < device.total_global_mem;
    Ok(VramContract {
        device_total_global_mem: device.total_global_mem,
        artifact_weight_and_scale_bytes,
        layer_norm_f32_bytes: layer_norm_f32_bytes as u64,
        shared_stack_workspace_bytes,
        resident_stack_hidden_bytes,
        model_head_resident_bytes,
        minimum_accounted_resident_bytes,
        unaccounted_device_bytes: device
            .total_global_mem
            .saturating_sub(minimum_accounted_resident_bytes),
        excludes_allocator_and_backend_overhead: true,
        fits_device,
    })
}

fn required_guard_names() -> Vec<String> {
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .map(str::to_string)
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

fn require_hip_kernel_guards() -> Result<(), String> {
    let invalid = required_guard_names()
        .into_iter()
        .filter(|name| std::env::var(name).ok().as_deref() != Some("1"))
        .collect::<Vec<_>>();
    if !invalid.is_empty() {
        return Err(format!(
            "SQ8 full-model validation requires these HIP-only guards to equal 1: {}",
            invalid.join(",")
        ));
    }
    Ok(())
}

fn validate_finite(values: &[f32], label: &str) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{label} contains non-finite value {value} at {index}"
        ));
    }
    Ok(())
}

fn last_row(values: &[f32], columns: usize) -> Result<&[f32], String> {
    if columns == 0 || values.len() != SEQUENCE_LEN * columns {
        return Err(format!(
            "last-row shape mismatch: values={} expected={}",
            values.len(),
            SEQUENCE_LEN * columns
        ));
    }
    Ok(&values[(SEQUENCE_LEN - 1) * columns..SEQUENCE_LEN * columns])
}

fn f32_bits_equal(lhs: &[f32], rhs: &[f32]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs)
            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
}

fn bf16_values_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn safe_relative_regular_file(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe vLLM oracle relative path: {relative}"));
    }
    let path = root.join(relative_path);
    let canonical_root = root
        .canonicalize()
        .map_err(|err| format!("failed to resolve {}: {err}", root.display()))?;
    let canonical_path = path
        .canonicalize()
        .map_err(|err| format!("failed to resolve {}: {err}", path.display()))?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(format!("vLLM oracle path escapes root: {relative}"));
    }
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|err| format!("failed to inspect {}: {err}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "vLLM oracle artifact is not a regular non-symlink file: {}",
            path.display()
        ));
    }
    Ok(path)
}

fn read_regular_file_exact(path: &Path, expected_bytes: u64) -> Result<Vec<u8>, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|err| format!("failed to inspect {}: {err}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "expected a regular non-symlink file: {}",
            path.display()
        ));
    }
    if metadata.len() != expected_bytes {
        return Err(format!(
            "file byte mismatch for {}: expected={expected_bytes} actual={}",
            path.display(),
            metadata.len()
        ));
    }
    std::fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))
}

fn sha256_regular_file(path: &Path) -> Result<String, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|err| format!("failed to inspect {}: {err}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "expected a regular non-symlink file: {}",
            path.display()
        ));
    }
    let mut file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; UPLOAD_CHUNK_BYTES];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| format!("{label} multiplication overflows"))
}

fn checked_add(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
    lhs.checked_add(rhs)
        .ok_or_else(|| format!("{label} addition overflows"))
}

fn byte_count(elements: usize, element_bytes: usize) -> Result<usize, String> {
    checked_mul(elements, element_bytes, "byte count")
}

fn write_json_no_clobber(path: &Path, result: &FullModelResult) -> Result<(), String> {
    let mut payload = serde_json::to_vec_pretty(result)
        .map_err(|err| format!("failed to serialize SQ8 full-model result: {err}"))?;
    payload.push(b'\n');
    publish_bytes_no_clobber(path, &payload)
}

fn publish_bytes_no_clobber(path: &Path, payload: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("output has no parent: {}", path.display()))?;
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "output filename is not UTF-8".to_string())?;
    let temporary = parent.join(format!(".{filename}.tmp.{}", std::process::id()));
    let write_result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|err| format!("failed to create {}: {err}", temporary.display()))?;
        file.write_all(payload)
            .map_err(|err| format!("failed to write {}: {err}", temporary.display()))?;
        file.sync_all()
            .map_err(|err| format!("failed to sync {}: {err}", temporary.display()))?;
        std::fs::hard_link(&temporary, path).map_err(|err| {
            format!(
                "failed to publish {} without clobbering: {err}",
                path.display()
            )
        })?;
        Ok::<(), String>(())
    })();
    let _ = std::fs::remove_file(&temporary);
    write_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_uses_lower_token_id_for_equal_logits() {
        let values = [1.0_f32, 3.0, 3.0, 2.0];
        let result = top_k(&values, 3).unwrap();
        assert_eq!(
            result
                .iter()
                .map(|entry| entry.token_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn percentile_interpolates_fixed_samples() {
        let values = [1.0_f64, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&values, 0.50), 2.5);
        assert!((percentile(&values, 0.95) - 3.85).abs() < 1.0e-12);
    }

    #[test]
    fn cpu_rmsnorm_matches_simple_vector() {
        let input = vec![2.0_f32; QWEN3_14B_HIDDEN_SIZE];
        let weight = vec![0.5_f32; QWEN3_14B_HIDDEN_SIZE];
        let output = cpu_rmsnorm(&input, &weight).unwrap();
        assert!(output.iter().all(|value| (*value - 0.5).abs() < 1.0e-6));
    }

    #[test]
    fn last_row_rejects_wrong_shape() {
        assert!(last_row(&[0.0_f32; 7], 1).is_err());
        let values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(last_row(&values, 2).unwrap(), &[14.0, 15.0]);
    }

    #[test]
    fn f32_bit_equality_distinguishes_signed_zero() {
        assert!(f32_bits_equal(&[1.0], &[1.0]));
        assert!(!f32_bits_equal(&[0.0], &[-0.0]));
    }

    #[test]
    fn no_clobber_publish_preserves_existing_destination() {
        let root = std::env::temp_dir().join(format!(
            "ullm-sq8-full-model-no-clobber-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir(&root).unwrap();
        let destination = root.join("result.json");
        std::fs::write(&destination, b"original\n").unwrap();

        assert!(publish_bytes_no_clobber(&destination, b"replacement\n").is_err());
        assert_eq!(std::fs::read(&destination).unwrap(), b"original\n");
        let temporary = root.join(format!(".result.json.tmp.{}", std::process::id()));
        assert!(!temporary.exists());

        std::fs::remove_dir_all(root).unwrap();
    }
}
