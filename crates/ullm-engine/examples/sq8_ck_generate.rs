// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use ullm_engine::loader::{
    PassthroughPayloadVerification, read_named_passthrough_f32, verify_named_passthrough_payload,
};
use ullm_engine::sq_canonical::{Sq8CanonicalArtifact, read_sq8_canonical_artifact};
use ullm_engine::sq_reference::{
    Sq8CorrectnessMetrics, compare_sq8_correctness, sq8_f32_le_sha256,
};
use ullm_engine::sq8_embedding_runtime::QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::sq8_generation_runtime::{
    QWEN3_14B_SQ8_GENERATION_EOS_TOKEN_ID, QWEN3_14B_SQ8_GENERATION_MAX_NEW_TOKENS,
    QWEN3_14B_SQ8_GENERATION_PROMPT_TOKEN_IDS, QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS,
    Qwen3Sq8GenerationRuntime, Sq8GenerationCompletionReason, Sq8GenerationRuntimeStatus,
    Sq8GenerationStepPhase,
};
use ullm_engine::sq8_layer_oracle::{QWEN3_14B_HEAD_DIM, QWEN3_14B_HIDDEN_SIZE};
use ullm_engine::sq8_layer_runtime::{
    QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
    Qwen3Sq8LayerNormValues, Sq8LayerExecutionProfile,
};
use ullm_engine::sq8_model_head_runtime::{
    QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_VOCAB_SIZE,
    Sq8ModelHeadDeviceIdentity, validate_qwen3_14b_sq8_r9700_device_info,
};
use ullm_engine::sq8_stack_runtime::{QWEN3_14B_SQ8_STACK_LAYERS, QWEN3_14B_SQ8_STACK_PROJECTIONS};
use ullm_runtime_sys::{DeviceInfo, RuntimeContext, device_count, device_info};

const SCHEMA_VERSION: &str = "ullm.sq8.generation.v1";
const ORACLE_SCHEMA_VERSION: &str = "ullm.qwen3_generation_oracle.v1";
const ORACLE_METADATA_BYTES: u64 = 29_465;
const UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const STEP_COUNT: usize = 8;
const TOP_K: usize = 10;
const MAX_RELATIVE_L2: f64 = 0.20;
const MIN_COSINE_SIMILARITY: f64 = 0.98;
const MIN_TOP_10_OVERLAP: usize = 3;

const EXPECTED_ARTIFACT_CONTENT_SHA256: &str =
    "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147";
const EXPECTED_PACKAGE_MANIFEST_SHA256: &str =
    "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb";
const EXPECTED_ORACLE_METADATA_SHA256: &str =
    "5fc03a28cd15409e84a7fd23fd51c0cbd6ec9cf8761a66d1f5ede7ddfe3226a0";
const EXPECTED_MODEL_REVISION: &str = "9a283b4a5efbc09ce247e0ae5b02b744739e525a";
const EXPECTED_CONFIG_SHA256: &str =
    "c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793";
const EXPECTED_INDEX_SHA256: &str =
    "6a9c8e17744118347080916d8f673b881941cf42989ee77266b14dc2062a7151";
const EXPECTED_GENERATED_TOKEN_IDS: [usize; STEP_COUNT] = [353, 10, 4_999, 1_725, 15, 16, 17, 18];

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
    source: OracleSource,
    prompt: OraclePrompt,
    generation: OracleGeneration,
    generated_token_ids: Vec<usize>,
    steps: Vec<OracleStep>,
    feedback: OracleFeedback,
    execution: OracleExecution,
    environment: OracleEnvironment,
}

#[derive(Debug, Deserialize)]
struct OracleSource {
    name: String,
    revision: OracleRevision,
    config: OracleConfig,
}

#[derive(Debug, Deserialize)]
struct OracleRevision {
    revision: String,
    revision_consistent: bool,
}

#[derive(Debug, Deserialize)]
struct OracleConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    tie_word_embeddings: bool,
    torch_dtype: String,
    use_cache: bool,
}

#[derive(Debug, Deserialize)]
struct OraclePrompt {
    token_ids: Vec<usize>,
    position_ids: Vec<usize>,
    attention: String,
    bos_inserted: bool,
    chat_template_applied: bool,
}

#[derive(Debug, Deserialize)]
struct OracleGeneration {
    method: String,
    temperature: f32,
    max_new_tokens: usize,
    min_new_tokens: usize,
    fixed_step_count: usize,
    eos_token_id: usize,
    early_stop_on_eos: bool,
    ignore_eos: bool,
    finish_reason: String,
    top_k_recorded: usize,
    topk_tie_breaker: String,
}

#[derive(Debug, Deserialize)]
struct OracleFeedback {
    all_feedback_edges_match: bool,
    feedback_edge_count: usize,
    step_zero_uses_prompt_last_token: bool,
    subsequent_steps_use_previous_generated_token: bool,
}

#[derive(Debug, Deserialize)]
struct OracleExecution {
    backend: String,
    runner: String,
    dtype: String,
    forward_token_counts: Vec<usize>,
    max_model_len: usize,
    max_num_batched_tokens: usize,
    max_num_seqs: usize,
    tensor_parallel_size: usize,
    pipeline_parallel_size: usize,
    enforce_eager: bool,
    enable_prefix_caching: bool,
    async_scheduling: bool,
    v1_multiprocessing: bool,
}

#[derive(Debug, Deserialize)]
struct OracleEnvironment {
    gpu: OracleGpu,
}

#[derive(Debug, Deserialize)]
struct OracleGpu {
    visible_device_index: usize,
    name: String,
    gfx: String,
    compute_capability: Vec<i32>,
    total_memory_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct OracleStep {
    step_index: usize,
    input_token_id: usize,
    input_position_id: usize,
    input_origin: String,
    feedback_from_step: Option<usize>,
    feedback_matches_previous_generated: Option<bool>,
    forward_token_count: usize,
    generated_token_id: usize,
    generated_token_position_id: usize,
    generated_matches_logits_top1: bool,
    final_hidden: OracleTensorDescriptor,
    logits: OracleTensorDescriptor,
    top_10: Vec<TopKEntry>,
}

#[derive(Debug, Deserialize)]
struct OracleTensorDescriptor {
    file: String,
    bytes: u64,
    sha256: String,
    shape: Vec<usize>,
    source_dtype: String,
    storage_dtype: String,
}

#[derive(Debug)]
struct OracleFixture {
    metadata_sha256: String,
    revision: String,
    steps: Vec<OracleStepFixture>,
}

#[derive(Debug)]
struct OracleStepFixture {
    generated_token_id: usize,
    final_hidden: Vec<f32>,
    final_hidden_sha256: String,
    logits: Vec<f32>,
    logits_sha256: String,
    top_10: Vec<TopKEntry>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct TopKEntry {
    token_id: usize,
    logit: f32,
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
struct GenerationGateResult {
    schema_version: &'static str,
    passed: bool,
    source: SourceIdentity,
    input: InputIdentity,
    payloads: PayloadEvidence,
    device: DeviceRecord,
    generation: GenerationRecord,
    steps: Vec<GenerationStepCheck>,
    execution: ExecutionRecord,
    timing: TimingRecord,
    allocator: AllocatorRecord,
}

#[derive(Debug, Serialize)]
struct SourceIdentity {
    artifact_content_sha256: String,
    artifact_config_sha256: String,
    artifact_index_sha256: String,
    package_manifest_sha256: String,
    vllm_oracle_metadata_sha256: String,
    model_revision: String,
}

#[derive(Debug, Serialize)]
struct InputIdentity {
    prompt_token_ids: [usize; QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS],
    prompt_position_ids: [usize; QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS],
    max_new_tokens: usize,
    eos_token_id: usize,
    sampling: &'static str,
}

#[derive(Debug, Serialize)]
struct PayloadEvidence {
    layer_count: usize,
    layer_norm_tensor_count: usize,
    layer_norms: Vec<PayloadIdentity>,
    embedding_payload_sha256: String,
    final_norm_payload_sha256: String,
    lm_head_payload_sha256: String,
    upload_chunk_bytes: usize,
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
struct GenerationRecord {
    request_id: u64,
    generated_token_ids: Vec<usize>,
    expected_generated_token_ids: [usize; STEP_COUNT],
    generated_token_ids_u32_le_sha256: String,
    decode_input_token_ids: Vec<usize>,
    decode_positions: Vec<usize>,
    completion_reason: &'static str,
    final_kv_len: usize,
    released_kv_blocks: usize,
    allocation_released: bool,
    feedback_verified: bool,
    exact_token_sequence: bool,
}

#[derive(Debug, Serialize)]
struct GenerationStepCheck {
    step_index: usize,
    phase: &'static str,
    input_token_id: Option<usize>,
    cache_position: Option<usize>,
    cache_len_after: usize,
    output_token_id: usize,
    expected_output_token_id: usize,
    output_logit: f32,
    device_final_hidden_health: TensorHealth,
    vllm_final_hidden_health: TensorHealth,
    device_logits_health: TensorHealth,
    vllm_logits_health: TensorHealth,
    vllm_final_hidden_sha256: String,
    vllm_logits_sha256: String,
    final_hidden: TensorGate,
    logits: TensorGate,
    device_top_10: Vec<TopKEntry>,
    vllm_top_10: Vec<TopKEntry>,
    top_1_exact: bool,
    top_10_overlap: usize,
    minimum_top_10_overlap: usize,
    started_at_ns: u128,
    completed_at_ns: u128,
    latency_ns: u128,
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

#[derive(Debug, Serialize)]
struct TensorGate {
    metrics: Sq8CorrectnessMetrics,
    max_relative_l2: f64,
    min_cosine_similarity: f64,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ExecutionRecord {
    runtime_status_after_load: &'static str,
    runtime_status_after_run: &'static str,
    profile: &'static str,
    stack_steps: Vec<StackStepRecord>,
    counters: ExecutionCounters,
    final_cache_lengths: Vec<usize>,
    generated_token_ids_sha256: String,
    required_hip_kernel_env: Vec<String>,
    feedback_verified: bool,
    allocation_released: bool,
    fallback_used: bool,
    host_staging_used: bool,
}

#[derive(Debug, Serialize)]
struct StackStepRecord {
    phase: &'static str,
    position: usize,
    sequence_len: usize,
    cache_len: usize,
    projection_calls: usize,
    activation_quantizations: usize,
    layer_d2d_copies: usize,
    kv_write_calls: usize,
    paged_attention_calls: usize,
    input_d2d_copies: usize,
    all_ck: bool,
    fallback_used: bool,
    host_staging_used: bool,
}

#[derive(Debug, Serialize)]
struct ExecutionCounters {
    embedding_gather_calls: usize,
    prompt_embedding_d2d_copies: usize,
    stack_input_d2d_copies: usize,
    projection_calls: usize,
    activation_quantizations: usize,
    layer_d2d_copies: usize,
    kv_write_calls: usize,
    paged_attention_calls: usize,
    model_head_calls: usize,
    model_head_d2d_copies: usize,
    result_readback_count: usize,
    execution_synchronization_count: usize,
    scheduler_prefill_completions: usize,
    scheduler_prefill_token_records: usize,
    scheduler_decode_advances: usize,
    scheduler_release_calls: usize,
    identity_check_count: usize,
}

#[derive(Debug, Serialize)]
struct TimingRecord {
    request_count: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    time_to_first_token_ns: u128,
    request_latency_ns: u128,
    decode_elapsed_ns: u128,
    requests_per_second: f64,
    generated_tokens_per_second: f64,
    total_tokens_per_second: f64,
    decode_tokens_per_second: Option<f64>,
}

#[derive(Debug, Serialize)]
struct AllocatorRecord {
    before: AllocatorStatsRecord,
    after_release: AllocatorStatsRecord,
}

#[derive(Debug, Serialize)]
struct AllocatorStatsRecord {
    block_size_tokens: u32,
    total_blocks: u32,
    free_blocks: usize,
    allocated_blocks: usize,
    free_runs: usize,
    largest_free_run: usize,
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
    let (norms, norm_identities) = load_all_verified_layer_norms(&options.package)?;
    let (runtime_index, device) = isolated_gfx1201_device()?;
    let mut context = RuntimeContext::create(runtime_index)?;
    let mut stream = context.create_stream()?;
    let mut runtime = Qwen3Sq8GenerationRuntime::load(
        &mut context,
        &mut stream,
        &artifact,
        &options.package,
        norms,
        UPLOAD_CHUNK_BYTES,
    )?;
    let load_status = runtime_status_name(runtime.status());
    if runtime.status() != Sq8GenerationRuntimeStatus::Ready {
        return Err(format!(
            "SQ8 generation runtime is not ready after load: {:?}",
            runtime.status()
        ));
    }
    runtime.load_report().validate_contract()?;
    validate_load_identity(
        runtime.load_report().device.clone(),
        &device,
        &runtime.load_report().artifact_content_sha256,
        &runtime.load_report().package_manifest_sha256,
    )?;
    let load_report = runtime.load_report().clone();

    let generation =
        runtime.run_fixed_synchronized(QWEN3_14B_SQ8_GENERATION_MAX_NEW_TOKENS, &mut stream)?;
    generation.validate_contract()?;
    let completed_status = runtime_status_name(runtime.status());
    if runtime.status() != Sq8GenerationRuntimeStatus::Completed {
        return Err(format!(
            "SQ8 generation runtime did not reach completed state: {:?}",
            runtime.status()
        ));
    }
    if generation.steps.len() != STEP_COUNT || oracle.steps.len() != STEP_COUNT {
        return Err(format!(
            "SQ8 generation comparison requires {STEP_COUNT} complete steps: ullm={} oracle={}",
            generation.steps.len(),
            oracle.steps.len()
        ));
    }

    let mut step_checks = Vec::with_capacity(STEP_COUNT);
    for (index, (actual, reference)) in generation.steps.iter().zip(oracle.steps.iter()).enumerate()
    {
        step_checks.push(compare_generation_step(index, actual, reference)?);
    }
    let exact_token_sequence =
        generation.completion.generated_token_ids == EXPECTED_GENERATED_TOKEN_IDS;
    let steps_passed = step_checks.iter().all(|step| step.passed);
    let passed = exact_token_sequence
        && steps_passed
        && generation.report.feedback_verified
        && generation.completion.allocation_released
        && !generation.report.fallback_used
        && !generation.report.host_staging_used;

    let report = &generation.report;
    let result = GenerationGateResult {
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
            vllm_oracle_metadata_sha256: oracle.metadata_sha256,
            model_revision: oracle.revision,
        },
        input: InputIdentity {
            prompt_token_ids: QWEN3_14B_SQ8_GENERATION_PROMPT_TOKEN_IDS,
            prompt_position_ids: [0, 1, 2, 3, 4, 5, 6, 7],
            max_new_tokens: generation.request.max_new_tokens,
            eos_token_id: generation.request.eos_token_id,
            sampling: "greedy_temperature_zero",
        },
        payloads: PayloadEvidence {
            layer_count: QWEN3_14B_SQ8_STACK_LAYERS,
            layer_norm_tensor_count: norm_identities.len(),
            layer_norms: norm_identities,
            embedding_payload_sha256: load_report.embedding_payload_sha256,
            final_norm_payload_sha256: load_report.final_norm_payload_sha256,
            lm_head_payload_sha256: load_report.lm_head_payload_sha256,
            upload_chunk_bytes: load_report.upload_chunk_bytes,
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
        generation: GenerationRecord {
            request_id: generation.request.request_id.0,
            generated_token_ids: generation.completion.generated_token_ids.clone(),
            expected_generated_token_ids: EXPECTED_GENERATED_TOKEN_IDS,
            generated_token_ids_u32_le_sha256: token_ids_sha256(
                &generation.completion.generated_token_ids,
            )?,
            decode_input_token_ids: generation.completion.decode_input_token_ids.clone(),
            decode_positions: generation.completion.decode_positions.clone(),
            completion_reason: completion_reason_name(generation.completion.reason),
            final_kv_len: generation.completion.final_kv_len,
            released_kv_blocks: generation.completion.released_kv_blocks,
            allocation_released: generation.completion.allocation_released,
            feedback_verified: report.feedback_verified,
            exact_token_sequence,
        },
        steps: step_checks,
        execution: ExecutionRecord {
            runtime_status_after_load: load_status,
            runtime_status_after_run: completed_status,
            profile: profile_name(report.profile)?,
            stack_steps: report
                .stack_steps
                .iter()
                .map(stack_step_record)
                .collect::<Result<Vec<_>, _>>()?,
            counters: ExecutionCounters {
                embedding_gather_calls: report.embedding_gather_calls,
                prompt_embedding_d2d_copies: report.prompt_embedding_d2d_copies,
                stack_input_d2d_copies: report.stack_input_d2d_copies,
                projection_calls: report.projection_calls,
                activation_quantizations: report.activation_quantizations,
                layer_d2d_copies: report.layer_d2d_copies,
                kv_write_calls: report.kv_write_calls,
                paged_attention_calls: report.paged_attention_calls,
                model_head_calls: report.model_head_calls,
                model_head_d2d_copies: report.model_head_d2d_copies,
                result_readback_count: report.result_readback_count,
                execution_synchronization_count: report.execution_synchronization_count,
                scheduler_prefill_completions: report.scheduler_prefill_completions,
                scheduler_prefill_token_records: report.scheduler_prefill_token_records,
                scheduler_decode_advances: report.scheduler_decode_advances,
                scheduler_release_calls: report.scheduler_release_calls,
                identity_check_count: report.identity_check_count,
            },
            final_cache_lengths: report.final_cache_lengths.to_vec(),
            generated_token_ids_sha256: report.generated_token_ids_sha256.clone(),
            required_hip_kernel_env: required_guard_names(),
            feedback_verified: report.feedback_verified,
            allocation_released: report.allocation_released,
            fallback_used: report.fallback_used,
            host_staging_used: report.host_staging_used,
        },
        timing: TimingRecord {
            request_count: generation.metrics.request_count,
            prompt_tokens: generation.metrics.prompt_tokens,
            generated_tokens: generation.metrics.generated_tokens,
            time_to_first_token_ns: generation.metrics.time_to_first_token_ns,
            request_latency_ns: generation.metrics.request_latency_ns,
            decode_elapsed_ns: generation.metrics.decode_elapsed_ns,
            requests_per_second: generation.metrics.requests_per_second,
            generated_tokens_per_second: generation.metrics.generated_tokens_per_second,
            total_tokens_per_second: generation.metrics.total_tokens_per_second,
            decode_tokens_per_second: generation.metrics.decode_tokens_per_second,
        },
        allocator: AllocatorRecord {
            before: allocator_record(generation.allocator_before),
            after_release: allocator_record(generation.allocator_after_release),
        },
    };
    write_json_no_clobber(&options.output, &result)?;
    if !result.passed {
        return Err(format!(
            "SQ8 generation gate failed; complete evidence was written to {}",
            options.output.display()
        ));
    }
    println!(
        "passed=true output={} tokens={:?} request_latency_ms={:.6} generated_tokens_per_second={:.3}",
        options.output.display(),
        result.generation.generated_token_ids,
        result.timing.request_latency_ns as f64 / 1_000_000.0,
        result.timing.generated_tokens_per_second,
    );
    Ok(())
}

fn parse_options() -> Result<Options, String> {
    let mut artifact = None;
    let mut package = None;
    let mut oracle = None;
    let mut output = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(usage)?;
        match flag.to_str() {
            Some("--artifact") => set_once(&mut artifact, value, "--artifact")?,
            Some("--package") => set_once(&mut package, value, "--package")?,
            Some("--oracle") => set_once(&mut oracle, value, "--oracle")?,
            Some("--output") => set_once(&mut output, value, "--output")?,
            _ => return Err(format!("unknown argument {:?}; {}", flag, usage())),
        }
    }
    let options = Options {
        artifact: PathBuf::from(artifact.ok_or_else(usage)?),
        package: PathBuf::from(package.ok_or_else(usage)?),
        oracle: PathBuf::from(oracle.ok_or_else(usage)?),
        output: PathBuf::from(output.ok_or_else(usage)?),
    };
    for (label, path) in [
        ("artifact", &options.artifact),
        ("package", &options.package),
        ("vLLM oracle", &options.oracle),
    ] {
        if !path.is_dir() {
            return Err(format!("{label} is not a directory: {}", path.display()));
        }
    }
    match std::fs::symlink_metadata(&options.output) {
        Ok(_) => {
            return Err(format!(
                "output already exists: {}",
                options.output.display()
            ));
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(format!(
                "failed to inspect {}: {err}",
                options.output.display()
            ));
        }
    }
    let parent = output_parent(&options.output)?;
    if !parent.is_dir() {
        return Err(format!(
            "output parent is not a directory: {}",
            parent.display()
        ));
    }
    Ok(options)
}

fn set_once(slot: &mut Option<OsString>, value: OsString, flag: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("duplicate argument {flag}"));
    }
    Ok(())
}

fn usage() -> String {
    "usage: sq8_ck_generate --artifact ARTIFACT_DIR --package THIN_PACKAGE --oracle VLLM_ORACLE_DIR --output OUTPUT_JSON".to_string()
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
        || manifest.coverage.selected_pair_count != QWEN3_14B_SQ8_STACK_PROJECTIONS as u64
        || manifest.coverage.unpaired_tensor_count != 0
        || manifest.quantized_tensors.len() != QWEN3_14B_SQ8_STACK_PROJECTIONS
    {
        return Err("SQ8 artifact checkpoint identity or full coverage mismatch".into());
    }
    Ok(())
}

fn load_oracle_fixture(root: &Path) -> Result<OracleFixture, String> {
    let metadata_path = root.join("metadata.json");
    let metadata_bytes = read_regular_file_exact(&metadata_path, ORACLE_METADATA_BYTES)?;
    let metadata_sha256 = sha256_bytes(&metadata_bytes);
    if metadata_sha256 != EXPECTED_ORACLE_METADATA_SHA256 {
        return Err(format!(
            "vLLM generation oracle metadata SHA-256 mismatch: expected={EXPECTED_ORACLE_METADATA_SHA256} actual={metadata_sha256}"
        ));
    }
    let metadata: OracleMetadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|err| format!("failed to parse {}: {err}", metadata_path.display()))?;
    validate_oracle_metadata(&metadata)?;

    let mut steps = Vec::with_capacity(STEP_COUNT);
    for step in &metadata.steps {
        let hidden_file = format!("steps/step-{:02}-final-hidden.f32", step.step_index);
        let logits_file = format!("steps/step-{:02}-logits.f32", step.step_index);
        validate_oracle_descriptor(&step.final_hidden, &hidden_file, &[QWEN3_14B_HIDDEN_SIZE])?;
        validate_oracle_descriptor(&step.logits, &logits_file, &[QWEN3_14B_VOCAB_SIZE])?;
        let final_hidden = read_oracle_tensor(root, &step.final_hidden)?;
        let logits = read_oracle_tensor(root, &step.logits)?;
        let recomputed_top_10 = top_k(&logits, TOP_K)?;
        if recomputed_top_10 != step.top_10
            || recomputed_top_10[0].token_id != step.generated_token_id
        {
            return Err(format!(
                "vLLM generation oracle step {} top-10/top-1 mismatch",
                step.step_index
            ));
        }
        steps.push(OracleStepFixture {
            generated_token_id: step.generated_token_id,
            final_hidden,
            final_hidden_sha256: step.final_hidden.sha256.clone(),
            logits,
            logits_sha256: step.logits.sha256.clone(),
            top_10: step.top_10.clone(),
        });
    }
    Ok(OracleFixture {
        metadata_sha256,
        revision: metadata.source.revision.revision,
        steps,
    })
}

fn validate_oracle_metadata(metadata: &OracleMetadata) -> Result<(), String> {
    let expected_positions = (0..QWEN3_14B_SQ8_GENERATION_PROMPT_TOKENS).collect::<Vec<_>>();
    let config = &metadata.source.config;
    if metadata.schema_version != ORACLE_SCHEMA_VERSION
        || metadata.source.name != "Qwen/Qwen3-14B-FP8"
        || metadata.source.revision.revision != EXPECTED_MODEL_REVISION
        || !metadata.source.revision.revision_consistent
        || config.hidden_size != QWEN3_14B_HIDDEN_SIZE
        || config.intermediate_size != 17_408
        || config.num_hidden_layers != QWEN3_14B_SQ8_STACK_LAYERS
        || config.num_attention_heads != 40
        || config.num_key_value_heads != 8
        || config.head_dim != QWEN3_14B_HEAD_DIM
        || config.vocab_size != QWEN3_14B_VOCAB_SIZE
        || config.rms_norm_eps.to_bits() != 1.0e-6_f64.to_bits()
        || config.rope_theta.to_bits() != 1_000_000.0_f64.to_bits()
        || config.tie_word_embeddings
        || config.torch_dtype != "bfloat16"
        || !config.use_cache
    {
        return Err("vLLM generation oracle schema/model identity mismatch".into());
    }
    if metadata.prompt.token_ids != QWEN3_14B_SQ8_GENERATION_PROMPT_TOKEN_IDS
        || metadata.prompt.position_ids != expected_positions
        || metadata.prompt.attention != "causal"
        || metadata.prompt.bos_inserted
        || metadata.prompt.chat_template_applied
    {
        return Err("vLLM generation oracle prompt semantics mismatch".into());
    }
    let generation = &metadata.generation;
    if generation.method != "greedy"
        || generation.temperature.to_bits() != 0.0_f32.to_bits()
        || generation.max_new_tokens != STEP_COUNT
        || generation.min_new_tokens != STEP_COUNT
        || generation.fixed_step_count != STEP_COUNT
        || generation.eos_token_id != QWEN3_14B_SQ8_GENERATION_EOS_TOKEN_ID
        || generation.early_stop_on_eos
        || !generation.ignore_eos
        || generation.finish_reason != "length"
        || generation.top_k_recorded != TOP_K
        || generation.topk_tie_breaker != "token_id_ascending"
        || metadata.generated_token_ids != EXPECTED_GENERATED_TOKEN_IDS
    {
        return Err("vLLM generation oracle generation semantics mismatch".into());
    }
    if !metadata.feedback.all_feedback_edges_match
        || metadata.feedback.feedback_edge_count != STEP_COUNT - 1
        || !metadata.feedback.step_zero_uses_prompt_last_token
        || !metadata
            .feedback
            .subsequent_steps_use_previous_generated_token
    {
        return Err("vLLM generation oracle feedback contract mismatch".into());
    }
    let execution = &metadata.execution;
    if execution.backend != "vLLM"
        || execution.runner != "generate"
        || execution.dtype != "bfloat16"
        || execution.forward_token_counts != [8, 1, 1, 1, 1, 1, 1, 1]
        || execution.max_model_len != 16
        || execution.max_num_batched_tokens != 8
        || execution.max_num_seqs != 1
        || execution.tensor_parallel_size != 1
        || execution.pipeline_parallel_size != 1
        || !execution.enforce_eager
        || execution.enable_prefix_caching
        || execution.async_scheduling
        || execution.v1_multiprocessing
    {
        return Err("vLLM generation oracle execution contract mismatch".into());
    }
    let gpu = &metadata.environment.gpu;
    if gpu.visible_device_index != 0
        || gpu.name != "AMD Radeon Graphics"
        || !gpu.gfx.eq_ignore_ascii_case("gfx1201")
        || gpu.compute_capability != [12, 0]
        || !(30 * 1024 * 1024 * 1024..=34 * 1024 * 1024 * 1024).contains(&gpu.total_memory_bytes)
    {
        return Err("vLLM generation oracle device identity mismatch".into());
    }
    if metadata.steps.len() != STEP_COUNT {
        return Err(format!(
            "vLLM generation oracle step count mismatch: expected={STEP_COUNT} actual={}",
            metadata.steps.len()
        ));
    }
    for (index, step) in metadata.steps.iter().enumerate() {
        let expected_input = if index == 0 {
            8
        } else {
            EXPECTED_GENERATED_TOKEN_IDS[index - 1]
        };
        let expected_origin = if index == 0 {
            "prompt_last_token"
        } else {
            "previous_step_generated_token"
        };
        if step.step_index != index
            || step.input_token_id != expected_input
            || step.input_position_id != 7 + index
            || step.input_origin != expected_origin
            || step.feedback_from_step != index.checked_sub(1)
            || step.feedback_matches_previous_generated != (index > 0).then_some(true)
            || step.forward_token_count != if index == 0 { 8 } else { 1 }
            || step.generated_token_id != EXPECTED_GENERATED_TOKEN_IDS[index]
            || step.generated_token_position_id != 8 + index
            || !step.generated_matches_logits_top1
            || step.top_10.len() != TOP_K
        {
            return Err(format!(
                "vLLM generation oracle step {index} semantics mismatch"
            ));
        }
    }
    Ok(())
}

fn validate_oracle_descriptor(
    descriptor: &OracleTensorDescriptor,
    expected_file: &str,
    expected_shape: &[usize],
) -> Result<(), String> {
    let expected_bytes = expected_shape
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| format!("oracle descriptor {expected_file} byte size overflows"))?;
    if descriptor.file != expected_file
        || descriptor.bytes != expected_bytes as u64
        || descriptor.shape != expected_shape
        || descriptor.source_dtype != "torch.bfloat16"
        || descriptor.storage_dtype != "float32_le"
        || !is_sha256(&descriptor.sha256)
    {
        return Err(format!(
            "vLLM generation oracle descriptor mismatch for {expected_file}"
        ));
    }
    Ok(())
}

fn read_oracle_tensor(
    root: &Path,
    descriptor: &OracleTensorDescriptor,
) -> Result<Vec<f32>, String> {
    let path = safe_relative_regular_file(root, &descriptor.file)?;
    let bytes = read_regular_file_exact(&path, descriptor.bytes)?;
    let actual_sha256 = sha256_bytes(&bytes);
    if actual_sha256 != descriptor.sha256 {
        return Err(format!(
            "vLLM generation oracle {} checksum mismatch: expected={} actual={actual_sha256}",
            descriptor.file, descriptor.sha256
        ));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    validate_finite(&values, &format!("vLLM oracle {}", descriptor.file))?;
    Ok(values)
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
    let unique_names = identities
        .iter()
        .map(|identity| identity.tensor_name.as_str())
        .collect::<BTreeSet<_>>();
    if norms.len() != QWEN3_14B_SQ8_STACK_LAYERS
        || identities.len() != QWEN3_14B_SQ8_STACK_LAYERS * 4
        || unique_names.len() != identities.len()
    {
        return Err("verified SQ8 layer norm count/identity mismatch".into());
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
            "SQ8 generation validation requires exactly one visible runtime HIP device, found {}",
            devices.len()
        ));
    }
    let (runtime_index, device) = devices.into_iter().next().expect("one HIP device");
    validate_qwen3_14b_sq8_r9700_device_info(&device)?;
    if device.device_id != 0 {
        return Err(format!(
            "SQ8 generation validation requires isolated HIP device 0, got {}",
            device.device_id
        ));
    }
    Ok((runtime_index, device))
}

fn validate_load_identity(
    load: Sq8ModelHeadDeviceIdentity,
    device: &DeviceInfo,
    artifact_sha256: &str,
    package_sha256: &str,
) -> Result<(), String> {
    if load.device_id != device.device_id
        || load.backend != device.backend
        || load.name != device.name
        || load.gcn_arch_name != device.gcn_arch_name
        || load.compute_major != device.compute_major
        || load.compute_minor != device.compute_minor
        || load.total_global_mem != device.total_global_mem
        || artifact_sha256 != EXPECTED_ARTIFACT_CONTENT_SHA256
        || package_sha256 != EXPECTED_PACKAGE_MANIFEST_SHA256
    {
        return Err("SQ8 generation resident load identity mismatch".into());
    }
    Ok(())
}

fn compare_generation_step(
    index: usize,
    actual: &ullm_engine::sq8_generation_runtime::Sq8GenerationStepResult,
    reference: &OracleStepFixture,
) -> Result<GenerationStepCheck, String> {
    actual.validate_contract()?;
    if actual.generated_index != index {
        return Err(format!(
            "SQ8 generation step order mismatch: expected={index} actual={}",
            actual.generated_index
        ));
    }
    let device_final_hidden_health = tensor_health(&actual.final_hidden)?;
    let device_logits_health = tensor_health(&actual.logits)?;
    if device_final_hidden_health.f32_le_sha256 != actual.final_hidden_f32_le_sha256
        || device_logits_health.f32_le_sha256 != actual.logits_f32_le_sha256
    {
        return Err(format!(
            "SQ8 generation step {index} typed tensor health/hash changed"
        ));
    }
    let vllm_final_hidden_health = tensor_health(&reference.final_hidden)?;
    let vllm_logits_health = tensor_health(&reference.logits)?;
    if vllm_final_hidden_health.f32_le_sha256 != reference.final_hidden_sha256
        || vllm_logits_health.f32_le_sha256 != reference.logits_sha256
    {
        return Err(format!(
            "vLLM generation oracle step {index} tensor hash changed after load"
        ));
    }
    let final_hidden = tensor_gate(&reference.final_hidden, &actual.final_hidden)?;
    let logits = tensor_gate(&reference.logits, &actual.logits)?;
    let device_top_10 = top_k(&actual.logits, TOP_K)?;
    let top_10_overlap = top_k_overlap(&device_top_10, &reference.top_10);
    let top_1_exact = actual.output_token_id == reference.generated_token_id
        && actual.output_token_id == EXPECTED_GENERATED_TOKEN_IDS[index]
        && device_top_10[0].token_id == reference.top_10[0].token_id;
    let passed =
        final_hidden.passed && logits.passed && top_1_exact && top_10_overlap >= MIN_TOP_10_OVERLAP;
    Ok(GenerationStepCheck {
        step_index: index,
        phase: generation_phase_name(actual.phase),
        input_token_id: actual.input_token_id,
        cache_position: actual.cache_position,
        cache_len_after: actual.cache_len_after,
        output_token_id: actual.output_token_id,
        expected_output_token_id: reference.generated_token_id,
        output_logit: actual.output_logit,
        device_final_hidden_health,
        vllm_final_hidden_health,
        device_logits_health,
        vllm_logits_health,
        vllm_final_hidden_sha256: reference.final_hidden_sha256.clone(),
        vllm_logits_sha256: reference.logits_sha256.clone(),
        final_hidden,
        logits,
        device_top_10,
        vllm_top_10: reference.top_10.clone(),
        top_1_exact,
        top_10_overlap,
        minimum_top_10_overlap: MIN_TOP_10_OVERLAP,
        started_at_ns: actual.started_at_ns,
        completed_at_ns: actual.completed_at_ns,
        latency_ns: actual.latency_ns,
        passed,
    })
}

fn tensor_gate(reference: &[f32], actual: &[f32]) -> Result<TensorGate, String> {
    let metrics = compare_sq8_correctness(reference, actual)?;
    let passed = metrics.nonfinite_count == 0
        && metrics.relative_l2.is_finite()
        && metrics.relative_l2 <= MAX_RELATIVE_L2
        && metrics.cosine_similarity.is_finite()
        && metrics.cosine_similarity >= MIN_COSINE_SIMILARITY;
    Ok(TensorGate {
        metrics,
        max_relative_l2: MAX_RELATIVE_L2,
        min_cosine_similarity: MIN_COSINE_SIMILARITY,
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
    Ok(TensorHealth {
        elements: values.len(),
        nonfinite,
        minimum,
        maximum,
        max_abs,
        f32_le_sha256: sq8_f32_le_sha256(values)?,
    })
}

fn top_k(values: &[f32], count: usize) -> Result<Vec<TopKEntry>, String> {
    if count == 0 || values.len() < count {
        return Err(format!(
            "top-k requires 0 < count <= values, got count={count} values={}",
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
        .take(count)
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

fn stack_step_record(
    step: &ullm_engine::sq8_generation_runtime::Sq8GenerationStackStepReport,
) -> Result<StackStepRecord, String> {
    Ok(StackStepRecord {
        phase: paged_phase_name(step.phase),
        position: step.position,
        sequence_len: step.sequence_len,
        cache_len: step.cache_len,
        projection_calls: step.projection_calls,
        activation_quantizations: step.activation_quantizations,
        layer_d2d_copies: step.layer_d2d_copies,
        kv_write_calls: step.kv_write_calls,
        paged_attention_calls: step.paged_attention_calls,
        input_d2d_copies: step.input_d2d_copies,
        all_ck: step.all_ck,
        fallback_used: step.fallback_used,
        host_staging_used: step.host_staging_used,
    })
}

fn allocator_record(value: ullm_engine::scheduler::KvBlockAllocatorStats) -> AllocatorStatsRecord {
    AllocatorStatsRecord {
        block_size_tokens: value.block_size_tokens,
        total_blocks: value.total_blocks,
        free_blocks: value.free_blocks,
        allocated_blocks: value.allocated_blocks,
        free_runs: value.free_runs,
        largest_free_run: value.largest_free_run,
    }
}

fn profile_name(value: Sq8LayerExecutionProfile) -> Result<&'static str, String> {
    match value {
        Sq8LayerExecutionProfile::Rdna4W8a8BlockCk => Ok("rdna4_w8a8_block_ck"),
        Sq8LayerExecutionProfile::ReferenceW8a16Block2d => {
            Err("generation report used the reference projection profile".into())
        }
    }
}

fn generation_phase_name(value: Sq8GenerationStepPhase) -> &'static str {
    match value {
        Sq8GenerationStepPhase::Prefill => "prefill",
        Sq8GenerationStepPhase::Decode => "decode",
    }
}

fn paged_phase_name(value: ullm_engine::sq8_stack_runtime::Sq8PagedStackPhase) -> &'static str {
    match value {
        ullm_engine::sq8_stack_runtime::Sq8PagedStackPhase::Prefill => "prefill",
        ullm_engine::sq8_stack_runtime::Sq8PagedStackPhase::Decode => "decode",
    }
}

fn completion_reason_name(value: Sq8GenerationCompletionReason) -> &'static str {
    match value {
        Sq8GenerationCompletionReason::Eos => "eos",
        Sq8GenerationCompletionReason::MaxNewTokens => "max_new_tokens",
    }
}

fn runtime_status_name(value: Sq8GenerationRuntimeStatus) -> &'static str {
    match value {
        Sq8GenerationRuntimeStatus::Ready => "ready",
        Sq8GenerationRuntimeStatus::Running => "running",
        Sq8GenerationRuntimeStatus::Completed => "completed",
        Sq8GenerationRuntimeStatus::Poisoned => "poisoned",
    }
}

fn required_guard_names() -> Vec<String> {
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV)
        .chain(["ULLM_REQUIRE_HIP_TOP1_KERNEL"])
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
            "SQ8 generation validation requires these HIP-only guards to equal 1: {}",
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

fn bf16_values_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn token_ids_sha256(values: &[usize]) -> Result<String, String> {
    if values.is_empty() {
        return Err("cannot hash an empty token ID sequence".into());
    }
    let mut digest = Sha256::new();
    for value in values {
        let token_id =
            u32::try_from(*value).map_err(|_| format!("token ID does not fit u32: {value}"))?;
        digest.update(token_id.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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

fn output_parent(path: &Path) -> Result<&Path, String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("output has no parent: {}", path.display()))?;
    if parent.as_os_str().is_empty() {
        Ok(Path::new("."))
    } else {
        Ok(parent)
    }
}

fn write_json_no_clobber(path: &Path, result: &GenerationGateResult) -> Result<(), String> {
    let mut payload = serde_json::to_vec_pretty(result)
        .map_err(|err| format!("failed to serialize SQ8 generation result: {err}"))?;
    payload.push(b'\n');
    publish_bytes_no_clobber(path, &payload)
}

fn publish_bytes_no_clobber(path: &Path, payload: &[u8]) -> Result<(), String> {
    let parent = output_parent(path)?;
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
