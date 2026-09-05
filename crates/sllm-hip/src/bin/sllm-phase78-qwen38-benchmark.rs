//! Resident, single-request Phase 78 benchmark for the exact Unsloth
//! Qwen3.8-27B NVFP4 artifact.
//!
//! The binary deliberately accepts configuration only through explicit
//! environment variables so the emitted JSON contains a compact, repeatable
//! execution contract. It never enables MTP, sampling, EOS termination, stop
//! strings, batching, or a non-HIP fallback.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    AllocationSnapshot, Backend, ExecutionSessionRequest, KvCacheEncoding, QWEN35_VOCAB_SIZE,
    QwenExecutionAudit, QwenRequestMemoryAudit, QwenResidentModel,
    UNSLOTH_QWEN38_NVFP4_MODEL_SHA256, UNSLOTH_QWEN38_NVFP4_MODEL_SIZE,
    UNSLOTH_QWEN38_NVFP4_REPOSITORY, UNSLOTH_QWEN38_NVFP4_REVISION,
    build_qwen35_unsloth_qwen38_nvfp4_graph, build_qwen38_nvfp4_weight_load_plan, read_model_lock,
    verify_unsloth_qwen38_nvfp4,
};
use sllm_hip::HipBackend;
use tokenizers::Tokenizer;

const MODEL_ENV: &str = "SLLM_PHASE78_MODEL_PATH";
const COMPAT_MODEL_ENV: &str = "SLLM_QWEN38_NVFP4_CACHE";
const TARGET_ENV: &str = "SLLM_PHASE78_TARGET";
const DEVICE_ENV: &str = "SLLM_PHASE78_DEVICE";
const WARMUPS_ENV: &str = "SLLM_PHASE78_WARMUPS";
const MEASURED_ENV: &str = "SLLM_PHASE78_MEASURED";
const ROWS_ENV: &str = "SLLM_PHASE78_ROWS";
const CHUNK_CAPACITY_ENV: &str = "SLLM_PHASE78_CHUNK_CAPACITY";

const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_MEASURED: usize = 10;
const MAX_REPETITIONS: usize = 100;
const PROMPT_CAPACITY: usize = 9_435;
const STATE_CAPACITY: u64 = 9_563;
const DEFAULT_CHUNK_CAPACITY: u64 = 1_024;
const MAX_CHUNK_CAPACITY: u64 = 8_192;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(600);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);
const FIXTURE_SHA256: &str =
    "sha256:50ae5d562b673cf68ea58ee93989356bdb5955693d47b1756331da3988081b80";
// The Qwen3.8 source artifact importer currently verifies the model and MTP
// safetensors but does not expose frontend assets. Keep the tokenizer's
// immutable artifact identity at this benchmark boundary until that API is
// available; decoding never participates in measured request intervals.
const QWEN38_TOKENIZER_SIZE_BYTES: u64 = 19_989_325;
const QWEN38_TOKENIZER_SHA256: &str =
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";
const QWEN38_TOKENIZER_VOCAB_SIZE: usize = 248_077;
const QWEN38_TOKENIZER_VOCAB_SPAN: u32 = 248_077;

const FIXED_PREFIX: [i32; 17] = [
    2, 106, 1_645, 108, 9_259, 236_776, 563, 107, 17, 23, 42, 255, 256, 257, 4_097, 65_537, 248_319,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RowSpec {
    prompt_tokens: usize,
    output_tokens: usize,
}

const ROWS: [RowSpec; 4] = [
    RowSpec {
        prompt_tokens: 17,
        output_tokens: 17,
    },
    RowSpec {
        prompt_tokens: 512,
        output_tokens: 32,
    },
    RowSpec {
        prompt_tokens: 2_048,
        output_tokens: 128,
    },
    RowSpec {
        prompt_tokens: 9_435,
        output_tokens: 128,
    },
];

#[derive(Debug)]
struct Config {
    target: String,
    device_index: u32,
    model_root: PathBuf,
    model_env: &'static str,
    warmups: usize,
    measured: usize,
    chunk_capacity: u64,
    rows: Vec<RowSpec>,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    benchmark_mode: &'static str,
    target: String,
    device_index: u32,
    model: ModelReport,
    protocol: ProtocolReport,
    fixture: FixtureReport,
    tokenizer: TokenizerReport,
    is_phase78_final: bool,
    repetitions: RepetitionReport,
    selector_environment: BTreeMap<String, Option<String>>,
    unsupported: Vec<UnsupportedReport>,
    setup: SetupReport,
    resident_ready_memory: AllocationReport,
    rows: Vec<RowReport>,
    cleanup: CleanupReport,
}

#[derive(Serialize)]
struct ModelReport {
    root: String,
    path_environment: &'static str,
    repository: &'static str,
    revision: &'static str,
    model_bytes: u64,
    model_sha256: &'static str,
}

#[derive(Serialize)]
struct ProtocolReport {
    active_requests: u32,
    parallel_requests: u32,
    batching: &'static str,
    kv_cache: &'static str,
    state_capacity_tokens: u64,
    prefill_chunk_capacity_tokens: u64,
    mtp: &'static str,
    generation: &'static str,
    eos_termination: bool,
    stop_sequences: bool,
    termination: &'static str,
    output_accounting: &'static str,
}

#[derive(Serialize)]
struct FixtureReport {
    schema_version: &'static str,
    construction: &'static str,
    total_tokens: usize,
    token_encoding: &'static str,
    sha256: String,
    fixed_prefix_17: [i32; 17],
}

#[derive(Clone, Serialize)]
struct TokenizerReport {
    schema_version: &'static str,
    source_file: &'static str,
    size_bytes: u64,
    sha256: String,
    tokenizer_vocab_size: usize,
    tokenizer_vocab_span: u32,
    model_vocab_size: usize,
    decode_mode: &'static str,
}

struct LockedTokenizer {
    tokenizer: Tokenizer,
    report: TokenizerReport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DecodedOutputReport {
    visible_tokens_sha256: String,
    visible_token_count: usize,
    generated_text_sha256: String,
    decoded_token_count: usize,
}

#[derive(Serialize)]
struct RepetitionReport {
    warmups_per_row: usize,
    measured_per_row: usize,
    measured_statistics: &'static str,
}

#[derive(Serialize)]
struct UnsupportedReport {
    field: &'static str,
    status: &'static str,
    reason: &'static str,
}

#[derive(Serialize)]
struct SetupReport {
    verify_ns: u128,
    graph_and_plan_ns: u128,
    resident_load_ns: u128,
    available_device_memory_bytes_before_load: Option<u64>,
    model_fingerprint: String,
    weight_plan_digest: String,
}

#[derive(Clone, Serialize)]
struct AllocationBucketReport {
    current_bytes: u64,
    high_water_bytes: u64,
}

#[derive(Clone, Serialize)]
struct AllocationReport {
    model_resident: AllocationBucketReport,
    request_state: AllocationBucketReport,
    workspace: AllocationBucketReport,
    current_bytes: u64,
    high_water_bytes: u64,
    poisoned: bool,
    high_water_scope: &'static str,
}

#[derive(Serialize)]
struct RowReport {
    prompt_tokens: usize,
    output_tokens: usize,
    prompt_prefix_sha256: String,
    deterministic_generated_tokens: bool,
    generated_tokens_sha256: String,
    visible_tokens_sha256: String,
    visible_token_count: usize,
    generated_text_sha256: String,
    decoded_token_count: usize,
    measured_summary: TimingSummary,
    runs: Vec<RunReport>,
}

#[derive(Serialize)]
struct RunReport {
    sample_kind: &'static str,
    sample_index: usize,
    prefill_output_rows: usize,
    decode_transition_count: usize,
    timing: TimingReport,
    generated_tokens: Vec<i32>,
    generated_tokens_sha256: String,
    visible_tokens_sha256: String,
    visible_token_count: usize,
    generated_text_sha256: String,
    decoded_token_count: usize,
    stop_reason: &'static str,
    audit: AuditReport,
    request_memory: RequestMemoryReport,
    allocation_before_request: AllocationReport,
    allocation_while_request_alive: AllocationReport,
    allocation_after_request_drop: AllocationReport,
}

#[derive(Clone, Serialize)]
struct TimingReport {
    request_setup_ns: u128,
    prefill_ns: u128,
    ttft_ns: u128,
    decode_ns: u128,
    e2e_ns: u128,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    output_tokens_per_e2e_second: f64,
    total_tokens_per_e2e_second: f64,
    tpot_ms: f64,
}

#[derive(Serialize)]
struct TimingSummary {
    measured_samples: usize,
    request_setup_ms: MedianMad,
    prefill_ms: MedianMad,
    ttft_ms: MedianMad,
    decode_ms: MedianMad,
    e2e_ms: MedianMad,
    prefill_tokens_per_second: MedianMad,
    decode_tokens_per_second: MedianMad,
    tpot_ms: MedianMad,
}

#[derive(Serialize)]
struct MedianMad {
    median: f64,
    mad: f64,
}

#[derive(Serialize)]
struct AuditReport {
    selected_backend: &'static str,
    target: String,
    completion_mode: &'static str,
    terminal_logit_non_finite_count: u64,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
    all_dispatches_hip: bool,
    segment_count: u64,
    boundary_count: u64,
    physical_queue_fence_count: u64,
    graph_replay_count: u64,
    graph_span_count: u64,
    graph_capture_kernel_node_count: u64,
    kv_append_attention_chain_count: u64,
    selected_kernel_counts: Vec<KernelCountReport>,
    complete_identity_map: &'static str,
}

#[derive(Serialize)]
struct KernelCountReport {
    kernel_id: u32,
    kernel_symbol: &'static str,
    dispatch_count: u64,
}

#[derive(Serialize)]
struct RequestMemoryReport {
    kv_layers: usize,
    kv_logical_capacity_tokens: Option<u64>,
    kv_observed_length_tokens: Option<u64>,
    kv_memory_kind: Option<String>,
    kv_physical_page_bytes: Option<u64>,
    kv_tokens_per_page: Option<u64>,
    kv_mapped_token_capacity: Option<u64>,
    kv_committed_bytes_per_plane: Option<u64>,
    kv_committed_bytes_all_layers_and_planes: u64,
    linear_attention_layers: usize,
    linear_attention_capacity_tokens: Option<u64>,
    linear_attention_observed_length_tokens: Option<u64>,
}

#[derive(Serialize)]
struct CleanupReport {
    allocation_after_resident_drop_before_shutdown: AllocationReport,
    retryable_cleanup: usize,
    durable_quarantine: usize,
    zero: bool,
}

#[derive(Serialize)]
struct FailureReport {
    schema_version: &'static str,
    state: &'static str,
    error: String,
}

fn main() -> ExitCode {
    if env::args_os().len() != 1 {
        return emit_failure("this benchmark accepts environment variables only".to_owned());
    }
    match Config::from_env().and_then(run) {
        Ok(report) => {
            let passed = report.state == "PASS";
            if let Err(error) = emit_json(io::stdout().lock(), &report) {
                eprintln!("Phase 78 benchmark JSON serialization failed: {error}");
                return ExitCode::from(2);
            }
            if passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => emit_failure(error),
    }
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let target = required_env(TARGET_ENV)?;
        if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
            return Err(format!("{TARGET_ENV} must be gfx1030 or gfx1201"));
        }
        let device_index = parse_env_or::<u32>(DEVICE_ENV, None)?;
        let (model_root, model_env) = match env::var_os(MODEL_ENV) {
            Some(path) => (PathBuf::from(path), MODEL_ENV),
            None => (
                PathBuf::from(
                    env::var_os(COMPAT_MODEL_ENV)
                        .ok_or_else(|| format!("{MODEL_ENV} or {COMPAT_MODEL_ENV} is required"))?,
                ),
                COMPAT_MODEL_ENV,
            ),
        };
        let warmups = parse_env_or(WARMUPS_ENV, Some(DEFAULT_WARMUPS))?;
        let measured = parse_env_or(MEASURED_ENV, Some(DEFAULT_MEASURED))?;
        let chunk_capacity = parse_env_or(CHUNK_CAPACITY_ENV, Some(DEFAULT_CHUNK_CAPACITY))?;
        if warmups > MAX_REPETITIONS {
            return Err(format!("{WARMUPS_ENV} must not exceed {MAX_REPETITIONS}"));
        }
        if measured == 0 || measured > MAX_REPETITIONS {
            return Err(format!("{MEASURED_ENV} must be in 1..={MAX_REPETITIONS}"));
        }
        if !(512..=MAX_CHUNK_CAPACITY).contains(&chunk_capacity)
            || !chunk_capacity.is_power_of_two()
        {
            return Err(format!(
                "{CHUNK_CAPACITY_ENV} must be a power of two in 512..={MAX_CHUNK_CAPACITY}"
            ));
        }
        let rows = match env::var(ROWS_ENV) {
            Ok(text) => parse_rows(&text)?,
            Err(env::VarError::NotPresent) => ROWS.to_vec(),
            Err(error) => return Err(format!("cannot read {ROWS_ENV}: {error}")),
        };
        Ok(Self {
            target,
            device_index,
            model_root,
            model_env,
            warmups,
            measured,
            chunk_capacity,
            rows,
        })
    }

    fn mode(&self) -> &'static str {
        match (self.warmups, self.measured) {
            (3, 10) => "phase78-final-3-warmup-10-measured",
            (1, 3) => "phase78-exploration-1-warmup-3-measured",
            _ => "custom-explicit-repetitions",
        }
    }

    fn is_phase78_final(&self) -> bool {
        self.warmups == DEFAULT_WARMUPS
            && self.measured == DEFAULT_MEASURED
            && self.rows.as_slice() == ROWS
    }
}

fn run(config: Config) -> Result<Report, String> {
    let prompt_fixture = fixed_prompt_fixture()?;
    let verify_started = Instant::now();
    let artifact = Arc::new(
        verify_unsloth_qwen38_nvfp4(&config.model_root).map_err(|error| error.to_string())?,
    );
    let verify_ns = verify_started.elapsed().as_nanos();
    let locked_tokenizer = load_locked_tokenizer(artifact.root())?;

    let graph_started = Instant::now();
    let lock_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/models/locks/qwen3.5-27b-bf16.json");
    let lock = read_model_lock(&lock_path).map_err(|error| error.to_string())?;
    let plan =
        build_qwen38_nvfp4_weight_load_plan(&lock, &artifact).map_err(|error| error.to_string())?;
    let plan_digest = plan.digest_hex();
    let graph = build_qwen35_unsloth_qwen38_nvfp4_graph(
        &lock,
        &plan,
        &artifact,
        config.chunk_capacity,
        STATE_CAPACITY,
        KvCacheEncoding::Fp16,
    )
    .map_err(|error| error.to_string())?;
    let graph_and_plan_ns = graph_started.elapsed().as_nanos();

    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(config.device_index, config.target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let available_device_memory_bytes_before_load = session
        .available_memory_bytes()
        .map_err(|error| error.to_string())?;

    let operation = (|| -> Result<_, String> {
        let load_started = Instant::now();
        let resident = QwenResidentModel::new_unsloth_qwen38_nvfp4(
            Arc::clone(&session),
            graph.clone(),
            plan,
            Arc::clone(&artifact),
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("model provisioning failed: {error}"))?;
        let resident_load_ns = load_started.elapsed().as_nanos();
        let resident_ready_memory = allocation_report(resident.memory_snapshot());
        if resident_ready_memory.poisoned || resident_ready_memory.model_resident.current_bytes == 0
        {
            return Err("resident model allocation snapshot is invalid".to_owned());
        }
        let model_fingerprint = resident.model_fingerprint().to_owned();
        let mut row_reports = Vec::with_capacity(config.rows.len());
        for row in &config.rows {
            row_reports.push(run_row(
                &session,
                &resident,
                &graph,
                &prompt_fixture,
                *row,
                config.warmups,
                config.measured,
                &config.target,
                &locked_tokenizer.tokenizer,
            )?);
        }
        drop(resident);
        Ok((
            row_reports,
            resident_ready_memory,
            resident_load_ns,
            model_fingerprint,
        ))
    })();

    let allocation_before_shutdown = allocation_report(session.memory_snapshot());
    let shutdown = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("session shutdown failed: {error}"));
    let (row_reports, resident_ready_memory, resident_load_ns, model_fingerprint) = match operation
    {
        Ok(report) => report,
        Err(error) => {
            let cleanup = shutdown
                .map(|report| {
                    format!(
                        "current_bytes={} poisoned={} retryable_cleanup={} durable_quarantine={}",
                        allocation_before_shutdown.current_bytes,
                        allocation_before_shutdown.poisoned,
                        report.retryable_cleanup,
                        report.durable_quarantine
                    )
                })
                .unwrap_or_else(|cleanup_error| cleanup_error);
            return Err(format!("{error}; post-error cleanup: {cleanup}"));
        }
    };
    let shutdown = shutdown?;
    let cleanup_zero = allocation_before_shutdown.current_bytes == 0
        && !allocation_before_shutdown.poisoned
        && shutdown.retryable_cleanup == 0
        && shutdown.durable_quarantine == 0;
    let state = if cleanup_zero { "PASS" } else { "FAIL" };
    let is_phase78_final = config.is_phase78_final();

    Ok(Report {
        schema_version: "phase78-qwen38-resident-benchmark-v3",
        state,
        benchmark_mode: config.mode(),
        target: config.target,
        device_index: config.device_index,
        model: ModelReport {
            root: config.model_root.display().to_string(),
            path_environment: config.model_env,
            repository: UNSLOTH_QWEN38_NVFP4_REPOSITORY,
            revision: UNSLOTH_QWEN38_NVFP4_REVISION,
            model_bytes: UNSLOTH_QWEN38_NVFP4_MODEL_SIZE,
            model_sha256: UNSLOTH_QWEN38_NVFP4_MODEL_SHA256,
        },
        protocol: ProtocolReport {
            active_requests: 1,
            parallel_requests: 1,
            batching: "disabled; rows and repetitions execute serially with one fresh request",
            kv_cache: "FP16",
            state_capacity_tokens: STATE_CAPACITY,
            prefill_chunk_capacity_tokens: config.chunk_capacity,
            mtp: "disabled; only non-MTP graph and prefill/decode APIs are called",
            generation: "greedy default device Argmax; no sampler or logits readback",
            eos_termination: false,
            stop_sequences: false,
            termination: "fixed total output-token budget; generated EOS-like IDs are not inspected",
            output_accounting: "generated_tokens starts with the terminal prefill Argmax, followed by output_tokens-1 decode() results; TPOT and decode throughput count only those decode transitions",
        },
        fixture: FixtureReport {
            schema_version: "phase78-qwen38-fixed-token-fixture-v1",
            construction: "tokens[0..17]=fixed_prefix_17; tokens[i>=17]=(i*7919+17)%248320",
            total_tokens: prompt_fixture.len(),
            token_encoding: "signed-i32 token IDs; SHA-256 over concatenated little-endian i32",
            sha256: hash_tokens(&prompt_fixture),
            fixed_prefix_17: FIXED_PREFIX,
        },
        tokenizer: locked_tokenizer.report,
        is_phase78_final,
        repetitions: RepetitionReport {
            warmups_per_row: config.warmups,
            measured_per_row: config.measured,
            measured_statistics: "median and median absolute deviation over measured runs only",
        },
        selector_environment: selector_environment(),
        unsupported: unsupported_reports(),
        setup: SetupReport {
            verify_ns,
            graph_and_plan_ns,
            resident_load_ns,
            available_device_memory_bytes_before_load,
            model_fingerprint,
            weight_plan_digest: plan_digest,
        },
        resident_ready_memory,
        rows: row_reports,
        cleanup: CleanupReport {
            allocation_after_resident_drop_before_shutdown: allocation_before_shutdown,
            retryable_cleanup: shutdown.retryable_cleanup,
            durable_quarantine: shutdown.durable_quarantine,
            zero: cleanup_zero,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn run_row(
    session: &Arc<sllm_core::ExecutionSession>,
    resident: &QwenResidentModel,
    graph: &sllm_core::QwenGraph,
    fixture: &[i32],
    row: RowSpec,
    warmups: usize,
    measured: usize,
    target: &str,
    tokenizer: &Tokenizer,
) -> Result<RowReport, String> {
    let prompt = fixture
        .get(..row.prompt_tokens)
        .ok_or_else(|| format!("fixture is shorter than {} tokens", row.prompt_tokens))?;
    let mut runs = Vec::with_capacity(warmups + measured);
    let mut expected_tokens: Option<Vec<i32>> = None;
    for (sample_kind, count) in [("warmup", warmups), ("measured", measured)] {
        for sample_index in 0..count {
            let report = run_one(
                session,
                resident,
                graph,
                prompt,
                row.output_tokens,
                sample_kind,
                sample_index,
                target,
                tokenizer,
            )?;
            if let Some(expected) = &expected_tokens {
                if expected != &report.generated_tokens {
                    return Err(format!(
                        "generated tokens changed for {}/{} on {sample_kind} sample {sample_index}",
                        row.prompt_tokens, row.output_tokens
                    ));
                }
            } else {
                expected_tokens = Some(report.generated_tokens.clone());
            }
            runs.push(report);
        }
    }
    let expected_tokens = expected_tokens.ok_or_else(|| "row has no benchmark runs".to_owned())?;
    let expected_run = runs
        .first()
        .ok_or_else(|| "row has no benchmark runs".to_owned())?;
    if runs.iter().any(|run| {
        run.visible_tokens_sha256 != expected_run.visible_tokens_sha256
            || run.visible_token_count != expected_run.visible_token_count
            || run.generated_text_sha256 != expected_run.generated_text_sha256
            || run.decoded_token_count != expected_run.decoded_token_count
    }) {
        return Err(format!(
            "decoded output changed for {}/{} across repetitions",
            row.prompt_tokens, row.output_tokens
        ));
    }
    let measured_timings = runs
        .iter()
        .filter(|run| run.sample_kind == "measured")
        .map(|run| run.timing.clone())
        .collect::<Vec<_>>();
    Ok(RowReport {
        prompt_tokens: row.prompt_tokens,
        output_tokens: row.output_tokens,
        prompt_prefix_sha256: hash_tokens(prompt),
        deterministic_generated_tokens: true,
        generated_tokens_sha256: hash_tokens(&expected_tokens),
        visible_tokens_sha256: expected_run.visible_tokens_sha256.clone(),
        visible_token_count: expected_run.visible_token_count,
        generated_text_sha256: expected_run.generated_text_sha256.clone(),
        decoded_token_count: expected_run.decoded_token_count,
        measured_summary: summarize_timings(&measured_timings)?,
        runs,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    session: &Arc<sllm_core::ExecutionSession>,
    resident: &QwenResidentModel,
    graph: &sllm_core::QwenGraph,
    prompt: &[i32],
    output_tokens: usize,
    sample_kind: &'static str,
    sample_index: usize,
    target: &str,
    tokenizer: &Tokenizer,
) -> Result<RunReport, String> {
    let before_request = allocation_report(session.memory_snapshot());
    // Phase 78 compares time-to-first-token and request E2E after the model is
    // resident.  Start both clocks before request-local state/queue creation;
    // keep the narrower prefill interval as a separate kernel/runtime metric.
    let e2e_started = Instant::now();
    let mut request = resident
        .new_request(graph.clone())
        .map_err(|error| format!("request creation failed: {error}"))?;
    let request_setup_elapsed = e2e_started.elapsed();
    let prefill_started = Instant::now();
    let prefill = request
        .prefill(prompt)
        .map_err(|error| format!("prefill failed: {error}"))?;
    let prefill_elapsed = prefill_started.elapsed();
    let ttft_elapsed = e2e_started.elapsed();
    if !matches!(prefill.token_ids().len(), 1) && prefill.token_ids().len() != prompt.len() {
        return Err(format!(
            "prefill returned unsupported row count {} for {} prompt tokens",
            prefill.token_ids().len(),
            prompt.len()
        ));
    }
    if prefill.selection().is_some() || prefill.last_logits().is_some() {
        return Err("prefill left the default Argmax/no-logits route".to_owned());
    }
    let mut current = *prefill
        .token_ids()
        .last()
        .ok_or_else(|| "prefill produced no terminal token".to_owned())?;
    validate_token(current)?;

    let mut generated = Vec::with_capacity(output_tokens);
    generated.push(current);
    let decode_started = Instant::now();
    for step in 1..output_tokens {
        let output = request
            .decode(current)
            .map_err(|error| format!("decode step {step} failed: {error}"))?;
        if output.token_ids().len() != 1
            || output.selection().is_some()
            || output.last_logits().is_some()
        {
            return Err(format!(
                "decode step {step} left the one-row default Argmax/no-logits route"
            ));
        }
        current = output.token_ids()[0];
        validate_token(current)?;
        generated.push(current);
    }
    let decode_elapsed = decode_started.elapsed();
    let e2e_elapsed = e2e_started.elapsed();
    if request_setup_elapsed.is_zero()
        || prefill_elapsed.is_zero()
        || ttft_elapsed.is_zero()
        || decode_elapsed.is_zero()
        || e2e_elapsed.is_zero()
    {
        return Err("a benchmark timing interval was zero".to_owned());
    }

    // Decode only after all request timing intervals have been captured. The
    // tokenizer is loaded from the verified artifact before the first request,
    // while this per-output detokenization remains outside measured time.
    let decoded = decoded_output_report(tokenizer, &generated)?;

    let audit = request
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    if audit.selected_backend() != "hip"
        || audit.target() != target
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
        || audit.fallback_used()
        || !audit.all_dispatches_hip()
    {
        return Err(format!("request dispatch audit is not HIP-only: {audit:?}"));
    }
    let audit = audit_report(&audit);
    let request_memory = request
        .memory_audit_snapshot()
        .map_err(|error| format!("request memory audit failed: {error}"))?;
    let request_memory = request_memory_report(&request_memory)?;
    let while_request_alive = allocation_report(session.memory_snapshot());
    drop(request);
    let after_request_drop = allocation_report(session.memory_snapshot());
    let timing = timing_report(
        request_setup_elapsed,
        prefill_elapsed,
        ttft_elapsed,
        decode_elapsed,
        e2e_elapsed,
        prompt.len(),
        output_tokens,
    );
    Ok(RunReport {
        sample_kind,
        sample_index,
        prefill_output_rows: prefill.token_ids().len(),
        decode_transition_count: generated.len() - 1,
        timing,
        generated_tokens_sha256: hash_tokens(&generated),
        visible_tokens_sha256: decoded.visible_tokens_sha256,
        visible_token_count: decoded.visible_token_count,
        generated_text_sha256: decoded.generated_text_sha256,
        decoded_token_count: decoded.decoded_token_count,
        generated_tokens: generated,
        stop_reason: "length",
        audit,
        request_memory,
        allocation_before_request: before_request,
        allocation_while_request_alive: while_request_alive,
        allocation_after_request_drop: after_request_drop,
    })
}

fn timing_report(
    request_setup: Duration,
    prefill: Duration,
    ttft: Duration,
    decode: Duration,
    e2e: Duration,
    prompt_tokens: usize,
    output_tokens: usize,
) -> TimingReport {
    let prefill_seconds = prefill.as_secs_f64();
    let decode_seconds = decode.as_secs_f64();
    let e2e_seconds = e2e.as_secs_f64();
    TimingReport {
        request_setup_ns: request_setup.as_nanos(),
        prefill_ns: prefill.as_nanos(),
        ttft_ns: ttft.as_nanos(),
        decode_ns: decode.as_nanos(),
        e2e_ns: e2e.as_nanos(),
        prefill_tokens_per_second: prompt_tokens as f64 / prefill_seconds,
        decode_tokens_per_second: (output_tokens - 1) as f64 / decode_seconds,
        output_tokens_per_e2e_second: output_tokens as f64 / e2e_seconds,
        total_tokens_per_e2e_second: (prompt_tokens + output_tokens) as f64 / e2e_seconds,
        tpot_ms: decode_seconds * 1_000.0 / (output_tokens - 1) as f64,
    }
}

fn summarize_timings(samples: &[TimingReport]) -> Result<TimingSummary, String> {
    if samples.is_empty() {
        return Err("measured timing set is empty".to_owned());
    }
    Ok(TimingSummary {
        measured_samples: samples.len(),
        request_setup_ms: median_mad(
            samples
                .iter()
                .map(|sample| sample.request_setup_ns as f64 / 1_000_000.0)
                .collect(),
        ),
        prefill_ms: median_mad(
            samples
                .iter()
                .map(|sample| sample.prefill_ns as f64 / 1_000_000.0)
                .collect(),
        ),
        ttft_ms: median_mad(
            samples
                .iter()
                .map(|sample| sample.ttft_ns as f64 / 1_000_000.0)
                .collect(),
        ),
        decode_ms: median_mad(
            samples
                .iter()
                .map(|sample| sample.decode_ns as f64 / 1_000_000.0)
                .collect(),
        ),
        e2e_ms: median_mad(
            samples
                .iter()
                .map(|sample| sample.e2e_ns as f64 / 1_000_000.0)
                .collect(),
        ),
        prefill_tokens_per_second: median_mad(
            samples
                .iter()
                .map(|sample| sample.prefill_tokens_per_second)
                .collect(),
        ),
        decode_tokens_per_second: median_mad(
            samples
                .iter()
                .map(|sample| sample.decode_tokens_per_second)
                .collect(),
        ),
        tpot_ms: median_mad(samples.iter().map(|sample| sample.tpot_ms).collect()),
    })
}

fn median_mad(mut values: Vec<f64>) -> MedianMad {
    let median_value = median(&mut values);
    let mut deviations = values
        .into_iter()
        .map(|value| (value - median_value).abs())
        .collect::<Vec<_>>();
    MedianMad {
        median: median_value,
        mad: median(&mut deviations),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn allocation_report(snapshot: AllocationSnapshot) -> AllocationReport {
    AllocationReport {
        model_resident: AllocationBucketReport {
            current_bytes: snapshot.model_resident().current_bytes(),
            high_water_bytes: snapshot.model_resident().high_water_bytes(),
        },
        request_state: AllocationBucketReport {
            current_bytes: snapshot.request_state().current_bytes(),
            high_water_bytes: snapshot.request_state().high_water_bytes(),
        },
        workspace: AllocationBucketReport {
            current_bytes: snapshot.workspace().current_bytes(),
            high_water_bytes: snapshot.workspace().high_water_bytes(),
        },
        current_bytes: snapshot.current_bytes(),
        high_water_bytes: snapshot.high_water_bytes(),
        poisoned: snapshot.poisoned(),
        high_water_scope: "cumulative execution-session allocation accounting",
    }
}

fn request_memory_report(audit: &QwenRequestMemoryAudit) -> Result<RequestMemoryReport, String> {
    let first = audit.kv_layers().first().copied();
    let physical = first.map(|layer| layer.physical());
    Ok(RequestMemoryReport {
        kv_layers: audit.kv_layers().len(),
        kv_logical_capacity_tokens: first.map(|layer| layer.logical_capacity_tokens()),
        kv_observed_length_tokens: first.map(|layer| layer.observed_length_tokens()),
        kv_memory_kind: physical.map(|value| format!("{:?}", value.memory_kind())),
        kv_physical_page_bytes: physical.map(|value| value.physical_page_bytes()),
        kv_tokens_per_page: physical.map(|value| value.tokens_per_page()),
        kv_mapped_token_capacity: physical.map(|value| value.mapped_token_capacity()),
        kv_committed_bytes_per_plane: physical.map(|value| value.committed_bytes_per_plane()),
        kv_committed_bytes_all_layers_and_planes: audit
            .committed_kv_bytes()
            .map_err(|error| error.to_string())?,
        linear_attention_layers: audit.linear_attention_layers(),
        linear_attention_capacity_tokens: audit.linear_attention_capacity_tokens(),
        linear_attention_observed_length_tokens: audit.linear_attention_observed_length_tokens(),
    })
}

fn audit_report(audit: &QwenExecutionAudit) -> AuditReport {
    const SELECTED_KERNELS: [(u32, &str); 32] = [
        (5, "matmul.fp8.outer.hipblaslt.v1"),
        (6, "matmul.fp8.outer.emulation.v1"),
        (11, "matmul.nvfp4.w4a4.block16.baseline.v1"),
        (58, "matmul.nvfp4.w4a4.block16.decode.v1"),
        (59, "matmul.nvfp4.w4a4.block16.prefill.row8_tiled256.v1"),
        (60, "matmul.fp8.outer.prefill.tiled16.v1"),
        (
            61,
            "matmul.nvfp4.w4a4.block16.prefill.row8_col8_tiled256.v1",
        ),
        (62, "matmul.nvfp4.w4a4.block16.prefill.dp4a64x64.v1"),
        (63, "matmul.fp8.outer.prefill.gfx1030.half2.128x64.v1"),
        (64, "matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x64.v1"),
        (65, "matmul.nvfp4.w4a4.decode.columns128.v1"),
        (66, "matmul.fp8.outer.decode.gfx1030.half2.wave4col32.v1"),
        (67, "matmul.nvfp4.w4a4.decode.dp4a.wave4col32.v1"),
        (68, "matmul.fp8.outer.decode.gfx1030.dword8.wave4col32.v1"),
        (
            69,
            "matmul.nvfp4.w4a4.prefill.gfx1201.wmma_f16scale128x64.v1",
        ),
        (70, "matmul.fp8.outer.prefill.gfx1030.f16_staging.v1"),
        (71, "matmul.fp8.outer.prefill.gfx1030.half2.64x64.v1"),
        (72, "matmul.nvfp4.w4a4.prefill.gfx1201.f16_staging.v1"),
        (
            73,
            "matmul.nvfp4.w4a4.decode.dp4a.activation_shared.wave4col32.v1",
        ),
        (74, "causal_attention.prefill.gfx1201_rocblas_gqa6_f32.v1"),
        (
            75,
            "matmul.fp8.outer.decode.gfx1030.activation_shared.wave4col32.v1",
        ),
        (
            76,
            "matmul.fp8.outer.decode.gfx1030.activation_shared.wave8col64.v1",
        ),
        (
            77,
            "causal_attention.prefill.gfx1201_rocblas_gqa6_f16_tail.v1",
        ),
        (78, "causal_attention.decode.gqa6_split_p128.fp16.v1"),
        (79, "linear_attention.gdn.row32_lds.v1"),
        (80, "matmul.nvfp4.w4a4.block16.prefill.dp4a64x64_k128.v1"),
        (81, "matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x32.v1"),
        (82, "matmul.fp8.outer.decode.gfx1030.lds_lut.wave4col32.v1"),
        (83, "matmul.nvfp4.w4a4.prefill.gfx1201.fp8_staging.v1"),
        (84, "matmul.nvfp4.w4a4.decode.scale_lut.v1"),
        (85, "matmul.fp8.outer.prefill.gfx1030.lds_lut.64x64.v1"),
        (86, "matmul.fp8.outer.prefill.gfx1030.f16_tile.v1"),
    ];
    AuditReport {
        selected_backend: audit.selected_backend(),
        target: audit.target().to_owned(),
        completion_mode: if audit.request_local_deferred_completion() {
            "deferred-request-local"
        } else {
            "profiled"
        },
        // Every terminal row goes through the device Argmax reduction.  That
        // kernel emits -1 if it observes any NaN or infinity; run_one rejects
        // the sentinel before this report can be constructed.
        terminal_logit_non_finite_count: 0,
        submission_count: audit.submission_count(),
        kernel_dispatch_count: audit.kernel_dispatch_count(),
        fallback_used: audit.fallback_used(),
        all_dispatches_hip: audit.all_dispatches_hip(),
        segment_count: audit.segment_count(),
        boundary_count: audit.boundary_count(),
        physical_queue_fence_count: audit.physical_queue_fence_count(),
        graph_replay_count: audit.graph_replay_count(),
        graph_span_count: audit.graph_span_count(),
        graph_capture_kernel_node_count: audit.graph_capture_kernel_node_count(),
        kv_append_attention_chain_count: audit.kv_append_attention_chain_count(),
        selected_kernel_counts: SELECTED_KERNELS
            .into_iter()
            .map(|(kernel_id, kernel_symbol)| KernelCountReport {
                kernel_id,
                kernel_symbol,
                dispatch_count: audit.kernel_dispatch_count_for(kernel_id, kernel_symbol),
            })
            .collect(),
        complete_identity_map: "unsupported: public audit exposes exact lookup but not identity iteration",
    }
}

fn load_locked_tokenizer(root: &std::path::Path) -> Result<LockedTokenizer, String> {
    let path = root.join("tokenizer.json");
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("stat verified tokenizer asset: {error}"))?;
    if !metadata.file_type().is_file() {
        return Err("verified tokenizer asset is not a regular file".to_owned());
    }
    if metadata.len() != QWEN38_TOKENIZER_SIZE_BYTES {
        return Err(format!(
            "verified tokenizer asset size changed: expected={} actual={}",
            QWEN38_TOKENIZER_SIZE_BYTES,
            metadata.len()
        ));
    }
    let bytes =
        fs::read(&path).map_err(|error| format!("read verified tokenizer asset: {error}"))?;
    if bytes.len() as u64 != QWEN38_TOKENIZER_SIZE_BYTES {
        return Err("verified tokenizer asset changed while it was read".to_owned());
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != QWEN38_TOKENIZER_SHA256 {
        return Err(format!(
            "verified tokenizer asset digest changed: expected={} actual={digest}",
            QWEN38_TOKENIZER_SHA256
        ));
    }
    let tokenizer = Tokenizer::from_bytes(&bytes)
        .map_err(|error| format!("verified tokenizer JSON is invalid: {error}"))?;
    let vocabulary = tokenizer.get_vocab(true);
    let vocabulary_span = vocabulary
        .values()
        .copied()
        .max()
        .map_or(0_u32, |id| id.saturating_add(1));
    if vocabulary.len() != QWEN38_TOKENIZER_VOCAB_SIZE
        || vocabulary_span != QWEN38_TOKENIZER_VOCAB_SPAN
        || vocabulary_span as usize > QWEN35_VOCAB_SIZE
    {
        return Err(format!(
            "verified tokenizer vocabulary changed: size={} span={} model_capacity={}",
            vocabulary.len(),
            vocabulary_span,
            QWEN35_VOCAB_SIZE
        ));
    }
    Ok(LockedTokenizer {
        tokenizer,
        report: TokenizerReport {
            schema_version: "phase78-qwen38-locked-tokenizer-v1",
            source_file: "tokenizer.json",
            size_bytes: QWEN38_TOKENIZER_SIZE_BYTES,
            sha256: format!("sha256:{QWEN38_TOKENIZER_SHA256}"),
            tokenizer_vocab_size: vocabulary.len(),
            tokenizer_vocab_span: vocabulary_span,
            model_vocab_size: QWEN35_VOCAB_SIZE,
            decode_mode: "preserve-special-tokens; fixed-budget visible IDs are not stop-filtered",
        },
    })
}

fn decoded_output_report(
    tokenizer: &Tokenizer,
    generated: &[i32],
) -> Result<DecodedOutputReport, String> {
    let ids = generated
        .iter()
        .copied()
        .map(|token| {
            u32::try_from(token).map_err(|_| format!("generated token cannot be decoded: {token}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for id in &ids {
        if tokenizer.id_to_token(*id).is_none() {
            return Err(format!(
                "generated token is absent from the locked tokenizer: {id}"
            ));
        }
    }
    // The Phase 78 protocol disables EOS and stop handling, so every generated
    // ID is visible even when the tokenizer marks it as a special token. This
    // intentionally mirrors the generated/visible token contract rather than
    // silently dropping special IDs during text decoding.
    let text = tokenizer
        .decode(&ids, false)
        .map_err(|error| format!("decode generated tokens: {error}"))?;
    Ok(DecodedOutputReport {
        visible_tokens_sha256: hash_tokens(generated),
        visible_token_count: generated.len(),
        generated_text_sha256: hash_text(&text),
        decoded_token_count: ids.len(),
    })
}

fn fixed_prompt_fixture() -> Result<Vec<i32>, String> {
    let mut tokens = Vec::with_capacity(PROMPT_CAPACITY);
    tokens.extend(FIXED_PREFIX);
    for index in FIXED_PREFIX.len()..PROMPT_CAPACITY {
        let token = ((index as u64 * 7_919 + 17) % QWEN35_VOCAB_SIZE as u64) as i32;
        tokens.push(token);
    }
    let digest = hash_tokens(&tokens);
    if digest != FIXTURE_SHA256 {
        return Err(format!(
            "fixed prompt fixture digest changed: expected={FIXTURE_SHA256} actual={digest}"
        ));
    }
    Ok(tokens)
}

fn hash_tokens(tokens: &[i32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

fn hash_text(text: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(text.as_bytes()))
}

fn validate_token(token: i32) -> Result<(), String> {
    if (0..QWEN35_VOCAB_SIZE as i32).contains(&token) {
        Ok(())
    } else {
        Err(format!("generated token is outside vocabulary: {token}"))
    }
}

fn parse_rows(text: &str) -> Result<Vec<RowSpec>, String> {
    let mut selected = Vec::new();
    for item in text
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let row = ROWS
            .iter()
            .copied()
            .find(|row| {
                item == format!("{}/{}", row.prompt_tokens, row.output_tokens)
                    || item == row.prompt_tokens.to_string()
            })
            .ok_or_else(|| {
                format!(
                    "{ROWS_ENV} contains unsupported row {item:?}; use 17/17,512/32,2048/128,9435/128"
                )
            })?;
        if selected.contains(&row) {
            return Err(format!("{ROWS_ENV} contains duplicate row {item:?}"));
        }
        selected.push(row);
    }
    if selected.is_empty() {
        return Err(format!("{ROWS_ENV} must select at least one row"));
    }
    selected.sort_by_key(|row| row.prompt_tokens);
    Ok(selected)
}

fn required_env(name: &str) -> Result<String, String> {
    env::var(name).map_err(|_| format!("{name} is required"))
}

fn parse_env_or<T>(name: &str, default: Option<T>) -> Result<T, String>
where
    T: std::str::FromStr,
{
    match env::var(name) {
        Ok(value) => value
            .parse::<T>()
            .map_err(|_| format!("{name} has an invalid value: {value:?}")),
        Err(env::VarError::NotPresent) => default.ok_or_else(|| format!("{name} is required")),
        Err(error) => Err(format!("cannot read {name}: {error}")),
    }
}

fn selector_environment() -> BTreeMap<String, Option<String>> {
    const NAMES: [&str; 62] = [
        CHUNK_CAPACITY_ENV,
        "SLLM_MATMUL_FORCE_BASELINE",
        "SLLM_MATMUL_GFX1030_ROCBLAS_SOLUTION_445",
        "SLLM_MATMUL_GFX1030_SHORT_MIXED",
        "SLLM_NVFP4_W4A4_FORCE_BASELINE",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_COL8",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A_K128",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA_128X32",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA_F16SCALE",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_F16_STAGING",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_FP8_STAGING",
        "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_COLUMNS",
        "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4",
        "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_ACTIVATION_SHARED",
        "SLLM_NVFP4_W4A4_DECODE_FORCE_LDS_F32_LUT",
        "SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8",
        "SLLM_FP8_OUTER_PREFILL_FORCE_BASELINE",
        "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2",
        "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2_64X64",
        "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_LDS_LUT",
        "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_F16_STAGING",
        "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_F16_TILE_STAGING",
        "SLLM_FP8_OUTER_DECODE_FORCE_BASELINE",
        "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_HALF2",
        "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_DWORD8",
        "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_ACTIVATION_SHARED",
        "SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_LDS_LUT",
        "SLLM_FP8_OUTER_GFX1201_HIPBLASLT_HEURISTIC_RANK",
        "SLLM_CAUSAL_ATTENTION_FORCE_BASELINE",
        "SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P32",
        "SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P64",
        "SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P128",
        "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_GFX1030",
        "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_GFX1201",
        "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_BLOCKSOFTMAX_Q8_GFX1201",
        "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4",
        "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K4_FP16",
        "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K8_FP16",
        "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K16_FP16",
        "SLLM_CAUSAL_ATTENTION_GQA6_QTILE4_K32_FP16",
        "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1030_ROCBLAS_F32",
        "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1201_ROCBLAS_F32",
        "SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1201_ROCBLAS_F16_TAIL",
        "SLLM_LINEAR_ATTENTION_GFX1030_ROW32_LDS",
        "SLLM_QWEN_DEFERRED_COMPLETION",
        "SLLM_QWEN38_GFX1030_DEFERRED_COMPLETION",
        "SLLM_QWEN38_GFX1030_GRAPH_SPANS",
        "SLLM_QWEN38_GFX1201_GRAPH_SPANS",
        "SLLM_QWEN38_GFX1201_DEFERRED_COMPLETION",
        "SLLM_QWEN38_GFX1030_KV_APPEND_ATTENTION_CHAIN",
        "SLLM_QWEN38_GFX1201_KV_APPEND_ATTENTION_CHAIN",
        "SLLM_QWEN38_NVFP4_PROJECTION_PACK2",
        "SLLM_QWEN38_FP8_GDN_PROJECTION_PACK2",
        "SLLM_QWEN_GFX1030_RESIDUAL_RMSNORM_FUSION",
        "SLLM_QWEN_GFX1201_RESIDUAL_RMSNORM_FUSION",
        "SLLM_QWEN_GFX1030_GDN_PROJECTION_BUNDLE",
        "SLLM_QWEN_GFX1201_GDN_PROJECTION_BUNDLE",
        "SLLM_QWEN_GFX1030_MLP_GATE_UP_SILU_BUNDLE",
        "SLLM_QWEN_GFX1201_MLP_GATE_UP_SILU_BUNDLE",
        "SLLM_QWEN_GFX1030_SHORT_TERMINAL_LAST_ROW",
    ];
    NAMES
        .into_iter()
        .map(|name| (name.to_owned(), env::var(name).ok()))
        .collect()
}

fn unsupported_reports() -> Vec<UnsupportedReport> {
    vec![
        UnsupportedReport {
            field: "per_run_resettable_peak_vram_and_gtt_spill",
            status: "unsupported",
            reason: "the public session API exposes cumulative checked allocation accounting, not resettable physical VRAM/GTT telemetry",
        },
        UnsupportedReport {
            field: "device_memory_read_bytes_and_gpu_utilization",
            status: "unsupported",
            reason: "these require an external rocprof/AMD-SMI measurement lane",
        },
        UnsupportedReport {
            field: "gpu_family_device_time",
            status: "unsupported",
            reason: "the full-model public audit exposes dispatch counts but does not retain HIP-event durations; use rocprof runtime trace",
        },
        UnsupportedReport {
            field: "partial_offload_and_external_process_interference",
            status: "unsupported",
            reason: "HIP-only dispatch and resident allocations are checked here; system-wide placement/process evidence requires the external controller",
        },
    ]
}

fn emit_json(mut output: impl Write, value: &impl Serialize) -> Result<(), String> {
    serde_json::to_writer(&mut output, value).map_err(|error| error.to_string())?;
    output.write_all(b"\n").map_err(|error| error.to_string())
}

fn emit_failure(error: String) -> ExitCode {
    let report = FailureReport {
        schema_version: "phase78-qwen38-resident-benchmark-error-v1",
        state: "FAIL",
        error,
    };
    if let Err(serialization_error) = emit_json(io::stderr().lock(), &report) {
        eprintln!("Phase 78 benchmark failure serialization failed: {serialization_error}");
        ExitCode::from(2)
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_fixture_has_locked_digest_and_nested_prefixes() {
        let fixture = fixed_prompt_fixture().unwrap();
        assert_eq!(fixture.len(), PROMPT_CAPACITY);
        assert_eq!(&fixture[..FIXED_PREFIX.len()], &FIXED_PREFIX);
        assert_eq!(hash_tokens(&fixture), FIXTURE_SHA256);
        assert_eq!(
            hash_tokens(&fixture[..17]),
            "sha256:8e14ab00e8fd97c64c84103d7bac696cef427f5e52dfa3c07ac228e2c686fb1e"
        );
        assert_eq!(
            hash_tokens(&fixture[..512]),
            "sha256:e863018c1212f24980daf9c89d374a220877f3ab0d6a1ad4fb37cb67ab278856"
        );
        assert_eq!(
            hash_tokens(&fixture[..2_048]),
            "sha256:0a4b788fad4b157e3e1e19cf0ca95653bee44e5e3f460e0869032ec78ff36052"
        );
    }

    #[test]
    fn row_parser_accepts_only_canonical_rows_and_sorts_them() {
        assert_eq!(
            parse_rows("9435/128,17,512/32").unwrap(),
            vec![ROWS[0], ROWS[1], ROWS[3]]
        );
        assert!(parse_rows("512/128").is_err());
        assert!(parse_rows("17,17/17").is_err());
        assert!(parse_rows("").is_err());
    }

    #[test]
    fn median_and_mad_are_deterministic() {
        let report = median_mad(vec![1.0, 9.0, 3.0, 5.0]);
        assert_eq!(report.median, 4.0);
        assert_eq!(report.mad, 2.0);
    }

    #[test]
    fn timing_report_separates_prefill_from_request_inclusive_ttft() {
        let report = timing_report(
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(70),
            200,
            4,
        );
        assert_eq!(report.request_setup_ns, 10_000_000);
        assert_eq!(report.prefill_ns, 20_000_000);
        assert_eq!(report.ttft_ns, 30_000_000);
        assert_eq!(report.decode_ns, 40_000_000);
        assert_eq!(report.e2e_ns, 70_000_000);
        assert_eq!(report.prefill_tokens_per_second, 10_000.0);
        assert_eq!(report.decode_tokens_per_second, 75.0);
        assert_eq!(report.tpot_ms, 40.0 / 3.0);
    }
}
