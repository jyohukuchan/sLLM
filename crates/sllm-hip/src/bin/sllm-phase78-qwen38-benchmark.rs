//! Resident, single-request Phase 78 benchmark for the exact Unsloth
//! Qwen3.8-27B NVFP4 artifact.
//!
//! The binary deliberately accepts configuration only through explicit
//! environment variables so the emitted JSON contains a compact, repeatable
//! execution contract. The default remains the Phase 78 greedy benchmark.
//! Phase 81 explicitly opts into host/GPU fixed sampling and matched input replay.
//! MTP is an explicit Phase83 opt-in using the fixed GPU-selector speculative
//! executor. Legacy mode never enables MTP, EOS termination, stop strings,
//! batching, or non-HIP fallback.

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
    AllocationSnapshot, Backend, ExecutionSessionRequest, KvCacheEncoding, OsSamplingRandom,
    QWEN35_VOCAB_SIZE, QwenExecutionAudit, QwenExecutionRequest, QwenRequestMemoryAudit,
    QwenResidentModel, SamplerChainConfigV1, SamplerChainV1, SamplingParametersV1,
    UNSLOTH_QWEN38_NVFP4_MODEL_SHA256, UNSLOTH_QWEN38_NVFP4_MODEL_SIZE,
    UNSLOTH_QWEN38_NVFP4_REPOSITORY, UNSLOTH_QWEN38_NVFP4_REVISION,
    build_qwen35_unsloth_qwen38_nvfp4_graph, build_qwen38_nvfp4_mtp_graph,
    build_qwen38_nvfp4_mtp_weight_load_plan, build_qwen38_nvfp4_weight_load_plan, read_model_lock,
    verify_unsloth_qwen38_nvfp4,
};
use sllm_frontend::{
    GenerationExecutorV1, Qwen35ChatMessageV1, Qwen35ChatTemplateV1, Qwen35RenderOptionsV1,
    QwenMtpGenerationExecutorV1, ThinkingModeV1, TokenizerFrontendV1,
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
const PHASE83_MODE_ENV: &str = "SLLM_PHASE83_MODE";
const PHASE83_KV_ENV: &str = "SLLM_PHASE83_KV";
const PHASE83_ROWS_ENV: &str = "SLLM_PHASE83_ROWS";
const PHASE83_SAMPLING_ENV: &str = "SLLM_PHASE83_SAMPLING";
const PHASE83_REPLAY_ENV: &str = "SLLM_PHASE83_REPLAY";
const PHASE83_FIXTURE_ONLY: &str = "SLLM_PHASE83_FIXTURE_ONLY";
const PHASE83_MTP_ENV: &str = "SLLM_PHASE83_MTP";
const PHASE83_MTP_WIDTH_ENV: &str = "SLLM_PHASE83_MTP_WIDTH";

const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_MEASURED: usize = 10;
const MAX_REPETITIONS: usize = 100;
const PROMPT_CAPACITY: usize = 9_435;
const STATE_CAPACITY: u64 = 9_563;
const DEFAULT_CHUNK_CAPACITY: u64 = 1_024;
const MAX_CHUNK_CAPACITY: u64 = 8_192;
const PHASE83_PROMPT_CAPACITY: usize = 8_192;
const PHASE83_MTP_MAX_DRAFT_WIDTH: usize = 3;
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

const PHASE83_ROWS: [RowSpec; 1] = [RowSpec {
    prompt_tokens: PHASE83_PROMPT_CAPACITY,
    output_tokens: 128,
}];
const PHASE83_DIAGNOSTIC_ROWS: [RowSpec; 2] = [
    RowSpec {
        prompt_tokens: 17,
        output_tokens: 17,
    },
    RowSpec {
        prompt_tokens: 129,
        output_tokens: 17,
    },
];
const PHASE83_ALLOWED_ROWS: [RowSpec; 3] = [
    PHASE83_ROWS[0],
    PHASE83_DIAGNOSTIC_ROWS[0],
    PHASE83_DIAGNOSTIC_ROWS[1],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PromptFixtureKind {
    Legacy,
    Coding8192,
}

/// Internal performance comparison only; public API sampling remains fixed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SamplingMode {
    #[default]
    Greedy,
    HostFixed,
    GpuFixed,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct SamplingBench {
    mode: SamplingMode,
    replay_inputs: bool,
    seed: u64,
}

impl SamplingBench {
    fn from_env() -> Result<Self, String> {
        Self::from_env_named("SLLM_PHASE81_SAMPLING", SamplingMode::Greedy)
    }

    fn from_phase83_env() -> Result<Self, String> {
        let sampling = Self::from_env_named(PHASE83_SAMPLING_ENV, SamplingMode::GpuFixed)?;
        let replay_inputs = match env::var(PHASE83_REPLAY_ENV).as_deref() {
            Err(env::VarError::NotPresent) | Ok("0") => false,
            Ok("1") => true,
            _ => return Err(format!("{PHASE83_REPLAY_ENV} must be 0 or 1")),
        };
        if replay_inputs {
            return Err(format!(
                "{PHASE83_REPLAY_ENV}=1 is unsupported in Phase83; generation must be autoregressive"
            ));
        }
        Ok(Self {
            replay_inputs,
            ..sampling
        })
    }

    fn from_env_named(name: &str, default: SamplingMode) -> Result<Self, String> {
        let mode = match env::var(name).as_deref() {
            Err(env::VarError::NotPresent) => default,
            Ok("greedy") => SamplingMode::Greedy,
            Ok("host-fixed") => SamplingMode::HostFixed,
            Ok("gpu-fixed") => SamplingMode::GpuFixed,
            _ => {
                return Err(format!("{name} must be greedy, host-fixed or gpu-fixed"));
            }
        };
        let replay_name = if name == PHASE83_SAMPLING_ENV {
            PHASE83_REPLAY_ENV
        } else {
            "SLLM_PHASE81_REPLAY"
        };
        let replay_inputs = match env::var(replay_name).as_deref() {
            Err(env::VarError::NotPresent) | Ok("0") => false,
            Ok("1") => true,
            _ => return Err(format!("{replay_name} must be 0 or 1")),
        };
        Ok(Self {
            mode,
            replay_inputs,
            seed: 123,
        })
    }

    fn input_token(self, step: usize, generated: i32) -> i32 {
        if self.replay_inputs {
            // Independent of selected tokens in every A/B/C run.
            ((step * 7919 + 17) % QWEN35_VOCAB_SIZE) as i32
        } else {
            generated
        }
    }

    fn generation(self) -> &'static str {
        match self.mode {
            SamplingMode::Greedy => "greedy device Argmax; no logits readback",
            SamplingMode::HostFixed => {
                "host sampling: temperature=1 top_k=20 top_p=0.95; full logits readback"
            }
            SamplingMode::GpuFixed => {
                "GPU sampling: temperature=1 top_k=20 top_p=0.95; 16-byte record readback"
            }
        }
    }
}

/// Phase83's MTP switch is deliberately narrower than the frontend's
/// general Qwen MTP width (1..=8). The benchmark uses the reviewed Qwen3.8
/// companion plan/graph and fixed GPU-selector executor for widths 1..=3.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct MtpConfig {
    enabled: bool,
    draft_width: usize,
}

impl MtpConfig {
    const fn disabled() -> Self {
        Self {
            enabled: false,
            draft_width: 0,
        }
    }

    fn from_phase83_env() -> Result<Self, String> {
        let mode = env::var(PHASE83_MTP_ENV).ok();
        let width = env::var(PHASE83_MTP_WIDTH_ENV).ok();
        parse_mtp_config(mode.as_deref(), width.as_deref())
    }

    fn report(self) -> MtpReport {
        if self.enabled {
            MtpReport {
                requested: true,
                draft_width: self.draft_width,
                supported_draft_widths: [1, 2, 3],
                execution: "fixed_gpu_sampler_speculative",
                contract: "Qwen3.8 companion resident and fixed GPU-selector speculative executor are active",
                timing_contract: "prefill_ns must include target prefill and MTP prefix draft priming; decode_ns includes proposal, verify, sampling, accept/reject, replay, and commit",
            }
        } else {
            MtpReport {
                requested: false,
                draft_width: 0,
                supported_draft_widths: [1, 2, 3],
                execution: "disabled",
                contract: "no MTP plan, graph, resident, or executor is constructed",
                timing_contract: "MTP timing fields are absent while the baseline path is disabled",
            }
        }
    }
}

fn parse_mtp_config(mode: Option<&str>, width: Option<&str>) -> Result<MtpConfig, String> {
    let enabled = match mode.unwrap_or("off") {
        "0" | "off" => false,
        "1" | "on" => true,
        _ => return Err(format!("{PHASE83_MTP_ENV} must be off or on")),
    };
    let draft_width = width.unwrap_or("2").parse::<usize>().map_err(|_| {
        format!("{PHASE83_MTP_WIDTH_ENV} must be an integer in 1..={PHASE83_MTP_MAX_DRAFT_WIDTH}")
    })?;
    if !(1..=PHASE83_MTP_MAX_DRAFT_WIDTH).contains(&draft_width) {
        return Err(format!(
            "{PHASE83_MTP_WIDTH_ENV} must be in 1..={PHASE83_MTP_MAX_DRAFT_WIDTH}"
        ));
    }
    Ok(MtpConfig {
        enabled,
        draft_width: if enabled { draft_width } else { 0 },
    })
}

/// Convert executor accounting into the benchmark contract. The executor's
/// committed rows are decode-only; the first prefill-selected token is
/// accounted separately. Rejected draft tokens are never output rows: each
/// proposal block contributes at most one replacement/continuation row plus
/// the draft rows actually accepted.
#[allow(clippy::too_many_arguments)]
fn build_mtp_run_report(
    draft_width: usize,
    proposal_blocks: u64,
    proposed_draft_tokens: u64,
    accepted_draft_tokens: u64,
    committed_target_rows: u64,
    prefill_selected_tokens: usize,
    committed_decode_tokens: usize,
    committed_output_tokens: usize,
    prefix_draft_priming_included: bool,
) -> Result<MtpRunReport, String> {
    if !(1..=PHASE83_MTP_MAX_DRAFT_WIDTH).contains(&draft_width) {
        return Err(format!(
            "MTP report width must be in 1..={PHASE83_MTP_MAX_DRAFT_WIDTH}"
        ));
    }
    if accepted_draft_tokens > proposed_draft_tokens {
        return Err("MTP accepted draft count exceeds proposed count".to_owned());
    }
    let max_proposed = proposal_blocks
        .checked_mul(draft_width as u64)
        .ok_or("MTP proposal count overflows")?;
    if proposed_draft_tokens > max_proposed {
        return Err("MTP proposed draft count exceeds block width".to_owned());
    }
    let max_committed = proposal_blocks
        .checked_add(accepted_draft_tokens)
        .ok_or("MTP committed count overflows")?;
    if committed_target_rows > max_committed {
        return Err(
            "MTP committed target rows include a rejected draft token; output accounting is invalid"
                .to_owned(),
        );
    }
    if accepted_draft_tokens > committed_target_rows {
        return Err("MTP accepted draft count exceeds committed target rows".to_owned());
    }
    if prefill_selected_tokens != 1 {
        return Err("MTP report must account for exactly one prefill-selected token".to_owned());
    }
    if committed_decode_tokens as u64 != committed_target_rows {
        return Err(format!(
            "MTP committed decode count differs from target rows: decode={} rows={committed_target_rows}",
            committed_decode_tokens
        ));
    }
    let expected_output_tokens = prefill_selected_tokens
        .checked_add(committed_decode_tokens)
        .ok_or("MTP output token count overflows")?;
    if committed_output_tokens != expected_output_tokens {
        return Err(format!(
            "MTP committed output count differs from prefill plus decode: output={} expected={expected_output_tokens}",
            committed_output_tokens
        ));
    }
    if !prefix_draft_priming_included {
        return Err("MTP prefill timing omitted prefix draft priming".to_owned());
    }
    Ok(MtpRunReport {
        draft_width,
        proposal_blocks,
        proposed_draft_tokens,
        accepted_draft_tokens,
        rejected_draft_tokens: proposed_draft_tokens - accepted_draft_tokens,
        committed_target_rows,
        prefill_selected_tokens,
        committed_decode_tokens,
        committed_output_tokens,
        prefix_draft_priming_included,
        target_kernel_dispatch_count: 0,
        draft_kernel_dispatch_count: 0,
        draft_fallback_used: false,
        draft_all_dispatches_hip: true,
    })
}

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
    sampling: SamplingBench,
    phase83: bool,
    fixture_kind: PromptFixtureKind,
    kv_cache: KvCacheEncoding,
    mtp: MtpConfig,
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
    sampling: SamplingBench,
    mtp: MtpReport,
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
struct MtpReport {
    requested: bool,
    draft_width: usize,
    supported_draft_widths: [usize; 3],
    execution: &'static str,
    contract: &'static str,
    timing_contract: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MtpRunReport {
    draft_width: usize,
    proposal_blocks: u64,
    proposed_draft_tokens: u64,
    accepted_draft_tokens: u64,
    rejected_draft_tokens: u64,
    committed_target_rows: u64,
    prefill_selected_tokens: usize,
    committed_decode_tokens: usize,
    committed_output_tokens: usize,
    prefix_draft_priming_included: bool,
    target_kernel_dispatch_count: u64,
    draft_kernel_dispatch_count: u64,
    draft_fallback_used: bool,
    draft_all_dispatches_hip: bool,
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
    chat_template_applied: bool,
    rendered_prompt_sha256: Option<String>,
    message_sha256: Option<String>,
}

struct PromptFixture {
    tokens: Vec<i32>,
    report: FixtureReport,
    message: Option<String>,
    rendered_prompt: Option<String>,
}

#[derive(Serialize)]
struct FixtureOnlyReport {
    schema_version: &'static str,
    state: &'static str,
    model_sha256: &'static str,
    fixture: FixtureReport,
    message: String,
    rendered_prompt: String,
    token_ids: Vec<i32>,
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
    row_scope: &'static str,
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
    cpu: CpuReport,
    generated_tokens: Vec<i32>,
    generated_tokens_sha256: String,
    visible_tokens_sha256: String,
    visible_token_count: usize,
    generated_text_sha256: String,
    decoded_token_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    mtp: Option<MtpRunReport>,
    stop_reason: &'static str,
    audit: AuditReport,
    request_memory: RequestMemoryReport,
    allocation_before_request: AllocationReport,
    allocation_while_request_alive: AllocationReport,
    allocation_after_request_drop: AllocationReport,
}

#[derive(Serialize)]
struct CpuReport {
    source: &'static str,
    prefill_user_system_ticks: Option<u64>,
    decode_user_system_ticks: Option<u64>,
}

// Linux /proc units are USER_HZ; report raw ticks rather than inventing a
// conversion. The controller records `getconf CLK_TCK` for the measured host.
fn process_cpu_ticks(enabled: bool) -> Option<u64> {
    if !enabled {
        return None;
    }
    parse_process_cpu_ticks(&fs::read_to_string("/proc/self/stat").ok()?)
}

fn parse_process_cpu_ticks(stat: &str) -> Option<u64> {
    // comm can contain spaces and parentheses; fields following its last ')'
    // start at state (field 3). utime/stime are fields 14/15.
    let fields = stat
        .rsplit_once(')')?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    fields
        .get(11)?
        .parse::<u64>()
        .ok()?
        .checked_add(fields.get(12)?.parse::<u64>().ok()?)
}

fn cpu_delta(start: Option<u64>, end: Option<u64>) -> Option<u64> {
    end?.checked_sub(start?)
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
    if env::var(PHASE83_FIXTURE_ONLY).as_deref() == Ok("1") {
        return emit_fixture_only();
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
        let phase83 = match env::var(PHASE83_MODE_ENV).as_deref() {
            Err(env::VarError::NotPresent) | Ok("0") | Ok("legacy") => false,
            Ok("coding8192") | Ok("1") => true,
            _ => {
                return Err(format!(
                    "{PHASE83_MODE_ENV} must be coding8192, 1, 0 or legacy"
                ));
            }
        };
        if !phase83
            && (env::var_os(PHASE83_KV_ENV).is_some()
                || env::var_os(PHASE83_ROWS_ENV).is_some()
                || env::var_os(PHASE83_SAMPLING_ENV).is_some()
                || env::var_os(PHASE83_REPLAY_ENV).is_some()
                || env::var_os(PHASE83_MTP_ENV).is_some()
                || env::var_os(PHASE83_MTP_WIDTH_ENV).is_some())
        {
            return Err(format!(
                "{PHASE83_MODE_ENV} must enable coding8192 before Phase83 settings are used"
            ));
        }
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
        let rows = if phase83 {
            match env::var(PHASE83_ROWS_ENV) {
                Ok(text) => parse_phase83_rows(&text)?,
                Err(env::VarError::NotPresent) => PHASE83_ROWS.to_vec(),
                Err(error) => return Err(format!("cannot read {PHASE83_ROWS_ENV}: {error}")),
            }
        } else {
            match env::var(ROWS_ENV) {
                Ok(text) => parse_rows(&text)?,
                Err(env::VarError::NotPresent) => ROWS.to_vec(),
                Err(error) => return Err(format!("cannot read {ROWS_ENV}: {error}")),
            }
        };
        let kv_cache = if phase83 {
            parse_phase83_kv()?
        } else {
            KvCacheEncoding::Fp16
        };
        let mtp = if phase83 {
            MtpConfig::from_phase83_env()?
        } else {
            MtpConfig::disabled()
        };
        let sampling = if phase83 {
            SamplingBench::from_phase83_env()?
        } else {
            SamplingBench::from_env()?
        };
        if mtp.enabled && sampling.mode != SamplingMode::GpuFixed {
            return Err(
                "SLLM_PHASE83_MTP=on requires SLLM_PHASE83_SAMPLING=gpu-fixed; ".to_owned(),
            );
        }
        Ok(Self {
            target,
            device_index,
            model_root,
            model_env,
            warmups,
            measured,
            chunk_capacity,
            rows,
            sampling,
            phase83,
            fixture_kind: if phase83 {
                PromptFixtureKind::Coding8192
            } else {
                PromptFixtureKind::Legacy
            },
            kv_cache,
            mtp,
        })
    }

    fn mode(&self) -> &'static str {
        if self.phase83 {
            return "phase83-coding-chat-template-8192-autoregressive";
        }
        if self.sampling.replay_inputs {
            return "phase81-matched-input-replay";
        }
        if self.sampling.mode != SamplingMode::Greedy {
            return "phase81-fixed-sampling-free-generation";
        }
        match (self.warmups, self.measured) {
            (3, 10) => "phase78-final-3-warmup-10-measured",
            (1, 3) => "phase78-exploration-1-warmup-3-measured",
            _ => "custom-explicit-repetitions",
        }
    }

    fn is_phase78_final(&self) -> bool {
        !self.phase83
            && self.sampling.mode == SamplingMode::Greedy
            && !self.sampling.replay_inputs
            && self.warmups == DEFAULT_WARMUPS
            && self.measured == DEFAULT_MEASURED
            && self.rows.as_slice() == ROWS
    }
}

fn run(config: Config) -> Result<Report, String> {
    let verify_started = Instant::now();
    let artifact = Arc::new(
        verify_unsloth_qwen38_nvfp4(&config.model_root).map_err(|error| error.to_string())?,
    );
    let verify_ns = verify_started.elapsed().as_nanos();
    let prompt_fixture = build_prompt_fixture(&artifact, config.fixture_kind)?;
    let mut locked_tokenizer = load_locked_tokenizer(artifact.root())?;
    if config.phase83 {
        locked_tokenizer.report.decode_mode =
            "preserve-special-tokens; Phase83 stop-token visibility follows the reviewed policy";
    }

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
        config.kv_cache,
    )
    .map_err(|error| error.to_string())?;
    let graph_and_plan_ns = graph_started.elapsed().as_nanos();

    // Build the exact companion plan/graph only for the explicit MTP opt-in;
    // the ordinary target-only path retains its historical resident setup.
    let mtp_plan = if config.mtp.enabled {
        Some(
            build_qwen38_nvfp4_mtp_weight_load_plan(&lock, &artifact)
                .map_err(|error| format!("Qwen3.8 MTP companion plan failed: {error}"))?,
        )
    } else {
        None
    };
    let mtp_graph = if let Some(plan) = mtp_plan.as_ref() {
        Some(
            build_qwen38_nvfp4_mtp_graph(&lock, plan, &artifact, STATE_CAPACITY, config.kv_cache)
                .map_err(|error| format!("Qwen3.8 MTP companion graph failed: {error}"))?,
        )
    } else {
        None
    };

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
        let mtp_resident = match (mtp_plan.as_ref(), mtp_graph.as_ref()) {
            (Some(plan), Some(graph)) => Some(provision_qwen38_mtp_resident(
                &resident,
                graph.clone(),
                plan.clone(),
                Arc::clone(&artifact),
            )?),
            (None, None) => None,
            _ => return Err("MTP companion plan/graph are only partially initialized".to_owned()),
        };
        let resident_load_ns = load_started.elapsed().as_nanos();
        let resident_ready_memory = allocation_report(session.memory_snapshot());
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
                &prompt_fixture.tokens,
                *row,
                config.warmups,
                config.measured,
                &config.target,
                &locked_tokenizer.tokenizer,
                config.sampling,
                if config.phase83 {
                    Some(lock.generation_stop_policy().stop_token_ids.as_slice())
                } else {
                    None
                },
                mtp_resident.as_ref(),
                mtp_graph.as_ref(),
                config.mtp,
                config.phase83,
            )?);
        }
        drop(mtp_resident);
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
        schema_version: if config.phase83 {
            "phase83-qwen38-resident-benchmark-v1"
        } else {
            "phase78-qwen38-resident-benchmark-v3"
        },
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
        sampling: config.sampling,
        mtp: config.mtp.report(),
        protocol: ProtocolReport {
            active_requests: 1,
            parallel_requests: 1,
            batching: "disabled; rows and repetitions execute serially with one fresh request",
            kv_cache: if config.phase83 {
                config.kv_cache.canonical_name()
            } else {
                "FP16"
            },
            state_capacity_tokens: STATE_CAPACITY,
            prefill_chunk_capacity_tokens: config.chunk_capacity,
            mtp: if config.mtp.enabled {
                "enabled; Qwen3.8 companion resident and fixed GPU-selector speculative executor"
            } else if config.phase83 {
                "disabled; Phase83 target-only baseline has no MTP resident or speculative executor"
            } else {
                "disabled; only non-MTP graph and prefill/decode APIs are called"
            },
            generation: config.sampling.generation(),
            eos_termination: config.phase83,
            stop_sequences: false,
            termination: if config.phase83 {
                "output budget or reviewed stop-token policy; autoregressive selected tokens"
            } else {
                "fixed total output-token budget; generated EOS-like IDs are not inspected"
            },
            output_accounting: if config.phase83 {
                "generated_tokens contains selected tokens including a terminal stop token; visible output fields omit stop IDs per the reviewed policy"
            } else {
                "generated_tokens contains selected tokens; sampling.replay_inputs replaces only subsequent inputs with (step*7919+17)%248320; TPOT and decode throughput count only those decode transitions"
            },
        },
        fixture: prompt_fixture.report,
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

/// Build the Phase83 fixed-sampler executor against the companion request pair.
#[allow(dead_code)]
fn construct_qwen38_mtp_executor(
    target: QwenExecutionRequest,
    mtp: QwenExecutionRequest,
    draft_width: usize,
) -> Result<QwenMtpGenerationExecutorV1, String> {
    QwenMtpGenerationExecutorV1::new_with_draft_width(target, mtp, draft_width)
        .map_err(|error| format!("Qwen3.8 MTP executor construction failed: {error}"))
}

/// Provision the companion resident used by the Phase83 speculative executor.
#[allow(dead_code)]
fn provision_qwen38_mtp_resident(
    target: &QwenResidentModel,
    graph: sllm_core::QwenGraph,
    plan: sllm_core::WeightLoadPlan,
    artifact: Arc<sllm_core::VerifiedUnslothQwen38Nvfp4>,
) -> Result<QwenResidentModel, String> {
    QwenResidentModel::new_unsloth_qwen38_nvfp4_mtp_shared(
        target,
        graph,
        plan,
        artifact,
        COMPLETION_TIMEOUT,
    )
    .map_err(|error| format!("Qwen3.8 MTP resident construction failed: {error}"))
}

fn emit_fixture_only() -> ExitCode {
    let result = (|| -> Result<FixtureOnlyReport, String> {
        let mode = env::var(PHASE83_MODE_ENV).unwrap_or_default();
        if mode != "coding8192" && mode != "1" {
            return Err(format!(
                "{PHASE83_FIXTURE_ONLY}=1 requires {PHASE83_MODE_ENV}=coding8192"
            ));
        }
        let model_root = match env::var_os(MODEL_ENV) {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from(
                env::var_os(COMPAT_MODEL_ENV)
                    .ok_or_else(|| format!("{MODEL_ENV} or {COMPAT_MODEL_ENV} is required"))?,
            ),
        };
        let artifact = verify_unsloth_qwen38_nvfp4(&model_root)
            .map_err(|error| format!("verify Qwen3.8 artifact: {error}"))?;
        let fixture = build_prompt_fixture(&artifact, PromptFixtureKind::Coding8192)?;
        let message = fixture
            .message
            .clone()
            .ok_or("Phase83 fixture message was not retained")?;
        let rendered_prompt = fixture
            .rendered_prompt
            .clone()
            .ok_or("Phase83 rendered prompt was not retained")?;
        Ok(FixtureOnlyReport {
            schema_version: "phase83-qwen38-fixture-only-v1",
            state: "PASS",
            model_sha256: UNSLOTH_QWEN38_NVFP4_MODEL_SHA256,
            fixture: fixture.report,
            message,
            rendered_prompt,
            token_ids: fixture.tokens,
        })
    })();
    match result {
        Ok(report) => {
            if let Err(error) = emit_json(io::stdout().lock(), &report) {
                eprintln!("Phase83 fixture-only JSON serialization failed: {error}");
                ExitCode::from(2)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => emit_failure(error),
    }
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
    sampling: SamplingBench,
    stop_token_ids: Option<&[u32]>,
    mtp_resident: Option<&QwenResidentModel>,
    mtp_graph: Option<&sllm_core::QwenGraph>,
    mtp: MtpConfig,
    phase83: bool,
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
                sampling,
                stop_token_ids,
                mtp_resident,
                mtp_graph,
                mtp,
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
        row_scope: if phase83 {
            if row == PHASE83_ROWS[0] {
                "phase83-default-acceptance-row"
            } else {
                "phase83-diagnostic-short-prefix-not-acceptance"
            }
        } else {
            "legacy-row"
        },
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
    sampling: SamplingBench,
    stop_token_ids: Option<&[u32]>,
    mtp_resident: Option<&QwenResidentModel>,
    mtp_graph: Option<&sllm_core::QwenGraph>,
    mtp: MtpConfig,
) -> Result<RunReport, String> {
    if mtp.enabled {
        return run_one_mtp(
            session,
            resident,
            graph,
            mtp_resident.ok_or("MTP resident is missing")?,
            mtp_graph.ok_or("MTP graph is missing")?,
            prompt,
            output_tokens,
            sample_kind,
            sample_index,
            target,
            tokenizer,
            sampling,
            stop_token_ids,
            mtp.draft_width,
        );
    }
    run_one_target(
        session,
        resident,
        graph,
        prompt,
        output_tokens,
        sample_kind,
        sample_index,
        target,
        tokenizer,
        sampling,
        stop_token_ids,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_one_target(
    session: &Arc<sllm_core::ExecutionSession>,
    resident: &QwenResidentModel,
    graph: &sllm_core::QwenGraph,
    prompt: &[i32],
    output_tokens: usize,
    sample_kind: &'static str,
    sample_index: usize,
    target: &str,
    tokenizer: &Tokenizer,
    sampling: SamplingBench,
    stop_token_ids: Option<&[u32]>,
) -> Result<RunReport, String> {
    let phase83_progress = stop_token_ids.is_some();
    if phase83_progress {
        eprintln!(
            "[phase83] start row={}/{} sample={}/{}",
            prompt.len(),
            output_tokens,
            sample_kind,
            sample_index
        );
    }
    let before_request = allocation_report(session.memory_snapshot());
    // Phase 78 compares time-to-first-token and request E2E after the model is
    // resident.  Start both clocks before request-local state/queue creation;
    // keep the narrower prefill interval as a separate kernel/runtime metric.
    let e2e_started = Instant::now();
    let mut request = resident
        .new_request(graph.clone())
        .map_err(|error| format!("request creation failed: {error}"))?;
    let request_setup_elapsed = e2e_started.elapsed();
    let mut sampling_state = if sampling.mode == SamplingMode::Greedy {
        None
    } else {
        let parameters =
            SamplingParametersV1::new(1.0, 0.95, 0.0, 0.0).map_err(|error| error.to_string())?;
        let prior = prompt.iter().map(|&token| token as u32).collect::<Vec<_>>();
        let sampler = SamplerChainV1::new(
            SamplerChainConfigV1::new(parameters)
                .with_top_k(20)
                .map_err(|error| error.to_string())?,
            &prior,
        )
        .map_err(|error| error.to_string())?;
        let random = OsSamplingRandom::for_parameters_and_seed(parameters, Some(sampling.seed))
            .map_err(|error| error.to_string())?;
        Some((sampler, random))
    };
    let observe_cpu = sampling.mode != SamplingMode::Greedy || sampling.replay_inputs;
    let prefill_cpu_started = process_cpu_ticks(observe_cpu);
    let prefill_started = Instant::now();
    let prefill = match sampling.mode {
        SamplingMode::Greedy => request.prefill(prompt),
        SamplingMode::HostFixed => request.prefill_with_last_logits(prompt),
        SamplingMode::GpuFixed => {
            let selector = sampling_state
                .as_ref()
                .expect("sampling mode state")
                .0
                .prepare_device_selector(QWEN35_VOCAB_SIZE, None, sampling.seed, 0)
                .map_err(|error| error.to_string())?;
            request.prefill_with_device_selector(prompt, &selector)
        }
    }
    .map_err(|error| format!("prefill failed: {error}"))?;
    let mut current = selected_token(&prefill, sampling.mode, &mut sampling_state)?;
    let prefill_elapsed = prefill_started.elapsed();
    let ttft_elapsed = e2e_started.elapsed();
    let prefill_cpu = cpu_delta(prefill_cpu_started, process_cpu_ticks(observe_cpu));
    validate_token(current)?;
    let mut generated = Vec::with_capacity(output_tokens);
    generated.push(current);
    let mut stop_reason = "length";
    if stop_token_ids.is_some_and(|ids| ids.contains(&(current as u32))) {
        stop_reason = "stop_token";
    }
    let decode_cpu_started = process_cpu_ticks(observe_cpu);
    let decode_started = Instant::now();
    for step in 1..output_tokens {
        if stop_reason == "stop_token" {
            break;
        }
        let input = sampling.input_token(step, current);
        let output = match sampling.mode {
            SamplingMode::Greedy => request.decode(input),
            SamplingMode::HostFixed => request.decode_with_last_logits(input),
            SamplingMode::GpuFixed => {
                let selector = sampling_state
                    .as_ref()
                    .expect("sampling mode state")
                    .0
                    .prepare_device_selector(QWEN35_VOCAB_SIZE, None, sampling.seed, step as u64)
                    .map_err(|error| error.to_string())?;
                request.decode_with_device_selector(input, &selector)
            }
        }
        .map_err(|error| format!("decode step {step} failed: {error}"))?;
        if output.token_ids().len() != 1 {
            return Err(format!("decode step {step} returned more than one row"));
        }
        current = selected_token(&output, sampling.mode, &mut sampling_state)?;
        validate_token(current)?;
        generated.push(current);
        if stop_token_ids.is_some_and(|ids| ids.contains(&(current as u32))) {
            stop_reason = "stop_token";
            break;
        }
    }
    let decode_elapsed = decode_started.elapsed();
    let e2e_elapsed = e2e_started.elapsed();
    let decode_cpu = cpu_delta(decode_cpu_started, process_cpu_ticks(observe_cpu));
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
    let decoded = decoded_output_report(tokenizer, &generated, stop_token_ids)?;

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
        generated.len(),
    );
    if phase83_progress {
        eprintln!(
            "[phase83] done row={}/{} sample={}/{} prefill_ms={:.3} decode_ms={:.3} generated={}",
            prompt.len(),
            output_tokens,
            sample_kind,
            sample_index,
            timing.prefill_ns as f64 / 1_000_000.0,
            timing.decode_ns as f64 / 1_000_000.0,
            generated.len()
        );
    }
    Ok(RunReport {
        sample_kind,
        sample_index,
        prefill_output_rows: prefill.token_ids().len(),
        decode_transition_count: generated.len() - 1,
        timing,
        cpu: CpuReport {
            source: "/proc/self/stat utime+stime; raw USER_HZ ticks; null when unavailable/disabled",
            prefill_user_system_ticks: prefill_cpu,
            decode_user_system_ticks: decode_cpu,
        },
        generated_tokens_sha256: hash_tokens(&generated),
        visible_tokens_sha256: decoded.visible_tokens_sha256,
        visible_token_count: decoded.visible_token_count,
        generated_text_sha256: decoded.generated_text_sha256,
        decoded_token_count: decoded.decoded_token_count,
        generated_tokens: generated,
        mtp: None,
        stop_reason,
        audit,
        request_memory,
        allocation_before_request: before_request,
        allocation_while_request_alive: while_request_alive,
        allocation_after_request_drop: after_request_drop,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_one_mtp(
    session: &Arc<sllm_core::ExecutionSession>,
    resident: &QwenResidentModel,
    graph: &sllm_core::QwenGraph,
    mtp_resident: &QwenResidentModel,
    mtp_graph: &sllm_core::QwenGraph,
    prompt: &[i32],
    output_tokens: usize,
    sample_kind: &'static str,
    sample_index: usize,
    target: &str,
    tokenizer: &Tokenizer,
    sampling: SamplingBench,
    stop_token_ids: Option<&[u32]>,
    draft_width: usize,
) -> Result<RunReport, String> {
    if sampling.mode != SamplingMode::GpuFixed {
        return Err("MTP requires the fixed GPU sampler".to_owned());
    }
    if sampling.replay_inputs {
        return Err("MTP requires autoregressive inputs; replay mode is unsupported".to_owned());
    }
    let phase83_progress = stop_token_ids.is_some();
    if phase83_progress {
        eprintln!(
            "[phase83] start row={}/{} sample={}/{} mtp_width={}",
            prompt.len(),
            output_tokens,
            sample_kind,
            sample_index,
            draft_width
        );
    }
    let before_request = allocation_report(session.memory_snapshot());
    let e2e_started = Instant::now();
    let target_request = resident
        .new_request(graph.clone())
        .map_err(|error| format!("MTP target request creation failed: {error}"))?;
    let mtp_request = mtp_resident
        .new_request(mtp_graph.clone())
        .map_err(|error| format!("MTP draft request creation failed: {error}"))?;
    let mut executor =
        QwenMtpGenerationExecutorV1::new_with_draft_width(target_request, mtp_request, draft_width)
            .map_err(|error| format!("MTP executor construction failed: {error}"))?;
    let request_setup_elapsed = e2e_started.elapsed();
    let sampling_state = {
        let parameters =
            SamplingParametersV1::new(1.0, 0.95, 0.0, 0.0).map_err(|error| error.to_string())?;
        let prior = prompt.iter().map(|&token| token as u32).collect::<Vec<_>>();
        let sampler = SamplerChainV1::new(
            SamplerChainConfigV1::new(parameters)
                .with_top_k(20)
                .map_err(|error| error.to_string())?,
            &prior,
        )
        .map_err(|error| error.to_string())?;
        let random = OsSamplingRandom::for_parameters_and_seed(parameters, Some(sampling.seed))
            .map_err(|error| error.to_string())?;
        (sampler, random)
    };
    let prompt_u32 = prompt
        .iter()
        .map(|&token| u32::try_from(token).map_err(|_| "prompt token is negative".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    let observe_cpu = true;
    let prefill_cpu_started = process_cpu_ticks(observe_cpu);
    let prefill_started = Instant::now();
    let prefill_selector = sampling_state
        .0
        .prepare_device_selector(QWEN35_VOCAB_SIZE, None, sampling.seed, 0)
        .map_err(|error| error.to_string())?;
    let prefill_step = match executor.prefill_with_device_selector(&prompt_u32, &prefill_selector) {
        Ok(step) => step,
        Err(error) => {
            executor.cancel();
            return Err(format!("MTP prefill failed: {error}"));
        }
    };
    let prefill_elapsed = prefill_started.elapsed();
    let ttft_elapsed = e2e_started.elapsed();
    let prefill_cpu = cpu_delta(prefill_cpu_started, process_cpu_ticks(observe_cpu));
    let mut current = i32::try_from(prefill_step.device_argmax())
        .map_err(|_| "MTP prefill selected token overflows i32".to_owned())?;
    validate_token(current)?;
    let mut generated = Vec::with_capacity(output_tokens);
    generated.push(current);
    let mut stop_reason = "length";
    if stop_token_ids.is_some_and(|ids| ids.contains(&(current as u32))) {
        stop_reason = "stop_token";
    }
    let decode_cpu_started = process_cpu_ticks(observe_cpu);
    let decode_started = Instant::now();
    let generation = (|| -> Result<(), String> {
        for step in 1..output_tokens {
            if stop_reason == "stop_token" {
                break;
            }
            let input = sampling.input_token(step, current);
            let selector = sampling_state
                .0
                .prepare_device_selector(QWEN35_VOCAB_SIZE, None, sampling.seed, step as u64)
                .map_err(|error| error.to_string())?;
            let output = executor
                .decode_with_device_selector(input as u32, &selector)
                .map_err(|error| format!("MTP decode step {step} failed: {error}"))?;
            current = i32::try_from(output.device_argmax())
                .map_err(|_| format!("MTP decode step {step} token overflows i32"))?;
            validate_token(current)?;
            generated.push(current);
            if stop_token_ids.is_some_and(|ids| ids.contains(&(current as u32))) {
                stop_reason = "stop_token";
                break;
            }
        }
        // A width>1 executor may still own accepted rows which have not been
        // presented to the sequential loop. Finish resolves that prefix and
        // rewinds every unpublished row before the request is audited.
        executor
            .finish()
            .map_err(|error| format!("MTP finalization failed: {error}"))?;
        Ok(())
    })();
    if let Err(error) = generation {
        executor.cancel();
        return Err(error);
    }
    let decode_elapsed = decode_started.elapsed();
    let e2e_elapsed = e2e_started.elapsed();
    let decode_cpu = cpu_delta(decode_cpu_started, process_cpu_ticks(observe_cpu));
    if request_setup_elapsed.is_zero()
        || prefill_elapsed.is_zero()
        || ttft_elapsed.is_zero()
        || decode_elapsed.is_zero()
        || e2e_elapsed.is_zero()
    {
        executor.cancel();
        return Err("a benchmark timing interval was zero".to_owned());
    }
    let decoded = match decoded_output_report(tokenizer, &generated, stop_token_ids) {
        Ok(decoded) => decoded,
        Err(error) => {
            executor.cancel();
            return Err(error);
        }
    };
    let target_audit = match executor.target().audit_snapshot() {
        Ok(audit) => audit,
        Err(error) => {
            executor.cancel();
            return Err(error.to_string());
        }
    };
    let draft_audit = match executor.mtp().audit_snapshot() {
        Ok(audit) => audit,
        Err(error) => {
            executor.cancel();
            return Err(error.to_string());
        }
    };
    if target_audit.selected_backend() != "hip"
        || target_audit.target() != target
        || target_audit.submission_count() == 0
        || target_audit.kernel_dispatch_count() == 0
        || target_audit.fallback_used()
        || !target_audit.all_dispatches_hip()
        || draft_audit.selected_backend() != "hip"
        || draft_audit.target() != target
        || draft_audit.submission_count() == 0
        || draft_audit.kernel_dispatch_count() == 0
        || draft_audit.fallback_used()
        || !draft_audit.all_dispatches_hip()
    {
        executor.cancel();
        return Err(format!(
            "MTP dispatch audits are not HIP-only: target={target_audit:?} draft={draft_audit:?}"
        ));
    }
    let mut mtp_report = build_mtp_run_report(
        draft_width,
        executor.proposal_blocks(),
        executor.proposed_draft_tokens(),
        executor.accepted_draft_tokens(),
        executor.committed_target_rows(),
        1,
        generated.len().saturating_sub(1),
        generated.len(),
        true,
    )?;
    mtp_report.target_kernel_dispatch_count = target_audit.kernel_dispatch_count();
    mtp_report.draft_kernel_dispatch_count = draft_audit.kernel_dispatch_count();
    mtp_report.draft_fallback_used = draft_audit.fallback_used();
    mtp_report.draft_all_dispatches_hip = draft_audit.all_dispatches_hip();
    let audit = audit_report(&target_audit);
    let request_memory = request_memory_report(
        &executor
            .target()
            .memory_audit_snapshot()
            .map_err(|error| format!("MTP target memory audit failed: {error}"))?,
    )?;
    let while_request_alive = allocation_report(session.memory_snapshot());
    let timing = timing_report(
        request_setup_elapsed,
        prefill_elapsed,
        ttft_elapsed,
        decode_elapsed,
        e2e_elapsed,
        prompt.len(),
        generated.len(),
    );
    drop(executor);
    let after_request_drop = allocation_report(session.memory_snapshot());
    if phase83_progress {
        eprintln!(
            "[phase83] done row={}/{} sample={}/{} mtp_width={} prefill_ms={:.3} decode_ms={:.3} generated={} accepted={}/{}",
            prompt.len(),
            output_tokens,
            sample_kind,
            sample_index,
            draft_width,
            timing.prefill_ns as f64 / 1_000_000.0,
            timing.decode_ns as f64 / 1_000_000.0,
            generated.len(),
            mtp_report.accepted_draft_tokens,
            mtp_report.proposed_draft_tokens,
        );
    }
    Ok(RunReport {
        sample_kind,
        sample_index,
        prefill_output_rows: 1,
        decode_transition_count: generated.len() - 1,
        timing,
        cpu: CpuReport {
            source: "/proc/self/stat utime+stime; raw USER_HZ ticks; null when unavailable/disabled",
            prefill_user_system_ticks: prefill_cpu,
            decode_user_system_ticks: decode_cpu,
        },
        generated_tokens_sha256: hash_tokens(&generated),
        visible_tokens_sha256: decoded.visible_tokens_sha256,
        visible_token_count: decoded.visible_token_count,
        generated_text_sha256: decoded.generated_text_sha256,
        decoded_token_count: decoded.decoded_token_count,
        generated_tokens: generated,
        mtp: Some(mtp_report),
        stop_reason,
        audit,
        request_memory,
        allocation_before_request: before_request,
        allocation_while_request_alive: while_request_alive,
        allocation_after_request_drop: after_request_drop,
    })
}

fn selected_token(
    output: &sllm_core::QwenExecutionOutput,
    mode: SamplingMode,
    sampling_state: &mut Option<(SamplerChainV1, OsSamplingRandom)>,
) -> Result<i32, String> {
    let token = *output.token_ids().last().ok_or("terminal token missing")?;
    match mode {
        SamplingMode::Greedy => {
            if output.selection().is_some() || output.last_logits().is_some() {
                return Err("greedy comparison left device Argmax/no-logits route".into());
            }
            Ok(token)
        }
        SamplingMode::HostFixed => {
            if output.selection().is_some() || output.last_logits().is_none() {
                return Err("host comparison did not return full logits".into());
            }
            let (sampler, random) = sampling_state.as_mut().ok_or("host sampler missing")?;
            sampler
                .select_token(token as u32, output.last_logits(), random)
                .map(|token| token as i32)
                .map_err(|error| error.to_string())
        }
        SamplingMode::GpuFixed => {
            let selection = output.selection().ok_or("GPU selection missing")?;
            if output.last_logits().is_some() || token as u32 != selection.token_id {
                return Err("GPU comparison returned logits or inconsistent selection".into());
            }
            Ok(token)
        }
    }
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
    let decode_transitions = output_tokens.saturating_sub(1);
    let decode_tokens_per_second = if decode_transitions == 0 {
        0.0
    } else {
        decode_transitions as f64 / decode_seconds
    };
    let tpot_ms = if decode_transitions == 0 {
        0.0
    } else {
        decode_seconds * 1_000.0 / decode_transitions as f64
    };
    TimingReport {
        request_setup_ns: request_setup.as_nanos(),
        prefill_ns: prefill.as_nanos(),
        ttft_ns: ttft.as_nanos(),
        decode_ns: decode.as_nanos(),
        e2e_ns: e2e.as_nanos(),
        prefill_tokens_per_second: prompt_tokens as f64 / prefill_seconds,
        decode_tokens_per_second,
        output_tokens_per_e2e_second: output_tokens as f64 / e2e_seconds,
        total_tokens_per_e2e_second: (prompt_tokens + output_tokens) as f64 / e2e_seconds,
        tpot_ms,
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
        (11, "matmul.nvfp4.w4a4.block16.packed.v1"),
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
        (66, "matmul.fp8.outer.decode.gfx1030.half2.wave4col32.v1"),
        (67, "matmul.nvfp4.w4a4.decode.dp4a.wave4col32.v1"),
        (68, "matmul.fp8.outer.decode.gfx1030.dword8.wave4col32.v1"),
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
        (80, "causal_attention.decode.wave8_split.staged.gfx1030.v1"),
        (82, "matmul.fp8.outer.decode.gfx1030.lds_lut.wave4col32.v1"),
        (84, "matmul.nvfp4.w4a4.decode.scale_lut.v1"),
        (87, "matmul.nvfp4.w4a4.block16.prefill.compensated64x64.v1"),
        (88, "matmul.nvfp4.w4a4.small_m.rowgrid.v1"),
        (89, "matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x64.kahan.v1"),
        (90, "matmul.nvfp4.w4a4.small_m.gfx1201.rowgrid.v1"),
        (
            91,
            "matmul.bf16_fp32.prefill.gfx1030.64x64.k32_transposed.v1",
        ),
        (92, "matmul.fp8.outer.decode.gfx1030.fused.m2_4.v1"),
        (93, "causal_attention.decode.wave32_split.staged.v1"),
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
    stop_token_ids: Option<&[u32]>,
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
    let visible_ids = match stop_token_ids {
        Some(stop_ids) => ids
            .iter()
            .copied()
            .filter(|id| !stop_ids.contains(id))
            .collect::<Vec<_>>(),
        None => ids.clone(),
    };
    let text = tokenizer
        .decode(&visible_ids, false)
        .map_err(|error| format!("decode generated tokens: {error}"))?;
    Ok(DecodedOutputReport {
        visible_tokens_sha256: hash_tokens(
            &visible_ids
                .iter()
                .copied()
                .map(|token| token as i32)
                .collect::<Vec<_>>(),
        ),
        visible_token_count: visible_ids.len(),
        generated_text_sha256: hash_text(&text),
        decoded_token_count: visible_ids.len(),
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

fn build_prompt_fixture(
    artifact: &sllm_core::VerifiedUnslothQwen38Nvfp4,
    kind: PromptFixtureKind,
) -> Result<PromptFixture, String> {
    match kind {
        PromptFixtureKind::Legacy => {
            let tokens = fixed_prompt_fixture()?;
            Ok(PromptFixture {
                report: FixtureReport {
                    schema_version: "phase78-qwen38-fixed-token-fixture-v1",
                    construction: "tokens[0..17]=fixed_prefix_17; tokens[i>=17]=(i*7919+17)%248320",
                    total_tokens: tokens.len(),
                    token_encoding: "signed-i32 token IDs; SHA-256 over concatenated little-endian i32",
                    sha256: hash_tokens(&tokens),
                    fixed_prefix_17: FIXED_PREFIX,
                    chat_template_applied: false,
                    rendered_prompt_sha256: None,
                    message_sha256: None,
                },
                tokens,
                message: None,
                rendered_prompt: None,
            })
        }
        PromptFixtureKind::Coding8192 => coding_prompt_fixture(artifact),
    }
}

fn coding_prompt_fixture(
    artifact: &sllm_core::VerifiedUnslothQwen38Nvfp4,
) -> Result<PromptFixture, String> {
    let tokenizer = TokenizerFrontendV1::from_unsloth_qwen38_nvfp4(artifact)
        .map_err(|error| format!("load Qwen3.8 frontend tokenizer: {error}"))?;
    let renderer = Qwen35ChatTemplateV1::from_unsloth_qwen38_nvfp4(artifact)
        .map_err(|error| format!("load Qwen3.8 chat template: {error}"))?;
    let (message, rendered, ids) = find_coding_fixture(&tokenizer, &renderer)?;
    if ids.len() != PHASE83_PROMPT_CAPACITY {
        return Err(format!(
            "Phase83 coding fixture has {} tokens, expected {}",
            ids.len(),
            PHASE83_PROMPT_CAPACITY
        ));
    }
    let tokens = ids
        .into_iter()
        .map(|token| i32::try_from(token).map_err(|_| format!("token ID overflows i32: {token}")))
        .collect::<Result<Vec<_>, _>>()?;
    let prefix: [i32; 17] = tokens
        .get(..FIXED_PREFIX.len())
        .ok_or("Phase83 coding fixture is shorter than its prefix report")?
        .try_into()
        .map_err(|_| "Phase83 coding fixture prefix conversion failed")?;
    let rendered_prompt_sha256 = hash_text(&rendered);
    let message_sha256 = hash_text(&message);
    Ok(PromptFixture {
        report: FixtureReport {
            schema_version: "phase83-qwen38-chat-coding-fixture-v1",
            construction: "Qwen3.8 reviewed chat template applied to a deterministic coding message; tokenizer output is exactly 8192 IDs",
            total_tokens: tokens.len(),
            token_encoding: "signed-i32 token IDs; SHA-256 over concatenated little-endian i32",
            sha256: hash_tokens(&tokens),
            fixed_prefix_17: prefix,
            chat_template_applied: true,
            rendered_prompt_sha256: Some(rendered_prompt_sha256),
            message_sha256: Some(message_sha256),
        },
        tokens,
        message: Some(message),
        rendered_prompt: Some(rendered),
    })
}

fn find_coding_fixture(
    tokenizer: &TokenizerFrontendV1,
    renderer: &Qwen35ChatTemplateV1,
) -> Result<(String, String, Vec<u32>), String> {
    const BASE: &str = "Implement and review this Rust coding task. Explain ownership, error handling, and complexity before presenting the final code.\n\n```rust\nfn checked_sum(values: &[i32]) -> Result<i32, &'static str> {\n    values.iter().try_fold(0_i32, |sum, value| sum.checked_add(*value).ok_or(\"overflow\"))\n}\n```\n";
    const UNIT: &str = "\n// Preserve this deterministic coding context and inspect each boundary carefully.\nfn boundary_case(value: i32) -> Option<i32> { value.checked_add(1) }\n";
    const FILLER: &str = " x";

    let render = |message: &str| -> Result<(String, Vec<u32>), String> {
        let rendered = renderer
            .render(
                &[Qwen35ChatMessageV1::user(message)],
                Qwen35RenderOptionsV1 {
                    add_generation_prompt: true,
                    thinking: ThinkingModeV1::Disabled,
                },
            )
            .map_err(|error| format!("render Phase83 coding fixture: {error}"))?;
        let ids = tokenizer
            .encode(&rendered)
            .map_err(|error| format!("tokenize Phase83 coding fixture: {error}"))?
            .as_slice()
            .to_vec();
        Ok((rendered, ids))
    };

    let token_count = |repeats: usize| -> Result<usize, String> {
        let message = format!("{BASE}{}", UNIT.repeat(repeats));
        Ok(render(&message)?.1.len())
    };
    let mut low = 0usize;
    let mut high = 1usize;
    while token_count(high)? < PHASE83_PROMPT_CAPACITY {
        high = high
            .checked_mul(2)
            .ok_or("Phase83 coding fixture repeat count overflow")?;
        if high > PHASE83_PROMPT_CAPACITY * 4 {
            return Err("unable to reach 8192 tokens with coding fixture context".to_owned());
        }
    }
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if token_count(middle)? < PHASE83_PROMPT_CAPACITY {
            low = middle;
        } else {
            high = middle;
        }
    }

    let mut bases = Vec::new();
    for repeats in [low, high] {
        let message = format!("{BASE}{}", UNIT.repeat(repeats));
        let (rendered, ids) = render(&message)?;
        bases.push((message, rendered, ids));
    }
    for (message, rendered, ids) in &bases {
        if ids.len() == PHASE83_PROMPT_CAPACITY {
            return Ok((message.clone(), rendered.clone(), ids.clone()));
        }
    }

    // BPE boundaries can leave a small gap between adjacent repeated units.
    // A bounded binary search over a token-friendly filler closes that gap
    // without carrying a large fixture into the repository.
    let (base_message, _, base_ids) = bases
        .iter()
        .find(|(_, _, ids)| ids.len() < PHASE83_PROMPT_CAPACITY)
        .cloned()
        .ok_or("Phase83 fixture search did not retain a lower bound")?;
    let mut filler_low = 0usize;
    let mut filler_high = PHASE83_PROMPT_CAPACITY;
    while filler_low + 1 < filler_high {
        let middle = filler_low + (filler_high - filler_low) / 2;
        let message = format!("{base_message}{}", FILLER.repeat(middle));
        if render(&message)?.1.len() < PHASE83_PROMPT_CAPACITY {
            filler_low = middle;
        } else {
            filler_high = middle;
        }
    }
    for filler_count in filler_low.saturating_sub(32)..=filler_high.saturating_add(32) {
        let message = format!("{base_message}{}", FILLER.repeat(filler_count));
        let (rendered, ids) = render(&message)?;
        if ids.len() == PHASE83_PROMPT_CAPACITY {
            return Ok((message, rendered, ids));
        }
    }
    Err(format!(
        "could not construct exactly {} chat-template tokens (lower bound had {} tokens)",
        PHASE83_PROMPT_CAPACITY,
        base_ids.len()
    ))
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
    parse_rows_from(text, &ROWS, ROWS_ENV)
}

fn parse_phase83_rows(text: &str) -> Result<Vec<RowSpec>, String> {
    parse_rows_from(text, &PHASE83_ALLOWED_ROWS, PHASE83_ROWS_ENV)
}

fn parse_rows_from(
    text: &str,
    allowed: &[RowSpec],
    env_name: &str,
) -> Result<Vec<RowSpec>, String> {
    let mut selected = Vec::new();
    for item in text
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let row = allowed
            .iter()
            .copied()
            .find(|row| {
                item == format!("{}/{}", row.prompt_tokens, row.output_tokens)
                    || item == row.prompt_tokens.to_string()
            })
            .ok_or_else(|| format!("{env_name} contains unsupported row {item:?}"))?;
        if selected.contains(&row) {
            return Err(format!("{env_name} contains duplicate row {item:?}"));
        }
        selected.push(row);
    }
    if selected.is_empty() {
        return Err(format!("{env_name} must select at least one row"));
    }
    selected.sort_by_key(|row| row.prompt_tokens);
    Ok(selected)
}

fn parse_phase83_kv() -> Result<KvCacheEncoding, String> {
    match env::var(PHASE83_KV_ENV) {
        Ok(value) if value.eq_ignore_ascii_case("fp16") => Ok(KvCacheEncoding::Fp16),
        Ok(value)
            if value.eq_ignore_ascii_case("mxfp8")
                || value.eq_ignore_ascii_case("mxfp8-e4")
                || value.eq_ignore_ascii_case("kv-mxfp8-e4") =>
        {
            Ok(KvCacheEncoding::Mxfp8E4)
        }
        Ok(value) => Err(format!(
            "{PHASE83_KV_ENV} must be fp16 or mxfp8, got {value:?}"
        )),
        Err(env::VarError::NotPresent) => Err(format!(
            "{PHASE83_KV_ENV} is required in Phase83 mode; choose fp16 or mxfp8"
        )),
        Err(error) => Err(format!("cannot read {PHASE83_KV_ENV}: {error}")),
    }
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
        PHASE83_MODE_ENV,
        PHASE83_KV_ENV,
        PHASE83_ROWS_ENV,
        PHASE83_SAMPLING_ENV,
        PHASE83_REPLAY_ENV,
        PHASE83_FIXTURE_ONLY,
        PHASE83_MTP_ENV,
        PHASE83_MTP_WIDTH_ENV,
        "SLLM_MATMUL_FORCE_BASELINE",
        "SLLM_MATMUL_GFX1030_ROCBLAS_SOLUTION_445",
        "SLLM_MATMUL_GFX1030_SHORT_MIXED",
        "SLLM_NVFP4_W4A4_FORCE_BASELINE",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_COL8",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA",
        "SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_F16_STAGING",
        "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4",
        "SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_ACTIVATION_SHARED",
        "SLLM_NVFP4_W4A4_DECODE_FORCE_LDS_F32_LUT",
        "SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8",
        "SLLM_FP8_OUTER_PREFILL_FORCE_BASELINE",
        "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2",
        "SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2_64X64",
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
    fn cpu_stat_parser_handles_parentheses_and_missing_fields() {
        assert_eq!(
            parse_process_cpu_ticks("12 (worker (test)) R 0 0 0 0 0 0 0 0 0 0 17 9 0"),
            Some(26)
        );
        assert_eq!(parse_process_cpu_ticks("12 (worker) R"), None);
        assert_eq!(cpu_delta(Some(3), Some(8)), Some(5));
        assert_eq!(cpu_delta(Some(8), Some(3)), None);
    }

    #[test]
    fn phase81_replay_inputs_do_not_depend_on_sampler_output() {
        for step in [1, 19, 20, 63, 64, 127] {
            let replay = SamplingBench {
                mode: SamplingMode::GpuFixed,
                replay_inputs: true,
                seed: 123,
            };
            assert_eq!(
                replay.input_token(step, 0),
                replay.input_token(step, 248319)
            );
            let free = SamplingBench {
                replay_inputs: false,
                ..replay
            };
            assert_eq!(free.input_token(step, 41), 41);
            assert!((0..QWEN35_VOCAB_SIZE as i32).contains(&replay.input_token(step, 0)));
        }
    }

    #[test]
    fn phase83_mtp_config_is_explicit_and_bounded() {
        assert_eq!(
            parse_mtp_config(None, None).unwrap(),
            MtpConfig {
                enabled: false,
                draft_width: 0
            }
        );
        assert_eq!(parse_mtp_config(Some("on"), None).unwrap().draft_width, 2);
        assert_eq!(
            parse_mtp_config(Some("on"), Some("3")).unwrap(),
            MtpConfig {
                enabled: true,
                draft_width: 3
            }
        );
        assert!(parse_mtp_config(Some("on"), Some("0")).is_err());
        assert!(parse_mtp_config(Some("on"), Some("4")).is_err());
        assert!(parse_mtp_config(Some("maybe"), Some("1")).is_err());
        for width in 1..=3 {
            let config = parse_mtp_config(Some("on"), Some(&width.to_string())).unwrap();
            assert_eq!(config.draft_width, width);
        }
        assert_eq!(
            parse_mtp_config(Some("on"), Some("3"))
                .unwrap()
                .report()
                .supported_draft_widths,
            [1, 2, 3]
        );
    }

    #[test]
    fn phase83_mtp_accounting_excludes_rejected_draft_rows() {
        let report = build_mtp_run_report(3, 2, 6, 4, 6, 1, 6, 7, true).unwrap();
        assert_eq!(report.rejected_draft_tokens, 2);
        assert_eq!(report.committed_target_rows, 6);
        assert_eq!(report.prefill_selected_tokens, 1);
        assert_eq!(report.committed_decode_tokens, 6);
        assert_eq!(report.committed_output_tokens, 7);
        // A final all-accept proposal can be truncated after its first
        // published row; accepted-but-unpublished rows stay out of output.
        let truncated = build_mtp_run_report(3, 1, 3, 1, 1, 1, 1, 2, true).unwrap();
        assert_eq!(truncated.committed_output_tokens, 2);
        // Early EOS after prefill has no speculative block or decode row.
        let early_eos = build_mtp_run_report(1, 0, 0, 0, 0, 1, 0, 1, true).unwrap();
        assert_eq!(early_eos.committed_output_tokens, 1);
        assert!(build_mtp_run_report(3, 2, 6, 4, 7, 1, 7, 8, true).is_err());
        assert!(build_mtp_run_report(3, 2, 6, 4, 6, 1, 6, 7, false).is_err());
    }

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
