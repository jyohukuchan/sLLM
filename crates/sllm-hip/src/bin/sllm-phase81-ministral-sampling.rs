//! Matched A/B/C sampling timings for the reviewed Ministral 3 GGUF path.
//!
//! The harness replays one fixed 17-token prompt and 17-token continuation.
//! A uses device Argmax, B reads the terminal BF16 row and applies the
//! backend-neutral fixed sampler on the host, and C uses the common device
//! selector with the Ministral profile (K0, top-p .95).  The BF16 row is
//! validated and discarded; it is never serialized into the report.

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, DeviceTokenSelectorRequestV1, ExecutionSessionRequest, MINISTRAL3_GRAPH_VOCAB_SIZE,
    MINISTRAL3_WEIGHT_LOCK_FINGERPRINT, Ministral3ExecutionOutput, Ministral3ResidentModel,
    OsSamplingRandom, SamplerChainConfigV1, SamplerChainV1, SamplingParametersV1,
    build_verified_ministral3_weight_load_plan, open_and_verify_official_ministral3_gguf,
    parse_ministral3_model_lock,
};
use sllm_hip::HipBackend;
use std::{
    env, fs,
    path::PathBuf,
    process::ExitCode,
    sync::Arc,
    time::{Duration, Instant},
};

const DEFAULT_GGUF: &str = "/home/homelab1/.cache/sllm/phase81-ministral3-official-bf16/Ministral-3-3B-Instruct-2512-BF16.gguf";
const MODEL_LOCK: &[u8] = include_bytes!(
    "../../../../docs/models/locks/ministral3-3b-instruct-2512-official-bf16-gguf.json"
);
const TOP_K: usize = 0;
const TOP_P: f32 = 0.95;
const TEMPERATURE: f32 = 1.0;
const SEED: u64 = 123;
const WARMUPS: usize = 1;
const MEASURED: usize = 3;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(600);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);

// Includes odd and non-power-of-two token IDs while keeping every ID inside
// the reviewed 131072-token Ministral vocabulary.
const PROMPT: [i32; 17] = [
    2, 106, 1_645, 108, 9_259, 23_676, 563, 107, 17, 23, 42, 255, 256, 257, 4_097, 65_537, 131_071,
];
const CONTINUATION: [i32; 17] = [
    42, 17, 23, 255, 256, 257, 4_097, 65_536, 65_537, 70_000, 70_001, 70_003, 90_000, 100_000,
    120_000, 131_070, 1,
];

#[derive(Clone, Debug)]
struct Case {
    id: &'static str,
    prompt: &'static [i32],
    continuation: &'static [i32],
}

const CASE: Case = Case {
    id: "ministral3-phase81-fixed-17x17",
    prompt: &PROMPT,
    continuation: &CONTINUATION,
};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Greedy,
    HostFixed,
    DeviceFixed,
}

impl Mode {
    const ALL: [Self; 3] = [Self::Greedy, Self::HostFixed, Self::DeviceFixed];
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    model_fingerprint: String,
    plan_digest_sha256: String,
    gguf_lfs_sha256: &'static str,
    lock_fingerprint: String,
    kv_encoding: &'static str,
    sampling_profile: SamplingProfile,
    fixture: FixtureReport,
    setup: SetupReport,
    rows: Vec<RunReport>,
    cleanup: CleanupReport,
}

#[derive(Serialize)]
struct SamplingProfile {
    temperature: f32,
    top_p: f32,
    top_k: usize,
    penalties: &'static str,
    seed: u64,
    warmups: usize,
    measured: usize,
    teacher_forced: bool,
}

#[derive(Serialize)]
struct FixtureReport {
    id: &'static str,
    prompt_tokens: usize,
    continuation_tokens: usize,
    prompt_sha256: String,
    continuation_sha256: String,
}

#[derive(Serialize)]
struct SetupReport {
    verify_gguf_ns: u128,
    build_plan_ns: u128,
    hip_connect_and_session_ns: u128,
    resident_load_ns: u128,
    available_memory_bytes_before_load: Option<u64>,
    resident_memory_bytes: u64,
}

#[derive(Serialize)]
struct RunReport {
    mode: Mode,
    sample_kind: &'static str,
    sample_index: usize,
    timing: TimingReport,
    cpu: CpuReport,
    generated_tokens_sha256: String,
    generated_token_count: usize,
    selector_records: usize,
    audit: AuditReport,
    allocation_before_request_bytes: u64,
    allocation_while_request_bytes: u64,
    allocation_after_request_drop_bytes: u64,
}

#[derive(Serialize)]
struct TimingReport {
    request_setup_ns: u128,
    prefill_ns: u128,
    ttft_ns: u128,
    decode_ns: u128,
    e2e_ns: u128,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    tpot_ms: f64,
}

#[derive(Serialize)]
struct CpuReport {
    source: &'static str,
    prefill_user_system_ticks: Option<u64>,
    decode_user_system_ticks: Option<u64>,
}

#[derive(Serialize)]
struct AuditReport {
    selected_backend: u32,
    target: String,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
}

#[derive(Serialize)]
struct CleanupReport {
    request_cleanup_pass: bool,
    final_current_bytes: u64,
    shutdown_retryable_cleanup: usize,
    shutdown_durable_quarantine: usize,
}

struct SamplingState {
    chain: SamplerChainV1,
    random: OsSamplingRandom,
}

struct Arguments {
    gguf: PathBuf,
    target: String,
    output: PathBuf,
}

fn parse_args() -> Result<Arguments, String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let (gguf, target, output) = match args.as_slice() {
        [target, output] => (
            env::var_os("SLLM_PHASE81_MINISTRAL3_GGUF")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_GGUF)),
            target.clone(),
            PathBuf::from(output),
        ),
        [gguf, target, output] => (PathBuf::from(gguf), target.clone(), PathBuf::from(output)),
        _ => {
            return Err(
                "usage: sllm-phase81-ministral-sampling [GGUF] TARGET OUTPUT_JSON (default GGUF may be overridden by SLLM_PHASE81_MINISTRAL3_GGUF)"
                    .to_owned(),
            )
        }
    };
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("TARGET must be exactly gfx1030 or gfx1201".to_owned());
    }
    Ok(Arguments {
        gguf,
        target,
        output,
    })
}

fn hash_tokens(tokens: &[i32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

// /proc units are USER_HZ; preserve raw ticks and do not invent a conversion.
fn process_cpu_ticks() -> Option<u64> {
    let stat = fs::read_to_string("/proc/self/stat").ok()?;
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
    match (start, end) {
        (Some(start), Some(end)) => end.checked_sub(start),
        _ => None,
    }
}

fn sampling_state(prompt: &[i32]) -> Result<SamplingState, Box<dyn std::error::Error>> {
    let parameters = SamplingParametersV1::new(TEMPERATURE, TOP_P, 0.0, 0.0)?;
    let config = SamplerChainConfigV1::new(parameters).with_top_k(TOP_K)?;
    let prior = prompt.iter().map(|&token| token as u32).collect::<Vec<_>>();
    Ok(SamplingState {
        chain: SamplerChainV1::new(config, &prior)?,
        random: OsSamplingRandom::for_parameters_and_seed(parameters, Some(SEED))?,
    })
}

fn selector(
    state: &SamplingState,
    counter: u64,
) -> Result<DeviceTokenSelectorRequestV1, Box<dyn std::error::Error>> {
    Ok(state
        .chain
        .prepare_device_selector(MINISTRAL3_GRAPH_VOCAB_SIZE, None, SEED, counter)?)
}

fn bf16_logits(bits: &[u16]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if bits.len() != MINISTRAL3_GRAPH_VOCAB_SIZE {
        return Err(format!(
            "terminal BF16 row has {} values, expected {}",
            bits.len(),
            MINISTRAL3_GRAPH_VOCAB_SIZE
        )
        .into());
    }
    let values = bits
        .iter()
        .map(|&value| f32::from_bits(u32::from(value) << 16))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("terminal BF16 row contains a non-finite value".into());
    }
    Ok(values)
}

fn selected_token(
    output: &Ministral3ExecutionOutput,
    mode: Mode,
    state: &mut Option<SamplingState>,
) -> Result<i32, Box<dyn std::error::Error>> {
    let token = *output
        .token_ids()
        .last()
        .ok_or("Ministral3 execution returned no selected token")?;
    if !(0..MINISTRAL3_GRAPH_VOCAB_SIZE as i32).contains(&token) {
        return Err("Ministral3 selected token is outside vocabulary".into());
    }
    match mode {
        Mode::Greedy => {
            if output.last_logits_bf16().is_some() || output.selection().is_some() {
                return Err("A greedy path returned logits or device selection metadata".into());
            }
        }
        Mode::HostFixed => {
            if output.selection().is_some() {
                return Err("B host path returned device selection metadata".into());
            }
            let bits = output
                .last_logits_bf16()
                .ok_or("B host path returned no terminal logits")?;
            let logits = bf16_logits(bits)?;
            let sampling = state.as_mut().ok_or("B sampler state is missing")?;
            let selected =
                sampling
                    .chain
                    .select_token(token as u32, Some(&logits), &mut sampling.random)?;
            return Ok(i32::try_from(selected)?);
        }
        Mode::DeviceFixed => {
            if output.last_logits_bf16().is_some() {
                return Err("C device path returned full logits".into());
            }
            let selection = output
                .selection()
                .ok_or("C device path returned no selection metadata")?;
            if selection.token_id != token as u32 {
                return Err("C selected token disagrees with selection metadata".into());
            }
        }
    }
    Ok(token)
}

fn run_one(
    session: &Arc<sllm_core::ExecutionSession>,
    resident: &Ministral3ResidentModel,
    case: &Case,
    target: &str,
    mode: Mode,
    sample_kind: &'static str,
    sample_index: usize,
) -> Result<RunReport, Box<dyn std::error::Error>> {
    let before_request = session.memory_snapshot();
    let request_started = Instant::now();
    let capacity = case
        .prompt
        .len()
        .checked_add(case.continuation.len())
        .and_then(|length| length.checked_add(1))
        .ok_or("request capacity overflow")?;
    let mut request = resident.new_request(case.prompt.len() as u64, capacity as u64)?;
    let request_setup = request_started.elapsed();
    let mut state = match mode {
        Mode::Greedy => None,
        Mode::HostFixed | Mode::DeviceFixed => Some(sampling_state(case.prompt)?),
    };

    let prefill_cpu_started = process_cpu_ticks();
    let prefill_started = Instant::now();
    let prefill = match mode {
        Mode::Greedy => request.prefill(case.prompt)?,
        Mode::HostFixed => request.prefill_with_last_logits(case.prompt)?,
        Mode::DeviceFixed => {
            let selector = selector(state.as_ref().ok_or("C sampler state is missing")?, 0)?;
            request.prefill_with_device_selector(case.prompt, &selector)?
        }
    };
    let prefill_elapsed = prefill_started.elapsed();
    let ttft = request_started.elapsed();
    let prefill_cpu = cpu_delta(prefill_cpu_started, process_cpu_ticks());
    let mut generated = vec![selected_token(&prefill, mode, &mut state)?];
    let mut selector_records = usize::from(prefill.selection().is_some());

    let decode_cpu_started = process_cpu_ticks();
    let decode_started = Instant::now();
    for (step, &input) in case.continuation.iter().enumerate() {
        let output = match mode {
            Mode::Greedy => request.decode(input)?,
            Mode::HostFixed => request.decode_with_last_logits(input)?,
            Mode::DeviceFixed => {
                let selector = selector(
                    state.as_ref().ok_or("C sampler state is missing")?,
                    (step + 1) as u64,
                )?;
                request.decode_with_device_selector(input, &selector)?
            }
        };
        selector_records += usize::from(output.selection().is_some());
        generated.push(selected_token(&output, mode, &mut state)?);
    }
    let decode_elapsed = decode_started.elapsed();
    let e2e_elapsed = request_started.elapsed();
    let decode_cpu = cpu_delta(decode_cpu_started, process_cpu_ticks());
    if prefill_elapsed.is_zero()
        || ttft.is_zero()
        || decode_elapsed.is_zero()
        || e2e_elapsed.is_zero()
    {
        return Err("a timing interval was zero".into());
    }
    if generated.len() != case.continuation.len() + 1 {
        return Err("teacher-forced output length differs from the expected replay".into());
    }

    let audit = request
        .last_audit()
        .ok_or("Ministral3 request did not publish dispatch audit")?;
    if audit.target() != target
        || audit.fallback_used()
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
    {
        return Err(format!("Ministral3 dispatch audit failed: {audit:?}").into());
    }
    let audit_report = AuditReport {
        selected_backend: audit.backend(),
        target: audit.target().to_owned(),
        submission_count: audit.submission_count(),
        kernel_dispatch_count: audit.kernel_dispatch_count(),
        fallback_used: audit.fallback_used(),
    };
    let while_request = session.memory_snapshot();
    drop(request);
    let after_request = session.memory_snapshot();
    if after_request.current_bytes() != before_request.current_bytes() {
        return Err("request cleanup changed the resident allocation baseline".into());
    }
    let prompt_tokens = case.prompt.len() as f64;
    let decode_tokens = case.continuation.len() as f64;
    let decode_seconds = decode_elapsed.as_secs_f64();
    let prefill_seconds = prefill_elapsed.as_secs_f64();
    Ok(RunReport {
        mode,
        sample_kind,
        sample_index,
        timing: TimingReport {
            request_setup_ns: request_setup.as_nanos(),
            prefill_ns: prefill_elapsed.as_nanos(),
            ttft_ns: ttft.as_nanos(),
            decode_ns: decode_elapsed.as_nanos(),
            e2e_ns: e2e_elapsed.as_nanos(),
            prefill_tokens_per_second: prompt_tokens / prefill_seconds,
            decode_tokens_per_second: decode_tokens / decode_seconds,
            tpot_ms: decode_seconds * 1_000.0 / decode_tokens,
        },
        cpu: CpuReport {
            source: "/proc/self/stat utime+stime; raw USER_HZ ticks",
            prefill_user_system_ticks: prefill_cpu,
            decode_user_system_ticks: decode_cpu,
        },
        generated_tokens_sha256: hash_tokens(&generated),
        generated_token_count: generated.len(),
        selector_records,
        audit: audit_report,
        allocation_before_request_bytes: before_request.current_bytes(),
        allocation_while_request_bytes: while_request.current_bytes(),
        allocation_after_request_drop_bytes: after_request.current_bytes(),
    })
}

fn run(arguments: Arguments) -> Result<Report, Box<dyn std::error::Error>> {
    let lock = parse_ministral3_model_lock(MODEL_LOCK)?;
    let verification_started = Instant::now();
    let verified = open_and_verify_official_ministral3_gguf(&arguments.gguf)?;
    let verify_ns = verification_started.elapsed().as_nanos();
    let plan_started = Instant::now();
    let (source, plan) = build_verified_ministral3_weight_load_plan(verified)?;
    if source.lock_fingerprint() != MINISTRAL3_WEIGHT_LOCK_FINGERPRINT
        || source.file_sha256() != lock.file_sha256()
    {
        return Err("Ministral3 weight source and tracked model lock differ".into());
    }
    let plan_digest_sha256 = format!("sha256:{:x}", Sha256::digest(plan.digest()));
    let plan_ns = plan_started.elapsed().as_nanos();

    let hip_started = Instant::now();
    let backend = HipBackend::connect()?;
    let session = backend
        .open_execution_session(ExecutionSessionRequest::new(0, arguments.target.clone())?)?;
    let hip_ns = hip_started.elapsed().as_nanos();
    let available_memory = session.available_memory_bytes()?;
    let resident_started = Instant::now();
    let resident = Ministral3ResidentModel::new_gguf(
        Arc::clone(&session),
        plan,
        Arc::new(source),
        COMPLETION_TIMEOUT,
    )?;
    let resident_ns = resident_started.elapsed().as_nanos();
    if resident.resident_bytes() == 0 {
        return Err("resident model allocation is empty".into());
    }

    let mut rows = Vec::new();
    for mode in Mode::ALL {
        for (sample_kind, count) in [("warmup", WARMUPS), ("measured", MEASURED)] {
            for sample_index in 0..count {
                rows.push(run_one(
                    &session,
                    &resident,
                    &CASE,
                    &arguments.target,
                    mode,
                    sample_kind,
                    sample_index,
                )?);
            }
        }
    }

    let resident_memory_bytes = resident.resident_bytes();
    drop(resident);
    let cleanup = session.shutdown(SHUTDOWN_TIMEOUT)?;
    let final_current = session.memory_snapshot().current_bytes();
    if final_current != 0 || cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("Ministral3 sampling cleanup retained runtime resources".into());
    }
    Ok(Report {
        schema_version: "phase81-ministral3-k0-sampling-v1",
        state: "PASS",
        target: arguments.target,
        device_index: 0,
        model_fingerprint: lock.fingerprint().to_owned(),
        plan_digest_sha256,
        gguf_lfs_sha256: sllm_core::MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256,
        lock_fingerprint: lock.fingerprint().to_owned(),
        kv_encoding: "fp16",
        sampling_profile: SamplingProfile {
            temperature: TEMPERATURE,
            top_p: TOP_P,
            top_k: TOP_K,
            penalties: "presence=0 frequency=0 repeat_penalty=1 repeat_last_n=0",
            seed: SEED,
            warmups: WARMUPS,
            measured: MEASURED,
            teacher_forced: true,
        },
        fixture: FixtureReport {
            id: CASE.id,
            prompt_tokens: CASE.prompt.len(),
            continuation_tokens: CASE.continuation.len(),
            prompt_sha256: hash_tokens(CASE.prompt),
            continuation_sha256: hash_tokens(CASE.continuation),
        },
        setup: SetupReport {
            verify_gguf_ns: verify_ns,
            build_plan_ns: plan_ns,
            hip_connect_and_session_ns: hip_ns,
            resident_load_ns: resident_ns,
            available_memory_bytes_before_load: available_memory,
            resident_memory_bytes,
        },
        rows,
        cleanup: CleanupReport {
            request_cleanup_pass: true,
            final_current_bytes: final_current,
            shutdown_retryable_cleanup: cleanup.retryable_cleanup,
            shutdown_durable_quarantine: cleanup.durable_quarantine,
        },
    })
}

fn main() -> ExitCode {
    let arguments = match parse_args() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("Phase 81 Ministral3 sampling argument error: {error}");
            return ExitCode::from(2);
        }
    };
    let output = arguments.output.clone();
    match run(arguments) {
        Ok(report) => match fs::write(&output, serde_json::to_vec_pretty(&report).unwrap()) {
            Ok(()) => {
                println!("PASS {}", output.display());
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("Phase 81 Ministral3 sampling output failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Phase 81 Ministral3 sampling failed: {error}");
            ExitCode::FAILURE
        }
    }
}
