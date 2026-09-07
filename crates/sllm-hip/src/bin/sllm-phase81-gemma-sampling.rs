//! Matched A/B/C sampling timings for the reviewed dense Gemma4 GGUF path.
//!
//! A, B, and C execute the same prompt and teacher-forced continuation in
//! fresh requests.  A uses device Argmax, B reads the full terminal logits
//! row and applies the backend-neutral sampler chain on the host, and C uses
//! the bounded device token selector.  Logits are validated and discarded;
//! this binary never writes a vocabulary-sized logits row to its JSON.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, DeviceTokenSelectorRequestV1, ExecutionSessionRequest, GEMMA4_VOCAB_SIZE,
    Gemma4ResidentModel, OsSamplingRandom, SamplerChainConfigV1, SamplerChainV1,
    SamplingParametersV1, build_verified_gguf_gemma_weight_load_plan, parse_gemma4_model_lock,
    read_derived_gguf_lock, verify_derived_gguf,
};
use sllm_hip::HipBackend;
use std::{
    env, fs,
    path::PathBuf,
    process::ExitCode,
    sync::Arc,
    time::{Duration, Instant},
};

const TOP_K: usize = 64;
const TOP_P: f32 = 0.95;
const TEMPERATURE: f32 = 1.0;
const SEED: u64 = 123;
const WARMUPS: usize = 1;
const MEASURED: usize = 3;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(600);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Case {
    id: String,
    prompt: Vec<i32>,
    continuation: Vec<i32>,
}

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
    id: String,
    prompt_tokens: usize,
    continuation_tokens: usize,
    prompt_sha256: String,
    continuation_sha256: String,
}

#[derive(Serialize)]
struct SetupReport {
    verify_derived_gguf_ns: u128,
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
    selected_backend: &'static str,
    target: String,
    submission_count: u64,
    kernel_dispatch_count: u64,
    segment_count: u64,
    boundary_count: u64,
    projection_pack_submission_count: u64,
    projection_pack_candidate_count: u64,
    projection_pack_scale_mismatch_count: u64,
    projection_pack_override_disabled: bool,
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

fn parse_args() -> Result<[String; 6], String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 6 {
        return Err(
            "usage: sllm-phase81-gemma-sampling MODEL_LOCK GGUF DERIVED_LOCK TARGET FIXTURE_JSON OUTPUT_JSON (single visible GPU, device 0)"
                .to_owned(),
        );
    }
    let args: [String; 6] = args
        .try_into()
        .map_err(|_| "expected exactly six arguments".to_owned())?;
    if !matches!(args[3].as_str(), "gfx1030" | "gfx1201") {
        return Err("TARGET must be exactly gfx1030 or gfx1201".to_owned());
    }
    Ok(args)
}

fn parse_fixture(path: &str) -> Result<Case, Box<dyn std::error::Error>> {
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(path)?)?;
    if cases.len() != 1 {
        return Err("fixture must contain exactly one case".into());
    }
    let case = cases.into_iter().next().expect("one fixture case");
    if case.id.is_empty() || case.prompt.is_empty() || case.continuation.is_empty() {
        return Err("fixture id, prompt, and continuation must be nonempty".into());
    }
    let valid = |token: &i32| (0..GEMMA4_VOCAB_SIZE as i32).contains(token);
    if case
        .prompt
        .iter()
        .chain(&case.continuation)
        .any(|token| !valid(token))
    {
        return Err("fixture contains a token outside the Gemma vocabulary".into());
    }
    Ok(case)
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
        .prepare_device_selector(GEMMA4_VOCAB_SIZE as usize, None, SEED, counter)?)
}

fn selected_token(
    output: &sllm_core::Gemma4ExecutionOutput,
    mode: Mode,
    state: &mut Option<SamplingState>,
) -> Result<i32, Box<dyn std::error::Error>> {
    let token = *output
        .token_ids()
        .last()
        .ok_or("Gemma execution returned no selected token")?;
    if !(0..GEMMA4_VOCAB_SIZE as i32).contains(&token) {
        return Err("Gemma selected token is outside vocabulary".into());
    }
    match mode {
        Mode::Greedy => {
            if output.last_logits().is_some() || output.selection().is_some() {
                return Err("A greedy path returned logits or device selection metadata".into());
            }
        }
        Mode::HostFixed => {
            if output.selection().is_some() {
                return Err("B host path returned device selection metadata".into());
            }
            let logits = output
                .last_logits()
                .ok_or("B host path returned no logits")?;
            if logits.len() != GEMMA4_VOCAB_SIZE as usize || logits.iter().any(|v| !v.is_finite()) {
                return Err("B host path returned invalid terminal logits".into());
            }
            let sampling = state.as_mut().ok_or("B sampler state is missing")?;
            let selected =
                sampling
                    .chain
                    .select_token(token as u32, Some(logits), &mut sampling.random)?;
            return Ok(i32::try_from(selected)?);
        }
        Mode::DeviceFixed => {
            if output.last_logits().is_some() {
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
    resident: &Gemma4ResidentModel,
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
        Mode::HostFixed | Mode::DeviceFixed => Some(sampling_state(&case.prompt)?),
    };

    let prefill_cpu_started = process_cpu_ticks();
    let prefill_started = Instant::now();
    let prefill = match mode {
        Mode::Greedy => request.prefill(&case.prompt)?,
        Mode::HostFixed => request.prefill_with_last_logits(&case.prompt)?,
        Mode::DeviceFixed => {
            let selector = selector(state.as_ref().ok_or("C sampler state is missing")?, 0)?;
            request.prefill_with_device_selector(&case.prompt, &selector)?
        }
    };
    let prefill_elapsed = prefill_started.elapsed();
    let ttft = request_started.elapsed();
    let prefill_cpu = cpu_delta(prefill_cpu_started, process_cpu_ticks());
    let mut generated = vec![selected_token(&prefill, mode, &mut state)?];
    let decode_cpu_started = process_cpu_ticks();
    let decode_started = Instant::now();
    let mut selector_records = usize::from(prefill.selection().is_some());
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

    let audit = request.audit_snapshot()?;
    if audit.target() != target
        || audit.fallback_used()
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
    {
        return Err(format!("Gemma dispatch audit failed: {audit:?}").into());
    }
    let audit_report = AuditReport {
        selected_backend: "hip",
        target: audit.target().to_owned(),
        submission_count: audit.submission_count(),
        kernel_dispatch_count: audit.kernel_dispatch_count(),
        segment_count: audit.segment_count(),
        boundary_count: audit.boundary_count(),
        projection_pack_submission_count: audit.projection_pack_submission_count(),
        projection_pack_candidate_count: audit.projection_pack_candidate_count(),
        projection_pack_scale_mismatch_count: audit.projection_pack_scale_mismatch_count(),
        projection_pack_override_disabled: audit.projection_pack_override_disabled(),
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

fn run(args: [String; 6]) -> Result<Report, Box<dyn std::error::Error>> {
    let fixture = parse_fixture(&args[4])?;
    let lock = parse_gemma4_model_lock(&fs::read(&args[0])?)?;
    let verification_started = Instant::now();
    let verified = verify_derived_gguf(
        read_derived_gguf_lock(PathBuf::from(&args[2]))?,
        PathBuf::from(&args[1]),
    )?;
    let verify_ns = verification_started.elapsed().as_nanos();
    let plan_started = Instant::now();
    let (source, plan) = build_verified_gguf_gemma_weight_load_plan(&lock, verified)?;
    let plan_ns = plan_started.elapsed().as_nanos();
    let plan_digest_sha256 = format!("sha256:{:x}", Sha256::digest(plan.digest()));
    let hip_started = Instant::now();
    let backend = HipBackend::connect()?;
    let session = Arc::new(
        backend.open_execution_session(ExecutionSessionRequest::new(0, args[3].clone())?)?,
    );
    let hip_ns = hip_started.elapsed().as_nanos();
    let available_memory = session.available_memory_bytes()?;
    let resident_started = Instant::now();
    let resident = Gemma4ResidentModel::new_gguf_quantized(
        Arc::clone(&session),
        lock.clone(),
        plan,
        Arc::new(source),
        COMPLETION_TIMEOUT,
    )?;
    let resident_ns = resident_started.elapsed().as_nanos();
    let resident_memory = resident.memory_snapshot();
    if resident_memory.model_resident().current_bytes() == 0 || resident_memory.poisoned() {
        return Err("resident model allocation snapshot is invalid".into());
    }

    let mut rows = Vec::new();
    for mode in Mode::ALL {
        for (sample_kind, count) in [("warmup", WARMUPS), ("measured", MEASURED)] {
            for sample_index in 0..count {
                rows.push(run_one(
                    &session,
                    &resident,
                    &fixture,
                    &args[3],
                    mode,
                    sample_kind,
                    sample_index,
                )?);
            }
        }
    }
    let resident_memory_bytes = resident_memory.model_resident().current_bytes();
    drop(resident);
    let cleanup = session.shutdown(SHUTDOWN_TIMEOUT)?;
    let final_current = session.memory_snapshot().current_bytes();
    if final_current != 0 || cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
        return Err("Gemma sampling cleanup retained runtime resources".into());
    }
    Ok(Report {
        schema_version: "phase81-gemma-dense-sampling-v1",
        state: "PASS",
        target: args[3].clone(),
        device_index: 0,
        model_fingerprint: lock.fingerprint().to_owned(),
        plan_digest_sha256,
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
            id: fixture.id,
            prompt_tokens: fixture.prompt.len(),
            continuation_tokens: fixture.continuation.len(),
            prompt_sha256: hash_tokens(&fixture.prompt),
            continuation_sha256: hash_tokens(&fixture.continuation),
        },
        setup: SetupReport {
            verify_derived_gguf_ns: verify_ns,
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
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("Phase 81 Gemma sampling argument error: {error}");
            return ExitCode::from(2);
        }
    };
    match run(args) {
        Ok(report) => {
            let output = env::args().nth(6).expect("output argument");
            match fs::write(&output, serde_json::to_vec_pretty(&report).unwrap()) {
                Ok(()) => {
                    println!("PASS {output}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("Phase 81 Gemma sampling output failed: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Err(error) => {
            eprintln!("Phase 81 Gemma sampling failed: {error}");
            ExitCode::FAILURE
        }
    }
}
