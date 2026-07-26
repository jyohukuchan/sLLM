use serde_json::json;
use std::env;
use std::ffi::{c_char, c_int, c_void};
use std::io::{self, Write};
use std::time::Instant;
use ullm_engine::sq8_embedding_runtime::QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::sq8_layer_runtime::{
    QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV,
    QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV,
    QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV,
};
use ullm_engine::sq8_model_head_runtime::{
    QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV, QWEN3_14B_VOCAB_SIZE,
    validate_qwen3_14b_sq8_r9700_device_info,
};
use ullm_engine::sq8_serving_runtime::{
    Qwen3Sq8ServingSession, Sq8CancellationToken, Sq8ServingAdvance, Sq8ServingPrefillMode,
    Sq8ServingRequest, Sq8ServingRuntimeStatus, load_qwen3_14b_sq8_serving_norms,
};
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_runtime_sys::{RuntimeContext, RuntimeStream, device_count, device_info};

const ARTIFACT: &str = "/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact";
const PACKAGE: &str = "/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package";
const UPLOAD_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_PROMPT_TOKENS: usize = 1024;
const DEFAULT_DECODE_WARMUP: usize = 4;
const DEFAULT_DECODE_MEASURED: usize = 16;
const RTLD_NOW: c_int = 2;
const ROCTX_LIBRARY: &[u8] = b"librocprofiler-sdk-roctx.so.1\0";
const ROCTX_PAUSE: &[u8] = b"roctxProfilerPause\0";
const ROCTX_RESUME: &[u8] = b"roctxProfilerResume\0";

type ProfileControlFn = unsafe extern "C" fn(u64) -> c_int;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Decode,
    Prefill,
}

#[derive(Debug)]
struct Args {
    phase: Phase,
    prompt_tokens: usize,
    warmup_steps: usize,
    measured_steps: usize,
    repeats: usize,
}

struct ProfileControl {
    _handle: *mut c_void,
    pause: ProfileControlFn,
    resume: ProfileControlFn,
}

unsafe impl Send for ProfileControl {}
unsafe impl Sync for ProfileControl {}

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ullm-sq8-r9700-phase0-profile: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let prompt = deterministic_tokens(args.prompt_tokens)?;
    require_environment()?;
    let runtime_index = isolated_r9700()?;
    let device = device_info(runtime_index).map_err(|error| error.to_string())?;

    // The SDK ROCTx library is deliberately used rather than the legacy compatibility DSO:
    // rocprofv3 observes this implementation's control and range events.
    ullm_engine::roctx::enable()?;
    let control = load_profile_control()?;

    // `rocprofv3 --selected-regions` starts its data context paused. Calling Pause before the
    // first Resume makes ROCm 7.2.1 attempt to stop a context that does not yet exist, so model
    // load/warmup remain excluded by the profiler mode itself. The only control calls issued by
    // this driver are paired Resume/Pause around a measured region.

    let artifact = read_sq8_canonical_artifact(ARTIFACT)?;
    let norms = load_qwen3_14b_sq8_serving_norms(PACKAGE, UPLOAD_CHUNK_BYTES)
        .map_err(|error| error.to_string())?;
    let mut context = RuntimeContext::create(runtime_index).map_err(|error| error.to_string())?;
    let mut stream = context.create_stream().map_err(|error| error.to_string())?;
    let load_started = Instant::now();
    let mut session = Qwen3Sq8ServingSession::load_with_prefill_mode(
        &mut context,
        &mut stream,
        &artifact,
        PACKAGE,
        norms,
        UPLOAD_CHUNK_BYTES,
        Sq8ServingPrefillMode::FixedM128Chunks,
    )
    .map_err(|error| error.to_string())?;
    let load_seconds = load_started.elapsed().as_secs_f64();

    run_complete_request(
        &mut session,
        &mut stream,
        "phase0-unprofiled-warmup",
        prompt.clone(),
        1,
    )?;

    write_json_line(json!({
        "schema_version": "ullm.sq8.r9700.prefill_comparison.driver.v1",
        "event": "configuration",
        "phase": phase_name(args.phase),
        "prompt_tokens": args.prompt_tokens,
        "decode_warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "repeats": args.repeats,
        "unprofiled_warmup": {
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": 1,
        },
        "prefill_mode": "m128-chunk128",
        "scope": "rocprofv3 --selected-regions starts paused; model load/warmup/prefill seed remain outside the selected region, and roctxProfilerResume/roctxProfilerPause surround only each measured phase",
        "load_seconds_unprofiled": load_seconds,
        "device": {
            "runtime_index": runtime_index,
            "device_id": device.device_id,
            "backend": device.backend,
            "name": device.name,
            "gcn_arch_name": device.gcn_arch_name,
            "compute_major": device.compute_major,
            "compute_minor": device.compute_minor,
            "total_global_mem": device.total_global_mem,
        },
        "roctx": {
            "library": "librocprofiler-sdk-roctx.so.1",
            "initial_state": "selected_regions_starts_paused; no pre-load Pause call",
        },
        "load": {
            "prefill_implementation": session.load_report().prefill_implementation,
            "artifact_content_sha256": session.load_report().artifact_content_sha256,
            "package_manifest_sha256": session.load_report().package_manifest_sha256,
            "total_kv_cache_bytes": session.load_report().total_kv_cache_bytes,
        }
    }))?;

    let mut elapsed_seconds = Vec::with_capacity(args.repeats);
    for repeat_index in 0..args.repeats {
        let result = match args.phase {
            Phase::Decode => profile_decode_repeat(
                &mut session,
                &mut stream,
                &control,
                &prompt,
                args.warmup_steps,
                args.measured_steps,
                repeat_index,
            )?,
            Phase::Prefill => profile_prefill_repeat(
                &mut session,
                &mut stream,
                &control,
                &prompt,
                repeat_index,
            )?,
        };
        elapsed_seconds.push(result.0);
        write_json_line(result.1)?;
    }

    let sum = elapsed_seconds.iter().sum::<f64>();
    let mean = sum / elapsed_seconds.len() as f64;
    let min = elapsed_seconds.iter().copied().fold(f64::INFINITY, f64::min);
    let max = elapsed_seconds.iter().copied().fold(0.0_f64, f64::max);
    let units = match args.phase {
        Phase::Decode => args.measured_steps,
        Phase::Prefill => args.prompt_tokens,
    };
    write_json_line(json!({
        "schema_version": "ullm.sq8.r9700.handwritten_phase0.driver.v1",
        "event": "summary",
        "phase": phase_name(args.phase),
        "repeats": args.repeats,
        "units_per_repeat": units,
        "mean_seconds": mean,
        "min_seconds": min,
        "max_seconds": max,
        "mean_units_per_second": (units as f64) / mean,
        "all_seconds": elapsed_seconds,
    }))?;
    Ok(())
}

fn profile_decode_repeat(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut RuntimeStream,
    control: &ProfileControl,
    prompt: &[usize],
    warmup_steps: usize,
    measured_steps: usize,
    repeat_index: usize,
) -> Result<(f64, serde_json::Value), String> {
    let total_new_tokens = 1usize
        .checked_add(warmup_steps)
        .and_then(|value| value.checked_add(measured_steps))
        .ok_or_else(|| "decode token count overflows".to_string())?;
    seed_decode_request(session, stream, repeat_index, prompt, total_new_tokens)?;
    for step_index in 0..warmup_steps {
        advance_decode(session, stream, false, step_index, warmup_steps)?;
    }
    let cache_start = session
        .snapshot()
        .cache_lengths
        .first()
        .copied()
        .ok_or_else(|| "decode snapshot has no cache lengths".to_string())?;
    let label = format!(
        "ullm.sq8.phase0.decode.steady.v1/repeat={repeat_index}/cache_start={cache_start}/steps={measured_steps}"
    );
    let resume_return = control.resume()?;
    let started = Instant::now();
    {
        let _range = ullm_engine::roctx::range(&label);
        for step_index in 0..measured_steps {
            advance_decode(
                session,
                stream,
                step_index + 1 == measured_steps,
                step_index,
                measured_steps,
            )?;
        }
    }
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let pause_return = control.pause()?;
    if session.status() != Sq8ServingRuntimeStatus::Finishing {
        return Err(format!(
            "profiled decode did not finish its bounded request: {:?}",
            session.status()
        ));
    }
    let final_cache_len = session
        .snapshot()
        .cache_lengths
        .first()
        .copied()
        .ok_or_else(|| "decode final snapshot has no cache lengths".to_string())?;
    session
        .finish_and_reset_synchronized(stream)
        .map_err(|error| error.to_string())?;
    Ok((
        elapsed_seconds,
        json!({
            "schema_version": "ullm.sq8.r9700.handwritten_phase0.driver.v1",
            "event": "measured_region",
            "phase": "decode",
            "repeat_index": repeat_index,
            "label": label,
            "cache_len_start": cache_start,
            "cache_len_end": final_cache_len,
            "measured_steps": measured_steps,
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": (measured_steps as f64) / elapsed_seconds,
            "roctx_resume_return": resume_return,
            "roctx_pause_return": pause_return,
            "excluded": ["model_load", "unprofiled_warmup_request", "prefill_seed", "decode_warmup_steps", "finish_and_reset"],
        }),
    ))
}

fn profile_prefill_repeat(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut RuntimeStream,
    control: &ProfileControl,
    prompt: &[usize],
    repeat_index: usize,
) -> Result<(f64, serde_json::Value), String> {
    let request_id = format!("phase0-prefill-{repeat_index}");
    session
        .start(
            Sq8ServingRequest::greedy_ignore_eos_for_testing(request_id, prompt.to_vec(), 1),
            Sq8CancellationToken::new(),
            stream,
        )
        .map_err(|error| error.to_string())?;
    if session.status() != Sq8ServingRuntimeStatus::Prefilling {
        return Err("prefill request did not enter Prefilling".to_string());
    }
    let label = format!(
        "ullm.sq8.phase0.prefill.m128.v1/repeat={repeat_index}/prompt_tokens={}",
        prompt.len()
    );
    let resume_return = control.resume()?;
    let started = Instant::now();
    let mut calls = 0usize;
    {
        let _range = ullm_engine::roctx::range(&label);
        while session.status() == Sq8ServingRuntimeStatus::Prefilling {
            match session
                .advance_synchronized(stream)
                .map_err(|error| error.to_string())?
            {
                Sq8ServingAdvance::PromptProgress { .. } | Sq8ServingAdvance::Token { .. } => {
                    calls += 1;
                }
                Sq8ServingAdvance::CancellationObserved => {
                    return Err("prefill profile unexpectedly observed cancellation".to_string());
                }
            }
        }
    }
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let pause_return = control.pause()?;
    if session.status() != Sq8ServingRuntimeStatus::Finishing {
        return Err(format!(
            "profiled prefill did not finish its one-token request: {:?}",
            session.status()
        ));
    }
    session
        .finish_and_reset_synchronized(stream)
        .map_err(|error| error.to_string())?;
    Ok((
        elapsed_seconds,
        json!({
            "schema_version": "ullm.sq8.r9700.handwritten_phase0.driver.v1",
            "event": "measured_region",
            "phase": "prefill",
            "repeat_index": repeat_index,
            "label": label,
            "prompt_tokens": prompt.len(),
            "prefill_advance_calls": calls,
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": (prompt.len() as f64) / elapsed_seconds,
            "roctx_resume_return": resume_return,
            "roctx_pause_return": pause_return,
            "excluded": ["model_load", "unprofiled_warmup_request", "request_start", "finish_and_reset"],
        }),
    ))
}

fn run_complete_request(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut RuntimeStream,
    request_id: &str,
    prompt: Vec<usize>,
    max_new_tokens: usize,
) -> Result<(), String> {
    session
        .start(
            Sq8ServingRequest::greedy_ignore_eos_for_testing(
                request_id,
                prompt,
                max_new_tokens,
            ),
            Sq8CancellationToken::new(),
            stream,
        )
        .map_err(|error| error.to_string())?;
    while session.status() != Sq8ServingRuntimeStatus::Finishing {
        match session
            .advance_synchronized(stream)
            .map_err(|error| error.to_string())?
        {
            Sq8ServingAdvance::PromptProgress { .. } | Sq8ServingAdvance::Token { .. } => {}
            Sq8ServingAdvance::CancellationObserved => {
                return Err("unprofiled warmup unexpectedly observed cancellation".to_string());
            }
        }
    }
    session
        .finish_and_reset_synchronized(stream)
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn seed_decode_request(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut RuntimeStream,
    repeat_index: usize,
    prompt: &[usize],
    max_new_tokens: usize,
) -> Result<(), String> {
    session
        .start(
            Sq8ServingRequest::greedy_ignore_eos_for_testing(
                format!("phase0-decode-{repeat_index}"),
                prompt.to_vec(),
                max_new_tokens,
            ),
            Sq8CancellationToken::new(),
            stream,
        )
        .map_err(|error| error.to_string())?;
    while session.status() == Sq8ServingRuntimeStatus::Prefilling {
        match session
            .advance_synchronized(stream)
            .map_err(|error| error.to_string())?
        {
            Sq8ServingAdvance::PromptProgress { .. } | Sq8ServingAdvance::Token { .. } => {}
            Sq8ServingAdvance::CancellationObserved => {
                return Err("decode seed unexpectedly observed cancellation".to_string());
            }
        }
    }
    if session.status() != Sq8ServingRuntimeStatus::Decoding {
        return Err(format!(
            "decode seed did not end at Decoding: {:?}",
            session.status()
        ));
    }
    Ok(())
}

fn advance_decode(
    session: &mut Qwen3Sq8ServingSession,
    stream: &mut RuntimeStream,
    should_finish: bool,
    step_index: usize,
    total_steps: usize,
) -> Result<(), String> {
    let terminal = match session
        .advance_synchronized(stream)
        .map_err(|error| error.to_string())?
    {
        Sq8ServingAdvance::Token { terminal_reason, .. } => terminal_reason.is_some(),
        Sq8ServingAdvance::PromptProgress { .. } => {
            return Err(format!("decode step {step_index} returned prompt progress"));
        }
        Sq8ServingAdvance::CancellationObserved => {
            return Err(format!("decode step {step_index} unexpectedly observed cancellation"));
        }
    };
    if terminal != should_finish {
        return Err(format!(
            "decode step {step_index}/{total_steps} terminal mismatch: terminal={terminal} expected={should_finish}"
        ));
    }
    Ok(())
}

fn deterministic_tokens(count: usize) -> Result<Vec<usize>, String> {
    if count == 0 {
        return Err("prompt token count must be positive".to_string());
    }
    Ok((0..count)
        .map(|index| (17usize + index.wrapping_mul(7919)) % QWEN3_14B_VOCAB_SIZE)
        .collect())
}

fn parse_args() -> Result<Args, String> {
    let mut phase = None;
    let mut prompt_tokens = DEFAULT_PROMPT_TOKENS;
    let mut warmup_steps = DEFAULT_DECODE_WARMUP;
    let mut measured_steps = DEFAULT_DECODE_MEASURED;
    let mut repeats = 1usize;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values.next().ok_or_else(usage)?;
        match flag.as_str() {
            "--phase" => {
                phase = Some(match value.as_str() {
                    "decode" => Phase::Decode,
                    "prefill" => Phase::Prefill,
                    _ => return Err(format!("invalid --phase {value:?}; {}", usage())),
                });
            }
            "--prompt-tokens" => prompt_tokens = parse_positive("--prompt-tokens", &value)?,
            "--warmup-steps" => warmup_steps = parse_nonnegative("--warmup-steps", &value)?,
            "--measured-steps" => measured_steps = parse_positive("--measured-steps", &value)?,
            "--repeats" => repeats = parse_positive("--repeats", &value)?,
            _ => return Err(format!("unknown argument {flag:?}; {}", usage())),
        }
    }
    let phase = phase.ok_or_else(usage)?;
    if prompt_tokens > 4096 {
        return Err("--prompt-tokens must not exceed 4096".to_string());
    }
    if phase == Phase::Decode
        && prompt_tokens
            .checked_add(1)
            .and_then(|value| value.checked_add(warmup_steps))
            .and_then(|value| value.checked_add(measured_steps))
            .is_none_or(|value| value > 4096)
    {
        return Err("decode prompt plus generated tokens exceeds the 4096-token context".to_string());
    }
    Ok(Args {
        phase,
        prompt_tokens,
        warmup_steps,
        measured_steps,
        repeats,
    })
}

fn usage() -> String {
    "usage: ullm-sq8-r9700-phase0-profile --phase decode|prefill [--prompt-tokens N] [--warmup-steps N] [--measured-steps N] [--repeats N]".to_string()
}

fn parse_positive(label: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be positive"));
    }
    Ok(parsed)
}

fn parse_nonnegative(label: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))
}

fn require_environment() -> Result<(), String> {
    if env::var("HIP_VISIBLE_DEVICES").ok().as_deref() != Some("1") {
        return Err("HIP_VISIBLE_DEVICES must be exactly 1 (R9700 isolation)".to_string());
    }
    let mut names = QWEN3_14B_SQ8_REQUIRED_HIP_KERNEL_ENV
        .into_iter()
        .chain(QWEN3_14B_SQ8_PAGED_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_MODEL_HEAD_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_EMBEDDING_REQUIRED_HIP_KERNEL_ENV)
        .chain(QWEN3_14B_SQ8_PREFILL_CHUNK_REQUIRED_HIP_KERNEL_ENV)
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    let missing = names
        .into_iter()
        .filter(|name| env::var(name).ok().as_deref() != Some("1"))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!("required SQ8 HIP guards are not all 1: {}", missing.join(",")))
    }
}

fn isolated_r9700() -> Result<u32, String> {
    let mut devices = Vec::new();
    for index in 1..device_count().map_err(|error| error.to_string())? {
        let info = device_info(index).map_err(|error| error.to_string())?;
        if info.backend == "hip" {
            devices.push((index, info));
        }
    }
    if devices.len() != 1 {
        return Err(format!("expected exactly one visible HIP device, found {}", devices.len()));
    }
    let (runtime_index, info) = devices.pop().expect("one visible HIP device");
    validate_qwen3_14b_sq8_r9700_device_info(&info)?;
    if info.device_id != 0 {
        return Err(format!("isolated R9700 must be HIP device 0, got {}", info.device_id));
    }
    Ok(runtime_index)
}

impl ProfileControl {
    fn pause(&self) -> Result<i32, String> {
        // SAFETY: resolved from the retained SDK ROCTx DSO with the exact C ABI.
        Ok(unsafe { (self.pause)(0) })
    }

    fn resume(&self) -> Result<i32, String> {
        // SAFETY: resolved from the retained SDK ROCTx DSO with the exact C ABI.
        Ok(unsafe { (self.resume)(0) })
    }
}

fn load_profile_control() -> Result<ProfileControl, String> {
    // SAFETY: the library name is static and NUL-terminated; RTLD_NOW is valid.
    let handle = unsafe { dlopen(ROCTX_LIBRARY.as_ptr().cast::<c_char>(), RTLD_NOW) };
    if handle.is_null() {
        return Err(format!("failed to load SDK ROCTx: {}", dl_error_message()));
    }
    let pause = load_symbol::<ProfileControlFn>(handle, ROCTX_PAUSE)?;
    let resume = load_symbol::<ProfileControlFn>(handle, ROCTX_RESUME)?;
    Ok(ProfileControl {
        _handle: handle,
        pause,
        resume,
    })
}

fn load_symbol<T>(handle: *mut c_void, name: &[u8]) -> Result<T, String> {
    // SAFETY: `handle` is an open dynamic-library handle and `name` is NUL-terminated.
    let symbol = unsafe { dlsym(handle, name.as_ptr().cast::<c_char>()) };
    if symbol.is_null() {
        return Err(format!(
            "SDK ROCTx is missing {}: {}",
            String::from_utf8_lossy(&name[..name.len().saturating_sub(1)]),
            dl_error_message()
        ));
    }
    // SAFETY: callers provide the known C ABI function type for the named symbol.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&symbol) })
}

fn dl_error_message() -> String {
    // SAFETY: dlerror returns either null or a NUL-terminated loader-owned string.
    let error = unsafe { dlerror() };
    if error.is_null() {
        "dynamic loader returned no detail".to_string()
    } else {
        // SAFETY: checked non-null above; POSIX guarantees a NUL-terminated error string.
        unsafe { std::ffi::CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Decode => "decode",
        Phase::Prefill => "prefill",
    }
}

fn write_json_line(value: serde_json::Value) -> Result<(), String> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    serde_json::to_writer(&mut stdout, &value).map_err(|error| error.to_string())?;
    stdout.write_all(b"\n").map_err(|error| error.to_string())?;
    stdout.flush().map_err(|error| error.to_string())
}
