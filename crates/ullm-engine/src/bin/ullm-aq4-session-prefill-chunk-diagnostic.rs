// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-chunk Qwen3.5 AQ4 session-prefill diagnostic on the isolated R9700.
//!
//! This intentionally drives the ordinary inference-session state machine. In particular, every
//! timed `prepare_advance()` invokes `Qwen35Aq4InferenceSession::prepare_prefill_chunk()`, whose
//! final chunk uses `min(total_len)` just as a worker request does. It is not a manual model
//! dispatch loop.
//!
//! Usage: `ullm-aq4-session-prefill-chunk-diagnostic [TOKEN_COUNT]`
//! If the positional argument is omitted, set `ULLM_AQ4_PREFILL_DIAGNOSTIC_TOKEN_COUNT`.

use serde_json::json;
use std::any::Any;
use std::env;
use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::ExitCode;
use std::time::Instant;

use ullm_engine::aq4_worker_backend::QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::inference_api::{CancellationToken, InferenceRequest, SamplingParams};
use ullm_engine::qwen35_aq4_head_runtime::PackageLmHeadMode;
use ullm_engine::qwen35_aq4_model_runtime::{
    Qwen35Aq4ModelLoadConfig, QWEN35_AQ4_CONTEXT_LENGTH, QWEN35_AQ4_KV_BLOCK_SIZE,
};
use ullm_engine::qwen35_aq4_session::{Qwen35Aq4InferenceSession, Qwen35Aq4SessionConfig};
use ullm_engine::worker_driver::{InferenceSession, PublishedAdvance, SessionAdvance};

const PACKAGE_DIR: &str = "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package";
const RUNTIME_HIP_DEVICE_INDEX: u32 = 1;
const REQUIRED_ARCHITECTURE: &str = "gfx1201";
const LOAD_CHUNK_BYTES: usize = 1024 * 1024;
const LM_HEAD_CHUNK_ROWS: usize = 8192;
const PREFILL_CHUNK_WIDTH: usize = 128;
const EOS_TOKEN_IDS: [usize; 2] = [248_044, 248_046];
const TOKEN_COUNT_ENV: &str = "ULLM_AQ4_PREFILL_DIAGNOSTIC_TOKEN_COUNT";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-aq4-session-prefill-chunk-diagnostic: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let token_count = token_count_from_caller()?;
    require_caller_environment()?;
    require_isolated_r9700("before model load")?;

    let session_config = Qwen35Aq4SessionConfig::greedy(1, EOS_TOKEN_IDS.to_vec())
        .with_prefill_chunk_tokens(PREFILL_CHUNK_WIDTH)?;
    let mut session = Qwen35Aq4InferenceSession::load(
        Qwen35Aq4ModelLoadConfig {
            package_dir: PACKAGE_DIR.into(),
            device_index: RUNTIME_HIP_DEVICE_INDEX,
            expected_architecture: Some(REQUIRED_ARCHITECTURE.to_string()),
            chunk_bytes: LOAD_CHUNK_BYTES,
            context_length: QWEN35_AQ4_CONTEXT_LENGTH,
            kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
            layer_indices: None,
            lm_head_mode: PackageLmHeadMode::GpuResidentF32,
            lm_head_chunk_rows: LM_HEAD_CHUNK_ROWS,
        },
        session_config,
    )?;

    require_isolated_r9700("after model load")?;
    if session.model().backend() != "hip" {
        return Err(format!(
            "loaded model backend must be hip, got {}",
            session.model().backend()
        ));
    }
    if token_count > session.model().geometry().context_length {
        return Err(format!(
            "diagnostic token count {token_count} exceeds loaded context length {}",
            session.model().geometry().context_length
        ));
    }

    let token_ids = deterministic_token_ids(token_count, session.model().geometry().vocab)?;
    let request = InferenceRequest::new_with_eos(
        "aq4-session-prefill-chunk-diagnostic",
        token_ids,
        1,
        EOS_TOKEN_IDS.to_vec(),
        SamplingParams::greedy_with_top_k(0, 1),
    );
    session.start_request(request, CancellationToken::new())?;

    drive_session_prefill(&mut session, token_count)?;
    session.shutdown()
}

fn token_count_from_caller() -> Result<usize, String> {
    let mut arguments = env::args().skip(1);
    let argument = arguments.next();
    if arguments.next().is_some() {
        return Err(format!(
            "usage: ullm-aq4-session-prefill-chunk-diagnostic [TOKEN_COUNT] (or set {TOKEN_COUNT_ENV})"
        ));
    }
    if argument.as_deref() == Some("--help") || argument.as_deref() == Some("-h") {
        return Err(format!(
            "usage: ullm-aq4-session-prefill-chunk-diagnostic [TOKEN_COUNT] (or set {TOKEN_COUNT_ENV})"
        ));
    }
    let value = match argument {
        Some(value) => value,
        None => env::var(TOKEN_COUNT_ENV).map_err(|_| {
            format!("provide TOKEN_COUNT as the sole positional argument or set {TOKEN_COUNT_ENV}")
        })?,
    };
    let token_count = value
        .parse::<usize>()
        .map_err(|error| format!("invalid token count {value:?}: {error}"))?;
    if token_count == 0 {
        return Err("token count must be positive".to_string());
    }
    Ok(token_count)
}

fn require_caller_environment() -> Result<(), String> {
    require_environment_value("HIP_VISIBLE_DEVICES")?;
    for name in QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV {
        require_environment_value(name)?;
    }
    Ok(())
}

fn require_environment_value(name: &str) -> Result<(), String> {
    if env::var(name).ok().as_deref() != Some("1") {
        return Err(format!(
            "{name} must be set to exactly 1 by the caller; this binary does not set GPU or kernel environment variables"
        ));
    }
    Ok(())
}

fn require_isolated_r9700(stage: &str) -> Result<(), String> {
    let count = ullm_runtime_sys::device_count()
        .map_err(|error| format!("failed to query runtime device count {stage}: {error}"))?;
    if count != RUNTIME_HIP_DEVICE_INDEX + 1 {
        return Err(format!(
            "{stage}: expected exactly CPU device 0 and isolated HIP device 1, found {count} runtime devices"
        ));
    }
    let info = ullm_runtime_sys::device_info(RUNTIME_HIP_DEVICE_INDEX)
        .map_err(|error| format!("failed to query runtime device 1 {stage}: {error}"))?;
    if info.backend != "hip" || info.gcn_arch_name != REQUIRED_ARCHITECTURE {
        return Err(format!(
            "{stage}: runtime device 1 must be HIP {REQUIRED_ARCHITECTURE}, got backend={} architecture={}",
            info.backend, info.gcn_arch_name
        ));
    }
    Ok(())
}

fn deterministic_token_ids(token_count: usize, vocab: usize) -> Result<Vec<usize>, String> {
    if vocab == 0 {
        return Err("loaded model has an empty vocabulary".to_string());
    }
    Ok((0..token_count)
        .map(|index| (17 + index * 7919) % vocab)
        .collect())
}

fn drive_session_prefill(
    session: &mut Qwen35Aq4InferenceSession,
    token_count: usize,
) -> Result<(), String> {
    let mut chunk_index = 0usize;
    let mut prompt_tokens_processed = 0usize;

    loop {
        let expected_width = PREFILL_CHUNK_WIDTH.min(
            token_count
                .checked_sub(prompt_tokens_processed)
                .ok_or_else(|| "prefill progress exceeded requested token count".to_string())?,
        );
        let started = Instant::now();
        let advance = match catch_unwind(AssertUnwindSafe(|| session.prepare_advance())) {
            Ok(Ok(advance)) => advance,
            Ok(Err(error)) => {
                emit_chunk_failure(
                    chunk_index,
                    expected_width,
                    started.elapsed(),
                    "error",
                    &error,
                );
                return Err(format!(
                    "prefill chunk {chunk_index} (execution_width={expected_width}) returned an error: {error}"
                ));
            }
            Err(payload) => {
                let error = panic_message(payload);
                emit_chunk_failure(
                    chunk_index,
                    expected_width,
                    started.elapsed(),
                    "panic",
                    &error,
                );
                return Err(format!(
                    "prefill chunk {chunk_index} (execution_width={expected_width}) panicked: {error}"
                ));
            }
        };

        match advance {
            SessionAdvance::PromptProgress {
                prompt_tokens_processed: reported_processed,
                cache_len,
                execution_width,
            } => {
                let elapsed = started.elapsed();
                if execution_width != expected_width
                    || cache_len != reported_processed
                    || reported_processed
                        != prompt_tokens_processed
                            .checked_add(expected_width)
                            .ok_or_else(|| "prefill progress overflowed".to_string())?
                {
                    let error = format!(
                        "unexpected prompt progress: processed={reported_processed}, cache_len={cache_len}, execution_width={execution_width}, expected_width={expected_width}, previous_processed={prompt_tokens_processed}"
                    );
                    emit_chunk_failure(
                        chunk_index,
                        expected_width,
                        elapsed,
                        "contract_error",
                        &error,
                    );
                    return Err(format!("prefill chunk {chunk_index}: {error}"));
                }
                write_json_line(json!({
                    "chunk_index": chunk_index,
                    "execution_width": execution_width,
                    "elapsed_seconds": elapsed.as_secs_f64(),
                }))?;
                prompt_tokens_processed = reported_processed;
                chunk_index = chunk_index
                    .checked_add(1)
                    .ok_or_else(|| "prefill chunk index overflowed".to_string())?;
            }
            SessionAdvance::Token { prepared, .. } => {
                if prompt_tokens_processed != token_count {
                    let elapsed = started.elapsed();
                    let error = format!(
                        "session prepared a generation token before prefill completed: processed={prompt_tokens_processed}, total={token_count}"
                    );
                    emit_chunk_failure(
                        chunk_index,
                        expected_width,
                        elapsed,
                        "contract_error",
                        &error,
                    );
                    return Err(error);
                }
                match session.publish_prepared(prepared, |_| Ok(()))? {
                    PublishedAdvance::Token {
                        terminal_reason: Some(_),
                        ..
                    } => return session.finish_and_reset().map(|_| ()),
                    PublishedAdvance::Token { .. } => {
                        return Err("one-token diagnostic request did not terminate".to_string());
                    }
                    PublishedAdvance::CancellationObserved => {
                        return Err(
                            "diagnostic request unexpectedly observed cancellation".to_string()
                        );
                    }
                }
            }
            SessionAdvance::CancellationObserved => {
                return Err("diagnostic request unexpectedly observed cancellation".to_string());
            }
        }
    }
}

fn emit_chunk_failure(
    chunk_index: usize,
    execution_width: usize,
    elapsed: std::time::Duration,
    failure_kind: &str,
    error: &str,
) {
    let _ = write_json_line(json!({
        "chunk_index": chunk_index,
        "execution_width": execution_width,
        "elapsed_seconds": elapsed.as_secs_f64(),
        "status": failure_kind,
        "error": error,
    }));
}

fn write_json_line(value: serde_json::Value) -> Result<(), String> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    serde_json::to_writer(&mut stdout, &value)
        .map_err(|error| format!("failed to serialize diagnostic JSON line: {error}"))?;
    stdout
        .write_all(b"\n")
        .and_then(|_| stdout.flush())
        .map_err(|error| format!("failed to write diagnostic JSON line: {error}"))
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
