//! Exact common-prefix terminal-logit comparison for the reviewed Ministral 3
//! GGUF and a fixed llama.cpp F32 oracle dump.
//!
//! The runner teacher-forces `Hello of the` as three transitions. It retains
//! the production FP16 KV recipe and requests the additive BF16 logit readback
//! only for evidence; the normal greedy execution path remains unchanged.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, MINISTRAL3_GRAPH_VOCAB_SIZE,
    MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256, MINISTRAL3_OFFICIAL_GGUF_REPOSITORY,
    MINISTRAL3_OFFICIAL_GGUF_REVISION, MINISTRAL3_WEIGHT_LOCK_FINGERPRINT, Ministral3DispatchAudit,
    Ministral3ResidentModel, VerifiedMinistral3WeightSource, build_ministral3_weight_load_plan,
    open_and_verify_official_ministral3_gguf,
};
use sllm_hip::HipBackend;

const COMPLETION_TIMEOUT: Duration = Duration::from_secs(300);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const PREFIXES: [&[i32]; 3] = [&[22_177], &[22_177, 1_307], &[22_177, 1_307, 1_278]];
const TRANSITION_TOKENS: [i32; 3] = [22_177, 1_307, 1_278];
const BENCHMARK_PREFILL_TOKENS: usize = 513;
const BENCHMARK_DECODE_TOKENS: usize = 8;
const BENCHMARK_WARMUPS: usize = 2;
const BENCHMARK_MEASURED: usize = 5;

#[derive(Debug, Serialize)]
struct TopValue {
    token_id: usize,
    logit: f32,
}

#[derive(Debug, Serialize)]
struct CandidateComparison {
    token_id: usize,
    reference_logit: f32,
    sllm_logit: f32,
    signed_error: f32,
}

#[derive(Debug, Serialize)]
struct RowReport {
    lane: &'static str,
    step: usize,
    prefix_token_ids: &'static [i32],
    reference_sha256: String,
    sllm_bf16_sha256: String,
    sllm_bf16_path: String,
    reference_top1: usize,
    sllm_top1: usize,
    top1_match: bool,
    reference_top10: Vec<TopValue>,
    sllm_top10: Vec<TopValue>,
    kld_reference_to_sllm: f64,
    mean_abs_logit_error: f64,
    root_mean_square_logit_error: f64,
    max_abs_logit_error: f32,
    token_3950: CandidateComparison,
    token_4304: CandidateComparison,
    reference_4304_minus_3950: f32,
    sllm_4304_minus_3950: f32,
    elapsed_ms: f64,
    submissions: u64,
    kernel_dispatches: u64,
}

#[derive(Debug, Serialize)]
struct BenchmarkSample {
    prefill_ms: f64,
    prefill_tokens_per_second: f64,
    decode_ms: Vec<f64>,
    decode_tokens_per_second: f64,
    generated_token_ids: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    timing_scope: &'static str,
    prefill_tokens: usize,
    decode_tokens: usize,
    warmups: usize,
    measured: usize,
    samples: Vec<BenchmarkSample>,
    median_prefill_ms: f64,
    median_prefill_tokens_per_second: f64,
    median_decode_ms_per_token: f64,
    median_decode_tokens_per_second: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    model_repository: &'static str,
    model_revision: &'static str,
    model_lfs_sha256: &'static str,
    weight_lock_fingerprint: &'static str,
    target: String,
    device_index: u32,
    kv_encoding: &'static str,
    graph_activation_dtype: &'static str,
    accumulation_dtype: &'static str,
    oracle_dtype: &'static str,
    resident_load_ms: f64,
    resident_bytes: u64,
    workspace_peak_bytes: u64,
    request_state_peak_bytes: u64,
    rows: Vec<RowReport>,
    benchmark: BenchmarkReport,
    retryable_cleanup: usize,
    durable_quarantine: usize,
    final_cleanup_empty: bool,
}

fn sha256(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn read_f32_row(path: &Path) -> Result<(Vec<f32>, String), String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let expected = MINISTRAL3_GRAPH_VOCAB_SIZE
        .checked_mul(4)
        .ok_or("reference byte count overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            path.display(),
            bytes.len()
        ));
    }
    let digest = sha256(&bytes);
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{} contains non-finite logits", path.display()));
    }
    Ok((values, digest))
}

fn bf16_row(bits: &[u16]) -> Result<(Vec<f32>, Vec<u8>, String), String> {
    if bits.len() != MINISTRAL3_GRAPH_VOCAB_SIZE {
        return Err(format!(
            "sLLM returned {} logits, expected {}",
            bits.len(),
            MINISTRAL3_GRAPH_VOCAB_SIZE
        ));
    }
    let bytes = bits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let values = bits
        .iter()
        .map(|&value| f32::from_bits(u32::from(value) << 16))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("sLLM BF16 logits contain non-finite values".to_owned());
    }
    let digest = sha256(&bytes);
    Ok((values, bytes, digest))
}

fn top(values: &[f32], count: usize) -> Vec<TopValue> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    });
    indices
        .into_iter()
        .take(count)
        .map(|token_id| TopValue {
            token_id,
            logit: values[token_id],
        })
        .collect()
}

fn kld(reference: &[f32], candidate: &[f32]) -> f64 {
    let reference_max = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let candidate_max = candidate.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let reference_sum = reference
        .iter()
        .map(|&value| f64::from(value - reference_max).exp())
        .sum::<f64>();
    let candidate_sum = candidate
        .iter()
        .map(|&value| f64::from(value - candidate_max).exp())
        .sum::<f64>();
    let reference_log_z = f64::from(reference_max) + reference_sum.ln();
    let candidate_log_z = f64::from(candidate_max) + candidate_sum.ln();
    reference
        .iter()
        .zip(candidate)
        .map(|(&left, &right)| {
            let left_log_probability = f64::from(left) - reference_log_z;
            let right_log_probability = f64::from(right) - candidate_log_z;
            left_log_probability.exp() * (left_log_probability - right_log_probability)
        })
        .sum()
}

fn validate_audit(audit: &Ministral3DispatchAudit, target: &str) -> Result<(), String> {
    if audit.backend() != 1
        || audit.target() != target
        || audit.fallback_used()
        || audit.submission_count() == 0
        || audit.kernel_dispatch_count() == 0
        || audit.dispatches().iter().any(|dispatch| {
            dispatch.backend != 1 || dispatch.target != target || dispatch.fallback_used
        })
    {
        return Err(format!(
            "transition is not exact HIP/no-fallback: {audit:?}"
        ));
    }
    Ok(())
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn benchmark_resident(
    resident: &Ministral3ResidentModel,
    target: &str,
) -> Result<BenchmarkReport, String> {
    let input = (0..BENCHMARK_PREFILL_TOKENS)
        .map(|index| TRANSITION_TOKENS[index % TRANSITION_TOKENS.len()])
        .collect::<Vec<_>>();
    let run_sample = || -> Result<BenchmarkSample, String> {
        let mut request = resident
            .new_request(
                BENCHMARK_PREFILL_TOKENS as u64,
                (BENCHMARK_PREFILL_TOKENS + BENCHMARK_DECODE_TOKENS) as u64,
            )
            .map_err(|error| format!("create benchmark request: {error}"))?;
        let prefill_started = Instant::now();
        let output = request
            .prefill(&input)
            .map_err(|error| format!("benchmark prefill: {error}"))?;
        let prefill_elapsed = prefill_started.elapsed();
        validate_audit(output.audit(), target)?;
        let mut token = output
            .token_ids()
            .last()
            .copied()
            .ok_or("benchmark prefill returned no token")?;
        let mut generated_token_ids = vec![token];
        let mut decode_ms = Vec::with_capacity(BENCHMARK_DECODE_TOKENS);
        for step in 0..BENCHMARK_DECODE_TOKENS {
            let decode_started = Instant::now();
            let output = request
                .decode(token)
                .map_err(|error| format!("benchmark decode {step}: {error}"))?;
            let decode_elapsed = decode_started.elapsed();
            validate_audit(output.audit(), target)?;
            token = output
                .token_ids()
                .last()
                .copied()
                .ok_or_else(|| format!("benchmark decode {step} returned no token"))?;
            generated_token_ids.push(token);
            decode_ms.push(decode_elapsed.as_secs_f64() * 1_000.0);
        }
        let prefill_seconds = prefill_elapsed.as_secs_f64();
        let decode_seconds = decode_ms.iter().sum::<f64>() / 1_000.0;
        Ok(BenchmarkSample {
            prefill_ms: prefill_seconds * 1_000.0,
            prefill_tokens_per_second: BENCHMARK_PREFILL_TOKENS as f64 / prefill_seconds,
            decode_ms,
            decode_tokens_per_second: BENCHMARK_DECODE_TOKENS as f64 / decode_seconds,
            generated_token_ids,
        })
    };
    for _ in 0..BENCHMARK_WARMUPS {
        let _ = run_sample()?;
    }
    let mut samples = Vec::with_capacity(BENCHMARK_MEASURED);
    for _ in 0..BENCHMARK_MEASURED {
        samples.push(run_sample()?);
    }
    let median_prefill_ms = median(samples.iter().map(|sample| sample.prefill_ms).collect());
    let median_prefill_tokens_per_second = median(
        samples
            .iter()
            .map(|sample| sample.prefill_tokens_per_second)
            .collect(),
    );
    let median_decode_ms_per_token = median(
        samples
            .iter()
            .map(|sample| sample.decode_ms.iter().sum::<f64>() / BENCHMARK_DECODE_TOKENS as f64)
            .collect(),
    );
    let median_decode_tokens_per_second = median(
        samples
            .iter()
            .map(|sample| sample.decode_tokens_per_second)
            .collect(),
    );
    Ok(BenchmarkReport {
        timing_scope: "resident production prefill/decode; request allocation excluded; no logit readback",
        prefill_tokens: BENCHMARK_PREFILL_TOKENS,
        decode_tokens: BENCHMARK_DECODE_TOKENS,
        warmups: BENCHMARK_WARMUPS,
        measured: BENCHMARK_MEASURED,
        samples,
        median_prefill_ms,
        median_prefill_tokens_per_second,
        median_decode_ms_per_token,
        median_decode_tokens_per_second,
    })
}

fn row_report(
    lane: &'static str,
    step: usize,
    reference_path: &Path,
    sllm_bits: &[u16],
    output_path: &Path,
    elapsed: Duration,
    audit: &Ministral3DispatchAudit,
) -> Result<RowReport, String> {
    let (reference, reference_sha256) = read_f32_row(reference_path)?;
    let (sllm, sllm_bytes, sllm_bf16_sha256) = bf16_row(sllm_bits)?;
    let raw_path = output_path.with_extension(format!("{lane}-step{step}.bf16"));
    if raw_path.exists() {
        return Err(format!("{} already exists", raw_path.display()));
    }
    fs::write(&raw_path, &sllm_bytes)
        .map_err(|error| format!("write {}: {error}", raw_path.display()))?;
    let mut absolute_sum = 0.0_f64;
    let mut square_sum = 0.0_f64;
    let mut maximum = 0.0_f32;
    for (&left, &right) in reference.iter().zip(&sllm) {
        let error = (left - right).abs();
        absolute_sum += f64::from(error);
        square_sum += f64::from(error) * f64::from(error);
        maximum = maximum.max(error);
    }
    let count = reference.len() as f64;
    let reference_top10 = top(&reference, 10);
    let sllm_top10 = top(&sllm, 10);
    let reference_top1 = reference_top10[0].token_id;
    let sllm_top1 = sllm_top10[0].token_id;
    let candidate = |token_id: usize| CandidateComparison {
        token_id,
        reference_logit: reference[token_id],
        sllm_logit: sllm[token_id],
        signed_error: sllm[token_id] - reference[token_id],
    };
    Ok(RowReport {
        lane,
        step,
        prefix_token_ids: PREFIXES[step],
        reference_sha256,
        sllm_bf16_sha256,
        sllm_bf16_path: raw_path.display().to_string(),
        reference_top1,
        sllm_top1,
        top1_match: reference_top1 == sllm_top1,
        reference_top10,
        sllm_top10,
        kld_reference_to_sllm: kld(&reference, &sllm),
        mean_abs_logit_error: absolute_sum / count,
        root_mean_square_logit_error: (square_sum / count).sqrt(),
        max_abs_logit_error: maximum,
        token_3950: candidate(3_950),
        token_4304: candidate(4_304),
        reference_4304_minus_3950: reference[4_304] - reference[3_950],
        sllm_4304_minus_3950: sllm[4_304] - sllm[3_950],
        elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
        submissions: audit.submission_count(),
        kernel_dispatches: audit.kernel_dispatch_count(),
    })
}

fn publish(report: &Report, output: &Path) -> Result<String, String> {
    if output.exists() {
        return Err(format!("{} already exists", output.display()));
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| format!("create {}: {error}", parent.display()))?;
    let bytes = serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?;
    let digest = sha256(&bytes);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|error| format!("create {}: {error}", output.display()))?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("write {}: {error}", output.display()))?;
    Ok(digest)
}

fn run(arguments: &[String]) -> Result<(Report, PathBuf), String> {
    if arguments.len() != 7 {
        return Err(
            "usage: GGUF DEVICE_INDEX TARGET LLAMA_STEP0_F32 LLAMA_STEP1_F32 LLAMA_STEP2_F32 OUTPUT_JSON"
                .to_owned(),
        );
    }
    let gguf_path = PathBuf::from(&arguments[0]);
    let device_index = arguments[1]
        .parse::<u32>()
        .map_err(|_| "device index must be u32".to_owned())?;
    let target = arguments[2].clone();
    if !matches!(target.as_str(), "gfx1030" | "gfx1201") {
        return Err("target must be exactly gfx1030 or gfx1201".to_owned());
    }
    let reference_paths = [
        PathBuf::from(&arguments[3]),
        PathBuf::from(&arguments[4]),
        PathBuf::from(&arguments[5]),
    ];
    let output_path = PathBuf::from(&arguments[6]);
    if output_path.exists()
        || ["incremental", "full-prefill"].into_iter().any(|lane| {
            (0..3).any(|step| {
                output_path
                    .with_extension(format!("{lane}-step{step}.bf16"))
                    .exists()
            })
        })
    {
        return Err("output report or BF16 companion already exists".to_owned());
    }
    for path in &reference_paths {
        let _ = read_f32_row(path)?;
    }

    let verified = open_and_verify_official_ministral3_gguf(&gguf_path)
        .map_err(|error| format!("verify official GGUF: {error}"))?;
    let source = Arc::new(
        VerifiedMinistral3WeightSource::from_verified_gguf(verified)
            .map_err(|error| format!("bind weight source: {error}"))?,
    );
    if source.repository() != MINISTRAL3_OFFICIAL_GGUF_REPOSITORY
        || source.revision() != MINISTRAL3_OFFICIAL_GGUF_REVISION
        || source.file_sha256() != MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256
        || source.lock_fingerprint() != MINISTRAL3_WEIGHT_LOCK_FINGERPRINT
    {
        return Err("reviewed Ministral 3 identity differs".to_owned());
    }
    let plan = build_ministral3_weight_load_plan(source.as_ref())
        .map_err(|error| format!("build weight plan: {error}"))?;
    let backend = HipBackend::connect().map_err(|error| format!("connect HIP: {error}"))?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| format!("session request: {error}"))?,
        )
        .map_err(|error| format!("open HIP session: {error}"))?;

    let execution = (|| -> Result<Report, String> {
        let load_started = Instant::now();
        let resident = Ministral3ResidentModel::new_gguf(
            Arc::clone(&session),
            plan,
            Arc::clone(&source),
            COMPLETION_TIMEOUT,
        )
        .map_err(|error| format!("provision resident model: {error}"))?;
        let resident_load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
        let resident_bytes = resident.resident_bytes();
        let mut request = resident
            .new_request(1, 3)
            .map_err(|error| format!("create request: {error}"))?;
        let mut rows = Vec::with_capacity(3);

        let started = Instant::now();
        let output = request
            .prefill_with_last_logits(&[TRANSITION_TOKENS[0]])
            .map_err(|error| format!("step 0 prefill: {error}"))?;
        validate_audit(output.audit(), &target)?;
        rows.push(row_report(
            "incremental",
            0,
            &reference_paths[0],
            output
                .last_logits_bf16()
                .ok_or("step 0 omitted requested logits")?,
            &output_path,
            started.elapsed(),
            output.audit(),
        )?);

        for step in 1..3 {
            let started = Instant::now();
            let output = request
                .decode_with_last_logits(TRANSITION_TOKENS[step])
                .map_err(|error| format!("step {step} decode: {error}"))?;
            validate_audit(output.audit(), &target)?;
            rows.push(row_report(
                "incremental",
                step,
                &reference_paths[step],
                output
                    .last_logits_bf16()
                    .ok_or_else(|| format!("step {step} omitted requested logits"))?,
                &output_path,
                started.elapsed(),
                output.audit(),
            )?);
        }
        drop(request);

        for step in 0..3 {
            let prefix = PREFIXES[step];
            let mut request = resident
                .new_request(prefix.len() as u64, prefix.len() as u64)
                .map_err(|error| format!("create full-prefill request {step}: {error}"))?;
            let started = Instant::now();
            let output = request
                .prefill_with_last_logits(prefix)
                .map_err(|error| format!("full-prefill step {step}: {error}"))?;
            validate_audit(output.audit(), &target)?;
            rows.push(row_report(
                "full-prefill",
                step,
                &reference_paths[step],
                output
                    .last_logits_bf16()
                    .ok_or_else(|| format!("full-prefill step {step} omitted logits"))?,
                &output_path,
                started.elapsed(),
                output.audit(),
            )?);
            drop(request);
        }
        let benchmark = benchmark_resident(&resident, &target)?;
        let active = session.memory_snapshot();
        let workspace_peak_bytes = active.workspace().high_water_bytes();
        let request_state_peak_bytes = active.request_state().high_water_bytes();
        drop(resident);
        let released = session.memory_snapshot();
        if released.poisoned() || released.current_bytes() != 0 {
            return Err(format!(
                "resident/request cleanup is incomplete: {released:?}"
            ));
        }
        Ok(Report {
            schema_version: "sllm-ministral3-logits-evidence-v3",
            state: "PASS",
            model_repository: MINISTRAL3_OFFICIAL_GGUF_REPOSITORY,
            model_revision: MINISTRAL3_OFFICIAL_GGUF_REVISION,
            model_lfs_sha256: MINISTRAL3_OFFICIAL_GGUF_LFS_SHA256,
            weight_lock_fingerprint: MINISTRAL3_WEIGHT_LOCK_FINGERPRINT,
            target,
            device_index,
            kv_encoding: "fp16",
            graph_activation_dtype: "bf16",
            accumulation_dtype: "fp32",
            oracle_dtype: "f32",
            resident_load_ms,
            resident_bytes,
            workspace_peak_bytes,
            request_state_peak_bytes,
            rows,
            benchmark,
            retryable_cleanup: 0,
            durable_quarantine: 0,
            final_cleanup_empty: false,
        })
    })();
    let mut report = match execution {
        Ok(report) => report,
        Err(error) => {
            let _ = session.shutdown(SHUTDOWN_TIMEOUT);
            return Err(error);
        }
    };
    let cleanup = session
        .shutdown(SHUTDOWN_TIMEOUT)
        .map_err(|error| format!("shutdown session: {error}"))?;
    let final_cleanup_empty = cleanup.retryable_cleanup == 0
        && cleanup.durable_quarantine == 0
        && session.memory_snapshot().current_bytes() == 0;
    if !final_cleanup_empty {
        return Err(format!("final cleanup is nonzero: {cleanup:?}"));
    }
    report.retryable_cleanup = cleanup.retryable_cleanup;
    report.durable_quarantine = cleanup.durable_quarantine;
    report.final_cleanup_empty = final_cleanup_empty;
    Ok((report, output_path))
}

fn main() -> ExitCode {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok((report, output)) => match publish(&report, &output) {
            Ok(digest) => {
                println!("{} {digest}", output.display());
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("Ministral 3 logits publication failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("Ministral 3 logits evidence failed: {error}");
            ExitCode::FAILURE
        }
    }
}
