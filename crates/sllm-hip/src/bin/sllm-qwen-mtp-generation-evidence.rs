//! Paired target-only versus exact-MTP generation-service evidence.

use serde::Serialize;
use sllm_core::{
    Backend, ExecutionSessionRequest, QwenComponentSelection, QwenResidentModel, SamplingError,
    SamplingRandomSource, build_qwen35_graph, build_qwen35_mtp_graph,
    build_verified_qwen_component_weight_load_plan, build_verified_weight_load_plan,
    read_model_lock,
};
use sllm_frontend::{
    GenerationCancellationV1, GenerationConfigV1, GenerationInputV1, GenerationServiceV1,
    Qwen35ChatMessageV1, Qwen35ChatTemplateV1, Qwen35RenderOptionsV1, QwenMtpGenerationExecutorV1,
    SpeculativeGenerationAdapterV1, ThinkingModeV1, TokenizerFrontendV1,
};
use sllm_hip::HipBackend;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const GUARD: &str = "SLLM_QWEN_MTP_GENERATION_GPU_EXECUTION";
const TIMEOUT: Duration = Duration::from_secs(180);

struct GreedyRandom;

impl SamplingRandomSource for GreedyRandom {
    fn next_unit_f64(&mut self) -> Result<f64, SamplingError> {
        Err(SamplingError::RandomSourceUnavailable)
    }
}

#[derive(Serialize)]
struct Sample {
    order: &'static str,
    target_only_ms: f64,
    mtp_ms: f64,
    speedup: f64,
    target_only_dispatches: u64,
    mtp_target_dispatches: u64,
    mtp_draft_dispatches: u64,
    proposal_blocks: u64,
    proposed_draft_tokens: u64,
    accepted_draft_tokens: u64,
    committed_target_rows: u64,
}

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    output_tokens: u32,
    warmups: u32,
    measured: u32,
    draft_width: usize,
    output_exact: bool,
    median_speedup: f64,
    mad_speedup: f64,
    p10_speedup: f64,
    p90_speedup: f64,
    off_first_median_speedup: Option<f64>,
    mtp_first_median_speedup: Option<f64>,
    accepted_tokens_per_proposal: f64,
    target_rows_per_output_token: f64,
    samples: Vec<Sample>,
    fallback_used: bool,
    cleanup_empty: bool,
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let position = percentile * (values.len().saturating_sub(1) as f64);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let weight = position - lower as f64;
        values[lower] * (1.0 - weight) + values[upper] * weight
    }
}

fn optional_median(values: Vec<f64>) -> Option<f64> {
    (!values.is_empty()).then(|| median(values))
}

fn main() -> Result<(), String> {
    if env::var(GUARD).as_deref() != Ok("1") {
        return Err(format!("{GUARD}=1 is required"));
    }
    let mut args = env::args().skip(1);
    let device_index = args
        .next()
        .ok_or("device index is required")?
        .parse::<u32>()
        .map_err(|_| "device index must be U32")?;
    let target = args.next().ok_or("target is required")?;
    let lock_path = PathBuf::from(args.next().ok_or("lock path is required")?);
    let cache_path = PathBuf::from(args.next().ok_or("cache path is required")?);
    let output_tokens = args
        .next()
        .unwrap_or_else(|| "32".to_owned())
        .parse::<u32>()
        .map_err(|_| "output tokens must be U32")?;
    let warmups = args
        .next()
        .unwrap_or_else(|| "1".to_owned())
        .parse::<u32>()
        .map_err(|_| "warmups must be U32")?;
    let measured = args
        .next()
        .unwrap_or_else(|| "3".to_owned())
        .parse::<u32>()
        .map_err(|_| "measured must be U32")?;
    let draft_width = args
        .next()
        .unwrap_or_else(|| "2".to_owned())
        .parse::<usize>()
        .map_err(|_| "draft width must be usize")?;
    if args.next().is_some()
        || measured == 0
        || output_tokens < 4
        || !(1..=2).contains(&draft_width)
    {
        return Err(
            "usage: DEVICE TARGET LOCK CACHE [OUTPUT>=4] [WARMUPS] [MEASURED>0] [DRAFT=1..2]"
                .to_owned(),
        );
    }

    let lock = read_model_lock(lock_path).map_err(|error| error.to_string())?;
    let cache = Arc::new(
        lock.verify_cache(cache_path)
            .map_err(|error| error.to_string())?,
    );
    let text_plan =
        build_verified_weight_load_plan(&lock, &cache).map_err(|error| error.to_string())?;
    let mtp_plan = build_verified_qwen_component_weight_load_plan(
        &lock,
        &cache,
        QwenComponentSelection::MTP_ONLY,
    )
    .map_err(|error| error.to_string())?;
    let tokenizer =
        TokenizerFrontendV1::from_verified_cache(&lock, &cache).map_err(|e| e.to_string())?;
    let renderer =
        Qwen35ChatTemplateV1::from_verified_cache(&lock, &cache).map_err(|e| e.to_string())?;
    let stop_policy = lock.generation_stop_policy().clone();
    let service = GenerationServiceV1::new(&tokenizer, Some(&renderer), &stop_policy)
        .map_err(|e| e.to_string())?;
    let input = service
        .prepare_input(&GenerationInputV1::Messages {
            messages: vec![Qwen35ChatMessageV1::user(
                "Write a concise Rust function that returns the sum of two i32 values.",
            )],
            options: Qwen35RenderOptionsV1 {
                add_generation_prompt: true,
                thinking: ThinkingModeV1::Disabled,
            },
        })
        .map_err(|error| error.to_string())?;
    let input_len = u64::try_from(input.len()).map_err(|_| "prompt length overflow")?;
    let capacity = u64::from(output_tokens) + input_len + 8;
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let graph_rows = input_len.max((draft_width + 1) as u64);
    let resident_graph = build_qwen35_graph(&lock, &text_plan, graph_rows, capacity)
        .map_err(|error| error.to_string())?;
    let text_resident = QwenResidentModel::new(
        Arc::clone(&session),
        resident_graph,
        text_plan.clone(),
        Arc::clone(&cache),
        TIMEOUT,
    )
    .map_err(|error| error.to_string())?;
    let mtp_graph =
        build_qwen35_mtp_graph(&lock, &mtp_plan, capacity).map_err(|error| error.to_string())?;
    let mtp_resident = QwenResidentModel::new(
        Arc::clone(&session),
        mtp_graph,
        mtp_plan.clone(),
        Arc::clone(&cache),
        TIMEOUT,
    )
    .map_err(|error| error.to_string())?;
    let request_graph = build_qwen35_graph(&lock, &text_plan, graph_rows, capacity)
        .map_err(|error| error.to_string())?;
    let config = GenerationConfigV1::new(
        output_tokens,
        sllm_core::SamplingParametersV1::greedy(),
        Vec::new(),
    )
    .map_err(|error| error.to_string())?;
    let total = warmups + measured;
    let mut samples = Vec::new();
    let mut output_exact = true;
    let mut fallback_used = false;

    for index in 0..total {
        let mtp_first = index % 2 == 1;
        let run_target = || -> Result<_, String> {
            let mut request = text_resident
                .new_request(request_graph.clone())
                .map_err(|error| error.to_string())?;
            let started = Instant::now();
            let result = service
                .generate_tokens(
                    &mut request,
                    &input,
                    &config,
                    &GenerationCancellationV1::new(),
                    &mut GreedyRandom,
                )
                .map_err(|error| error.to_string())?;
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            let audit = request
                .audit_snapshot()
                .map_err(|error| error.to_string())?;
            Ok((result, elapsed, audit))
        };
        let run_mtp = || -> Result<_, String> {
            let target_request = text_resident
                .new_request(request_graph.clone())
                .map_err(|error| error.to_string())?;
            let mtp_request = mtp_resident
                .new_request(
                    build_qwen35_mtp_graph(&lock, &mtp_plan, capacity)
                        .map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;
            let mut adapter = SpeculativeGenerationAdapterV1::new(
                QwenMtpGenerationExecutorV1::new_with_draft_width(
                    target_request,
                    mtp_request,
                    draft_width,
                )
                .map_err(|error| error.to_string())?,
            );
            let started = Instant::now();
            let result = service
                .generate_tokens(
                    &mut adapter,
                    &input,
                    &config,
                    &GenerationCancellationV1::new(),
                    &mut GreedyRandom,
                )
                .map_err(|error| error.to_string())?;
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            let target_audit = adapter
                .inner()
                .target()
                .audit_snapshot()
                .map_err(|error| error.to_string())?;
            let mtp_audit = adapter
                .inner()
                .mtp()
                .audit_snapshot()
                .map_err(|error| error.to_string())?;
            Ok((
                result,
                elapsed,
                target_audit,
                mtp_audit,
                adapter.inner().proposal_blocks(),
                adapter.inner().proposed_draft_tokens(),
                adapter.inner().accepted_draft_tokens(),
                adapter.inner().committed_target_rows(),
            ))
        };
        let (off, on) = if mtp_first {
            let on = run_mtp()?;
            let off = run_target()?;
            (off, on)
        } else {
            let off = run_target()?;
            let on = run_mtp()?;
            (off, on)
        };
        output_exact &= off.0.generated_token_ids() == on.0.generated_token_ids()
            && off.0.finish_reason() == on.0.finish_reason()
            && off.0.usage() == on.0.usage();
        fallback_used |= off.2.fallback_used() || on.2.fallback_used() || on.3.fallback_used();
        if index >= warmups {
            samples.push(Sample {
                order: if mtp_first { "mtp/off" } else { "off/mtp" },
                target_only_ms: off.1,
                mtp_ms: on.1,
                speedup: off.1 / on.1,
                target_only_dispatches: off.2.kernel_dispatch_count(),
                mtp_target_dispatches: on.2.kernel_dispatch_count(),
                mtp_draft_dispatches: on.3.kernel_dispatch_count(),
                proposal_blocks: on.4,
                proposed_draft_tokens: on.5,
                accepted_draft_tokens: on.6,
                committed_target_rows: on.7,
            });
        }
    }
    if !output_exact || fallback_used {
        return Err(format!(
            "paired generation failed: output_exact={output_exact}, fallback={fallback_used}"
        ));
    }
    let speedups = samples
        .iter()
        .map(|sample| sample.speedup)
        .collect::<Vec<_>>();
    let median_speedup = median(speedups.clone());
    let mad_speedup = median(
        speedups
            .iter()
            .map(|speedup| (speedup - median_speedup).abs())
            .collect(),
    );
    let p10_speedup = percentile(speedups.clone(), 0.10);
    let p90_speedup = percentile(speedups, 0.90);
    let off_first_median_speedup = optional_median(
        samples
            .iter()
            .filter(|sample| sample.order == "off/mtp")
            .map(|sample| sample.speedup)
            .collect(),
    );
    let mtp_first_median_speedup = optional_median(
        samples
            .iter()
            .filter(|sample| sample.order == "mtp/off")
            .map(|sample| sample.speedup)
            .collect(),
    );
    let proposed = samples
        .iter()
        .map(|sample| sample.proposed_draft_tokens)
        .sum::<u64>();
    let accepted = samples
        .iter()
        .map(|sample| sample.accepted_draft_tokens)
        .sum::<u64>();
    let target_rows = samples
        .iter()
        .map(|sample| sample.committed_target_rows)
        .sum::<u64>();
    let measured_output = u64::from(output_tokens) * u64::from(measured);
    let accepted_tokens_per_proposal = if proposed == 0 {
        0.0
    } else {
        accepted as f64 / proposed as f64
    };
    let target_rows_per_output_token = target_rows as f64 / measured_output as f64;
    drop(mtp_resident);
    drop(text_resident);
    let cleanup = session
        .shutdown(Duration::from_secs(30))
        .map_err(|error| error.to_string())?;
    let cleanup_empty = cleanup.retryable_cleanup == 0 && cleanup.durable_quarantine == 0;
    if !cleanup_empty {
        return Err("generation evidence cleanup was not empty".to_owned());
    }
    println!(
        "{}",
        serde_json::to_string(&Report {
            schema_version: "qwen35-mtp-generation-v1",
            state: "PASS",
            target,
            output_tokens,
            warmups,
            measured,
            draft_width,
            output_exact,
            median_speedup,
            mad_speedup,
            p10_speedup,
            p90_speedup,
            off_first_median_speedup,
            mtp_first_median_speedup,
            accepted_tokens_per_proposal,
            target_rows_per_output_token,
            samples,
            fallback_used,
            cleanup_empty,
        })
        .map_err(|error| error.to_string())?
    );
    Ok(())
}
