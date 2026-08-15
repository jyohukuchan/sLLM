//! Full real-weight Qwen3.5 vision encoder/projector GPU evidence.

use serde::Serialize;
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, ExecutionSessionRequest, QwenMultimodalImageEmbedding, QwenResidentModel,
    QwenVisionExecutionInput, QwenVisionResidentModel, assemble_qwen35_multimodal_prompt,
    build_qwen35_graph, build_qwen35_multimodal_graph, build_verified_qwen35_vision_manifest,
    build_verified_weight_load_plan, read_model_lock,
};
use sllm_hip::HipBackend;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(300);
const GUARD: &str = "SLLM_QWEN_VISION_GPU_EXECUTION";

#[derive(Serialize)]
struct Report {
    schema_version: &'static str,
    state: &'static str,
    target: String,
    device_index: u32,
    lock_fingerprint: String,
    manifest_digest: String,
    grid_thw: [u32; 3],
    patch_sha256: String,
    projected_sha256: String,
    projected_prefix_bf16: Vec<u16>,
    patch_tokens: usize,
    visual_tokens: usize,
    dispatches: u64,
    fallback_used: bool,
    all_dispatches_hip: bool,
    deterministic_replay: bool,
    text_prefill_token: i32,
    text_decode_token: i32,
    text_dispatches: u64,
    text_fallback_used: bool,
    elapsed_ms: u128,
    cleanup_empty: bool,
}

fn hash_f32(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("sha256:{:x}", hash.finalize())
}

fn hash_bf16(values: &[u16]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("sha256:{:x}", hash.finalize())
}

fn run() -> Result<Report, String> {
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
    if args.next().is_some() || !matches!(target.as_str(), "gfx1030" | "gfx1201" | "gfx942") {
        return Err("usage: DEVICE TARGET LOCK CACHE".to_owned());
    }
    let started = Instant::now();
    let lock = read_model_lock(lock_path).map_err(|error| error.to_string())?;
    let cache = Arc::new(
        lock.verify_cache(cache_path)
            .map_err(|error| error.to_string())?,
    );
    let manifest =
        build_verified_qwen35_vision_manifest(&lock, &cache).map_err(|error| error.to_string())?;
    let manifest_digest = manifest.digest_hex();
    let backend = HipBackend::connect().map_err(|error| error.to_string())?;
    let session = backend
        .open_execution_session(
            ExecutionSessionRequest::new(device_index, target.clone())
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let resident =
        QwenVisionResidentModel::new(Arc::clone(&session), Arc::clone(&cache), manifest, TIMEOUT)
            .map_err(|error| error.to_string())?;

    // Minimum locked image area gives a 16x16 patch grid. Values model a
    // deterministic normalized RGB raster without retaining an image artifact.
    let patches = (0..256 * 1_536)
        .map(|index| ((index % 509) as f32 / 254.0) - 1.0)
        .collect::<Vec<_>>();
    let input = QwenVisionExecutionInput {
        grid_thw: [1, 16, 16],
        patch_width: 1_536,
        patches,
    };
    let first = resident
        .execute(&input)
        .map_err(|error| error.to_string())?;
    let second = resident
        .execute(&input)
        .map_err(|error| error.to_string())?;
    let deterministic_replay = first.embeddings_bf16() == second.embeddings_bf16();
    if !deterministic_replay {
        return Err("vision deterministic replay changed projected embeddings".to_owned());
    }
    let text_plan =
        build_verified_weight_load_plan(&lock, &cache).map_err(|error| error.to_string())?;
    let prompt = std::iter::once(248_053_u32)
        .chain(std::iter::repeat_n(248_056_u32, first.visual_tokens()))
        .chain([248_054, 248_046])
        .collect::<Vec<_>>();
    let assembled = assemble_qwen35_multimodal_prompt(
        cache.as_ref(),
        &prompt,
        248_056,
        &[QwenMultimodalImageEmbedding {
            grid_thw: input.grid_thw,
            embeddings_bf16: first.embeddings_bf16().to_vec(),
        }],
    )
    .map_err(|error| error.to_string())?;
    let capacity = prompt.len() as u64 + 2;
    let seed =
        build_qwen35_graph(&lock, &text_plan, 1, capacity).map_err(|error| error.to_string())?;
    let text_resident = QwenResidentModel::new(
        Arc::clone(&session),
        seed,
        text_plan.clone(),
        Arc::clone(&cache),
        TIMEOUT,
    )
    .map_err(|error| error.to_string())?;
    let graph = build_qwen35_multimodal_graph(&lock, &text_plan, prompt.len() as u64, capacity)
        .map_err(|error| error.to_string())?;
    let mut text_request = text_resident
        .new_request(graph)
        .map_err(|error| error.to_string())?;
    let prompt_i32 = prompt.iter().map(|token| *token as i32).collect::<Vec<_>>();
    let prefill = text_request
        .prefill_multimodal_with_last_logits(
            &prompt_i32,
            &assembled.embeddings_bf16,
            &assembled.positions,
        )
        .map_err(|error| error.to_string())?;
    let text_prefill_token = *prefill
        .token_ids()
        .last()
        .ok_or("text prefill token absent")?;
    let decode = text_request
        .decode(text_prefill_token)
        .map_err(|error| error.to_string())?;
    let text_decode_token = decode.token_ids()[0];
    let text_audit = text_request
        .audit_snapshot()
        .map_err(|error| error.to_string())?;
    if text_audit.fallback_used() || !text_audit.all_dispatches_hip() {
        return Err("multimodal text execution used fallback or non-HIP dispatch".to_owned());
    }
    let report = Report {
        schema_version: "qwen35-vision-gpu-evidence-v1",
        state: "PASS",
        target,
        device_index,
        lock_fingerprint: lock.fingerprint().to_owned(),
        manifest_digest,
        grid_thw: input.grid_thw,
        patch_sha256: hash_f32(&input.patches),
        projected_sha256: hash_bf16(first.embeddings_bf16()),
        projected_prefix_bf16: first.embeddings_bf16()[..16].to_vec(),
        patch_tokens: first.patch_tokens(),
        visual_tokens: first.visual_tokens(),
        dispatches: first.dispatches() + second.dispatches(),
        fallback_used: first.fallback_used() || second.fallback_used(),
        all_dispatches_hip: first.all_dispatches_hip() && second.all_dispatches_hip(),
        deterministic_replay,
        text_prefill_token,
        text_decode_token,
        text_dispatches: text_audit.kernel_dispatch_count(),
        text_fallback_used: text_audit.fallback_used(),
        elapsed_ms: started.elapsed().as_millis(),
        cleanup_empty: false,
    };
    drop(text_request);
    drop(text_resident);
    drop(resident);
    let cleanup = session
        .shutdown(Duration::from_secs(30))
        .map_err(|error| error.to_string())?;
    let cleanup_empty = cleanup.retryable_cleanup == 0 && cleanup.durable_quarantine == 0;
    if !cleanup_empty {
        return Err("vision evidence cleanup was not empty".to_owned());
    }
    Ok(Report {
        cleanup_empty,
        ..report
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(value) => {
                println!("{value}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("vision evidence serialization failed: {error}");
                ExitCode::from(2)
            }
        },
        Err(error) => {
            eprintln!("vision evidence failed: {error}");
            ExitCode::from(2)
        }
    }
}
