// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Full-model AQ4_0 KV-cache dtype measurement harness.
//!
//! This binary deliberately owns no GPU-isolation or service lifecycle policy:
//! its caller must hold `/run/ullm/r9700.lock`, expose only the R9700, and set
//! the production AQ4 guards before it is started.  It records the selected
//! KV storage dtype from the process environment, but never changes it.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ullm_engine::aq4_worker_backend::QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV;
use ullm_engine::execution_batch::ExecutionPhase;
use ullm_engine::kv_cache_dtype::{KvCacheDtypes, KvCacheLayout};
use ullm_engine::qwen35_aq4_head_runtime::{PackageLmHeadMode, PackageTokenLogit};
use ullm_engine::qwen35_aq4_model_runtime::{
    QWEN35_AQ4_CONTEXT_LENGTH, QWEN35_AQ4_KV_BLOCK_SIZE, Qwen35Aq4CalibrationObserver,
    Qwen35Aq4ModelLoadConfig, Qwen35Aq4ModelRuntime,
};
use ullm_engine::qwen35_aq4_session::{QWEN35_AQ4_ROPE_BASE, QWEN35_AQ4_ROTARY_DIM};
use ullm_engine::qwen35_package_contract::PackageDecoderLayerKind;

const DEFAULT_PACKAGE: &str = "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/package";
const DEFAULT_DEVICE_INDEX: u32 = 1;
const REQUIRED_ARCHITECTURE: &str = "gfx1201";
const LOAD_CHUNK_BYTES: usize = 1024 * 1024;
const LM_HEAD_CHUNK_ROWS: usize = 8192;
const TOP_K: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Capacity,
    Prefill,
    Decode,
    Generate,
    Snapshot,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "capacity" => Ok(Self::Capacity),
            "prefill" => Ok(Self::Prefill),
            "decode" => Ok(Self::Decode),
            "generate" => Ok(Self::Generate),
            "snapshot" => Ok(Self::Snapshot),
            _ => Err(format!(
                "unknown --mode {value:?}; choose capacity, prefill, decode, generate, or snapshot"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Capacity => "capacity",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Generate => "generate",
            Self::Snapshot => "snapshot",
        }
    }
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    output: PathBuf,
    package_dir: PathBuf,
    device_index: u32,
    context_length: usize,
    token_count: usize,
    prefix_tokens: usize,
    generated_tokens: usize,
    repeats: usize,
    warmup: usize,
    prefill_width: usize,
    token_ids_file: Option<PathBuf>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ullm-aq4-kv-cache-dtype-measure: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let dtypes = KvCacheDtypes::from_env()?;
    require_caller_environment(dtypes)?;
    require_isolated_r9700(args.device_index, "before model load")?;

    let mut model = Qwen35Aq4ModelRuntime::load(Qwen35Aq4ModelLoadConfig {
        package_dir: args.package_dir.clone(),
        device_index: args.device_index,
        expected_architecture: Some(REQUIRED_ARCHITECTURE.to_string()),
        chunk_bytes: LOAD_CHUNK_BYTES,
        context_length: args.context_length,
        kv_block_size: QWEN35_AQ4_KV_BLOCK_SIZE,
        layer_indices: None,
        lm_head_mode: PackageLmHeadMode::GpuResidentF32,
        lm_head_chunk_rows: LM_HEAD_CHUNK_ROWS,
    })?;
    require_isolated_r9700(args.device_index, "after model load")?;
    if model.backend() != "hip" {
        return Err(format!(
            "AQ4 dtype measurement requires HIP backend, got {}",
            model.backend()
        ));
    }
    if model.geometry().context_length != args.context_length {
        return Err("loaded context length differs from requested context length".into());
    }

    let common = common_record(&args, dtypes, &model)?;
    let result = match args.mode {
        Mode::Capacity => capacity_record(common, dtypes, &model)?,
        Mode::Prefill => prefill_record(common, &args, &mut model)?,
        Mode::Decode => decode_record(common, &args, &mut model)?,
        Mode::Generate => generate_record(common, &args, &mut model)?,
        Mode::Snapshot => snapshot_record(common, &args, &mut model)?,
    };
    write_json(&args.output, &result)?;
    model.shutdown_synchronized()?;
    Ok(())
}

fn common_record(
    args: &Args,
    dtypes: KvCacheDtypes,
    model: &Qwen35Aq4ModelRuntime,
) -> Result<serde_json::Map<String, Value>, String> {
    let geometry = model.geometry();
    let mut record = serde_json::Map::new();
    record.insert(
        "schema".into(),
        Value::String("ullm.aq4_kv_cache_dtype_measure.v0.1".into()),
    );
    record.insert("mode".into(), Value::String(args.mode.as_str().into()));
    record.insert("format_id".into(), Value::String("AQ4_0".into()));
    record.insert(
        "package_dir".into(),
        Value::String(args.package_dir.display().to_string()),
    );
    record.insert("device_index".into(), json!(args.device_index));
    record.insert("backend".into(), Value::String(model.backend().into()));
    record.insert(
        "device_name".into(),
        Value::String(model.device_name().into()),
    );
    record.insert(
        "device_total_global_mem_bytes".into(),
        json!(model.device_total_global_mem()),
    );
    record.insert(
        "kv_cache_dtype".into(),
        json!({"key": dtypes.key.as_str(), "value": dtypes.value.as_str()}),
    );
    record.insert(
        "geometry".into(),
        json!({
            "vocab": geometry.vocab,
            "hidden": geometry.hidden,
            "context_length": geometry.context_length,
            "block_size": geometry.block_size,
            "cache_blocks": geometry.cache_blocks,
            "self_attention": geometry.self_attention.as_ref().map(|attention| json!({
                "q_heads": attention.q_heads,
                "kv_heads": attention.kv_heads,
                "head_dim": attention.head_dim,
                "value_dim": attention.value_dim,
            })),
        }),
    );
    Ok(record)
}

fn capacity_record(
    mut record: serde_json::Map<String, Value>,
    dtypes: KvCacheDtypes,
    model: &Qwen35Aq4ModelRuntime,
) -> Result<Value, String> {
    let geometry = model.geometry();
    let attention = geometry
        .self_attention
        .as_ref()
        .ok_or_else(|| "AQ4 package has no full-attention geometry".to_string())?;
    let physical_tokens = geometry
        .block_size
        .checked_mul(geometry.cache_blocks)
        .ok_or_else(|| "physical KV token capacity overflows".to_string())?;
    let layout = KvCacheLayout::new(
        dtypes,
        physical_tokens,
        attention.kv_heads,
        attention.head_dim,
        attention.value_dim,
    )?;
    let self_attention_layers = geometry
        .layers
        .iter()
        .filter(|layer| layer.kind == PackageDecoderLayerKind::SelfAttention)
        .count();
    let one_layer = layout.total_bytes()?;
    let all_layers = one_layer
        .checked_mul(self_attention_layers)
        .ok_or_else(|| "all-layer KV allocation byte count overflows".to_string())?;
    record.insert(
        "capacity".into(),
        json!({
            "model_load_succeeded": true,
            "requested_context_tokens": geometry.context_length,
            "physical_kv_tokens": physical_tokens,
            "self_attention_layers": self_attention_layers,
            "per_self_attention_layer": {
                "k_payload_bytes": layout.k_payload_bytes,
                "v_payload_bytes": layout.v_payload_bytes,
                "k_scale_bytes": layout.k_scale_bytes,
                "v_scale_bytes": layout.v_scale_bytes,
                "total_bytes": one_layer,
            },
            "all_self_attention_layers_total_bytes": all_layers,
        }),
    );
    Ok(Value::Object(record))
}

fn prefill_record(
    mut record: serde_json::Map<String, Value>,
    args: &Args,
    model: &mut Qwen35Aq4ModelRuntime,
) -> Result<Value, String> {
    let tokens = input_tokens(args, model.geometry().vocab, args.token_count)?;
    validate_prefill_tokens(&tokens, model.geometry().context_length)?;
    let mut warmups = Vec::with_capacity(args.warmup);
    for index in 0..args.warmup {
        let (seconds, _) = timed_prefill(model, &tokens, args.prefill_width, false)?;
        warmups.push(seconds);
        model.reset_all_request_state_synchronized()?;
        require_nonzero_elapsed(seconds, &format!("prefill warmup {index}"))?;
    }
    let mut elapsed = Vec::with_capacity(args.repeats);
    let mut final_top = Vec::new();
    for index in 0..args.repeats {
        let (seconds, top) = timed_prefill(model, &tokens, args.prefill_width, true)?;
        require_nonzero_elapsed(seconds, &format!("prefill repeat {index}"))?;
        elapsed.push(seconds);
        final_top = top;
        model.reset_all_request_state_synchronized()?;
    }
    record.insert(
        "prefill".into(),
        timing_record(
            tokens.len(),
            args.prefill_width,
            &warmups,
            &elapsed,
            &final_top,
        )?,
    );
    Ok(Value::Object(record))
}

fn decode_record(
    mut record: serde_json::Map<String, Value>,
    args: &Args,
    model: &mut Qwen35Aq4ModelRuntime,
) -> Result<Value, String> {
    if args.generated_tokens == 0 {
        return Err("--generated-tokens must be positive for decode".into());
    }
    let max_prefix = model
        .geometry()
        .context_length
        .checked_sub(args.generated_tokens)
        .ok_or_else(|| "decode generated tokens exceed context length".to_string())?;
    let requested_prefix = args.prefix_tokens.min(max_prefix);
    let tokens = input_tokens(args, model.geometry().vocab, requested_prefix)?;
    if tokens.len() > max_prefix {
        return Err(format!(
            "decode prefix has {} tokens but context only leaves {max_prefix} before {} decode tokens",
            tokens.len(),
            args.generated_tokens
        ));
    }
    validate_prefill_tokens(&tokens, model.geometry().context_length)?;
    let mut warmups = Vec::with_capacity(args.warmup);
    for index in 0..args.warmup {
        let (seconds, _) = timed_decode(model, &tokens, args.prefill_width, args.generated_tokens)?;
        require_nonzero_elapsed(seconds, &format!("decode warmup {index}"))?;
        warmups.push(seconds);
        model.reset_all_request_state_synchronized()?;
    }
    let mut elapsed = Vec::with_capacity(args.repeats);
    let mut generated = Vec::new();
    let mut final_top = Vec::new();
    for index in 0..args.repeats {
        let (seconds, output, top) =
            timed_decode_with_top(model, &tokens, args.prefill_width, args.generated_tokens)?;
        require_nonzero_elapsed(seconds, &format!("decode repeat {index}"))?;
        elapsed.push(seconds);
        generated = output;
        final_top = top;
        model.reset_all_request_state_synchronized()?;
    }
    let mut timing = timing_record(
        args.generated_tokens,
        args.prefill_width,
        &warmups,
        &elapsed,
        &final_top,
    )?;
    timing["prefix_tokens"] = json!(tokens.len());
    timing["generated_token_ids_last_repeat"] = json!(generated);
    record.insert("decode".into(), timing);
    Ok(Value::Object(record))
}

fn generate_record(
    mut record: serde_json::Map<String, Value>,
    args: &Args,
    model: &mut Qwen35Aq4ModelRuntime,
) -> Result<Value, String> {
    if args.generated_tokens == 0 {
        return Err("--generated-tokens must be positive for generate".into());
    }
    let tokens = input_tokens(args, model.geometry().vocab, args.prefix_tokens)?;
    let max_total = tokens
        .len()
        .checked_add(args.generated_tokens)
        .ok_or_else(|| "generation token count overflows".to_string())?;
    if max_total > model.geometry().context_length {
        return Err(format!(
            "generation needs {max_total} cache positions, exceeding context length {}",
            model.geometry().context_length
        ));
    }
    validate_prefill_tokens(&tokens, model.geometry().context_length)?;
    prefill(
        model,
        &tokens,
        args.prefill_width,
        ExecutionPhase::ColdPrefill,
    )?;
    model.synchronize()?;
    let mut generated = Vec::with_capacity(args.generated_tokens);
    let mut top = model.top_logits_from_last_layer(TOP_K, "aq4-kv-dtype-generate-initial")?;
    for step in 0..args.generated_tokens {
        let next = top
            .first()
            .ok_or_else(|| "AQ4 generation returned no top logit".to_string())?
            .token_id;
        generated.push(next);
        if step + 1 != args.generated_tokens {
            let position = tokens.len() + step;
            model.dispatch_token_for_phase(
                next,
                QWEN35_AQ4_ROTARY_DIM,
                QWEN35_AQ4_ROPE_BASE,
                position,
                position,
                ExecutionPhase::Decode,
                false,
                "aq4-kv-dtype-generate-decode",
            )?;
            top = model.top_logits_from_last_layer(TOP_K, "aq4-kv-dtype-generate-top")?;
        }
    }
    record.insert(
        "generation".into(),
        json!({
            "input_token_count": tokens.len(),
            "input_token_ids_sha256": token_ids_sha256(&tokens),
            "generated_token_ids": generated,
            "topk_after_last_processed_token": top_logits_json(&top),
        }),
    );
    Ok(Value::Object(record))
}

fn snapshot_record(
    mut record: serde_json::Map<String, Value>,
    args: &Args,
    model: &mut Qwen35Aq4ModelRuntime,
) -> Result<Value, String> {
    let tokens = input_tokens(args, model.geometry().vocab, args.token_count)?;
    validate_prefill_tokens(&tokens, model.geometry().context_length)?;
    prefill(
        model,
        &tokens,
        args.prefill_width,
        ExecutionPhase::ColdPrefill,
    )?;
    model.synchronize()?;
    let top = model.top_logits_from_last_layer(TOP_K, "aq4-kv-dtype-snapshot-top")?;
    let epoch = model
        .last_generation_state_epoch()
        .ok_or_else(|| "snapshot has no generation-state epoch".to_string())?;
    let mut observer = StateHashObserver::default();
    model.visit_last_generation_state(epoch, &mut observer)?;
    let summary = observer.summary()?;
    record.insert(
        "snapshot".into(),
        json!({
            "input_token_count": tokens.len(),
            "input_token_ids_sha256": token_ids_sha256(&tokens),
            "topk": top_logits_json(&top),
            "state": summary,
        }),
    );
    Ok(Value::Object(record))
}

fn timed_prefill(
    model: &mut Qwen35Aq4ModelRuntime,
    tokens: &[usize],
    prefill_width: usize,
    keep_top: bool,
) -> Result<(f64, Vec<PackageTokenLogit>), String> {
    let started = Instant::now();
    prefill(model, tokens, prefill_width, ExecutionPhase::ColdPrefill)?;
    let top = model.top_logits_from_last_layer(TOP_K, "aq4-kv-dtype-prefill-top")?;
    model.synchronize()?;
    let seconds = started.elapsed().as_secs_f64();
    Ok((seconds, if keep_top { top } else { Vec::new() }))
}

fn timed_decode(
    model: &mut Qwen35Aq4ModelRuntime,
    tokens: &[usize],
    prefill_width: usize,
    generated_tokens: usize,
) -> Result<(f64, Vec<usize>), String> {
    let (seconds, output, _) =
        timed_decode_with_top(model, tokens, prefill_width, generated_tokens)?;
    Ok((seconds, output))
}

fn timed_decode_with_top(
    model: &mut Qwen35Aq4ModelRuntime,
    tokens: &[usize],
    prefill_width: usize,
    generated_tokens: usize,
) -> Result<(f64, Vec<usize>, Vec<PackageTokenLogit>), String> {
    prefill(model, tokens, prefill_width, ExecutionPhase::ColdPrefill)?;
    model.synchronize()?;
    let mut top = model.top_logits_from_last_layer(TOP_K, "aq4-kv-dtype-decode-prime")?;
    let mut output = Vec::with_capacity(generated_tokens);
    let started = Instant::now();
    for step in 0..generated_tokens {
        let next = top
            .first()
            .ok_or_else(|| "AQ4 decode returned no top logit".to_string())?
            .token_id;
        output.push(next);
        let position = tokens
            .len()
            .checked_add(step)
            .ok_or_else(|| "decode position overflows".to_string())?;
        model.dispatch_token_for_phase(
            next,
            QWEN35_AQ4_ROTARY_DIM,
            QWEN35_AQ4_ROPE_BASE,
            position,
            position,
            ExecutionPhase::Decode,
            false,
            "aq4-kv-dtype-decode",
        )?;
        top = model.top_logits_from_last_layer(TOP_K, "aq4-kv-dtype-decode-top")?;
    }
    model.synchronize()?;
    Ok((started.elapsed().as_secs_f64(), output, top))
}

fn prefill(
    model: &mut Qwen35Aq4ModelRuntime,
    tokens: &[usize],
    prefill_width: usize,
    phase: ExecutionPhase,
) -> Result<(), String> {
    if !(2..=128).contains(&prefill_width) {
        return Err("--prefill-width must be in 2..=128".into());
    }
    let mut offset = 0usize;
    while offset < tokens.len() {
        let width = prefill_width.min(tokens.len() - offset);
        if width == 1 {
            model.dispatch_token_for_phase(
                tokens[offset],
                QWEN35_AQ4_ROTARY_DIM,
                QWEN35_AQ4_ROPE_BASE,
                offset,
                offset,
                phase,
                false,
                "aq4-kv-dtype-prefill-m1-tail",
            )?;
        } else {
            let step = model.dispatch_prefill_chunk_for_phase(
                &tokens[offset..offset + width],
                QWEN35_AQ4_ROTARY_DIM,
                QWEN35_AQ4_ROPE_BASE,
                offset,
                phase,
                false,
                "aq4-kv-dtype-prefill",
            )?;
            if step.execution_width != width {
                return Err(format!(
                    "native prefill executed M={} for requested M={width}",
                    step.execution_width
                ));
            }
        }
        offset += width;
    }
    Ok(())
}

fn timing_record(
    work_tokens: usize,
    prefill_width: usize,
    warmups: &[f64],
    elapsed: &[f64],
    final_top: &[PackageTokenLogit],
) -> Result<Value, String> {
    if elapsed.is_empty() {
        return Err("--repeats must be positive".into());
    }
    let mean_seconds = elapsed.iter().sum::<f64>() / elapsed.len() as f64;
    if !mean_seconds.is_finite() || mean_seconds <= 0.0 {
        return Err("timed mean is invalid".into());
    }
    let tokens_per_second = work_tokens as f64 / mean_seconds;
    Ok(json!({
        "work_tokens": work_tokens,
        "prefill_width": prefill_width,
        "warmup_elapsed_seconds": warmups,
        "elapsed_seconds": elapsed,
        "mean_elapsed_seconds": mean_seconds,
        "min_elapsed_seconds": elapsed.iter().copied().fold(f64::INFINITY, f64::min),
        "max_elapsed_seconds": elapsed.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "tokens_per_second_mean": tokens_per_second,
        "topk_after_last_repeat": top_logits_json(final_top),
    }))
}

fn input_tokens(args: &Args, vocab: usize, default_count: usize) -> Result<Vec<usize>, String> {
    let tokens = if let Some(path) = &args.token_ids_file {
        let raw = fs::read(path).map_err(|error| {
            format!(
                "failed to read --token-ids-file {}: {error}",
                path.display()
            )
        })?;
        serde_json::from_slice::<Vec<usize>>(&raw)
            .map_err(|error| format!("--token-ids-file must be a JSON array of usize: {error}"))?
    } else {
        deterministic_token_ids(default_count, vocab)?
    };
    if tokens.is_empty() {
        return Err("input token sequence must be non-empty".into());
    }
    if tokens.iter().any(|token| *token >= vocab) {
        return Err(format!(
            "input token sequence contains an id outside vocabulary {vocab}"
        ));
    }
    Ok(tokens)
}

fn deterministic_token_ids(count: usize, vocab: usize) -> Result<Vec<usize>, String> {
    if count == 0 || vocab == 0 {
        return Err("deterministic token count and vocabulary must be positive".into());
    }
    Ok((0..count)
        .map(|index| (17usize.wrapping_add(index.wrapping_mul(7919))) % vocab)
        .collect())
}

fn validate_prefill_tokens(tokens: &[usize], context_length: usize) -> Result<(), String> {
    if tokens.len() > context_length {
        return Err(format!(
            "input has {} tokens, exceeding loaded context length {context_length}",
            tokens.len()
        ));
    }
    Ok(())
}

fn token_ids_sha256(tokens: &[usize]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update((*token as u64).to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn top_logits_json(entries: &[PackageTokenLogit]) -> Value {
    Value::Array(
        entries
            .iter()
            .map(|entry| json!({"token_id": entry.token_id, "logit": entry.logit}))
            .collect(),
    )
}

fn require_nonzero_elapsed(seconds: f64, label: &str) -> Result<(), String> {
    if !seconds.is_finite() || seconds <= 0.0 {
        Err(format!("{label} elapsed time is invalid: {seconds}"))
    } else {
        Ok(())
    }
}

fn require_caller_environment(dtypes: KvCacheDtypes) -> Result<(), String> {
    require_environment_value("HIP_VISIBLE_DEVICES")?;
    for name in QWEN35_AQ4_REQUIRED_HIP_KERNEL_ENV {
        require_environment_value(name)?;
    }
    if dtypes != KvCacheDtypes::default() {
        for name in [
            "ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_KERNEL",
            "ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_SPLIT_KERNEL",
            "ULLM_REQUIRE_HIP_TYPED_PAGED_KV_WRITE_KERNEL",
        ] {
            require_environment_value(name)?;
        }
    }
    Ok(())
}

fn require_environment_value(name: &str) -> Result<(), String> {
    if env::var(name).ok().as_deref() != Some("1") {
        return Err(format!(
            "{name} must be set to exactly 1 by the caller; this binary never changes GPU or kernel environment variables"
        ));
    }
    Ok(())
}

fn require_isolated_r9700(device_index: u32, stage: &str) -> Result<(), String> {
    let count = ullm_runtime_sys::device_count()
        .map_err(|error| format!("failed to query runtime device count {stage}: {error}"))?;
    if count != device_index + 1 {
        return Err(format!(
            "{stage}: expected exactly CPU device 0 plus isolated HIP device {device_index}, found {count} runtime devices"
        ));
    }
    let info = ullm_runtime_sys::device_info(device_index).map_err(|error| {
        format!("failed to query runtime device {device_index} {stage}: {error}")
    })?;
    if info.backend != "hip" || info.gcn_arch_name != REQUIRED_ARCHITECTURE {
        return Err(format!(
            "{stage}: runtime device {device_index} must be HIP {REQUIRED_ARCHITECTURE}, got backend={} architecture={}",
            info.backend, info.gcn_arch_name
        ));
    }
    Ok(())
}

fn write_json(path: &PathBuf, value: &Value) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("output path {} has no parent", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "failed to create output directory {}: {error}",
            parent.display()
        )
    })?;
    let rendered = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to serialize measurement output: {error}"))?;
    fs::write(path, rendered)
        .map_err(|error| format!("failed to write output {}: {error}", path.display()))
}

#[derive(Default)]
struct StateHashObserver {
    expected_hidden: Option<usize>,
    expected_logits: Option<usize>,
    hidden_seen: usize,
    logits_seen: usize,
    hidden_hash: Sha256,
    logits_hash: Sha256,
    topk: Vec<PackageTokenLogit>,
}

impl StateHashObserver {
    fn update_hash(hash: &mut Sha256, values: &[f32], label: &str) -> Result<(), String> {
        for value in values {
            if !value.is_finite() {
                return Err(format!("snapshot {label} contains a non-finite value"));
            }
            hash.update(value.to_le_bytes());
        }
        Ok(())
    }

    fn summary(self) -> Result<Value, String> {
        let expected_hidden = self
            .expected_hidden
            .ok_or_else(|| "snapshot observer did not begin".to_string())?;
        let expected_logits = self
            .expected_logits
            .ok_or_else(|| "snapshot observer did not begin".to_string())?;
        if self.hidden_seen != expected_hidden || self.logits_seen != expected_logits {
            return Err("snapshot observer ended with incomplete vectors".into());
        }
        Ok(json!({
            "hidden_elements": expected_hidden,
            "hidden_f32le_sha256": format!("{:x}", self.hidden_hash.finalize()),
            "logit_elements": expected_logits,
            "logits_f32le_sha256": format!("{:x}", self.logits_hash.finalize()),
            "full_logit_topk": top_logits_json(&self.topk),
        }))
    }

    fn update_topk(&mut self, start: usize, values: &[f32]) {
        for (offset, logit) in values.iter().copied().enumerate() {
            self.topk.push(PackageTokenLogit {
                token_id: start + offset,
                logit,
            });
        }
        self.topk.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        self.topk.truncate(TOP_K);
    }
}

impl Qwen35Aq4CalibrationObserver for StateHashObserver {
    fn begin(&mut self, hidden_elements: usize, logit_elements: usize) -> Result<(), String> {
        if self.expected_hidden.replace(hidden_elements).is_some()
            || self.expected_logits.replace(logit_elements).is_some()
        {
            return Err("snapshot observer began more than once".into());
        }
        Ok(())
    }

    fn observe_hidden_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        if start != self.hidden_seen {
            return Err("snapshot hidden chunks are not contiguous".into());
        }
        Self::update_hash(&mut self.hidden_hash, values, "hidden")?;
        self.hidden_seen = self
            .hidden_seen
            .checked_add(values.len())
            .ok_or_else(|| "snapshot hidden element count overflows".to_string())?;
        Ok(())
    }

    fn observe_logit_chunk(&mut self, start: usize, values: &[f32]) -> Result<(), String> {
        if start != self.logits_seen {
            return Err("snapshot logit chunks are not contiguous".into());
        }
        Self::update_hash(&mut self.logits_hash, values, "logits")?;
        self.update_topk(start, values);
        self.logits_seen = self
            .logits_seen
            .checked_add(values.len())
            .ok_or_else(|| "snapshot logit element count overflows".to_string())?;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        let expected_hidden = self
            .expected_hidden
            .ok_or_else(|| "snapshot observer finished before begin".to_string())?;
        let expected_logits = self
            .expected_logits
            .ok_or_else(|| "snapshot observer finished before begin".to_string())?;
        if self.hidden_seen != expected_hidden || self.logits_seen != expected_logits {
            return Err("snapshot observer finished with incomplete vectors".into());
        }
        Ok(())
    }
}

fn parse_args() -> Result<Args, String> {
    let mut mode = None;
    let mut output = None;
    let mut package_dir = PathBuf::from(DEFAULT_PACKAGE);
    let mut device_index = DEFAULT_DEVICE_INDEX;
    let mut context_length = QWEN35_AQ4_CONTEXT_LENGTH;
    let mut token_count = 128usize;
    let mut prefix_tokens = 3968usize;
    let mut generated_tokens = 64usize;
    let mut repeats = 5usize;
    let mut warmup = 1usize;
    let mut prefill_width = 128usize;
    let mut token_ids_file = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--mode" => mode = Some(Mode::parse(&required(&mut arguments, "--mode")?)?),
            "--output" => output = Some(PathBuf::from(required(&mut arguments, "--output")?)),
            "--package" => package_dir = PathBuf::from(required(&mut arguments, "--package")?),
            "--device-index" => {
                device_index = parse_u32(
                    "--device-index",
                    &required(&mut arguments, "--device-index")?,
                )?
            }
            "--context-length" => {
                context_length = parse_usize(
                    "--context-length",
                    &required(&mut arguments, "--context-length")?,
                )?
            }
            "--token-count" => {
                token_count =
                    parse_usize("--token-count", &required(&mut arguments, "--token-count")?)?
            }
            "--prefix-tokens" => {
                prefix_tokens = parse_usize(
                    "--prefix-tokens",
                    &required(&mut arguments, "--prefix-tokens")?,
                )?
            }
            "--generated-tokens" => {
                generated_tokens = parse_usize(
                    "--generated-tokens",
                    &required(&mut arguments, "--generated-tokens")?,
                )?
            }
            "--repeats" => {
                repeats = parse_usize("--repeats", &required(&mut arguments, "--repeats")?)?
            }
            "--warmup" => warmup = parse_usize("--warmup", &required(&mut arguments, "--warmup")?)?,
            "--prefill-width" => {
                prefill_width = parse_usize(
                    "--prefill-width",
                    &required(&mut arguments, "--prefill-width")?,
                )?
            }
            "--token-ids-file" => {
                token_ids_file = Some(PathBuf::from(required(&mut arguments, "--token-ids-file")?))
            }
            "--help" | "-h" => return Err(usage().to_string()),
            _ => return Err(format!("unknown argument {argument:?}; {}", usage())),
        }
    }
    let mode = mode.ok_or_else(|| format!("--mode is required; {}", usage()))?;
    let output = output.ok_or_else(|| format!("--output is required; {}", usage()))?;
    if context_length == 0 || token_count == 0 || prefix_tokens == 0 || repeats == 0 {
        return Err("context/token/prefix counts and repeats must be positive".into());
    }
    if !(2..=128).contains(&prefill_width) {
        return Err("--prefill-width must be in 2..=128".into());
    }
    Ok(Args {
        mode,
        output,
        package_dir,
        device_index,
        context_length,
        token_count,
        prefix_tokens,
        generated_tokens,
        repeats,
        warmup,
        prefill_width,
        token_ids_file,
    })
}

fn required(arguments: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{name} needs a value"))
}

fn parse_usize(name: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

fn parse_u32(name: &str, value: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

fn usage() -> &'static str {
    "usage: ullm-aq4-kv-cache-dtype-measure --mode capacity|prefill|decode|generate|snapshot --output FILE [--package DIR] [--device-index N] [--context-length N] [--token-count N] [--prefix-tokens N] [--generated-tokens N] [--repeats N] [--warmup N] [--prefill-width 2..128] [--token-ids-file JSON_ARRAY]"
}
