use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use sllm_core::{
    Backend, ExecutionSessionRequest, Gemma4ModelLock, Gemma4ResidentModel, KvCacheEncoding,
    KvCacheSelection, KvCacheSelectionRequest, KvCacheSelectionSource, KvFp8PhysicalVariant,
    ModelLock, OsSamplingRandom, QWEN_RUNTIME_MAX_CONTEXT_TOKENS, QwenComponentSelection,
    QwenExecutionRequest, QwenMultimodalImageEmbedding, QwenMultimodalPrompt, QwenResidentModel,
    QwenVisionExecutionInput, QwenVisionResidentModel, ReviewedModelLock, SamplingParametersV1,
    VerifiedCache, VerifiedGgufGemmaSource, VerifiedGgufQwen35Moe, VerifiedGgufWeightSource,
    WeightClassification, assemble_gguf_qwen35_multimodal_prompt,
    assemble_qwen35_multimodal_prompt, build_gguf_qwen35_moe_weight_load_plan,
    build_qwen35_fp8_fnuz_graph, build_qwen35_fp8_graph, build_qwen35_gguf_fp8_graph,
    build_qwen35_gguf_moe_execution_graph, build_qwen35_graph,
    build_qwen35_graph_with_kv_cache_encoding, build_qwen35_graph_with_kv_cache_selection,
    build_qwen35_mtp_graph, build_qwen35_multimodal_graph, build_qwen35_nvfp4_graph,
    build_verified_gguf_gemma_weight_load_plan, build_verified_gguf_qwen_weight_load_plan,
    build_verified_gguf_qwen35_vision_manifest, build_verified_qwen_component_weight_load_plan,
    build_verified_qwen35_vision_manifest, builtin_reviewed_model_lock, qwen_graph_memory_estimate,
    qwen_prefill_chunk_candidates, qwen35_moe_generation_stop_policy, read_derived_gguf_lock,
    resolve_kv_cache_selection, verify_derived_gguf, verify_fp8_sidecar, verify_gguf_qwen35_moe,
    verify_nvfp4_sidecar,
};
use sllm_frontend::{
    BoundedImageBytesV1, DecodeModeV1, GenerationCancellationV1, GenerationConfigV1,
    GenerationExecutorV1, GenerationInputV1 as ServiceGenerationInputV1, GenerationReportV1,
    GenerationServiceError, GenerationServiceV1, GenerationStepV1, GenerationStopControllerV1,
    GenerationStopPolicyV1, GenericTemplateInputV1, GenericTemplateMessagesInputV1,
    GenericTemplateProviderV1, InputTokenCountInputV1, ProcessedVisionInputV1, Qwen35ChatMessageV1,
    Qwen35ChatTemplateV1, Qwen35RenderOptionsV1, Qwen35VisionProcessorV1,
    QwenMtpGenerationExecutorV1, SpeculativeGenerationAdapterV1, ThinkingModeV1, TokenIdsV1,
    TokenPieceV1, TokenizerFrontendV1, TokenizerUtilityServiceV1, gemma4_generation_stop_policy,
};
use sllm_hip::HipBackend;

use crate::benchmark::{
    BenchmarkEvent, BenchmarkSampleInput, BenchmarkTimeline, BenchmarkTiming,
    DIRECT_BENCHMARK_SCHEMA_VERSION, MonotonicClock, RENDER_TOKENIZE_BENCHMARK_SCHEMA_VERSION,
    allocation_snapshot_value, compare_control_sample, control_comparison_contract,
    validate_fixed_input_token_ids, validate_model_ready_snapshot, validate_peak_vram_snapshot,
    validate_request_cleanup_snapshot, validate_resident_drop_snapshot, validate_sample_count,
    validate_snapshot_accounting,
};

const REPORT_SCHEMA: &str = "model-frontend-cli-report-v1";
const MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOKEN_IDS: usize = 1_048_576;
const MAX_TOKEN_IDS_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EMBEDDING_INPUTS: usize = 256;
const MAX_PHASE42_AGGREGATE_BYTES: usize = 96 * 1024 * 1024;
const MAX_NEW_TOKENS: u32 = 4096;
const MAX_BENCHMARK_DIRECT_NEW_TOKENS: u32 = 20_000;
const MAX_BENCHMARK_CONTEXT_LENGTH: u64 = QWEN_RUNTIME_MAX_CONTEXT_TOKENS;
const DEFAULT_BENCHMARK_COMPLETION_TIMEOUT_SECONDS: u64 = 120;
const MAX_BENCHMARK_COMPLETION_TIMEOUT_SECONDS: u64 = 86_400;
const MAX_PREFILL_CHUNK_TOKENS: u64 = 16_384;
const MAX_MTP_DRAFT_WIDTH: u8 = QwenMtpGenerationExecutorV1::MAX_DRAFT_WIDTH as u8;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Eq, PartialEq)]
enum GenerationInput {
    Prompt(String),
    Messages {
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
    },
}

fn prepare_qwen_cli_images(
    input: &GenerationInput,
    image_paths: &[PathBuf],
) -> Result<(GenerationInput, Vec<ProcessedVisionInputV1>), String> {
    if image_paths.is_empty() {
        return Ok((input.clone(), Vec::new()));
    }
    let GenerationInput::Messages { messages, options } = input else {
        return Err("--image requires chat input via --message".to_owned());
    };
    let encoded = image_paths
        .iter()
        .map(|path| {
            BoundedImageBytesV1::from_local_path(path)
                .map_err(|error| format!("image `{}` could not be read: {error}", path.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let decoded = encoded
        .iter()
        .map(|image| {
            image
                .decode_rgb()
                .map_err(|error| format!("local image could not be decoded: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let processed = Qwen35VisionProcessorV1
        .process_many(&decoded)
        .map_err(|error| format!("local image preprocessing failed: {error}"))?;
    let mut messages = messages.clone();
    let Some(Qwen35ChatMessageV1::User { content }) = messages.last_mut() else {
        return Err("--image requires the final --message to have role user".to_owned());
    };
    let mut prefix = String::new();
    for image in &processed {
        prefix.push_str("<|vision_start|>");
        for _ in 0..image.visual_tokens {
            prefix.push_str("<|image_pad|>");
        }
        prefix.push_str("<|vision_end|>");
    }
    prefix.push_str(content);
    *content = prefix;
    Ok((
        GenerationInput::Messages {
            messages,
            options: *options,
        },
        processed,
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CliFp8Provider {
    Native,
    NativeFnuz,
    ConvertedBf16,
    Nvfp4PackedDequant,
}

impl CliFp8Provider {
    const fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::NativeFnuz => "native-fnuz",
            Self::ConvertedBf16 => "converted-bf16",
            Self::Nvfp4PackedDequant => "nvfp4-packed-dequant",
        }
    }
}

fn select_cli_gguf_fp8_provider(target: &str) -> Result<CliFp8Provider, String> {
    match target {
        "gfx1201" => Ok(CliFp8Provider::Native),
        "gfx942" => Ok(CliFp8Provider::NativeFnuz),
        _ => Err(format!(
            "embedded E4M3FN GGUF recipe requires exact gfx1201 native or gfx942 native-fnuz provider; exact target {target} is unsupported"
        )),
    }
}

fn cli_gguf_fp8_provider_label(provider: CliFp8Provider) -> &'static str {
    match provider {
        CliFp8Provider::Native => "gguf-native",
        CliFp8Provider::NativeFnuz => "native-fnuz",
        CliFp8Provider::ConvertedBf16 | CliFp8Provider::Nvfp4PackedDequant => {
            unreachable!("GGUF embedded FP8 uses a native provider")
        }
    }
}

fn cli_fp8_dtype(provider: CliFp8Provider) -> sllm_core::DType {
    match provider {
        CliFp8Provider::Native => sllm_core::DType::F8E4M3Fn,
        CliFp8Provider::NativeFnuz => sllm_core::DType::F8E4M3FnuZ,
        CliFp8Provider::ConvertedBf16 | CliFp8Provider::Nvfp4PackedDequant => {
            unreachable!("GGUF embedded FP8 uses a native provider")
        }
    }
}

const fn cli_fp8_weight_encoding(provider: Option<CliFp8Provider>) -> &'static str {
    match provider {
        Some(CliFp8Provider::ConvertedBf16) => "bf16-converted-from-ocp-e4m3fn",
        Some(CliFp8Provider::NativeFnuz) => "e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32",
        Some(CliFp8Provider::Nvfp4PackedDequant) => "nvfp4-e2m1-block16-e4m3fn-tensor-f32",
        Some(CliFp8Provider::Native) => "ocp-e4m3fn-outer-f32",
        None => "bf16",
    }
}

fn select_cli_fp8_provider(
    has_sidecar: bool,
    requested: Option<CliFp8Provider>,
    target: &str,
) -> Result<Option<CliFp8Provider>, String> {
    if !has_sidecar {
        return if requested.is_none() {
            Ok(None)
        } else {
            Err("--fp8-provider requires an FP8 sidecar".to_owned())
        };
    }
    let selected = requested.unwrap_or(match target {
        "gfx1201" => CliFp8Provider::Native,
        "gfx942" => CliFp8Provider::NativeFnuz,
        _ => CliFp8Provider::ConvertedBf16,
    });
    if selected == CliFp8Provider::Nvfp4PackedDequant {
        return if matches!(target, "gfx1201" | "gfx1030") {
            Ok(Some(selected))
        } else {
            Err(format!(
                "NVFP4 packed-dequant provider is incompatible with exact target {target}"
            ))
        };
    }
    let valid = matches!(
        (selected, target),
        (CliFp8Provider::Native, "gfx1201")
            | (CliFp8Provider::NativeFnuz, "gfx942")
            | (CliFp8Provider::ConvertedBf16, "gfx1030")
    );
    if !valid {
        return Err(format!(
            "FP8 provider {} is incompatible with exact target {target}",
            selected.label()
        ));
    }
    Ok(Some(selected))
}

fn resolve_cli_kv_cache_selection(
    requested: Option<KvCacheEncoding>,
    target: &str,
    model_fingerprint: &str,
    dense_text: bool,
    full_attention: bool,
    head_dim: usize,
) -> Result<KvCacheSelection, String> {
    resolve_kv_cache_selection(KvCacheSelectionRequest::new(
        requested,
        target,
        model_fingerprint,
        dense_text,
        full_attention,
        true,
        head_dim,
    ))
    .map_err(|error| error.to_string())
}

const fn kv_selection_source_name(source: KvCacheSelectionSource) -> &'static str {
    match source {
        KvCacheSelectionSource::Explicit => "process-explicit",
        KvCacheSelectionSource::Mxfp8E4Default => "mxfp8-e4-default",
        KvCacheSelectionSource::ModelFixedFp16 => "model-fixed-fp16",
    }
}

const fn kv_physical_variant_name(
    variant: KvFp8PhysicalVariant,
    standard_mxfp8: bool,
) -> &'static str {
    match (variant, standard_mxfp8) {
        (KvFp8PhysicalVariant::OcpE5M2, true) => "E5M2-OCP",
        (KvFp8PhysicalVariant::OcpE4M3Fn, _) => "E4M3-OCP",
        (KvFp8PhysicalVariant::E4M3FnuZ, _) => "E4M3-FNUZ",
        (KvFp8PhysicalVariant::OcpE5M2, false) => "E5M2-software",
    }
}

fn kv_descriptor_id(selection: KvCacheSelection) -> Option<String> {
    selection
        .block16_descriptor()
        .map(|descriptor| {
            format!(
                "{}-v{}",
                descriptor.encoding().canonical_name(),
                descriptor.format_version(),
            )
        })
        .or_else(|| {
            selection.mxfp8_descriptor().map(|descriptor| {
                format!(
                    "{}-v{}",
                    descriptor.encoding().canonical_name(),
                    descriptor.format_version(),
                )
            })
        })
}

fn kv_selection_report(selection: KvCacheSelection) -> Value {
    json!({
        "requested": selection.requested().map(KvCacheEncoding::canonical_name).unwrap_or("auto"),
        "resolved": selection.resolved().canonical_name(),
        "selection_source": kv_selection_source_name(selection.source()),
        "reason": selection.reason(),
        "physical_variant": selection
            .physical_variant()
            .map(|variant| kv_physical_variant_name(variant, selection.mxfp8_descriptor().is_some())),
        "descriptor_id": kv_descriptor_id(selection),
        "policy_version": selection.policy_version(),
    })
}

#[derive(Debug, PartialEq)]
struct GenerateRequest {
    input: GenerationInput,
    image_paths: Vec<PathBuf>,
    max_new_tokens: u32,
    prefill_chunk_tokens: Option<u64>,
    mtp_draft_width: Option<u8>,
    sampling: SamplingParametersV1,
    seed: Option<u64>,
    stop_strings: Vec<String>,
    device_index: u32,
    target: String,
    /// `None` is the public `auto` request. It remains distinct from explicit
    /// `fp16` until the model lock and exact target are both known.
    kv_cache_encoding: Option<KvCacheEncoding>,
    fp8_manifest: Option<PathBuf>,
    fp8_artifact: Option<PathBuf>,
    fp8_provider: Option<CliFp8Provider>,
}

struct CliQwenMultimodalExecutor<'a> {
    inner: &'a mut QwenExecutionRequest,
    prompt: &'a QwenMultimodalPrompt,
    prefilled: bool,
}

impl GenerationExecutorV1 for CliQwenMultimodalExecutor<'_> {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        _include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if self.prefilled {
            return Err(GenerationServiceError::Execution(
                "multimodal prefill was requested twice".to_owned(),
            ));
        }
        let token_ids = input_token_ids
            .iter()
            .map(|token| i32::try_from(*token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = self
            .inner
            .prefill_multimodal_with_last_logits(
                &token_ids,
                &self.prompt.embeddings_bf16,
                &self.prompt.positions,
            )
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        self.prefilled = true;
        multimodal_step(output)
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = if include_last_logits {
            self.inner.decode_with_last_logits(token)
        } else {
            self.inner.decode(token)
        }
        .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        multimodal_step(output)
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }
}

fn multimodal_step(
    output: sllm_core::QwenExecutionOutput,
) -> Result<GenerationStepV1, GenerationServiceError> {
    let argmax = output
        .token_ids()
        .last()
        .copied()
        .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
    Ok(GenerationStepV1::new(
        u32::try_from(argmax).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
        output.last_logits().map(ToOwned::to_owned),
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchmarkLane {
    Direct,
    RenderTokenize,
}

impl BenchmarkLane {
    fn schema_version(self) -> &'static str {
        match self {
            Self::Direct => DIRECT_BENCHMARK_SCHEMA_VERSION,
            Self::RenderTokenize => RENDER_TOKENIZE_BENCHMARK_SCHEMA_VERSION,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum BenchmarkInput {
    TokenIds(TokenIdsV1),
    Messages {
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
    },
}

#[derive(Debug, Eq, PartialEq)]
struct BenchmarkRequest {
    lane: BenchmarkLane,
    row_id: String,
    model_size: String,
    case_id: String,
    input: BenchmarkInput,
    max_new_tokens: u32,
    ignore_eos: bool,
    context_length: Option<u64>,
    completion_timeout_seconds: Option<u64>,
    prefill_chunk_tokens: Option<u64>,
    device_index: u32,
    target: String,
    greedy: bool,
    warmups: u32,
    measured: u32,
    kv_cache_encoding: Option<KvCacheEncoding>,
    fp8_manifest: Option<PathBuf>,
    fp8_artifact: Option<PathBuf>,
    fp8_provider: Option<CliFp8Provider>,
}

trait GreedyExecution {
    fn prefill_last(&mut self, input_token_ids: &[i32]) -> Result<i32, String>;
    fn decode_one(&mut self, token_id: i32) -> Result<i32, String>;
}

impl GreedyExecution for QwenExecutionRequest {
    fn prefill_last(&mut self, input_token_ids: &[i32]) -> Result<i32, String> {
        let output = self
            .prefill(input_token_ids)
            .map_err(|error| format!("Qwen prefill failed: {error}"))?;
        output
            .token_ids()
            .last()
            .copied()
            .ok_or_else(|| "Qwen prefill published no argmax token".to_owned())
    }

    fn decode_one(&mut self, token_id: i32) -> Result<i32, String> {
        let output = self
            .decode(token_id)
            .map_err(|error| format!("Qwen decode failed: {error}"))?;
        if output.token_ids().len() != 1 {
            return Err("Qwen decode published a non-singleton argmax".to_owned());
        }
        Ok(output.token_ids()[0])
    }
}

struct GenerationOutcome {
    report: GenerationReportV1,
    #[cfg_attr(not(test), allow(dead_code))]
    decode_steps: u32,
}

#[cfg(test)]
fn run_greedy_generation(
    executor: &mut impl GreedyExecution,
    policy: &GenerationStopPolicyV1,
    max_new_tokens: u32,
    input_token_ids: &[u32],
) -> Result<GenerationOutcome, String> {
    run_greedy_generation_timed(executor, policy, max_new_tokens, input_token_ids, None)
}

fn run_greedy_generation_timed(
    executor: &mut impl GreedyExecution,
    policy: &GenerationStopPolicyV1,
    max_new_tokens: u32,
    input_token_ids: &[u32],
    timing: Option<(&mut BenchmarkTimeline, MonotonicClock)>,
) -> Result<GenerationOutcome, String> {
    let input_i32 = input_token_ids
        .iter()
        .map(|token| {
            i32::try_from(*token).map_err(|_| "generation input token does not fit I32".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut timing = timing;
    if let Some((timeline, clock)) = timing.as_mut() {
        timeline.record(BenchmarkEvent::PrefillSubmit, clock.now_ns())?;
    }
    let mut generated = executor.prefill_last(&input_i32)?;
    if let Some((timeline, clock)) = timing.as_mut() {
        timeline.record(BenchmarkEvent::PrefillComplete, clock.now_ns())?;
        timeline.record(BenchmarkEvent::FirstToken, clock.now_ns())?;
    }
    let mut controller = GenerationStopControllerV1::new_with_input_token_ids(
        policy,
        max_new_tokens,
        input_token_ids,
    )
    .map_err(|_| "generation stop policy could not be initialized".to_owned())?;
    let mut decode_steps = 0_u32;
    loop {
        let generated_u32 =
            u32::try_from(generated).map_err(|_| "Qwen argmax token was negative".to_owned())?;
        let decision = controller
            .observe_generated(generated_u32)
            .map_err(|_| "generated token violated the stop policy".to_owned())?;
        let Some(decode_input) = decision.decode_input_token_id() else {
            if let Some((timeline, clock)) = timing.as_mut() {
                timeline.record(BenchmarkEvent::Stop, clock.now_ns())?;
            }
            break;
        };
        generated = executor.decode_one(
            i32::try_from(decode_input).map_err(|_| "decode token does not fit I32".to_owned())?,
        )?;
        decode_steps = decode_steps
            .checked_add(1)
            .ok_or_else(|| "decode step count overflowed".to_owned())?;
        if let Some((timeline, clock)) = timing.as_mut() {
            timeline.record(BenchmarkEvent::LaterToken, clock.now_ns())?;
        }
    }
    Ok(GenerationOutcome {
        report: controller.into_report(),
        decode_steps,
    })
}

fn benchmark_stop_policy(
    policy: &GenerationStopPolicyV1,
    ignore_eos: bool,
) -> GenerationStopPolicyV1 {
    if !ignore_eos {
        return policy.clone();
    }
    let mut policy = policy.clone();
    // Keep the reviewed stop policy shape valid while making the benchmark's
    // explicit ignore-EOS mode unable to stop on any model vocabulary token.
    policy.stop_token_ids = vec![u32::MAX];
    policy
}

fn benchmark_state_capacity(
    input_len: u64,
    max_new_tokens: u32,
    context_length: Option<u64>,
) -> Result<u64, String> {
    let required = input_len
        .checked_add(u64::from(max_new_tokens))
        .ok_or_else(|| "benchmark state capacity overflowed".to_owned())?;
    let capacity = context_length.unwrap_or(required);
    if capacity < required {
        return Err(format!(
            "benchmark context length {capacity} is smaller than required input+output {required}"
        ));
    }
    if capacity > MAX_BENCHMARK_CONTEXT_LENGTH {
        return Err(format!(
            "benchmark context length must be in [1,{MAX_BENCHMARK_CONTEXT_LENGTH}]"
        ));
    }
    Ok(capacity)
}

fn benchmark_completion_timeout(seconds: Option<u64>) -> Result<Duration, String> {
    let seconds = seconds.unwrap_or(DEFAULT_BENCHMARK_COMPLETION_TIMEOUT_SECONDS);
    if seconds == 0 || seconds > MAX_BENCHMARK_COMPLETION_TIMEOUT_SECONDS {
        return Err(format!(
            "benchmark completion timeout must be in [1,{MAX_BENCHMARK_COMPLETION_TIMEOUT_SECONDS}] seconds"
        ));
    }
    Ok(Duration::from_secs(seconds))
}

fn validate_benchmark_protocol(warmups: u32, measured: u32) -> Result<(), String> {
    validate_sample_count(warmups, measured)?;
    if !matches!((warmups, measured), (3, 10) | (1, 3)) {
        return Err(
            "benchmark protocol requires exactly 3 warmups and 10 measured requests, or 1 warmup and 3 measured requests"
                .to_owned(),
        );
    }
    Ok(())
}

fn correctness_reference_from_warmup(sample: &Value) -> Result<Value, String> {
    let section = |name: &str| {
        sample
            .get(name)
            .cloned()
            .ok_or_else(|| format!("benchmark correctness reference {name} is absent"))
    };
    let mut comparison = control_comparison_contract();
    if let Some(object) = comparison.as_object_mut() {
        object.insert(
            "scope".to_owned(),
            json!("first_warmup_reference_against_every_remaining_warmup_and_measured_sample"),
        );
        object.insert("reference_source".to_owned(), json!("warmups.samples[0]"));
    }
    Ok(json!({
        "label": "correctness-reference",
        "execution_path": "first-warmup-sample",
        "timing_instrumentation": "on",
        "included_in_performance_statistics": false,
        "source": {
            "kind": "warmup-sample",
            "sample_index": 0,
            "request_count": 0,
        },
        "tokens": section("tokens")?,
        "stop": section("stop")?,
        "audit": section("audit")?,
        "memory": section("memory")?,
        "cleanup": {
            "reference_sample": true,
            "request_dropped": true,
            "allocator_cleanup_validated": true,
            "retryable_cleanup": 0,
            "durable_quarantine": 0,
        },
        "comparison": comparison,
    }))
}

#[derive(Debug, PartialEq)]
enum Operation {
    Verify,
    Tokenize {
        text: String,
        pieces: bool,
    },
    Render {
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
    },
    Decode {
        ids: TokenIdsV1,
        mode: DecodeModeV1,
    },
    ApplyTemplate {
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
        custom_template: Option<CustomTemplateSpec>,
    },
    InputTokens {
        text: Option<String>,
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
        custom_template: Option<CustomTemplateSpec>,
    },
    Embeddings {
        texts: Vec<String>,
        token_inputs: Vec<TokenIdsV1>,
        device_index: u32,
        target: String,
    },
    Rerank {
        query: String,
        documents: Vec<String>,
        top_n: Option<usize>,
        device_index: u32,
        target: String,
    },
    Infill {
        prefix: String,
        suffix: String,
    },
    Generate(GenerateRequest),
    Benchmark(BenchmarkRequest),
}

#[derive(Clone)]
struct CustomTemplateSpec {
    path: PathBuf,
    digest: String,
    kwargs: Map<String, Value>,
    provider: Option<GenericTemplateProviderV1>,
}

impl std::fmt::Debug for CustomTemplateSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CustomTemplateSpec")
            .field("digest", &self.digest)
            .field("kwargs", &self.kwargs)
            .field("loaded", &self.provider.is_some())
            .finish()
    }
}

impl PartialEq for CustomTemplateSpec {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.digest == other.digest
            && self.kwargs == other.kwargs
            && self
                .provider
                .as_ref()
                .map(GenericTemplateProviderV1::digest)
                == other
                    .provider
                    .as_ref()
                    .map(GenericTemplateProviderV1::digest)
    }
}

#[derive(Debug, PartialEq)]
struct Request {
    gguf: Option<PathBuf>,
    derived_lock: Option<PathBuf>,
    operation: Operation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ModelIdentity {
    repo_id: String,
    resolved_revision: String,
    lock_fingerprint: String,
}

fn utility_tokenize(
    tokenizer: &TokenizerFrontendV1,
    text: &str,
    include_pieces: bool,
) -> Result<Value, String> {
    let utility = TokenizerUtilityServiceV1::new(tokenizer, None);
    let result = if include_pieces {
        utility.tokenize_with_pieces(text)
    } else {
        utility.tokenize_default(text)
    }
    .map_err(|error| error.to_string())?;
    let pieces = result.pieces().map(|pieces| {
        pieces
            .iter()
            .map(|piece| match piece {
                TokenPieceV1::Utf8(value) => json!({"kind":"utf8", "text":value}),
                TokenPieceV1::Bytes(value) => json!({"kind":"bytes", "bytes":value}),
            })
            .collect::<Vec<_>>()
    });
    Ok(json!({
        "kind": "tokenize",
        "version": result.version(),
        "count": result.count(),
        "token_ids": result.token_ids().as_slice(),
        "pieces": pieces,
        "tokenizer_fingerprint": tokenizer.snapshot().fingerprint(),
    }))
}

fn utility_detokenize(
    tokenizer: &TokenizerFrontendV1,
    ids: &TokenIdsV1,
    mode: DecodeModeV1,
) -> Result<Value, String> {
    let utility = TokenizerUtilityServiceV1::new(tokenizer, None);
    let text = utility
        .detokenize(ids, mode)
        .map_err(|error| error.to_string())?;
    Ok(json!({
        "kind": "detokenize",
        "text": text,
        "token_count": ids.len(),
        "tokenizer_fingerprint": tokenizer.snapshot().fingerprint(),
    }))
}

fn utility_apply_template(
    tokenizer: &TokenizerFrontendV1,
    renderer: &Qwen35ChatTemplateV1,
    messages: &[Qwen35ChatMessageV1],
    options: Qwen35RenderOptionsV1,
) -> Result<Value, String> {
    let utility = TokenizerUtilityServiceV1::new(tokenizer, Some(renderer));
    let result = utility
        .apply_template(messages, options)
        .map_err(|error| error.to_string())?;
    let identity = result.identity();
    Ok(json!({
        "kind": "apply-template",
        "version": result.version(),
        "text": result.rendered(),
        "prompt": result.rendered(),
        "count": result.count(),
        "token_ids": result.token_ids().as_slice(),
        "tokenizer_fingerprint": tokenizer.snapshot().fingerprint(),
        "template": {
            "kind": identity.kind(),
            "version": identity.version(),
            "consistency_label": identity.consistency_label(),
            "digest": identity.digest(),
            "size_bytes": identity.size_bytes(),
        }
    }))
}

fn generic_message_values(messages: &[Qwen35ChatMessageV1]) -> Vec<Value> {
    messages
        .iter()
        .map(|message| match message {
            Qwen35ChatMessageV1::System { content } => {
                json!({"role":"system", "content": content})
            }
            Qwen35ChatMessageV1::User { content } => {
                json!({"role":"user", "content": content})
            }
            Qwen35ChatMessageV1::Assistant {
                content,
                reasoning_content,
            } => {
                let mut value = json!({"role":"assistant", "content": content});
                if let Some(reasoning_content) = reasoning_content {
                    value["reasoning_content"] = Value::String(reasoning_content.clone());
                }
                value
            }
        })
        .collect()
}

fn generic_special_tokens(tokenizer: &TokenizerFrontendV1) -> Map<String, Value> {
    let snapshot = tokenizer.snapshot();
    let mut tokens = Map::new();
    for role in snapshot.special_roles() {
        tokens.insert(
            role.role().to_owned(),
            Value::String(role.content().to_owned()),
        );
    }
    tokens.insert(
        "eos_token".to_owned(),
        Value::String(snapshot.tokenizer_eos().token().to_owned()),
    );
    tokens
}

fn generic_template_input(
    tokenizer: &TokenizerFrontendV1,
    messages: &[Qwen35ChatMessageV1],
    options: Qwen35RenderOptionsV1,
    kwargs: Map<String, Value>,
) -> Result<GenericTemplateInputV1, String> {
    let input = GenericTemplateMessagesInputV1::from_parts(
        generic_message_values(messages),
        kwargs,
        generic_special_tokens(tokenizer),
        options.add_generation_prompt,
        matches!(options.thinking, ThinkingModeV1::Enabled),
        None,
    )
    .map_err(|error| error.to_string())?;
    Ok(GenericTemplateInputV1::messages(input))
}

fn custom_template_identity_json(result: &sllm_frontend::ApplyTemplateResultV1) -> Value {
    let identity = result.identity();
    let generic = result.generic_identity();
    json!({
        "kind": identity.kind(),
        "version": identity.version(),
        "consistency_label": identity.consistency_label(),
        "digest": identity.digest(),
        "size_bytes": identity.size_bytes(),
        "template_digest": generic.map(|value| value.template_digest()),
        "source_size_bytes": generic.map(|value| value.source_size_bytes()),
        "kwargs_digest": generic.map(|value| value.kwargs_digest()),
        "rendered_digest": generic.map(|value| value.rendered_digest()),
        "rendered_bytes_digest": generic.map(|value| value.rendered_digest()),
        "rendered_size_bytes": generic.map(|value| value.rendered_size_bytes()),
        "profile_version": generic.map(|value| value.profile_version()),
    })
}

fn utility_apply_custom_template(
    tokenizer: &TokenizerFrontendV1,
    provider: &GenericTemplateProviderV1,
    messages: &[Qwen35ChatMessageV1],
    options: Qwen35RenderOptionsV1,
    kwargs: Map<String, Value>,
) -> Result<Value, String> {
    let utility = TokenizerUtilityServiceV1::new(tokenizer, None);
    let input = generic_template_input(tokenizer, messages, options, kwargs)?;
    let result = utility
        .apply_generic_template(provider, input)
        .map_err(|error| error.to_string())?;
    Ok(json!({
        "kind": "apply-template",
        "version": result.version(),
        "text": result.rendered(),
        "prompt": result.rendered(),
        "count": result.count(),
        "token_ids": result.token_ids().as_slice(),
        "tokenizer_fingerprint": tokenizer.snapshot().fingerprint(),
        "template": custom_template_identity_json(&result),
    }))
}

fn utility_input_tokens_custom(
    tokenizer: &TokenizerFrontendV1,
    provider: &GenericTemplateProviderV1,
    messages: &[Qwen35ChatMessageV1],
    options: Qwen35RenderOptionsV1,
    kwargs: Map<String, Value>,
) -> Result<Value, String> {
    let utility = TokenizerUtilityServiceV1::new(tokenizer, None);
    let input = generic_template_input(tokenizer, messages, options, kwargs)?;
    let result = utility
        .apply_generic_template(provider, input)
        .map_err(|error| error.to_string())?;
    Ok(json!({
        "kind": "input-tokens",
        "count": result.count(),
        "input_kind": "custom-messages",
        "tokenizer_fingerprint": tokenizer.snapshot().fingerprint(),
        "template": custom_template_identity_json(&result),
    }))
}

fn utility_input_tokens(
    tokenizer: &TokenizerFrontendV1,
    renderer: Option<&Qwen35ChatTemplateV1>,
    text: Option<&str>,
    messages: &[Qwen35ChatMessageV1],
    options: Qwen35RenderOptionsV1,
) -> Result<Value, String> {
    let utility = TokenizerUtilityServiceV1::new(tokenizer, renderer);
    let (count, input_kind) = if let Some(text) = text {
        (
            utility
                .input_token_count(InputTokenCountInputV1::RawText(text))
                .map_err(|error| error.to_string())?,
            "raw",
        )
    } else {
        (
            utility
                .input_token_count(InputTokenCountInputV1::Messages { messages, options })
                .map_err(|error| error.to_string())?,
            "messages",
        )
    };
    Ok(json!({
        "kind": "input-tokens",
        "count": count,
        "input_kind": input_kind,
        "tokenizer_fingerprint": tokenizer.snapshot().fingerprint(),
    }))
}

fn prepare_embedding_inputs(
    tokenizer: &TokenizerFrontendV1,
    texts: &[String],
    token_inputs: &[TokenIdsV1],
    max_context_tokens: u64,
) -> Result<Vec<Vec<i32>>, String> {
    if texts.is_empty() == token_inputs.is_empty() {
        return Err("embedding input must contain either text or token IDs".to_owned());
    }
    let vocab_size = tokenizer.snapshot().vocab_size();
    let mut result = Vec::with_capacity(texts.len().max(token_inputs.len()));
    if !texts.is_empty() {
        for text in texts {
            if text.is_empty() {
                return Err("embedding text must not be empty".to_owned());
            }
            let ids = tokenizer
                .encode(text)
                .map_err(|error| format!("embedding text could not be tokenized: {error}"))?;
            if ids.is_empty() {
                return Err("embedding text produced an empty token sequence".to_owned());
            }
            if u64::try_from(ids.len()).unwrap_or(u64::MAX) > max_context_tokens {
                return Err(format!(
                    "embedding input exceeds the model context limit {max_context_tokens}"
                ));
            }
            result.push(
                ids.as_slice()
                    .iter()
                    .map(|id| {
                        i32::try_from(*id).map_err(|_| "token ID does not fit i32".to_owned())
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    } else {
        for ids in token_inputs {
            if ids.is_empty() {
                return Err("embedding token sequence must not be empty".to_owned());
            }
            if u64::try_from(ids.len()).unwrap_or(u64::MAX) > max_context_tokens {
                return Err(format!(
                    "embedding input exceeds the model context limit {max_context_tokens}"
                ));
            }
            for id in ids.as_slice() {
                if u64::from(*id) >= vocab_size {
                    return Err(format!(
                        "embedding token ID {id} is outside the tokenizer vocabulary"
                    ));
                }
            }
            result.push(
                ids.as_slice()
                    .iter()
                    .map(|id| {
                        i32::try_from(*id).map_err(|_| "token ID does not fit i32".to_owned())
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    Ok(result)
}

fn embedding_response(vectors: Vec<Vec<f32>>, token_counts: &[usize]) -> Result<Value, String> {
    if vectors.len() != token_counts.len() {
        return Err("embedding result count differs from input count".to_owned());
    }
    let mut total_tokens = 0_u64;
    let mut data = Vec::with_capacity(vectors.len());
    for (index, (vector, tokens)) in vectors.into_iter().zip(token_counts).enumerate() {
        total_tokens = total_tokens
            .checked_add(
                u64::try_from(*tokens)
                    .map_err(|_| "embedding token count overflowed".to_owned())?,
            )
            .ok_or_else(|| "embedding usage overflowed".to_owned())?;
        if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
            return Err("embedding output was empty or non-finite".to_owned());
        }
        let squared_norm = vector
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>();
        if !squared_norm.is_finite() || (squared_norm - 1.0).abs() > 1.0e-4 {
            return Err("embedding output was not L2-normalized".to_owned());
        }
        data.push(json!({"index": index, "embedding": vector, "object": "embedding"}));
    }
    Ok(json!({
        "kind": "embeddings",
        "profile": sllm_core::EMBEDDING_POOL_PROFILE_V1,
        "object": "list",
        "data": data,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }))
}

fn rerank_from_embedding_response(
    response: &Value,
    query: &str,
    documents: &[String],
    top_n: Option<usize>,
) -> Result<Value, String> {
    let data = response
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| "embedding response omitted data".to_owned())?;
    if data.len() != documents.len() + 1 {
        return Err("rerank embedding count differs from input count".to_owned());
    }
    let query_vector = data[0]
        .get("embedding")
        .and_then(Value::as_array)
        .ok_or_else(|| "rerank query embedding was malformed".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| "rerank embedding was non-numeric".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut scored = Vec::with_capacity(documents.len());
    for (index, _document) in documents.iter().enumerate() {
        let values = data[index + 1]
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| "rerank document embedding was malformed".to_owned())?;
        if values.len() != query_vector.len() {
            return Err("rerank embedding dimensions differed".to_owned());
        }
        let score = query_vector
            .iter()
            .zip(values)
            .map(|(left, right)| {
                right
                    .as_f64()
                    .map(|right| left * right)
                    .ok_or_else(|| "rerank embedding was non-numeric".to_owned())
            })
            .try_fold(0.0_f64, |sum, value| value.map(|value| sum + value))?;
        if !score.is_finite() {
            return Err("rerank score was non-finite".to_owned());
        }
        scored.push((index, score as f32));
    }
    scored.sort_by(|(left_index, left_score), (right_index, right_score)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_index.cmp(right_index))
    });
    if let Some(top_n) = top_n {
        scored.truncate(top_n);
    }
    let usage = response.get("usage").cloned().unwrap_or_else(|| json!({}));
    Ok(json!({
        "kind": "rerank",
        "profile": sllm_core::COSINE_EMBEDDING_RERANK_PROFILE_V1,
        "results": scored.into_iter().map(|(index, score)| json!({
            "index": index,
            "relevance_score": score,
            "document": documents[index],
        })).collect::<Vec<_>>(),
        "query": query,
        "usage": usage,
    }))
}

trait ModelFrontendBackend {
    fn identity(&self) -> ModelIdentity;
    fn verify(&self) -> Result<Value, String>;
    fn tokenize(&self, text: &str) -> Result<Value, String>;
    fn tokenize_with_pieces(&self, text: &str) -> Result<Value, String> {
        self.tokenize(text)
    }
    fn render(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String>;
    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String>;
    fn apply_template(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        self.render(messages, options)
    }
    fn apply_template_custom(
        &self,
        _messages: &[Qwen35ChatMessageV1],
        _options: Qwen35RenderOptionsV1,
        _provider: &GenericTemplateProviderV1,
        _kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        Err("custom template is unsupported for this model backend".to_owned())
    }
    fn input_tokens(
        &self,
        text: Option<&str>,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        if let Some(text) = text {
            let result = self.tokenize(text)?;
            let count = result
                .get("count")
                .and_then(Value::as_u64)
                .ok_or_else(|| "tokenizer result omitted count".to_owned())?;
            return Ok(json!({"kind":"input-tokens", "count":count}));
        }
        let result = self.apply_template(messages, options)?;
        let rendered = result
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| "template result omitted text".to_owned())?;
        let tokenized = self.tokenize(rendered)?;
        let count = tokenized
            .get("count")
            .and_then(Value::as_u64)
            .ok_or_else(|| "tokenizer result omitted count".to_owned())?;
        Ok(json!({"kind":"input-tokens", "count":count}))
    }
    fn input_tokens_custom(
        &self,
        _messages: &[Qwen35ChatMessageV1],
        _options: Qwen35RenderOptionsV1,
        _provider: &GenericTemplateProviderV1,
        _kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        Err("custom template is unsupported for this model backend".to_owned())
    }
    fn detokenize(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        self.decode(ids, mode)
    }
    fn embeddings(
        &self,
        _texts: &[String],
        _token_inputs: &[TokenIdsV1],
        _device_index: u32,
        _target: &str,
    ) -> Result<Value, String> {
        Err("embedding execution is unavailable for this CLI model/backend".to_owned())
    }
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_n: Option<usize>,
        device_index: u32,
        target: &str,
    ) -> Result<Value, String> {
        let mut texts = Vec::with_capacity(documents.len() + 1);
        texts.push(query.to_owned());
        texts.extend(documents.iter().cloned());
        let response = self.embeddings(&texts, &[], device_index, target)?;
        rerank_from_embedding_response(&response, query, documents, top_n)
    }
    fn infill(&self, _prefix: &str, _suffix: &str) -> Result<Value, String> {
        Err("infill is unavailable: no verified production FIM capability".to_owned())
    }
    fn generate(&self, request: &GenerateRequest) -> Result<Value, String>;
    fn benchmark(
        &self,
        request: &BenchmarkRequest,
        timing: BenchmarkTiming,
    ) -> Result<Value, String>;
}

struct ProductionBackend {
    lock: ModelLock,
    lock_path: PathBuf,
    source: QwenDenseSource,
}

enum QwenDenseSource {
    // Kept for converter/development adapters; the public parser constructs GGUF only.
    #[allow(dead_code)]
    Cache(Arc<VerifiedCache>),
    Gguf(Arc<VerifiedGgufWeightSource>),
}

struct MoeProductionBackend {
    source: Arc<VerifiedGgufQwen35Moe>,
}

struct GemmaProductionBackend {
    lock: Gemma4ModelLock,
    source: Arc<VerifiedGgufGemmaSource>,
}

impl GemmaProductionBackend {
    fn open(lock: Gemma4ModelLock, request: &Request) -> Result<Self, String> {
        let gguf_path = request
            .gguf
            .as_ref()
            .expect("public parser requires a GGUF path");
        let derived_path = request
            .derived_lock
            .as_ref()
            .expect("public parser requires a derived lock");
        let derived = read_derived_gguf_lock(derived_path)
            .map_err(|error| format!("derived GGUF lock is invalid: {error}"))?;
        let verified = verify_derived_gguf(derived, gguf_path)
            .map_err(|error| format!("GGUF does not match its derived lock: {error}"))?;
        let (source, _) = build_verified_gguf_gemma_weight_load_plan(&lock, verified)
            .map_err(|error| format!("GGUF Gemma load plan is invalid: {error}"))?;
        let source = Arc::new(source);
        Ok(Self { lock, source })
    }

    fn tokenizer(&self) -> Result<TokenizerFrontendV1, String> {
        TokenizerFrontendV1::from_gemma4_gguf(&self.lock, self.source.gguf()).map_err(|error| {
            format!("verified Gemma 4 tokenizer could not be constructed: {error}")
        })
    }
}

impl ModelFrontendBackend for GemmaProductionBackend {
    fn identity(&self) -> ModelIdentity {
        ModelIdentity {
            repo_id: self.source.repository().to_owned(),
            resolved_revision: self.source.resolved_revision().to_owned(),
            lock_fingerprint: self.lock.fingerprint().to_owned(),
        }
    }

    fn verify(&self) -> Result<Value, String> {
        let plan = self
            .source
            .build_weight_load_plan(&self.lock)
            .map_err(|_| "GGUF tensors do not form the Gemma 4 load plan".to_owned())?;
        let verified_files = 1;
        let tensor_count = self.source.gguf().tensors().len();
        let weight_encoding = "mixed-nvfp4-w4a4-fp8-w8a8";
        let recipe_digest = Some(self.source.recipe_digest());
        let loadable = plan
            .entries
            .iter()
            .filter(|entry| entry.classification != WeightClassification::KnownUnconsumed)
            .count();
        Ok(json!({
            "kind": "verify-model",
            "model_kind": "gemma4-dense",
            "prompt_mode": self.lock.model.tokenizer_contract.prompt_mode,
            "chat_template": self.lock.supports_chat_messages(),
            "locked_files": self.lock.model.files.len(),
            "verified_files": verified_files,
            "tensor_count": tensor_count,
            "weight_entries": plan.entries.len(),
            "loadable_entries": loadable,
            "known_unconsumed_entries": plan.entries.len() - loadable,
            "total_destination_bytes": plan.total_destination_bytes,
            "plan_digest": plan.digest_hex(),
            "weight_encoding": weight_encoding,
            "recipe_digest": recipe_digest,
        }))
    }

    fn tokenize(&self, text: &str) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_tokenize(&tokenizer, text, false)
    }

    fn tokenize_with_pieces(&self, text: &str) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_tokenize(&tokenizer, text, true)
    }

    fn render(&self, _: &[Qwen35ChatMessageV1], _: Qwen35RenderOptionsV1) -> Result<Value, String> {
        Err("google/gemma-4-12B has no locked chat template; use a raw prompt".to_owned())
    }

    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let mut result = utility_detokenize(&tokenizer, ids, mode)?;
        result["kind"] = Value::from("decode");
        Ok(result)
    }

    fn detokenize(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_detokenize(&tokenizer, ids, mode)
    }

    fn embeddings(
        &self,
        texts: &[String],
        token_inputs: &[TokenIdsV1],
        device_index: u32,
        target: &str,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let inputs = prepare_embedding_inputs(
            &tokenizer,
            texts,
            token_inputs,
            self.lock.model.architecture.text.max_position_embeddings,
        )?;
        let token_counts = inputs.iter().map(Vec::len).collect::<Vec<_>>();
        let max_tokens = token_counts.iter().copied().max().unwrap_or(0);
        let token_count =
            u64::try_from(max_tokens).map_err(|_| "embedding token count overflowed".to_owned())?;
        let plan = self
            .source
            .build_weight_load_plan(&self.lock)
            .map_err(|error| format!("embedding load plan is unavailable: {error}"))?;
        let _graph = sllm_core::build_gemma4_graph(&self.lock, &plan, token_count, 0, token_count)
            .map_err(|error| format!("embedding graph is unavailable: {error}"))?;
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(device_index, target.to_owned())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let execution = (|| -> Result<Vec<Vec<f32>>, String> {
            let resident = Gemma4ResidentModel::new_gguf_quantized(
                Arc::clone(&session),
                self.lock.clone(),
                plan.clone(),
                Arc::clone(&self.source),
                COMPLETION_TIMEOUT,
            )
            .map_err(|error| format!("embedding resident provisioning failed: {error}"))?;
            let mut vectors = Vec::with_capacity(inputs.len());
            for input in &inputs {
                let mut owner = resident
                    .new_request(u64::try_from(input.len()).unwrap_or(0), token_count)
                    .map_err(|error| format!("embedding request provisioning failed: {error}"))?;
                let output = owner
                    .prefill_with_embeddings(input)
                    .map_err(|error| format!("embedding execution failed: {error}"))?;
                let audit = owner
                    .audit_snapshot()
                    .map_err(|_| "embedding dispatch audit was empty or invalid".to_owned())?;
                if audit.target() != target || audit.fallback_used() {
                    return Err(
                        "embedding dispatch audit differs from the exact HIP target".to_owned()
                    );
                }
                let rows = output.final_hidden_states_bf16().ok_or_else(|| {
                    "embedding execution did not publish final-normalized hidden rows".to_owned()
                })?;
                let hidden = self.lock.model.architecture.text.hidden_size as usize;
                let expected = input
                    .len()
                    .checked_mul(hidden)
                    .ok_or_else(|| "embedding hidden shape overflowed".to_owned())?;
                if rows.len() != expected {
                    return Err(
                        "embedding hidden readback shape differed from the model contract"
                            .to_owned(),
                    );
                }
                let vector = sllm_core::EmbeddingPoolV1::new()
                    .pool_bf16(rows, input.len(), hidden)
                    .map_err(|error| format!("embedding pooling failed: {error}"))?;
                vectors.push(vector.as_slice().to_vec());
            }
            Ok(vectors)
        })();
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|error| format!("HIP session cleanup failed: {error}"))?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        embedding_response(execution?, &token_counts)
    }

    fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
        let started = Instant::now();
        if request.prefill_chunk_tokens.is_some() || request.mtp_draft_width.is_some() {
            return Err(
                "--prefill-chunk-tokens and --mtp-draft-width are supported only for dense Qwen generation"
                    .to_owned(),
            );
        }
        if !request.image_paths.is_empty() {
            return Err("--image is supported only by Qwen3.5 vision models".to_owned());
        }
        if request.fp8_manifest.is_some()
            || request.fp8_artifact.is_some()
            || request.fp8_provider.is_some()
        {
            return Err(
                "Gemma 4 first-class model input does not accept development sidecar flags"
                    .to_owned(),
            );
        }
        let prompt = match &request.input {
            GenerationInput::Prompt(prompt) => prompt,
            GenerationInput::Messages { .. } => {
                return Err(
                    "google/gemma-4-12B has no locked chat template; use a raw prompt".to_owned(),
                );
            }
        };
        let tokenizer = self.tokenizer()?;
        let stop_policy = gemma4_generation_stop_policy(&self.lock)
            .map_err(|error| format!("Gemma stop policy is invalid: {error}"))?;
        let service = GenerationServiceV1::new(&tokenizer, None, &stop_policy)
            .map_err(|error| format!("generation service could not be constructed: {error}"))?;
        let input = service
            .prepare_input(&ServiceGenerationInputV1::Prompt(prompt.clone()))
            .map_err(|error| format!("generation input preparation failed: {error}"))?;
        let input_len = u64::try_from(input.len())
            .map_err(|_| "generation input token count overflowed".to_owned())?;
        let state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or_else(|| "generation state capacity overflowed".to_owned())?;
        let plan = self
            .source
            .build_weight_load_plan(&self.lock)
            .map_err(|_| "GGUF tensors do not form the Gemma 4 load plan".to_owned())?;
        let plan_digest = plan.digest_hex();
        let model_fingerprint = self.lock.fingerprint().to_owned();
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session_request =
            ExecutionSessionRequest::new(request.device_index, request.target.clone())
                .map_err(|_| "invalid exact HIP session request".to_owned())?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;

        let execution = (|| -> Result<Value, String> {
            let resident = Gemma4ResidentModel::new_gguf_quantized(
                Arc::clone(&session),
                self.lock.clone(),
                plan,
                Arc::clone(&self.source),
                COMPLETION_TIMEOUT,
            )
            .map_err(|error| format!("Gemma resident provisioning failed: {error}"))?;
            let mut owner = resident
                .new_request(input_len, state_capacity)
                .map_err(|error| format!("Gemma request provisioning failed: {error}"))?;
            let config = GenerationConfigV1::new(
                request.max_new_tokens,
                request.sampling,
                request.stop_strings.clone(),
            )
            .map_err(|error| format!("generation configuration is invalid: {error}"))?;
            let cancellation = GenerationCancellationV1::new();
            let mut random =
                OsSamplingRandom::for_parameters_and_seed(request.sampling, request.seed)
                    .map_err(|error| format!("sampling random source failed: {error}"))?;
            let report = service
                .generate_tokens(&mut owner, &input, &config, &cancellation, &mut random)
                .map_err(|error| format!("generation service failed: {error}"))?;
            let audit = owner
                .audit_snapshot()
                .map_err(|_| "Gemma dispatch audit was empty or invalid".to_owned())?;
            if audit.target() != request.target || audit.fallback_used() {
                return Err("Gemma dispatch audit differs from the exact target".to_owned());
            }
            Ok(json!({
                "kind": "generate",
                "input_kind": "prompt",
                "input_token_ids": report.input_token_ids(),
                "generated_token_ids": report.generated_token_ids(),
                "visible_token_ids": report.visible_token_ids(),
                "decode_input_token_ids": report.decode_input_token_ids(),
                "output_text": report.output_text(),
                "finish_reason": report.finish_reason().as_str(),
                "stop_reason": {
                    "version": 1,
                    "reason_version": 1,
                    "kind": report.finish_reason().as_str(),
                    "token_id": report.stop_token_id(),
                    "matched_string": report.matched_stop(),
                },
                "usage": {
                    "prompt_tokens": report.usage().prompt_tokens(),
                    "completion_tokens": report.usage().completion_tokens(),
                    "total_tokens": report.usage().total_tokens(),
                },
                "sampling": {
                    "temperature": request.sampling.temperature(),
                    "top_p": request.sampling.top_p(),
                    "presence_penalty": request.sampling.presence_penalty(),
                    "frequency_penalty": request.sampling.frequency_penalty(),
                },
                "execution": {
                    "selected_backend": "hip",
                    "target": audit.target(),
                    "device_index": request.device_index,
                    "model_fingerprint": model_fingerprint,
                    "plan_digest": plan_digest,
                    "prefill_tokens": input.len(),
                    "decode_steps": report.decode_steps(),
                    "fallback_used": audit.fallback_used(),
                    "submission_count": audit.submission_count(),
                    "kernel_dispatch_count": audit.kernel_dispatch_count(),
                    "segment_count": audit.segment_count(),
                    "boundary_count": audit.boundary_count(),
                    "all_dispatches_hip": true,
                    "weight_encoding": "mixed-nvfp4-w4a4-fp8-w8a8",
                    "fp8_provider": null,
                },
            }))
        })();
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|_| "HIP session cleanup failed".to_owned())?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        let mut result = execution?;
        let object = result
            .as_object_mut()
            .ok_or_else(|| "generation result was not an object".to_owned())?;
        object.insert(
            "timing_ns".to_owned(),
            Value::from(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)),
        );
        object.insert(
            "cleanup".to_owned(),
            json!({"retryable_cleanup": 0, "durable_quarantine": 0}),
        );
        Ok(result)
    }

    fn benchmark(&self, _: &BenchmarkRequest, _: BenchmarkTiming) -> Result<Value, String> {
        Err("Gemma 4 benchmark is unavailable until its exact executor is active".to_owned())
    }
}

fn open_production_backend(request: &Request) -> Result<Box<dyn ModelFrontendBackend>, String> {
    let gguf_path = request
        .gguf
        .as_ref()
        .expect("public parser requires a GGUF path");
    let derived_path = request
        .derived_lock
        .as_ref()
        .expect("public parser requires a derived lock");
    let derived = read_derived_gguf_lock(derived_path)
        .map_err(|error| format!("derived GGUF lock is invalid: {error}"))?;
    if derived.semantic_model_id.starts_with("qwen35moe:") {
        let verified = verify_derived_gguf(derived, gguf_path)
            .map_err(|error| format!("GGUF does not match its derived lock: {error}"))?;
        let source = verify_gguf_qwen35_moe(verified)
            .map_err(|error| format!("MoE GGUF is invalid: {error}"))?;
        return Ok(Box::new(MoeProductionBackend {
            source: Arc::new(source),
        }));
    }
    let reviewed = builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
        .map_err(|error| format!("derived GGUF source identity is unsupported: {error}"))?;
    match reviewed {
        ReviewedModelLock::Qwen35(lock) => {
            ProductionBackend::open(lock, request).map(|backend| Box::new(backend) as Box<_>)
        }
        ReviewedModelLock::Gemma4(lock) => {
            GemmaProductionBackend::open(lock, request).map(|backend| Box::new(backend) as Box<_>)
        }
    }
}

impl MoeProductionBackend {
    fn plan(&self) -> Result<sllm_core::WeightLoadPlan, String> {
        build_gguf_qwen35_moe_weight_load_plan(&self.source).map_err(|error| error.to_string())
    }

    fn tokenizer(&self) -> Result<TokenizerFrontendV1, String> {
        TokenizerFrontendV1::from_qwen35_moe_gguf(&self.source).map_err(|error| error.to_string())
    }

    fn renderer(&self) -> Result<Qwen35ChatTemplateV1, String> {
        Qwen35ChatTemplateV1::from_qwen35_moe_gguf(&self.source).map_err(|error| error.to_string())
    }

    fn graph(
        &self,
        plan: &sllm_core::WeightLoadPlan,
        token_count: u64,
        state_capacity: u64,
    ) -> Result<sllm_core::QwenGraph, String> {
        build_qwen35_gguf_moe_execution_graph(&self.source, plan, token_count, state_capacity)
            .map_err(|error| error.to_string())
    }
}

impl ModelFrontendBackend for MoeProductionBackend {
    fn identity(&self) -> ModelIdentity {
        ModelIdentity {
            repo_id: sllm_core::QWEN35_MOE_REPOSITORY.to_owned(),
            resolved_revision: sllm_core::QWEN35_MOE_REVISION.to_owned(),
            lock_fingerprint: sllm_core::QWEN35_MOE_MODEL_FINGERPRINT.to_owned(),
        }
    }

    fn verify(&self) -> Result<Value, String> {
        let plan = self.plan()?;
        let tensor_count = self.source.gguf().tensors().len();
        Ok(json!({
            "kind": "verify-model",
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "tensor_count": tensor_count,
            "source_kind": "gguf",
            "weight_entries": plan.entries.len(),
            "total_destination_bytes": plan.total_destination_bytes,
            "plan_digest": plan.digest_hex(),
            "weight_encoding": "ocp-mxfp4-e2m1-block32-e8m0-mixed",
        }))
    }

    fn tokenize(&self, text: &str) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_tokenize(&tokenizer, text, false)
    }

    fn tokenize_with_pieces(&self, text: &str) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_tokenize(&tokenizer, text, true)
    }

    fn render(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let renderer = self.renderer()?;
        let text = renderer
            .render(messages, options)
            .map_err(|error| error.to_string())?;
        Ok(json!({"kind":"render","text":text}))
    }

    fn apply_template(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let renderer = self.renderer()?;
        utility_apply_template(&tokenizer, &renderer, messages, options)
    }

    fn apply_template_custom(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
        provider: &GenericTemplateProviderV1,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_apply_custom_template(&tokenizer, provider, messages, options, kwargs)
    }

    fn input_tokens(
        &self,
        text: Option<&str>,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let renderer = self.renderer().ok();
        utility_input_tokens(&tokenizer, renderer.as_ref(), text, messages, options)
    }

    fn input_tokens_custom(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
        provider: &GenericTemplateProviderV1,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_input_tokens_custom(&tokenizer, provider, messages, options, kwargs)
    }

    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let mut result = utility_detokenize(&tokenizer, ids, mode)?;
        result["kind"] = Value::from("decode");
        Ok(result)
    }

    fn detokenize(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_detokenize(&tokenizer, ids, mode)
    }

    fn embeddings(
        &self,
        texts: &[String],
        token_inputs: &[TokenIdsV1],
        device_index: u32,
        target: &str,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let inputs = prepare_embedding_inputs(
            &tokenizer,
            texts,
            token_inputs,
            QWEN_RUNTIME_MAX_CONTEXT_TOKENS,
        )?;
        let token_counts = inputs.iter().map(Vec::len).collect::<Vec<_>>();
        let token_count = u64::try_from(token_counts.iter().copied().max().unwrap_or(0))
            .map_err(|_| "embedding token count overflowed".to_owned())?;
        let plan = self.plan()?;
        let graph = self
            .graph(&plan, token_count, token_count)
            .map_err(|error| format!("embedding graph is unavailable: {error}"))?;
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(device_index, target.to_owned())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let execution = (|| -> Result<Vec<Vec<f32>>, String> {
            let resident = QwenResidentModel::new_gguf_moe(
                Arc::clone(&session),
                graph.clone(),
                plan.clone(),
                Arc::clone(&self.source),
                COMPLETION_TIMEOUT,
            )
            .map_err(|error| format!("embedding resident provisioning failed: {error}"))?;
            let mut vectors = Vec::with_capacity(inputs.len());
            for input in &inputs {
                let mut owner = resident
                    .new_request(graph.clone())
                    .map_err(|error| format!("embedding request provisioning failed: {error}"))?;
                let output = owner
                    .prefill_with_embeddings(input)
                    .map_err(|error| format!("embedding execution failed: {error}"))?;
                let audit = owner
                    .audit_snapshot()
                    .map_err(|error| format!("embedding dispatch audit failed: {error}"))?;
                if audit.target() != target || audit.fallback_used() || !audit.all_dispatches_hip()
                {
                    return Err("embedding dispatch audit is not exact HIP/no-fallback".to_owned());
                }
                let rows = output.final_hidden_states_bf16().ok_or_else(|| {
                    "embedding execution did not publish final-normalized hidden rows".to_owned()
                })?;
                let hidden = sllm_core::QWEN35_HIDDEN_SIZE;
                if rows.len()
                    != input
                        .len()
                        .checked_mul(hidden)
                        .ok_or_else(|| "embedding hidden shape overflowed".to_owned())?
                {
                    return Err(
                        "embedding hidden readback shape differed from the model contract"
                            .to_owned(),
                    );
                }
                vectors.push(
                    sllm_core::EmbeddingPoolV1::new()
                        .pool_bf16(rows, input.len(), hidden)
                        .map_err(|error| format!("embedding pooling failed: {error}"))?
                        .as_slice()
                        .to_vec(),
                );
            }
            Ok(vectors)
        })();
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|_| "HIP session cleanup failed".to_owned())?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        embedding_response(execution?, &token_counts)
    }

    fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
        let started = Instant::now();
        if request.prefill_chunk_tokens.is_some() || request.mtp_draft_width.is_some() {
            return Err(
                "--prefill-chunk-tokens and --mtp-draft-width are supported only for dense Qwen generation"
                    .to_owned(),
            );
        }
        if !request.image_paths.is_empty() {
            return Err("Qwen3.5 MoE production path is text-only".to_owned());
        }
        if request.fp8_manifest.is_some()
            || request.fp8_artifact.is_some()
            || request.fp8_provider.is_some()
        {
            return Err("Qwen3.5 MoE selects its mixed MXFP4 recipe internally".to_owned());
        }
        let tokenizer = self.tokenizer()?;
        let renderer = match &request.input {
            GenerationInput::Prompt(_) => None,
            GenerationInput::Messages { .. } => Some(self.renderer()?),
        };
        let stop_policy = qwen35_moe_generation_stop_policy();
        let service = GenerationServiceV1::new(&tokenizer, renderer.as_ref(), &stop_policy)
            .map_err(|error| error.to_string())?;
        let (input_kind, service_input) = match &request.input {
            GenerationInput::Prompt(prompt) => {
                ("prompt", ServiceGenerationInputV1::Prompt(prompt.clone()))
            }
            GenerationInput::Messages { messages, options } => (
                "messages",
                ServiceGenerationInputV1::Messages {
                    messages: messages.clone(),
                    options: *options,
                },
            ),
        };
        let input = service
            .prepare_input(&service_input)
            .map_err(|error| error.to_string())?;
        let input_len = u64::try_from(input.len()).map_err(|_| "input is too long")?;
        let state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or("state capacity overflow")?;
        let plan = self.plan()?;
        let graph = self.graph(&plan, input_len, state_capacity)?;
        let backend = HipBackend::connect().map_err(|error| error.to_string())?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(request.device_index, request.target.clone())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
        let result = (|| -> Result<Value, String> {
            let resident = QwenResidentModel::new_gguf_moe(
                Arc::clone(&session),
                graph.clone(),
                plan.clone(),
                Arc::clone(&self.source),
                COMPLETION_TIMEOUT,
            )
            .map_err(|error| error.to_string())?;
            let mut owner = resident
                .new_request(graph)
                .map_err(|error| error.to_string())?;
            let config = GenerationConfigV1::new(
                request.max_new_tokens,
                request.sampling,
                request.stop_strings.clone(),
            )
            .map_err(|error| error.to_string())?;
            let cancellation = GenerationCancellationV1::new();
            let mut random =
                OsSamplingRandom::for_parameters_and_seed(request.sampling, request.seed)
                    .map_err(|error| error.to_string())?;
            let report = service
                .generate_tokens(&mut owner, &input, &config, &cancellation, &mut random)
                .map_err(|error| error.to_string())?;
            let audit = owner.audit_snapshot().map_err(|error| error.to_string())?;
            if audit.target() != request.target
                || audit.fallback_used()
                || !audit.all_dispatches_hip()
            {
                return Err("MoE dispatch audit is not exact HIP/no-fallback".to_owned());
            }
            Ok(json!({
                "kind":"generate",
                "input_kind":input_kind,
                "input_token_ids":report.input_token_ids(),
                "generated_token_ids":report.generated_token_ids(),
                "visible_token_ids":report.visible_token_ids(),
                "decode_input_token_ids":report.decode_input_token_ids(),
                "output_text":report.output_text(),
                "finish_reason":report.finish_reason().as_str(),
                "usage":{
                    "prompt_tokens":report.usage().prompt_tokens(),
                    "completion_tokens":report.usage().completion_tokens(),
                    "total_tokens":report.usage().total_tokens(),
                },
                "execution":{
                    "selected_backend":audit.selected_backend(),
                    "target":audit.target(),
                    "device_index":request.device_index,
                    "model_fingerprint":sllm_core::QWEN35_MOE_MODEL_FINGERPRINT,
                    "plan_digest":plan.digest_hex(),
                    "fallback_used":audit.fallback_used(),
                    "submission_count":audit.submission_count(),
                    "kernel_dispatch_count":audit.kernel_dispatch_count(),
                    "all_dispatches_hip":audit.all_dispatches_hip(),
                    "weight_encoding":"ocp-mxfp4-e2m1-block32-e8m0-mixed",
                    "fp8_provider":null,
                },
            }))
        })();
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|error| error.to_string())?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        let mut result = result?;
        result.as_object_mut().unwrap().insert(
            "timing_ns".to_owned(),
            Value::from(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)),
        );
        Ok(result)
    }

    fn benchmark(&self, _: &BenchmarkRequest, _: BenchmarkTiming) -> Result<Value, String> {
        Err("Qwen3.5 MoE benchmark uses the Phase 19 evidence runner".to_owned())
    }
}

impl ProductionBackend {
    fn open(lock: ModelLock, request: &Request) -> Result<Self, String> {
        let source = match (&request.gguf, &request.derived_lock) {
            (Some(gguf_path), Some(derived_path)) => {
                let derived = read_derived_gguf_lock(derived_path)
                    .map_err(|error| format!("derived GGUF lock is invalid: {error}"))?;
                let verified = verify_derived_gguf(derived, gguf_path)
                    .map_err(|error| format!("GGUF does not match its derived lock: {error}"))?;
                let (source, _) = build_verified_gguf_qwen_weight_load_plan(
                    &lock,
                    verified,
                    QwenComponentSelection::TEXT_ONLY,
                )
                .map_err(|error| format!("GGUF load plan is invalid: {error}"))?;
                QwenDenseSource::Gguf(Arc::new(source))
            }
            _ => unreachable!("public parser requires paired GGUF paths"),
        };
        Ok(Self {
            lock,
            lock_path: PathBuf::new(),
            source,
        })
    }

    fn cache(&self) -> Result<&Arc<VerifiedCache>, String> {
        match &self.source {
            QwenDenseSource::Cache(cache) => Ok(cache),
            QwenDenseSource::Gguf(_) => Err(
                "this operation requires the legacy development importer and is unavailable for GGUF"
                    .to_owned(),
            ),
        }
    }

    fn load_plan(
        &self,
        selection: QwenComponentSelection,
    ) -> Result<sllm_core::WeightLoadPlan, String> {
        match &self.source {
            QwenDenseSource::Cache(cache) => {
                build_verified_qwen_component_weight_load_plan(&self.lock, cache, selection)
            }
            QwenDenseSource::Gguf(source) => {
                source.build_qwen_weight_load_plan(&self.lock, selection)
            }
        }
        .map_err(|error| format!("verified tensors do not form the fixed model load plan: {error}"))
    }

    fn validate_cli_mtp_weight_plan(
        &self,
        plan: &sllm_core::WeightLoadPlan,
        state_capacity: u64,
    ) -> Result<(), String> {
        if let QwenDenseSource::Gguf(source) = &self.source {
            let mtp_prefix = &self.lock.model().architecture.mtp.tensor_prefix;
            if let Some(entry) = plan.entries.iter().find(|entry| {
                entry.tensor_name.starts_with(mtp_prefix)
                    && source.recipe_binding(&entry.tensor_name).is_some()
            }) {
                return Err(format!(
                    "MTP draft tensor {} is recipe-backed; the reviewed draft path requires BF16 MTP weights",
                    entry.tensor_name
                ));
            }
        }
        build_qwen35_mtp_graph(&self.lock, plan, state_capacity)
            .map(|_| ())
            .map_err(|error| {
                format!("MTP weight plan is incompatible with the fixed graph: {error}")
            })
    }

    fn tokenizer(&self) -> Result<TokenizerFrontendV1, String> {
        match &self.source {
            QwenDenseSource::Cache(cache) => {
                TokenizerFrontendV1::from_verified_cache(&self.lock, cache)
            }
            QwenDenseSource::Gguf(source) => {
                TokenizerFrontendV1::from_qwen35_gguf(&self.lock, source.gguf())
            }
        }
        .map_err(|error| format!("verified tokenizer could not be constructed: {error}"))
    }

    fn renderer(&self) -> Result<Qwen35ChatTemplateV1, String> {
        match &self.source {
            QwenDenseSource::Cache(cache) => {
                Qwen35ChatTemplateV1::from_verified_cache(&self.lock, cache)
            }
            QwenDenseSource::Gguf(source) => {
                Qwen35ChatTemplateV1::from_qwen35_gguf(&self.lock, source.gguf())
            }
        }
        .map_err(|_| "verified chat renderer could not be constructed".to_owned())
    }

    fn build_plain_graph(
        &self,
        plan: &sllm_core::WeightLoadPlan,
        token_count: u64,
        state_capacity: u64,
        target: &str,
        kv_cache_encoding: KvCacheEncoding,
        kv_cache_selection: Option<KvCacheSelection>,
    ) -> Result<sllm_core::QwenGraph, sllm_core::QwenGraphError> {
        match &self.source {
            QwenDenseSource::Gguf(source) if source.has_fp8_recipe() => {
                let provider = select_cli_gguf_fp8_provider(target)
                    .map_err(sllm_core::QwenGraphError::InvalidModel)?;
                build_qwen35_gguf_fp8_graph(
                    &self.lock,
                    plan,
                    source,
                    token_count,
                    state_capacity,
                    cli_fp8_dtype(provider),
                    kv_cache_encoding,
                )
            }
            _ => match kv_cache_selection {
                Some(selection) => build_qwen35_graph_with_kv_cache_selection(
                    &self.lock,
                    plan,
                    token_count,
                    state_capacity,
                    selection,
                ),
                None => build_qwen35_graph_with_kv_cache_encoding(
                    &self.lock,
                    plan,
                    token_count,
                    state_capacity,
                    kv_cache_encoding,
                ),
            },
        }
    }
}

impl ModelFrontendBackend for ProductionBackend {
    fn identity(&self) -> ModelIdentity {
        ModelIdentity {
            repo_id: self.lock.model().repo_id.clone(),
            resolved_revision: self.lock.model().resolved_revision.clone(),
            lock_fingerprint: self.lock.fingerprint().to_owned(),
        }
    }

    fn verify(&self) -> Result<Value, String> {
        let plan = self.load_plan(QwenComponentSelection::TEXT_ONLY)?;
        let loadable = plan
            .entries
            .iter()
            .filter(|entry| entry.classification != WeightClassification::KnownUnconsumed)
            .count();
        let (source_kind, verified_files, tensor_count) = match &self.source {
            QwenDenseSource::Cache(cache) => (
                "development-cache",
                cache.files.len(),
                cache.tensors().count(),
            ),
            QwenDenseSource::Gguf(source) => ("gguf", 1, source.gguf().tensors().len()),
        };
        Ok(json!({
            "kind": "verify-model",
            "source_kind": source_kind,
            "locked_files": self.lock.model().files.len(),
            "verified_files": verified_files,
            "tensor_count": tensor_count,
            "weight_entries": plan.entries.len(),
            "loadable_entries": loadable,
            "known_unconsumed_entries": plan.entries.len() - loadable,
            "total_destination_bytes": plan.total_destination_bytes,
            "plan_digest": plan.digest_hex(),
        }))
    }

    fn tokenize(&self, text: &str) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_tokenize(&tokenizer, text, false)
    }

    fn tokenize_with_pieces(&self, text: &str) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_tokenize(&tokenizer, text, true)
    }

    fn render(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let renderer = self.renderer()?;
        let text = renderer
            .render(messages, options)
            .map_err(|_| "chat messages could not be rendered".to_owned())?;
        Ok(json!({"kind": "render", "text": text}))
    }

    fn apply_template(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let renderer = self.renderer()?;
        utility_apply_template(&tokenizer, &renderer, messages, options)
    }

    fn apply_template_custom(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
        provider: &GenericTemplateProviderV1,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_apply_custom_template(&tokenizer, provider, messages, options, kwargs)
    }

    fn input_tokens(
        &self,
        text: Option<&str>,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let renderer = self.renderer().ok();
        utility_input_tokens(&tokenizer, renderer.as_ref(), text, messages, options)
    }

    fn input_tokens_custom(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
        provider: &GenericTemplateProviderV1,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_input_tokens_custom(&tokenizer, provider, messages, options, kwargs)
    }

    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let mut result = utility_detokenize(&tokenizer, ids, mode)?;
        result["kind"] = Value::from("decode");
        Ok(result)
    }

    fn detokenize(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        utility_detokenize(&tokenizer, ids, mode)
    }

    fn embeddings(
        &self,
        texts: &[String],
        token_inputs: &[TokenIdsV1],
        device_index: u32,
        target: &str,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let inputs = prepare_embedding_inputs(
            &tokenizer,
            texts,
            token_inputs,
            QWEN_RUNTIME_MAX_CONTEXT_TOKENS,
        )?;
        let token_counts = inputs.iter().map(Vec::len).collect::<Vec<_>>();
        let max_tokens = token_counts.iter().copied().max().unwrap_or(0);
        let token_count =
            u64::try_from(max_tokens).map_err(|_| "embedding token count overflowed".to_owned())?;
        let plan = self.load_plan(QwenComponentSelection::TEXT_ONLY)?;
        let graph = self
            .build_plain_graph(
                &plan,
                token_count,
                token_count,
                target,
                KvCacheEncoding::Fp16,
                None,
            )
            .map_err(|error| format!("embedding graph is unavailable: {error}"))?;
        let source = match &self.source {
            QwenDenseSource::Gguf(source) => Arc::clone(source),
            QwenDenseSource::Cache(_) => {
                return Err("embedding requires a verified GGUF source".to_owned());
            }
        };
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(device_index, target.to_owned())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let execution = (|| -> Result<Vec<Vec<f32>>, String> {
            let resident = QwenResidentModel::new_gguf(
                Arc::clone(&session),
                graph.clone(),
                plan.clone(),
                source,
                COMPLETION_TIMEOUT,
            )
            .map_err(|error| format!("embedding resident provisioning failed: {error}"))?;
            let mut vectors = Vec::with_capacity(inputs.len());
            for input in &inputs {
                let mut owner = resident
                    .new_request(graph.clone())
                    .map_err(|error| format!("embedding request provisioning failed: {error}"))?;
                let output = owner
                    .prefill_with_embeddings(input)
                    .map_err(|error| format!("embedding execution failed: {error}"))?;
                let audit = owner
                    .audit_snapshot()
                    .map_err(|error| format!("embedding dispatch audit failed: {error}"))?;
                if audit.target() != target || audit.fallback_used() || !audit.all_dispatches_hip()
                {
                    return Err("embedding dispatch audit is not exact HIP/no-fallback".to_owned());
                }
                let rows = output.final_hidden_states_bf16().ok_or_else(|| {
                    "embedding execution did not publish final-normalized hidden rows".to_owned()
                })?;
                let hidden = sllm_core::QWEN35_HIDDEN_SIZE;
                let expected = input
                    .len()
                    .checked_mul(hidden)
                    .ok_or_else(|| "embedding hidden shape overflowed".to_owned())?;
                if rows.len() != expected {
                    return Err(
                        "embedding hidden readback shape differed from the model contract"
                            .to_owned(),
                    );
                }
                let vector = sllm_core::EmbeddingPoolV1::new()
                    .pool_bf16(rows, input.len(), hidden)
                    .map_err(|error| format!("embedding pooling failed: {error}"))?;
                vectors.push(vector.as_slice().to_vec());
            }
            Ok(vectors)
        })();
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|_| "HIP session cleanup failed".to_owned())?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        embedding_response(execution?, &token_counts)
    }

    fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
        let started = Instant::now();
        let (effective_input, processed_images) =
            prepare_qwen_cli_images(&request.input, &request.image_paths)?;
        if matches!(self.source, QwenDenseSource::Gguf(_))
            && (request.fp8_manifest.is_some()
                || request.fp8_artifact.is_some()
                || request.fp8_provider.is_some())
        {
            return Err(
                "GGUF carries its own quantization recipe and cannot be combined with legacy sidecar flags"
                    .to_owned(),
            );
        }
        let tokenizer = self.tokenizer()?;
        let renderer = match &effective_input {
            GenerationInput::Prompt(_) => None,
            GenerationInput::Messages { messages, options } => {
                let _ = (messages, options);
                Some(self.renderer()?)
            }
        };
        let service = GenerationServiceV1::new(
            &tokenizer,
            renderer.as_ref(),
            self.lock.generation_stop_policy(),
        )
        .map_err(|error| format!("generation service could not be constructed: {error}"))?;
        let (input_kind, service_input) = match &effective_input {
            GenerationInput::Prompt(prompt) => {
                ("prompt", ServiceGenerationInputV1::Prompt(prompt.clone()))
            }
            GenerationInput::Messages { messages, options } => (
                "messages",
                ServiceGenerationInputV1::Messages {
                    messages: messages.clone(),
                    options: *options,
                },
            ),
        };
        let input = service
            .prepare_input(&service_input)
            .map_err(|error| format!("generation input preparation failed: {error}"))?;
        let input_len = u64::try_from(input.len())
            .map_err(|_| "generation input token count overflowed".to_owned())?;
        let logical_state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or_else(|| "generation state capacity overflowed".to_owned())?;
        let plan = self.load_plan(QwenComponentSelection::TEXT_ONLY)?;
        let head_dim = usize::try_from(self.lock.model().architecture.text_config.head_dim)
            .map_err(|_| "Qwen KV head dimension overflowed usize".to_owned())?;
        let kv_selection = resolve_cli_kv_cache_selection(
            request.kv_cache_encoding,
            &request.target,
            self.lock.fingerprint(),
            true,
            true,
            head_dim,
        )?;
        let kv_cache_encoding = kv_selection.resolved();
        let embedded_fp8 = matches!(
            &self.source,
            QwenDenseSource::Gguf(source) if source.has_fp8_recipe()
        );
        let embedded_fp8_provider = embedded_fp8
            .then(|| select_cli_gguf_fp8_provider(&request.target))
            .transpose()?;
        let nvfp4_requested = request.fp8_provider == Some(CliFp8Provider::Nvfp4PackedDequant);
        let nvfp4_sidecar = match (
            nvfp4_requested,
            &request.fp8_manifest,
            &request.fp8_artifact,
        ) {
            (true, Some(manifest), Some(artifact)) => Some(Arc::new(
                verify_nvfp4_sidecar(manifest, artifact, &self.lock_path, &self.lock)
                    .map_err(|error| format!("NVFP4 sidecar verification failed: {error}"))?,
            )),
            (true, _, _) => {
                return Err(
                    "NVFP4 generation requires manifest, artifact, and --nvfp4-provider packed-dequant"
                        .to_owned(),
                );
            }
            (false, _, _) => None,
        };
        let sidecar = match (
            nvfp4_requested,
            &request.fp8_manifest,
            &request.fp8_artifact,
        ) {
            (false, Some(manifest), Some(artifact)) => Some(Arc::new(
                verify_fp8_sidecar(manifest, artifact, &self.lock_path, &self.lock)
                    .map_err(|error| format!("FP8 sidecar verification failed: {error}"))?,
            )),
            (false, None, None) => None,
            (true, _, _) => None,
            _ => return Err("FP8 generation requires both manifest and artifact".to_owned()),
        };
        let has_sidecar = sidecar.is_some() || nvfp4_sidecar.is_some();
        if (kv_cache_encoding.is_kv_fp8_block16() || kv_cache_encoding.is_kv_mxfp8())
            && (has_sidecar || embedded_fp8)
        {
            return Err(
                "block-scaled KV FP8 is currently scoped to Qwen3.5-4B BF16 text weights"
                    .to_owned(),
            );
        }
        if !processed_images.is_empty()
            && (has_sidecar || kv_cache_encoding != KvCacheEncoding::Fp16)
        {
            return Err(
                "vision requests currently require BF16 text weights and FP16 KV cache".to_owned(),
            );
        }
        let fp8_provider =
            select_cli_fp8_provider(has_sidecar, request.fp8_provider, &request.target)?;
        let mtp_plan = resolve_cli_mtp_plan(
            request.mtp_draft_width,
            !processed_images.is_empty(),
            has_sidecar,
            embedded_fp8,
            &request.target,
            kv_cache_encoding,
            request.sampling,
            self.lock.fingerprint(),
        )?;
        let (state_capacity, mtp_state_slack_tokens) =
            cli_state_capacity_with_mtp_slack(logical_state_capacity, mtp_plan.effective_width)?;
        let mut mtp_weight_plan = if mtp_plan.enabled {
            let plan = self
                .load_plan(QwenComponentSelection::MTP_ONLY)
                .map_err(|error| {
                    if mtp_plan.selection == "forced" {
                        format!(
                            "forced MTP is unavailable: verified MTP weight plan could not be loaded: {error}"
                        )
                    } else {
                        error
                    }
                })?;
            self.validate_cli_mtp_weight_plan(&plan, state_capacity)
                .map_err(|error| {
                    if mtp_plan.selection == "forced" {
                        format!("forced MTP is unavailable: {error}")
                    } else {
                        error
                    }
                })?;
            Some(plan)
        } else {
            None
        };
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session_request =
            ExecutionSessionRequest::new(request.device_index, request.target.clone())
                .map_err(|_| "invalid exact HIP session request".to_owned())?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let placement_total_memory_bytes = session
            .total_memory_bytes()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "HIP backend omitted total device memory".to_owned())?;
        let placement_available_memory_bytes = session
            .available_memory_bytes()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "HIP backend omitted available device memory".to_owned())?;
        if request.prefill_chunk_tokens.is_some() && !processed_images.is_empty() {
            return Err(
                "--prefill-chunk-tokens is supported only for text-only Qwen generation".to_owned(),
            );
        }
        let chunk_candidates = if processed_images.is_empty() {
            cli_prefill_chunk_candidates(
                request.prefill_chunk_tokens,
                placement_total_memory_bytes,
                input_len,
            )?
        } else {
            vec![input_len]
        };
        let build_graph = |chunk_rows: u64| {
            let text_rows = if mtp_plan.enabled {
                let target_block_rows = mtp_plan
                    .effective_width
                    .map(|width| u64::from(width) + 1)
                    .unwrap_or(2);
                chunk_rows.max(target_block_rows)
            } else {
                chunk_rows
            };
            if !processed_images.is_empty() {
                build_qwen35_multimodal_graph(&self.lock, &plan, input_len, state_capacity)
            } else if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
                build_qwen35_nvfp4_graph(
                    &self.lock,
                    &plan,
                    nvfp4_sidecar,
                    chunk_rows,
                    state_capacity,
                )
            } else {
                match (&sidecar, fp8_provider) {
                    (Some(_), Some(CliFp8Provider::ConvertedBf16)) => {
                        build_qwen35_graph(&self.lock, &plan, text_rows, state_capacity)
                    }
                    (Some(sidecar), Some(CliFp8Provider::NativeFnuz)) => {
                        build_qwen35_fp8_fnuz_graph(
                            &self.lock,
                            &plan,
                            sidecar,
                            chunk_rows,
                            state_capacity,
                        )
                    }
                    (Some(sidecar), Some(_)) => build_qwen35_fp8_graph(
                        &self.lock,
                        &plan,
                        sidecar,
                        chunk_rows,
                        state_capacity,
                    ),
                    (None, None) => self.build_plain_graph(
                        &plan,
                        text_rows,
                        state_capacity,
                        &request.target,
                        kv_cache_encoding,
                        Some(kv_selection),
                    ),
                    _ => unreachable!("quantized provider selection validated sidecar state"),
                }
            }
        };
        let mut rejected = Vec::new();
        let mut selected = None;
        for chunk_rows in chunk_candidates {
            let graph = build_graph(chunk_rows).map_err(|error| {
                format!("generation graph does not satisfy the fixed Qwen contract: {error}")
            })?;
            let estimate = qwen_graph_memory_estimate(&graph, &plan, placement_total_memory_bytes)
                .map_err(|error| error.to_string())?;
            if estimate.required_bytes() <= placement_available_memory_bytes {
                selected = Some((graph, estimate));
                break;
            }
            rejected.push(format!("{}:{}", chunk_rows, estimate.required_bytes()));
        }
        let (graph, placement) = selected.ok_or_else(|| {
            format!(
                "no prefill chunk fits available device memory {}; candidates chunk:required [{}]",
                placement_available_memory_bytes,
                rejected.join(",")
            )
        })?;
        let request_graph = graph.clone();
        let prefill_chunk_capacity_tokens = graph.token_count();
        let plan_digest = plan.digest_hex();
        let model_fingerprint = self.lock.fingerprint().to_owned();

        let execution = (|| -> Result<Value, String> {
            let (mut owner, _resident) = if let Some(sidecar) = nvfp4_sidecar {
                let resident = QwenResidentModel::new_nvfp4(
                    Arc::clone(&session),
                    graph,
                    plan.clone(),
                    Arc::clone(self.cache()?),
                    Arc::clone(&sidecar),
                    COMPLETION_TIMEOUT,
                )
                .map_err(|error| format!("Qwen NVFP4 resident provisioning failed: {error}"))?;
                let owner = resident
                    .new_request(request_graph)
                    .map_err(|error| format!("Qwen NVFP4 request provisioning failed: {error}"))?;
                (owner, Some(resident))
            } else if let Some(sidecar) = sidecar {
                let resident = match fp8_provider {
                    Some(CliFp8Provider::ConvertedBf16) => {
                        QwenResidentModel::new_fp8_converted_bf16(
                            Arc::clone(&session),
                            graph,
                            plan.clone(),
                            Arc::clone(self.cache()?),
                            Arc::clone(&sidecar),
                            COMPLETION_TIMEOUT,
                        )
                    }
                    Some(CliFp8Provider::NativeFnuz) => QwenResidentModel::new_fp8_fnuz(
                        Arc::clone(&session),
                        graph,
                        plan.clone(),
                        Arc::clone(self.cache()?),
                        Arc::clone(&sidecar),
                        COMPLETION_TIMEOUT,
                    ),
                    Some(CliFp8Provider::Native) => QwenResidentModel::new_fp8(
                        Arc::clone(&session),
                        graph,
                        plan.clone(),
                        Arc::clone(self.cache()?),
                        Arc::clone(&sidecar),
                        COMPLETION_TIMEOUT,
                    ),
                    Some(CliFp8Provider::Nvfp4PackedDequant) | None => {
                        unreachable!("FP8 sidecar requires an FP8 provider")
                    }
                }
                .map_err(|error| format!("Qwen FP8 resident provisioning failed: {error}"))?;
                let owner = resident
                    .new_request(request_graph)
                    .map_err(|error| format!("Qwen FP8 request provisioning failed: {error}"))?;
                (owner, Some(resident))
            } else {
                match &self.source {
                    QwenDenseSource::Cache(cache) => {
                        let owner = QwenExecutionRequest::new(
                            Arc::clone(&session),
                            graph,
                            plan,
                            Arc::clone(cache),
                            COMPLETION_TIMEOUT,
                        )
                        .map_err(|error| format!("Qwen request provisioning failed: {error}"))?;
                        (owner, None)
                    }
                    QwenDenseSource::Gguf(source) => {
                        let resident = QwenResidentModel::new_gguf(
                            Arc::clone(&session),
                            graph,
                            plan.clone(),
                            Arc::clone(source),
                            COMPLETION_TIMEOUT,
                        )
                        .map_err(|error| {
                            format!("Qwen GGUF resident provisioning failed: {error}")
                        })?;
                        let owner = resident.new_request(request_graph).map_err(|error| {
                            format!("Qwen GGUF request provisioning failed: {error}")
                        })?;
                        (owner, Some(resident))
                    }
                }
            };
            let vision_bundle = if processed_images.is_empty() {
                None
            } else {
                let (manifest, resident) = match &self.source {
                    QwenDenseSource::Cache(cache) => {
                        let manifest = build_verified_qwen35_vision_manifest(&self.lock, cache)
                            .map_err(|error| {
                                format!("vision manifest validation failed: {error}")
                            })?;
                        let resident = QwenVisionResidentModel::new(
                            Arc::clone(&session),
                            Arc::clone(cache),
                            manifest.clone(),
                            COMPLETION_TIMEOUT,
                        )
                        .map_err(|error| format!("vision resident provisioning failed: {error}"))?;
                        (manifest, resident)
                    }
                    QwenDenseSource::Gguf(source) => {
                        let manifest =
                            build_verified_gguf_qwen35_vision_manifest(&self.lock, source)
                                .map_err(|error| {
                                    format!("GGUF vision manifest validation failed: {error}")
                                })?;
                        let resident = QwenVisionResidentModel::new_gguf(
                            Arc::clone(&session),
                            Arc::clone(source),
                            manifest.clone(),
                            COMPLETION_TIMEOUT,
                        )
                        .map_err(|error| {
                            format!("GGUF vision resident provisioning failed: {error}")
                        })?;
                        (manifest, resident)
                    }
                };
                let images = processed_images
                    .iter()
                    .map(|image| {
                        resident
                            .execute(&QwenVisionExecutionInput {
                                grid_thw: image.grid_thw,
                                patch_width: image.patch_width,
                                patches: image.patches.clone(),
                            })
                            .map(|output| QwenMultimodalImageEmbedding {
                                grid_thw: image.grid_thw,
                                embeddings_bf16: output.embeddings_bf16().to_vec(),
                            })
                            .map_err(|error| format!("vision encode failed: {error}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let prompt = match &self.source {
                    QwenDenseSource::Cache(cache) => assemble_qwen35_multimodal_prompt(
                        cache.as_ref(),
                        &input,
                        manifest.image_pad_token,
                        &images,
                    ),
                    QwenDenseSource::Gguf(source) => assemble_gguf_qwen35_multimodal_prompt(
                        source.as_ref(),
                        &input,
                        manifest.image_pad_token,
                        &images,
                    ),
                }
                .map_err(|error| format!("multimodal prompt assembly failed: {error}"))?;
                Some((resident, prompt))
            };
            let config = GenerationConfigV1::new(
                request.max_new_tokens,
                request.sampling,
                request.stop_strings.clone(),
            )
            .map_err(|error| format!("generation configuration is invalid: {error}"))?;
            let cancellation = GenerationCancellationV1::new();
            let mut random =
                OsSamplingRandom::for_parameters_and_seed(request.sampling, request.seed)
                    .map_err(|error| format!("sampling random source failed: {error}"))?;
            let mut mtp_proposal_blocks = None;
            let mut mtp_proposed_draft_tokens = None;
            let mut mtp_accepted_draft_tokens = None;
            let (report, audit, prefill_chunk_count) =
                if let Some((_, prompt)) = vision_bundle.as_ref() {
                    let mut executor = CliQwenMultimodalExecutor {
                        inner: &mut owner,
                        prompt,
                        prefilled: false,
                    };
                    let report = service.generate_tokens(
                        &mut executor,
                        &input,
                        &config,
                        &cancellation,
                        &mut random,
                    );
                    let audit = owner.audit_snapshot();
                    let prefill_chunk_count = owner.prefill_chunk_count();
                    (report, audit, prefill_chunk_count)
                } else if mtp_plan.enabled {
                    let mtp_weight_plan = mtp_weight_plan.take().ok_or_else(|| {
                        "MTP selection omitted its verified weight plan".to_owned()
                    })?;
                    let mtp_graph =
                        build_qwen35_mtp_graph(&self.lock, &mtp_weight_plan, state_capacity)
                            .map_err(|error| format!("MTP request graph failed: {error}"))?;
                    let mtp_resident = match &self.source {
                        QwenDenseSource::Cache(cache) => QwenResidentModel::new(
                            Arc::clone(&session),
                            mtp_graph.clone(),
                            mtp_weight_plan,
                            Arc::clone(cache),
                            COMPLETION_TIMEOUT,
                        ),
                        QwenDenseSource::Gguf(source) => QwenResidentModel::new_gguf(
                            Arc::clone(&session),
                            mtp_graph.clone(),
                            mtp_weight_plan,
                            Arc::clone(source),
                            COMPLETION_TIMEOUT,
                        ),
                    }
                    .map_err(|error| format!("MTP resident provisioning failed: {error}"))?;
                    let mtp_owner = mtp_resident
                        .new_request(mtp_graph)
                        .map_err(|error| format!("MTP request provisioning failed: {error}"))?;
                    let draft_width = usize::from(mtp_plan.effective_width.ok_or_else(|| {
                        "MTP selection omitted an effective draft width".to_owned()
                    })?);
                    let mut executor = SpeculativeGenerationAdapterV1::new(
                        QwenMtpGenerationExecutorV1::new_with_draft_width(
                            owner,
                            mtp_owner,
                            draft_width,
                        )
                        .map_err(|error| error.to_string())?,
                    );
                    let report = service.generate_tokens(
                        &mut executor,
                        &input,
                        &config,
                        &cancellation,
                        &mut random,
                    );
                    let audit = executor.inner().target().audit_snapshot();
                    let prefill_chunk_count = executor.inner().target().prefill_chunk_count();
                    mtp_proposal_blocks = Some(executor.inner().proposal_blocks());
                    mtp_proposed_draft_tokens = Some(executor.inner().proposed_draft_tokens());
                    mtp_accepted_draft_tokens = Some(executor.inner().accepted_draft_tokens());
                    drop(executor);
                    drop(mtp_resident);
                    (report, audit, prefill_chunk_count)
                } else {
                    let report = service.generate_tokens(
                        &mut owner,
                        &input,
                        &config,
                        &cancellation,
                        &mut random,
                    );
                    let audit = owner.audit_snapshot();
                    let prefill_chunk_count = owner.prefill_chunk_count();
                    (report, audit, prefill_chunk_count)
                };
            let report = report.map_err(|error| format!("generation service failed: {error}"))?;
            let audit = audit.map_err(|_| "Qwen dispatch audit was empty or invalid".to_owned())?;
            if audit.target() != request.target {
                return Err(
                    "Qwen dispatch audit target differs from the requested target".to_owned(),
                );
            }
            let mtp_rejected_draft_tokens =
                match (mtp_proposed_draft_tokens, mtp_accepted_draft_tokens) {
                    (Some(proposed), Some(accepted)) => Some(proposed.saturating_sub(accepted)),
                    _ => None,
                };
            let mut execution_report = json!({
                "selected_backend": audit.selected_backend(),
                "target": audit.target(),
                "device_index": request.device_index,
                "model_fingerprint": model_fingerprint,
                "plan_digest": plan_digest,
                "prefill_tokens": input.len(),
                "logical_state_capacity_tokens": logical_state_capacity,
                "allocated_state_capacity_tokens": state_capacity,
                "mtp_state_slack_tokens": mtp_state_slack_tokens,
                "prefill_chunk_requested_tokens": request.prefill_chunk_tokens,
                "prefill_chunk_selection": if request.prefill_chunk_tokens.is_some() {
                    "explicit"
                } else {
                    "auto"
                },
                "prefill_chunk_capacity_tokens": prefill_chunk_capacity_tokens,
                "prefill_chunk_count": prefill_chunk_count,
                "mtp_selection": mtp_plan.selection,
                "mtp_draft_width_requested": mtp_plan.requested_width,
                "mtp_draft_width_effective": mtp_plan.effective_width,
                "mtp_target_block_rows": mtp_plan
                    .effective_width
                    .map(|width| u64::from(width) + 1),
                "mtp_proposal_blocks": mtp_proposal_blocks,
                "mtp_proposed_draft_tokens": mtp_proposed_draft_tokens,
                "mtp_accepted_draft_tokens": mtp_accepted_draft_tokens,
                "mtp_rejected_draft_tokens": mtp_rejected_draft_tokens,
                "placement_total_memory_bytes": placement_total_memory_bytes,
                "placement_available_memory_bytes": placement_available_memory_bytes,
                "placement_required_bytes": placement.required_bytes(),
                "placement_model_resident_bytes": placement.model_resident_bytes(),
                "placement_request_state_bytes": placement.request_state_bytes(),
                "placement_safety_reserve_bytes": placement.safety_reserve_bytes(),
                "workspace_separate_allocation_bytes": placement.workspace_baseline_bytes(),
                "workspace_arena_bytes": placement.workspace_arena_bytes(),
                "decode_steps": report.decode_steps(),
                "fallback_used": audit.fallback_used(),
                "submission_count": audit.submission_count(),
                "kernel_dispatch_count": audit.kernel_dispatch_count(),
                "segment_count": audit.segment_count(),
                "boundary_count": audit.boundary_count(),
                "all_dispatches_hip": audit.all_dispatches_hip(),
                "weight_encoding": cli_fp8_weight_encoding(embedded_fp8_provider.or(fp8_provider)),
                "kv_cache_encoding": kv_cache_encoding.canonical_name(),
                "kv_cache_selection": kv_selection_report(kv_selection),
                "fp8_provider": embedded_fp8_provider
                    .map(cli_gguf_fp8_provider_label)
                    .or_else(|| fp8_provider.map(CliFp8Provider::label)),
                "image_count": processed_images.len(),
            });
            let execution_object = execution_report
                .as_object_mut()
                .ok_or_else(|| "execution report was not an object".to_owned())?;
            execution_object.insert(
                "mtp_weight_encoding".to_owned(),
                if mtp_plan.enabled {
                    Value::from("bf16")
                } else {
                    Value::Null
                },
            );
            execution_object.insert(
                "mtp_kv_cache_encoding".to_owned(),
                if mtp_plan.enabled {
                    Value::from("fp16")
                } else {
                    Value::Null
                },
            );
            Ok(json!({
                "kind": "generate",
                "input_kind": input_kind,
                "input_token_ids": report.input_token_ids(),
                "generated_token_ids": report.generated_token_ids(),
                "visible_token_ids": report.visible_token_ids(),
                "decode_input_token_ids": report.decode_input_token_ids(),
                "output_text": report.output_text(),
                "finish_reason": report.finish_reason().as_str(),
                "stop_reason": {
                    "version": 1,
                    "reason_version": 1,
                    "kind": report.finish_reason().as_str(),
                    "token_id": report.stop_token_id(),
                    "matched_string": report.matched_stop(),
                },
                "usage": {
                    "prompt_tokens": report.usage().prompt_tokens(),
                    "completion_tokens": report.usage().completion_tokens(),
                    "total_tokens": report.usage().total_tokens(),
                },
                "sampling": {
                    "temperature": request.sampling.temperature(),
                    "top_p": request.sampling.top_p(),
                    "presence_penalty": request.sampling.presence_penalty(),
                    "frequency_penalty": request.sampling.frequency_penalty(),
                },
                "execution": execution_report,
            }))
        })();
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|_| "HIP session cleanup failed".to_owned())?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        let mut result = execution?;
        let object = result
            .as_object_mut()
            .ok_or_else(|| "generation result was not an object".to_owned())?;
        object.insert(
            "timing_ns".to_owned(),
            Value::from(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)),
        );
        object.insert(
            "cleanup".to_owned(),
            json!({"retryable_cleanup": 0, "durable_quarantine": 0}),
        );
        Ok(result)
    }

    fn benchmark(
        &self,
        request: &BenchmarkRequest,
        timing: BenchmarkTiming,
    ) -> Result<Value, String> {
        if matches!(self.source, QwenDenseSource::Gguf(_))
            && (request.fp8_manifest.is_some()
                || request.fp8_artifact.is_some()
                || request.fp8_provider.is_some())
        {
            return Err(
                "GGUF carries its own quantization recipe and cannot be combined with legacy sidecar flags"
                    .to_owned(),
            );
        }
        validate_benchmark_protocol(request.warmups, request.measured)?;
        if !request.greedy {
            return Err("benchmark requires explicit --greedy mode".to_owned());
        }
        if request.ignore_eos && request.lane != BenchmarkLane::Direct {
            return Err("--ignore-eos is supported only for benchmark direct lane".to_owned());
        }
        if request.prefill_chunk_tokens.is_some() && request.lane != BenchmarkLane::Direct {
            return Err(
                "--prefill-chunk-tokens is supported only for benchmark direct lane".to_owned(),
            );
        }
        let completion_timeout = benchmark_completion_timeout(request.completion_timeout_seconds)?;
        let expected_model_size = match self.lock.model().repo_id.as_str() {
            "Qwen/Qwen3.5-2B" => "2B",
            "Qwen/Qwen3.5-4B" => "4B",
            "Qwen/Qwen3.5-9B" => "9B",
            _ => return Err("benchmark model is outside the fixed Qwen size contract".to_owned()),
        };
        if request.model_size != expected_model_size {
            return Err("benchmark model-size identity differs from the locked model".to_owned());
        }
        if request.row_id.is_empty() || request.case_id.is_empty() {
            return Err("benchmark row and case identities must not be empty".to_owned());
        }

        let tokenizer = matches!(&request.input, BenchmarkInput::Messages { .. })
            .then(|| self.tokenizer())
            .transpose()?;
        let renderer = matches!(&request.input, BenchmarkInput::Messages { .. })
            .then(|| self.renderer())
            .transpose()?;
        let seed_input = match (&request.input, &renderer, &tokenizer) {
            (BenchmarkInput::TokenIds(ids), None, None) => ids.clone(),
            (BenchmarkInput::Messages { messages, options }, Some(renderer), Some(tokenizer)) => {
                let rendered = renderer
                    .render(messages, *options)
                    .map_err(|_| "chat messages could not be rendered".to_owned())?;
                tokenizer
                    .encode(&rendered)
                    .map_err(|_| "benchmark input could not be tokenized")?
            }
            _ => return Err("benchmark lane and input shape do not match".to_owned()),
        };
        if seed_input.is_empty() {
            return Err("benchmark input token IDs must not be empty".to_owned());
        }
        let input_len = u64::try_from(seed_input.len())
            .map_err(|_| "benchmark input token count overflowed".to_owned())?;
        let state_capacity =
            benchmark_state_capacity(input_len, request.max_new_tokens, request.context_length)?;
        let stop_policy =
            benchmark_stop_policy(self.lock.generation_stop_policy(), request.ignore_eos);
        let model_load_start_ns = timing.model_load_start_ns();
        let plan = self.load_plan(QwenComponentSelection::TEXT_ONLY)?;
        let head_dim = usize::try_from(self.lock.model().architecture.text_config.head_dim)
            .map_err(|_| "Qwen KV head dimension overflowed usize".to_owned())?;
        let kv_selection = resolve_cli_kv_cache_selection(
            request.kv_cache_encoding,
            &request.target,
            self.lock.fingerprint(),
            true,
            true,
            head_dim,
        )?;
        let kv_cache_encoding = kv_selection.resolved();
        let embedded_fp8 = matches!(
            &self.source,
            QwenDenseSource::Gguf(source) if source.has_fp8_recipe()
        );
        let embedded_fp8_provider = embedded_fp8
            .then(|| select_cli_gguf_fp8_provider(&request.target))
            .transpose()?;
        let nvfp4_requested = request.fp8_provider == Some(CliFp8Provider::Nvfp4PackedDequant);
        let nvfp4_sidecar = match (
            nvfp4_requested,
            &request.fp8_manifest,
            &request.fp8_artifact,
        ) {
            (true, Some(manifest), Some(artifact)) => Some(Arc::new(
                verify_nvfp4_sidecar(manifest, artifact, &self.lock_path, &self.lock)
                    .map_err(|error| format!("NVFP4 sidecar verification failed: {error}"))?,
            )),
            (true, _, _) => {
                return Err(
                    "NVFP4 benchmark requires manifest, artifact, and --nvfp4-provider packed-dequant"
                        .to_owned(),
                );
            }
            (false, _, _) => None,
        };
        let sidecar = match (
            nvfp4_requested,
            &request.fp8_manifest,
            &request.fp8_artifact,
        ) {
            (false, Some(manifest), Some(artifact)) => Some(Arc::new(
                verify_fp8_sidecar(manifest, artifact, &self.lock_path, &self.lock)
                    .map_err(|error| format!("FP8 sidecar verification failed: {error}"))?,
            )),
            (false, None, None) => None,
            (true, _, _) => None,
            _ => return Err("FP8 benchmark requires both manifest and artifact".to_owned()),
        };
        let has_sidecar = sidecar.is_some() || nvfp4_sidecar.is_some();
        if (kv_cache_encoding.is_kv_fp8_block16() || kv_cache_encoding.is_kv_mxfp8())
            && (has_sidecar || embedded_fp8)
        {
            return Err(
                "block-scaled KV FP8 is currently scoped to Qwen3.5-4B BF16 text weights"
                    .to_owned(),
            );
        }
        let fp8_provider =
            select_cli_fp8_provider(has_sidecar, request.fp8_provider, &request.target)?;
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session_request =
            ExecutionSessionRequest::new(request.device_index, request.target.clone())
                .map_err(|_| "invalid exact HIP session request".to_owned())?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let placement_total_memory_bytes = session
            .total_memory_bytes()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "HIP backend omitted total device memory".to_owned())?;
        let placement_available_memory_bytes = session
            .available_memory_bytes()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "HIP backend omitted available device memory".to_owned())?;
        let chunk_candidates = cli_prefill_chunk_candidates(
            request.prefill_chunk_tokens,
            placement_total_memory_bytes,
            input_len,
        )?;
        let placement_candidate_tokens = chunk_candidates.clone();
        let build_graph = |graph_token_count: u64| {
            if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
                build_qwen35_nvfp4_graph(
                    &self.lock,
                    &plan,
                    nvfp4_sidecar,
                    graph_token_count,
                    state_capacity,
                )
            } else {
                match (&sidecar, fp8_provider) {
                    (Some(_), Some(CliFp8Provider::ConvertedBf16)) => {
                        build_qwen35_graph(&self.lock, &plan, graph_token_count, state_capacity)
                    }
                    (Some(sidecar), Some(CliFp8Provider::NativeFnuz)) => {
                        build_qwen35_fp8_fnuz_graph(
                            &self.lock,
                            &plan,
                            sidecar,
                            graph_token_count,
                            state_capacity,
                        )
                    }
                    (Some(sidecar), Some(_)) => build_qwen35_fp8_graph(
                        &self.lock,
                        &plan,
                        sidecar,
                        graph_token_count,
                        state_capacity,
                    ),
                    (None, None) => self.build_plain_graph(
                        &plan,
                        graph_token_count,
                        state_capacity,
                        &request.target,
                        kv_cache_encoding,
                        Some(kv_selection),
                    ),
                    _ => unreachable!("quantized provider selection validated sidecar state"),
                }
            }
        };
        let mut rejected = Vec::new();
        let mut selected = None;
        for graph_token_count in chunk_candidates {
            let graph = build_graph(graph_token_count).map_err(|error| {
                format!("benchmark graph does not satisfy the fixed Qwen contract: {error}")
            })?;
            let estimate = qwen_graph_memory_estimate(&graph, &plan, placement_total_memory_bytes)
                .map_err(|error| error.to_string())?;
            if estimate.required_bytes() <= placement_available_memory_bytes {
                selected = Some((graph_token_count, graph, estimate));
                break;
            }
            rejected.push(format!(
                "{}:{}",
                graph_token_count,
                estimate.required_bytes()
            ));
        }
        let (graph_token_count, first_graph, placement) = selected.ok_or_else(|| {
            format!(
                "no benchmark prefill chunk fits available device memory {}; candidates chunk:required [{}]",
                placement_available_memory_bytes,
                rejected.join(",")
            )
        })?;
        let placement_diagnostic = json!({
            "selection": if request.prefill_chunk_tokens.is_some() {
                "explicit"
            } else {
                "automatic"
            },
            "candidate_tokens": placement_candidate_tokens,
            "rejected_chunk_required_bytes": rejected,
            "selected_chunk_tokens": graph_token_count,
            "total_memory_bytes": placement_total_memory_bytes,
            "available_memory_bytes": placement_available_memory_bytes,
            "required_bytes": placement.required_bytes(),
            "model_resident_bytes": placement.model_resident_bytes(),
            "workspace_baseline_bytes": placement.workspace_baseline_bytes(),
            "workspace_arena_bytes": placement.workspace_arena_bytes(),
            "request_state_bytes": placement.request_state_bytes(),
            "safety_reserve_bytes": placement.safety_reserve_bytes(),
        });

        let execution = (|| -> Result<Value, String> {
            let resident = if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
                QwenResidentModel::new_nvfp4(
                    Arc::clone(&session),
                    first_graph,
                    plan.clone(),
                    Arc::clone(self.cache()?),
                    Arc::clone(nvfp4_sidecar),
                    completion_timeout,
                )
            } else {
                match (&sidecar, fp8_provider) {
                    (Some(sidecar), Some(CliFp8Provider::ConvertedBf16)) => {
                        QwenResidentModel::new_fp8_converted_bf16(
                            Arc::clone(&session),
                            first_graph,
                            plan.clone(),
                            Arc::clone(self.cache()?),
                            Arc::clone(sidecar),
                            completion_timeout,
                        )
                    }
                    (Some(sidecar), Some(CliFp8Provider::NativeFnuz)) => {
                        QwenResidentModel::new_fp8_fnuz(
                            Arc::clone(&session),
                            first_graph,
                            plan.clone(),
                            Arc::clone(self.cache()?),
                            Arc::clone(sidecar),
                            completion_timeout,
                        )
                    }
                    (Some(sidecar), Some(_)) => QwenResidentModel::new_fp8(
                        Arc::clone(&session),
                        first_graph,
                        plan.clone(),
                        Arc::clone(self.cache()?),
                        Arc::clone(sidecar),
                        completion_timeout,
                    ),
                    (None, None) => match &self.source {
                        QwenDenseSource::Cache(cache) => QwenResidentModel::new(
                            Arc::clone(&session),
                            first_graph,
                            plan.clone(),
                            Arc::clone(cache),
                            completion_timeout,
                        ),
                        QwenDenseSource::Gguf(source) => QwenResidentModel::new_gguf(
                            Arc::clone(&session),
                            first_graph,
                            plan.clone(),
                            Arc::clone(source),
                            completion_timeout,
                        ),
                    },
                    _ => unreachable!("quantized provider selection validated sidecar state"),
                }
            }
            .map_err(|error| format!("Qwen resident model provisioning failed: {error}"))?;
            let model_ready_ns = timing.now_ns();
            let model_ready_snapshot = session.memory_snapshot();
            let model_ready_memory = allocation_snapshot_value(model_ready_snapshot);
            let model_resident_high_water_bytes =
                validate_model_ready_snapshot(&model_ready_memory)?;
            let ready_model_current_bytes = model_ready_snapshot.model_resident().current_bytes();

            let run_sample = |sample_index: u32| -> Result<Value, String> {
                let request_start_ns = timing.now_ns();
                let input = match (&request.input, &renderer, &tokenizer) {
                    (BenchmarkInput::TokenIds(ids), None, None) => ids.clone(),
                    (
                        BenchmarkInput::Messages { messages, options },
                        Some(renderer),
                        Some(tokenizer),
                    ) => {
                        let rendered = renderer
                            .render(messages, *options)
                            .map_err(|_| "chat messages could not be rendered".to_owned())?;
                        tokenizer
                            .encode(&rendered)
                            .map_err(|_| "benchmark input could not be tokenized".to_owned())?
                    }
                    _ => return Err("benchmark lane and input shape do not match".to_owned()),
                };
                validate_fixed_input_token_ids(seed_input.as_slice(), input.as_slice())?;
                let graph = if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
                    build_qwen35_nvfp4_graph(
                        &self.lock,
                        &plan,
                        nvfp4_sidecar,
                        graph_token_count,
                        state_capacity,
                    )
                } else {
                    match (&sidecar, fp8_provider) {
                        (Some(_), Some(CliFp8Provider::ConvertedBf16)) => {
                            build_qwen35_graph(&self.lock, &plan, graph_token_count, state_capacity)
                        }
                        (Some(sidecar), Some(CliFp8Provider::NativeFnuz)) => {
                            build_qwen35_fp8_fnuz_graph(
                                &self.lock,
                                &plan,
                                sidecar,
                                graph_token_count,
                                state_capacity,
                            )
                        }
                        (Some(sidecar), Some(_)) => build_qwen35_fp8_graph(
                            &self.lock,
                            &plan,
                            sidecar,
                            graph_token_count,
                            state_capacity,
                        ),
                        (None, None) => self.build_plain_graph(
                            &plan,
                            graph_token_count,
                            state_capacity,
                            &request.target,
                            kv_cache_encoding,
                            Some(kv_selection),
                        ),
                        _ => unreachable!("quantized provider selection validated sidecar state"),
                    }
                }
                .map_err(|error| {
                    format!("benchmark request graph does not satisfy the Qwen contract: {error}")
                })?;
                let mut owner = match resident.new_request(graph) {
                    Ok(owner) => owner,
                    Err(error) => {
                        let cleanup_memory = allocation_snapshot_value(session.memory_snapshot());
                        validate_request_cleanup_snapshot(
                            &cleanup_memory,
                            ready_model_current_bytes,
                        )?;
                        return Err(format!(
                            "Qwen benchmark request provisioning failed: {error}"
                        ));
                    }
                };
                let request_memory = allocation_snapshot_value(session.memory_snapshot());
                let mut timeline = BenchmarkTimeline::new(request_start_ns);
                let outcome = match validate_snapshot_accounting(&request_memory, "timed request") {
                    Ok(()) => run_greedy_generation_timed(
                        &mut owner,
                        &stop_policy,
                        request.max_new_tokens,
                        input.as_slice(),
                        Some((&mut timeline, timing.request_clock())),
                    ),
                    Err(error) => Err(error),
                };
                let (audit, kv_memory) = if outcome.is_ok() {
                    let audit = owner.audit_snapshot().map_err(|_| {
                        "Qwen benchmark dispatch audit was empty or invalid".to_owned()
                    });
                    let memory = owner.memory_audit_snapshot().map_err(|error| {
                        format!("Qwen benchmark KV memory audit failed: {error}")
                    })?;
                    let layers = memory
                        .kv_layers()
                        .iter()
                        .map(|layer| {
                            let physical = layer.physical();
                            json!({
                                "layer": layer.layer(),
                                "logical_capacity_tokens": layer.logical_capacity_tokens(),
                                "observed_length_tokens": layer.observed_length_tokens(),
                                "memory_kind": match physical.memory_kind() {
                                    sllm_core::KvMemoryKind::VirtualContiguous => "virtual-contiguous",
                                    sllm_core::KvMemoryKind::ContiguousResident => "contiguous-resident",
                                },
                                "physical_page_bytes": physical.physical_page_bytes(),
                                "tokens_per_page": physical.tokens_per_page(),
                                "mapped_token_capacity": physical.mapped_token_capacity(),
                                "committed_bytes_per_plane": physical.committed_bytes_per_plane(),
                            })
                        })
                        .collect::<Vec<_>>();
                    (
                        Some(audit),
                        Some(json!({
                            "kv_layer_count": layers.len(),
                            "committed_kv_bytes": memory
                                .committed_kv_bytes()
                                .map_err(|error| format!("Qwen benchmark KV committed-byte audit failed: {error}"))?,
                            "layers": layers,
                        })),
                    )
                } else {
                    (None, None)
                };
                drop(owner);
                let cleanup_timestamp_ns = timing.now_ns();
                let cleanup_memory = allocation_snapshot_value(session.memory_snapshot());
                validate_request_cleanup_snapshot(&cleanup_memory, ready_model_current_bytes)?;
                let outcome = outcome?;
                let audit =
                    audit.ok_or_else(|| "timed dispatch audit was not collected".to_owned())??;
                if audit.target() != request.target
                    || audit.fallback_used()
                    || !audit.all_dispatches_hip()
                {
                    return Err(
                        "Qwen benchmark dispatch audit is not exact HIP/no-fallback".to_owned()
                    );
                }
                let report = outcome.report;
                let stop = report
                    .stop_reason()
                    .ok_or_else(|| "benchmark generation ended without a stop reason".to_owned())?;
                let stop_value = json!({
                    "version": stop.version(),
                    "reason_version": stop.reason_version(),
                    "kind": stop.reason_token(),
                    "token_id": stop.token_id(),
                });
                let audit_value = json!({
                    "selected_backend": audit.selected_backend(),
                    "target": audit.target(),
                    "device_index": request.device_index,
                    "model_fingerprint": self.lock.fingerprint(),
                    "plan_digest": plan.digest_hex(),
                    "fallback_used": audit.fallback_used(),
                    "submission_count": audit.submission_count(),
                    "kernel_dispatch_count": audit.kernel_dispatch_count(),
                    "segment_count": audit.segment_count(),
                    "boundary_count": audit.boundary_count(),
                    "all_dispatches_hip": audit.all_dispatches_hip(),
                });
                timeline.record(BenchmarkEvent::Cleanup, cleanup_timestamp_ns)?;
                let sample = timeline.finish(BenchmarkSampleInput {
                    input_token_ids: input.as_slice(),
                    generated_token_ids: report.generated_token_ids(),
                    visible_token_ids: report.visible_token_ids(),
                    decode_input_token_ids: report.decode_input_token_ids(),
                    stop: stop_value,
                    audit: audit_value,
                    memory: json!({
                        "request_start": request_memory,
                        "after_cleanup": cleanup_memory,
                        "kv": kv_memory,
                    }),
                    cleanup: json!({
                        "sample_index": sample_index,
                        "request_dropped": true,
                        "allocator_cleanup_validated": true,
                        "retryable_cleanup": 0,
                        "durable_quarantine": 0,
                    }),
                })?;
                Ok(sample)
            };

            let mut warmup_samples = Vec::with_capacity(request.warmups as usize);
            for index in 0..request.warmups {
                warmup_samples.push(run_sample(index)?);
            }
            let mut measured_samples = Vec::with_capacity(request.measured as usize);
            for index in 0..request.measured {
                measured_samples.push(run_sample(index)?);
            }
            let control_reference = warmup_samples.first().ok_or_else(|| {
                "benchmark correctness reference requires at least one warmup sample".to_owned()
            })?;
            let correctness_control = correctness_reference_from_warmup(control_reference)?;
            for sample in warmup_samples.iter().skip(1).chain(measured_samples.iter()) {
                compare_control_sample(&correctness_control, sample)?;
            }
            let all_samples = warmup_samples.iter().chain(measured_samples.iter());
            let mut submission_count = 0_u64;
            let mut kernel_dispatch_count = 0_u64;
            let mut segment_count = 0_u64;
            let mut boundary_count = 0_u64;
            for sample in all_samples {
                let audit = sample
                    .get("audit")
                    .and_then(Value::as_object)
                    .ok_or_else(|| "benchmark sample audit was not an object".to_owned())?;
                submission_count = submission_count
                    .checked_add(
                        audit
                            .get("submission_count")
                            .and_then(Value::as_u64)
                            .ok_or_else(|| {
                                "benchmark sample submission count was missing".to_owned()
                            })?,
                    )
                    .ok_or_else(|| "benchmark submission count overflowed".to_owned())?;
                kernel_dispatch_count = kernel_dispatch_count
                    .checked_add(
                        audit
                            .get("kernel_dispatch_count")
                            .and_then(Value::as_u64)
                            .ok_or_else(|| {
                                "benchmark sample dispatch count was missing".to_owned()
                            })?,
                    )
                    .ok_or_else(|| "benchmark dispatch count overflowed".to_owned())?;
                segment_count = segment_count
                    .checked_add(
                        audit
                            .get("segment_count")
                            .and_then(Value::as_u64)
                            .ok_or_else(|| {
                                "benchmark sample segment count was missing".to_owned()
                            })?,
                    )
                    .ok_or_else(|| "benchmark segment count overflowed".to_owned())?;
                boundary_count = boundary_count
                    .checked_add(
                        audit
                            .get("boundary_count")
                            .and_then(Value::as_u64)
                            .ok_or_else(|| {
                                "benchmark sample boundary count was missing".to_owned()
                            })?,
                    )
                    .ok_or_else(|| "benchmark boundary count overflowed".to_owned())?;
            }
            drop(resident);
            let final_snapshot = session.memory_snapshot();
            let final_memory = allocation_snapshot_value(final_snapshot);
            validate_resident_drop_snapshot(&final_memory)?;
            validate_peak_vram_snapshot(&final_memory, model_resident_high_water_bytes)?;
            let (lane, lane_definition, tokenizer_enabled, render_enabled) = match request.lane {
                BenchmarkLane::Direct => (
                    "direct",
                    "pretokenized direct engine: request start excludes render/tokenize",
                    false,
                    false,
                ),
                BenchmarkLane::RenderTokenize => (
                    "render-tokenize",
                    "CLI end-to-end: request start includes chat render and tokenizer encode",
                    true,
                    true,
                ),
            };
            let config = if request.lane == BenchmarkLane::Direct {
                json!({
                    "input_token_ids": seed_input.as_slice(),
                    "input_token_count": seed_input.len(),
                    "max_new_tokens": request.max_new_tokens,
                    "ignore_eos": request.ignore_eos,
                    "context_length": request.context_length,
                    "effective_context_length": state_capacity,
                    "prefill_chunk_tokens": request.prefill_chunk_tokens,
                    "effective_prefill_chunk_tokens": graph_token_count,
                    "prefill_chunk_selection": if request.prefill_chunk_tokens.is_some() {
                        "explicit"
                    } else {
                        "automatic"
                    },
                    "prefill_chunk_candidates": placement_diagnostic["candidate_tokens"].clone(),
                    "prefill_chunk_rejections": placement_diagnostic["rejected_chunk_required_bytes"].clone(),
                    "completion_timeout_seconds": request
                        .completion_timeout_seconds
                        .unwrap_or(DEFAULT_BENCHMARK_COMPLETION_TIMEOUT_SECONDS),
                    "greedy": request.greedy,
                    "warmups": request.warmups,
                    "measured": request.measured,
                    "lane": lane,
                    "tokenizer": tokenizer_enabled,
                    "render": render_enabled,
                    "kv_cache_encoding": kv_cache_encoding.canonical_name(),
                    "kv_cache_selection": kv_selection_report(kv_selection),
                    "stop_policy": {
                        "stop_token_ids": if request.ignore_eos {
                            Vec::<u32>::new()
                        } else {
                            vec![248046, 248044]
                        },
                        "ignore_eos": request.ignore_eos,
                        "visible_stop_tokens": false,
                    },
                })
            } else {
                json!({
                    "input_token_ids": seed_input.as_slice(),
                    "input_token_count": seed_input.len(),
                    "max_new_tokens": request.max_new_tokens,
                    "ignore_eos": request.ignore_eos,
                    "context_length": request.context_length,
                    "effective_context_length": state_capacity,
                    "prefill_chunk_tokens": request.prefill_chunk_tokens,
                    "effective_prefill_chunk_tokens": graph_token_count,
                    "prefill_chunk_selection": if request.prefill_chunk_tokens.is_some() {
                        "explicit"
                    } else {
                        "automatic"
                    },
                    "prefill_chunk_candidates": placement_diagnostic["candidate_tokens"].clone(),
                    "prefill_chunk_rejections": placement_diagnostic["rejected_chunk_required_bytes"].clone(),
                    "completion_timeout_seconds": request
                        .completion_timeout_seconds
                        .unwrap_or(DEFAULT_BENCHMARK_COMPLETION_TIMEOUT_SECONDS),
                    "greedy": request.greedy,
                    "warmups": request.warmups,
                    "measured": request.measured,
                    "tokenizer": tokenizer_enabled,
                    "render": render_enabled,
                    "kv_cache_encoding": kv_cache_encoding.canonical_name(),
                    "kv_cache_selection": kv_selection_report(kv_selection),
                    "stop_policy": {
                        "stop_token_ids": if request.ignore_eos {
                            Vec::<u32>::new()
                        } else {
                            vec![248046, 248044]
                        },
                        "ignore_eos": request.ignore_eos,
                        "visible_stop_tokens": false,
                    },
                })
            };
            Ok(json!({
                "benchmark_schema_version": request.lane.schema_version(),
                "state": "PASS",
                "lane": lane,
                "lane_definition": lane_definition,
                "row": {
                    "row_id": request.row_id,
                    "model_size": request.model_size,
                    "case_id": request.case_id,
                    "input_token_ids": seed_input.as_slice(),
                    "input_token_count": seed_input.len(),
                    "requested_output_tokens": request.max_new_tokens,
                },
                "identities": {
                    "engine": "sllm",
                    "backend": "hip",
                    "session_id": session.id().raw(),
                    "device_index": request.device_index,
                    "target": request.target,
                    "model": {
                        "model_size": request.model_size,
                        "repo_id": self.lock.model().repo_id,
                        "resolved_revision": self.lock.model().resolved_revision,
                        "lock_fingerprint": self.lock.fingerprint(),
                    },
                    "binding": {
                        "model_fingerprint": self.lock.fingerprint(),
                        "plan_digest": plan.digest_hex(),
                    },
                },
                "model_load": {
                    "event": "model_load",
                    "start_ns": model_load_start_ns,
                    "model_ready_ns": model_ready_ns,
                    "duration_ns": model_ready_ns.checked_sub(model_load_start_ns)
                        .ok_or_else(|| "model load duration underflowed".to_owned())?,
                    "load_count": 1,
                },
                "config": config,
                "memory": {
                    "placement_total_memory_bytes": placement_total_memory_bytes,
                    "placement_available_memory_bytes": placement_available_memory_bytes,
                    "placement_required_bytes": placement.required_bytes(),
                    "placement_model_resident_bytes": placement.model_resident_bytes(),
                    "placement_request_state_bytes": placement.request_state_bytes(),
                    "placement_safety_reserve_bytes": placement.safety_reserve_bytes(),
                    "workspace_separate_allocation_bytes": placement.workspace_baseline_bytes(),
                    "workspace_arena_bytes": placement.workspace_arena_bytes(),
                    "model_ready": model_ready_memory,
                    "after_model_drop": final_memory,
                    "model_resident_high_water_bytes": model_resident_high_water_bytes,
                    "resident_vram_bytes": model_resident_high_water_bytes,
                    "resident_vram_source": "model_resident_allocator_high_water",
                    "peak_vram_bytes": final_snapshot.high_water_bytes(),
                    "peak_source": "runtime_allocator",
                },
                "audit": {
                    "selected_backend": "hip",
                    "target": request.target,
                    "device_index": request.device_index,
                    "model_fingerprint": self.lock.fingerprint(),
                    "plan_digest": plan.digest_hex(),
                    "submission_count": submission_count,
                    "kernel_dispatch_count": kernel_dispatch_count,
                    "segment_count": segment_count,
                    "boundary_count": boundary_count,
                    "fallback_used": false,
                    "all_dispatches_hip": true,
                    "model_load_count": 1,
                    "weight_encoding": cli_fp8_weight_encoding(embedded_fp8_provider.or(fp8_provider)),
                    "fp8_provider": embedded_fp8_provider
                        .map(cli_gguf_fp8_provider_label)
                        .or_else(|| fp8_provider.map(CliFp8Provider::label)),
                    "request_model_load_count": 0,
                    "model_reused": true,
                    "sample_count": request.warmups + request.measured,
                    "correctness_control_request_count": 0,
                    "correctness_control_source": "first-warmup-sample",
                    "correctness_control_reference_sample_index": 0,
                    "total_request_count": request.warmups + request.measured,
                },
                "cleanup": {
                    "correctness_control_request_count": 0,
                    "correctness_control_source": "first-warmup-sample",
                    "correctness_control_reference_sample_index": 0,
                    "warmup_request_count": request.warmups,
                    "measured_request_count": request.measured,
                    "request_cleanup_count": request.warmups + request.measured,
                    "performance_sample_count": request.warmups + request.measured,
                    "all_requests_dropped": true,
                    "retryable_cleanup": 0,
                    "durable_quarantine": 0,
                },
                "correctness_control": correctness_control,
                "warmups": {"count": request.warmups, "samples": warmup_samples},
                "measured": {"count": request.measured, "samples": measured_samples},
            }))
        })();
        let cleanup_result = session.shutdown(SHUTDOWN_TIMEOUT);
        let mut result = match execution {
            Ok(result) => result,
            Err(execution_error) => {
                let execution_error = format!(
                    "Qwen benchmark placement diagnostics={placement_diagnostic}; {execution_error}"
                );
                return match cleanup_result {
                    Ok(_) => Err(execution_error),
                    Err(cleanup_error) => Err(format!(
                        "{execution_error}; HIP session cleanup failed: {cleanup_error}"
                    )),
                };
            }
        };
        let cleanup =
            cleanup_result.map_err(|error| format!("HIP session cleanup failed: {error}"))?;
        let object = result
            .as_object_mut()
            .ok_or_else(|| "benchmark result was not an object".to_owned())?;
        object.insert(
            "session_cleanup".to_owned(),
            json!({
                "retryable_cleanup": cleanup.retryable_cleanup,
                "durable_quarantine": cleanup.durable_quarantine,
            }),
        );
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        Ok(result)
    }
}

pub(crate) fn run(
    command: &str,
    arguments: impl Iterator<Item = String>,
) -> Result<String, String> {
    let mut request = parse(command, arguments)?;
    load_custom_template(&mut request)?;
    let benchmark_timing = (command == "benchmark").then(BenchmarkTiming::start);
    let backend = open_production_backend(&request)?;
    match benchmark_timing {
        Some(timing) => {
            execute_with_timing(command, request.operation, backend.as_ref(), Some(timing))
        }
        None => execute(command, request.operation, backend.as_ref()),
    }
}

fn load_custom_template(request: &mut Request) -> Result<(), String> {
    let custom_template = match &mut request.operation {
        Operation::ApplyTemplate {
            custom_template: Some(spec),
            ..
        }
        | Operation::InputTokens {
            custom_template: Some(spec),
            ..
        } => Some(spec),
        _ => None,
    };
    if let Some(spec) = custom_template {
        if spec.provider.is_none() {
            spec.provider = Some(crate::template_file::read_verified_template(
                &spec.path,
                &spec.digest,
            )?);
        }
    }
    Ok(())
}

fn execute(
    command: &str,
    operation: Operation,
    backend: &dyn ModelFrontendBackend,
) -> Result<String, String> {
    execute_with_timing(command, operation, backend, None)
}

fn execute_with_timing(
    command: &str,
    operation: Operation,
    backend: &dyn ModelFrontendBackend,
    benchmark_timing: Option<BenchmarkTiming>,
) -> Result<String, String> {
    let result = match operation {
        Operation::Verify => backend.verify()?,
        Operation::Tokenize { text, pieces } => {
            if pieces {
                backend.tokenize_with_pieces(&text)?
            } else {
                backend.tokenize(&text)?
            }
        }
        Operation::Render { messages, options } => backend.render(&messages, options)?,
        Operation::Decode { ids, mode } => {
            if command == "detokenize" {
                backend.detokenize(&ids, mode)?
            } else {
                backend.decode(&ids, mode)?
            }
        }
        Operation::ApplyTemplate {
            messages,
            options,
            custom_template,
        } => match custom_template {
            Some(spec) => {
                let CustomTemplateSpec {
                    kwargs,
                    provider: Some(provider),
                    ..
                } = spec
                else {
                    return Err("custom template was not prepared".to_owned());
                };
                backend.apply_template_custom(&messages, options, &provider, kwargs)?
            }
            None => backend.apply_template(&messages, options)?,
        },
        Operation::InputTokens {
            text,
            messages,
            options,
            custom_template,
        } => match custom_template {
            Some(spec) => {
                let CustomTemplateSpec {
                    kwargs,
                    provider: Some(provider),
                    ..
                } = spec
                else {
                    return Err("custom template was not prepared".to_owned());
                };
                backend.input_tokens_custom(&messages, options, &provider, kwargs)?
            }
            None => backend.input_tokens(text.as_deref(), &messages, options)?,
        },
        Operation::Embeddings {
            texts,
            token_inputs,
            device_index,
            target,
        } => backend.embeddings(&texts, &token_inputs, device_index, &target)?,
        Operation::Rerank {
            query,
            documents,
            top_n,
            device_index,
            target,
        } => backend.rerank(&query, &documents, top_n, device_index, &target)?,
        Operation::Infill { prefix, suffix } => backend.infill(&prefix, &suffix)?,
        Operation::Generate(request) => backend.generate(&request)?,
        Operation::Benchmark(request) => backend.benchmark(
            &request,
            benchmark_timing
                .ok_or_else(|| "benchmark timing origin was not initialized".to_owned())?,
        )?,
    };
    if command == "benchmark" {
        return serde_json::to_string(&result)
            .map_err(|_| "benchmark report could not be serialized".to_owned());
    }
    serialize_report(command, &backend.identity(), result)
}

fn serialize_report(
    command: &str,
    identity: &ModelIdentity,
    result: Value,
) -> Result<String, String> {
    let generation = command == "generate" || command == "completions" || command == "infill";
    let model_execution = generation || command == "embeddings" || command == "rerank";
    serde_json::to_string(&json!({
        "schema_version": REPORT_SCHEMA,
        "command": command,
        "state": "PASS",
        "model": {
            "repo_id": identity.repo_id,
            "resolved_revision": identity.resolved_revision,
            "lock_fingerprint": identity.lock_fingerprint,
        },
        "scope": {
            "offline": true,
            "gpu_execution": model_execution,
            "model_execution": model_execution,
            "generation": generation,
        },
        "result": result,
    }))
    .map_err(|_| "model frontend report could not be serialized".to_owned())
}

fn parse(command: &str, arguments: impl Iterator<Item = String>) -> Result<Request, String> {
    // The CLI keeps one generation state machine for the legacy `generate`
    // command and the Phase 42 Completions alias.  The report still retains
    // the user-facing command name in the caller.
    let command = if command == "completions" {
        "generate"
    } else {
        command
    };
    let mut gguf = None;
    let mut derived_lock = None;
    let mut chat_template_file: Option<PathBuf> = None;
    let mut chat_template_digest: Option<String> = None;
    let mut template_kwargs: Option<Map<String, Value>> = None;
    let mut text = None;
    let mut embedding_texts = Vec::new();
    let mut token_ids = None;
    let mut embedding_token_inputs = Vec::new();
    let mut messages = Vec::new();
    let mut thinking = None;
    let mut no_generation_prompt = false;
    let mut skip_special_tokens = false;
    let mut include_pieces = false;
    let mut prompt = None;
    let mut max_new_tokens = None;
    let mut prefill_chunk_tokens = None;
    let mut mtp_draft_width = None;
    let mut device_index = None;
    let mut target = None;
    let mut kv_cache_encoding = None;
    let mut greedy = false;
    let mut temperature = None;
    let mut top_p = None;
    let mut presence_penalty = None;
    let mut frequency_penalty = None;
    let mut seed = None;
    let mut stop_strings = Vec::new();
    let mut benchmark_lane = None;
    let mut benchmark_row_id = None;
    let mut benchmark_model_size = None;
    let mut benchmark_case_id = None;
    let mut benchmark_warmups = None;
    let mut benchmark_measured = None;
    let mut benchmark_ignore_eos = false;
    let mut benchmark_context_length = None;
    let mut benchmark_completion_timeout_seconds = None;
    let fp8_manifest: Option<PathBuf> = None;
    let fp8_artifact: Option<PathBuf> = None;
    let fp8_provider: Option<CliFp8Provider> = None;
    let mut image_paths = Vec::new();
    let mut query = None;
    let mut documents = Vec::new();
    let mut top_n = None;
    let mut infill_prefix = None;
    let mut infill_suffix = None;
    let mut message_bytes = 0_usize;
    let mut phase42_aggregate_bytes = 0_usize;
    let mut rerank_document_bytes = 0_usize;
    let mut arguments = arguments.peekable();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--gguf" => set_once(&mut gguf, take_value(&mut arguments, "--gguf")?, "--gguf")?,
            "--derived-lock" => set_once(
                &mut derived_lock,
                take_value(&mut arguments, "--derived-lock")?,
                "--derived-lock",
            )?,
            "--chat-template-file" if command == "apply-template" || command == "input-tokens" => {
                let value = take_value(&mut arguments, "--chat-template-file")?;
                if value.is_empty() {
                    return Err("--chat-template-file must not be empty".to_owned());
                }
                set_once(
                    &mut chat_template_file,
                    PathBuf::from(value),
                    "--chat-template-file",
                )?;
            }
            "--chat-template-digest"
                if command == "apply-template" || command == "input-tokens" =>
            {
                let value = take_value(&mut arguments, "--chat-template-digest")?;
                set_once(&mut chat_template_digest, value, "--chat-template-digest")?;
            }
            "--template-kwargs-json"
                if command == "apply-template" || command == "input-tokens" =>
            {
                let value = take_value(&mut arguments, "--template-kwargs-json")?;
                set_once(
                    &mut template_kwargs,
                    crate::template_file::parse_kwargs_json(&value)?,
                    "--template-kwargs-json",
                )?;
            }
            "--text"
                if command == "tokenize"
                    || command == "input-tokens"
                    || command == "embeddings" =>
            {
                let value = take_value(&mut arguments, "--text")?;
                if value.len() > MAX_TEXT_BYTES {
                    return Err("--text exceeds the 16 MiB input limit".to_owned());
                }
                if command == "embeddings" {
                    if value.is_empty() {
                        return Err("embedding --text must not be empty".to_owned());
                    }
                    if embedding_texts.len() == MAX_EMBEDDING_INPUTS {
                        return Err(format!(
                            "embeddings accepts at most {MAX_EMBEDDING_INPUTS} --text values"
                        ));
                    }
                    phase42_aggregate_bytes = phase42_aggregate_bytes
                        .checked_add(value.len())
                        .ok_or_else(|| "embedding input size overflow".to_owned())?;
                    if phase42_aggregate_bytes > MAX_PHASE42_AGGREGATE_BYTES {
                        return Err("embedding inputs exceed the 96 MiB aggregate limit".to_owned());
                    }
                    embedding_texts.push(value);
                } else {
                    set_once(&mut text, value, "--text")?;
                }
            }
            "--tokens" if command == "decode" || command == "detokenize" => {
                let value = take_value(&mut arguments, "--tokens")?;
                set_once(&mut token_ids, parse_token_ids(&value)?, "--tokens")?;
            }
            "--tokens" if command == "embeddings" => {
                let value = take_value(&mut arguments, "--tokens")?;
                if embedding_token_inputs.len() == MAX_EMBEDDING_INPUTS {
                    return Err(format!(
                        "embeddings accepts at most {MAX_EMBEDDING_INPUTS} --tokens values"
                    ));
                }
                phase42_aggregate_bytes = phase42_aggregate_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| "embedding input size overflow".to_owned())?;
                if phase42_aggregate_bytes > MAX_PHASE42_AGGREGATE_BYTES {
                    return Err("embedding inputs exceed the 96 MiB aggregate limit".to_owned());
                }
                embedding_token_inputs.push(parse_token_ids(&value)?);
            }
            "--input-token-ids" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--input-token-ids")?;
                set_once(
                    &mut token_ids,
                    parse_token_ids(&value)?,
                    "--input-token-ids",
                )?;
            }
            "--input-token-ids-file" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--input-token-ids-file")?;
                set_once(
                    &mut token_ids,
                    parse_token_ids_file(Path::new(&value))?,
                    "--input-token-ids-file",
                )?;
            }
            "--skip-special-tokens" if command == "decode" || command == "detokenize" => {
                if skip_special_tokens {
                    return Err("duplicate --skip-special-tokens".to_owned());
                }
                skip_special_tokens = true;
            }
            "--pieces" if command == "tokenize" => {
                if include_pieces {
                    return Err("duplicate --pieces".to_owned());
                }
                include_pieces = true;
            }
            "--prompt" if command == "generate" => {
                let value = take_value(&mut arguments, "--prompt")?;
                if value.is_empty() {
                    return Err("--prompt must not be empty".to_owned());
                }
                if value.len() > MAX_TEXT_BYTES {
                    return Err("--prompt exceeds the 16 MiB input limit".to_owned());
                }
                set_once(&mut prompt, value, "--prompt")?;
            }
            "--image" if command == "generate" => {
                let value = take_value(&mut arguments, "--image")?;
                if image_paths.len() == 2 {
                    return Err("generate accepts at most two --image PATH values".to_owned());
                }
                image_paths.push(PathBuf::from(value));
            }
            "--max-new-tokens" if command == "generate" || command == "benchmark" => {
                let value = take_value(&mut arguments, "--max-new-tokens")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err("--max-new-tokens must be an unsigned decimal U32".to_owned());
                }
                let parsed = value
                    .parse::<u32>()
                    .map_err(|_| "--max-new-tokens must be an unsigned decimal U32".to_owned())?;
                let max_allowed = if command == "benchmark" {
                    MAX_BENCHMARK_DIRECT_NEW_TOKENS
                } else {
                    MAX_NEW_TOKENS
                };
                if parsed == 0 || parsed > max_allowed {
                    return Err(format!("--max-new-tokens must be in [1,{max_allowed}]"));
                }
                set_once(&mut max_new_tokens, parsed, "--max-new-tokens")?;
            }
            "--ignore-eos" if command == "benchmark" => {
                if benchmark_ignore_eos {
                    return Err("duplicate --ignore-eos".to_owned());
                }
                benchmark_ignore_eos = true;
            }
            "--context-length" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--context-length")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err("--context-length must be an unsigned decimal U64".to_owned());
                }
                let parsed = value
                    .parse::<u64>()
                    .map_err(|_| "--context-length must be an unsigned decimal U64".to_owned())?;
                if parsed == 0 || parsed > MAX_BENCHMARK_CONTEXT_LENGTH {
                    return Err(format!(
                        "--context-length must be in [1,{MAX_BENCHMARK_CONTEXT_LENGTH}]"
                    ));
                }
                set_once(&mut benchmark_context_length, parsed, "--context-length")?;
            }
            "--completion-timeout-seconds" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--completion-timeout-seconds")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err(
                        "--completion-timeout-seconds must be an unsigned decimal U64".to_owned(),
                    );
                }
                let parsed = value.parse::<u64>().map_err(|_| {
                    "--completion-timeout-seconds must be an unsigned decimal U64".to_owned()
                })?;
                if parsed == 0 || parsed > MAX_BENCHMARK_COMPLETION_TIMEOUT_SECONDS {
                    return Err(format!(
                        "--completion-timeout-seconds must be in [1,{MAX_BENCHMARK_COMPLETION_TIMEOUT_SECONDS}]"
                    ));
                }
                set_once(
                    &mut benchmark_completion_timeout_seconds,
                    parsed,
                    "--completion-timeout-seconds",
                )?;
            }
            "--prefill-chunk-tokens" if command == "generate" || command == "benchmark" => {
                let value = take_value(&mut arguments, "--prefill-chunk-tokens")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err("--prefill-chunk-tokens must be an unsigned decimal U64".to_owned());
                }
                let parsed = value.parse::<u64>().map_err(|_| {
                    "--prefill-chunk-tokens must be an unsigned decimal U64".to_owned()
                })?;
                if parsed == 0 || parsed > MAX_PREFILL_CHUNK_TOKENS {
                    return Err(format!(
                        "--prefill-chunk-tokens must be in [1,{MAX_PREFILL_CHUNK_TOKENS}]"
                    ));
                }
                set_once(&mut prefill_chunk_tokens, parsed, "--prefill-chunk-tokens")?;
            }
            "--mtp-draft-width" if command == "generate" => {
                let value = take_value(&mut arguments, "--mtp-draft-width")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err("--mtp-draft-width must be an unsigned decimal U8".to_owned());
                }
                let parsed = value
                    .parse::<u8>()
                    .map_err(|_| "--mtp-draft-width must be an unsigned decimal U8".to_owned())?;
                if parsed > MAX_MTP_DRAFT_WIDTH {
                    return Err(format!(
                        "--mtp-draft-width must be in [0,{MAX_MTP_DRAFT_WIDTH}]"
                    ));
                }
                set_once(&mut mtp_draft_width, parsed, "--mtp-draft-width")?;
            }
            "--device-index"
                if command == "generate"
                    || command == "benchmark"
                    || command == "embeddings"
                    || command == "rerank" =>
            {
                let value = take_value(&mut arguments, "--device-index")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err("--device-index must be an unsigned decimal U32".to_owned());
                }
                let parsed = value
                    .parse::<u32>()
                    .map_err(|_| "--device-index must be an unsigned decimal U32".to_owned())?;
                set_once(&mut device_index, parsed, "--device-index")?;
            }
            "--target"
                if command == "generate"
                    || command == "benchmark"
                    || command == "embeddings"
                    || command == "rerank" =>
            {
                let value = take_value(&mut arguments, "--target")?;
                if !matches!(
                    value.as_str(),
                    "gfx1030" | "gfx1201" | "gfx942" | "gfx942:sramecc+:xnack-"
                ) {
                    return Err(
                        "--target must be gfx1030, gfx1201, gfx942, or gfx942:sramecc+:xnack-"
                            .to_owned(),
                    );
                }
                set_once(&mut target, value, "--target")?;
            }
            "--kv-cache-encoding" if command == "generate" || command == "benchmark" => {
                let value = take_value(&mut arguments, "--kv-cache-encoding")?;
                let parsed = match value.as_str() {
                    "fp16" => KvCacheEncoding::Fp16,
                    "fp8" => KvCacheEncoding::Fp8E4M3Fn,
                    "fp8-static" => KvCacheEncoding::Fp8E4M3FnStatic,
                    "nvfp4" => KvCacheEncoding::Nvfp4,
                    "kv-mxfp8-e4" => KvCacheEncoding::Mxfp8E4,
                    "kv-mxfp8-e5" => KvCacheEncoding::Mxfp8E5,
                    _ => {
                        return Err(
                            "--kv-cache-encoding must be fp16, fp8, fp8-static, nvfp4, kv-mxfp8-e4, or kv-mxfp8-e5"
                                .to_owned(),
                        );
                    }
                };
                set_once(&mut kv_cache_encoding, parsed, "--kv-cache-encoding")?;
            }
            "--greedy" if command == "generate" || command == "benchmark" => {
                if greedy {
                    return Err("duplicate --greedy".to_owned());
                }
                greedy = true;
            }
            "--temperature" if command == "generate" => {
                let value = take_value(&mut arguments, "--temperature")?;
                set_once(
                    &mut temperature,
                    parse_f32(&value, "--temperature")?,
                    "--temperature",
                )?;
            }
            "--top-p" if command == "generate" => {
                let value = take_value(&mut arguments, "--top-p")?;
                set_once(&mut top_p, parse_f32(&value, "--top-p")?, "--top-p")?;
            }
            "--presence-penalty" if command == "generate" => {
                let value = take_value(&mut arguments, "--presence-penalty")?;
                set_once(
                    &mut presence_penalty,
                    parse_f32(&value, "--presence-penalty")?,
                    "--presence-penalty",
                )?;
            }
            "--frequency-penalty" if command == "generate" => {
                let value = take_value(&mut arguments, "--frequency-penalty")?;
                set_once(
                    &mut frequency_penalty,
                    parse_f32(&value, "--frequency-penalty")?,
                    "--frequency-penalty",
                )?;
            }
            "--seed" if command == "generate" => {
                let value = take_value(&mut arguments, "--seed")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err("--seed must be an unsigned decimal U64".to_owned());
                }
                set_once(
                    &mut seed,
                    value
                        .parse::<u64>()
                        .map_err(|_| "--seed must be an unsigned decimal U64".to_owned())?,
                    "--seed",
                )?;
            }
            "--stop" if command == "generate" => {
                let value = take_value(&mut arguments, "--stop")?;
                if value.is_empty() {
                    return Err("--stop must not be empty".to_owned());
                }
                stop_strings.push(value);
            }
            "--lane" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--lane")?;
                let parsed = match value.as_str() {
                    "direct" => BenchmarkLane::Direct,
                    "render-tokenize" => BenchmarkLane::RenderTokenize,
                    _ => return Err("--lane must be direct or render-tokenize".to_owned()),
                };
                set_once(&mut benchmark_lane, parsed, "--lane")?;
            }
            "--row-id" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--row-id")?;
                set_once(&mut benchmark_row_id, value, "--row-id")?;
            }
            "--model-size" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--model-size")?;
                if !matches!(value.as_str(), "2B" | "4B" | "9B") {
                    return Err("--model-size must be 2B, 4B, or 9B".to_owned());
                }
                set_once(&mut benchmark_model_size, value, "--model-size")?;
            }
            "--case-id" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--case-id")?;
                if value.is_empty()
                    || value.len() > 128
                    || !value.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
                    })
                {
                    return Err("--case-id must be a bounded lowercase identifier".to_owned());
                }
                set_once(&mut benchmark_case_id, value, "--case-id")?;
            }
            "--warmups" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--warmups")?;
                let parsed = parse_bounded_count(&value, "--warmups")?;
                set_once(&mut benchmark_warmups, parsed, "--warmups")?;
            }
            "--measured" if command == "benchmark" => {
                let value = take_value(&mut arguments, "--measured")?;
                let parsed = parse_bounded_count(&value, "--measured")?;
                set_once(&mut benchmark_measured, parsed, "--measured")?;
            }
            "--message"
                if command == "render"
                    || command == "apply-template"
                    || command == "input-tokens"
                    || command == "generate"
                    || command == "benchmark" =>
            {
                let value = take_value(&mut arguments, "--message")?;
                message_bytes = message_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| "--message input size overflow".to_owned())?;
                let max_messages = if command == "apply-template" || command == "input-tokens" {
                    1_024
                } else {
                    4_096
                };
                if message_bytes > MAX_TEXT_BYTES || messages.len() == max_messages {
                    return Err("render message input exceeds the bounded CLI limit".to_owned());
                }
                messages.push(parse_message(&value)?);
            }
            "--thinking"
                if command == "render"
                    || command == "apply-template"
                    || command == "input-tokens"
                    || command == "generate"
                    || command == "benchmark" =>
            {
                let value = match take_value(&mut arguments, "--thinking")?.as_str() {
                    "default" => ThinkingModeV1::TemplateDefault,
                    "enabled" => ThinkingModeV1::Enabled,
                    "disabled" => ThinkingModeV1::Disabled,
                    _ => return Err("--thinking must be default, enabled, or disabled".to_owned()),
                };
                set_once(&mut thinking, value, "--thinking")?;
            }
            "--no-generation-prompt" if command == "render" || command == "apply-template" => {
                if no_generation_prompt {
                    return Err("duplicate --no-generation-prompt".to_owned());
                }
                no_generation_prompt = true;
            }
            "--query" if command == "rerank" => {
                let value = take_value(&mut arguments, "--query")?;
                if value.is_empty() || value.len() > MAX_TEXT_BYTES {
                    return Err(
                        "--query must be non-empty and within the 16 MiB input limit".to_owned(),
                    );
                }
                set_once(&mut query, value, "--query")?;
            }
            "--document" if command == "rerank" => {
                let value = take_value(&mut arguments, "--document")?;
                if value.is_empty() || value.len() > MAX_TEXT_BYTES {
                    return Err(
                        "--document must be non-empty and within the 16 MiB input limit".to_owned(),
                    );
                }
                if documents.len() == 256 {
                    return Err("rerank accepts at most 256 --document values".to_owned());
                }
                rerank_document_bytes = rerank_document_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| "rerank input size overflow".to_owned())?;
                if rerank_document_bytes > MAX_TEXT_BYTES {
                    return Err("rerank documents exceed the 16 MiB aggregate limit".to_owned());
                }
                documents.push(value);
            }
            "--top-n" if command == "rerank" => {
                let value = take_value(&mut arguments, "--top-n")?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| "--top-n must be an unsigned decimal".to_owned())?;
                set_once(&mut top_n, parsed, "--top-n")?;
            }
            "--prefix" if command == "infill" => {
                let value = take_value(&mut arguments, "--prefix")?;
                if value.len() > MAX_TEXT_BYTES {
                    return Err("--prefix exceeds the 16 MiB input limit".to_owned());
                }
                set_once(&mut infill_prefix, value, "--prefix")?;
            }
            "--suffix" if command == "infill" => {
                let value = take_value(&mut arguments, "--suffix")?;
                if value.len() > MAX_TEXT_BYTES {
                    return Err("--suffix exceeds the 16 MiB input limit".to_owned());
                }
                set_once(&mut infill_suffix, value, "--suffix")?;
            }
            value => return Err(format!("unexpected argument `{value}` for `{command}`")),
        }
    }

    let gguf = Some(PathBuf::from(
        gguf.ok_or_else(|| "missing required --gguf PATH".to_owned())?,
    ));
    let derived_lock =
        Some(PathBuf::from(derived_lock.ok_or_else(|| {
            "missing required --derived-lock PATH".to_owned()
        })?));
    let custom_template = match (chat_template_file, chat_template_digest) {
        (Some(path), Some(digest)) => Some(CustomTemplateSpec {
            path,
            digest,
            kwargs: template_kwargs.unwrap_or_default(),
            provider: None,
        }),
        (None, None) => {
            if template_kwargs.is_some() {
                return Err(
                    "--template-kwargs-json requires both custom template file and digest"
                        .to_owned(),
                );
            }
            None
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(
                "custom template requires both --chat-template-file and --chat-template-digest"
                    .to_owned(),
            );
        }
    };
    let operation = match command {
        "verify-model" => Operation::Verify,
        "tokenize" => Operation::Tokenize {
            text: text.ok_or_else(|| "missing required --text TEXT".to_owned())?,
            pieces: include_pieces,
        },
        "render" => {
            if messages.is_empty() {
                return Err("render requires at least one --message ROLE:CONTENT".to_owned());
            }
            Operation::Render {
                messages,
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: !no_generation_prompt,
                    thinking: thinking.unwrap_or(ThinkingModeV1::TemplateDefault),
                },
            }
        }
        "apply-template" => {
            if messages.is_empty() {
                return Err(
                    "apply-template requires at least one --message ROLE:CONTENT".to_owned(),
                );
            }
            Operation::ApplyTemplate {
                messages,
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: !no_generation_prompt,
                    thinking: thinking.unwrap_or(ThinkingModeV1::TemplateDefault),
                },
                custom_template,
            }
        }
        "input-tokens" => {
            if (text.is_some() && !messages.is_empty()) || (text.is_none() && messages.is_empty()) {
                return Err(
                    "input-tokens requires exactly one --text or at least one --message".to_owned(),
                );
            }
            if text.is_some() && custom_template.is_some() {
                return Err("custom template requires --message input".to_owned());
            }
            Operation::InputTokens {
                text,
                messages,
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: !no_generation_prompt,
                    thinking: thinking.unwrap_or(ThinkingModeV1::TemplateDefault),
                },
                custom_template,
            }
        }
        "decode" | "detokenize" => Operation::Decode {
            ids: token_ids.ok_or_else(|| "missing required --tokens IDS".to_owned())?,
            mode: if skip_special_tokens {
                DecodeModeV1::SkipSpecialTokens
            } else {
                DecodeModeV1::PreserveSpecialTokens
            },
        },
        "embeddings" => {
            if embedding_texts.is_empty() && embedding_token_inputs.is_empty() {
                return Err("embeddings requires at least one --text or --tokens".to_owned());
            }
            if !embedding_texts.is_empty() && !embedding_token_inputs.is_empty() {
                return Err("embeddings does not mix --text and --tokens inputs".to_owned());
            }
            Operation::Embeddings {
                texts: embedding_texts,
                token_inputs: embedding_token_inputs,
                device_index: device_index
                    .ok_or_else(|| "embeddings requires --device-index".to_owned())?,
                target: target.ok_or_else(|| "embeddings requires --target".to_owned())?,
            }
        }
        "rerank" => {
            let query = query.ok_or_else(|| "rerank requires --query TEXT".to_owned())?;
            if documents.is_empty() {
                return Err("rerank requires at least one --document TEXT".to_owned());
            }
            if let Some(top_n) = top_n {
                if top_n == 0 || top_n > documents.len() {
                    return Err("--top-n must be in 1..=document count".to_owned());
                }
            }
            Operation::Rerank {
                query,
                documents,
                top_n,
                device_index: device_index
                    .ok_or_else(|| "rerank requires --device-index".to_owned())?,
                target: target.ok_or_else(|| "rerank requires --target".to_owned())?,
            }
        }
        "infill" => Operation::Infill {
            prefix: infill_prefix.ok_or_else(|| "infill requires --prefix TEXT".to_owned())?,
            suffix: infill_suffix.ok_or_else(|| "infill requires --suffix TEXT".to_owned())?,
        },
        "generate" => {
            if greedy && temperature.is_some() {
                return Err("generate accepts --greedy or --temperature, not both".to_owned());
            }
            let options = Qwen35RenderOptionsV1 {
                add_generation_prompt: true,
                thinking: thinking.unwrap_or(ThinkingModeV1::TemplateDefault),
            };
            let input = match (prompt, messages.is_empty()) {
                (Some(prompt), true) => {
                    if thinking.is_some() {
                        return Err("--thinking is valid only with generate --message".to_owned());
                    }
                    GenerationInput::Prompt(prompt)
                }
                (None, false) => GenerationInput::Messages { messages, options },
                (Some(_), false) => {
                    return Err(
                        "generate accepts either --prompt or --message, not both".to_owned()
                    );
                }
                (None, true) => {
                    return Err("generate requires --prompt or at least one --message".to_owned());
                }
            };
            Operation::Generate(GenerateRequest {
                input,
                image_paths,
                max_new_tokens: max_new_tokens
                    .ok_or_else(|| "generate requires --max-new-tokens".to_owned())?,
                prefill_chunk_tokens,
                mtp_draft_width,
                sampling: SamplingParametersV1::new(
                    if greedy {
                        0.0
                    } else {
                        temperature.unwrap_or(1.0)
                    },
                    top_p.unwrap_or(1.0),
                    presence_penalty.unwrap_or(0.0),
                    frequency_penalty.unwrap_or(0.0),
                )
                .map_err(|error| format!("invalid generation sampling parameters: {error}"))?,
                seed,
                stop_strings,
                device_index: device_index
                    .ok_or_else(|| "generate requires --device-index".to_owned())?,
                target: target.ok_or_else(|| "generate requires --target".to_owned())?,
                kv_cache_encoding,
                fp8_manifest,
                fp8_artifact,
                fp8_provider,
            })
        }
        "benchmark" => {
            if !greedy {
                return Err("benchmark requires explicit --greedy mode".to_owned());
            }
            let lane = benchmark_lane.unwrap_or(BenchmarkLane::RenderTokenize);
            let warmups = benchmark_warmups.unwrap_or(3);
            let measured = benchmark_measured.unwrap_or(10);
            validate_benchmark_protocol(warmups, measured)?;
            let model_size =
                benchmark_model_size.ok_or_else(|| "benchmark requires --model-size".to_owned())?;
            if lane == BenchmarkLane::RenderTokenize
                && kv_cache_encoding.is_some()
                && kv_cache_encoding != Some(KvCacheEncoding::Fp16)
            {
                return Err(
                    "benchmark render-tokenize lane requires --kv-cache-encoding fp16".to_owned(),
                );
            }
            let row_id = benchmark_row_id.unwrap_or_else(|| match lane {
                BenchmarkLane::Direct => "cli-direct".to_owned(),
                BenchmarkLane::RenderTokenize => "cli-render-tokenize".to_owned(),
            });
            let case_id = benchmark_case_id.unwrap_or_else(|| match lane {
                BenchmarkLane::Direct => "direct".to_owned(),
                BenchmarkLane::RenderTokenize => "render-tokenize".to_owned(),
            });
            let input = match lane {
                BenchmarkLane::Direct => {
                    if !messages.is_empty() || thinking.is_some() {
                        return Err(
                            "benchmark direct lane accepts only pretokenized input".to_owned()
                        );
                    }
                    BenchmarkInput::TokenIds(token_ids.ok_or_else(|| {
                        "benchmark direct lane requires --input-token-ids IDS or --input-token-ids-file PATH".to_owned()
                    })?)
                }
                BenchmarkLane::RenderTokenize => {
                    if token_ids.is_some() {
                        return Err(
                            "benchmark render-tokenize lane does not accept pretokenized input"
                                .to_owned(),
                        );
                    }
                    if messages.is_empty() {
                        return Err(
                            "benchmark render-tokenize lane requires --message ROLE:CONTENT"
                                .to_owned(),
                        );
                    }
                    BenchmarkInput::Messages {
                        messages,
                        options: Qwen35RenderOptionsV1 {
                            add_generation_prompt: true,
                            thinking: thinking.unwrap_or(ThinkingModeV1::TemplateDefault),
                        },
                    }
                }
            };
            if lane == BenchmarkLane::RenderTokenize && prefill_chunk_tokens.is_some() {
                return Err(
                    "benchmark --prefill-chunk-tokens is supported only for the direct lane"
                        .to_owned(),
                );
            }
            let requested_max_new_tokens =
                max_new_tokens.ok_or_else(|| "benchmark requires --max-new-tokens".to_owned())?;
            if lane == BenchmarkLane::RenderTokenize && requested_max_new_tokens > MAX_NEW_TOKENS {
                return Err(format!(
                    "benchmark render-tokenize --max-new-tokens must be in [1,{MAX_NEW_TOKENS}]"
                ));
            }
            if lane == BenchmarkLane::RenderTokenize && benchmark_ignore_eos {
                return Err("--ignore-eos is supported only for benchmark direct lane".to_owned());
            }
            Operation::Benchmark(BenchmarkRequest {
                lane,
                row_id,
                model_size,
                case_id,
                input,
                max_new_tokens: requested_max_new_tokens,
                ignore_eos: benchmark_ignore_eos,
                context_length: benchmark_context_length,
                completion_timeout_seconds: benchmark_completion_timeout_seconds,
                prefill_chunk_tokens,
                device_index: device_index
                    .ok_or_else(|| "benchmark requires --device-index".to_owned())?,
                target: target.ok_or_else(|| "benchmark requires --target".to_owned())?,
                greedy,
                warmups,
                measured,
                kv_cache_encoding,
                fp8_manifest,
                fp8_artifact,
                fp8_provider,
            })
        }
        _ => return Err("internal unsupported model command".to_owned()),
    };
    Ok(Request {
        gguf,
        derived_lock,
        operation,
    })
}

fn parse_bounded_count(value: &str, flag: &str) -> Result<u32, String> {
    if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
        return Err(format!("{flag} must be an unsigned decimal U32"));
    }
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("{flag} must be an unsigned decimal U32"))?;
    if parsed > 100 {
        return Err(format!("{flag} must be in [0,100]"));
    }
    Ok(parsed)
}

fn cli_prefill_chunk_candidates(
    explicit_tokens: Option<u64>,
    total_memory_bytes: u64,
    input_tokens: u64,
) -> Result<Vec<u64>, String> {
    if input_tokens == 0 {
        return Err("prefill chunk selection requires non-zero prompt tokens".to_owned());
    }
    if let Some(tokens) = explicit_tokens {
        if tokens == 0 || tokens > MAX_PREFILL_CHUNK_TOKENS {
            return Err(format!(
                "--prefill-chunk-tokens must be in [1,{MAX_PREFILL_CHUNK_TOKENS}]"
            ));
        }
        // Keep short prompts unpadded, matching the automatic selector's
        // effective-row policy. The explicit path still has exactly one
        // candidate, so placement failure cannot fall back to another size.
        return Ok(vec![input_tokens.min(tokens)]);
    }
    qwen_prefill_chunk_candidates(total_memory_bytes, input_tokens)
        .map_err(|error| error.to_string())
}

fn cli_state_capacity_with_mtp_slack(
    logical_state_capacity: u64,
    effective_width: Option<u8>,
) -> Result<(u64, u64), String> {
    let slack_tokens = effective_width.map(u64::from).unwrap_or(0);
    let allocated_state_capacity = logical_state_capacity
        .checked_add(slack_tokens)
        .ok_or_else(|| "MTP state capacity overflowed".to_owned())?;
    if allocated_state_capacity > QWEN_RUNTIME_MAX_CONTEXT_TOKENS {
        return Err(format!(
            "MTP state capacity {allocated_state_capacity} exceeds runtime context limit {QWEN_RUNTIME_MAX_CONTEXT_TOKENS}"
        ));
    }
    Ok((allocated_state_capacity, slack_tokens))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CliMtpPlan {
    selection: &'static str,
    enabled: bool,
    requested_width: Option<u8>,
    effective_width: Option<u8>,
}

#[allow(clippy::too_many_arguments)]
fn resolve_cli_mtp_plan(
    requested_width: Option<u8>,
    has_images: bool,
    has_sidecar: bool,
    embedded_fp8: bool,
    target: &str,
    kv_cache_encoding: KvCacheEncoding,
    sampling: SamplingParametersV1,
    model_fingerprint: &str,
) -> Result<CliMtpPlan, String> {
    match requested_width {
        Some(0) => Ok(CliMtpPlan {
            selection: "target-only",
            enabled: false,
            requested_width,
            effective_width: None,
        }),
        Some(width) => {
            if width > MAX_MTP_DRAFT_WIDTH {
                return Err(format!(
                    "--mtp-draft-width must be in [0,{MAX_MTP_DRAFT_WIDTH}]"
                ));
            }
            let incompatibility = if has_images {
                Some("vision requests have no verified Qwen MTP executor path")
            } else if has_sidecar {
                Some("FP8/NVFP4 sidecar weights have no verified Qwen MTP executor path")
            } else if target != "gfx1201" && target != "gfx942" {
                Some("forced MTP is reviewed only for exact gfx1201 or gfx942")
            } else if embedded_fp8 && kv_cache_encoding != KvCacheEncoding::Fp8E4M3Fn {
                Some("embedded FP8 forced MTP requires the dynamic FP8 target KV cache encoding")
            } else if !embedded_fp8 && kv_cache_encoding != KvCacheEncoding::Fp16 {
                Some("BF16 forced MTP requires the FP16 KV cache encoding")
            } else if sampling.requires_logits() {
                Some("forced MTP requires greedy sampling without logits")
            } else if model_fingerprint != sllm_core::QWEN35_4B_FINGERPRINT {
                Some("forced MTP requires the reviewed fixed Qwen3.5-4B model")
            } else {
                None
            };
            if let Some(reason) = incompatibility {
                return Err(format!("forced MTP is incompatible: {reason}"));
            }
            Ok(CliMtpPlan {
                selection: "forced",
                enabled: true,
                requested_width,
                effective_width: Some(width),
            })
        }
        None => {
            let enabled = !has_images
                && !has_sidecar
                && !embedded_fp8
                && target == "gfx1201"
                && kv_cache_encoding == KvCacheEncoding::Fp16
                && !sampling.requires_logits()
                && model_fingerprint == sllm_core::QWEN35_4B_FINGERPRINT;
            Ok(CliMtpPlan {
                selection: "auto",
                enabled,
                requested_width: None,
                effective_width: enabled.then_some(1),
            })
        }
    }
}

fn parse_f32(value: &str, flag: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("{flag} must be a finite decimal F32"))?;
    if !parsed.is_finite() {
        return Err(format!("{flag} must be a finite decimal F32"));
    }
    Ok(parsed)
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("duplicate {flag}"));
    }
    Ok(())
}

fn take_value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_token_ids(value: &str) -> Result<TokenIdsV1, String> {
    if value.is_empty() {
        return Err("--tokens must not be empty".to_owned());
    }
    let mut ids = Vec::new();
    for item in value.split(',') {
        if ids.len() == MAX_TOKEN_IDS {
            return Err("--tokens exceeds the 1048576-ID input limit".to_owned());
        }
        if item.is_empty() || item.bytes().any(|byte| !byte.is_ascii_digit()) {
            return Err("--tokens must be comma-separated unsigned decimal IDs".to_owned());
        }
        ids.push(
            item.parse::<u32>()
                .map_err(|_| "--tokens contains an ID outside u32".to_owned())?,
        );
    }
    Ok(TokenIdsV1::from_slice(&ids))
}

fn parse_token_ids_file(path: &Path) -> Result<TokenIdsV1, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat --input-token-ids-file: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("--input-token-ids-file must be a regular non-symlink file".to_owned());
    }
    if metadata.len() == 0 || metadata.len() > MAX_TOKEN_IDS_FILE_BYTES {
        return Err(format!(
            "--input-token-ids-file must contain between 1 and {MAX_TOKEN_IDS_FILE_BYTES} bytes"
        ));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read --input-token-ids-file: {error}"))?;
    if bytes.len() as u64 != metadata.len() {
        return Err("--input-token-ids-file changed while being read".to_owned());
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| "--input-token-ids-file must be UTF-8 ASCII token IDs".to_owned())?;
    let text = text.strip_suffix('\n').unwrap_or(text);
    parse_token_ids(text)
}

fn parse_message(value: &str) -> Result<Qwen35ChatMessageV1, String> {
    let (role, content) = value
        .split_once(':')
        .ok_or_else(|| "--message must use ROLE:CONTENT".to_owned())?;
    if content.len() > MAX_TEXT_BYTES {
        return Err("--message content exceeds the 16 MiB input limit".to_owned());
    }
    match role {
        "system" => Ok(Qwen35ChatMessageV1::system(content)),
        "user" => Ok(Qwen35ChatMessageV1::user(content)),
        "assistant" => Ok(Qwen35ChatMessageV1::assistant(content, None)),
        _ => Err("message role must be system, user, or assistant".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correctness_reference_reuses_first_warmup_without_extra_request() {
        let warmup = json!({
            "tokens": {
                "input_token_ids": [1, 3, 17],
                "generated_token_ids": [7, 8, 9],
                "visible_token_ids": [7, 8, 9],
                "decode_input_token_ids": [7, 8],
            },
            "stop": {"kind": "max_new_tokens", "token_id": null},
            "audit": {
                "selected_backend": "hip",
                "target": "gfx1030",
                "device_index": 0,
                "model_fingerprint": "model",
                "plan_digest": "plan",
                "fallback_used": false,
                "all_dispatches_hip": true,
                "submission_count": 12,
                "kernel_dispatch_count": 12,
                "segment_count": 3,
                "boundary_count": 4,
            },
            "memory": {"request_start": {}, "after_cleanup": {}},
        });
        let reference = correctness_reference_from_warmup(&warmup).unwrap();
        assert_eq!(reference["label"], "correctness-reference");
        assert_eq!(reference["execution_path"], "first-warmup-sample");
        assert_eq!(reference["timing_instrumentation"], "on");
        assert_eq!(reference["source"]["request_count"], 0);
        assert_eq!(reference["tokens"], warmup["tokens"]);
        assert_eq!(reference["audit"], warmup["audit"]);
        assert_eq!(reference["cleanup"]["reference_sample"], true);
        assert_eq!(
            reference["comparison"]["reference_source"],
            "warmups.samples[0]"
        );
        assert_eq!(
            reference["comparison"]["scope"],
            "first_warmup_reference_against_every_remaining_warmup_and_measured_sample"
        );
    }

    #[test]
    fn fp8_provider_defaults_preserve_target_specific_performance_policy() {
        assert_eq!(
            select_cli_fp8_provider(true, None, "gfx1201").unwrap(),
            Some(CliFp8Provider::Native)
        );
        assert_eq!(
            select_cli_fp8_provider(true, None, "gfx1030").unwrap(),
            Some(CliFp8Provider::ConvertedBf16)
        );
        assert_eq!(
            select_cli_fp8_provider(true, None, "gfx942").unwrap(),
            Some(CliFp8Provider::NativeFnuz)
        );
    }

    #[test]
    fn embedded_gguf_fp8_provider_accepts_only_verified_native_targets() {
        assert_eq!(
            select_cli_gguf_fp8_provider("gfx1201").unwrap(),
            CliFp8Provider::Native
        );
        assert_eq!(
            select_cli_gguf_fp8_provider("gfx942").unwrap(),
            CliFp8Provider::NativeFnuz
        );
        assert_eq!(
            cli_fp8_dtype(select_cli_gguf_fp8_provider("gfx1201").unwrap()),
            sllm_core::DType::F8E4M3Fn
        );
        assert_eq!(
            cli_fp8_dtype(select_cli_gguf_fp8_provider("gfx942").unwrap()),
            sllm_core::DType::F8E4M3FnuZ
        );
    }

    #[test]
    fn embedded_gguf_fp8_provider_rejects_rdna2_and_non_exact_targets() {
        for target in ["gfx1030", "gfx1200", "gfx942:sramecc+:xnack-", "unknown"] {
            let error = select_cli_gguf_fp8_provider(target).unwrap_err();
            assert!(error.contains(target), "{error}");
            assert!(error.contains("native-fnuz"), "{error}");
        }
    }
    use sllm_core::{SamplingError, SamplingRandomSource};
    use sllm_frontend::{
        GenerationExecutorV1, GenerationServiceError, GenerationStepV1, GenerationTextFrontendV1,
    };
    use std::collections::VecDeque;

    struct TinyBackend;

    struct SequenceExecution {
        outputs: VecDeque<i32>,
        prefill_inputs: Vec<Vec<i32>>,
        decode_inputs: Vec<i32>,
        cancelled: bool,
    }

    impl SequenceExecution {
        fn new(outputs: impl IntoIterator<Item = i32>) -> Self {
            Self {
                outputs: outputs.into_iter().collect(),
                prefill_inputs: Vec::new(),
                decode_inputs: Vec::new(),
                cancelled: false,
            }
        }

        fn next(&mut self) -> Result<i32, String> {
            self.outputs
                .pop_front()
                .ok_or_else(|| "fake sequence exhausted".to_owned())
        }
    }

    impl GreedyExecution for SequenceExecution {
        fn prefill_last(&mut self, input_token_ids: &[i32]) -> Result<i32, String> {
            self.prefill_inputs.push(input_token_ids.to_vec());
            self.next()
        }

        fn decode_one(&mut self, token_id: i32) -> Result<i32, String> {
            self.decode_inputs.push(token_id);
            self.next()
        }
    }

    impl GenerationExecutorV1 for SequenceExecution {
        fn prefill(
            &mut self,
            input_token_ids: &[u32],
            include_last_logits: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            assert!(!include_last_logits);
            let input = input_token_ids
                .iter()
                .map(|token| i32::try_from(*token).unwrap())
                .collect::<Vec<_>>();
            self.prefill_inputs.push(input);
            Ok(GenerationStepV1::new(
                u32::try_from(self.next().map_err(GenerationServiceError::Execution)?).unwrap(),
                None,
            ))
        }

        fn decode(
            &mut self,
            token_id: u32,
            include_last_logits: bool,
        ) -> Result<GenerationStepV1, GenerationServiceError> {
            assert!(!include_last_logits);
            self.decode_inputs.push(i32::try_from(token_id).unwrap());
            Ok(GenerationStepV1::new(
                u32::try_from(self.next().map_err(GenerationServiceError::Execution)?).unwrap(),
                None,
            ))
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }
    }

    struct NumericFrontend;

    impl GenerationTextFrontendV1 for NumericFrontend {
        fn encode_generation(&self, _: &str) -> Result<Vec<u32>, GenerationServiceError> {
            Ok(vec![1, 3, 17])
        }

        fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError> {
            Ok(token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","))
        }
    }

    struct NeverRandom;

    impl SamplingRandomSource for NeverRandom {
        fn next_unit_f64(&mut self) -> Result<f64, SamplingError> {
            Err(SamplingError::InvalidRandomValue)
        }
    }

    fn qwen_stop_policy() -> GenerationStopPolicyV1 {
        sllm_core::parse_model_lock(include_bytes!(
            "../../../docs/models/locks/qwen3.5-4b-bf16.json"
        ))
        .unwrap()
        .generation_stop_policy()
        .clone()
    }

    impl ModelFrontendBackend for TinyBackend {
        fn identity(&self) -> ModelIdentity {
            ModelIdentity {
                repo_id: "Qwen/Qwen3.5-4B".to_owned(),
                resolved_revision: "8".repeat(40),
                lock_fingerprint: format!("sha256:{}", "3".repeat(64)),
            }
        }

        fn verify(&self) -> Result<Value, String> {
            Ok(json!({
                "kind": "verify-model", "locked_files": 3, "verified_files": 3,
                "tensor_count": 17, "weight_entries": 17, "loadable_entries": 3,
                "known_unconsumed_entries": 14, "total_destination_bytes": 17,
                "plan_digest": format!("sha256:{}", "9".repeat(64)),
            }))
        }

        fn tokenize(&self, text: &str) -> Result<Value, String> {
            assert_eq!(text, "abc");
            Ok(json!({"kind": "tokenize", "count": 3, "token_ids": [1, 3, 17]}))
        }

        fn render(
            &self,
            messages: &[Qwen35ChatMessageV1],
            options: Qwen35RenderOptionsV1,
        ) -> Result<Value, String> {
            assert_eq!(messages.len(), 1);
            assert_eq!(options.thinking, ThinkingModeV1::Disabled);
            Ok(json!({"kind": "render", "text": "rendered"}))
        }

        fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
            assert_eq!(ids.as_slice(), &[1, 3, 17]);
            assert_eq!(mode, DecodeModeV1::SkipSpecialTokens);
            Ok(json!({"kind": "decode", "text": "decoded"}))
        }

        fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
            assert_eq!(request.max_new_tokens, 3);
            assert_eq!(request.device_index, 0);
            assert_eq!(request.target, "gfx1030");
            assert!(matches!(request.input, GenerationInput::Prompt(ref text) if text == "abc"));
            Ok(json!({
                "kind": "generate",
                "input_token_ids": [1, 3, 17],
                "generated_token_ids": [7, 8, 9],
                "visible_token_ids": [7, 8, 9],
                "decode_input_token_ids": [7, 8],
                "output_text": "generated",
                "stop_reason": {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": null},
                "execution": {"selected_backend": "hip", "target": "gfx1030", "device_index": 0, "fallback_used": false},
                "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
            }))
        }

        fn benchmark(
            &self,
            request: &BenchmarkRequest,
            timing: BenchmarkTiming,
        ) -> Result<Value, String> {
            assert_eq!(request.row_id, "host-test");
            assert_eq!(request.model_size, "4B");
            assert_eq!(request.case_id, "host-test");
            assert_eq!(request.max_new_tokens, 3);
            assert_eq!(request.warmups, 3);
            assert_eq!(request.measured, 10);
            let (lane, lane_definition, tokenizer, render) = match request.lane {
                BenchmarkLane::Direct => {
                    assert!(matches!(request.input, BenchmarkInput::TokenIds(_)));
                    (
                        "direct",
                        "pretokenized direct engine: request start excludes render/tokenize",
                        false,
                        false,
                    )
                }
                BenchmarkLane::RenderTokenize => {
                    assert!(matches!(request.input, BenchmarkInput::Messages { .. }));
                    (
                        "render-tokenize",
                        "CLI end-to-end: request start includes chat render and tokenizer encode",
                        true,
                        true,
                    )
                }
            };
            let config = if request.lane == BenchmarkLane::Direct {
                json!({"lane": lane, "warmups": request.warmups, "measured": request.measured, "tokenizer": tokenizer, "render": render, "kv_cache_encoding": "fp16", "ignore_eos": request.ignore_eos, "context_length": request.context_length, "completion_timeout_seconds": request.completion_timeout_seconds, "prefill_chunk_tokens": request.prefill_chunk_tokens})
            } else {
                json!({"warmups": request.warmups, "measured": request.measured, "tokenizer": tokenizer, "render": render, "ignore_eos": request.ignore_eos, "context_length": request.context_length, "completion_timeout_seconds": request.completion_timeout_seconds, "prefill_chunk_tokens": request.prefill_chunk_tokens})
            };
            Ok(json!({
                "benchmark_schema_version": request.lane.schema_version(),
                "state": "PASS",
                "lane": lane,
                "lane_definition": lane_definition,
                "row": {"row_id": request.row_id, "model_size": request.model_size, "case_id": request.case_id, "input_token_ids": [1, 3, 17], "input_token_count": 3, "requested_output_tokens": 3},
                "identities": {"target": request.target, "device_index": request.device_index},
                "model_load": {"event": "model_load", "start_ns": timing.model_load_start_ns(), "model_ready_ns": 1, "duration_ns": 1, "load_count": 1},
                "config": config,
                "memory": {},
                "audit": {"model_load_count": 1, "request_model_load_count": 0, "model_reused": true},
                "cleanup": {},
                "warmups": {"count": request.warmups, "samples": []},
                "measured": {"count": request.measured, "samples": []},
            }))
        }
    }

    fn parse_args(command: &str, args: &[&str]) -> Result<Request, String> {
        parse(command, args.iter().map(|value| (*value).to_owned()))
    }

    #[test]
    fn all_model_entrances_parse_without_touching_hip() {
        let common = ["--gguf", "model.gguf", "--derived-lock", "model.lock.json"];
        assert_eq!(
            parse_args("verify-model", &common).unwrap().operation,
            Operation::Verify
        );
        assert!(matches!(
            parse_args(
                "tokenize",
                &[
                    "--gguf",
                    "model.gguf",
                    "--derived-lock",
                    "model.lock.json",
                    "--text",
                    "abc"
                ]
            )
            .unwrap()
            .operation,
            Operation::Tokenize { .. }
        ));
        assert!(matches!(
            parse_args(
                "render",
                &[
                    "--message",
                    "user:a:b",
                    "--thinking",
                    "disabled",
                    "--gguf",
                    "model.gguf",
                    "--derived-lock",
                    "model.lock.json"
                ]
            )
            .unwrap()
            .operation,
            Operation::Render { .. }
        ));
        assert!(matches!(
            parse_args(
                "decode",
                &[
                    "--tokens",
                    "1,3,17",
                    "--skip-special-tokens",
                    "--gguf",
                    "model.gguf",
                    "--derived-lock",
                    "model.lock.json"
                ]
            )
            .unwrap()
            .operation,
            Operation::Decode { .. }
        ));
        assert!(matches!(
            parse_args(
                "generate",
                &[
                    "--gguf",
                    "model.gguf",
                    "--derived-lock",
                    "model.lock.json",
                    "--prompt",
                    "abc",
                    "--max-new-tokens",
                    "3",
                    "--device-index",
                    "0",
                    "--target",
                    "gfx1030",
                    "--seed",
                    "18446744073709551615",
                    "--greedy"
                ]
            )
            .unwrap()
            .operation,
            Operation::Generate(GenerateRequest {
                input: GenerationInput::Prompt(_),
                max_new_tokens: 3,
                seed: Some(u64::MAX),
                device_index: 0,
                kv_cache_encoding: None,
                ..
            })
        ));
        let low_bit_kv = parse_args(
            "generate",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--prompt",
                "abc",
                "--max-new-tokens",
                "1",
                "--device-index",
                "0",
                "--target",
                "gfx1201",
                "--kv-cache-encoding",
                "fp8-static",
                "--greedy",
            ],
        )
        .unwrap();
        assert!(matches!(
            low_bit_kv.operation,
            Operation::Generate(GenerateRequest {
                kv_cache_encoding: Some(KvCacheEncoding::Fp8E4M3FnStatic),
                ..
            })
        ));
        let benchmark = parse_args(
            "benchmark",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--lane",
                "render-tokenize",
                "--row-id",
                "host-test",
                "--model-size",
                "4B",
                "--case-id",
                "host-test",
                "--message",
                "user:abc",
                "--max-new-tokens",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
        )
        .unwrap();
        assert!(matches!(
            benchmark.operation,
            Operation::Benchmark(BenchmarkRequest {
                warmups: 3,
                measured: 10,
                ..
            })
        ));
        assert!(
            parse_args(
                "benchmark",
                &[
                    "--gguf",
                    "model.gguf",
                    "--derived-lock",
                    "model.lock.json",
                    "--nvfp4-manifest",
                    "sidecar.json"
                ]
            )
            .is_err()
        );
        let unicode = parse_args(
            "generate",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--message",
                "user:雪とGPU",
                "--thinking",
                "disabled",
                "--max-new-tokens",
                "17",
                "--device-index",
                "0",
                "--target",
                "gfx1201",
                "--greedy",
            ],
        )
        .unwrap();
        assert!(matches!(
            unicode.operation,
            Operation::Generate(GenerateRequest {
                input: GenerationInput::Messages { .. },
                max_new_tokens: 17,
                target,
                ..
            }) if target == "gfx1201"
        ));
        let image = parse_args(
            "generate",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--message",
                "user:describe",
                "--image",
                "fixture.png",
                "--max-new-tokens",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
        )
        .unwrap();
        assert!(matches!(
            image.operation,
            Operation::Generate(GenerateRequest { ref image_paths, .. })
                if image_paths == &[PathBuf::from("fixture.png")]
        ));
    }

    #[test]
    fn custom_template_flags_are_explicitly_bound_to_message_operations() {
        let request = parse_args(
            "apply-template",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--message",
                "user:hello",
                "--chat-template-file",
                "template.jinja",
                "--chat-template-digest",
                "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "--template-kwargs-json",
                r#"{"temperature":0.5}"#,
            ],
        )
        .unwrap();
        assert!(matches!(
            request.operation,
            Operation::ApplyTemplate {
                custom_template: Some(CustomTemplateSpec {
                    path,
                    digest,
                    kwargs,
                    ..
                }),
                ..
            } if path == std::path::Path::new("template.jinja")
                && digest.starts_with("sha256:")
                && kwargs.get("temperature") == Some(&Value::from(0.5))
        ));

        let missing_digest = parse_args(
            "apply-template",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--message",
                "user:hello",
                "--chat-template-file",
                "template.jinja",
            ],
        )
        .unwrap_err();
        assert!(missing_digest.contains("requires both"));

        let raw_text = parse_args(
            "input-tokens",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--text",
                "hello",
                "--chat-template-file",
                "template.jinja",
                "--chat-template-digest",
                "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            ],
        )
        .unwrap_err();
        assert!(raw_text.contains("requires --message"));

        let unsupported = parse_args(
            "tokenize",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--text",
                "hello",
                "--chat-template-file",
                "secret-template.jinja",
            ],
        )
        .unwrap_err();
        assert!(!unsupported.contains("secret-template.jinja"));
    }

    #[test]
    fn prefill_chunk_override_is_bounded_and_preserves_auto_selection() {
        let common = [
            "--gguf",
            "model.gguf",
            "--derived-lock",
            "model.lock.json",
            "--prompt",
            "abc",
            "--max-new-tokens",
            "3",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--greedy",
        ];
        let omitted = parse_args("generate", &common).unwrap();
        assert!(matches!(
            omitted.operation,
            Operation::Generate(GenerateRequest {
                prefill_chunk_tokens: None,
                ..
            })
        ));
        for value in [512_u64, 2_048, 4_096, 8_192, 16_384] {
            let mut args = common.to_vec();
            let value_string = value.to_string();
            args.extend(["--prefill-chunk-tokens", value_string.as_str()]);
            let request = parse_args("generate", &args).unwrap();
            assert!(matches!(
                request.operation,
                Operation::Generate(GenerateRequest {
                    prefill_chunk_tokens: Some(parsed),
                    ..
                }) if parsed == value
            ));
        }
        for value in ["0", "16385", "-1", "not-a-number"] {
            let mut args = common.to_vec();
            args.extend(["--prefill-chunk-tokens", value]);
            assert!(parse_args("generate", &args).is_err(), "value {value}");
        }
        assert_eq!(
            cli_prefill_chunk_candidates(Some(512), 32 * 1024 * 1024 * 1024, 10_001).unwrap(),
            [512]
        );
        assert_eq!(
            cli_prefill_chunk_candidates(None, 32 * 1024 * 1024 * 1024, 10_001).unwrap(),
            qwen_prefill_chunk_candidates(32 * 1024 * 1024 * 1024, 10_001).unwrap()
        );
    }

    #[test]
    fn kv_parser_and_resolver_preserve_auto_and_canonical_public_names() {
        let common = [
            "--gguf",
            "model.gguf",
            "--derived-lock",
            "model.lock.json",
            "--prompt",
            "abc",
            "--max-new-tokens",
            "1",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
            "--greedy",
        ];
        let omitted = parse_args("generate", &common).unwrap();
        assert!(matches!(
            omitted.operation,
            Operation::Generate(GenerateRequest {
                kv_cache_encoding: None,
                ..
            })
        ));
        for (name, expected) in [
            ("fp16", KvCacheEncoding::Fp16),
            ("kv-mxfp8-e4", KvCacheEncoding::Mxfp8E4),
            ("kv-mxfp8-e5", KvCacheEncoding::Mxfp8E5),
        ] {
            let mut args = common.to_vec();
            args.extend(["--kv-cache-encoding", name]);
            let request = parse_args("generate", &args).unwrap();
            assert!(matches!(
                request.operation,
                Operation::Generate(GenerateRequest {
                    kv_cache_encoding: Some(actual),
                    ..
                }) if actual == expected
            ));
        }
        for alias in [
            "fp8-e4-block16",
            "kv-fp8-e4m3-block16",
            "kv-fp8-e5m2-block16",
            "KV-FP8-E4-BLOCK16",
            "mxfp8-e4",
            "kv-mxfp8-e4-block32",
            "KV-MXFP8-E4",
        ] {
            let mut args = common.to_vec();
            args.extend(["--kv-cache-encoding", alias]);
            assert!(parse_args("generate", &args).is_err(), "alias {alias}");
        }

        let auto = resolve_cli_kv_cache_selection(
            None,
            "gfx1201",
            sllm_core::QWEN35_4B_FINGERPRINT,
            true,
            true,
            256,
        )
        .unwrap();
        let auto_report = kv_selection_report(auto);
        assert_eq!(auto_report["requested"], "auto");
        assert_eq!(auto_report["resolved"], "kv-mxfp8-e4");
        assert_eq!(auto_report["selection_source"], "mxfp8-e4-default");

        assert!(
            resolve_cli_kv_cache_selection(
                Some(KvCacheEncoding::Fp8E4M3Block16),
                "gfx942:sramecc+:xnack-",
                sllm_core::QWEN35_4B_FINGERPRINT,
                true,
                true,
                256,
            )
            .is_err()
        );

        for (encoding, target, physical_variant) in [
            (KvCacheEncoding::Mxfp8E4, "gfx1201", "E4M3-OCP"),
            (KvCacheEncoding::Mxfp8E5, "gfx1030", "E5M2-OCP"),
        ] {
            let resolved = resolve_cli_kv_cache_selection(
                Some(encoding),
                target,
                sllm_core::QWEN35_4B_FINGERPRINT,
                true,
                true,
                256,
            )
            .unwrap();
            let report = kv_selection_report(resolved);
            assert_eq!(report["resolved"], encoding.canonical_name());
            assert_eq!(report["selection_source"], "process-explicit");
            assert_eq!(report["physical_variant"], physical_variant);
            assert_eq!(
                report["descriptor_id"],
                format!("{}-v1", encoding.canonical_name())
            );
        }
        for target in ["gfx1030", "gfx1201", "gfx942:sramecc+:xnack-"] {
            let resolved = resolve_cli_kv_cache_selection(
                Some(KvCacheEncoding::Mxfp8E4),
                target,
                sllm_core::QWEN35_4B_FINGERPRINT,
                true,
                true,
                256,
            )
            .unwrap();
            assert_eq!(
                resolved.physical_variant(),
                Some(KvFp8PhysicalVariant::OcpE4M3Fn)
            );
        }
        for (encoding, target, fingerprint, dense, full_attention, head_dim) in [
            (
                KvCacheEncoding::Mxfp8E5,
                "gfx1201",
                sllm_core::QWEN35_4B_FINGERPRINT,
                true,
                true,
                256,
            ),
            (
                KvCacheEncoding::Mxfp8E4,
                "gfx1201",
                "wrong-model",
                true,
                true,
                256,
            ),
            (
                KvCacheEncoding::Mxfp8E4,
                "gfx1201",
                sllm_core::QWEN35_4B_FINGERPRINT,
                false,
                true,
                256,
            ),
            (
                KvCacheEncoding::Mxfp8E4,
                "gfx1201",
                sllm_core::QWEN35_4B_FINGERPRINT,
                true,
                false,
                256,
            ),
            (
                KvCacheEncoding::Mxfp8E4,
                "gfx1201",
                sllm_core::QWEN35_4B_FINGERPRINT,
                true,
                true,
                128,
            ),
        ] {
            assert!(
                resolve_cli_kv_cache_selection(
                    Some(encoding),
                    target,
                    fingerprint,
                    dense,
                    full_attention,
                    head_dim,
                )
                .is_err(),
                "MXFP8 must fail closed outside its reviewed target/model/shape scope"
            );
        }
    }

    #[test]
    fn mtp_draft_width_parse_and_admission_are_fail_closed() {
        let common = [
            "--gguf",
            "model.gguf",
            "--derived-lock",
            "model.lock.json",
            "--prompt",
            "abc",
            "--max-new-tokens",
            "3",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--greedy",
        ];
        let omitted = parse_args("generate", &common).unwrap();
        assert!(matches!(
            omitted.operation,
            Operation::Generate(GenerateRequest {
                mtp_draft_width: None,
                ..
            })
        ));
        for value in [0_u8, 1, 8] {
            let mut args = common.to_vec();
            let value_string = value.to_string();
            args.extend(["--mtp-draft-width", value_string.as_str()]);
            let request = parse_args("generate", &args).unwrap();
            assert!(matches!(
                request.operation,
                Operation::Generate(GenerateRequest {
                    mtp_draft_width: Some(parsed),
                    ..
                }) if parsed == value
            ));
        }
        for value in ["9", "255", "-1", "not-a-number"] {
            let mut args = common.to_vec();
            args.extend(["--mtp-draft-width", value]);
            assert!(parse_args("generate", &args).is_err(), "value {value}");
        }

        let sampling = SamplingParametersV1::new(0.0, 1.0, 0.0, 0.0).unwrap();
        let auto = resolve_cli_mtp_plan(
            None,
            false,
            false,
            false,
            "gfx1201",
            KvCacheEncoding::Fp16,
            sampling,
            sllm_core::QWEN35_4B_FINGERPRINT,
        )
        .unwrap();
        assert_eq!(auto.selection, "auto");
        assert!(auto.enabled);
        assert_eq!(auto.effective_width, Some(1));
        let auto_gfx942 = resolve_cli_mtp_plan(
            None,
            false,
            false,
            false,
            "gfx942",
            KvCacheEncoding::Fp16,
            sampling,
            sllm_core::QWEN35_4B_FINGERPRINT,
        )
        .unwrap();
        assert_eq!(auto_gfx942.selection, "auto");
        assert!(!auto_gfx942.enabled);
        assert_eq!(auto_gfx942.effective_width, None);

        let forced = resolve_cli_mtp_plan(
            Some(8),
            false,
            false,
            false,
            "gfx942",
            KvCacheEncoding::Fp16,
            sampling,
            sllm_core::QWEN35_4B_FINGERPRINT,
        )
        .unwrap();
        assert_eq!(forced.selection, "forced");
        assert!(forced.enabled);
        assert_eq!(forced.effective_width, Some(8));

        for target in ["gfx1201", "gfx942"] {
            let embedded_fp8_forced = resolve_cli_mtp_plan(
                Some(2),
                false,
                false,
                true,
                target,
                KvCacheEncoding::Fp8E4M3Fn,
                sampling,
                sllm_core::QWEN35_4B_FINGERPRINT,
            )
            .unwrap();
            assert_eq!(embedded_fp8_forced.selection, "forced");
            assert!(embedded_fp8_forced.enabled);
            assert_eq!(embedded_fp8_forced.effective_width, Some(2));
        }

        let target_only = resolve_cli_mtp_plan(
            Some(0),
            true,
            true,
            true,
            "unsupported",
            KvCacheEncoding::Nvfp4,
            SamplingParametersV1::new(1.0, 1.0, 0.0, 0.0).unwrap(),
            "unreviewed",
        )
        .unwrap();
        assert_eq!(target_only.selection, "target-only");
        assert!(!target_only.enabled);
        assert_eq!(target_only.effective_width, None);

        for (target, kv, expected) in [
            ("gfx1030", KvCacheEncoding::Fp16, "exact gfx1201 or gfx942"),
            ("gfx942", KvCacheEncoding::Fp8E4M3Fn, "FP16 KV"),
        ] {
            let error = resolve_cli_mtp_plan(
                Some(1),
                false,
                false,
                false,
                target,
                kv,
                sampling,
                sllm_core::QWEN35_4B_FINGERPRINT,
            )
            .unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
        for kv in [
            KvCacheEncoding::Fp16,
            KvCacheEncoding::Fp8E4M3FnStatic,
            KvCacheEncoding::Nvfp4,
        ] {
            let error = resolve_cli_mtp_plan(
                Some(1),
                false,
                false,
                true,
                "gfx942",
                kv,
                sampling,
                sllm_core::QWEN35_4B_FINGERPRINT,
            )
            .unwrap_err();
            assert!(error.contains("dynamic FP8"), "{error}");
        }
        let error = resolve_cli_mtp_plan(
            Some(1),
            false,
            true,
            true,
            "gfx942",
            KvCacheEncoding::Fp8E4M3Fn,
            sampling,
            sllm_core::QWEN35_4B_FINGERPRINT,
        )
        .unwrap_err();
        assert!(error.contains("sidecar"), "{error}");
    }

    #[test]
    fn mtp_state_capacity_adds_bounded_slack_without_truncating_budget() {
        for width in [1_u8, 2, 8] {
            assert_eq!(
                cli_state_capacity_with_mtp_slack(17, Some(width)).unwrap(),
                (17 + u64::from(width), u64::from(width))
            );
        }
        assert_eq!(
            cli_state_capacity_with_mtp_slack(17, None).unwrap(),
            (17, 0)
        );
        assert!(cli_state_capacity_with_mtp_slack(u64::MAX, Some(1)).is_err());
        assert!(
            cli_state_capacity_with_mtp_slack(QWEN_RUNTIME_MAX_CONTEXT_TOKENS, Some(1)).is_err()
        );
    }

    #[test]
    fn greedy_controller_excludes_stop_tokens_and_stops_exactly_at_budget() {
        let policy = qwen_stop_policy();

        let mut first_stop = SequenceExecution::new([248046]);
        let outcome = run_greedy_generation(&mut first_stop, &policy, 3, &[1, 3, 17]).unwrap();
        assert_eq!(outcome.report.generated_token_ids(), &[248046]);
        assert!(outcome.report.visible_token_ids().is_empty());
        assert!(outcome.report.decode_input_token_ids().is_empty());
        assert_eq!(outcome.report.stop_token_id(), Some(248046));
        assert_eq!(outcome.decode_steps, 0);

        let mut second_stop = SequenceExecution::new([7, 248044]);
        let outcome = run_greedy_generation(&mut second_stop, &policy, 3, &[1, 3, 17]).unwrap();
        assert_eq!(outcome.report.generated_token_ids(), &[7, 248044]);
        assert_eq!(outcome.report.visible_token_ids(), &[7]);
        assert_eq!(outcome.report.decode_input_token_ids(), &[7]);
        assert_eq!(second_stop.decode_inputs, [7]);
        assert_eq!(outcome.report.stop_token_id(), Some(248044));

        for budget in [1_u32, 3, 17, 255, 256, 257] {
            let mut executor = SequenceExecution::new(std::iter::repeat_n(7, budget as usize));
            let outcome =
                run_greedy_generation(&mut executor, &policy, budget, &[1, 3, 17]).unwrap();
            assert_eq!(outcome.report.generated_token_ids().len(), budget as usize);
            assert_eq!(outcome.report.visible_token_ids().len(), budget as usize);
            assert_eq!(
                outcome.report.decode_input_token_ids().len(),
                budget.saturating_sub(1) as usize
            );
            assert_eq!(outcome.report.reason_token(), Some("max_new_tokens"));
            assert_eq!(outcome.decode_steps, budget.saturating_sub(1));
        }
    }

    #[test]
    fn greedy_controller_rejects_negative_or_exhausted_executor_output() {
        let policy = qwen_stop_policy();
        let mut negative = SequenceExecution::new([-1]);
        assert!(run_greedy_generation(&mut negative, &policy, 3, &[1]).is_err());

        let mut exhausted = SequenceExecution::new([7]);
        assert!(run_greedy_generation(&mut exhausted, &policy, 3, &[1]).is_err());
    }

    #[test]
    fn normal_generation_and_timed_generation_have_identical_token_semantics() {
        let policy = qwen_stop_policy();
        let mut normal = SequenceExecution::new([7, 8, 248044]);
        let expected = run_greedy_generation(&mut normal, &policy, 5, &[1, 3, 17]).unwrap();
        let mut timed = SequenceExecution::new([7, 8, 248044]);
        let mut timeline = BenchmarkTimeline::new(0);
        let actual = run_greedy_generation_timed(
            &mut timed,
            &policy,
            5,
            &[1, 3, 17],
            Some((&mut timeline, MonotonicClock::start())),
        )
        .unwrap();
        assert_eq!(expected.report, actual.report);
        assert_eq!(expected.decode_steps, actual.decode_steps);
    }

    #[test]
    fn temperature_zero_service_matches_phase3_greedy_token_semantics() {
        let policy = qwen_stop_policy();
        let mut phase3 = SequenceExecution::new([7, 8, 9]);
        let expected = run_greedy_generation(&mut phase3, &policy, 3, &[1, 3, 17]).unwrap();

        let frontend = NumericFrontend;
        let service = GenerationServiceV1::new(&frontend, None, &policy).unwrap();
        let config = GenerationConfigV1::new(3, SamplingParametersV1::greedy(), vec![]).unwrap();
        let mut a3 = SequenceExecution::new([7, 8, 9]);
        let actual = service
            .generate_tokens(
                &mut a3,
                &[1, 3, 17],
                &config,
                &GenerationCancellationV1::new(),
                &mut NeverRandom,
            )
            .unwrap();
        assert_eq!(
            actual.generated_token_ids(),
            expected.report.generated_token_ids()
        );
        assert_eq!(
            actual.visible_token_ids(),
            expected.report.visible_token_ids()
        );
        assert_eq!(
            actual.decode_input_token_ids(),
            expected.report.decode_input_token_ids()
        );
        assert_eq!(actual.decode_steps(), expected.decode_steps);
        assert_eq!(actual.finish_reason().as_str(), "length");
        assert!(!a3.cancelled);
    }

    #[test]
    fn tiny_backend_executes_all_success_entrances() {
        let cases = [
            ("verify-model", vec!["--gguf", "x", "--derived-lock", "y"]),
            (
                "tokenize",
                vec!["--gguf", "x", "--derived-lock", "y", "--text", "abc"],
            ),
            (
                "render",
                vec![
                    "--gguf",
                    "x",
                    "--derived-lock",
                    "y",
                    "--message",
                    "user:abc",
                    "--thinking",
                    "disabled",
                ],
            ),
            (
                "decode",
                vec![
                    "--gguf",
                    "x",
                    "--derived-lock",
                    "y",
                    "--tokens",
                    "1,3,17",
                    "--skip-special-tokens",
                ],
            ),
            (
                "generate",
                vec![
                    "--gguf",
                    "x",
                    "--derived-lock",
                    "y",
                    "--prompt",
                    "abc",
                    "--max-new-tokens",
                    "3",
                    "--device-index",
                    "0",
                    "--target",
                    "gfx1030",
                    "--greedy",
                ],
            ),
            (
                "benchmark",
                vec![
                    "--gguf",
                    "x",
                    "--derived-lock",
                    "y",
                    "--lane",
                    "direct",
                    "--row-id",
                    "host-test",
                    "--model-size",
                    "4B",
                    "--case-id",
                    "host-test",
                    "--input-token-ids",
                    "1,3,17",
                    "--max-new-tokens",
                    "3",
                    "--device-index",
                    "0",
                    "--target",
                    "gfx1030",
                    "--greedy",
                ],
            ),
        ];
        for (command, args) in cases {
            let request = parse_args(command, &args).unwrap();
            let output = if command == "benchmark" {
                execute_with_timing(
                    command,
                    request.operation,
                    &TinyBackend,
                    Some(BenchmarkTiming::start()),
                )
                .unwrap()
            } else {
                execute(command, request.operation, &TinyBackend).unwrap()
            };
            let document: Value = serde_json::from_str(&output).unwrap();
            if command == "benchmark" {
                assert_eq!(document["lane"], "direct");
                assert_eq!(
                    document["benchmark_schema_version"],
                    "engine-performance-direct-v2"
                );
                assert_eq!(document["config"]["lane"], "direct");
                assert_eq!(document["config"]["kv_cache_encoding"], "fp16");
            } else {
                assert_eq!(document["command"], command);
                assert_eq!(document["result"]["kind"], command);
                assert_eq!(document["state"], "PASS");
            }
        }
    }

    #[test]
    fn benchmark_public_cli_keeps_the_render_tokenize_lane() {
        assert_eq!(
            BenchmarkLane::RenderTokenize.schema_version(),
            "engine-performance-render-v1"
        );
        let cli_request = parse_args(
            "benchmark",
            &[
                "--gguf",
                "x",
                "--derived-lock",
                "y",
                "--lane",
                "render-tokenize",
                "--model-size",
                "4B",
                "--message",
                "user:abc",
                "--max-new-tokens",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
        )
        .unwrap();
        assert!(matches!(
            cli_request.operation,
            Operation::Benchmark(BenchmarkRequest {
                lane: BenchmarkLane::RenderTokenize,
                ..
            })
        ));
    }

    #[test]
    fn benchmark_direct_lane_accepts_large_pretokenized_input_and_kv_encoding() {
        assert_eq!(
            BenchmarkLane::Direct.schema_version(),
            "engine-performance-direct-v2"
        );
        let input_ids = std::iter::repeat_n("23066", 10_001)
            .collect::<Vec<_>>()
            .join(",");
        let request = parse_args(
            "benchmark",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--lane",
                "direct",
                "--model-size",
                "4B",
                "--case-id",
                "long-10001",
                "--input-token-ids",
                input_ids.as_str(),
                "--max-new-tokens",
                "2",
                "--device-index",
                "0",
                "--target",
                "gfx942",
                "--kv-cache-encoding",
                "fp8",
                "--greedy",
            ],
        )
        .unwrap();
        assert!(matches!(
            request.operation,
            Operation::Benchmark(BenchmarkRequest {
                lane: BenchmarkLane::Direct,
                input: BenchmarkInput::TokenIds(ref ids),
                kv_cache_encoding: Some(KvCacheEncoding::Fp8E4M3Fn),
                ..
            }) if ids.len() == 10_001 && ids.as_slice().iter().all(|id| *id == 23_066)
        ));
    }

    #[test]
    fn benchmark_direct_lane_reads_bounded_pretokenized_input_file() {
        let path = std::env::temp_dir().join(format!(
            "sllm-phase49-input-token-ids-{}-{}.csv",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&path, "23066,23066,23066\n").unwrap();
        let path_text = path.to_str().unwrap().to_owned();
        let request = parse_args(
            "benchmark",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--lane",
                "direct",
                "--model-size",
                "4B",
                "--input-token-ids-file",
                path_text.as_str(),
                "--max-new-tokens",
                "2",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
        )
        .unwrap();
        fs::remove_file(path).unwrap();
        assert!(matches!(
            request.operation,
            Operation::Benchmark(BenchmarkRequest {
                lane: BenchmarkLane::Direct,
                input: BenchmarkInput::TokenIds(ref ids),
                ..
            }) if ids.as_slice() == [23_066, 23_066, 23_066]
        ));
    }

    #[test]
    fn benchmark_direct_phase49_controls_parse_and_normal_generate_stays_bounded() {
        let request = parse_args(
            "benchmark",
            &[
                "--gguf",
                "model.gguf",
                "--derived-lock",
                "model.lock.json",
                "--lane",
                "direct",
                "--model-size",
                "4B",
                "--case-id",
                "decode-20000",
                "--input-token-ids",
                "1,3,17",
                "--max-new-tokens",
                "20000",
                "--ignore-eos",
                "--context-length",
                "131072",
                "--completion-timeout-seconds",
                "3600",
                "--prefill-chunk-tokens",
                "16384",
                "--warmups",
                "1",
                "--measured",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
        )
        .unwrap();
        assert!(matches!(
            request.operation,
            Operation::Benchmark(BenchmarkRequest {
                lane: BenchmarkLane::Direct,
                max_new_tokens: 20_000,
                ignore_eos: true,
                context_length: Some(131_072),
                completion_timeout_seconds: Some(3_600),
                prefill_chunk_tokens: Some(16_384),
                warmups: 1,
                measured: 3,
                ..
            })
        ));

        assert!(
            parse_args(
                "generate",
                &[
                    "--gguf",
                    "model.gguf",
                    "--derived-lock",
                    "model.lock.json",
                    "--prompt",
                    "abc",
                    "--max-new-tokens",
                    "20000",
                    "--device-index",
                    "0",
                    "--target",
                    "gfx1030",
                    "--greedy",
                ],
            )
            .is_err()
        );
    }

    #[test]
    fn benchmark_phase49_controls_reject_wrong_scope_and_invalid_protocols() {
        let render_common = [
            "--gguf",
            "model.gguf",
            "--derived-lock",
            "model.lock.json",
            "--lane",
            "render-tokenize",
            "--model-size",
            "4B",
            "--message",
            "user:abc",
            "--max-new-tokens",
            "3",
            "--device-index",
            "0",
            "--target",
            "gfx1030",
            "--greedy",
        ];
        let mut ignore_eos = render_common.to_vec();
        ignore_eos.push("--ignore-eos");
        assert!(parse_args("benchmark", &ignore_eos).is_err());

        let mut long_render = render_common.to_vec();
        let max_tokens_index = long_render.iter().position(|value| *value == "3").unwrap();
        long_render[max_tokens_index] = "20000";
        assert!(parse_args("benchmark", &long_render).is_err());

        let mut chunked_render = render_common.to_vec();
        chunked_render.extend(["--prefill-chunk-tokens", "16384"]);
        assert!(parse_args("benchmark", &chunked_render).is_err());
        assert!(validate_benchmark_protocol(3, 10).is_ok());
        assert!(validate_benchmark_protocol(1, 3).is_ok());
        assert!(validate_benchmark_protocol(1, 10).is_err());
        assert!(benchmark_state_capacity(32, 20_000, Some(131_072)).is_ok());
        assert!(benchmark_state_capacity(100, 20_000, Some(20_000)).is_err());
        assert!(benchmark_completion_timeout(Some(86_400)).is_ok());
        assert!(benchmark_completion_timeout(Some(86_401)).is_err());
    }

    #[test]
    fn benchmark_ignore_eos_runs_through_qwen_stop_ids_to_max_budget() {
        let policy = qwen_stop_policy();
        let ignored_policy = benchmark_stop_policy(&policy, true);
        let mut executor = SequenceExecution::new([248_046, 248_044, 7]);
        let outcome =
            run_greedy_generation(&mut executor, &ignored_policy, 3, &[1, 3, 17]).unwrap();
        assert_eq!(outcome.report.generated_token_ids(), &[248_046, 248_044, 7]);
        assert_eq!(outcome.report.visible_token_ids(), &[248_046, 248_044, 7]);
        assert_eq!(outcome.report.decode_input_token_ids(), &[248_046, 248_044]);
        assert_eq!(outcome.report.reason_token(), Some("max_new_tokens"));
        assert_eq!(outcome.decode_steps, 2);
    }

    #[test]
    fn benchmark_lanes_reject_cross_input_shapes_and_missing_direct_ids() {
        let common = ["--gguf", "model.gguf", "--derived-lock", "model.lock.json"];
        let direct_with_message = [
            "--gguf",
            "model.gguf",
            "--derived-lock",
            "model.lock.json",
            "--lane",
            "direct",
            "--model-size",
            "4B",
            "--message",
            "user:abc",
            "--input-token-ids",
            "1,3,17",
            "--max-new-tokens",
            "2",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--greedy",
        ];
        assert!(parse_args("benchmark", &direct_with_message).is_err());

        let render_with_ids = [
            common[0],
            common[1],
            common[2],
            common[3],
            "--lane",
            "render-tokenize",
            "--model-size",
            "4B",
            "--message",
            "user:abc",
            "--input-token-ids",
            "1,3,17",
            "--max-new-tokens",
            "2",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--greedy",
        ];
        assert!(parse_args("benchmark", &render_with_ids).is_err());

        let render_with_low_bit_kv = [
            common[0],
            common[1],
            common[2],
            common[3],
            "--lane",
            "render-tokenize",
            "--model-size",
            "4B",
            "--message",
            "user:abc",
            "--max-new-tokens",
            "2",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--kv-cache-encoding",
            "fp8",
            "--greedy",
        ];
        assert!(parse_args("benchmark", &render_with_low_bit_kv).is_err());

        let direct_without_ids = [
            common[0],
            common[1],
            common[2],
            common[3],
            "--lane",
            "direct",
            "--model-size",
            "4B",
            "--max-new-tokens",
            "2",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--greedy",
        ];
        assert!(parse_args("benchmark", &direct_without_ids).is_err());
    }

    #[test]
    fn malformed_and_cross_command_arguments_fail_closed() {
        assert!(parse_args("tokenize", &["--lock", "x", "--cache", "y"]).is_err());
        assert!(
            parse_args(
                "decode",
                &["--lock", "x", "--cache", "y", "--tokens", "1,,2"]
            )
            .is_err()
        );
        for arguments in [
            vec!["--lock", "x", "--cache", "y", "--prompt", "x"],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "x",
                "--message",
                "user:y",
                "--max-new-tokens",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "x",
                "--max-new-tokens",
                "0",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "x",
                "--max-new-tokens",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx9999",
                "--greedy",
            ],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "x",
                "--thinking",
                "disabled",
                "--max-new-tokens",
                "3",
                "--device-index",
                "0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
            vec![
                "--lock",
                "x",
                "--cache",
                "y",
                "--prompt",
                "",
                "--max-new-tokens",
                "+3",
                "--device-index",
                "+0",
                "--target",
                "gfx1030",
                "--greedy",
            ],
        ] {
            assert!(parse_args("generate", &arguments).is_err());
        }
        assert!(
            parse_args(
                "render",
                &["--lock", "x", "--cache", "y", "--message", "tool:x"]
            )
            .is_err()
        );
        assert!(
            parse_args(
                "verify-model",
                &["--lock", "x", "--lock", "z", "--cache", "y"]
            )
            .is_err()
        );
        assert!(
            parse_args(
                "decode",
                &[
                    "--lock",
                    "x",
                    "--cache",
                    "y",
                    "--tokens",
                    "1",
                    "--skip-special-tokens",
                    "--skip-special-tokens"
                ]
            )
            .is_err()
        );
        assert!(
            parse_args(
                "render",
                &[
                    "--lock",
                    "x",
                    "--cache",
                    "y",
                    "--message",
                    "user:x",
                    "--thinking",
                    "enabled",
                    "--thinking",
                    "disabled"
                ]
            )
            .is_err()
        );
        assert!(
            parse_args(
                "verify-model",
                &["--lock", "x", "--cache", "y", "--text", "x"]
            )
            .is_err()
        );
    }

    #[test]
    fn token_boundaries_include_non_aligned_values() {
        assert_eq!(parse_token_ids("1,3,17").unwrap().as_slice(), &[1, 3, 17]);
        assert!(parse_token_ids("").is_err());
        assert!(parse_token_ids("4294967296").is_err());
        assert!(parse_token_ids("+1").is_err());
    }

    #[test]
    fn serialized_success_uses_the_versioned_closed_envelope() {
        let lock = sllm_core::parse_model_lock(include_bytes!(
            "../../../docs/models/locks/qwen3.5-4b-bf16.json"
        ))
        .unwrap();
        let identity = ModelIdentity {
            repo_id: lock.model().repo_id.clone(),
            resolved_revision: lock.model().resolved_revision.clone(),
            lock_fingerprint: lock.fingerprint().to_owned(),
        };
        for (command, result) in [
            (
                "tokenize",
                json!({"kind": "tokenize", "count": 3, "token_ids": [1, 3, 17]}),
            ),
            ("render", json!({"kind": "render", "text": "prompt"})),
            ("decode", json!({"kind": "decode", "text": "text"})),
            (
                "generate",
                json!({"kind": "generate", "generated_token_ids": [1]}),
            ),
        ] {
            let output = serialize_report(command, &identity, result).unwrap();
            assert!(!output.contains('\n'));
            let document: Value = serde_json::from_str(&output).unwrap();
            assert_eq!(document.as_object().unwrap().len(), 6);
            assert_eq!(document["schema_version"], REPORT_SCHEMA);
            assert_eq!(document["command"], command);
            assert_eq!(document["state"], "PASS");
            assert_eq!(document["result"]["kind"], command);
            assert_eq!(document["scope"]["gpu_execution"], command == "generate");
        }
    }

    #[test]
    fn embedding_and_rerank_reports_mark_model_execution_scope() {
        let identity = ModelIdentity {
            repo_id: "model".to_owned(),
            resolved_revision: "revision".to_owned(),
            lock_fingerprint: "fingerprint".to_owned(),
        };
        for command in ["embeddings", "rerank"] {
            let output = serialize_report(command, &identity, json!({"kind": command})).unwrap();
            let document: Value = serde_json::from_str(&output).unwrap();
            assert_eq!(document["scope"]["gpu_execution"], true);
            assert_eq!(document["scope"]["model_execution"], true);
            assert_eq!(document["scope"]["generation"], false);
        }
    }

    #[test]
    fn serialized_generate_report_contains_dispatch_audit_fields() {
        let identity = ModelIdentity {
            repo_id: "Qwen/Qwen3.5-4B".to_owned(),
            resolved_revision: "8".repeat(40),
            lock_fingerprint: format!("sha256:{}", "3".repeat(64)),
        };
        let result = json!({
            "kind": "generate",
            "input_kind": "prompt",
            "input_token_ids": [9419],
            "generated_token_ids": [220],
            "visible_token_ids": [220],
            "decode_input_token_ids": [],
            "output_text": "Hello",
            "stop_reason": {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": null},
            "execution": {
                "selected_backend": "hip",
                "target": "gfx1030",
                "device_index": 0,
                "model_fingerprint": identity.lock_fingerprint,
                "plan_digest": format!("sha256:{}", "9".repeat(64)),
                "prefill_tokens": 1,
                "decode_steps": 0,
                "fallback_used": false,
                "submission_count": 1,
                "kernel_dispatch_count": 1,
                "segment_count": 1,
                "boundary_count": 1,
                "all_dispatches_hip": true,
                "weight_encoding": "bf16",
                "fp8_provider": null,
            },
            "timing_ns": 1,
            "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
        });
        let document: Value =
            serde_json::from_str(&serialize_report("generate", &identity, result).unwrap())
                .unwrap();
        assert_eq!(document["scope"]["offline"], true);
        assert_eq!(document["scope"]["generation"], true);
        assert_eq!(document["result"]["execution"]["selected_backend"], "hip");
        assert_eq!(document["result"]["execution"]["submission_count"], 1);
        assert_eq!(document["result"]["execution"]["kernel_dispatch_count"], 1);
        assert_eq!(document["result"]["execution"]["segment_count"], 1);
        assert_eq!(document["result"]["execution"]["boundary_count"], 1);
        assert_eq!(document["result"]["execution"]["all_dispatches_hip"], true);
        assert_eq!(document["result"]["execution"]["weight_encoding"], "bf16");
        assert_eq!(document["result"]["execution"]["fp8_provider"], Value::Null);
    }
}
