use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use sllm_core::{
    Backend, CheckpointIdentity, CheckpointStore, ExecutionSessionRequest,
    GEMMA4_12B_IT_FINGERPRINT, Gemma4ModelLock, Gemma4MoeExecutionOutput,
    Gemma4MoeExecutionRequest, Gemma4MoeResidentModel, Gemma4MtpModelLock, Gemma4MtpResidentModel,
    Gemma4ResidentModel, KvCacheEncoding, KvCacheSelection, KvCacheSelectionRequest,
    KvCacheSelectionSource, KvFp8PhysicalVariant, MINISTRAL3_GRAPH_MAX_CONTEXT,
    MINISTRAL3_MODEL_ALIAS, MINISTRAL3_MODEL_LOCK_FINGERPRINT, Ministral3ModelLock,
    Ministral3ResidentModel, ModelLock, OsSamplingRandom, QWEN_RUNTIME_MAX_CONTEXT_TOKENS,
    QwenComponentSelection, QwenExecutionRequest, QwenMultimodalImageEmbedding,
    QwenMultimodalPrompt, QwenResidentModel, QwenVisionExecutionInput, QwenVisionResidentModel,
    ReviewedModelLock, SamplingParametersV1, SessionCheckpoint, VerifiedCache,
    VerifiedGgufGemma4Moe, VerifiedGgufGemmaSource, VerifiedGgufQwen35Moe,
    VerifiedGgufWeightSource, VerifiedMinistral3WeightSource, WeightClassification,
    assemble_gguf_qwen35_multimodal_prompt, assemble_qwen35_multimodal_prompt,
    build_gemma4_moe_gguf_graph, build_gemma4_moe_resident_weight_load_plan,
    build_gemma4_mtp_graph, build_gguf_gemma4_moe_weight_load_plan,
    build_gguf_qwen35_moe_weight_load_plan, build_ministral3_weight_load_plan,
    build_qwen35_fp8_fnuz_graph, build_qwen35_fp8_graph, build_qwen35_gguf_fp8_graph,
    build_qwen35_gguf_moe_execution_graph, build_qwen35_graph,
    build_qwen35_graph_with_kv_cache_encoding, build_qwen35_graph_with_kv_cache_selection,
    build_qwen35_mtp_graph, build_qwen35_multimodal_graph, build_qwen35_nvfp4_graph,
    build_verified_gemma4_mtp_weight_load_plan, build_verified_gguf_gemma_weight_load_plan,
    build_verified_gguf_qwen_weight_load_plan, build_verified_gguf_qwen35_vision_manifest,
    build_verified_qwen_component_weight_load_plan, build_verified_qwen35_vision_manifest,
    builtin_reviewed_model_lock, open_and_verify_official_ministral3_gguf,
    parse_gemma4_mtp_model_lock, parse_ministral3_model_lock, qwen_graph_memory_estimate,
    qwen_prefill_chunk_candidates, qwen35_moe_generation_stop_policy, read_derived_gguf_lock,
    resolve_kv_cache_selection, verify_derived_gguf, verify_fp8_sidecar, verify_gguf_gemma4_moe,
    verify_gguf_gemma4_mtp, verify_gguf_qwen35_moe, verify_nvfp4_sidecar,
};
use sllm_frontend::{
    BoundedImageBytesV1, ChatTemplateRendererV1, DecodeModeV1, Gemma4MoeChatTemplateV1,
    Gemma4MtpGenerationExecutorV1, GenerationCancellationV1, GenerationConfigV1,
    GenerationExecutorV1, GenerationInputV1 as ServiceGenerationInputV1, GenerationReportV1,
    GenerationServiceError, GenerationServiceV1, GenerationStepV1, GenerationStopControllerV1,
    GenerationStopPolicyV1, GenericTemplateInputV1, GenericTemplateMessagesInputV1,
    GenericTemplateProviderV1, InputTokenCountInputV1, Ministral3TextFrontendV1,
    ProcessedVisionInputV1, Qwen35ChatMessageV1, Qwen35ChatTemplateV1, Qwen35RenderOptionsV1,
    Qwen35VisionProcessorV1, QwenMtpGenerationExecutorV1, SpeculativeGenerationAdapterV1,
    ThinkingModeV1, TokenIdsV1, TokenPieceV1, TokenizerFrontendV1, TokenizerUtilityServiceV1,
    gemma4_generation_stop_policy, gemma4_moe_generation_stop_policy,
    ministral3_generation_stop_policy,
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
use crate::chat::{
    ChatBackendErrorV1, ChatFinishReasonV1, ChatGenerationRequestV1, ChatGenerationResultV1,
    ChatThinkingModeV1,
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
const GEMMA4_MTP_CONTEXT_TOKENS: u64 = 2_048;
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
// The reviewed Gemma 4 MoE sliding-ring contract permits a single prefill
// transition up to the 1,024-token window.  Longer prompts are continued as
// one-token transitions on the same request-local KV state.
const GEMMA4_MOE_PREFILL_CHUNK_TOKENS: u64 = 1_024;

fn gemma4_moe_checkpoint_suffix<'a>(
    input: &'a [u32],
    prefix: &[u32],
) -> Result<&'a [u32], ChatBackendErrorV1> {
    if prefix.is_empty() || input.len() <= prefix.len() || input[..prefix.len()] != *prefix {
        return Err(ChatBackendErrorV1::CheckpointUnavailable);
    }
    Ok(&input[prefix.len()..])
}

fn gemma4_moe_checkpoint_visible_text(text: &str, reverse_prompts: &[String]) -> String {
    let end = reverse_prompts
        .iter()
        .filter(|prompt| !prompt.is_empty())
        .filter_map(|prompt| text.find(prompt))
        .min()
        .unwrap_or(text.len());
    text[..end].to_owned()
}

fn validate_gemma4_moe_chat_kv_cache_encoding(
    encoding: Option<KvCacheEncoding>,
) -> Result<(), &'static str> {
    match encoding {
        None | Some(KvCacheEncoding::Fp8E4M3FnStatic) => Ok(()),
        Some(_) => Err(
            "Gemma 4 MoE chat uses its fixed static FP8 E4M3 KV contract; only --kv-cache-encoding fp8-static is supported",
        ),
    }
}

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

/// Generation-service adapter for the reviewed Gemma 4 MoE executor.
///
/// The executor intentionally exposes only device Argmax today.  Keeping the
/// adapter explicit makes the unsupported sampling surface fail closed instead
/// of silently pretending that a CPU logits path is available.
struct CliGemma4MoeExecutor {
    inner: Gemma4MoeExecutionRequest,
    prefilled: bool,
    /// A restored checkpoint already owns the historical prefix.  Its graph
    /// is intentionally still a fresh start-position-zero graph; continuation
    /// input must therefore be submitted as one-token transitions.
    restored: bool,
    restored_prefix_len: usize,
    published_since_transition: bool,
    submission_count: u64,
    kernel_dispatch_count: u64,
    segment_count: u64,
    boundary_count: u64,
    fallback_used: bool,
    target: Option<String>,
}

impl CliGemma4MoeExecutor {
    fn new(inner: Gemma4MoeExecutionRequest) -> Self {
        Self::new_with_mode(inner, false)
    }

    fn new_restored_with_prefix(inner: Gemma4MoeExecutionRequest, prefix_len: usize) -> Self {
        let mut executor = Self::new_with_mode(inner, true);
        executor.restored_prefix_len = prefix_len;
        executor
    }

    fn new_with_mode(inner: Gemma4MoeExecutionRequest, restored: bool) -> Self {
        Self {
            inner,
            prefilled: false,
            restored,
            restored_prefix_len: 0,
            published_since_transition: false,
            submission_count: 0,
            kernel_dispatch_count: 0,
            segment_count: 0,
            boundary_count: 0,
            fallback_used: false,
            target: None,
        }
    }

    fn state_image(&self) -> Result<sllm_core::Gemma4MoeStateImageV1, String> {
        self.inner
            .export_state_image()
            .map_err(|error| format!("Gemma 4 MoE state image export failed: {error}"))
    }

    fn absorb(&mut self, output: &Gemma4MoeExecutionOutput) {
        let audit = output.audit();
        self.submission_count = self
            .submission_count
            .saturating_add(audit.submission_count());
        self.kernel_dispatch_count = self
            .kernel_dispatch_count
            .saturating_add(audit.kernel_dispatch_count());
        self.segment_count = self.segment_count.saturating_add(audit.segment_count());
        self.boundary_count = self.boundary_count.saturating_add(audit.boundary_count());
        self.fallback_used |= audit.fallback_used();
        if self.target.is_none() {
            self.target = Some(audit.target().to_owned());
        }
    }

    fn audit_json(&self, requested_target: &str) -> Result<Value, String> {
        let target = self
            .target
            .as_deref()
            .ok_or_else(|| "Gemma 4 MoE execution did not publish a dispatch audit".to_owned())?;
        if target != requested_target || self.fallback_used {
            return Err("Gemma 4 MoE dispatch audit is not exact HIP/no-fallback".to_owned());
        }
        Ok(json!({
            "selected_backend": "hip",
            "target": target,
            "submission_count": self.submission_count,
            "kernel_dispatch_count": self.kernel_dispatch_count,
            "segment_count": self.segment_count,
            "boundary_count": self.boundary_count,
            "fallback_used": self.fallback_used,
            "all_dispatches_hip": true,
        }))
    }
}

fn gemma4_moe_prefill_terminal_argmax(
    token_ids: &[i32],
    expected_rows: usize,
) -> Result<i32, GenerationServiceError> {
    if token_ids.len() != expected_rows {
        return Err(GenerationServiceError::Execution(format!(
            "Gemma 4 MoE prefill argmax row count differs: expected {expected_rows}, got {}",
            token_ids.len()
        )));
    }
    token_ids
        .last()
        .copied()
        .ok_or(GenerationServiceError::MissingDeviceArgmax)
}

impl GenerationExecutorV1 for CliGemma4MoeExecutor {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        _include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if self.prefilled {
            return Err(GenerationServiceError::Execution(
                "Gemma 4 MoE prefill was requested twice".to_owned(),
            ));
        }
        if input_token_ids.is_empty() {
            return Err(GenerationServiceError::EmptyPromptTokens);
        }
        let mut token = 0_i32;
        let continuation_start = if self.restored {
            // The restored request is quiescent at the checkpoint boundary.
            // Feed every suffix token through execute_next so no historical
            // token is re-appended to the imported KV image.
            self.restored_prefix_len
        } else {
            let first_chunk_len = input_token_ids
                .len()
                .min(GEMMA4_MOE_PREFILL_CHUNK_TOKENS as usize);
            let ids = input_token_ids[..first_chunk_len]
                .iter()
                .map(|token| {
                    i32::try_from(*token).map_err(|_| GenerationServiceError::TokenIdOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output = self
                .inner
                .execute(&ids)
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
            // Wide prefill publishes one device Argmax per input row.  The
            // generation service consumes only the terminal row, while the
            // exact row count remains part of the fail-closed contract.
            token = gemma4_moe_prefill_terminal_argmax(output.token_ids(), ids.len())?;
            self.absorb(&output);
            // A prompt continuation is still part of prefill and uses the
            // request's same opaque KV states. It remains request-local until
            // the final prefill argmax is handed to the generation service.
            first_chunk_len
        };
        for prompt_token in &input_token_ids[continuation_start..] {
            let prompt_token = i32::try_from(*prompt_token)
                .map_err(|_| GenerationServiceError::TokenIdOverflow)?;
            let output = self
                .inner
                .execute_next(&[prompt_token])
                .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
            if output.token_ids().len() != 1 {
                return Err(GenerationServiceError::Execution(
                    "Gemma 4 MoE prompt continuation published a non-singleton argmax".to_owned(),
                ));
            }
            token = output
                .token_ids()
                .last()
                .copied()
                .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
            self.absorb(&output);
        }
        self.prefilled = true;
        // The service may publish the returned argmax immediately after this
        // method returns. Treat the transition as externally visible already
        // so a SIGINT in that handoff cannot rewind a token that was exposed.
        self.published_since_transition = true;
        Ok(GenerationStepV1::new(
            u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            None,
        ))
    }

    fn decode(
        &mut self,
        token_id: u32,
        _include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if !self.prefilled {
            return Err(GenerationServiceError::Execution(
                "Gemma 4 MoE decode was requested before prefill".to_owned(),
            ));
        }
        // GenerationService publishes the selected completion token before it
        // asks the executor to consume it.  Once decode starts, a cancellation
        // must drop this request rather than rewind a token visible to the
        // caller.
        self.published_since_transition = true;
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = self
            .inner
            .execute_next(&[token])
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let argmax = output
            .token_ids()
            .last()
            .copied()
            .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
        if output.token_ids().len() != 1 {
            return Err(GenerationServiceError::Execution(
                "Gemma 4 MoE decode published a non-singleton argmax".to_owned(),
            ));
        }
        self.absorb(&output);
        Ok(GenerationStepV1::new(
            u32::try_from(argmax).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            None,
        ))
    }

    fn cancel(&mut self) {
        if self.inner.transition_committed() && !self.published_since_transition {
            let _ = self.inner.cancel_last_transition();
        }
    }
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
    mtp_draft_width: Option<u8>,
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

/// Runs the benchmark's exact greedy loop while retaining the same
/// publication timestamps for target-only and speculative executors.  The
/// regular generation service deliberately hides these callbacks; benchmark
/// timing therefore drives the already-public executor contract directly.
fn run_generation_executor_timed(
    executor: &mut impl GenerationExecutorV1,
    policy: &GenerationStopPolicyV1,
    max_new_tokens: u32,
    input_token_ids: &[u32],
    timing: (&mut BenchmarkTimeline, MonotonicClock),
) -> Result<GenerationOutcome, String> {
    let (timeline, clock) = timing;
    timeline.record(BenchmarkEvent::PrefillSubmit, clock.now_ns())?;
    let first = executor
        .prefill(input_token_ids, false)
        .map_err(|error| format!("Gemma prefill failed: {error}"))?;
    timeline.record(BenchmarkEvent::PrefillComplete, clock.now_ns())?;
    timeline.record(BenchmarkEvent::FirstToken, clock.now_ns())?;
    let mut controller = GenerationStopControllerV1::new_with_input_token_ids(
        policy,
        max_new_tokens,
        input_token_ids,
    )
    .map_err(|error| format!("generation stop policy could not be initialized: {error}"))?;
    let mut generated = first.device_argmax();
    let mut decode_steps = 0_u32;
    loop {
        let decision = controller
            .observe_generated(generated)
            .map_err(|error| format!("generated token violated the stop policy: {error}"))?;
        let Some(decode_input) = decision.decode_input_token_id() else {
            timeline.record(BenchmarkEvent::Stop, clock.now_ns())?;
            executor
                .finish()
                .map_err(|error| format!("generation executor finish failed: {error}"))?;
            timeline.record(BenchmarkEvent::Cleanup, clock.now_ns())?;
            return Ok(GenerationOutcome {
                report: controller.into_report(),
                decode_steps,
            });
        };
        let step = executor
            .decode(decode_input, false)
            .map_err(|error| format!("Gemma decode failed: {error}"))?;
        decode_steps = decode_steps
            .checked_add(1)
            .ok_or_else(|| "Gemma decode step count overflowed".to_owned())?;
        generated = step.device_argmax();
        timeline.record(BenchmarkEvent::LaterToken, clock.now_ns())?;
    }
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
    mtp_assistant_gguf: Option<PathBuf>,
    mtp_assistant_derived_lock: Option<PathBuf>,
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

struct Gemma4MoeProductionBackend {
    source: Arc<VerifiedGgufGemma4Moe>,
}

struct Ministral3ProductionBackend {
    lock: Ministral3ModelLock,
    frontend: Ministral3TextFrontendV1,
    source: Arc<VerifiedMinistral3WeightSource>,
    plan: sllm_core::WeightLoadPlan,
}

impl Ministral3ProductionBackend {
    fn open(path: &Path) -> Result<Self, String> {
        let lock = parse_ministral3_model_lock(include_bytes!(
            "../../../docs/models/locks/ministral3-3b-instruct-2512-official-bf16-gguf.json"
        ))
        .map_err(|error| format!("reviewed Ministral 3 model lock is invalid: {error}"))?;
        let verified = open_and_verify_official_ministral3_gguf(path)
            .map_err(|error| format!("official Ministral 3 GGUF is invalid: {error}"))?;
        let frontend = Ministral3TextFrontendV1::from_verified_gguf(&verified)
            .map_err(|error| format!("Ministral 3 frontend is invalid: {error}"))?;
        let source = Arc::new(
            VerifiedMinistral3WeightSource::from_verified_gguf(verified)
                .map_err(|error| format!("Ministral 3 weight source is invalid: {error}"))?,
        );
        let plan = build_ministral3_weight_load_plan(source.as_ref())
            .map_err(|error| format!("Ministral 3 weight plan is invalid: {error}"))?;
        Ok(Self {
            lock,
            frontend,
            source,
            plan,
        })
    }

    fn render_text(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<String, String> {
        if !matches!(
            options.thinking,
            ThinkingModeV1::TemplateDefault | ThinkingModeV1::Disabled
        ) {
            return Err("Ministral 3 does not expose a reviewed thinking mode".to_owned());
        }
        self.frontend
            .renderer()
            .render(messages)
            .map_err(|error| format!("Ministral 3 chat messages are invalid: {error}"))
    }

    fn reject_generation_extensions(&self, request: &GenerateRequest) -> Result<(), String> {
        if !request.image_paths.is_empty() {
            return Err("Ministral 3 production is text-only; --image is unsupported".to_owned());
        }
        if request.mtp_draft_width.is_some_and(|width| width != 0) {
            return Err("Ministral 3 does not support MTP generation".to_owned());
        }
        if request.prefill_chunk_tokens.is_some() {
            return Err("Ministral 3 does not yet expose chunked prefill".to_owned());
        }
        if request
            .kv_cache_encoding
            .is_some_and(|value| value != KvCacheEncoding::Fp16)
        {
            return Err("Ministral 3 uses its fixed FP16 KV cache".to_owned());
        }
        if request.fp8_manifest.is_some()
            || request.fp8_artifact.is_some()
            || request.fp8_provider.is_some()
        {
            return Err("Ministral 3 official GGUF cannot be combined with sidecars".to_owned());
        }
        if request.sampling.temperature() != 0.0
            || request.sampling.top_p() != 1.0
            || request.sampling.presence_penalty() != 0.0
            || request.sampling.frequency_penalty() != 0.0
            || request.seed.is_some()
        {
            return Err("Ministral 3 currently supports greedy generation only".to_owned());
        }
        Ok(())
    }
}

impl ModelFrontendBackend for Ministral3ProductionBackend {
    fn identity(&self) -> ModelIdentity {
        ModelIdentity {
            repo_id: self.lock.repository().to_owned(),
            resolved_revision: self.lock.revision().to_owned(),
            lock_fingerprint: self.lock.fingerprint().to_owned(),
        }
    }

    fn verify(&self) -> Result<Value, String> {
        Ok(json!({
            "kind": "verify-model",
            "architecture": "Ministral3ForCausalLM",
            "source_kind": "official-gguf",
            "model_alias": MINISTRAL3_MODEL_ALIAS,
            "model_fingerprint": MINISTRAL3_MODEL_LOCK_FINGERPRINT,
            "tensor_count": self.source.gguf().tensors().len(),
            "weight_entries": self.plan.entries.len(),
            "total_destination_bytes": self.plan.total_destination_bytes,
            "plan_digest": self.plan.digest_hex(),
            "weight_encoding": "bf16",
            "kv_cache_encoding": "fp16",
        }))
    }

    fn tokenize(&self, text: &str) -> Result<Value, String> {
        let ids = self
            .frontend
            .tokenizer()
            .encode(text)
            .map_err(|error| error.to_string())?;
        Ok(json!({
            "kind": "tokenize",
            "version": 1,
            "count": ids.len(),
            "token_ids": ids.as_slice(),
            "pieces": null,
            "tokenizer_fingerprint": sllm_frontend::MINISTRAL3_TOKENIZER_SHA256,
        }))
    }

    fn render(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        Ok(json!({"kind":"render", "text":self.render_text(messages, options)?}))
    }

    fn apply_template(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let rendered = self.render_text(messages, options)?;
        let ids = self
            .frontend
            .tokenizer()
            .encode(&rendered)
            .map_err(|error| error.to_string())?;
        Ok(json!({
            "kind": "apply-template",
            "version": 1,
            "text": rendered,
            "prompt": rendered,
            "count": ids.len(),
            "token_ids": ids.as_slice(),
            "tokenizer_fingerprint": sllm_frontend::MINISTRAL3_TOKENIZER_SHA256,
            "template": {
                "kind": "reviewed-model-template",
                "version": 1,
                "consistency_label": "ministral3-official-gguf-text-v1",
                "digest": sllm_frontend::MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SHA256,
                "size_bytes": sllm_frontend::MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SIZE_BYTES,
            }
        }))
    }

    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let text = self
            .frontend
            .tokenizer()
            .decode(ids, mode)
            .map_err(|error| error.to_string())?;
        Ok(json!({
            "kind":"decode",
            "text":text,
            "token_count":ids.len(),
            "tokenizer_fingerprint":sllm_frontend::MINISTRAL3_TOKENIZER_SHA256,
        }))
    }

    fn detokenize(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let mut value = self.decode(ids, mode)?;
        value["kind"] = Value::from("detokenize");
        Ok(value)
    }

    fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
        self.reject_generation_extensions(request)?;
        let started = Instant::now();
        let (input_kind, prompt) = match &request.input {
            GenerationInput::Prompt(prompt) => ("prompt", prompt.clone()),
            GenerationInput::Messages { messages, options } => {
                ("messages", self.render_text(messages, *options)?)
            }
        };
        let stop_policy = ministral3_generation_stop_policy()
            .map_err(|error| format!("Ministral 3 stop policy is invalid: {error}"))?;
        let service = GenerationServiceV1::new(&self.frontend, None, &stop_policy)
            .map_err(|error| format!("Ministral 3 generation service failed: {error}"))?;
        let input = service
            .prepare_input(&ServiceGenerationInputV1::Prompt(prompt))
            .map_err(|error| format!("Ministral 3 input preparation failed: {error}"))?;
        let input_len = u64::try_from(input.len())
            .map_err(|_| "Ministral 3 input length overflowed".to_owned())?;
        let state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or_else(|| "Ministral 3 state capacity overflowed".to_owned())?;
        if input_len == 0 || state_capacity > MINISTRAL3_GRAPH_MAX_CONTEXT {
            return Err("Ministral 3 request exceeds the reviewed context".to_owned());
        }
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(request.device_index, request.target.clone())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let execution = (|| -> Result<Value, String> {
            let resident = Ministral3ResidentModel::new_gguf(
                Arc::clone(&session),
                self.plan.clone(),
                Arc::clone(&self.source),
                COMPLETION_TIMEOUT,
            )
            .map_err(|error| format!("Ministral 3 resident provisioning failed: {error}"))?;
            let mut owner = resident
                .new_request(input_len, state_capacity)
                .map_err(|error| format!("Ministral 3 request provisioning failed: {error}"))?;
            let config = GenerationConfigV1::new(
                request.max_new_tokens,
                request.sampling,
                request.stop_strings.clone(),
            )
            .map_err(|error| format!("generation configuration is invalid: {error}"))?;
            let cancellation = GenerationCancellationV1::new();
            let mut random = OsSamplingRandom::for_parameters_and_seed(request.sampling, None)
                .map_err(|error| format!("sampling random source failed: {error}"))?;
            let report = service
                .generate_tokens(&mut owner, &input, &config, &cancellation, &mut random)
                .map_err(|error| format!("Ministral 3 generation failed: {error}"))?;
            let audit = owner
                .last_audit()
                .ok_or_else(|| "Ministral 3 dispatch audit is absent".to_owned())?;
            if audit.target() != request.target || audit.fallback_used() {
                return Err("Ministral 3 dispatch audit is not exact HIP/no-fallback".to_owned());
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
                "stop_reason":{
                    "version":1,
                    "reason_version":1,
                    "kind":report.finish_reason().as_str(),
                    "token_id":report.stop_token_id(),
                    "matched_string":report.matched_stop(),
                },
                "usage":{
                    "prompt_tokens":report.usage().prompt_tokens(),
                    "completion_tokens":report.usage().completion_tokens(),
                    "total_tokens":report.usage().total_tokens(),
                },
                "sampling":{"temperature":0.0,"top_p":1.0,"presence_penalty":0.0,"frequency_penalty":0.0},
                "execution":{
                    "selected_backend":"hip",
                    "target":audit.target(),
                    "device_index":request.device_index,
                    "model_fingerprint":self.lock.fingerprint(),
                    "plan_digest":self.plan.digest_hex(),
                    "prefill_tokens":input.len(),
                    "logical_state_capacity_tokens":state_capacity,
                    "allocated_state_capacity_tokens":state_capacity,
                    "decode_steps":report.decode_steps(),
                    "fallback_used":audit.fallback_used(),
                    "submission_count":audit.submission_count(),
                    "kernel_dispatch_count":audit.kernel_dispatch_count(),
                    "all_dispatches_hip":true,
                    "weight_encoding":"bf16",
                    "kv_cache_encoding":"fp16",
                },
                "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
            }))
        })();
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|error| format!("HIP session cleanup failed: {error}"))?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("Ministral 3 HIP session cleanup was not empty".to_owned());
        }
        execution
    }

    fn benchmark(
        &self,
        _request: &BenchmarkRequest,
        _timing: BenchmarkTiming,
    ) -> Result<Value, String> {
        Err("Ministral 3 is integrated into normal CLI/API/WebUI generation; the internal CLI evidence benchmark is not exposed for this direct artifact"
            .to_owned())
    }
}

impl Gemma4MoeProductionBackend {
    fn plan(&self) -> Result<sllm_core::WeightLoadPlan, String> {
        build_gguf_gemma4_moe_weight_load_plan(&self.source)
            .map_err(|error| format!("Gemma 4 MoE GGUF load plan is invalid: {error}"))
    }

    fn resident_plan(&self) -> Result<sllm_core::WeightLoadPlan, String> {
        build_gemma4_moe_resident_weight_load_plan(self.source.as_ref())
            .map_err(|error| format!("Gemma 4 MoE resident load plan is invalid: {error}"))
    }

    fn tokenizer(&self) -> Result<TokenizerFrontendV1, String> {
        TokenizerFrontendV1::from_gemma4_moe_gguf(self.source.gguf())
            .map_err(|error| format!("Gemma 4 MoE tokenizer is invalid: {error}"))
    }

    fn renderer(&self) -> Result<Gemma4MoeChatTemplateV1, String> {
        Gemma4MoeChatTemplateV1::from_gemma4_moe_gguf(self.source.gguf())
            .map_err(|error| format!("Gemma 4 MoE chat template is invalid: {error}"))
    }

    fn generation_input(
        &self,
        input: &GenerationInput,
    ) -> Result<(ServiceGenerationInputV1, &'static str), String> {
        match input {
            GenerationInput::Prompt(prompt) => {
                Ok((ServiceGenerationInputV1::Prompt(prompt.clone()), "prompt"))
            }
            GenerationInput::Messages { messages, options } => Ok((
                ServiceGenerationInputV1::Messages {
                    messages: messages.clone(),
                    options: *options,
                },
                "messages",
            )),
        }
    }
}

impl ModelFrontendBackend for Gemma4MoeProductionBackend {
    fn identity(&self) -> ModelIdentity {
        ModelIdentity {
            repo_id: sllm_core::GEMMA4_MOE_SEMANTIC_REPOSITORY.to_owned(),
            resolved_revision: sllm_core::GEMMA4_MOE_SEMANTIC_REVISION.to_owned(),
            lock_fingerprint: sllm_core::GEMMA4_MOE_MODEL_FINGERPRINT.to_owned(),
        }
    }

    fn verify(&self) -> Result<Value, String> {
        let plan = self.plan()?;
        let loadable = plan
            .entries
            .iter()
            .filter(|entry| entry.classification != WeightClassification::KnownUnconsumed)
            .count();
        Ok(json!({
            "kind": "verify-model",
            "architecture": "Gemma4ForCausalLM",
            "model_kind": "gemma4-moe",
            "source_kind": "gguf",
            "source_file_sha256": self.source.file_sha256(),
            "tensor_count": self.source.gguf().tensors().len(),
            "weight_entries": plan.entries.len(),
            "loadable_entries": loadable,
            "known_unconsumed_entries": plan.entries.len() - loadable,
            "total_destination_bytes": plan.total_destination_bytes,
            "plan_digest": plan.digest_hex(),
            "weight_encoding": "nvfp4-e2m1-block16-e4m3fn-tensor-f32-routed",
            "kv_cache_encoding": "fp8-e4m3fn-static-unit-scale",
            "expert_count": self.source.config().expert_count,
            "selected_expert_count": self.source.config().selected_expert_count,
        }))
    }

    fn tokenize(&self, text: &str) -> Result<Value, String> {
        utility_tokenize(&self.tokenizer()?, text, false)
    }

    fn tokenize_with_pieces(&self, text: &str) -> Result<Value, String> {
        utility_tokenize(&self.tokenizer()?, text, true)
    }

    fn render(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let renderer = self.renderer()?;
        let rendered = renderer
            .renderer()
            .render(messages, options)
            .map_err(|error| error.to_string())?;
        let template = rendered.generic_identity().map(|identity| {
            json!({
                "kind": "generic-jinja-v1",
                "version": identity.version(),
                "digest": identity.template_digest(),
                "size_bytes": identity.source_size_bytes(),
                "rendered_digest": identity.rendered_digest(),
            })
        });
        Ok(json!({
            "kind": "render",
            "text": rendered.rendered(),
            "template": template,
        }))
    }

    fn apply_template(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let renderer = self.renderer()?;
        utility_apply_custom_template(
            &tokenizer,
            renderer.provider(),
            messages,
            options,
            Map::new(),
        )
    }

    fn apply_template_custom(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
        provider: &GenericTemplateProviderV1,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        utility_apply_custom_template(&self.tokenizer()?, provider, messages, options, kwargs)
    }

    fn input_tokens(
        &self,
        text: Option<&str>,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String> {
        if let Some(text) = text {
            return utility_input_tokens(&self.tokenizer()?, None, Some(text), messages, options);
        }
        let renderer = self.renderer()?;
        utility_input_tokens_custom(
            &self.tokenizer()?,
            renderer.provider(),
            messages,
            options,
            Map::new(),
        )
    }

    fn input_tokens_custom(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
        provider: &GenericTemplateProviderV1,
        kwargs: Map<String, Value>,
    ) -> Result<Value, String> {
        utility_input_tokens_custom(&self.tokenizer()?, provider, messages, options, kwargs)
    }

    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let mut result = utility_detokenize(&self.tokenizer()?, ids, mode)?;
        result["kind"] = Value::from("decode");
        Ok(result)
    }

    fn detokenize(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        utility_detokenize(&self.tokenizer()?, ids, mode)
    }

    fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
        let started = Instant::now();
        if request.sampling != SamplingParametersV1::greedy() {
            return Err(
                "Gemma 4 MoE currently exposes device Argmax only; non-greedy sampling is unavailable"
                    .to_owned(),
            );
        }
        if request.prefill_chunk_tokens.is_some() || request.mtp_draft_width.is_some() {
            return Err(
                "--prefill-chunk-tokens and --mtp-draft-width are unavailable for Gemma 4 MoE"
                    .to_owned(),
            );
        }
        if !request.image_paths.is_empty() {
            return Err("Gemma 4 MoE CLI generation is text-only".to_owned());
        }
        if request.fp8_manifest.is_some()
            || request.fp8_artifact.is_some()
            || request.fp8_provider.is_some()
        {
            return Err(
                "Gemma 4 MoE GGUF carries its own NVFP4 recipe; sidecar flags are unavailable"
                    .to_owned(),
            );
        }
        if let Some(encoding) = request.kv_cache_encoding
            && encoding != KvCacheEncoding::Fp8E4M3FnStatic
        {
            return Err(
                "Gemma 4 MoE requires its static FP8 E4M3 KV contract; use no KV flag or fp8-static"
                    .to_owned(),
            );
        }
        let tokenizer = self.tokenizer()?;
        let renderer = self.renderer()?;
        let stop_policy = gemma4_moe_generation_stop_policy()
            .map_err(|error| format!("Gemma 4 MoE stop policy is invalid: {error}"))?;
        let service = GenerationServiceV1::new_with_chat_renderer(
            &tokenizer,
            Some(ChatTemplateRendererV1::generic_with_config(
                renderer.provider(),
                renderer.config().clone(),
            )),
            &stop_policy,
        )
        .map_err(|error| format!("generation service could not be constructed: {error}"))?;
        let (service_input, input_kind) = self.generation_input(&request.input)?;
        let input = service
            .prepare_input(&service_input)
            .map_err(|error| format!("generation input preparation failed: {error}"))?;
        if input.is_empty() {
            return Err("Gemma 4 MoE generation input token sequence is empty".to_owned());
        }
        let input_len = u64::try_from(input.len())
            .map_err(|_| "Gemma 4 MoE input token count overflowed".to_owned())?;
        let max_context = u64::from(self.source.config().max_position_embeddings);
        let state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or_else(|| "Gemma 4 MoE state capacity overflowed".to_owned())?;
        if state_capacity > max_context {
            return Err(format!(
                "Gemma 4 MoE input plus output tokens exceed context limit {max_context}"
            ));
        }
        let plan = self.resident_plan()?;
        let plan_digest = plan.digest_hex();
        let execution_state_capacity = state_capacity.max(GEMMA4_MOE_PREFILL_CHUNK_TOKENS);
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(request.device_index, request.target.clone())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let execution = (|| -> Result<Value, String> {
            let graph = build_gemma4_moe_gguf_graph(
                &self.source,
                input_len.min(GEMMA4_MOE_PREFILL_CHUNK_TOKENS),
                0,
                execution_state_capacity,
            )
            .map_err(|error| format!("Gemma 4 MoE execution graph is invalid: {error}"))?;
            let resident = Gemma4MoeResidentModel::provision(
                Arc::clone(&session),
                Arc::clone(&self.source),
                plan.clone(),
                COMPLETION_TIMEOUT,
            )
            .map_err(|error| format!("Gemma 4 MoE resident provisioning failed: {error}"))?;
            let owner = resident
                .new_request(graph)
                .map_err(|error| format!("Gemma 4 MoE request provisioning failed: {error}"))?;
            let mut executor = CliGemma4MoeExecutor::new(owner);
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
                .generate_tokens(&mut executor, &input, &config, &cancellation, &mut random)
                .map_err(|error| format!("Gemma 4 MoE generation failed: {error}"))?;
            let audit = executor.audit_json(&request.target)?;
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
                "execution": {
                    "selected_backend": "hip",
                    "target": request.target,
                    "device_index": request.device_index,
                    "model_fingerprint": sllm_core::GEMMA4_MOE_MODEL_FINGERPRINT,
                    "plan_digest": plan_digest,
                    "prefill_tokens": report.usage().prompt_tokens(),
                    "logical_state_capacity": state_capacity,
                    "allocated_state_capacity": execution_state_capacity,
                    "decode_steps": report.decode_steps(),
                    "fallback_used": audit["fallback_used"],
                    "submission_count": audit["submission_count"],
                    "kernel_dispatch_count": audit["kernel_dispatch_count"],
                    "segment_count": audit["segment_count"],
                    "boundary_count": audit["boundary_count"],
                    "all_dispatches_hip": true,
                    "weight_encoding": "nvfp4-e2m1-block16-e4m3fn-tensor-f32-routed",
                    "kv_cache_encoding": "fp8-e4m3fn-static-unit-scale",
                    "fp8_provider": null,
                },
            }))
        })();
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|error| format!("HIP session cleanup failed: {error}"))?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        let mut result = execution?;
        let object = result
            .as_object_mut()
            .ok_or_else(|| "Gemma 4 MoE generation result was not an object".to_owned())?;
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
        if request.model_size != "26B-A4B" {
            return Err("Gemma 4 MoE benchmark requires --model-size 26B-A4B".to_owned());
        }
        if !request.greedy {
            return Err("Gemma 4 MoE benchmark requires explicit --greedy mode".to_owned());
        }
        if request.prefill_chunk_tokens.is_some()
            || request.fp8_manifest.is_some()
            || request.fp8_artifact.is_some()
            || request.fp8_provider.is_some()
        {
            return Err(
                "Gemma 4 MoE benchmark does not accept sidecar or chunk overrides".to_owned(),
            );
        }
        if request.kv_cache_encoding.is_some()
            && request.kv_cache_encoding != Some(KvCacheEncoding::Fp8E4M3FnStatic)
        {
            return Err("Gemma 4 MoE benchmark requires static FP8 E4M3 KV".to_owned());
        }
        validate_benchmark_protocol(request.warmups, request.measured)?;
        let completion_timeout = benchmark_completion_timeout(request.completion_timeout_seconds)?;
        let tokenizer = self.tokenizer()?;
        let renderer = self.renderer()?;
        let stop_policy = gemma4_moe_generation_stop_policy()
            .map_err(|error| format!("Gemma 4 MoE stop policy is invalid: {error}"))?;
        let stop_policy = benchmark_stop_policy(&stop_policy, request.ignore_eos);
        let service = GenerationServiceV1::new_with_chat_renderer(
            &tokenizer,
            Some(ChatTemplateRendererV1::generic_with_config(
                renderer.provider(),
                renderer.config().clone(),
            )),
            &stop_policy,
        )
        .map_err(|error| format!("generation service could not be constructed: {error}"))?;
        let seed_input = match &request.input {
            BenchmarkInput::TokenIds(ids) => ids.clone(),
            BenchmarkInput::Messages { messages, options } => {
                let ids = service
                    .prepare_input(&ServiceGenerationInputV1::Messages {
                        messages: messages.clone(),
                        options: *options,
                    })
                    .map_err(|error| error.to_string())?;
                TokenIdsV1::from_slice(&ids)
            }
        };
        if seed_input.is_empty() {
            return Err("Gemma 4 MoE benchmark input token IDs must not be empty".to_owned());
        }
        let input_len = u64::try_from(seed_input.len())
            .map_err(|_| "Gemma 4 MoE benchmark input length overflowed".to_owned())?;
        let state_capacity =
            benchmark_state_capacity(input_len, request.max_new_tokens, request.context_length)?;
        let execution_state_capacity = state_capacity.max(GEMMA4_MOE_PREFILL_CHUNK_TOKENS);
        if execution_state_capacity > u64::from(self.source.config().max_position_embeddings) {
            return Err("Gemma 4 MoE benchmark context exceeds model context".to_owned());
        }
        let plan = self.resident_plan()?;
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(request.device_index, request.target.clone())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let model_load_start_ns = timing.model_load_start_ns();
        let execution = (|| -> Result<Value, String> {
            let resident = Gemma4MoeResidentModel::provision(
                Arc::clone(&session),
                Arc::clone(&self.source),
                plan.clone(),
                completion_timeout,
            )
            .map_err(|error| format!("Gemma 4 MoE resident provisioning failed: {error}"))?;
            let model_ready_ns = timing.now_ns();
            let mut samples = Vec::new();
            for sample_index in 0..(request.warmups + request.measured) {
                let request_start_ns = timing.now_ns();
                let graph = build_gemma4_moe_gguf_graph(
                    &self.source,
                    input_len.min(GEMMA4_MOE_PREFILL_CHUNK_TOKENS),
                    0,
                    execution_state_capacity,
                )
                .map_err(|error| error.to_string())?;
                let owner = resident
                    .new_request(graph)
                    .map_err(|error| error.to_string())?;
                let mut executor = CliGemma4MoeExecutor::new(owner);
                let config = GenerationConfigV1::new(
                    request.max_new_tokens,
                    SamplingParametersV1::greedy(),
                    Vec::new(),
                )
                .map_err(|error| error.to_string())?;
                let cancellation = GenerationCancellationV1::new();
                let mut random =
                    OsSamplingRandom::for_parameters_and_seed(SamplingParametersV1::greedy(), None)
                        .map_err(|error| error.to_string())?;
                let report = service
                    .generate_tokens(
                        &mut executor,
                        seed_input.as_slice(),
                        &config,
                        &cancellation,
                        &mut random,
                    )
                    .map_err(|error| error.to_string())?;
                let audit = executor.audit_json(&request.target)?;
                let mut timeline = BenchmarkTimeline::new(request_start_ns);
                timeline.record(BenchmarkEvent::PrefillSubmit, request_start_ns)?;
                let now = timing.now_ns();
                timeline.record(BenchmarkEvent::PrefillComplete, now)?;
                timeline.record(BenchmarkEvent::FirstToken, now)?;
                for _ in report.generated_token_ids().iter().skip(1) {
                    let later = timing.now_ns();
                    timeline.record(BenchmarkEvent::LaterToken, later)?;
                }
                let stop = timing.now_ns();
                timeline.record(BenchmarkEvent::Stop, stop)?;
                let cleanup = timing.now_ns();
                timeline.record(BenchmarkEvent::Cleanup, cleanup)?;
                let generated = report.generated_token_ids();
                let visible = report.visible_token_ids();
                let decode_inputs = report.decode_input_token_ids();
                let sample = timeline.finish(BenchmarkSampleInput {
                    input_token_ids: seed_input.as_slice(),
                    generated_token_ids: generated,
                    visible_token_ids: visible,
                    decode_input_token_ids: decode_inputs,
                    stop: json!({
                        "finish_reason": report.finish_reason().as_str(),
                        "stop_token_id": report.stop_token_id(),
                        "matched_string": report.matched_stop(),
                    }),
                    audit,
                    memory: json!({"sample_index": sample_index}),
                    cleanup: json!({
                        "sample_index": sample_index,
                        "request_dropped": true,
                        "retryable_cleanup": 0,
                        "durable_quarantine": 0,
                    }),
                })?;
                samples.push(sample);
            }
            let warmup_samples = samples[..request.warmups as usize].to_vec();
            let measured_samples = samples[request.warmups as usize..].to_vec();
            let correctness_control = correctness_reference_from_warmup(
                warmup_samples
                    .first()
                    .ok_or_else(|| "benchmark requires a warmup sample".to_owned())?,
            )?;
            for sample in warmup_samples.iter().skip(1).chain(measured_samples.iter()) {
                compare_control_sample(&correctness_control, sample)?;
            }
            Ok(json!({
                "benchmark_schema_version": request.lane.schema_version(),
                "state": "PASS",
                "lane": match request.lane { BenchmarkLane::Direct => "direct", BenchmarkLane::RenderTokenize => "render-tokenize" },
                "lane_definition": "Gemma 4 MoE exact HIP greedy generation",
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
                        "repo_id": sllm_core::GEMMA4_MOE_SEMANTIC_REPOSITORY,
                        "resolved_revision": sllm_core::GEMMA4_MOE_SEMANTIC_REVISION,
                        "lock_fingerprint": sllm_core::GEMMA4_MOE_MODEL_FINGERPRINT,
                    },
                    "binding": {
                        "model_fingerprint": sllm_core::GEMMA4_MOE_MODEL_FINGERPRINT,
                        "plan_digest": plan.digest_hex(),
                    },
                },
                "model_load": {
                    "event": "model_load",
                    "start_ns": model_load_start_ns,
                    "model_ready_ns": model_ready_ns,
                    "duration_ns": model_ready_ns.saturating_sub(model_load_start_ns),
                    "load_count": 1,
                },
                "config": {
                    "input_token_count": seed_input.len(),
                    "max_new_tokens": request.max_new_tokens,
                    "context_length": request.context_length,
                    "requested_state_capacity": state_capacity,
                    "effective_context_length": execution_state_capacity,
                    "allocated_state_capacity": execution_state_capacity,
                    "greedy": true,
                    "warmups": request.warmups,
                    "measured": request.measured,
                    "kv_cache_encoding": "fp8-e4m3fn-static-unit-scale",
                },
                "memory": {},
                "audit": {
                    "selected_backend": "hip",
                    "target": request.target,
                    "device_index": request.device_index,
                    "model_fingerprint": sllm_core::GEMMA4_MOE_MODEL_FINGERPRINT,
                    "plan_digest": plan.digest_hex(),
                    "fallback_used": false,
                    "all_dispatches_hip": true,
                    "model_load_count": 1,
                    "model_reused": true,
                    "sample_count": request.warmups + request.measured,
                },
                "cleanup": {
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
        let cleanup = session
            .shutdown(SHUTDOWN_TIMEOUT)
            .map_err(|error| format!("HIP session cleanup failed: {error}"))?;
        let mut result = execution?;
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        result["session_cleanup"] = json!({
            "retryable_cleanup": cleanup.retryable_cleanup,
            "durable_quarantine": cleanup.durable_quarantine,
        });
        Ok(result)
    }
}

/// Persistent adapter used by the interactive CLI. It verifies and opens the
/// exact Gemma MoE artifact once, then reuses the tokenizer, renderer, HIP
/// session, and resident weights across turns. Each turn receives a fresh
/// request-local owner; when a committed state image/checkpoint is available,
/// the owner imports that prefix and executes only the new suffix.
pub(crate) struct Gemma4MoeCliChatBackend {
    source: Arc<VerifiedGgufGemma4Moe>,
    tokenizer: TokenizerFrontendV1,
    renderer: Gemma4MoeChatTemplateV1,
    stop_policy: GenerationStopPolicyV1,
    _plan: sllm_core::WeightLoadPlan,
    session: Arc<sllm_core::ExecutionSession>,
    resident: Option<Gemma4MoeResidentModel>,
    _device_index: u32,
    target: String,
    context_length: u64,
    shutdown_timeout: Duration,
    checkpoint_store: Arc<CheckpointStore>,
    current_checkpoint: Option<SessionCheckpoint>,
    checkpoint_loaded_explicitly: bool,
    pending_state: Option<(sllm_core::Gemma4MoeStateImageV1, Vec<u32>)>,
    active_cancellation: Option<GenerationCancellationV1>,
}

impl Gemma4MoeCliChatBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open(
        gguf: PathBuf,
        derived_lock: PathBuf,
        device_index: u32,
        target: String,
        context_length: u32,
        kv_cache_encoding: Option<KvCacheEncoding>,
        completion_timeout: Duration,
        shutdown_timeout: Duration,
        checkpoint_directory: PathBuf,
        checkpoint_quota_bytes: u64,
    ) -> Result<Self, String> {
        let derived = read_derived_gguf_lock(&derived_lock)
            .map_err(|error| format!("chat derived GGUF lock is invalid: {error}"))?;
        if !derived.semantic_model_id.starts_with("gemma4moe:") {
            return Err("chat Gemma 4 MoE adapter received a non-Gemma MoE artifact".to_owned());
        }
        let verified = verify_derived_gguf(derived, &gguf)
            .map_err(|error| format!("chat GGUF does not match its derived lock: {error}"))?;
        let source = verify_gguf_gemma4_moe(verified)
            .map_err(|error| format!("chat Gemma 4 MoE GGUF is invalid: {error}"))?;
        let source = Arc::new(source);
        // Chat's public default follows Qwen's 1M-token recommendation.  The
        // reviewed Gemma artifact has a smaller fixed model window, so use the
        // model limit rather than rejecting an otherwise valid default.
        let context_length =
            u64::from(context_length).min(u64::from(source.config().max_position_embeddings));
        let model_context_length = u64::from(source.config().max_position_embeddings);
        if context_length == 0 {
            return Err(format!(
                "chat context length must be within the Gemma 4 MoE model limit {model_context_length}"
            ));
        }
        validate_gemma4_moe_chat_kv_cache_encoding(kv_cache_encoding).map_err(ToOwned::to_owned)?;
        if completion_timeout.is_zero() || shutdown_timeout.is_zero() {
            return Err("chat completion and shutdown timeouts must be nonzero".to_owned());
        }
        let checkpoint_store = Arc::new(
            CheckpointStore::new(&checkpoint_directory, checkpoint_quota_bytes)
                .map_err(|error| format!("chat checkpoint store is invalid: {error}"))?,
        );
        let tokenizer = TokenizerFrontendV1::from_gemma4_moe_gguf(source.gguf())
            .map_err(|error| format!("chat Gemma 4 MoE tokenizer is invalid: {error}"))?;
        let renderer = Gemma4MoeChatTemplateV1::from_gemma4_moe_gguf(source.gguf())
            .map_err(|error| format!("chat Gemma 4 MoE chat template is invalid: {error}"))?;
        let stop_policy = gemma4_moe_generation_stop_policy()
            .map_err(|error| format!("chat Gemma 4 MoE stop policy is invalid: {error}"))?;
        let plan = build_gemma4_moe_resident_weight_load_plan(source.as_ref())
            .map_err(|error| format!("chat Gemma 4 MoE resident load plan is invalid: {error}"))?;
        let hip = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session = hip
            .open_execution_session(
                ExecutionSessionRequest::new(device_index, target.clone())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let resident = Gemma4MoeResidentModel::provision(
            Arc::clone(&session),
            Arc::clone(&source),
            plan.clone(),
            completion_timeout,
        )
        .map_err(|error| format!("chat Gemma 4 MoE resident provisioning failed: {error}"))?;
        Ok(Self {
            source,
            tokenizer,
            renderer,
            stop_policy,
            _plan: plan,
            session,
            resident: Some(resident),
            _device_index: device_index,
            target,
            context_length,
            shutdown_timeout,
            checkpoint_store,
            current_checkpoint: None,
            checkpoint_loaded_explicitly: false,
            pending_state: None,
            active_cancellation: None,
        })
    }

    pub(crate) fn set_cancellation(&mut self, cancellation: GenerationCancellationV1) {
        self.active_cancellation = Some(cancellation);
    }

    fn checkpoint_capacity(identity: &CheckpointIdentity) -> Result<u64, String> {
        let (_, capacity) = identity
            .target_semantics
            .rsplit_once(":capacity=")
            .ok_or_else(|| "Gemma 4 MoE checkpoint capacity is absent".to_owned())?;
        let capacity = capacity
            .parse::<u64>()
            .map_err(|_| "Gemma 4 MoE checkpoint capacity is invalid".to_owned())?;
        if capacity < GEMMA4_MOE_PREFILL_CHUNK_TOKENS {
            return Err(
                "Gemma 4 MoE checkpoint capacity is below the physical ring window".to_owned(),
            );
        }
        Ok(capacity)
    }

    fn renderer_identity(&self) -> Result<String, String> {
        let digest = self
            .source
            .gguf()
            .metadata_value("tokenizer.chat_template.sha256")
            .and_then(|value| match value {
                sllm_core::GgufValue::String(value) => Some(value.as_str()),
                _ => None,
            })
            .ok_or_else(|| "Gemma 4 MoE chat template digest is absent".to_owned())?;
        Ok(format!("gemma4moe-generic-jinja-v1:{digest}"))
    }

    fn checkpoint_context_policy_digest(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"sllm-gemma4-moe-cli-checkpoint-context-v1");
        digest.update(self.context_length.to_le_bytes());
        digest.finalize().into()
    }

    fn checkpoint_identity(
        &self,
        image: &sllm_core::Gemma4MoeStateImageV1,
        tokens: &[u32],
    ) -> Result<CheckpointIdentity, String> {
        if image.model_fingerprint() != sllm_core::GEMMA4_MOE_MODEL_FINGERPRINT
            || image.plan_digest() != self._plan.digest()
        {
            return Err("Gemma 4 MoE state image identity differs from the resident".to_owned());
        }
        CheckpointIdentity::for_tokens(
            sllm_core::GEMMA4_MOE_MODEL_FINGERPRINT,
            format!("derived-artifact:{}", self.source.file_sha256()),
            "adapter:none-v1",
            self.renderer_identity()?,
            self.tokenizer.snapshot().fingerprint(),
            format!(
                "{}:nvfp4-e2m1-block16-fp8-e4m3fn-static:capacity={}",
                self.target,
                image.state_capacity()
            ),
            self._plan.digest_hex(),
            tokens,
            KvCacheEncoding::Fp8E4M3FnStatic,
            image.kv_descriptor_digest(),
            self.checkpoint_context_policy_digest(),
        )
        .map_err(|error| format!("Gemma 4 MoE checkpoint identity is invalid: {error}"))
    }

    fn history_tokens(
        &self,
        messages: &[Qwen35ChatMessageV1],
        output_text: &str,
    ) -> Result<Vec<u32>, String> {
        let mut history = messages.to_vec();
        history.push(Qwen35ChatMessageV1::assistant(output_text, None));
        let rendered = self
            .renderer
            .renderer()
            .render(
                &history,
                Qwen35RenderOptionsV1 {
                    add_generation_prompt: false,
                    thinking: ThinkingModeV1::Disabled,
                },
            )
            .map_err(|error| format!("Gemma 4 MoE history render failed: {error}"))?;
        Ok(self
            .tokenizer
            .encode(rendered.rendered())
            .map_err(|error| format!("Gemma 4 MoE history tokenization failed: {error}"))?
            .as_slice()
            .to_vec())
    }

    fn rebuild_state_image(
        &self,
        tokens: &[u32],
        state_capacity: u64,
    ) -> Result<sllm_core::Gemma4MoeStateImageV1, String> {
        if tokens.is_empty() || u64::try_from(tokens.len()).unwrap_or(u64::MAX) > state_capacity {
            return Err("Gemma 4 MoE checkpoint history exceeds its state capacity".to_owned());
        }
        let graph = build_gemma4_moe_gguf_graph(
            &self.source,
            u64::try_from(tokens.len())
                .map_err(|_| "Gemma 4 MoE checkpoint token count overflowed".to_owned())?
                .min(GEMMA4_MOE_PREFILL_CHUNK_TOKENS),
            0,
            state_capacity,
        )
        .map_err(|error| format!("Gemma 4 MoE checkpoint graph is invalid: {error}"))?;
        let owner = self
            .resident
            .as_ref()
            .ok_or_else(|| "Gemma 4 MoE resident is shut down".to_owned())?
            .new_request(graph)
            .map_err(|error| format!("Gemma 4 MoE checkpoint request failed: {error}"))?;
        let mut executor = CliGemma4MoeExecutor::new(owner);
        executor
            .prefill(tokens, false)
            .map_err(|error| format!("Gemma 4 MoE checkpoint prefill failed: {error}"))?;
        executor.state_image()
    }

    fn stage_checkpoint_state(
        &mut self,
        messages: &[Qwen35ChatMessageV1],
        output_text: &str,
        state_capacity: u64,
    ) -> Result<(), ChatBackendErrorV1> {
        let tokens = self
            .history_tokens(messages, output_text)
            .map_err(|_| ChatBackendErrorV1::Failed)?;
        let image = self
            .rebuild_state_image(&tokens, state_capacity)
            .map_err(|_| ChatBackendErrorV1::Failed)?;
        self.pending_state = Some((image, tokens));
        Ok(())
    }

    fn candidate_checkpoint(
        &self,
        conversation: &[u8],
    ) -> Result<SessionCheckpoint, ChatBackendErrorV1> {
        let (image, tokens) = self
            .pending_state
            .as_ref()
            .ok_or(ChatBackendErrorV1::CheckpointUnavailable)?;
        let identity = self
            .checkpoint_identity(image, tokens)
            .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)?;
        image
            .to_checkpoint(
                identity,
                tokens,
                conversation,
                &[],
                &[],
                &[],
                image.committed_length(),
                image.committed_length(),
                1,
            )
            .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)
    }

    pub(crate) fn load_named_checkpoint(
        &mut self,
        name: &str,
    ) -> Result<Vec<u8>, ChatBackendErrorV1> {
        if self.pending_state.is_some() {
            return Err(ChatBackendErrorV1::CheckpointUnavailable);
        }
        let checkpoint = self
            .checkpoint_store
            .load_validated(name)
            .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)?;
        let capacity = Self::checkpoint_capacity(&checkpoint.header.identity)
            .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)?;
        let identity = &checkpoint.header.identity;
        let expected_target = format!(
            "{}:nvfp4-e2m1-block16-fp8-e4m3fn-static:capacity={capacity}",
            self.target
        );
        let expected_derived = format!("derived-artifact:{}", self.source.file_sha256());
        let expected_renderer = self
            .renderer_identity()
            .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)?;
        if identity.model_lock_fingerprint != sllm_core::GEMMA4_MOE_MODEL_FINGERPRINT
            || identity.derived_artifact_identity != expected_derived
            || identity.adapter_identity != "adapter:none-v1"
            || identity.renderer_identity != expected_renderer
            || identity.tokenizer_identity != self.tokenizer.snapshot().fingerprint()
            || identity.target_semantics != expected_target
            || identity.plan_digest != self._plan.digest_hex()
            || identity.kv_encoding != KvCacheEncoding::Fp8E4M3FnStatic
            || identity.context_policy_digest != self.checkpoint_context_policy_digest()
            || capacity > self.context_length
        {
            return Err(ChatBackendErrorV1::CheckpointUnavailable);
        }
        let conversation = checkpoint.payload.conversation.clone();
        self.current_checkpoint = Some(checkpoint);
        self.checkpoint_loaded_explicitly = true;
        Ok(conversation)
    }
}

impl Drop for Gemma4MoeCliChatBackend {
    fn drop(&mut self) {
        // Release all model-resident buffers before closing the session. The
        // session shutdown contract otherwise observes live resident owners.
        let _ = self.resident.take();
        let _ = self.session.shutdown(self.shutdown_timeout);
    }
}

impl crate::chat::ChatBackendV1 for Gemma4MoeCliChatBackend {
    fn generate(
        &mut self,
        request: &ChatGenerationRequestV1,
    ) -> Result<ChatGenerationResultV1, ChatBackendErrorV1> {
        if request.thinking == ChatThinkingModeV1::Enabled || request.reasoning_budget.is_some() {
            // The reviewed Gemma 4 MoE CLI owner is device-Argmax only and
            // has no reasoning controller/visible-vs-hidden token contract.
            return Err(ChatBackendErrorV1::Failed);
        }
        let thinking = match request.thinking {
            ChatThinkingModeV1::Default => ThinkingModeV1::TemplateDefault,
            ChatThinkingModeV1::Enabled => unreachable!("enabled thinking was rejected above"),
            ChatThinkingModeV1::Disabled => ThinkingModeV1::Disabled,
        };
        let service = GenerationServiceV1::new_with_chat_renderer(
            &self.tokenizer,
            Some(ChatTemplateRendererV1::generic_with_config(
                self.renderer.provider(),
                self.renderer.config().clone(),
            )),
            &self.stop_policy,
        )
        .map_err(|_| ChatBackendErrorV1::Failed)?;
        let input = service
            .prepare_input(&ServiceGenerationInputV1::Messages {
                messages: request.messages.clone(),
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: true,
                    thinking,
                },
            })
            .map_err(|_| ChatBackendErrorV1::Failed)?;
        if input.is_empty() {
            return Err(ChatBackendErrorV1::Failed);
        }
        let input_len = u64::try_from(input.len()).map_err(|_| ChatBackendErrorV1::Failed)?;
        let state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or(ChatBackendErrorV1::Failed)?;
        if state_capacity > self.context_length {
            return Err(ChatBackendErrorV1::Failed);
        }
        let checkpoint = self.current_checkpoint.clone();
        let checkpoint = if let Some(checkpoint) = checkpoint {
            let compatible_prefix =
                gemma4_moe_checkpoint_suffix(&input, &checkpoint.payload.token_history).is_ok();
            let compatible_capacity = Self::checkpoint_capacity(&checkpoint.header.identity)
                .ok()
                .is_some_and(|capacity| state_capacity <= capacity);
            if !compatible_prefix || !compatible_capacity {
                if self.checkpoint_loaded_explicitly {
                    return Err(ChatBackendErrorV1::CheckpointUnavailable);
                }
                // Keep the prior committed checkpoint until this fresh
                // request commits successfully; cancellation or execution
                // failure must leave the last turn recoverable.
                None
            } else {
                Some(checkpoint)
            }
        } else {
            None
        };
        let (mut executor, execution_state_capacity) = if let Some(checkpoint) = checkpoint.as_ref()
        {
            let prefix_len = checkpoint.payload.token_history.len();
            let checkpoint_capacity = Self::checkpoint_capacity(&checkpoint.header.identity)
                .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)?;
            let graph = build_gemma4_moe_gguf_graph(&self.source, 1, 0, checkpoint_capacity)
                .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)?;
            let resident = self.resident.as_ref().ok_or(ChatBackendErrorV1::Failed)?;
            let owner = resident
                .new_request_from_checkpoint(checkpoint, graph, &checkpoint.header.identity)
                .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)?;
            (
                CliGemma4MoeExecutor::new_restored_with_prefix(owner, prefix_len),
                checkpoint_capacity,
            )
        } else {
            let execution_state_capacity = state_capacity.max(GEMMA4_MOE_PREFILL_CHUNK_TOKENS);
            let graph = build_gemma4_moe_gguf_graph(
                &self.source,
                input_len.min(GEMMA4_MOE_PREFILL_CHUNK_TOKENS),
                0,
                execution_state_capacity,
            )
            .map_err(|_| ChatBackendErrorV1::Failed)?;
            let resident = self.resident.as_ref().ok_or(ChatBackendErrorV1::Failed)?;
            let owner = resident
                .new_request(graph)
                .map_err(|_| ChatBackendErrorV1::Failed)?;
            (CliGemma4MoeExecutor::new(owner), execution_state_capacity)
        };
        let config = GenerationConfigV1::new(
            request.max_new_tokens,
            SamplingParametersV1::greedy(),
            request.stop_sequences.clone(),
        )
        .map_err(|_| ChatBackendErrorV1::Failed)?;
        let cancellation = self.active_cancellation.take().unwrap_or_default();
        let mut random =
            OsSamplingRandom::for_parameters_and_seed(SamplingParametersV1::greedy(), None)
                .map_err(|_| ChatBackendErrorV1::Failed)?;
        let result =
            service.generate_tokens(&mut executor, &input, &config, &cancellation, &mut random);
        let result = match result {
            Ok(_result) if cancellation.is_cancelled() => {
                return Err(ChatBackendErrorV1::Cancelled);
            }
            Ok(result) => result,
            Err(_) if cancellation.is_cancelled() => return Err(ChatBackendErrorV1::Cancelled),
            Err(_) => return Err(ChatBackendErrorV1::Failed),
        };
        if cancellation.is_cancelled() {
            return Err(ChatBackendErrorV1::Cancelled);
        }
        executor
            .audit_json(&self.target)
            .map_err(|_| ChatBackendErrorV1::Failed)?;
        // Rebase to the completed rendered history before publishing a
        // checkpoint. Generation leaves its last selected token unconsumed;
        // rebuilding from the canonical assistant transcript gives the
        // checkpoint an exact token/state correspondence and also makes the
        // next persistent turn eligible for suffix restore.
        let checkpoint_text =
            gemma4_moe_checkpoint_visible_text(result.output_text(), &request.reverse_prompts);
        self.stage_checkpoint_state(
            &request.messages,
            &checkpoint_text,
            execution_state_capacity,
        )?;
        let result = json!({
            "output_text": result.output_text(),
            "finish_reason": result.finish_reason().as_str(),
        });
        let text = result
            .get("output_text")
            .and_then(Value::as_str)
            .ok_or(ChatBackendErrorV1::Failed)?
            .to_owned();
        let finish_reason = match result
            .get("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("length")
        {
            "stop" => ChatFinishReasonV1::Stop,
            "reverse_prompt" => ChatFinishReasonV1::ReversePrompt,
            "length" => ChatFinishReasonV1::Length,
            _ => return Err(ChatBackendErrorV1::Failed),
        };
        Ok(ChatGenerationResultV1 {
            text,
            reasoning: None,
            finish_reason,
            cancelled: false,
        })
    }

    fn save_checkpoint(
        &mut self,
        name: &str,
        conversation: &[u8],
    ) -> Result<(), ChatBackendErrorV1> {
        let checkpoint = self.candidate_checkpoint(conversation)?;
        self.checkpoint_store
            .save(name, &checkpoint)
            .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)?;
        self.current_checkpoint = Some(checkpoint);
        self.checkpoint_loaded_explicitly = false;
        self.pending_state = None;
        Ok(())
    }

    fn load_checkpoint(&mut self, name: &str) -> Result<Option<Vec<u8>>, ChatBackendErrorV1> {
        self.load_named_checkpoint(name).map(Some)
    }

    fn commit_turn(&mut self, conversation: &[u8]) -> Result<(), ChatBackendErrorV1> {
        let checkpoint = self.candidate_checkpoint(conversation)?;
        self.current_checkpoint = Some(checkpoint);
        self.checkpoint_loaded_explicitly = false;
        self.pending_state = None;
        Ok(())
    }

    fn abort_turn(&mut self) -> Result<(), ChatBackendErrorV1> {
        // The request-local owner is dropped by generate on every return path;
        // only the previously committed checkpoint remains reusable.
        self.pending_state = None;
        Ok(())
    }
}

struct GemmaProductionBackend {
    lock: Gemma4ModelLock,
    source: Arc<VerifiedGgufGemmaSource>,
    mtp_assistant_lock: Option<Gemma4MtpModelLock>,
    mtp_assistant_source: Option<Arc<sllm_core::VerifiedGgufGemma4Mtp>>,
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
        let (mtp_assistant_lock, mtp_assistant_source) = match (
            request.mtp_assistant_gguf.as_ref(),
            request.mtp_assistant_derived_lock.as_ref(),
        ) {
            (None, None) => (None, None),
            (Some(gguf_path), Some(derived_path)) => {
                let assistant_lock = parse_gemma4_mtp_model_lock(include_bytes!(
                    "../../../docs/models/locks/gemma4-12b-it-assistant-bf16.json"
                ))
                .map_err(|error| format!("Gemma MTP assistant lock is invalid: {error}"))?;
                let expected_semantic_id = format!(
                    "gemma4mtp-pair:{}:{}",
                    lock.fingerprint(),
                    assistant_lock.fingerprint()
                );
                let derived = read_derived_gguf_lock(derived_path)
                    .map_err(|error| format!("Gemma MTP derived lock is invalid: {error}"))?;
                if derived.semantic_model_id != expected_semantic_id
                    || derived.source_lock_fingerprints
                        != [
                            lock.fingerprint().to_owned(),
                            assistant_lock.fingerprint().to_owned(),
                        ]
                {
                    return Err(
                        "Gemma MTP assistant derived lock is not the canonical target/assistant pair"
                            .to_owned(),
                    );
                }
                let verified = verify_derived_gguf(derived, gguf_path).map_err(|error| {
                    format!("Gemma MTP assistant GGUF does not match its derived lock: {error}")
                })?;
                let source = verify_gguf_gemma4_mtp(verified.gguf, &assistant_lock, &lock)
                    .map_err(|error| format!("Gemma MTP assistant GGUF is invalid: {error}"))?;
                (Some(assistant_lock), Some(Arc::new(source)))
            }
            _ => {
                return Err(
                    "Gemma MTP assistant source requires both --mtp-assistant-gguf and --mtp-assistant-derived-lock"
                        .to_owned(),
                );
            }
        };
        Ok(Self {
            lock,
            source,
            mtp_assistant_lock,
            mtp_assistant_source,
        })
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
        if request.prefill_chunk_tokens.is_some() {
            return Err("--prefill-chunk-tokens is unsupported for Gemma 4 generation".to_owned());
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
        let mtp_plan = resolve_cli_gemma4_mtp_plan(
            request.mtp_draft_width,
            &request.target,
            request.kv_cache_encoding,
            request.sampling,
            self.lock.fingerprint(),
        )?;
        let state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or_else(|| "generation state capacity overflowed".to_owned())?;
        if mtp_plan.enabled && state_capacity > GEMMA4_MTP_CONTEXT_TOKENS {
            return Err(format!(
                "forced Gemma MTP context {} exceeds the initial reviewed limit {}",
                state_capacity, GEMMA4_MTP_CONTEXT_TOKENS
            ));
        }
        let (allocated_state_capacity, mtp_state_slack_tokens) = if mtp_plan.enabled {
            if self.mtp_assistant_lock.is_none() || self.mtp_assistant_source.is_none() {
                return Err(
                    "forced Gemma MTP requires both verified target-paired assistant GGUF flags"
                        .to_owned(),
                );
            }
            (
                state_capacity
                    .checked_add(1)
                    .ok_or_else(|| "Gemma MTP state capacity overflowed".to_owned())?,
                1_u64,
            )
        } else {
            (state_capacity, 0_u64)
        };
        let plan = self
            .source
            .build_weight_load_plan(&self.lock)
            .map_err(|_| "GGUF tensors do not form the Gemma 4 load plan".to_owned())?;
        let plan_digest = plan.digest_hex();
        let model_fingerprint = self.lock.fingerprint().to_owned();
        let mtp_assistant_plan = if mtp_plan.enabled {
            let lock = self
                .mtp_assistant_lock
                .as_ref()
                .ok_or_else(|| "Gemma MTP assistant lock is unavailable".to_owned())?;
            let source = self
                .mtp_assistant_source
                .as_ref()
                .ok_or_else(|| "Gemma MTP assistant source is unavailable".to_owned())?;
            Some(
                build_verified_gemma4_mtp_weight_load_plan(lock, source.as_ref()).map_err(
                    |error| format!("Gemma MTP assistant load plan is invalid: {error}"),
                )?,
            )
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
                .new_request(input_len, allocated_state_capacity)
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
            let (
                report,
                audit,
                mtp_proposal_blocks,
                mtp_proposed_draft_tokens,
                mtp_accepted_draft_tokens,
            ) = if mtp_plan.enabled {
                let assistant_lock = self
                    .mtp_assistant_lock
                    .as_ref()
                    .ok_or_else(|| "Gemma MTP assistant lock is unavailable".to_owned())?;
                let assistant_source = self
                    .mtp_assistant_source
                    .as_ref()
                    .ok_or_else(|| "Gemma MTP assistant source is unavailable".to_owned())?;
                let assistant_plan = mtp_assistant_plan
                    .as_ref()
                    .ok_or_else(|| "Gemma MTP assistant load plan is unavailable".to_owned())?;
                if input_len == 0 {
                    return Err("Gemma MTP requires a non-empty tokenized prompt".to_owned());
                }
                let assistant_graph = build_gemma4_mtp_graph(
                    assistant_lock,
                    assistant_source.as_ref(),
                    assistant_plan,
                    input_len,
                    input_len - 1,
                )
                .map_err(|error| format!("Gemma MTP request graph failed: {error}"))?;
                let assistant_resident = Gemma4MtpResidentModel::provision(
                    Arc::clone(&session),
                    assistant_lock,
                    assistant_source.as_ref(),
                    assistant_plan.clone(),
                    COMPLETION_TIMEOUT,
                )
                .map_err(|error| {
                    format!("Gemma MTP assistant resident provisioning failed: {error}")
                })?;
                let assistant_owner =
                    assistant_resident
                        .new_request(&assistant_graph)
                        .map_err(|error| {
                            format!("Gemma MTP assistant request provisioning failed: {error}")
                        })?;
                let executor =
                    Gemma4MtpGenerationExecutorV1::new_with_draft_width(owner, assistant_owner, 1)
                        .map_err(|error| {
                            format!("Gemma MTP executor could not be constructed: {error}")
                        })?;
                let mut adapter = SpeculativeGenerationAdapterV1::new(executor);
                let report = service
                    .generate_tokens(&mut adapter, &input, &config, &cancellation, &mut random)
                    .map_err(|error| format!("Gemma MTP generation service failed: {error}"))?;
                let accounting = adapter.accounting();
                let audit = adapter
                    .inner()
                    .target()
                    .audit_snapshot()
                    .map_err(|_| "Gemma MTP dispatch audit was empty or invalid".to_owned())?;
                (
                    report,
                    audit,
                    Some(accounting.proposal_blocks()),
                    Some(accounting.proposed_tokens()),
                    Some(accounting.accepted_tokens()),
                )
            } else {
                let report = service
                    .generate_tokens(&mut owner, &input, &config, &cancellation, &mut random)
                    .map_err(|error| format!("generation service failed: {error}"))?;
                let audit = owner
                    .audit_snapshot()
                    .map_err(|_| "Gemma dispatch audit was empty or invalid".to_owned())?;
                (report, audit, None, None, None)
            };
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
                    "logical_state_capacity_tokens": state_capacity,
                    "allocated_state_capacity_tokens": allocated_state_capacity,
                    "mtp_state_slack_tokens": mtp_state_slack_tokens,
                    "decode_steps": report.decode_steps(),
                    "mtp_selection": mtp_plan.selection,
                    "mtp_draft_width_requested": mtp_plan.requested_width,
                    "mtp_draft_width_effective": mtp_plan.effective_width,
                    "mtp_proposal_blocks": mtp_proposal_blocks,
                    "mtp_proposed_draft_tokens": mtp_proposed_draft_tokens,
                    "mtp_accepted_draft_tokens": mtp_accepted_draft_tokens,
                    "mtp_rejected_draft_tokens": match (mtp_proposed_draft_tokens, mtp_accepted_draft_tokens) {
                        (Some(proposed), Some(accepted)) => Some(proposed.saturating_sub(accepted)),
                        _ => None,
                    },
                    "fallback_used": audit.fallback_used(),
                    "submission_count": audit.submission_count(),
                    "kernel_dispatch_count": audit.kernel_dispatch_count(),
                    "segment_count": audit.segment_count(),
                    "boundary_count": audit.boundary_count(),
                    "all_dispatches_hip": true,
                    "weight_encoding": if mtp_plan.enabled {
                        "mixed-nvfp4-w4a4-fp8-w8a8+gemma4-mtp-bf16"
                    } else {
                        "mixed-nvfp4-w4a4-fp8-w8a8"
                    },
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

    fn benchmark(
        &self,
        request: &BenchmarkRequest,
        timing: BenchmarkTiming,
    ) -> Result<Value, String> {
        if request.model_size != "12B" {
            return Err("Gemma 4 benchmark requires --model-size 12B".to_owned());
        }
        if !request.greedy {
            return Err("Gemma 4 benchmark requires explicit --greedy mode".to_owned());
        }
        if request.lane != BenchmarkLane::Direct {
            return Err(
                "Gemma 4 benchmark currently supports only the direct pretokenized lane".to_owned(),
            );
        }
        if request.prefill_chunk_tokens.is_some()
            || request.fp8_manifest.is_some()
            || request.fp8_artifact.is_some()
            || request.fp8_provider.is_some()
        {
            return Err(
                "Gemma 4 benchmark does not accept sidecar or prefill-chunk overrides".to_owned(),
            );
        }
        if request
            .kv_cache_encoding
            .is_some_and(|encoding| encoding != KvCacheEncoding::Fp16)
        {
            return Err("Gemma 4 benchmark requires the FP16 KV cache encoding".to_owned());
        }
        validate_benchmark_protocol(request.warmups, request.measured)?;
        let completion_timeout = benchmark_completion_timeout(request.completion_timeout_seconds)?;
        let seed_input = match &request.input {
            BenchmarkInput::TokenIds(ids) => ids.clone(),
            BenchmarkInput::Messages { .. } => {
                return Err("Gemma 4 benchmark direct lane requires pretokenized input".to_owned());
            }
        };
        if seed_input.is_empty() {
            return Err("Gemma 4 benchmark input token IDs must not be empty".to_owned());
        }
        let input_len = u64::try_from(seed_input.len())
            .map_err(|_| "Gemma 4 benchmark input length overflowed".to_owned())?;
        let logical_state_capacity =
            benchmark_state_capacity(input_len, request.max_new_tokens, request.context_length)?;
        let mtp_plan = resolve_cli_gemma4_mtp_plan(
            request.mtp_draft_width,
            &request.target,
            request.kv_cache_encoding,
            SamplingParametersV1::greedy(),
            self.lock.fingerprint(),
        )?;
        if mtp_plan.enabled && logical_state_capacity > GEMMA4_MTP_CONTEXT_TOKENS {
            return Err(format!(
                "forced Gemma MTP context {logical_state_capacity} exceeds the initial reviewed limit {GEMMA4_MTP_CONTEXT_TOKENS}"
            ));
        }
        let (state_capacity, mtp_state_slack_tokens) = if mtp_plan.enabled {
            if self.mtp_assistant_lock.is_none() || self.mtp_assistant_source.is_none() {
                return Err(
                    "forced Gemma MTP requires both verified target-paired assistant GGUF flags"
                        .to_owned(),
                );
            }
            (
                logical_state_capacity
                    .checked_add(1)
                    .ok_or_else(|| "Gemma MTP state capacity overflowed".to_owned())?,
                1_u64,
            )
        } else {
            (logical_state_capacity, 0_u64)
        };
        let target_plan = self
            .source
            .build_weight_load_plan(&self.lock)
            .map_err(|error| format!("Gemma 4 benchmark target load plan is invalid: {error}"))?;
        let target_plan_digest = target_plan.digest_hex();
        let mtp_assistant_plan = if mtp_plan.enabled {
            let assistant_lock = self
                .mtp_assistant_lock
                .as_ref()
                .ok_or_else(|| "Gemma MTP assistant lock is unavailable".to_owned())?;
            let assistant_source = self
                .mtp_assistant_source
                .as_ref()
                .ok_or_else(|| "Gemma MTP assistant source is unavailable".to_owned())?;
            Some(
                build_verified_gemma4_mtp_weight_load_plan(
                    assistant_lock,
                    assistant_source.as_ref(),
                )
                .map_err(|error| format!("Gemma MTP assistant load plan is invalid: {error}"))?,
            )
        } else {
            None
        };
        let stop_policy = gemma4_generation_stop_policy(&self.lock)
            .map_err(|error| format!("Gemma stop policy is invalid: {error}"))?;
        let stop_policy = benchmark_stop_policy(&stop_policy, request.ignore_eos);
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(request.device_index, request.target.clone())
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| format!("exact HIP execution session could not be opened: {error}"))?;
        let model_load_start_ns = timing.model_load_start_ns();
        let execution = (|| -> Result<Value, String> {
            let target_resident = Gemma4ResidentModel::new_gguf_quantized(
                Arc::clone(&session),
                self.lock.clone(),
                target_plan.clone(),
                Arc::clone(&self.source),
                completion_timeout,
            )
            .map_err(|error| format!("Gemma resident provisioning failed: {error}"))?;
            let assistant_resident = if mtp_plan.enabled {
                let assistant_lock = self
                    .mtp_assistant_lock
                    .as_ref()
                    .ok_or_else(|| "Gemma MTP assistant lock is unavailable".to_owned())?;
                let assistant_source = self
                    .mtp_assistant_source
                    .as_ref()
                    .ok_or_else(|| "Gemma MTP assistant source is unavailable".to_owned())?;
                let assistant_plan = mtp_assistant_plan
                    .as_ref()
                    .ok_or_else(|| "Gemma MTP assistant load plan is unavailable".to_owned())?;
                Some(
                    Gemma4MtpResidentModel::provision(
                        Arc::clone(&session),
                        assistant_lock,
                        assistant_source.as_ref(),
                        assistant_plan.clone(),
                        completion_timeout,
                    )
                    .map_err(|error| {
                        format!("Gemma MTP assistant resident provisioning failed: {error}")
                    })?,
                )
            } else {
                None
            };
            let model_ready_ns = timing.now_ns();
            let model_ready_memory = allocation_snapshot_value(session.memory_snapshot());
            let model_resident_high_water_bytes =
                validate_model_ready_snapshot(&model_ready_memory)?;
            let ready_model_current_bytes =
                session.memory_snapshot().model_resident().current_bytes();
            let run_sample = |sample_index: u32| -> Result<Value, String> {
                let request_start_ns = timing.now_ns();
                let request_memory: Value;
                let (outcome, audit, mtp_accounting) = if mtp_plan.enabled {
                    let assistant_lock = self
                        .mtp_assistant_lock
                        .as_ref()
                        .ok_or_else(|| "Gemma MTP assistant lock is unavailable".to_owned())?;
                    let assistant_resident = assistant_resident
                        .as_ref()
                        .ok_or_else(|| "Gemma MTP assistant resident is unavailable".to_owned())?;
                    let assistant_source = self
                        .mtp_assistant_source
                        .as_ref()
                        .ok_or_else(|| "Gemma MTP assistant source is unavailable".to_owned())?;
                    let assistant_plan = mtp_assistant_plan
                        .as_ref()
                        .ok_or_else(|| "Gemma MTP assistant load plan is unavailable".to_owned())?;
                    let assistant_graph = build_gemma4_mtp_graph(
                        assistant_lock,
                        assistant_source.as_ref(),
                        assistant_plan,
                        input_len,
                        input_len - 1,
                    )
                    .map_err(|error| format!("Gemma MTP request graph failed: {error}"))?;
                    let target = target_resident
                        .new_request(input_len, state_capacity)
                        .map_err(|error| {
                            format!("Gemma benchmark request provisioning failed: {error}")
                        })?;
                    let assistant =
                        assistant_resident
                            .new_request(&assistant_graph)
                            .map_err(|error| {
                                format!("Gemma MTP assistant request provisioning failed: {error}")
                            })?;
                    request_memory = allocation_snapshot_value(session.memory_snapshot());
                    let executor =
                        Gemma4MtpGenerationExecutorV1::new_with_draft_width(target, assistant, 1)
                            .map_err(|error| {
                            format!("Gemma MTP executor could not be constructed: {error}")
                        })?;
                    let mut adapter = SpeculativeGenerationAdapterV1::new(executor);
                    let mut timeline = BenchmarkTimeline::new(request_start_ns);
                    let outcome = run_generation_executor_timed(
                        &mut adapter,
                        &stop_policy,
                        request.max_new_tokens,
                        seed_input.as_slice(),
                        (&mut timeline, timing.request_clock()),
                    )?;
                    let accounting = adapter.accounting();
                    let audit = adapter.inner().target().audit_snapshot().map_err(|error| {
                        format!("Gemma MTP dispatch audit was empty or invalid: {error}")
                    })?;
                    let sample = timeline.finish(BenchmarkSampleInput {
                        input_token_ids: seed_input.as_slice(),
                        generated_token_ids: outcome.report.generated_token_ids(),
                        visible_token_ids: outcome.report.visible_token_ids(),
                        decode_input_token_ids: outcome.report.decode_input_token_ids(),
                        stop: json!({
                            "version": outcome.report.stop_reason().map(|stop| stop.version()),
                            "reason_version": outcome.report.stop_reason().map(|stop| stop.reason_version()),
                            "kind": outcome.report.reason_token(),
                            "token_id": outcome.report.stop_token_id(),
                        }),
                        audit: json!({
                            "selected_backend": "hip",
                            "target": audit.target(),
                            "device_index": request.device_index,
                            "model_fingerprint": self.lock.fingerprint(),
                            "plan_digest": target_plan_digest,
                            "fallback_used": audit.fallback_used(),
                            "submission_count": audit.submission_count(),
                            "kernel_dispatch_count": audit.kernel_dispatch_count(),
                            "segment_count": audit.segment_count(),
                            "boundary_count": audit.boundary_count(),
                            "all_dispatches_hip": true,
                            "mtp_proposal_blocks": accounting.proposal_blocks(),
                            "mtp_proposed_draft_tokens": accounting.proposed_tokens(),
                            "mtp_accepted_draft_tokens": accounting.accepted_tokens(),
                            "mtp_rejected_draft_tokens": accounting.rejected_tokens(),
                        }),
                        memory: json!({
                            "sample_index": sample_index,
                            "request_start": request_memory,
                        }),
                        cleanup: json!({
                            "sample_index": sample_index,
                            "request_dropped": true,
                            "allocator_cleanup_validated": true,
                            "retryable_cleanup": 0,
                            "durable_quarantine": 0,
                        }),
                    })?;
                    drop(adapter);
                    let cleanup_memory = allocation_snapshot_value(session.memory_snapshot());
                    validate_request_cleanup_snapshot(&cleanup_memory, ready_model_current_bytes)?;
                    let mut sample = sample;
                    sample["memory"]["after_cleanup"] = cleanup_memory;
                    (sample, audit, Some(accounting))
                } else {
                    let mut owner = target_resident
                        .new_request(input_len, state_capacity)
                        .map_err(|error| {
                            format!("Gemma benchmark request provisioning failed: {error}")
                        })?;
                    request_memory = allocation_snapshot_value(session.memory_snapshot());
                    let mut timeline = BenchmarkTimeline::new(request_start_ns);
                    let outcome = run_generation_executor_timed(
                        &mut owner,
                        &stop_policy,
                        request.max_new_tokens,
                        seed_input.as_slice(),
                        (&mut timeline, timing.request_clock()),
                    )?;
                    let audit = owner.audit_snapshot().map_err(|error| {
                        format!("Gemma dispatch audit was empty or invalid: {error}")
                    })?;
                    let sample = timeline.finish(BenchmarkSampleInput {
                        input_token_ids: seed_input.as_slice(),
                        generated_token_ids: outcome.report.generated_token_ids(),
                        visible_token_ids: outcome.report.visible_token_ids(),
                        decode_input_token_ids: outcome.report.decode_input_token_ids(),
                        stop: json!({
                            "version": outcome.report.stop_reason().map(|stop| stop.version()),
                            "reason_version": outcome.report.stop_reason().map(|stop| stop.reason_version()),
                            "kind": outcome.report.reason_token(),
                            "token_id": outcome.report.stop_token_id(),
                        }),
                        audit: json!({
                            "selected_backend": "hip",
                            "target": audit.target(),
                            "device_index": request.device_index,
                            "model_fingerprint": self.lock.fingerprint(),
                            "plan_digest": target_plan_digest,
                            "fallback_used": audit.fallback_used(),
                            "submission_count": audit.submission_count(),
                            "kernel_dispatch_count": audit.kernel_dispatch_count(),
                            "segment_count": audit.segment_count(),
                            "boundary_count": audit.boundary_count(),
                            "all_dispatches_hip": true,
                            "mtp_proposal_blocks": null,
                            "mtp_proposed_draft_tokens": null,
                            "mtp_accepted_draft_tokens": null,
                            "mtp_rejected_draft_tokens": null,
                        }),
                        memory: json!({
                            "sample_index": sample_index,
                            "request_start": request_memory,
                        }),
                        cleanup: json!({
                            "sample_index": sample_index,
                            "request_dropped": true,
                            "allocator_cleanup_validated": true,
                            "retryable_cleanup": 0,
                            "durable_quarantine": 0,
                        }),
                    })?;
                    drop(owner);
                    let cleanup_memory = allocation_snapshot_value(session.memory_snapshot());
                    validate_request_cleanup_snapshot(&cleanup_memory, ready_model_current_bytes)?;
                    let mut sample = sample;
                    sample["memory"]["after_cleanup"] = cleanup_memory;
                    (sample, audit, None)
                };
                if audit.target() != request.target || audit.fallback_used() {
                    return Err(
                        "Gemma benchmark dispatch audit is not exact HIP/no-fallback".to_owned(),
                    );
                }
                let _ = mtp_accounting;
                Ok(outcome)
            };
            let mut samples = Vec::with_capacity((request.warmups + request.measured) as usize);
            for index in 0..(request.warmups + request.measured) {
                samples.push(run_sample(index)?);
            }
            let warmup_samples = samples[..request.warmups as usize].to_vec();
            let measured_samples = samples[request.warmups as usize..].to_vec();
            let correctness_control = correctness_reference_from_warmup(
                warmup_samples
                    .first()
                    .ok_or_else(|| "benchmark requires a warmup sample".to_owned())?,
            )?;
            for sample in warmup_samples.iter().skip(1).chain(measured_samples.iter()) {
                compare_control_sample(&correctness_control, sample)?;
            }
            let mut submission_count = 0_u64;
            let mut kernel_dispatch_count = 0_u64;
            let mut segment_count = 0_u64;
            let mut boundary_count = 0_u64;
            let mut mtp_proposal_blocks = 0_u64;
            let mut mtp_proposed_draft_tokens = 0_u64;
            let mut mtp_accepted_draft_tokens = 0_u64;
            let mut mtp_rejected_draft_tokens = 0_u64;
            for sample in &samples {
                let audit = sample
                    .get("audit")
                    .and_then(Value::as_object)
                    .ok_or_else(|| "benchmark sample audit was not an object".to_owned())?;
                submission_count = submission_count
                    .checked_add(
                        audit["submission_count"]
                            .as_u64()
                            .ok_or_else(|| "benchmark submission count was missing".to_owned())?,
                    )
                    .ok_or_else(|| "benchmark submission count overflowed".to_owned())?;
                kernel_dispatch_count = kernel_dispatch_count
                    .checked_add(
                        audit["kernel_dispatch_count"]
                            .as_u64()
                            .ok_or_else(|| "benchmark dispatch count was missing".to_owned())?,
                    )
                    .ok_or_else(|| "benchmark dispatch count overflowed".to_owned())?;
                segment_count = segment_count
                    .checked_add(
                        audit["segment_count"]
                            .as_u64()
                            .ok_or_else(|| "benchmark segment count was missing".to_owned())?,
                    )
                    .ok_or_else(|| "benchmark segment count overflowed".to_owned())?;
                boundary_count = boundary_count
                    .checked_add(
                        audit["boundary_count"]
                            .as_u64()
                            .ok_or_else(|| "benchmark boundary count was missing".to_owned())?,
                    )
                    .ok_or_else(|| "benchmark boundary count overflowed".to_owned())?;
                for (field, total) in [
                    ("mtp_proposal_blocks", &mut mtp_proposal_blocks),
                    ("mtp_proposed_draft_tokens", &mut mtp_proposed_draft_tokens),
                    ("mtp_accepted_draft_tokens", &mut mtp_accepted_draft_tokens),
                    ("mtp_rejected_draft_tokens", &mut mtp_rejected_draft_tokens),
                ] {
                    if let Some(value) = audit.get(field).and_then(Value::as_u64) {
                        *total = total
                            .checked_add(value)
                            .ok_or_else(|| "benchmark MTP accounting overflowed".to_owned())?;
                    }
                }
            }
            drop(assistant_resident);
            drop(target_resident);
            let final_memory = allocation_snapshot_value(session.memory_snapshot());
            validate_resident_drop_snapshot(&final_memory)?;
            validate_peak_vram_snapshot(&final_memory, model_resident_high_water_bytes)?;
            Ok(json!({
                "benchmark_schema_version": request.lane.schema_version(),
                "state": "PASS",
                "lane": "direct",
                "lane_definition": "Gemma 4 12B Dense exact HIP greedy generation",
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
                        "repo_id": self.lock.model.repo_id,
                        "resolved_revision": self.lock.model.resolved_revision,
                        "lock_fingerprint": self.lock.fingerprint(),
                    },
                    "binding": {
                        "model_fingerprint": self.lock.fingerprint(),
                        "plan_digest": target_plan_digest,
                    },
                },
                "model_load": {
                    "event": "model_load",
                    "start_ns": model_load_start_ns,
                    "model_ready_ns": model_ready_ns,
                    "duration_ns": model_ready_ns.saturating_sub(model_load_start_ns),
                    "load_count": 1,
                },
                "config": {
                    "input_token_ids": seed_input.as_slice(),
                    "input_token_count": seed_input.len(),
                    "max_new_tokens": request.max_new_tokens,
                    "ignore_eos": request.ignore_eos,
                    "context_length": request.context_length,
                    "effective_context_length": logical_state_capacity,
                    "allocated_state_capacity": state_capacity,
                    "mtp_state_slack_tokens": mtp_state_slack_tokens,
                    "greedy": true,
                    "warmups": request.warmups,
                    "measured": request.measured,
                    "kv_cache_encoding": "fp16",
                    "mtp_selection": mtp_plan.selection,
                    "mtp_draft_width_requested": mtp_plan.requested_width,
                    "mtp_draft_width_effective": mtp_plan.effective_width,
                },
                "memory": {
                    "model_ready": model_ready_memory,
                    "after_model_drop": final_memory,
                    "model_resident_high_water_bytes": model_resident_high_water_bytes,
                    "resident_vram_bytes": model_resident_high_water_bytes,
                    "resident_vram_source": "model_resident_allocator_high_water",
                    "peak_vram_bytes": final_memory["high_water_bytes"],
                    "peak_source": "runtime_allocator",
                },
                "audit": {
                    "selected_backend": "hip",
                    "target": request.target,
                    "device_index": request.device_index,
                    "model_fingerprint": self.lock.fingerprint(),
                    "plan_digest": target_plan_digest,
                    "submission_count": submission_count,
                    "kernel_dispatch_count": kernel_dispatch_count,
                    "segment_count": segment_count,
                    "boundary_count": boundary_count,
                    "fallback_used": false,
                    "all_dispatches_hip": true,
                    "model_load_count": 1,
                    "model_reused": true,
                    "sample_count": request.warmups + request.measured,
                    "weight_encoding": if mtp_plan.enabled {
                        "mixed-nvfp4-w4a4-fp8-w8a8+gemma4-mtp-bf16"
                    } else {
                        "mixed-nvfp4-w4a4-fp8-w8a8"
                    },
                    "mtp_proposal_blocks": mtp_plan.enabled.then_some(mtp_proposal_blocks),
                    "mtp_proposed_draft_tokens": mtp_plan.enabled.then_some(mtp_proposed_draft_tokens),
                    "mtp_accepted_draft_tokens": mtp_plan.enabled.then_some(mtp_accepted_draft_tokens),
                    "mtp_rejected_draft_tokens": mtp_plan.enabled.then_some(mtp_rejected_draft_tokens),
                },
                "cleanup": {
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
        if cleanup.retryable_cleanup != 0 || cleanup.durable_quarantine != 0 {
            return Err("HIP session cleanup was not empty".to_owned());
        }
        result["session_cleanup"] = json!({
            "retryable_cleanup": cleanup.retryable_cleanup,
            "durable_quarantine": cleanup.durable_quarantine,
        });
        Ok(result)
    }
}

fn open_production_backend(request: &Request) -> Result<Box<dyn ModelFrontendBackend>, String> {
    let gguf_path = request
        .gguf
        .as_ref()
        .expect("public parser requires a GGUF path");
    let Some(derived_path) = request.derived_lock.as_ref() else {
        if request.mtp_assistant_gguf.is_some() || request.mtp_assistant_derived_lock.is_some() {
            return Err(
                "Ministral 3 direct official GGUF does not support MTP assistants".to_owned(),
            );
        }
        return Ministral3ProductionBackend::open(gguf_path)
            .map(|backend| Box::new(backend) as Box<_>);
    };
    let derived = read_derived_gguf_lock(derived_path)
        .map_err(|error| format!("derived GGUF lock is invalid: {error}"))?;
    if derived.semantic_model_id.starts_with("qwen35moe:") {
        if request.mtp_assistant_gguf.is_some() || request.mtp_assistant_derived_lock.is_some() {
            return Err(
                "--mtp-assistant-gguf/--mtp-assistant-derived-lock are supported only for dense Gemma 4"
                    .to_owned(),
            );
        }
        let verified = verify_derived_gguf(derived, gguf_path)
            .map_err(|error| format!("GGUF does not match its derived lock: {error}"))?;
        let source = verify_gguf_qwen35_moe(verified)
            .map_err(|error| format!("MoE GGUF is invalid: {error}"))?;
        return Ok(Box::new(MoeProductionBackend {
            source: Arc::new(source),
        }));
    }
    if derived.semantic_model_id.starts_with("gemma4moe:") {
        if request.mtp_assistant_gguf.is_some() || request.mtp_assistant_derived_lock.is_some() {
            return Err(
                "--mtp-assistant-gguf/--mtp-assistant-derived-lock are supported only for dense Gemma 4"
                    .to_owned(),
            );
        }
        let verified = verify_derived_gguf(derived, gguf_path)
            .map_err(|error| format!("GGUF does not match its derived lock: {error}"))?;
        let source = verify_gguf_gemma4_moe(verified)
            .map_err(|error| format!("Gemma 4 MoE GGUF is invalid: {error}"))?;
        return Ok(Box::new(Gemma4MoeProductionBackend {
            source: Arc::new(source),
        }));
    }
    let reviewed = builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
        .map_err(|error| format!("derived GGUF source identity is unsupported: {error}"))?;
    match reviewed {
        ReviewedModelLock::Qwen35(lock) => {
            if request.mtp_assistant_gguf.is_some() || request.mtp_assistant_derived_lock.is_some()
            {
                return Err(
                    "--mtp-assistant-gguf/--mtp-assistant-derived-lock are supported only for dense Gemma 4"
                        .to_owned(),
                );
            }
            ProductionBackend::open(lock, request).map(|backend| Box::new(backend) as Box<_>)
        }
        ReviewedModelLock::Gemma4(lock) => {
            GemmaProductionBackend::open(lock, request).map(|backend| Box::new(backend) as Box<_>)
        }
        ReviewedModelLock::Ministral3(_) => {
            Err("Ministral 3 is a direct official GGUF and must omit --derived-lock".to_owned())
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
    let mut mtp_assistant_gguf = None;
    let mut mtp_assistant_derived_lock = None;
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
            "--mtp-assistant-gguf" if command == "generate" || command == "benchmark" => set_once(
                &mut mtp_assistant_gguf,
                take_value(&mut arguments, "--mtp-assistant-gguf")?,
                "--mtp-assistant-gguf",
            )?,
            "--mtp-assistant-derived-lock" if command == "generate" || command == "benchmark" => {
                set_once(
                    &mut mtp_assistant_derived_lock,
                    take_value(&mut arguments, "--mtp-assistant-derived-lock")?,
                    "--mtp-assistant-derived-lock",
                )?
            }
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
            "--mtp-draft-width" if command == "generate" || command == "benchmark" => {
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
                if !matches!(value.as_str(), "2B" | "4B" | "9B" | "12B" | "26B-A4B") {
                    return Err("--model-size must be 2B, 4B, 9B, 12B, or 26B-A4B".to_owned());
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
    let derived_lock = derived_lock.map(PathBuf::from);
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
            match (
                mtp_assistant_gguf.as_ref(),
                mtp_assistant_derived_lock.as_ref(),
            ) {
                (Some(_), Some(_)) | (None, None) => {}
                _ => {
                    return Err(
                        "Gemma MTP assistant source requires both --mtp-assistant-gguf and --mtp-assistant-derived-lock"
                            .to_owned(),
                    );
                }
            }
            if mtp_draft_width != Some(1)
                && (mtp_assistant_gguf.is_some() || mtp_assistant_derived_lock.is_some())
            {
                return Err(
                    "Gemma MTP assistant source is valid only with --mtp-draft-width 1".to_owned(),
                );
            }
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
            match (
                mtp_assistant_gguf.as_ref(),
                mtp_assistant_derived_lock.as_ref(),
            ) {
                (Some(_), Some(_)) | (None, None) => {}
                _ => {
                    return Err(
                        "Gemma MTP assistant source requires both --mtp-assistant-gguf and --mtp-assistant-derived-lock"
                            .to_owned(),
                    );
                }
            }
            if mtp_draft_width != Some(1)
                && (mtp_assistant_gguf.is_some() || mtp_assistant_derived_lock.is_some())
            {
                return Err(
                    "Gemma MTP assistant source is valid only with --mtp-draft-width 1".to_owned(),
                );
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
                mtp_draft_width,
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
        mtp_assistant_gguf: mtp_assistant_gguf.map(PathBuf::from),
        mtp_assistant_derived_lock: mtp_assistant_derived_lock.map(PathBuf::from),
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

/// Admission contract for the reviewed Gemma 4 12B MTP companion.
///
/// The public draft-width flag is shared with Qwen, but Gemma's first
/// production path is intentionally narrower: omitted/zero means target-only
/// and one requests the paired BF16 assistant.  There is no implicit assistant
/// lookup or width clamping here; the caller must provide the canonical pair
/// bundle to the execution owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CliGemma4MtpPlan {
    selection: &'static str,
    enabled: bool,
    requested_width: Option<u8>,
    effective_width: Option<u8>,
}

fn resolve_cli_gemma4_mtp_plan(
    requested_width: Option<u8>,
    target: &str,
    kv_cache_encoding: Option<KvCacheEncoding>,
    sampling: SamplingParametersV1,
    model_fingerprint: &str,
) -> Result<CliGemma4MtpPlan, String> {
    match requested_width {
        None | Some(0) => Ok(CliGemma4MtpPlan {
            selection: "target-only",
            enabled: false,
            requested_width,
            effective_width: None,
        }),
        Some(width) => {
            if width != 1 {
                return Err(
                    "Gemma 4 MTP supports --mtp-draft-width 0 (target-only) or 1 only".to_owned(),
                );
            }
            let reason = if target != "gfx1201" {
                Some("forced Gemma MTP is reviewed only for exact gfx1201")
            } else if model_fingerprint != GEMMA4_12B_IT_FINGERPRINT {
                Some("forced Gemma MTP requires the reviewed Gemma 4 12B target")
            } else if kv_cache_encoding.is_some_and(|encoding| encoding != KvCacheEncoding::Fp16) {
                Some("forced Gemma MTP requires the FP16 KV cache encoding")
            } else if sampling.requires_logits() {
                Some("forced Gemma MTP requires greedy sampling without logits")
            } else {
                None
            };
            if let Some(reason) = reason {
                return Err(format!("forced Gemma MTP is incompatible: {reason}"));
            }
            Ok(CliGemma4MtpPlan {
                selection: "forced",
                enabled: true,
                requested_width,
                effective_width: Some(1),
            })
        }
    }
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
    fn gemma4_moe_prefill_selects_terminal_argmax_from_exact_row_count() {
        assert_eq!(
            gemma4_moe_prefill_terminal_argmax(&[11, 12, 13], 3).unwrap(),
            13
        );
        assert!(matches!(
            gemma4_moe_prefill_terminal_argmax(&[11], 3),
            Err(GenerationServiceError::Execution(message))
                if message.contains("expected 3, got 1")
        ));
        assert!(matches!(
            gemma4_moe_prefill_terminal_argmax(&[], 0),
            Err(GenerationServiceError::MissingDeviceArgmax)
        ));
    }

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
    fn gemma4_mtp_admission_is_target_only_by_default_and_greedy_width_one_only() {
        let greedy = SamplingParametersV1::new(0.0, 1.0, 0.0, 0.0).unwrap();
        let target_only = resolve_cli_gemma4_mtp_plan(
            None,
            "gfx1030",
            None,
            SamplingParametersV1::new(1.0, 1.0, 0.0, 0.0).unwrap(),
            "unreviewed",
        )
        .unwrap();
        assert_eq!(target_only.selection, "target-only");
        assert!(!target_only.enabled);
        assert_eq!(target_only.requested_width, None);
        assert_eq!(target_only.effective_width, None);

        let zero = resolve_cli_gemma4_mtp_plan(
            Some(0),
            "gfx1201",
            Some(KvCacheEncoding::Nvfp4),
            SamplingParametersV1::new(1.0, 1.0, 0.0, 0.0).unwrap(),
            "unreviewed",
        )
        .unwrap();
        assert_eq!(zero.selection, "target-only");
        assert!(!zero.enabled);

        let enabled = resolve_cli_gemma4_mtp_plan(
            Some(1),
            "gfx1201",
            None,
            greedy,
            GEMMA4_12B_IT_FINGERPRINT,
        )
        .unwrap();
        assert_eq!(enabled.selection, "forced");
        assert!(enabled.enabled);
        assert_eq!(enabled.effective_width, Some(1));

        for width in [2, 8] {
            let error = resolve_cli_gemma4_mtp_plan(
                Some(width),
                "gfx1201",
                None,
                greedy,
                GEMMA4_12B_IT_FINGERPRINT,
            )
            .unwrap_err();
            assert!(error.contains("width 0") && error.contains("or 1 only"));
        }
        let error = resolve_cli_gemma4_mtp_plan(
            Some(1),
            "gfx1030",
            None,
            greedy,
            GEMMA4_12B_IT_FINGERPRINT,
        )
        .unwrap_err();
        assert!(error.contains("gfx1201"));
        let error = resolve_cli_gemma4_mtp_plan(
            Some(1),
            "gfx1201",
            Some(KvCacheEncoding::Mxfp8E4),
            greedy,
            GEMMA4_12B_IT_FINGERPRINT,
        )
        .unwrap_err();
        assert!(error.contains("FP16 KV"));
        let error = resolve_cli_gemma4_mtp_plan(
            Some(1),
            "gfx1201",
            None,
            SamplingParametersV1::new(1.0, 0.9, 0.0, 0.0).unwrap(),
            GEMMA4_12B_IT_FINGERPRINT,
        )
        .unwrap_err();
        assert!(error.contains("greedy"));
    }

    #[test]
    fn gemma4_mtp_context_limit_is_explicit_and_non_default() {
        assert_eq!(GEMMA4_MTP_CONTEXT_TOKENS, 2_048);
        const {
            assert!(GEMMA4_MTP_CONTEXT_TOKENS < QWEN_RUNTIME_MAX_CONTEXT_TOKENS);
        }
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
    fn gemma_mtp_assistant_flags_are_an_explicit_width_one_pair() {
        let common = [
            "--gguf",
            "target.gguf",
            "--derived-lock",
            "target.lock.json",
            "--prompt",
            "abc",
            "--max-new-tokens",
            "3",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
            "--greedy",
        ];
        let mut missing_lock = common.to_vec();
        missing_lock.extend(["--mtp-assistant-gguf", "assistant.gguf"]);
        assert!(
            parse_args("generate", &missing_lock)
                .unwrap_err()
                .contains("requires both")
        );

        // Width one remains a valid shared flag for Qwen's existing MTP
        // route; Gemma enforces its assistant pair once its target identity
        // is known by the backend.
        let mut target_only_pairless = common.to_vec();
        target_only_pairless.extend(["--mtp-draft-width", "1"]);
        assert!(parse_args("generate", &target_only_pairless).is_ok());

        let mut missing_width = common.to_vec();
        missing_width.extend([
            "--mtp-assistant-gguf",
            "assistant.gguf",
            "--mtp-assistant-derived-lock",
            "assistant.lock.json",
        ]);
        assert!(
            parse_args("generate", &missing_width)
                .unwrap_err()
                .contains("only with --mtp-draft-width 1")
        );

        let mut pair = common.to_vec();
        pair.extend([
            "--mtp-draft-width",
            "1",
            "--mtp-assistant-gguf",
            "assistant.gguf",
            "--mtp-assistant-derived-lock",
            "assistant.lock.json",
        ]);
        let request = parse_args("generate", &pair).unwrap();
        assert_eq!(
            request.mtp_assistant_gguf,
            Some(PathBuf::from("assistant.gguf"))
        );
        assert_eq!(
            request.mtp_assistant_derived_lock,
            Some(PathBuf::from("assistant.lock.json"))
        );
        assert!(matches!(
            request.operation,
            Operation::Generate(GenerateRequest {
                mtp_draft_width: Some(1),
                ..
            })
        ));

        let mut benchmark_pair = [
            "--gguf",
            "target.gguf",
            "--derived-lock",
            "target.lock.json",
            "--lane",
            "direct",
            "--model-size",
            "12B",
            "--input-token-ids",
            "1,3,17",
            "--max-new-tokens",
            "3",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
            "--greedy",
            "--mtp-draft-width",
            "1",
            "--mtp-assistant-gguf",
            "assistant.gguf",
            "--mtp-assistant-derived-lock",
            "assistant.lock.json",
        ]
        .to_vec();
        let benchmark_request = parse_args("benchmark", &benchmark_pair).unwrap();
        assert!(matches!(
            benchmark_request.operation,
            Operation::Benchmark(BenchmarkRequest {
                lane: BenchmarkLane::Direct,
                model_size,
                mtp_draft_width: Some(1),
                input: BenchmarkInput::TokenIds(ref ids),
                ..
            }) if model_size == "12B" && ids.as_slice() == [1, 3, 17]
        ));
        benchmark_pair.extend(["--kv-cache-encoding", "mxfp8-e4"]);
        assert!(parse_args("benchmark", &benchmark_pair).is_err());
    }

    #[test]
    fn gemma_benchmark_executor_report_keeps_timed_token_contract() {
        let mut executor = SequenceExecution::new([7, 8, 9]);
        let mut timeline = BenchmarkTimeline::new(0);
        let outcome = run_generation_executor_timed(
            &mut executor,
            &qwen_stop_policy(),
            3,
            &[1, 3, 17],
            (&mut timeline, MonotonicClock::start()),
        )
        .unwrap();
        let sample = timeline
            .finish(BenchmarkSampleInput {
                input_token_ids: outcome.report.input_token_ids(),
                generated_token_ids: outcome.report.generated_token_ids(),
                visible_token_ids: outcome.report.visible_token_ids(),
                decode_input_token_ids: outcome.report.decode_input_token_ids(),
                stop: json!({
                    "version": outcome.report.stop_reason().map(|stop| stop.version()),
                    "reason_version": outcome.report.stop_reason().map(|stop| stop.reason_version()),
                    "kind": outcome.report.reason_token(),
                    "token_id": outcome.report.stop_token_id(),
                }),
                audit: json!({
                    "selected_backend": "hip",
                    "target": "gfx1201",
                    "device_index": 0,
                    "model_fingerprint": "sha256:test",
                    "plan_digest": "sha256:test",
                    "fallback_used": false,
                    "all_dispatches_hip": true,
                    "submission_count": 1,
                    "kernel_dispatch_count": 1,
                    "segment_count": 1,
                    "boundary_count": 1,
                }),
                memory: json!({}),
                cleanup: json!({"retryable_cleanup": 0, "durable_quarantine": 0}),
            })
            .unwrap();
        assert_eq!(sample["tokens"]["input_token_ids"], json!([1, 3, 17]));
        assert_eq!(sample["tokens"]["generated_token_ids"], json!([7, 8, 9]));
        assert_eq!(sample["tokens"]["visible_token_ids"], json!([7, 8, 9]));
        assert_eq!(sample["derived"]["decode_tokens"], 2);
        assert_eq!(outcome.decode_steps, 2);
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

    #[test]
    fn gemma_checkpoint_continuation_uses_only_new_suffix() {
        let prefix = [11_u32, 22, 33];
        let input = [11_u32, 22, 33, 44, 55];
        assert_eq!(
            gemma4_moe_checkpoint_suffix(&input, &prefix).unwrap(),
            &[44_u32, 55]
        );
    }

    #[test]
    fn gemma_checkpoint_continuation_rejects_non_prefix_or_empty_suffix() {
        assert_eq!(
            gemma4_moe_checkpoint_suffix(&[11_u32, 99], &[11, 22]),
            Err(ChatBackendErrorV1::CheckpointUnavailable)
        );
        let prefix = [11_u32, 22];
        assert_eq!(
            gemma4_moe_checkpoint_suffix(&prefix, &prefix),
            Err(ChatBackendErrorV1::CheckpointUnavailable)
        );
    }

    #[test]
    fn gemma_checkpoint_history_excludes_reverse_prompt_marker() {
        let reverse = vec!["<next>".to_owned(), "STOP".to_owned()];
        assert_eq!(
            gemma4_moe_checkpoint_visible_text("answer<next>tail", &reverse),
            "answer"
        );
        assert_eq!(
            gemma4_moe_checkpoint_visible_text("answer", &reverse),
            "answer"
        );
    }

    #[test]
    fn gemma_chat_accepts_default_or_explicit_static_fp8_only() {
        assert!(validate_gemma4_moe_chat_kv_cache_encoding(None).is_ok());
        assert!(
            validate_gemma4_moe_chat_kv_cache_encoding(Some(KvCacheEncoding::Fp8E4M3FnStatic))
                .is_ok()
        );
        for encoding in [
            KvCacheEncoding::Fp16,
            KvCacheEncoding::Fp8E4M3Fn,
            KvCacheEncoding::Nvfp4,
            KvCacheEncoding::Fp8E4M3Block16,
            KvCacheEncoding::Fp8E5M2Block16,
            KvCacheEncoding::Mxfp8E4,
            KvCacheEncoding::Mxfp8E5,
        ] {
            assert!(validate_gemma4_moe_chat_kv_cache_encoding(Some(encoding)).is_err());
        }
    }
}
