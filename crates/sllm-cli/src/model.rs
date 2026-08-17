use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sllm_core::{
    Backend, ExecutionSessionRequest, Gemma4ModelLock, Gemma4ResidentModel, ModelLock,
    OsSamplingRandom, QwenComponentSelection, QwenExecutionRequest, QwenMultimodalImageEmbedding,
    QwenMultimodalPrompt, QwenResidentModel, QwenVisionExecutionInput, QwenVisionResidentModel,
    ReviewedModelLock, SamplingParametersV1, VerifiedCache, VerifiedGgufGemmaSource,
    VerifiedGgufQwen35Moe, VerifiedGgufWeightSource, WeightClassification,
    assemble_gguf_qwen35_multimodal_prompt, assemble_qwen35_multimodal_prompt,
    build_gguf_qwen35_moe_weight_load_plan, build_qwen35_fp8_fnuz_graph, build_qwen35_fp8_graph,
    build_qwen35_gguf_fp8_graph, build_qwen35_gguf_moe_execution_graph, build_qwen35_graph,
    build_qwen35_mtp_graph, build_qwen35_multimodal_graph, build_qwen35_nvfp4_graph,
    build_verified_gguf_gemma_weight_load_plan, build_verified_gguf_qwen_weight_load_plan,
    build_verified_gguf_qwen35_vision_manifest, build_verified_qwen_component_weight_load_plan,
    build_verified_qwen35_vision_manifest, builtin_reviewed_model_lock,
    qwen35_moe_generation_stop_policy, read_derived_gguf_lock, verify_derived_gguf,
    verify_fp8_sidecar, verify_gguf_qwen35_moe, verify_nvfp4_sidecar,
};
use sllm_frontend::{
    BoundedImageBytesV1, DecodeModeV1, GenerationCancellationV1, GenerationConfigV1,
    GenerationExecutorV1, GenerationInputV1 as ServiceGenerationInputV1, GenerationReportV1,
    GenerationServiceError, GenerationServiceV1, GenerationStepV1, GenerationStopControllerV1,
    GenerationStopPolicyV1, ProcessedVisionInputV1, Qwen35ChatMessageV1, Qwen35ChatTemplateV1,
    Qwen35RenderOptionsV1, Qwen35VisionProcessorV1, QwenMtpGenerationExecutorV1,
    SpeculativeGenerationAdapterV1, ThinkingModeV1, TokenIdsV1, TokenizerFrontendV1,
    gemma4_generation_stop_policy,
};
use sllm_hip::HipBackend;

use crate::benchmark::{
    BenchmarkEvent, BenchmarkSampleInput, BenchmarkTimeline, BenchmarkTiming, MonotonicClock,
    RENDER_TOKENIZE_BENCHMARK_SCHEMA_VERSION, allocation_snapshot_value, compare_control_sample,
    control_comparison_contract, validate_fixed_input_token_ids, validate_model_ready_snapshot,
    validate_peak_vram_snapshot, validate_request_cleanup_snapshot,
    validate_resident_drop_snapshot, validate_sample_count, validate_snapshot_accounting,
};

const REPORT_SCHEMA: &str = "model-frontend-cli-report-v1";
const MAX_TEXT_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOKEN_IDS: usize = 1_048_576;
const MAX_NEW_TOKENS: u32 = 4096;
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

#[derive(Debug, PartialEq)]
struct GenerateRequest {
    input: GenerationInput,
    image_paths: Vec<PathBuf>,
    max_new_tokens: u32,
    sampling: SamplingParametersV1,
    seed: Option<u64>,
    stop_strings: Vec<String>,
    device_index: u32,
    target: String,
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
    RenderTokenize,
}

impl BenchmarkLane {
    fn schema_version(self) -> &'static str {
        RENDER_TOKENIZE_BENCHMARK_SCHEMA_VERSION
    }
}

#[derive(Debug, Eq, PartialEq)]
enum BenchmarkInput {
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
    device_index: u32,
    target: String,
    greedy: bool,
    warmups: u32,
    measured: u32,
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

#[derive(Debug, PartialEq)]
enum Operation {
    Verify,
    Tokenize {
        text: String,
    },
    Render {
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
    },
    Decode {
        ids: TokenIdsV1,
        mode: DecodeModeV1,
    },
    Generate(GenerateRequest),
    Benchmark(BenchmarkRequest),
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

trait ModelFrontendBackend {
    fn identity(&self) -> ModelIdentity;
    fn verify(&self) -> Result<Value, String>;
    fn tokenize(&self, text: &str) -> Result<Value, String>;
    fn render(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<Value, String>;
    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String>;
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
        let ids = tokenizer
            .encode(text)
            .map_err(|_| "text could not be tokenized".to_owned())?;
        Ok(json!({
            "kind": "tokenize",
            "prompt_mode": self.lock.model.tokenizer_contract.prompt_mode,
            "count": ids.len(),
            "token_ids": ids.as_slice(),
        }))
    }

    fn render(&self, _: &[Qwen35ChatMessageV1], _: Qwen35RenderOptionsV1) -> Result<Value, String> {
        Err("google/gemma-4-12B has no locked chat template; use a raw prompt".to_owned())
    }

    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let text = tokenizer
            .decode(ids, mode)
            .map_err(|_| "token IDs could not be decoded".to_owned())?;
        Ok(json!({"kind": "decode", "text": text}))
    }

    fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
        let started = Instant::now();
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
            .map_err(|_| "exact HIP execution session could not be opened".to_owned())?;

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
        let ids = tokenizer.encode(text).map_err(|error| error.to_string())?;
        Ok(json!({"kind":"tokenize","count":ids.len(),"token_ids":ids.as_slice()}))
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

    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let text = tokenizer
            .decode(ids, mode)
            .map_err(|error| error.to_string())?;
        Ok(json!({"kind":"decode","text":text}))
    }

    fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
        let started = Instant::now();
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
    ) -> Result<sllm_core::QwenGraph, sllm_core::QwenGraphError> {
        match &self.source {
            QwenDenseSource::Gguf(source) if source.has_fp8_recipe() => {
                if target != "gfx1201" {
                    return Err(sllm_core::QwenGraphError::InvalidModel(
                        "the embedded E4M3FN GGUF recipe currently requires the native gfx1201 provider"
                            .to_owned(),
                    ));
                }
                build_qwen35_gguf_fp8_graph(
                    &self.lock,
                    plan,
                    source,
                    token_count,
                    state_capacity,
                    sllm_core::DType::F8E4M3Fn,
                    sllm_core::KvCacheEncoding::Fp16,
                )
            }
            _ => build_qwen35_graph(&self.lock, plan, token_count, state_capacity),
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
        let ids = tokenizer
            .encode(text)
            .map_err(|_| "text could not be tokenized".to_owned())?;
        Ok(json!({"kind": "tokenize", "count": ids.len(), "token_ids": ids.as_slice()}))
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

    fn decode(&self, ids: &TokenIdsV1, mode: DecodeModeV1) -> Result<Value, String> {
        let tokenizer = self.tokenizer()?;
        let text = tokenizer
            .decode(ids, mode)
            .map_err(|_| "token IDs could not be decoded".to_owned())?;
        Ok(json!({"kind": "decode", "text": text}))
    }

    fn generate(&self, request: &GenerateRequest) -> Result<Value, String> {
        let started = Instant::now();
        let (effective_input, processed_images) =
            prepare_qwen_cli_images(&request.input, &request.image_paths)?;
        if matches!(self.source, QwenDenseSource::Gguf(_)) {
            if request.fp8_manifest.is_some()
                || request.fp8_artifact.is_some()
                || request.fp8_provider.is_some()
            {
                return Err(
                    "GGUF carries its own quantization recipe and cannot be combined with legacy sidecar flags"
                        .to_owned(),
                );
            }
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
        let state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or_else(|| "generation state capacity overflowed".to_owned())?;
        let plan = self.load_plan(QwenComponentSelection::TEXT_ONLY)?;
        let embedded_fp8 = matches!(
            &self.source,
            QwenDenseSource::Gguf(source) if source.has_fp8_recipe()
        );
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
        if !processed_images.is_empty() && has_sidecar {
            return Err("vision requests currently require the BF16 text artifact".to_owned());
        }
        let fp8_provider =
            select_cli_fp8_provider(has_sidecar, request.fp8_provider, &request.target)?;
        let mtp_candidate = processed_images.is_empty()
            && !has_sidecar
            && !embedded_fp8
            && request.target == "gfx1201"
            && !request.sampling.requires_logits()
            && self.lock.fingerprint() == sllm_core::QWEN35_4B_FINGERPRINT;
        let text_graph_rows = if mtp_candidate {
            input_len.max(2)
        } else {
            input_len
        };
        let graph = if !processed_images.is_empty() {
            build_qwen35_multimodal_graph(&self.lock, &plan, input_len, state_capacity)
        } else if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
            build_qwen35_nvfp4_graph(&self.lock, &plan, nvfp4_sidecar, input_len, state_capacity)
        } else {
            match (&sidecar, fp8_provider) {
                (Some(_), Some(CliFp8Provider::ConvertedBf16)) => {
                    build_qwen35_graph(&self.lock, &plan, input_len, state_capacity)
                }
                (Some(sidecar), Some(CliFp8Provider::NativeFnuz)) => build_qwen35_fp8_fnuz_graph(
                    &self.lock,
                    &plan,
                    sidecar,
                    input_len,
                    state_capacity,
                ),
                (Some(sidecar), Some(_)) => {
                    build_qwen35_fp8_graph(&self.lock, &plan, sidecar, input_len, state_capacity)
                }
                (None, None) => {
                    self.build_plain_graph(&plan, text_graph_rows, state_capacity, &request.target)
                }
                _ => unreachable!("quantized provider selection validated sidecar state"),
            }
        }
        .map_err(|error| {
            format!("generation graph does not satisfy the fixed Qwen contract: {error}")
        })?;
        let plan_digest = plan.digest_hex();
        let model_fingerprint = self.lock.fingerprint().to_owned();

        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session_request =
            ExecutionSessionRequest::new(request.device_index, request.target.clone())
                .map_err(|_| "invalid exact HIP session request".to_owned())?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|_| "exact HIP execution session could not be opened".to_owned())?;

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
                let request_graph = build_qwen35_nvfp4_graph(
                    &self.lock,
                    &plan,
                    &sidecar,
                    input_len,
                    state_capacity,
                )
                .map_err(|error| format!("Qwen NVFP4 request graph failed: {error}"))?;
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
                let request_graph = match fp8_provider {
                    Some(CliFp8Provider::ConvertedBf16) => {
                        build_qwen35_graph(&self.lock, &plan, input_len, state_capacity)
                    }
                    Some(CliFp8Provider::NativeFnuz) => build_qwen35_fp8_fnuz_graph(
                        &self.lock,
                        &plan,
                        &sidecar,
                        input_len,
                        state_capacity,
                    ),
                    Some(CliFp8Provider::Native) => build_qwen35_fp8_graph(
                        &self.lock,
                        &plan,
                        &sidecar,
                        input_len,
                        state_capacity,
                    ),
                    Some(CliFp8Provider::Nvfp4PackedDequant) | None => {
                        unreachable!("FP8 sidecar requires an FP8 provider")
                    }
                }
                .map_err(|error| format!("Qwen FP8 request graph failed: {error}"))?;
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
                        let request_graph = self
                            .build_plain_graph(&plan, input_len, state_capacity, &request.target)
                            .map_err(|error| format!("Qwen GGUF request graph failed: {error}"))?;
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
            let (report, audit) = if let Some((_, prompt)) = vision_bundle.as_ref() {
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
                (report, audit)
            } else if mtp_candidate {
                let mtp_plan = self.load_plan(QwenComponentSelection::MTP_ONLY)?;
                let mtp_graph = build_qwen35_mtp_graph(&self.lock, &mtp_plan, state_capacity)
                    .map_err(|error| format!("MTP request graph failed: {error}"))?;
                let mtp_resident = match &self.source {
                    QwenDenseSource::Cache(cache) => QwenResidentModel::new(
                        Arc::clone(&session),
                        mtp_graph.clone(),
                        mtp_plan,
                        Arc::clone(cache),
                        COMPLETION_TIMEOUT,
                    ),
                    QwenDenseSource::Gguf(source) => QwenResidentModel::new_gguf(
                        Arc::clone(&session),
                        mtp_graph.clone(),
                        mtp_plan,
                        Arc::clone(source),
                        COMPLETION_TIMEOUT,
                    ),
                }
                .map_err(|error| format!("MTP resident provisioning failed: {error}"))?;
                let mtp_owner = mtp_resident
                    .new_request(mtp_graph)
                    .map_err(|error| format!("MTP request provisioning failed: {error}"))?;
                let mut executor = SpeculativeGenerationAdapterV1::new(
                    QwenMtpGenerationExecutorV1::new_with_draft_width(owner, mtp_owner, 1)
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
                drop(executor);
                drop(mtp_resident);
                (report, audit)
            } else {
                let report = service.generate_tokens(
                    &mut owner,
                    &input,
                    &config,
                    &cancellation,
                    &mut random,
                );
                let audit = owner.audit_snapshot();
                (report, audit)
            };
            let report = report.map_err(|error| format!("generation service failed: {error}"))?;
            let audit = audit.map_err(|_| "Qwen dispatch audit was empty or invalid".to_owned())?;
            if audit.target() != request.target {
                return Err(
                    "Qwen dispatch audit target differs from the requested target".to_owned(),
                );
            }
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
                    "selected_backend": audit.selected_backend(),
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
                    "all_dispatches_hip": audit.all_dispatches_hip(),
                    "weight_encoding": if embedded_fp8 { "ocp-e4m3fn-outer-f32" } else { match fp8_provider { Some(CliFp8Provider::ConvertedBf16) => "bf16-converted-from-ocp-e4m3fn", Some(CliFp8Provider::NativeFnuz) => "e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32", Some(CliFp8Provider::Nvfp4PackedDequant) => "nvfp4-e2m1-block16-e4m3fn-tensor-f32", Some(_) => "ocp-e4m3fn-outer-f32", None => "bf16" } },
                    "fp8_provider": if embedded_fp8 { Some("gguf-native") } else { fp8_provider.map(CliFp8Provider::label) },
                    "image_count": processed_images.len(),
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
        validate_sample_count(request.warmups, request.measured)?;
        if request.warmups != 3 || request.measured != 10 {
            return Err(
                "benchmark protocol requires exactly 3 warmups and 10 measured requests".to_owned(),
            );
        }
        if !request.greedy {
            return Err("benchmark requires explicit --greedy mode".to_owned());
        }
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

        let tokenizer = self.tokenizer()?;
        let renderer = self.renderer()?;
        let BenchmarkInput::Messages { messages, options } = &request.input;
        let rendered = renderer
            .render(messages, *options)
            .map_err(|_| "chat messages could not be rendered".to_owned())?;
        let seed_input = tokenizer
            .encode(&rendered)
            .map_err(|_| "benchmark input could not be tokenized")?;
        if seed_input.is_empty() {
            return Err("benchmark input token IDs must not be empty".to_owned());
        }
        let input_len = u64::try_from(seed_input.len())
            .map_err(|_| "benchmark input token count overflowed".to_owned())?;
        let state_capacity = input_len
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or_else(|| "benchmark state capacity overflowed".to_owned())?;
        let model_load_start_ns = timing.model_load_start_ns();
        let plan = self.load_plan(QwenComponentSelection::TEXT_ONLY)?;
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
        let fp8_provider =
            select_cli_fp8_provider(has_sidecar, request.fp8_provider, &request.target)?;
        let first_graph = if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
            build_qwen35_nvfp4_graph(&self.lock, &plan, nvfp4_sidecar, input_len, state_capacity)
        } else {
            match (&sidecar, fp8_provider) {
                (Some(_), Some(CliFp8Provider::ConvertedBf16)) => {
                    build_qwen35_graph(&self.lock, &plan, input_len, state_capacity)
                }
                (Some(sidecar), Some(CliFp8Provider::NativeFnuz)) => build_qwen35_fp8_fnuz_graph(
                    &self.lock,
                    &plan,
                    sidecar,
                    input_len,
                    state_capacity,
                ),
                (Some(sidecar), Some(_)) => {
                    build_qwen35_fp8_graph(&self.lock, &plan, sidecar, input_len, state_capacity)
                }
                (None, None) => {
                    self.build_plain_graph(&plan, input_len, state_capacity, &request.target)
                }
                _ => unreachable!("quantized provider selection validated sidecar state"),
            }
        }
        .map_err(|error| {
            format!("benchmark graph does not satisfy the fixed Qwen contract: {error}")
        })?;
        let backend = HipBackend::connect().map_err(|_| "HIP backend is unavailable".to_owned())?;
        let session_request =
            ExecutionSessionRequest::new(request.device_index, request.target.clone())
                .map_err(|_| "invalid exact HIP session request".to_owned())?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|_| "exact HIP execution session could not be opened".to_owned())?;

        let execution = (|| -> Result<Value, String> {
            let resident = if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
                QwenResidentModel::new_nvfp4(
                    Arc::clone(&session),
                    first_graph,
                    plan.clone(),
                    Arc::clone(self.cache()?),
                    Arc::clone(nvfp4_sidecar),
                    COMPLETION_TIMEOUT,
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
                            COMPLETION_TIMEOUT,
                        )
                    }
                    (Some(sidecar), Some(CliFp8Provider::NativeFnuz)) => {
                        QwenResidentModel::new_fp8_fnuz(
                            Arc::clone(&session),
                            first_graph,
                            plan.clone(),
                            Arc::clone(self.cache()?),
                            Arc::clone(sidecar),
                            COMPLETION_TIMEOUT,
                        )
                    }
                    (Some(sidecar), Some(_)) => QwenResidentModel::new_fp8(
                        Arc::clone(&session),
                        first_graph,
                        plan.clone(),
                        Arc::clone(self.cache()?),
                        Arc::clone(sidecar),
                        COMPLETION_TIMEOUT,
                    ),
                    (None, None) => match &self.source {
                        QwenDenseSource::Cache(cache) => QwenResidentModel::new(
                            Arc::clone(&session),
                            first_graph,
                            plan.clone(),
                            Arc::clone(cache),
                            COMPLETION_TIMEOUT,
                        ),
                        QwenDenseSource::Gguf(source) => QwenResidentModel::new_gguf(
                            Arc::clone(&session),
                            first_graph,
                            plan.clone(),
                            Arc::clone(source),
                            COMPLETION_TIMEOUT,
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

            let control_graph = if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
                build_qwen35_nvfp4_graph(
                    &self.lock,
                    &plan,
                    nvfp4_sidecar,
                    input_len,
                    state_capacity,
                )
            } else {
                match (&sidecar, fp8_provider) {
                    (Some(_), Some(CliFp8Provider::ConvertedBf16)) => {
                        build_qwen35_graph(&self.lock, &plan, input_len, state_capacity)
                    }
                    (Some(sidecar), Some(CliFp8Provider::NativeFnuz)) => {
                        build_qwen35_fp8_fnuz_graph(
                            &self.lock,
                            &plan,
                            sidecar,
                            input_len,
                            state_capacity,
                        )
                    }
                    (Some(sidecar), Some(_)) => build_qwen35_fp8_graph(
                        &self.lock,
                        &plan,
                        sidecar,
                        input_len,
                        state_capacity,
                    ),
                    (None, None) => self.build_plain_graph(
                        &plan,
                        input_len,
                        state_capacity,
                        &request.target,
                    ),
                    _ => unreachable!("quantized provider selection validated sidecar state"),
                }
            }
            .map_err(|error| {
                format!("benchmark correctness-control graph does not satisfy the Qwen contract: {error}")
            })?;
            let mut control_owner = match resident.new_request(control_graph) {
                Ok(owner) => owner,
                Err(error) => {
                    let cleanup_memory = allocation_snapshot_value(session.memory_snapshot());
                    validate_request_cleanup_snapshot(&cleanup_memory, ready_model_current_bytes)?;
                    return Err(format!(
                        "Qwen benchmark correctness-control provisioning failed: {error}"
                    ));
                }
            };
            let control_request_memory = allocation_snapshot_value(session.memory_snapshot());
            let control_outcome = match validate_snapshot_accounting(
                &control_request_memory,
                "correctness-control request",
            ) {
                Ok(()) => run_greedy_generation(
                    &mut control_owner,
                    self.lock.generation_stop_policy(),
                    request.max_new_tokens,
                    seed_input.as_slice(),
                ),
                Err(error) => Err(error),
            };
            let control_audit = if control_outcome.is_ok() {
                Some(control_owner.audit_snapshot().map_err(|_| {
                    "Qwen correctness-control dispatch audit was empty or invalid".to_owned()
                }))
            } else {
                None
            };
            drop(control_owner);
            let control_cleanup_memory = allocation_snapshot_value(session.memory_snapshot());
            validate_request_cleanup_snapshot(&control_cleanup_memory, ready_model_current_bytes)?;
            let control_outcome = control_outcome?;
            let control_audit = control_audit.ok_or_else(|| {
                "correctness-control dispatch audit was not collected".to_owned()
            })??;
            if control_audit.target() != request.target {
                return Err(
                    "Qwen correctness-control dispatch audit target differs from requested target"
                        .to_owned(),
                );
            }
            let control_report = control_outcome.report;
            let control_stop = control_report.stop_reason().ok_or_else(|| {
                "correctness-control generation ended without a stop reason".to_owned()
            })?;
            let control_stop_value = json!({
                "version": control_stop.version(),
                "reason_version": control_stop.reason_version(),
                "kind": control_stop.reason_token(),
                "token_id": control_stop.token_id(),
            });
            let control_audit_value = json!({
                "selected_backend": control_audit.selected_backend(),
                "target": control_audit.target(),
                "device_index": request.device_index,
                "model_fingerprint": self.lock.fingerprint(),
                "plan_digest": plan.digest_hex(),
                "fallback_used": control_audit.fallback_used(),
                "submission_count": control_audit.submission_count(),
                "kernel_dispatch_count": control_audit.kernel_dispatch_count(),
                "segment_count": control_audit.segment_count(),
                "boundary_count": control_audit.boundary_count(),
                "all_dispatches_hip": control_audit.all_dispatches_hip(),
            });
            let correctness_control = json!({
                "label": "correctness-only",
                "execution_path": "normal-untimed",
                "timing_instrumentation": "off",
                "included_in_performance_statistics": false,
                "tokens": {
                    "input_token_ids": control_report.input_token_ids(),
                    "generated_token_ids": control_report.generated_token_ids(),
                    "visible_token_ids": control_report.visible_token_ids(),
                    "decode_input_token_ids": control_report.decode_input_token_ids(),
                },
                "stop": control_stop_value,
                "audit": control_audit_value,
                "memory": {
                    "request_start": control_request_memory,
                    "after_cleanup": control_cleanup_memory,
                },
                "cleanup": {
                    "request_dropped": true,
                    "allocator_cleanup_validated": true,
                },
                "comparison": control_comparison_contract(),
            });

            let run_sample = |sample_index: u32| -> Result<Value, String> {
                let request_start_ns = timing.now_ns();
                let BenchmarkInput::Messages { messages, options } = &request.input;
                let rendered = renderer
                    .render(messages, *options)
                    .map_err(|_| "chat messages could not be rendered".to_owned())?;
                let input = tokenizer
                    .encode(&rendered)
                    .map_err(|_| "benchmark input could not be tokenized".to_owned())?;
                validate_fixed_input_token_ids(seed_input.as_slice(), input.as_slice())?;
                let graph = if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
                    build_qwen35_nvfp4_graph(
                        &self.lock,
                        &plan,
                        nvfp4_sidecar,
                        input_len,
                        state_capacity,
                    )
                } else {
                    match (&sidecar, fp8_provider) {
                        (Some(_), Some(CliFp8Provider::ConvertedBf16)) => {
                            build_qwen35_graph(&self.lock, &plan, input_len, state_capacity)
                        }
                        (Some(sidecar), Some(CliFp8Provider::NativeFnuz)) => {
                            build_qwen35_fp8_fnuz_graph(
                                &self.lock,
                                &plan,
                                sidecar,
                                input_len,
                                state_capacity,
                            )
                        }
                        (Some(sidecar), Some(_)) => build_qwen35_fp8_graph(
                            &self.lock,
                            &plan,
                            sidecar,
                            input_len,
                            state_capacity,
                        ),
                        (None, None) => {
                            build_qwen35_graph(&self.lock, &plan, input_len, state_capacity)
                        }
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
                        self.lock.generation_stop_policy(),
                        request.max_new_tokens,
                        input.as_slice(),
                        Some((&mut timeline, timing.request_clock())),
                    ),
                    Err(error) => Err(error),
                };
                let audit = if outcome.is_ok() {
                    Some(owner.audit_snapshot().map_err(|_| {
                        "Qwen benchmark dispatch audit was empty or invalid".to_owned()
                    }))
                } else {
                    None
                };
                drop(owner);
                let cleanup_timestamp_ns = timing.now_ns();
                let cleanup_memory = allocation_snapshot_value(session.memory_snapshot());
                validate_request_cleanup_snapshot(&cleanup_memory, ready_model_current_bytes)?;
                let outcome = outcome?;
                let audit =
                    audit.ok_or_else(|| "timed dispatch audit was not collected".to_owned())??;
                if audit.target() != request.target {
                    return Err(
                        "Qwen benchmark dispatch audit target differs from requested target"
                            .to_owned(),
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
                    }),
                    cleanup: json!({
                        "sample_index": sample_index,
                        "request_dropped": true,
                        "allocator_cleanup_validated": true,
                        "retryable_cleanup": 0,
                        "durable_quarantine": 0,
                    }),
                })?;
                compare_control_sample(&correctness_control, &sample)?;
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
            let lane_definition =
                "CLI end-to-end: request start includes chat render and tokenizer encode";
            Ok(json!({
                "benchmark_schema_version": request.lane.schema_version(),
                "state": "PASS",
                "lane": "render-tokenize",
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
                "config": {
                    "input_token_ids": seed_input.as_slice(),
                    "input_token_count": seed_input.len(),
                    "max_new_tokens": request.max_new_tokens,
                    "greedy": request.greedy,
                    "warmups": request.warmups,
                    "measured": request.measured,
                    "tokenizer": true,
                    "render": true,
                    "stop_policy": {
                        "stop_token_ids": [248046, 248044],
                        "visible_stop_tokens": false,
                    },
                },
                "memory": {
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
                    "submission_count": submission_count,
                    "kernel_dispatch_count": kernel_dispatch_count,
                    "segment_count": segment_count,
                    "boundary_count": boundary_count,
                    "fallback_used": false,
                    "all_dispatches_hip": true,
                    "model_load_count": 1,
                    "weight_encoding": match fp8_provider { Some(CliFp8Provider::ConvertedBf16) => "bf16-converted-from-ocp-e4m3fn", Some(CliFp8Provider::NativeFnuz) => "e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32", Some(CliFp8Provider::Nvfp4PackedDequant) => "nvfp4-e2m1-block16-e4m3fn-tensor-f32", Some(_) => "ocp-e4m3fn-outer-f32", None => "bf16" },
                    "fp8_provider": fp8_provider.map(CliFp8Provider::label),
                    "request_model_load_count": 0,
                    "model_reused": true,
                    "sample_count": request.warmups + request.measured,
                    "correctness_control_request_count": 1,
                    "total_request_count": request.warmups + request.measured + 1,
                },
                "cleanup": {
                    "correctness_control_request_count": 1,
                    "warmup_request_count": request.warmups,
                    "measured_request_count": request.measured,
                    "request_cleanup_count": request.warmups + request.measured + 1,
                    "performance_sample_count": request.warmups + request.measured,
                    "all_requests_dropped": true,
                    "correctness_control_dropped": true,
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
            .map_err(|_| "HIP session cleanup failed".to_owned())?;
        let mut result = execution?;
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
    let request = parse(command, arguments)?;
    let benchmark_timing = (command == "benchmark").then(BenchmarkTiming::start);
    let backend = open_production_backend(&request)?;
    match benchmark_timing {
        Some(timing) => {
            execute_with_timing(command, request.operation, backend.as_ref(), Some(timing))
        }
        None => execute(command, request.operation, backend.as_ref()),
    }
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
        Operation::Tokenize { text } => backend.tokenize(&text)?,
        Operation::Render { messages, options } => backend.render(&messages, options)?,
        Operation::Decode { ids, mode } => backend.decode(&ids, mode)?,
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
    let generation = command == "generate";
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
            "gpu_execution": generation,
            "model_execution": generation,
            "generation": generation,
        },
        "result": result,
    }))
    .map_err(|_| "model frontend report could not be serialized".to_owned())
}

fn parse(command: &str, arguments: impl Iterator<Item = String>) -> Result<Request, String> {
    let mut gguf = None;
    let mut derived_lock = None;
    let mut text = None;
    let mut token_ids = None;
    let mut messages = Vec::new();
    let mut thinking = None;
    let mut no_generation_prompt = false;
    let mut skip_special_tokens = false;
    let mut prompt = None;
    let mut max_new_tokens = None;
    let mut device_index = None;
    let mut target = None;
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
    let fp8_manifest: Option<PathBuf> = None;
    let fp8_artifact: Option<PathBuf> = None;
    let fp8_provider: Option<CliFp8Provider> = None;
    let mut image_paths = Vec::new();
    let mut message_bytes = 0_usize;
    let mut arguments = arguments.peekable();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--gguf" => set_once(&mut gguf, take_value(&mut arguments, "--gguf")?, "--gguf")?,
            "--derived-lock" => set_once(
                &mut derived_lock,
                take_value(&mut arguments, "--derived-lock")?,
                "--derived-lock",
            )?,
            "--text" if command == "tokenize" => {
                let value = take_value(&mut arguments, "--text")?;
                if value.len() > MAX_TEXT_BYTES {
                    return Err("--text exceeds the 16 MiB input limit".to_owned());
                }
                set_once(&mut text, value, "--text")?;
            }
            "--tokens" if command == "decode" => {
                let value = take_value(&mut arguments, "--tokens")?;
                set_once(&mut token_ids, parse_token_ids(&value)?, "--tokens")?;
            }
            "--skip-special-tokens" if command == "decode" => {
                if skip_special_tokens {
                    return Err("duplicate --skip-special-tokens".to_owned());
                }
                skip_special_tokens = true;
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
                if parsed == 0 || parsed > MAX_NEW_TOKENS {
                    return Err(format!("--max-new-tokens must be in [1,{MAX_NEW_TOKENS}]"));
                }
                set_once(&mut max_new_tokens, parsed, "--max-new-tokens")?;
            }
            "--device-index" if command == "generate" || command == "benchmark" => {
                let value = take_value(&mut arguments, "--device-index")?;
                if value.is_empty() || value.bytes().any(|byte| !byte.is_ascii_digit()) {
                    return Err("--device-index must be an unsigned decimal U32".to_owned());
                }
                let parsed = value
                    .parse::<u32>()
                    .map_err(|_| "--device-index must be an unsigned decimal U32".to_owned())?;
                set_once(&mut device_index, parsed, "--device-index")?;
            }
            "--target" if command == "generate" || command == "benchmark" => {
                let value = take_value(&mut arguments, "--target")?;
                if value != "gfx1030" && value != "gfx1201" && value != "gfx942" {
                    return Err("--target must be gfx1030, gfx1201, or gfx942".to_owned());
                }
                set_once(&mut target, value, "--target")?;
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
                    "render-tokenize" => BenchmarkLane::RenderTokenize,
                    _ => return Err("--lane must be render-tokenize".to_owned()),
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
                if command == "render" || command == "generate" || command == "benchmark" =>
            {
                let value = take_value(&mut arguments, "--message")?;
                message_bytes = message_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| "--message input size overflow".to_owned())?;
                if message_bytes > MAX_TEXT_BYTES || messages.len() == 4096 {
                    return Err("render message input exceeds the bounded CLI limit".to_owned());
                }
                messages.push(parse_message(&value)?);
            }
            "--thinking"
                if command == "render" || command == "generate" || command == "benchmark" =>
            {
                let value = match take_value(&mut arguments, "--thinking")?.as_str() {
                    "default" => ThinkingModeV1::TemplateDefault,
                    "enabled" => ThinkingModeV1::Enabled,
                    "disabled" => ThinkingModeV1::Disabled,
                    _ => return Err("--thinking must be default, enabled, or disabled".to_owned()),
                };
                set_once(&mut thinking, value, "--thinking")?;
            }
            "--no-generation-prompt" if command == "render" => {
                if no_generation_prompt {
                    return Err("duplicate --no-generation-prompt".to_owned());
                }
                no_generation_prompt = true;
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
    let operation = match command {
        "verify-model" => Operation::Verify,
        "tokenize" => Operation::Tokenize {
            text: text.ok_or_else(|| "missing required --text TEXT".to_owned())?,
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
        "decode" => Operation::Decode {
            ids: token_ids.ok_or_else(|| "missing required --tokens IDS".to_owned())?,
            mode: if skip_special_tokens {
                DecodeModeV1::SkipSpecialTokens
            } else {
                DecodeModeV1::PreserveSpecialTokens
            },
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
            validate_sample_count(warmups, measured)?;
            if warmups != 3 || measured != 10 {
                return Err(
                    "benchmark protocol requires exactly 3 warmups and 10 measured requests"
                        .to_owned(),
                );
            }
            let model_size =
                benchmark_model_size.ok_or_else(|| "benchmark requires --model-size".to_owned())?;
            let row_id = benchmark_row_id.unwrap_or_else(|| "cli-render-tokenize".to_owned());
            let case_id = benchmark_case_id.unwrap_or_else(|| "render-tokenize".to_owned());
            if messages.is_empty() {
                return Err(
                    "benchmark render-tokenize lane requires --message ROLE:CONTENT".to_owned(),
                );
            }
            let input = BenchmarkInput::Messages {
                messages,
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: true,
                    thinking: thinking.unwrap_or(ThinkingModeV1::TemplateDefault),
                },
            };
            Operation::Benchmark(BenchmarkRequest {
                lane,
                row_id,
                model_size,
                case_id,
                input,
                max_new_tokens: max_new_tokens
                    .ok_or_else(|| "benchmark requires --max-new-tokens".to_owned())?,
                device_index: device_index
                    .ok_or_else(|| "benchmark requires --device-index".to_owned())?,
                target: target.ok_or_else(|| "benchmark requires --target".to_owned())?,
                greedy,
                warmups,
                measured,
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
            assert_eq!(request.lane, BenchmarkLane::RenderTokenize);
            assert!(matches!(request.input, BenchmarkInput::Messages { .. }));
            assert_eq!(request.row_id, "host-test");
            assert_eq!(request.model_size, "4B");
            assert_eq!(request.case_id, "host-test");
            assert_eq!(request.max_new_tokens, 3);
            assert_eq!(request.warmups, 3);
            assert_eq!(request.measured, 10);
            Ok(json!({
                "benchmark_schema_version": RENDER_TOKENIZE_BENCHMARK_SCHEMA_VERSION,
                "state": "PASS",
                "lane": "render-tokenize",
                "lane_definition": "pretokenized direct engine: request start excludes render/tokenize",
                "row": {"row_id": request.row_id, "model_size": request.model_size, "case_id": request.case_id, "input_token_ids": [1, 3, 17], "input_token_count": 3, "requested_output_tokens": 3},
                "identities": {"target": request.target, "device_index": request.device_index},
                "model_load": {"event": "model_load", "start_ns": timing.model_load_start_ns(), "model_ready_ns": 1, "duration_ns": 1, "load_count": 1},
                "config": {"warmups": request.warmups, "measured": request.measured, "tokenizer": false, "render": false},
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
        ];
        for (command, args) in cases {
            let request = parse_args(command, &args).unwrap();
            let output = execute(command, request.operation, &TinyBackend).unwrap();
            let document: Value = serde_json::from_str(&output).unwrap();
            assert_eq!(document["command"], command);
            assert_eq!(document["result"]["kind"], command);
            assert_eq!(document["state"], "PASS");
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
