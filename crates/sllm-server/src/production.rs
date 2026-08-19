//! Production model backends for the profile-v1 transport.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use sllm_core::{
    AllocationSnapshot, Backend, ExecutionSession, ExecutionSessionRequest,
    GEMMA4_RECOMMENDED_CONTEXT_TOKENS, Gemma4ModelLock, Gemma4ResidentModel, KvCacheEncoding,
    ModelLock, OsSamplingRandom, QWEN35_RECOMMENDED_CONTEXT_TOKENS, QwenComponentSelection,
    QwenExecutionRequest, QwenMultimodalImageEmbedding, QwenMultimodalPrompt, QwenResidentModel,
    QwenVisionExecutionInput, QwenVisionManifest, QwenVisionResidentModel, ReviewedModelLock,
    VerifiedCache, VerifiedFp8Sidecar, VerifiedGgufQwen35Moe, VerifiedGgufWeightSource,
    VerifiedNvfp4Sidecar, VerifiedQwen35Moe, WeightLoadPlan,
    assemble_gguf_qwen35_multimodal_prompt, assemble_qwen35_multimodal_prompt,
    build_gguf_qwen35_moe_weight_load_plan, build_qwen35_fp8_fnuz_graph, build_qwen35_fp8_graph,
    build_qwen35_gguf_fp8_graph, build_qwen35_gguf_moe_execution_graph,
    build_qwen35_graph_with_kv_cache_encoding, build_qwen35_moe_execution_graph,
    build_qwen35_mtp_graph, build_qwen35_multimodal_graph, build_qwen35_nvfp4_graph,
    build_verified_gguf_gemma_weight_load_plan, build_verified_gguf_qwen_weight_load_plan,
    build_verified_gguf_qwen35_vision_manifest, builtin_reviewed_model_lock,
    qwen_graph_memory_estimate, qwen_prefill_chunk_candidates, qwen35_moe_generation_stop_policy,
    read_derived_gguf_lock, verify_derived_gguf, verify_gguf_qwen35_moe,
};
use sllm_frontend::{
    GenerationCancellationV1, GenerationExecutorV1, GenerationInputV1, GenerationOutputSinkV1,
    GenerationServiceError, GenerationServiceV1, GenerationStepV1, GenerationStopPolicyV1,
    Qwen35ChatMessageV1, Qwen35ChatTemplateV1, Qwen35RenderOptionsV1, QwenMtpGenerationExecutorV1,
    SpeculativeGenerationAdapterV1, TokenizerFrontendV1, gemma4_generation_stop_policy,
};
use sllm_hip::HipBackend;

use crate::api::ChatContentPartV1;
use crate::{
    BackendCompletionV1, BackendErrorV1, ChatCompletionRequestV1, ChatGenerationBackendV1,
    FinishReasonV1, GenerationDeltaSinkV1, TokenUsageV1,
};

const MAX_RETAINED_REQUEST_AUDITS: usize = 64;
const GEMMA4_RAW_CHAT_MAX_BYTES: usize = 16 * 1024 * 1024;
const GEMMA4_STATIC_FP8_KV_BYTES_PER_TOKEN: u64 = 172_032;

#[derive(Clone, Debug)]
pub struct QwenBackendConfigV1 {
    pub gguf_path: PathBuf,
    pub derived_lock_path: PathBuf,
    pub device_index: u32,
    pub target: String,
    pub completion_timeout: Duration,
    pub shutdown_timeout: Duration,
    pub context_length: u32,
    pub kv_cache_encoding: KvCacheEncoding,
}

impl QwenBackendConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        if self.target.is_empty()
            || !self.target.is_ascii()
            || self.completion_timeout.is_zero()
            || self.shutdown_timeout.is_zero()
            || self.context_length == 0
        {
            return Err(BackendErrorV1::new(
                "Qwen backend target, context length, and timeouts must be valid and nonzero",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Gemma4BackendConfigV1 {
    pub gguf_path: PathBuf,
    pub derived_lock_path: PathBuf,
    pub device_index: u32,
    pub target: String,
    pub completion_timeout: Duration,
    pub shutdown_timeout: Duration,
    pub context_length: u32,
}

impl Gemma4BackendConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        if self.target.is_empty()
            || !self.target.is_ascii()
            || self.completion_timeout.is_zero()
            || self.shutdown_timeout.is_zero()
            || self.context_length == 0
        {
            return Err(BackendErrorV1::new(
                "Gemma backend target, context length, and timeouts must be valid and nonzero",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ProductionRequestAuditV1 {
    pub outcome: String,
    pub target: String,
    pub weight_encoding: String,
    pub kv_cache_encoding: String,
    pub fp8_provider: Option<String>,
    pub prompt_tokens: u64,
    pub requested_max_completion_tokens: u32,
    pub completion_tokens: Option<u64>,
    pub elapsed_ns: u64,
    pub selected_backend: Option<String>,
    pub fallback_used: Option<bool>,
    pub all_dispatches_hip: Option<bool>,
    pub submission_count: Option<u64>,
    pub kernel_dispatch_count: Option<u64>,
    pub full_attention_layers: usize,
    pub linear_attention_layers: usize,
    pub logical_kv_capacity_tokens: Option<u64>,
    pub observed_kv_length_tokens: Option<u64>,
    pub physical_page_bytes: Option<u64>,
    pub kv_memory_kind: Option<String>,
    pub tokens_per_page: Option<u64>,
    pub mapped_kv_capacity_tokens: Option<u64>,
    pub committed_kv_bytes: Option<u64>,
    pub prefill_chunk_capacity_tokens: Option<u64>,
    pub prefill_chunk_count: Option<u64>,
    pub placement_total_memory_bytes: Option<u64>,
    pub placement_available_memory_bytes: Option<u64>,
    pub placement_required_bytes: Option<u64>,
    pub placement_incremental_required_bytes: Option<u64>,
    pub workspace_separate_allocation_bytes: Option<u64>,
    pub workspace_arena_bytes: Option<u64>,
    pub allocated_request_state_bytes: u64,
    pub allocated_workspace_bytes: u64,
    pub cleanup_request_state_bytes: u64,
    pub cleanup_workspace_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProductionShutdownAuditV1 {
    pub schema_version: String,
    pub target: String,
    pub model_fingerprint: String,
    pub plan_digest: String,
    pub model_ready_current_bytes: u64,
    pub final_current_bytes: u64,
    pub final_request_state_bytes: u64,
    pub final_workspace_bytes: u64,
    pub retryable_cleanup: usize,
    pub durable_quarantine: usize,
    pub requests: Vec<ProductionRequestAuditV1>,
}

pub struct QwenChatBackendV1 {
    state: Mutex<Option<QwenBackendStateV1>>,
    audits: Mutex<Vec<ProductionRequestAuditV1>>,
    shutdown_timeout: Duration,
    identity: BackendIdentityV1,
}

struct QwenBackendStateV1 {
    lock: Option<ModelLock>,
    moe_artifact: Option<Arc<VerifiedQwen35Moe>>,
    gguf_moe: Option<Arc<VerifiedGgufQwen35Moe>>,
    stop_policy: GenerationStopPolicyV1,
    tokenizer: TokenizerFrontendV1,
    renderer: Qwen35ChatTemplateV1,
    plan: WeightLoadPlan,
    resident: QwenResidentModel,
    mtp_resident: Option<QwenResidentModel>,
    mtp_plan: Option<WeightLoadPlan>,
    session: Arc<ExecutionSession>,
    target: String,
    model_ready_current_bytes: u64,
    sidecar: Option<Arc<VerifiedFp8Sidecar>>,
    nvfp4_sidecar: Option<Arc<VerifiedNvfp4Sidecar>>,
    fp8_provider: Option<String>,
    cache: Option<Arc<VerifiedCache>>,
    gguf_source: Option<Arc<VerifiedGgufWeightSource>>,
    vision_manifest: Option<QwenVisionManifest>,
    vision_resident: Option<QwenVisionResidentModel>,
    completion_timeout: Duration,
    kv_cache_encoding: KvCacheEncoding,
}

pub struct Gemma4ChatBackendV1 {
    state: Mutex<Option<Gemma4BackendStateV1>>,
    audits: Mutex<Vec<ProductionRequestAuditV1>>,
    shutdown_timeout: Duration,
    identity: BackendIdentityV1,
}

struct Gemma4BackendStateV1 {
    _lock: Gemma4ModelLock,
    tokenizer: TokenizerFrontendV1,
    stop_policy: GenerationStopPolicyV1,
    _plan: WeightLoadPlan,
    resident: Gemma4ResidentModel,
    session: Arc<ExecutionSession>,
    target: String,
    model_ready_current_bytes: u64,
    weight_encoding: String,
    kv_bytes_per_token: u64,
}

#[derive(Clone)]
struct BackendIdentityV1 {
    target: String,
    model_fingerprint: String,
    plan_digest: String,
    model_ready_current_bytes: u64,
    context_length: u32,
    recommended_context_tokens: u32,
}

impl QwenChatBackendV1 {
    pub fn open(config: QwenBackendConfigV1) -> Result<Self, BackendErrorV1> {
        config.validate()?;
        let derived = read_derived_gguf_lock(&config.derived_lock_path).map_err(|error| {
            BackendErrorV1::new(format!("derived GGUF lock validation failed: {error}"))
        })?;
        if derived.semantic_model_id.starts_with("qwen35moe:") {
            if config.kv_cache_encoding != KvCacheEncoding::Fp16 {
                return Err(BackendErrorV1::new(
                    "Qwen MoE currently requires FP16 KV cache",
                ));
            }
            return Self::open_gguf_moe(config, derived);
        }
        let lock = match builtin_reviewed_model_lock(&derived.source_lock_fingerprints).map_err(
            |error| BackendErrorV1::new(format!("built-in model lock resolution failed: {error}")),
        )? {
            ReviewedModelLock::Qwen35(lock) => lock,
            ReviewedModelLock::Gemma4(_) => {
                return Err(BackendErrorV1::new(
                    "Qwen backend requires a derived GGUF for a reviewed Qwen model",
                ));
            }
        };
        let verified = verify_derived_gguf(derived, &config.gguf_path)
            .map_err(|error| BackendErrorV1::new(format!("GGUF verification failed: {error}")))?;
        let (source, plan) = build_verified_gguf_qwen_weight_load_plan(
            &lock,
            verified,
            QwenComponentSelection::TEXT_ONLY,
        )
        .map_err(|error| {
            BackendErrorV1::new(format!("verified GGUF model load plan failed: {error}"))
        })?;
        let source = Arc::new(source);
        let tokenizer =
            TokenizerFrontendV1::from_qwen35_gguf(&lock, source.gguf()).map_err(|error| {
                BackendErrorV1::new(format!("verified tokenizer construction failed: {error}"))
            })?;
        let renderer =
            Qwen35ChatTemplateV1::from_qwen35_gguf(&lock, source.gguf()).map_err(|error| {
                BackendErrorV1::new(format!(
                    "verified chat renderer construction failed: {error}"
                ))
            })?;
        if source.has_fp8_recipe() && config.target != "gfx1201" {
            return Err(BackendErrorV1::new(
                "the embedded E4M3FN GGUF recipe currently requires the native gfx1201 provider",
            ));
        }
        let seed_graph = if source.has_fp8_recipe() {
            build_qwen35_gguf_fp8_graph(
                &lock,
                &plan,
                &source,
                1,
                1,
                sllm_core::DType::F8E4M3Fn,
                config.kv_cache_encoding,
            )
        } else {
            build_qwen35_graph_with_kv_cache_encoding(&lock, &plan, 1, 1, config.kv_cache_encoding)
        }
        .map_err(|error| {
            BackendErrorV1::new(format!("resident seed graph construction failed: {error}"))
        })?;
        let backend = HipBackend::connect()
            .map_err(|error| BackendErrorV1::new(format!("HIP backend is unavailable: {error}")))?;
        let session_request = ExecutionSessionRequest::new(config.device_index, &config.target)
            .map_err(|error| BackendErrorV1::new(format!("HIP session request failed: {error}")))?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|error| {
                BackendErrorV1::new(format!("exact HIP execution session failed: {error}"))
            })?;
        let resident = QwenResidentModel::new_gguf(
            Arc::clone(&session),
            seed_graph,
            plan.clone(),
            Arc::clone(&source),
            config.completion_timeout,
        )
        .map_err(|error| BackendErrorV1::new(format!("resident model load failed: {error}")))?;
        let vision_manifest = if lock.fingerprint() == sllm_core::QWEN35_4B_FINGERPRINT {
            Some(
                build_verified_gguf_qwen35_vision_manifest(&lock, &source).map_err(|error| {
                    BackendErrorV1::new(format!("GGUF vision manifest validation failed: {error}"))
                })?,
            )
        } else {
            None
        };
        let (mtp_resident, mtp_plan) = if !source.has_fp8_recipe()
            && config.target == "gfx1201"
            && config.kv_cache_encoding == KvCacheEncoding::Fp16
            && lock.fingerprint() == sllm_core::QWEN35_4B_FINGERPRINT
        {
            let mtp_plan = source
                .build_qwen_weight_load_plan(&lock, QwenComponentSelection::MTP_ONLY)
                .map_err(|error| {
                    BackendErrorV1::new(format!("GGUF MTP load plan validation failed: {error}"))
                })?;
            let mtp_graph = build_qwen35_mtp_graph(&lock, &mtp_plan, 1).map_err(|error| {
                BackendErrorV1::new(format!("GGUF MTP resident graph failed: {error}"))
            })?;
            let mtp_resident = QwenResidentModel::new_gguf(
                Arc::clone(&session),
                mtp_graph,
                mtp_plan.clone(),
                Arc::clone(&source),
                config.completion_timeout,
            )
            .map_err(|error| {
                BackendErrorV1::new(format!("GGUF MTP resident load failed: {error}"))
            })?;
            (Some(mtp_resident), Some(mtp_plan))
        } else {
            (None, None)
        };
        let ready = session.memory_snapshot();
        require_clean_request_memory(ready, "model-ready")?;
        let model_ready_current_bytes = ready.model_resident().current_bytes();
        if model_ready_current_bytes == 0 || ready.current_bytes() != model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "model-ready allocation accounting is not resident-only",
            ));
        }
        let fp8_provider = source.has_fp8_recipe().then(|| "gguf-native".to_owned());
        let identity = BackendIdentityV1 {
            target: config.target.clone(),
            model_fingerprint: lock.fingerprint().to_owned(),
            plan_digest: plan.digest_hex(),
            model_ready_current_bytes,
            context_length: config.context_length,
            recommended_context_tokens: QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32,
        };
        Ok(Self {
            state: Mutex::new(Some(QwenBackendStateV1 {
                stop_policy: lock.generation_stop_policy().clone(),
                lock: Some(lock),
                moe_artifact: None,
                gguf_moe: None,
                tokenizer,
                renderer,
                plan,
                resident,
                mtp_resident,
                mtp_plan,
                session,
                target: config.target,
                model_ready_current_bytes,
                sidecar: None,
                nvfp4_sidecar: None,
                fp8_provider,
                cache: None,
                gguf_source: Some(source),
                vision_manifest,
                vision_resident: None,
                completion_timeout: config.completion_timeout,
                kv_cache_encoding: config.kv_cache_encoding,
            })),
            audits: Mutex::new(Vec::new()),
            shutdown_timeout: config.shutdown_timeout,
            identity,
        })
    }

    fn open_gguf_moe(
        config: QwenBackendConfigV1,
        derived: sllm_core::DerivedGgufLock,
    ) -> Result<Self, BackendErrorV1> {
        let verified = verify_derived_gguf(derived, &config.gguf_path)
            .map_err(|error| BackendErrorV1::new(format!("GGUF verification failed: {error}")))?;
        let source = Arc::new(verify_gguf_qwen35_moe(verified).map_err(|error| {
            BackendErrorV1::new(format!("Qwen3.5 MoE GGUF validation failed: {error}"))
        })?);
        let tokenizer = TokenizerFrontendV1::from_qwen35_moe_gguf(&source).map_err(|error| {
            BackendErrorV1::new(format!("MoE tokenizer construction failed: {error}"))
        })?;
        let renderer = Qwen35ChatTemplateV1::from_qwen35_moe_gguf(&source)
            .map_err(|error| BackendErrorV1::new(format!("MoE chat renderer failed: {error}")))?;
        let plan = build_gguf_qwen35_moe_weight_load_plan(&source).map_err(|error| {
            BackendErrorV1::new(format!("MoE GGUF load plan validation failed: {error}"))
        })?;
        let seed_graph =
            build_qwen35_gguf_moe_execution_graph(&source, &plan, 1, 1).map_err(|error| {
                BackendErrorV1::new(format!("MoE GGUF resident graph failed: {error}"))
            })?;
        let backend = HipBackend::connect()
            .map_err(|error| BackendErrorV1::new(format!("HIP backend is unavailable: {error}")))?;
        let session = backend
            .open_execution_session(
                ExecutionSessionRequest::new(config.device_index, &config.target).map_err(
                    |error| BackendErrorV1::new(format!("HIP session request failed: {error}")),
                )?,
            )
            .map_err(|error| BackendErrorV1::new(format!("HIP session failed: {error}")))?;
        let resident = QwenResidentModel::new_gguf_moe(
            Arc::clone(&session),
            seed_graph,
            plan.clone(),
            Arc::clone(&source),
            config.completion_timeout,
        )
        .map_err(|error| BackendErrorV1::new(format!("MoE GGUF resident load failed: {error}")))?;
        let ready = session.memory_snapshot();
        require_clean_request_memory(ready, "MoE GGUF model-ready")?;
        let model_ready_current_bytes = ready.model_resident().current_bytes();
        if model_ready_current_bytes == 0 || ready.current_bytes() != model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "MoE GGUF model-ready accounting is not resident-only",
            ));
        }
        let identity = BackendIdentityV1 {
            target: config.target.clone(),
            model_fingerprint: sllm_core::QWEN35_MOE_MODEL_FINGERPRINT.to_owned(),
            plan_digest: plan.digest_hex(),
            model_ready_current_bytes,
            context_length: config.context_length,
            recommended_context_tokens: QWEN35_RECOMMENDED_CONTEXT_TOKENS as u32,
        };
        Ok(Self {
            state: Mutex::new(Some(QwenBackendStateV1 {
                lock: None,
                moe_artifact: None,
                gguf_moe: Some(source),
                stop_policy: qwen35_moe_generation_stop_policy(),
                tokenizer,
                renderer,
                plan,
                resident,
                mtp_resident: None,
                mtp_plan: None,
                session,
                target: config.target,
                model_ready_current_bytes,
                sidecar: None,
                nvfp4_sidecar: None,
                fp8_provider: Some("ocp-mxfp4-w4a4-mixed".to_owned()),
                cache: None,
                gguf_source: None,
                vision_manifest: None,
                vision_resident: None,
                completion_timeout: config.completion_timeout,
                kv_cache_encoding: KvCacheEncoding::Fp16,
            })),
            audits: Mutex::new(Vec::new()),
            shutdown_timeout: config.shutdown_timeout,
            identity,
        })
    }

    pub fn request_audits(&self) -> Vec<ProductionRequestAuditV1> {
        self.audits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.identity.model_fingerprint
    }

    pub const fn context_length(&self) -> u32 {
        self.identity.context_length
    }

    pub const fn recommended_context_tokens(&self) -> u32 {
        self.identity.recommended_context_tokens
    }

    pub fn target(&self) -> &str {
        &self.identity.target
    }

    pub fn shutdown(&self) -> Result<ProductionShutdownAuditV1, BackendErrorV1> {
        let state = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?
            .take()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is already shut down"))?;
        let QwenBackendStateV1 {
            resident,
            mtp_resident,
            vision_resident,
            session,
            model_ready_current_bytes,
            ..
        } = state;
        drop(resident);
        drop(mtp_resident);
        drop(vision_resident);
        let before_shutdown = session.memory_snapshot();
        if before_shutdown.current_bytes() != 0 {
            return Err(BackendErrorV1::new(format!(
                "resident drop left {} tracked device bytes",
                before_shutdown.current_bytes()
            )));
        }
        let report = session.shutdown(self.shutdown_timeout).map_err(|error| {
            BackendErrorV1::new(format!("HIP session shutdown failed: {error}"))
        })?;
        let final_memory = session.memory_snapshot();
        if final_memory.current_bytes() != 0
            || report.retryable_cleanup != 0
            || report.durable_quarantine != 0
        {
            return Err(BackendErrorV1::new(
                "HIP session shutdown did not reach a zero-cleanup terminal state",
            ));
        }
        Ok(ProductionShutdownAuditV1 {
            schema_version: "openai-chat-production-shutdown-v1".to_owned(),
            target: self.identity.target.clone(),
            model_fingerprint: self.identity.model_fingerprint.clone(),
            plan_digest: self.identity.plan_digest.clone(),
            model_ready_current_bytes,
            final_current_bytes: final_memory.current_bytes(),
            final_request_state_bytes: final_memory.request_state().current_bytes(),
            final_workspace_bytes: final_memory.workspace().current_bytes(),
            retryable_cleanup: report.retryable_cleanup,
            durable_quarantine: report.durable_quarantine,
            requests: self.request_audits(),
        })
    }

    fn record_audit(&self, audit: ProductionRequestAuditV1) {
        let mut audits = self
            .audits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if audits.len() == MAX_RETAINED_REQUEST_AUDITS {
            audits.remove(0);
        }
        audits.push(audit);
    }
}

impl Gemma4ChatBackendV1 {
    pub fn open(config: Gemma4BackendConfigV1) -> Result<Self, BackendErrorV1> {
        config.validate()?;
        let derived = read_derived_gguf_lock(&config.derived_lock_path).map_err(|error| {
            BackendErrorV1::new(format!("derived GGUF lock validation failed: {error}"))
        })?;
        let lock = match builtin_reviewed_model_lock(&derived.source_lock_fingerprints).map_err(
            |error| BackendErrorV1::new(format!("built-in model lock resolution failed: {error}")),
        )? {
            ReviewedModelLock::Gemma4(lock) => lock,
            ReviewedModelLock::Qwen35(_) => {
                return Err(BackendErrorV1::new(
                    "Gemma backend requires a derived GGUF for a reviewed Gemma 4 model",
                ));
            }
        };
        let verified = verify_derived_gguf(derived, &config.gguf_path)
            .map_err(|error| BackendErrorV1::new(format!("GGUF verification failed: {error}")))?;
        let (source, plan) =
            build_verified_gguf_gemma_weight_load_plan(&lock, verified).map_err(|error| {
                BackendErrorV1::new(format!("GGUF Gemma load plan failed: {error}"))
            })?;
        let source = Arc::new(source);
        let tokenizer =
            TokenizerFrontendV1::from_gemma4_gguf(&lock, source.gguf()).map_err(|error| {
                BackendErrorV1::new(format!(
                    "verified Gemma tokenizer construction failed: {error}"
                ))
            })?;
        let stop_policy = gemma4_generation_stop_policy(&lock).map_err(|error| {
            BackendErrorV1::new(format!("Gemma stop policy construction failed: {error}"))
        })?;
        let backend = HipBackend::connect()
            .map_err(|error| BackendErrorV1::new(format!("HIP backend is unavailable: {error}")))?;
        let session_request = ExecutionSessionRequest::new(config.device_index, &config.target)
            .map_err(|error| BackendErrorV1::new(format!("HIP session request failed: {error}")))?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|error| {
                BackendErrorV1::new(format!("exact HIP execution session failed: {error}"))
            })?;
        let resident = Gemma4ResidentModel::new_gguf_quantized(
            Arc::clone(&session),
            lock.clone(),
            plan.clone(),
            source,
            config.completion_timeout,
        )
        .map_err(|error| BackendErrorV1::new(format!("resident model load failed: {error}")))?;
        let ready = session.memory_snapshot();
        require_clean_request_memory(ready, "model-ready")?;
        let model_ready_current_bytes = ready.model_resident().current_bytes();
        if model_ready_current_bytes == 0 || ready.current_bytes() != model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "Gemma model-ready allocation accounting is not resident-only",
            ));
        }
        let identity = BackendIdentityV1 {
            target: config.target.clone(),
            model_fingerprint: lock.fingerprint().to_owned(),
            plan_digest: plan.digest_hex(),
            model_ready_current_bytes,
            context_length: config.context_length,
            recommended_context_tokens: GEMMA4_RECOMMENDED_CONTEXT_TOKENS as u32,
        };
        Ok(Self {
            state: Mutex::new(Some(Gemma4BackendStateV1 {
                _lock: lock,
                tokenizer,
                stop_policy,
                _plan: plan,
                resident,
                session,
                target: config.target,
                model_ready_current_bytes,
                weight_encoding: "mixed-nvfp4-w4a4-fp8-w8a8".to_owned(),
                kv_bytes_per_token: GEMMA4_STATIC_FP8_KV_BYTES_PER_TOKEN,
            })),
            audits: Mutex::new(Vec::new()),
            shutdown_timeout: config.shutdown_timeout,
            identity,
        })
    }

    pub fn request_audits(&self) -> Vec<ProductionRequestAuditV1> {
        self.audits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.identity.model_fingerprint
    }

    pub const fn context_length(&self) -> u32 {
        self.identity.context_length
    }

    pub const fn recommended_context_tokens(&self) -> u32 {
        self.identity.recommended_context_tokens
    }

    pub fn target(&self) -> &str {
        &self.identity.target
    }

    pub fn shutdown(&self) -> Result<ProductionShutdownAuditV1, BackendErrorV1> {
        let state = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Gemma backend state is poisoned"))?
            .take()
            .ok_or_else(|| BackendErrorV1::new("Gemma backend is already shut down"))?;
        let Gemma4BackendStateV1 {
            resident, session, ..
        } = state;
        drop(resident);
        let before_shutdown = session.memory_snapshot();
        if before_shutdown.current_bytes() != 0 {
            return Err(BackendErrorV1::new(format!(
                "Gemma resident drop left {} tracked device bytes",
                before_shutdown.current_bytes()
            )));
        }
        let report = session.shutdown(self.shutdown_timeout).map_err(|error| {
            BackendErrorV1::new(format!("HIP session shutdown failed: {error}"))
        })?;
        let final_memory = session.memory_snapshot();
        if final_memory.current_bytes() != 0
            || report.retryable_cleanup != 0
            || report.durable_quarantine != 0
        {
            return Err(BackendErrorV1::new(
                "HIP session shutdown did not reach a zero-cleanup terminal state",
            ));
        }
        Ok(ProductionShutdownAuditV1 {
            schema_version: "openai-chat-production-shutdown-v1".to_owned(),
            target: self.identity.target.clone(),
            model_fingerprint: self.identity.model_fingerprint.clone(),
            plan_digest: self.identity.plan_digest.clone(),
            model_ready_current_bytes: self.identity.model_ready_current_bytes,
            final_current_bytes: final_memory.current_bytes(),
            final_request_state_bytes: final_memory.request_state().current_bytes(),
            final_workspace_bytes: final_memory.workspace().current_bytes(),
            retryable_cleanup: report.retryable_cleanup,
            durable_quarantine: report.durable_quarantine,
            requests: self.request_audits(),
        })
    }

    fn record_audit(&self, audit: ProductionRequestAuditV1) {
        let mut audits = self
            .audits
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if audits.len() == MAX_RETAINED_REQUEST_AUDITS {
            audits.remove(0);
        }
        audits.push(audit);
    }
}

impl ChatGenerationBackendV1 for QwenChatBackendV1 {
    fn generate(
        &self,
        request: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        let started = Instant::now();
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Qwen backend state is poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| BackendErrorV1::new("Qwen backend is shut down"))?;
        let ready = state.session.memory_snapshot();
        require_clean_request_memory(ready, "request admission")?;
        if ready.model_resident().current_bytes() != state.model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "model-resident accounting changed before request admission",
            ));
        }

        let service = GenerationServiceV1::new(
            &state.tokenizer,
            Some(&state.renderer),
            &state.stop_policy,
        )
        .map_err(|error| BackendErrorV1::new(format!("generation service failed: {error}")))?;
        let input = GenerationInputV1::Messages {
            messages: request
                .messages()
                .iter()
                .map(|message| message.inner().clone())
                .collect(),
            options: Qwen35RenderOptionsV1 {
                add_generation_prompt: true,
                thinking: request.reasoning().thinking(),
            },
        };
        let prompt = service.prepare_input(&input).map_err(|error| {
            BackendErrorV1::new(format!("generation input preparation failed: {error}"))
        })?;
        let processed_images = request
            .messages()
            .iter()
            .flat_map(|message| message.parts())
            .filter_map(|part| match part {
                ChatContentPartV1::Image(image) => Some(image),
                ChatContentPartV1::Text(_) => None,
            })
            .collect::<Vec<_>>();
        let multimodal_prompt = if processed_images.is_empty() {
            None
        } else {
            if state.moe_artifact.is_some() || state.gguf_moe.is_some() {
                return Err(BackendErrorV1::new(
                    "Qwen3.5 MoE production path is text-only",
                ));
            }
            if state.sidecar.is_some() || state.nvfp4_sidecar.is_some() {
                return Err(BackendErrorV1::new(
                    "vision requests currently require the BF16 text artifact",
                ));
            }
            let vision_manifest = state.vision_manifest.clone().ok_or_else(|| {
                BackendErrorV1::new("vision requests require the fixed Qwen3.5-4B model")
            })?;
            if state.vision_resident.is_none() {
                state.vision_resident = Some(
                    match (&state.cache, &state.gguf_source) {
                        (Some(cache), None) => QwenVisionResidentModel::new(
                            Arc::clone(&state.session),
                            Arc::clone(cache),
                            vision_manifest.clone(),
                            state.completion_timeout,
                        ),
                        (None, Some(source)) => QwenVisionResidentModel::new_gguf(
                            Arc::clone(&state.session),
                            Arc::clone(source),
                            vision_manifest.clone(),
                            state.completion_timeout,
                        ),
                        _ => {
                            return Err(BackendErrorV1::new(
                                "vision requires exactly one verified dense weight source",
                            ));
                        }
                    }
                    .map_err(|error| {
                        BackendErrorV1::new(format!("vision resident load failed: {error}"))
                    })?,
                );
                let ready = state.session.memory_snapshot();
                require_clean_request_memory(ready, "vision model-ready")?;
                state.model_ready_current_bytes = ready.model_resident().current_bytes();
            }
            let vision = state
                .vision_resident
                .as_ref()
                .expect("vision resident was initialized");
            let images = processed_images
                .iter()
                .map(|image| {
                    let output = vision
                        .execute(&QwenVisionExecutionInput {
                            grid_thw: image.grid_thw,
                            patch_width: image.patch_width,
                            patches: image.patches.clone(),
                        })
                        .map_err(|error| {
                            BackendErrorV1::new(format!("vision encode failed: {error}"))
                        })?;
                    Ok(QwenMultimodalImageEmbedding {
                        grid_thw: image.grid_thw,
                        embeddings_bf16: output.embeddings_bf16().to_vec(),
                    })
                })
                .collect::<Result<Vec<_>, BackendErrorV1>>()?;
            let assembled = match (&state.cache, &state.gguf_source) {
                (Some(cache), None) => assemble_qwen35_multimodal_prompt(
                    cache,
                    &prompt,
                    vision_manifest.image_pad_token,
                    &images,
                ),
                (None, Some(source)) => assemble_gguf_qwen35_multimodal_prompt(
                    source,
                    &prompt,
                    vision_manifest.image_pad_token,
                    &images,
                ),
                _ => {
                    return Err(BackendErrorV1::new(
                        "multimodal assembly requires exactly one verified dense weight source",
                    ));
                }
            }
            .map_err(|error| {
                BackendErrorV1::new(format!("multimodal prompt assembly failed: {error}"))
            })?;
            Some(assembled)
        };
        let prompt_tokens = u64::try_from(prompt.len())
            .map_err(|_| BackendErrorV1::new("prompt token count overflowed u64"))?;
        let state_capacity = prompt_tokens
            .checked_add(u64::from(request.generation().max_new_tokens()))
            .ok_or_else(|| BackendErrorV1::new("request state capacity overflowed u64"))?;
        if state_capacity > u64::from(self.identity.context_length) {
            return Err(BackendErrorV1::new(format!(
                "request requires {state_capacity} context tokens but the server was started with --context-length {}",
                self.identity.context_length
            )));
        }
        if multimodal_prompt.is_some() && state.kv_cache_encoding != KvCacheEncoding::Fp16 {
            return Err(BackendErrorV1::new(
                "multimodal Qwen requests currently require FP16 KV cache",
            ));
        }
        let placement_total_memory_bytes = state
            .session
            .total_memory_bytes()
            .map_err(|error| BackendErrorV1::new(error.to_string()))?
            .ok_or_else(|| BackendErrorV1::new("backend omitted total device memory"))?;
        let placement_available_memory_bytes = state
            .session
            .available_memory_bytes()
            .map_err(|error| BackendErrorV1::new(error.to_string()))?
            .ok_or_else(|| BackendErrorV1::new("backend omitted available device memory"))?;
        let chunk_candidates = if multimodal_prompt.is_some() {
            vec![prompt_tokens]
        } else {
            qwen_prefill_chunk_candidates(placement_total_memory_bytes, prompt_tokens)
                .map_err(|error| BackendErrorV1::new(error.to_string()))?
        };
        let mtp_target = state.mtp_resident.is_some()
            && !request.generation().sampling().requires_logits()
            && multimodal_prompt.is_none();
        let build_graph = |chunk_rows: u64| {
            let target_rows = if mtp_target {
                chunk_rows.max(2)
            } else {
                chunk_rows
            };
            if let Some(artifact) = &state.moe_artifact {
                build_qwen35_moe_execution_graph(artifact, &state.plan, chunk_rows, state_capacity)
            } else if let Some(source) = &state.gguf_moe {
                build_qwen35_gguf_moe_execution_graph(
                    source,
                    &state.plan,
                    chunk_rows,
                    state_capacity,
                )
            } else if multimodal_prompt.is_some() {
                build_qwen35_multimodal_graph(
                    state.lock.as_ref().expect("dense Qwen lock"),
                    &state.plan,
                    prompt_tokens,
                    state_capacity,
                )
            } else if let Some(nvfp4_sidecar) = &state.nvfp4_sidecar {
                build_qwen35_nvfp4_graph(
                    state.lock.as_ref().expect("dense Qwen lock"),
                    &state.plan,
                    nvfp4_sidecar,
                    chunk_rows,
                    state_capacity,
                )
            } else if let Some(source) = state
                .gguf_source
                .as_ref()
                .filter(|source| source.has_fp8_recipe())
            {
                build_qwen35_gguf_fp8_graph(
                    state.lock.as_ref().expect("dense Qwen lock"),
                    &state.plan,
                    source,
                    chunk_rows,
                    state_capacity,
                    sllm_core::DType::F8E4M3Fn,
                    state.kv_cache_encoding,
                )
            } else {
                match (&state.sidecar, state.fp8_provider.as_deref()) {
                    (Some(_), Some("converted-bf16")) | (None, None) => {
                        build_qwen35_graph_with_kv_cache_encoding(
                            state.lock.as_ref().expect("dense Qwen lock"),
                            &state.plan,
                            target_rows,
                            state_capacity,
                            state.kv_cache_encoding,
                        )
                    }
                    (Some(sidecar), Some("native-fnuz")) => build_qwen35_fp8_fnuz_graph(
                        state.lock.as_ref().expect("dense Qwen lock"),
                        &state.plan,
                        sidecar,
                        chunk_rows,
                        state_capacity,
                    ),
                    (Some(sidecar), Some(_)) => build_qwen35_fp8_graph(
                        state.lock.as_ref().expect("dense Qwen lock"),
                        &state.plan,
                        sidecar,
                        chunk_rows,
                        state_capacity,
                    ),
                    _ => unreachable!("validated FP8 server state has a selected provider"),
                }
            }
        };
        let mut rejected = Vec::new();
        let mut selected = None;
        for chunk_rows in chunk_candidates {
            let graph = build_graph(chunk_rows)
                .map_err(|error| BackendErrorV1::new(format!("request graph failed: {error}")))?;
            let estimate =
                qwen_graph_memory_estimate(&graph, &state.plan, placement_total_memory_bytes)
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?;
            let incremental_required = estimate
                .required_bytes()
                .checked_sub(estimate.model_resident_bytes())
                .ok_or_else(|| {
                    BackendErrorV1::new("request placement underflowed graph model-resident bytes")
                })?;
            if incremental_required <= placement_available_memory_bytes {
                selected = Some((graph, estimate, incremental_required));
                break;
            }
            rejected.push(format!("{chunk_rows}:{incremental_required}"));
        }
        let (graph, placement, placement_incremental_required_bytes) = selected.ok_or_else(|| {
            BackendErrorV1::new(format!(
                "no prefill chunk fits available device memory {}; candidates chunk:incremental-required [{}]",
                placement_available_memory_bytes,
                rejected.join(",")
            ))
        })?;
        let prefill_chunk_capacity_tokens = graph.token_count();
        let mut owner = state
            .resident
            .new_request_for_session(Arc::clone(&state.session), graph)
            .map_err(|error| {
                BackendErrorV1::new(format!("request provisioning failed: {error}"))
            })?;
        let mut allocated = state.session.memory_snapshot();
        let mut random = OsSamplingRandom::for_parameters_and_seed(
            request.generation().sampling(),
            request.sampling_seed(),
        )
        .map_err(|error| BackendErrorV1::new(format!("sampling source failed: {error}")))?;
        let mut output_sink = OutputSinkAdapterV1 { inner: sink };
        let (outcome, dispatch, memory, prefill_chunk_count) = if let Some(multimodal_prompt) =
            multimodal_prompt.as_ref()
        {
            let mut executor = QwenMultimodalExecutorV1 {
                inner: &mut owner,
                prompt: multimodal_prompt,
                prefilled: false,
            };
            let outcome = service.generate_tokens_with_sink(
                &mut executor,
                &prompt,
                request.generation(),
                cancellation,
                &mut random,
                &mut output_sink,
            );
            let dispatch = owner.audit_snapshot().ok();
            let memory = owner.memory_audit_snapshot().ok();
            let prefill_chunk_count = owner.prefill_chunk_count();
            drop(owner);
            (outcome, dispatch, memory, Some(prefill_chunk_count))
        } else if !request.generation().sampling().requires_logits()
            && let (Some(mtp_resident), Some(mtp_plan)) = (&state.mtp_resident, &state.mtp_plan)
        {
            let mtp_graph = build_qwen35_mtp_graph(
                state.lock.as_ref().expect("MTP requires dense Qwen lock"),
                mtp_plan,
                state_capacity,
            )
            .map_err(|error| BackendErrorV1::new(format!("MTP request graph failed: {error}")))?;
            let mtp_owner = mtp_resident
                .new_request_for_session(Arc::clone(&state.session), mtp_graph)
                .map_err(|error| {
                    BackendErrorV1::new(format!("MTP request provisioning failed: {error}"))
                })?;
            allocated = state.session.memory_snapshot();
            let mut executor = SpeculativeGenerationAdapterV1::new(
                QwenMtpGenerationExecutorV1::new_with_draft_width(owner, mtp_owner, 1)
                    .map_err(|error| BackendErrorV1::new(error.to_string()))?,
            );
            let outcome = service.generate_tokens_with_sink(
                &mut executor,
                &prompt,
                request.generation(),
                cancellation,
                &mut random,
                &mut output_sink,
            );
            let dispatch = executor.inner().target().audit_snapshot().ok();
            let memory = executor.inner().target().memory_audit_snapshot().ok();
            let prefill_chunk_count = executor.inner().target().prefill_chunk_count();
            drop(executor);
            (outcome, dispatch, memory, Some(prefill_chunk_count))
        } else {
            let outcome = service.generate_tokens_with_sink(
                &mut owner,
                &prompt,
                request.generation(),
                cancellation,
                &mut random,
                &mut output_sink,
            );
            let dispatch = owner.audit_snapshot().ok();
            let memory = owner.memory_audit_snapshot().ok();
            let prefill_chunk_count = owner.prefill_chunk_count();
            drop(owner);
            (outcome, dispatch, memory, Some(prefill_chunk_count))
        };
        let cleanup = state.session.memory_snapshot();
        let cleanup_result =
            require_clean_request_memory(cleanup, "request cleanup").and_then(|()| {
                if cleanup.model_resident().current_bytes() == state.model_ready_current_bytes {
                    Ok(())
                } else {
                    Err(BackendErrorV1::new(
                        "model-resident accounting changed after request cleanup",
                    ))
                }
            });
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let generation_result = outcome
            .map_err(|error| BackendErrorV1::new(format!("generation failed: {error}")))
            .and_then(|result| {
                let dispatch = dispatch.as_ref().ok_or_else(|| {
                    BackendErrorV1::new("completed generation has no dispatch audit")
                })?;
                if dispatch.selected_backend() != "hip"
                    || dispatch.target() != state.target
                    || dispatch.fallback_used()
                    || !dispatch.all_dispatches_hip()
                {
                    return Err(BackendErrorV1::new(
                        "completed generation dispatch audit is not exact HIP/no-fallback",
                    ));
                }
                if memory.is_none() {
                    return Err(BackendErrorV1::new(
                        "completed generation has no physical-memory audit",
                    ));
                }
                let finish_reason = match result.finish_reason() {
                    sllm_frontend::FinishReasonV1::Stop => FinishReasonV1::Stop,
                    sllm_frontend::FinishReasonV1::Length => FinishReasonV1::Length,
                };
                let usage = result.usage();
                Ok(BackendCompletionV1 {
                    finish_reason,
                    usage: TokenUsageV1::new(usage.prompt_tokens(), usage.completion_tokens())
                        .map_err(|error| BackendErrorV1::new(error.to_string()))?,
                })
            });
        let result = match (generation_result, cleanup_result) {
            (_, Err(error)) => Err(error),
            (result, Ok(())) => result,
        };
        let first_kv = memory.as_ref().and_then(|audit| audit.kv_layers().first());
        let committed_kv_bytes = memory
            .as_ref()
            .and_then(|audit| audit.committed_kv_bytes().ok());
        let completion_tokens = result
            .as_ref()
            .ok()
            .map(|value| value.usage.completion_tokens);
        self.record_audit(ProductionRequestAuditV1 {
            outcome: if cancellation.is_cancelled() {
                "cancelled".to_owned()
            } else if result.is_ok() {
                "completed".to_owned()
            } else {
                "failed".to_owned()
            },
            target: state.target.clone(),
            weight_encoding: match state.fp8_provider.as_deref() {
                None => "bf16".to_owned(),
                Some("converted-bf16") => "bf16-converted-from-ocp-e4m3fn".to_owned(),
                Some("native-fnuz") => "e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32".to_owned(),
                Some("nvfp4-packed-dequant") => "nvfp4-e2m1-block16-e4m3fn-tensor-f32".to_owned(),
                Some("ocp-mxfp4-w4a4-mixed") => "ocp-mxfp4-e2m1-block32-e8m0-mixed".to_owned(),
                Some(_) => "ocp-e4m3fn-outer-f32".to_owned(),
            },
            kv_cache_encoding: match state.kv_cache_encoding {
                KvCacheEncoding::Fp16 => "fp16",
                KvCacheEncoding::Fp8E4M3Fn => "fp8",
                KvCacheEncoding::Fp8E4M3FnStatic => "fp8-static",
                KvCacheEncoding::Nvfp4 => "nvfp4",
            }
            .to_owned(),
            fp8_provider: state.fp8_provider.clone(),
            prompt_tokens,
            requested_max_completion_tokens: request.generation().max_new_tokens(),
            completion_tokens,
            elapsed_ns,
            selected_backend: dispatch
                .as_ref()
                .map(|audit| audit.selected_backend().to_owned()),
            fallback_used: dispatch.as_ref().map(|audit| audit.fallback_used()),
            all_dispatches_hip: dispatch.as_ref().map(|audit| audit.all_dispatches_hip()),
            submission_count: dispatch.as_ref().map(|audit| audit.submission_count()),
            kernel_dispatch_count: dispatch.as_ref().map(|audit| audit.kernel_dispatch_count()),
            full_attention_layers: memory.as_ref().map_or(0, |audit| audit.kv_layers().len()),
            linear_attention_layers: memory
                .as_ref()
                .map_or(0, |audit| audit.linear_attention_layers()),
            logical_kv_capacity_tokens: first_kv.map(|layer| layer.logical_capacity_tokens()),
            observed_kv_length_tokens: first_kv.map(|layer| layer.observed_length_tokens()),
            physical_page_bytes: first_kv.map(|layer| layer.physical().physical_page_bytes()),
            kv_memory_kind: first_kv.map(|layer| match layer.physical().memory_kind() {
                sllm_core::KvMemoryKind::VirtualContiguous => "virtual-contiguous".to_owned(),
                sllm_core::KvMemoryKind::ContiguousResident => "contiguous-resident".to_owned(),
            }),
            tokens_per_page: first_kv.map(|layer| layer.physical().tokens_per_page()),
            mapped_kv_capacity_tokens: first_kv
                .map(|layer| layer.physical().mapped_token_capacity()),
            committed_kv_bytes,
            prefill_chunk_capacity_tokens: Some(prefill_chunk_capacity_tokens),
            prefill_chunk_count,
            placement_total_memory_bytes: Some(placement_total_memory_bytes),
            placement_available_memory_bytes: Some(placement_available_memory_bytes),
            placement_required_bytes: Some(placement.required_bytes()),
            placement_incremental_required_bytes: Some(placement_incremental_required_bytes),
            workspace_separate_allocation_bytes: Some(placement.workspace_baseline_bytes()),
            workspace_arena_bytes: Some(placement.workspace_arena_bytes()),
            allocated_request_state_bytes: allocated.request_state().current_bytes(),
            allocated_workspace_bytes: allocated.workspace().current_bytes(),
            cleanup_request_state_bytes: cleanup.request_state().current_bytes(),
            cleanup_workspace_bytes: cleanup.workspace().current_bytes(),
        });
        result
    }
}

impl ChatGenerationBackendV1 for Gemma4ChatBackendV1 {
    fn generate(
        &self,
        request: &ChatCompletionRequestV1,
        cancellation: &GenerationCancellationV1,
        sink: &mut dyn GenerationDeltaSinkV1,
    ) -> Result<BackendCompletionV1, BackendErrorV1> {
        let started = Instant::now();
        if request.reasoning().enabled() || request.reasoning().separate_reasoning() {
            return Err(BackendErrorV1::new(
                "Gemma 4 base raw-text profile does not support reasoning mode",
            ));
        }
        let mut state_guard = self
            .state
            .lock()
            .map_err(|_| BackendErrorV1::new("Gemma backend state is poisoned"))?;
        let state = state_guard
            .as_mut()
            .ok_or_else(|| BackendErrorV1::new("Gemma backend is shut down"))?;
        let ready = state.session.memory_snapshot();
        require_clean_request_memory(ready, "Gemma request admission")?;
        if ready.model_resident().current_bytes() != state.model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "Gemma model-resident accounting changed before request admission",
            ));
        }

        let service = GenerationServiceV1::new(&state.tokenizer, None, &state.stop_policy)
            .map_err(|error| BackendErrorV1::new(format!("generation service failed: {error}")))?;
        let rendered = render_gemma4_raw_messages(request.messages())?;
        let prompt = service
            .prepare_input(&GenerationInputV1::Prompt(rendered))
            .map_err(|error| {
                BackendErrorV1::new(format!("generation input preparation failed: {error}"))
            })?;
        let prompt_tokens = u64::try_from(prompt.len())
            .map_err(|_| BackendErrorV1::new("prompt token count overflowed u64"))?;
        let state_capacity = prompt_tokens
            .checked_add(u64::from(request.generation().max_new_tokens()))
            .ok_or_else(|| BackendErrorV1::new("request state capacity overflowed u64"))?;
        if state_capacity > u64::from(self.identity.context_length) {
            return Err(BackendErrorV1::new(format!(
                "request requires {state_capacity} context tokens but the server was started with --context-length {}",
                self.identity.context_length
            )));
        }
        let mut owner = state
            .resident
            .new_request_for_session(Arc::clone(&state.session), prompt_tokens, state_capacity)
            .map_err(|error| {
                BackendErrorV1::new(format!("request provisioning failed: {error}"))
            })?;
        let allocated = state.session.memory_snapshot();
        let mut random = OsSamplingRandom::for_parameters_and_seed(
            request.generation().sampling(),
            request.sampling_seed(),
        )
        .map_err(|error| BackendErrorV1::new(format!("sampling source failed: {error}")))?;
        let mut output_sink = OutputSinkAdapterV1 { inner: sink };
        let outcome = service.generate_tokens_with_sink(
            &mut owner,
            &prompt,
            request.generation(),
            cancellation,
            &mut random,
            &mut output_sink,
        );
        let dispatch = owner.audit_snapshot().ok();
        let observed_length = owner.committed_length();
        drop(owner);
        let cleanup = state.session.memory_snapshot();
        let cleanup_result = require_clean_request_memory(cleanup, "Gemma request cleanup")
            .and_then(|()| {
                if cleanup.model_resident().current_bytes() == state.model_ready_current_bytes {
                    Ok(())
                } else {
                    Err(BackendErrorV1::new(
                        "Gemma model-resident accounting changed after request cleanup",
                    ))
                }
            });
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let generation_result = outcome
            .map_err(|error| BackendErrorV1::new(format!("generation failed: {error}")))
            .and_then(|result| {
                let dispatch = dispatch.as_ref().ok_or_else(|| {
                    BackendErrorV1::new("completed Gemma generation has no dispatch audit")
                })?;
                if dispatch.target() != state.target || dispatch.fallback_used() {
                    return Err(BackendErrorV1::new(
                        "completed Gemma generation is not exact HIP/no-fallback",
                    ));
                }
                let finish_reason = match result.finish_reason() {
                    sllm_frontend::FinishReasonV1::Stop => FinishReasonV1::Stop,
                    sllm_frontend::FinishReasonV1::Length => FinishReasonV1::Length,
                };
                let usage = result.usage();
                Ok(BackendCompletionV1 {
                    finish_reason,
                    usage: TokenUsageV1::new(usage.prompt_tokens(), usage.completion_tokens())
                        .map_err(|error| BackendErrorV1::new(error.to_string()))?,
                })
            });
        let result = match (generation_result, cleanup_result) {
            (_, Err(error)) => Err(error),
            (result, Ok(())) => result,
        };
        let completion_tokens = result
            .as_ref()
            .ok()
            .map(|value| value.usage.completion_tokens);
        self.record_audit(ProductionRequestAuditV1 {
            outcome: if cancellation.is_cancelled() {
                "cancelled".to_owned()
            } else if result.is_ok() {
                "completed".to_owned()
            } else {
                "failed".to_owned()
            },
            target: state.target.clone(),
            weight_encoding: state.weight_encoding.clone(),
            kv_cache_encoding: "fp8-static".to_owned(),
            fp8_provider: None,
            prompt_tokens,
            requested_max_completion_tokens: request.generation().max_new_tokens(),
            completion_tokens,
            elapsed_ns,
            selected_backend: dispatch.as_ref().map(|_| "hip".to_owned()),
            fallback_used: dispatch.as_ref().map(|audit| audit.fallback_used()),
            all_dispatches_hip: dispatch.as_ref().map(|audit| !audit.fallback_used()),
            submission_count: dispatch.as_ref().map(|audit| audit.submission_count()),
            kernel_dispatch_count: dispatch.as_ref().map(|audit| audit.kernel_dispatch_count()),
            full_attention_layers: 8,
            linear_attention_layers: 0,
            logical_kv_capacity_tokens: Some(state_capacity),
            observed_kv_length_tokens: Some(observed_length),
            physical_page_bytes: None,
            kv_memory_kind: Some("contiguous-resident".to_owned()),
            tokens_per_page: None,
            mapped_kv_capacity_tokens: Some(state_capacity),
            committed_kv_bytes: observed_length.checked_mul(state.kv_bytes_per_token),
            prefill_chunk_capacity_tokens: None,
            prefill_chunk_count: None,
            placement_total_memory_bytes: None,
            placement_available_memory_bytes: None,
            placement_required_bytes: None,
            placement_incremental_required_bytes: None,
            workspace_separate_allocation_bytes: None,
            workspace_arena_bytes: None,
            allocated_request_state_bytes: allocated.request_state().current_bytes(),
            allocated_workspace_bytes: allocated.workspace().current_bytes(),
            cleanup_request_state_bytes: cleanup.request_state().current_bytes(),
            cleanup_workspace_bytes: cleanup.workspace().current_bytes(),
        });
        result
    }
}

fn render_gemma4_raw_messages(messages: &[crate::ChatMessageV1]) -> Result<String, BackendErrorV1> {
    let mut rendered = String::new();
    for message in messages {
        let (role, content) = match message.inner() {
            Qwen35ChatMessageV1::System { content } => ("System", content),
            Qwen35ChatMessageV1::User { content } => ("User", content),
            Qwen35ChatMessageV1::Assistant {
                content,
                reasoning_content,
            } => {
                if reasoning_content.is_some() {
                    return Err(BackendErrorV1::new(
                        "Gemma 4 base raw-text profile rejects reasoning history",
                    ));
                }
                ("Assistant", content)
            }
        };
        rendered.push_str(role);
        rendered.push_str(": ");
        rendered.push_str(content);
        rendered.push('\n');
        if rendered.len() > GEMMA4_RAW_CHAT_MAX_BYTES {
            return Err(BackendErrorV1::new(
                "Gemma raw chat transcript exceeds the host byte limit",
            ));
        }
    }
    rendered.push_str("Assistant:");
    if rendered.len() > GEMMA4_RAW_CHAT_MAX_BYTES {
        return Err(BackendErrorV1::new(
            "Gemma raw chat transcript exceeds the host byte limit",
        ));
    }
    Ok(rendered)
}

struct OutputSinkAdapterV1<'a> {
    inner: &'a mut dyn GenerationDeltaSinkV1,
}

struct QwenMultimodalExecutorV1<'a> {
    inner: &'a mut QwenExecutionRequest,
    prompt: &'a QwenMultimodalPrompt,
    prefilled: bool,
}

impl GenerationExecutorV1 for QwenMultimodalExecutorV1<'_> {
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
        let device_argmax = output
            .token_ids()
            .last()
            .copied()
            .ok_or(GenerationServiceError::MissingDeviceArgmax)
            .and_then(|token| {
                u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)
            })?;
        Ok(GenerationStepV1::new(
            device_argmax,
            output.last_logits().map(ToOwned::to_owned),
        ))
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
        let device_argmax = output
            .token_ids()
            .last()
            .copied()
            .ok_or(GenerationServiceError::MissingDeviceArgmax)
            .and_then(|token| {
                u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)
            })?;
        Ok(GenerationStepV1::new(
            device_argmax,
            output.last_logits().map(ToOwned::to_owned),
        ))
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }
}

impl GenerationOutputSinkV1 for OutputSinkAdapterV1<'_> {
    fn publish(&mut self, delta: &str) -> Result<(), GenerationServiceError> {
        self.inner
            .publish(delta)
            .map_err(|error| GenerationServiceError::Output(error.to_string()))
    }
}

fn require_clean_request_memory(
    snapshot: AllocationSnapshot,
    boundary: &str,
) -> Result<(), BackendErrorV1> {
    if snapshot.poisoned()
        || snapshot.request_state().current_bytes() != 0
        || snapshot.workspace().current_bytes() != 0
    {
        return Err(BackendErrorV1::new(format!(
            "{boundary} has nonzero or poisoned request allocation accounting"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatMessageV1;

    fn message(inner: Qwen35ChatMessageV1) -> ChatMessageV1 {
        let content = match &inner {
            Qwen35ChatMessageV1::System { content }
            | Qwen35ChatMessageV1::User { content }
            | Qwen35ChatMessageV1::Assistant { content, .. } => content.clone(),
        };
        ChatMessageV1 {
            inner,
            parts: vec![crate::api::ChatContentPartV1::Text(content)],
        }
    }

    #[test]
    fn gemma_raw_transcript_is_versioned_by_exact_roles_and_unicode() {
        let rendered = render_gemma4_raw_messages(&[
            message(Qwen35ChatMessageV1::system("方針")),
            message(Qwen35ChatMessageV1::user("こんにちは🌙")),
            message(Qwen35ChatMessageV1::assistant("了解", None)),
            message(Qwen35ChatMessageV1::user("続けて")),
        ])
        .unwrap();
        assert_eq!(
            rendered,
            "System: 方針\nUser: こんにちは🌙\nAssistant: 了解\nUser: 続けて\nAssistant:"
        );
        assert!(
            render_gemma4_raw_messages(&[message(Qwen35ChatMessageV1::assistant(
                "visible",
                Some("hidden".to_owned()),
            ))])
            .is_err()
        );
    }

    #[test]
    fn gemma_raw_transcript_checks_both_sides_of_the_byte_cap() {
        let overhead = "User: \nAssistant:".len();
        let accepted = "x".repeat(GEMMA4_RAW_CHAT_MAX_BYTES - overhead);
        assert_eq!(
            render_gemma4_raw_messages(&[message(Qwen35ChatMessageV1::user(accepted))])
                .unwrap()
                .len(),
            GEMMA4_RAW_CHAT_MAX_BYTES
        );
        let rejected = "x".repeat(GEMMA4_RAW_CHAT_MAX_BYTES - overhead + 1);
        assert!(
            render_gemma4_raw_messages(&[message(Qwen35ChatMessageV1::user(rejected))]).is_err()
        );
    }
}
