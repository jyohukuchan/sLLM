//! Production model backends for the profile-v1 transport.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use sllm_core::{
    AllocationSnapshot, Backend, ExecutionSession, ExecutionSessionRequest, Gemma4ModelLock,
    Gemma4ResidentModel, ModelLock, OsSamplingRandom, QwenResidentModel, ReviewedModelLock,
    VerifiedFp8Sidecar, VerifiedNvfp4Sidecar, WeightLoadPlan, build_qwen35_fp8_fnuz_graph,
    build_qwen35_fp8_graph, build_qwen35_graph, build_qwen35_nvfp4_graph,
    build_verified_gemma4_weight_load_plan, build_verified_weight_load_plan, read_model_lock,
    read_reviewed_model_lock, verify_fp8_sidecar, verify_nvfp4_sidecar,
};
use sllm_frontend::{
    GenerationCancellationV1, GenerationInputV1, GenerationOutputSinkV1, GenerationServiceError,
    GenerationServiceV1, GenerationStopPolicyV1, Qwen35ChatMessageV1, Qwen35ChatTemplateV1,
    Qwen35RenderOptionsV1, TokenizerFrontendV1, gemma4_generation_stop_policy,
};
use sllm_hip::HipBackend;

use crate::{
    BackendCompletionV1, BackendErrorV1, ChatCompletionRequestV1, ChatGenerationBackendV1,
    FinishReasonV1, GenerationDeltaSinkV1, TokenUsageV1,
};

const MAX_RETAINED_REQUEST_AUDITS: usize = 64;
const GEMMA4_RAW_CHAT_MAX_BYTES: usize = 16 * 1024 * 1024;
const GEMMA4_KV_BYTES_PER_TOKEN: u64 = 344_064;

#[derive(Clone, Debug)]
pub struct QwenBackendConfigV1 {
    pub lock_path: PathBuf,
    pub cache_path: PathBuf,
    pub device_index: u32,
    pub target: String,
    pub completion_timeout: Duration,
    pub shutdown_timeout: Duration,
    pub fp8_manifest_path: Option<PathBuf>,
    pub fp8_artifact_path: Option<PathBuf>,
    pub fp8_provider: Option<String>,
}

impl QwenBackendConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        if self.target.is_empty()
            || !self.target.is_ascii()
            || self.completion_timeout.is_zero()
            || self.shutdown_timeout.is_zero()
        {
            return Err(BackendErrorV1::new(
                "Qwen backend target and timeouts must be valid and nonzero",
            ));
        }
        let sidecar = self.fp8_manifest_path.is_some() && self.fp8_artifact_path.is_some();
        if self.fp8_manifest_path.is_some() != self.fp8_artifact_path.is_some()
            || (!sidecar && self.fp8_provider.is_some())
        {
            return Err(BackendErrorV1::new(
                "FP8 server configuration requires manifest and artifact together",
            ));
        }
        if sidecar {
            let provider = self
                .fp8_provider
                .as_deref()
                .unwrap_or(match self.target.as_str() {
                    "gfx1201" => "native",
                    "gfx942" => "native-fnuz",
                    _ => "converted-bf16",
                });
            let valid = matches!(
                (provider, self.target.as_str()),
                ("native", "gfx1201")
                    | ("native-fnuz", "gfx942")
                    | ("emulation" | "converted-bf16", "gfx1030")
                    | ("nvfp4-packed-dequant", "gfx1030" | "gfx1201")
            );
            if !valid {
                return Err(BackendErrorV1::new(
                    "FP8 server provider is incompatible with the exact target",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Gemma4BackendConfigV1 {
    pub lock_path: PathBuf,
    pub cache_path: PathBuf,
    pub device_index: u32,
    pub target: String,
    pub completion_timeout: Duration,
    pub shutdown_timeout: Duration,
}

impl Gemma4BackendConfigV1 {
    pub fn validate(&self) -> Result<(), BackendErrorV1> {
        if self.target.is_empty()
            || !self.target.is_ascii()
            || self.completion_timeout.is_zero()
            || self.shutdown_timeout.is_zero()
        {
            return Err(BackendErrorV1::new(
                "Gemma backend target and timeouts must be valid and nonzero",
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
    lock: ModelLock,
    tokenizer: TokenizerFrontendV1,
    renderer: Qwen35ChatTemplateV1,
    plan: WeightLoadPlan,
    resident: QwenResidentModel,
    session: Arc<ExecutionSession>,
    target: String,
    model_ready_current_bytes: u64,
    sidecar: Option<Arc<VerifiedFp8Sidecar>>,
    nvfp4_sidecar: Option<Arc<VerifiedNvfp4Sidecar>>,
    fp8_provider: Option<String>,
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
}

#[derive(Clone)]
struct BackendIdentityV1 {
    target: String,
    model_fingerprint: String,
    plan_digest: String,
    model_ready_current_bytes: u64,
}

impl QwenChatBackendV1 {
    pub fn open(config: QwenBackendConfigV1) -> Result<Self, BackendErrorV1> {
        config.validate()?;
        let lock = read_model_lock(&config.lock_path).map_err(|error| {
            BackendErrorV1::new(format!("model lock validation failed: {error}"))
        })?;
        let cache = Arc::new(lock.verify_cache(&config.cache_path).map_err(|error| {
            BackendErrorV1::new(format!("model cache verification failed: {error}"))
        })?);
        let tokenizer =
            TokenizerFrontendV1::from_verified_cache(&lock, &cache).map_err(|error| {
                BackendErrorV1::new(format!("verified tokenizer construction failed: {error}"))
            })?;
        let renderer =
            Qwen35ChatTemplateV1::from_verified_cache(&lock, &cache).map_err(|error| {
                BackendErrorV1::new(format!(
                    "verified chat renderer construction failed: {error}"
                ))
            })?;
        let plan = build_verified_weight_load_plan(&lock, &cache).map_err(|error| {
            BackendErrorV1::new(format!("verified model load plan failed: {error}"))
        })?;
        let nvfp4_requested = config.fp8_provider.as_deref() == Some("nvfp4-packed-dequant");
        let nvfp4_sidecar = match (
            nvfp4_requested,
            &config.fp8_manifest_path,
            &config.fp8_artifact_path,
        ) {
            (true, Some(manifest), Some(artifact)) => Some(Arc::new(
                verify_nvfp4_sidecar(manifest, artifact, &config.lock_path, &lock).map_err(
                    |error| {
                        BackendErrorV1::new(format!("NVFP4 sidecar validation failed: {error}"))
                    },
                )?,
            )),
            (true, _, _) => unreachable!("validated NVFP4 configuration has paired paths"),
            (false, _, _) => None,
        };
        let sidecar = match (
            nvfp4_requested,
            &config.fp8_manifest_path,
            &config.fp8_artifact_path,
        ) {
            (false, Some(manifest), Some(artifact)) => Some(Arc::new(
                verify_fp8_sidecar(manifest, artifact, &config.lock_path, &lock).map_err(
                    |error| BackendErrorV1::new(format!("FP8 sidecar validation failed: {error}")),
                )?,
            )),
            (false, None, None) | (true, _, _) => None,
            _ => unreachable!("validated FP8 configuration has paired paths"),
        };
        let fp8_provider = if nvfp4_requested {
            Some("nvfp4-packed-dequant".to_owned())
        } else {
            sidecar.as_ref().map(|_| {
                config
                    .fp8_provider
                    .clone()
                    .unwrap_or_else(|| match config.target.as_str() {
                        "gfx1201" => "native".to_owned(),
                        "gfx942" => "native-fnuz".to_owned(),
                        _ => "converted-bf16".to_owned(),
                    })
            })
        };
        let seed_graph = if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
            build_qwen35_nvfp4_graph(&lock, &plan, nvfp4_sidecar, 1, 1)
        } else {
            match (&sidecar, fp8_provider.as_deref()) {
                (Some(_), Some("converted-bf16")) | (None, None) => {
                    build_qwen35_graph(&lock, &plan, 1, 1)
                }
                (Some(sidecar), Some("native-fnuz")) => {
                    build_qwen35_fp8_fnuz_graph(&lock, &plan, sidecar, 1, 1)
                }
                (Some(sidecar), Some(_)) => build_qwen35_fp8_graph(&lock, &plan, sidecar, 1, 1),
                _ => unreachable!("validated FP8 configuration has a selected provider"),
            }
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
        let resident = if let Some(nvfp4_sidecar) = &nvfp4_sidecar {
            QwenResidentModel::new_nvfp4(
                Arc::clone(&session),
                seed_graph,
                plan.clone(),
                Arc::clone(&cache),
                Arc::clone(nvfp4_sidecar),
                config.completion_timeout,
            )
        } else {
            match (&sidecar, fp8_provider.as_deref()) {
                (Some(sidecar), Some("converted-bf16")) => {
                    QwenResidentModel::new_fp8_converted_bf16(
                        Arc::clone(&session),
                        seed_graph,
                        plan.clone(),
                        Arc::clone(&cache),
                        Arc::clone(sidecar),
                        config.completion_timeout,
                    )
                }
                (Some(sidecar), Some("native-fnuz")) => QwenResidentModel::new_fp8_fnuz(
                    Arc::clone(&session),
                    seed_graph,
                    plan.clone(),
                    Arc::clone(&cache),
                    Arc::clone(sidecar),
                    config.completion_timeout,
                ),
                (Some(sidecar), Some(_)) => QwenResidentModel::new_fp8(
                    Arc::clone(&session),
                    seed_graph,
                    plan.clone(),
                    Arc::clone(&cache),
                    Arc::clone(sidecar),
                    config.completion_timeout,
                ),
                (None, None) => QwenResidentModel::new(
                    Arc::clone(&session),
                    seed_graph,
                    plan.clone(),
                    Arc::clone(&cache),
                    config.completion_timeout,
                ),
                _ => unreachable!("validated FP8 configuration has a selected provider"),
            }
        }
        .map_err(|error| BackendErrorV1::new(format!("resident model load failed: {error}")))?;
        let ready = session.memory_snapshot();
        require_clean_request_memory(ready, "model-ready")?;
        let model_ready_current_bytes = ready.model_resident().current_bytes();
        if model_ready_current_bytes == 0 || ready.current_bytes() != model_ready_current_bytes {
            return Err(BackendErrorV1::new(
                "model-ready allocation accounting is not resident-only",
            ));
        }
        let identity = BackendIdentityV1 {
            target: config.target.clone(),
            model_fingerprint: lock.fingerprint().to_owned(),
            plan_digest: plan.digest_hex(),
            model_ready_current_bytes,
        };
        Ok(Self {
            state: Mutex::new(Some(QwenBackendStateV1 {
                lock,
                tokenizer,
                renderer,
                plan,
                resident,
                session,
                target: config.target,
                model_ready_current_bytes,
                sidecar,
                nvfp4_sidecar,
                fp8_provider,
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
            resident, session, ..
        } = state;
        drop(resident);
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

impl Gemma4ChatBackendV1 {
    pub fn open(config: Gemma4BackendConfigV1) -> Result<Self, BackendErrorV1> {
        config.validate()?;
        let lock = match read_reviewed_model_lock(&config.lock_path).map_err(|error| {
            BackendErrorV1::new(format!("model lock validation failed: {error}"))
        })? {
            ReviewedModelLock::Gemma4(lock) => lock,
            ReviewedModelLock::Qwen35(_) => {
                return Err(BackendErrorV1::new(
                    "Gemma backend requires a reviewed Gemma 4 lock",
                ));
            }
        };
        let cache = lock.verify_cache(&config.cache_path).map_err(|error| {
            BackendErrorV1::new(format!("model cache verification failed: {error}"))
        })?;
        let tokenizer =
            TokenizerFrontendV1::from_gemma4_verified_cache(&lock, &cache).map_err(|error| {
                BackendErrorV1::new(format!(
                    "verified Gemma tokenizer construction failed: {error}"
                ))
            })?;
        let stop_policy = gemma4_generation_stop_policy(&lock).map_err(|error| {
            BackendErrorV1::new(format!("Gemma stop policy construction failed: {error}"))
        })?;
        let plan = build_verified_gemma4_weight_load_plan(&lock, &cache).map_err(|error| {
            BackendErrorV1::new(format!("verified Gemma load plan failed: {error}"))
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
        let resident = Gemma4ResidentModel::new(
            Arc::clone(&session),
            lock.clone(),
            plan.clone(),
            &cache,
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
            state.lock.generation_stop_policy(),
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
        let prompt_tokens = u64::try_from(prompt.len())
            .map_err(|_| BackendErrorV1::new("prompt token count overflowed u64"))?;
        let state_capacity = prompt_tokens
            .checked_add(u64::from(request.generation().max_new_tokens()))
            .ok_or_else(|| BackendErrorV1::new("request state capacity overflowed u64"))?;
        let graph = if let Some(nvfp4_sidecar) = &state.nvfp4_sidecar {
            build_qwen35_nvfp4_graph(
                &state.lock,
                &state.plan,
                nvfp4_sidecar,
                prompt_tokens,
                state_capacity,
            )
        } else {
            match (&state.sidecar, state.fp8_provider.as_deref()) {
                (Some(_), Some("converted-bf16")) | (None, None) => {
                    build_qwen35_graph(&state.lock, &state.plan, prompt_tokens, state_capacity)
                }
                (Some(sidecar), Some("native-fnuz")) => build_qwen35_fp8_fnuz_graph(
                    &state.lock,
                    &state.plan,
                    sidecar,
                    prompt_tokens,
                    state_capacity,
                ),
                (Some(sidecar), Some(_)) => build_qwen35_fp8_graph(
                    &state.lock,
                    &state.plan,
                    sidecar,
                    prompt_tokens,
                    state_capacity,
                ),
                _ => unreachable!("validated FP8 server state has a selected provider"),
            }
        }
        .map_err(|error| BackendErrorV1::new(format!("request graph failed: {error}")))?;
        let mut owner = state
            .resident
            .new_request_for_session(Arc::clone(&state.session), graph)
            .map_err(|error| {
                BackendErrorV1::new(format!("request provisioning failed: {error}"))
            })?;
        let allocated = state.session.memory_snapshot();
        let mut random = OsSamplingRandom::for_parameters(request.generation().sampling())
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
        let memory = owner.memory_audit_snapshot().ok();
        drop(owner);
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
                Some(_) => "ocp-e4m3fn-outer-f32".to_owned(),
            },
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
        let mut owner = state
            .resident
            .new_request_for_session(Arc::clone(&state.session), prompt_tokens, state_capacity)
            .map_err(|error| {
                BackendErrorV1::new(format!("request provisioning failed: {error}"))
            })?;
        let allocated = state.session.memory_snapshot();
        let mut random = OsSamplingRandom::for_parameters(request.generation().sampling())
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
            weight_encoding: "bf16".to_owned(),
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
            committed_kv_bytes: observed_length.checked_mul(GEMMA4_KV_BYTES_PER_TOKEN),
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
        ChatMessageV1 { inner }
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
