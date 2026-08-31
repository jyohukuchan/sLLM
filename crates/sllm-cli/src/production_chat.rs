//! Production adapter for the Phase 44 `chat` command.
//!
//! This module owns only CLI/runtime configuration and maps the typed
//! model-independent chat callback to a persistent Qwen, Gemma 4 MoE, or
//! direct official Ministral 3 owner. All adapters retain their verified
//! resident/session across turns; other model/runtime combinations fail before
//! model admission.

use std::io;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use sllm_core::{
    Backend, ExecutionSession, ExecutionSessionRequest, KvCacheEncoding, KvCacheSelectionRequest,
    MINISTRAL3_CONTEXT_LENGTH, MINISTRAL3_GRAPH_MAX_CONTEXT, MINISTRAL3_GRAPH_ORIGINAL_CONTEXT,
    MINISTRAL3_MODEL_ALIAS, MINISTRAL3_MODEL_LOCK_FINGERPRINT, MINISTRAL3_WEIGHT_LOCK_FINGERPRINT,
    Ministral3ModelLock, Ministral3ResidentModel, OsSamplingRandom,
    QWEN35_RECOMMENDED_CONTEXT_TOKENS, ReviewedModelLock, SamplingParametersV1,
    VerifiedMinistral3WeightSource, build_ministral3_weight_load_plan, builtin_reviewed_model_lock,
    open_and_verify_official_ministral3_gguf, parse_ministral3_model_lock, read_derived_gguf_lock,
    resolve_kv_cache_selection,
};
use sllm_frontend::{
    GenerationCancellationV1, GenerationConfigV1, GenerationInputV1, GenerationServiceV1,
    GenerationStopPolicyV1, Ministral3TextFrontendV1, ThinkingModeV1,
    ministral3_generation_stop_policy,
};
use sllm_server::{
    CheckpointStartupConfigV1, ContextWindowStartupConfigV1, DraftStartupConfigV1,
    KvCacheExplicitSourceV1, KvCacheSelectionReportV1, Phase41ProductionConfigV1,
    PrefixCacheStartupConfigV1, QwenBackendConfigV1, QwenPersistentChatFinishReasonV1,
    QwenPersistentChatSessionConfigV1, QwenPersistentChatSessionV1,
    QwenPersistentChatTurnRequestV1,
};

use crate::chat::{
    ChatBackendErrorV1, ChatBackendV1, ChatFinishReasonV1, ChatGenerationRequestV1,
    ChatGenerationResultV1, ChatThinkingModeV1,
};

const DEFAULT_COMPLETION_TIMEOUT_SECONDS_V1: u64 = 120;
const DEFAULT_SHUTDOWN_TIMEOUT_SECONDS_V1: u64 = 30;
const MAX_CONTEXT_LENGTH_V1: u32 = 1_048_576;

/// The synchronous backend callback cannot carry a cancellation argument in
/// the model-independent CLI trait.  Keep exactly one in-flight token in a
/// small shared registry so the SIGINT listener can cancel the current turn
/// without installing an unsafe process-global handler.
#[derive(Clone, Debug, Default)]
struct CancellationRegistryV1 {
    active: Arc<Mutex<Option<GenerationCancellationV1>>>,
}

impl CancellationRegistryV1 {
    fn register(&self) -> GenerationCancellationV1 {
        let cancellation = GenerationCancellationV1::new();
        *self.active.lock().expect("cancellation registry poisoned") = Some(cancellation.clone());
        cancellation
    }

    fn clear(&self) {
        *self.active.lock().expect("cancellation registry poisoned") = None;
    }

    fn cancel_current(&self) {
        if let Some(cancellation) = self
            .active
            .lock()
            .expect("cancellation registry poisoned")
            .as_ref()
        {
            cancellation.cancel();
        }
    }
}

/// Owns a tokio signal listener for the lifetime of one production `chat`
/// invocation.  A current-thread runtime is isolated to this listener thread;
/// model execution remains synchronous on the CLI thread.
struct SigintListenerV1 {
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl SigintListenerV1 {
    fn start(registry: CancellationRegistryV1) -> Result<Self, String> {
        #[cfg(not(unix))]
        {
            let _ = registry;
            return Err("chat SIGINT listener is unsupported on this target".to_owned());
        }

        #[cfg(unix)]
        {
            let (shutdown, mut shutdown_receiver) = tokio::sync::oneshot::channel();
            let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);
            let thread = thread::Builder::new()
                .name("sllm-chat-sigint".to_owned())
                .spawn(move || {
                    let runtime = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(runtime) => runtime,
                        Err(_) => {
                            let _ = ready_sender.send(false);
                            return;
                        }
                    };
                    let interrupt = {
                        let _guard = runtime.enter();
                        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                    };
                    let mut interrupt = match interrupt {
                        Ok(interrupt) => interrupt,
                        Err(_) => {
                            let _ = ready_sender.send(false);
                            return;
                        }
                    };
                    let _ = ready_sender.send(true);
                    runtime.block_on(async move {
                        loop {
                            tokio::select! {
                                signal = interrupt.recv() => {
                                    if signal.is_none() {
                                        break;
                                    }
                                    registry.cancel_current();
                                }
                                _ = &mut shutdown_receiver => break,
                            }
                        }
                    });
                })
                .map_err(|_| "chat SIGINT listener failed to start".to_owned())?;
            if ready_receiver.recv().ok() != Some(true) {
                let _ = thread.join();
                return Err("chat SIGINT listener failed to initialize".to_owned());
            }
            Ok(Self {
                shutdown: Some(shutdown),
                thread: Some(thread),
            })
        }
    }

    fn shutdown(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProductionChatConfigV1 {
    gguf: std::path::PathBuf,
    derived_lock: Option<std::path::PathBuf>,
    device_index: u32,
    target: String,
    context_length: u32,
    kv_cache_encoding: Option<KvCacheEncoding>,
    completion_timeout_seconds: u64,
    shutdown_timeout_seconds: u64,
    checkpoint_directory: std::path::PathBuf,
    checkpoint_quota_bytes: u64,
}

fn parse_u64(value: &str, option: &str, min: u64, max: u64) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("{option} value is invalid"))?;
    if !(min..=max).contains(&parsed) {
        return Err(format!("{option} value is out of range"));
    }
    Ok(parsed)
}

fn parse_u32(value: &str, option: &str, min: u32, max: u32) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("{option} value is invalid"))?;
    if !(min..=max).contains(&parsed) {
        return Err(format!("{option} value is out of range"));
    }
    Ok(parsed)
}

fn validate_path_argument(path: &std::path::Path, option: &str) -> Result<(), String> {
    if path.as_os_str().is_empty() || path.to_string_lossy().contains('\0') {
        return Err(format!("{option} value is invalid"));
    }
    Ok(())
}

fn is_chat_flag(flag: &str) -> bool {
    matches!(
        flag,
        "--help"
            | "-h"
            | "--prompt"
            | "--message"
            | "--prompt-file"
            | "--system"
            | "--reverse-prompt"
            | "--stop"
            | "--max-new-tokens"
            | "--thinking"
            | "--reasoning-budget"
            | "--checkpoint-load"
            | "--checkpoint-save"
    )
}

fn is_production_flag(flag: &str) -> bool {
    matches!(
        flag,
        "--gguf"
            | "--derived-lock"
            | "--device-index"
            | "--target"
            | "--context-length"
            | "--kv-cache-encoding"
            | "--completion-timeout-seconds"
            | "--shutdown-timeout-seconds"
            | "--checkpoint-directory"
            | "--checkpoint-quota-bytes"
    )
}

fn split_args(
    arguments: impl IntoIterator<Item = String>,
) -> Result<(ProductionChatConfigV1, Vec<String>), String> {
    let raw = arguments.into_iter().collect::<Vec<_>>();
    if raw
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        return Err("help".to_owned());
    }
    let mut gguf = None;
    let mut derived_lock = None;
    let mut device_index = None;
    let mut target = None;
    let mut context_length = None;
    let mut kv_cache_encoding = None;
    let mut completion_timeout_seconds = None;
    let mut shutdown_timeout_seconds = None;
    let mut checkpoint_directory = None;
    let mut checkpoint_quota_bytes = None;
    let mut chat_args = Vec::new();
    let mut index = 0;
    while index < raw.len() {
        let flag = &raw[index];
        if is_production_flag(flag) {
            let value = raw
                .get(index + 1)
                .filter(|value| !value.starts_with('-'))
                .ok_or_else(|| "chat model/runtime option requires a value".to_owned())?;
            match flag.as_str() {
                "--gguf" if gguf.is_none() => gguf = Some(std::path::PathBuf::from(value)),
                "--derived-lock" if derived_lock.is_none() => {
                    derived_lock = Some(std::path::PathBuf::from(value))
                }
                "--device-index" if device_index.is_none() => {
                    device_index = Some(parse_u32(value, flag, 0, u32::MAX)?)
                }
                "--target" if target.is_none() => target = Some(value.clone()),
                "--context-length" if context_length.is_none() => {
                    context_length = Some(parse_u32(value, flag, 1, MAX_CONTEXT_LENGTH_V1)?)
                }
                "--kv-cache-encoding" if kv_cache_encoding.is_none() => {
                    kv_cache_encoding = Some(value.clone())
                }
                "--completion-timeout-seconds" if completion_timeout_seconds.is_none() => {
                    completion_timeout_seconds = Some(parse_u64(value, flag, 1, 86_400)?)
                }
                "--shutdown-timeout-seconds" if shutdown_timeout_seconds.is_none() => {
                    shutdown_timeout_seconds = Some(parse_u64(value, flag, 1, 3_600)?)
                }
                "--checkpoint-directory" if checkpoint_directory.is_none() => {
                    checkpoint_directory = Some(std::path::PathBuf::from(value))
                }
                "--checkpoint-quota-bytes" if checkpoint_quota_bytes.is_none() => {
                    checkpoint_quota_bytes = Some(parse_u64(value, flag, 1, u64::MAX)?)
                }
                _ => return Err("duplicate chat model/runtime option".to_owned()),
            }
            index += 2;
        } else if is_chat_flag(flag) {
            chat_args.push(flag.clone());
            if flag != "--help" && flag != "-h" {
                let value = raw
                    .get(index + 1)
                    .ok_or_else(|| "chat option requires a value".to_owned())?;
                chat_args.push(value.clone());
                index += 2;
            } else {
                index += 1;
            }
        } else {
            return Err("unknown chat model/runtime option".to_owned());
        }
    }
    let gguf = gguf.ok_or_else(|| "chat requires --gguf".to_owned())?;
    let device_index = device_index.ok_or_else(|| "chat requires --device-index".to_owned())?;
    let target = target.ok_or_else(|| "chat requires --target".to_owned())?;
    if !matches!(target.as_str(), "gfx1030" | "gfx1201" | "gfx942") {
        return Err("chat target is unsupported".to_owned());
    }
    if target.is_empty() || !target.is_ascii() {
        return Err("chat target is invalid".to_owned());
    }
    let kv_cache_encoding = match kv_cache_encoding.as_deref() {
        None => None,
        Some("fp16") => Some(KvCacheEncoding::Fp16),
        Some("fp8-static") => Some(KvCacheEncoding::Fp8E4M3FnStatic),
        Some("kv-mxfp8-e4") => Some(KvCacheEncoding::Mxfp8E4),
        Some(_) => {
            return Err(
                "chat KV cache encoding must be fp16, fp8-static, or kv-mxfp8-e4".to_owned(),
            );
        }
    };
    let checkpoint_directory =
        checkpoint_directory.ok_or_else(|| "chat requires --checkpoint-directory".to_owned())?;
    validate_path_argument(&gguf, "--gguf")?;
    if let Some(derived_lock) = derived_lock.as_deref() {
        validate_path_argument(derived_lock, "--derived-lock")?;
    }
    validate_path_argument(&checkpoint_directory, "--checkpoint-directory")?;
    let checkpoint_quota_bytes = checkpoint_quota_bytes
        .ok_or_else(|| "chat requires --checkpoint-quota-bytes".to_owned())?;
    let default_context_length = if derived_lock.is_none() {
        MINISTRAL3_GRAPH_ORIGINAL_CONTEXT as u32
    } else {
        u32::try_from(QWEN35_RECOMMENDED_CONTEXT_TOKENS).expect("Qwen recommended context fits u32")
    };
    Ok((
        ProductionChatConfigV1 {
            gguf,
            derived_lock,
            device_index,
            target,
            context_length: context_length.unwrap_or(default_context_length),
            kv_cache_encoding,
            completion_timeout_seconds: completion_timeout_seconds
                .unwrap_or(DEFAULT_COMPLETION_TIMEOUT_SECONDS_V1),
            shutdown_timeout_seconds: shutdown_timeout_seconds
                .unwrap_or(DEFAULT_SHUTDOWN_TIMEOUT_SECONDS_V1),
            checkpoint_directory,
            checkpoint_quota_bytes,
        },
        chat_args,
    ))
}

fn phase41_disabled() -> Phase41ProductionConfigV1 {
    Phase41ProductionConfigV1 {
        prefix_cache: PrefixCacheStartupConfigV1::Disabled,
        context_window: ContextWindowStartupConfigV1::Disabled,
        checkpoint: CheckpointStartupConfigV1::Disabled,
        draft: DraftStartupConfigV1::Disabled,
    }
}

fn validate_direct_chat_options(
    config: &ProductionChatConfigV1,
    chat_args: &[String],
) -> Result<(), String> {
    if config.derived_lock.is_some() {
        return Ok(());
    }
    let mut index = 0;
    while index < chat_args.len() {
        match chat_args[index].as_str() {
            "--thinking" => {
                if chat_args.get(index + 1).map(String::as_str) == Some("enabled") {
                    return Err("Ministral 3 does not support thinking/reasoning".to_owned());
                }
                index += 2;
            }
            "--reasoning-budget" => {
                return Err("Ministral 3 does not support thinking/reasoning".to_owned());
            }
            "--checkpoint-load" | "--checkpoint-save" => {
                // These are intentionally passed through to the backend so
                // the public chat contract reports CheckpointUnavailable.
                index += 2;
            }
            _ => index += 1,
        }
    }
    Ok(())
}

/// The direct official Ministral 3 chat owner deliberately keeps the model
/// resident and HIP session alive for the whole CLI invocation. Each turn is
/// still request-local: the complete rendered transcript is submitted to a
/// fresh `Ministral3ExecutionRequest`, so no mutable conversation/KV state is
/// carried across turns.
struct Ministral3CliChatBackend {
    _lock: Ministral3ModelLock,
    frontend: Ministral3TextFrontendV1,
    stop_policy: GenerationStopPolicyV1,
    resident: Option<Ministral3ResidentModel>,
    session: Option<Arc<ExecutionSession>>,
    shutdown_timeout: Duration,
    context_length: u32,
    target: String,
}

impl Ministral3CliChatBackend {
    fn open(config: &ProductionChatConfigV1) -> Result<Self, String> {
        if config.context_length > MINISTRAL3_CONTEXT_LENGTH
            || config.context_length > MINISTRAL3_GRAPH_MAX_CONTEXT as u32
        {
            return Err(format!(
                "Ministral 3 context length must be in [1,{MINISTRAL3_CONTEXT_LENGTH}]"
            ));
        }
        if config
            .kv_cache_encoding
            .is_some_and(|encoding| encoding != KvCacheEncoding::Fp16)
        {
            return Err("Ministral 3 supports only its fixed FP16 KV cache".to_owned());
        }
        let lock = parse_ministral3_model_lock(include_bytes!(
            "../../../docs/models/locks/ministral3-3b-instruct-2512-official-bf16-gguf.json"
        ))
        .map_err(|error| format!("reviewed Ministral 3 model lock is invalid: {error}"))?;
        if lock.fingerprint() != MINISTRAL3_MODEL_LOCK_FINGERPRINT
            || lock.aliases() != [MINISTRAL3_MODEL_ALIAS.to_owned()]
        {
            return Err("reviewed Ministral 3 model lock is not canonical".to_owned());
        }
        let verified = open_and_verify_official_ministral3_gguf(&config.gguf)
            .map_err(|error| format!("official Ministral 3 GGUF verification failed: {error}"))?;
        if verified.expected_lfs_sha256() != lock.file_sha256()
            || verified.repository() != lock.repository()
            || verified.revision() != lock.revision()
        {
            return Err(
                "official Ministral 3 GGUF identity differs from the reviewed model lock"
                    .to_owned(),
            );
        }
        let source = Arc::new(
            VerifiedMinistral3WeightSource::from_verified_gguf(verified).map_err(|error| {
                format!("Ministral 3 weight source verification failed: {error}")
            })?,
        );
        if source.lock_fingerprint() != MINISTRAL3_WEIGHT_LOCK_FINGERPRINT
            || source.repository() != lock.repository()
            || source.revision() != lock.revision()
            || source.file_sha256() != lock.file_sha256()
        {
            return Err("Ministral 3 weight source identity is not canonical".to_owned());
        }
        let plan = build_ministral3_weight_load_plan(source.as_ref())
            .map_err(|error| format!("Ministral 3 weight load plan failed: {error}"))?;
        let frontend = Ministral3TextFrontendV1::from_verified_gguf(source.verified())
            .map_err(|error| format!("Ministral 3 frontend construction failed: {error}"))?;
        let stop_policy = ministral3_generation_stop_policy()
            .map_err(|error| format!("Ministral 3 stop policy failed: {error}"))?;
        let backend = sllm_hip::HipBackend::connect()
            .map_err(|error| format!("HIP backend is unavailable: {error}"))?;
        let session_request = ExecutionSessionRequest::new(config.device_index, &config.target)
            .map_err(|error| format!("HIP session request failed: {error}"))?;
        let session = backend
            .open_execution_session(session_request)
            .map_err(|error| format!("exact HIP execution session failed: {error}"))?;
        let resident = Ministral3ResidentModel::new_gguf(
            Arc::clone(&session),
            plan,
            source,
            Duration::from_secs(config.completion_timeout_seconds),
        )
        .map_err(|error| format!("Ministral 3 resident load failed: {error}"))?;
        Ok(Self {
            _lock: lock,
            frontend,
            stop_policy,
            resident: Some(resident),
            session: Some(session),
            shutdown_timeout: Duration::from_secs(config.shutdown_timeout_seconds),
            context_length: config.context_length,
            target: config.target.clone(),
        })
    }

    fn generate_with_cancellation(
        &mut self,
        request: &ChatGenerationRequestV1,
        cancellation: &GenerationCancellationV1,
    ) -> Result<ChatGenerationResultV1, ChatBackendErrorV1> {
        if request.thinking == ChatThinkingModeV1::Enabled || request.reasoning_budget.is_some() {
            return Err(ChatBackendErrorV1::Failed);
        }
        let mut stop_sequences = request.stop_sequences.clone();
        stop_sequences.extend(request.reverse_prompts.iter().cloned());
        let rendered = self
            .frontend
            .renderer()
            .render(&request.messages)
            .map_err(|_| ChatBackendErrorV1::Failed)?;
        let service = GenerationServiceV1::new(&self.frontend, None, &self.stop_policy)
            .map_err(|_| ChatBackendErrorV1::Failed)?;
        let prompt = service
            .prepare_input(&GenerationInputV1::Prompt(rendered))
            .map_err(|_| ChatBackendErrorV1::Failed)?;
        let prompt_tokens = u64::try_from(prompt.len()).map_err(|_| ChatBackendErrorV1::Failed)?;
        let state_capacity = prompt_tokens
            .checked_add(u64::from(request.max_new_tokens))
            .ok_or(ChatBackendErrorV1::Failed)?;
        if prompt.is_empty() || state_capacity > u64::from(self.context_length) {
            return Err(ChatBackendErrorV1::Failed);
        }
        let resident = self.resident.as_ref().ok_or(ChatBackendErrorV1::Failed)?;
        let mut owner = resident
            .new_request(prompt_tokens, state_capacity)
            .map_err(|_| ChatBackendErrorV1::Failed)?;
        let config = GenerationConfigV1::new(
            request.max_new_tokens,
            SamplingParametersV1::greedy(),
            stop_sequences,
        )
        .map_err(|_| ChatBackendErrorV1::Failed)?;
        let mut random =
            OsSamplingRandom::for_parameters_and_seed(SamplingParametersV1::greedy(), None)
                .map_err(|_| ChatBackendErrorV1::Failed)?;
        let result =
            service.generate_tokens(&mut owner, &prompt, &config, cancellation, &mut random);
        if cancellation.is_cancelled() {
            return Err(ChatBackendErrorV1::Cancelled);
        }
        let result = result.map_err(|_| ChatBackendErrorV1::Failed)?;
        let audit = owner.last_audit().ok_or(ChatBackendErrorV1::Failed)?;
        if audit.target() != self.target || audit.fallback_used() {
            return Err(ChatBackendErrorV1::Failed);
        }
        let finish_reason = if result.matched_stop().is_some_and(|stop| {
            request
                .reverse_prompts
                .iter()
                .any(|reverse| reverse == stop)
        }) {
            ChatFinishReasonV1::ReversePrompt
        } else {
            match result.finish_reason() {
                sllm_frontend::FinishReasonV1::Stop => ChatFinishReasonV1::Stop,
                sllm_frontend::FinishReasonV1::Length => ChatFinishReasonV1::Length,
            }
        };
        Ok(ChatGenerationResultV1 {
            text: result.output_text().to_owned(),
            reasoning: None,
            finish_reason,
            cancelled: false,
        })
    }
}

impl Drop for Ministral3CliChatBackend {
    fn drop(&mut self) {
        // Resident buffers must be released before asking the session to drain
        // and close; otherwise shutdown would correctly reject live bytes.
        let _ = self.resident.take();
        if let Some(session) = self.session.take() {
            let _ = session.shutdown(self.shutdown_timeout);
        }
    }
}

fn open_backend(
    config: ProductionChatConfigV1,
    cancellation_registry: CancellationRegistryV1,
) -> Result<ProductionChatBackendV1, String> {
    let Some(derived_lock_path) = config.derived_lock.as_ref() else {
        let backend = Ministral3CliChatBackend::open(&config)?;
        return Ok(ProductionChatBackendV1 {
            session: None,
            gemma_moe: None,
            ministral3: Some(backend),
            cancellation_registry,
        });
    };
    let derived = read_derived_gguf_lock(derived_lock_path)
        .map_err(|_| "chat derived GGUF lock is invalid".to_owned())?;
    if derived.semantic_model_id.starts_with("gemma4moe:") {
        let backend = crate::model::Gemma4MoeCliChatBackend::open(
            config.gguf,
            derived_lock_path.clone(),
            config.device_index,
            config.target,
            config.context_length,
            config.kv_cache_encoding,
            Duration::from_secs(config.completion_timeout_seconds),
            Duration::from_secs(config.shutdown_timeout_seconds),
            config.checkpoint_directory,
            config.checkpoint_quota_bytes,
        )?;
        return Ok(ProductionChatBackendV1 {
            session: None,
            gemma_moe: Some(backend),
            ministral3: None,
            cancellation_registry,
        });
    }
    let lock = match builtin_reviewed_model_lock(&derived.source_lock_fingerprints)
        .map_err(|_| "chat model lock is unsupported".to_owned())?
    {
        ReviewedModelLock::Qwen35(lock) => lock,
        ReviewedModelLock::Gemma4(_) => {
            return Err("chat requires a reviewed Qwen model".to_owned());
        }
        ReviewedModelLock::Ministral3(_) => {
            return Err("derived GGUF cannot select the direct Ministral 3 backend".to_owned());
        }
    };
    let head_dim = usize::try_from(lock.model().architecture.text_config.head_dim)
        .map_err(|_| "chat KV head dimension is invalid".to_owned())?;
    let kv_selection = resolve_kv_cache_selection(KvCacheSelectionRequest::new(
        config.kv_cache_encoding,
        &config.target,
        lock.fingerprint(),
        true,
        true,
        true,
        head_dim,
    ))
    .map_err(|error| error.to_string())?;
    let kv_selection_report =
        KvCacheSelectionReportV1::from_core(kv_selection, KvCacheExplicitSourceV1::Process);
    let backend = QwenBackendConfigV1 {
        gguf_path: config.gguf,
        derived_lock_path: derived_lock_path.clone(),
        device_index: config.device_index,
        target: config.target,
        completion_timeout: Duration::from_secs(config.completion_timeout_seconds),
        shutdown_timeout: Duration::from_secs(config.shutdown_timeout_seconds),
        context_length: config.context_length,
        kv_cache_encoding: kv_selection.resolved(),
        kv_cache_resolved_selection: Some(kv_selection),
        kv_cache_selection: Some(kv_selection_report),
        phase41: phase41_disabled(),
        adapter_catalog: None,
    };
    let session = QwenPersistentChatSessionV1::open(QwenPersistentChatSessionConfigV1 {
        backend,
        checkpoint_directory: config.checkpoint_directory,
        checkpoint_quota_bytes: config.checkpoint_quota_bytes,
    })
    .map_err(|_| "chat model/runtime failed to open".to_owned())?;
    Ok(ProductionChatBackendV1 {
        session: Some(session),
        gemma_moe: None,
        ministral3: None,
        cancellation_registry,
    })
}

pub(crate) struct ProductionChatBackendV1 {
    session: Option<QwenPersistentChatSessionV1>,
    gemma_moe: Option<crate::model::Gemma4MoeCliChatBackend>,
    ministral3: Option<Ministral3CliChatBackend>,
    cancellation_registry: CancellationRegistryV1,
}

impl Drop for ProductionChatBackendV1 {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            let _ = session.shutdown();
        }
        // The Gemma 4 MoE adapter owns its persistent HIP session and closes
        // it from its Drop implementation.
        let _ = self.gemma_moe.take();
        let _ = self.ministral3.take();
    }
}

impl ChatBackendV1 for ProductionChatBackendV1 {
    fn generate(
        &mut self,
        request: &ChatGenerationRequestV1,
    ) -> Result<ChatGenerationResultV1, ChatBackendErrorV1> {
        if let Some(backend) = self.gemma_moe.as_mut() {
            let cancellation = self.cancellation_registry.register();
            backend.set_cancellation(cancellation.clone());
            let result = backend.generate(request);
            let was_cancelled = cancellation.is_cancelled();
            self.cancellation_registry.clear();
            return match result {
                Ok(_result) if was_cancelled => Err(ChatBackendErrorV1::Cancelled),
                Ok(result) => Ok(result),
                Err(ChatBackendErrorV1::Failed) if was_cancelled => {
                    Err(ChatBackendErrorV1::Cancelled)
                }
                Err(error) => Err(error),
            };
        }
        if let Some(backend) = self.ministral3.as_mut() {
            let cancellation = self.cancellation_registry.register();
            let result = backend.generate_with_cancellation(request, &cancellation);
            let was_cancelled = cancellation.is_cancelled();
            self.cancellation_registry.clear();
            return match result {
                Ok(_result) if was_cancelled => Err(ChatBackendErrorV1::Cancelled),
                Ok(result) => Ok(result),
                Err(ChatBackendErrorV1::Failed) if was_cancelled => {
                    Err(ChatBackendErrorV1::Cancelled)
                }
                Err(error) => Err(error),
            };
        }
        let session = self.session.as_mut().ok_or(ChatBackendErrorV1::Failed)?;
        let cancellation = self.cancellation_registry.register();
        let thinking = match request.thinking {
            ChatThinkingModeV1::Default => ThinkingModeV1::TemplateDefault,
            ChatThinkingModeV1::Enabled => ThinkingModeV1::Enabled,
            ChatThinkingModeV1::Disabled => ThinkingModeV1::Disabled,
        };
        let result = session.turn_with_cancellation(
            QwenPersistentChatTurnRequestV1 {
                messages: request.messages.clone(),
                max_new_tokens: request.max_new_tokens,
                stop_sequences: request.stop_sequences.clone(),
                reverse_prompts: request.reverse_prompts.clone(),
                thinking,
                reasoning_budget: request.reasoning_budget,
            },
            &cancellation,
        );
        let was_cancelled = cancellation.is_cancelled();
        self.cancellation_registry.clear();
        let result = match result {
            Ok(_) if was_cancelled => return Err(ChatBackendErrorV1::Cancelled),
            Ok(result) => result,
            Err(_) if was_cancelled => return Err(ChatBackendErrorV1::Cancelled),
            Err(_) => return Err(ChatBackendErrorV1::Failed),
        };
        let finish_reason = match result.finish_reason {
            QwenPersistentChatFinishReasonV1::Stop => ChatFinishReasonV1::Stop,
            QwenPersistentChatFinishReasonV1::ReversePrompt => ChatFinishReasonV1::ReversePrompt,
            QwenPersistentChatFinishReasonV1::Length => ChatFinishReasonV1::Length,
        };
        Ok(ChatGenerationResultV1 {
            text: result.text,
            reasoning: result.reasoning,
            finish_reason,
            cancelled: false,
        })
    }

    fn load_checkpoint(&mut self, name: &str) -> Result<Option<Vec<u8>>, ChatBackendErrorV1> {
        if let Some(backend) = self.gemma_moe.as_mut() {
            return backend.load_checkpoint(name);
        }
        if self.ministral3.is_some() {
            let _ = name;
            return Err(ChatBackendErrorV1::CheckpointUnavailable);
        }
        let session = self.session.as_mut().ok_or(ChatBackendErrorV1::Failed)?;
        session
            .load_checkpoint(name)
            .map(Some)
            .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)
    }

    fn save_checkpoint(
        &mut self,
        name: &str,
        conversation: &[u8],
    ) -> Result<(), ChatBackendErrorV1> {
        if let Some(backend) = self.gemma_moe.as_mut() {
            return backend.save_checkpoint(name, conversation);
        }
        if self.ministral3.is_some() {
            let _ = (name, conversation);
            return Err(ChatBackendErrorV1::CheckpointUnavailable);
        }
        let session = self.session.as_mut().ok_or(ChatBackendErrorV1::Failed)?;
        session
            .save_checkpoint(name, conversation)
            .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)
    }

    fn commit_turn(&mut self, conversation: &[u8]) -> Result<(), ChatBackendErrorV1> {
        if let Some(backend) = self.gemma_moe.as_mut() {
            return backend.commit_turn(conversation);
        }
        if self.ministral3.is_some() {
            let _ = conversation;
            return Ok(());
        }
        let session = self.session.as_mut().ok_or(ChatBackendErrorV1::Failed)?;
        session
            .commit_turn(conversation)
            .map_err(|_| ChatBackendErrorV1::Failed)
    }

    fn abort_turn(&mut self) -> Result<(), ChatBackendErrorV1> {
        if let Some(backend) = self.gemma_moe.as_mut() {
            return backend.abort_turn();
        }
        if self.ministral3.is_some() {
            return Ok(());
        }
        let session = self.session.as_mut().ok_or(ChatBackendErrorV1::Failed)?;
        session
            .discard_pending_turn()
            .map_err(|_| ChatBackendErrorV1::Failed)
    }
}

pub(crate) fn run<I>(arguments: I) -> Result<(), String>
where
    I: Iterator<Item = String>,
{
    let (config, chat_args) = match split_args(arguments) {
        Ok(value) => value,
        Err(error) if error == "help" => {
            return crate::chat::run(["--help".to_owned()].into_iter());
        }
        Err(error) => return Err(error),
    };
    validate_direct_chat_options(&config, &chat_args)?;
    let options = match crate::chat::preflight(chat_args.into_iter())? {
        Some(options) => options,
        None => return crate::chat::run(["--help".to_owned()].into_iter()),
    };
    let cancellation_registry = CancellationRegistryV1::default();
    let mut backend = open_backend(config, cancellation_registry.clone())?;
    let listener = SigintListenerV1::start(cancellation_registry)?;
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut output = io::stdout();
    let result = crate::chat::run_prepared(options, &mut backend, &mut output, &mut input);
    listener.shutdown();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_registry_only_targets_current_turn() {
        let registry = CancellationRegistryV1::default();
        let cancellation = registry.register();
        assert!(!cancellation.is_cancelled());
        registry.cancel_current();
        assert!(cancellation.is_cancelled());
        registry.clear();
        let next = registry.register();
        assert!(!next.is_cancelled());
        registry.clear();
        assert!(!next.is_cancelled());
    }

    #[test]
    fn parser_defaults_qwen_kv_and_accepts_explicit_mxfp8_or_fp16() {
        let error = split_args(["--gguf", "m"].into_iter().map(str::to_owned)).unwrap_err();
        assert!(error.contains("--device-index"));
        let mut args = vec![
            "--gguf",
            "m",
            "--derived-lock",
            "l",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--checkpoint-directory",
            "c",
            "--checkpoint-quota-bytes",
            "1024",
            "--kv-cache-encoding",
            "fp8",
        ];
        let error = split_args(args.drain(..).map(str::to_owned)).unwrap_err();
        assert!(error.contains("fp16"));

        let base = [
            "--gguf",
            "m",
            "--derived-lock",
            "l",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--checkpoint-directory",
            "c",
            "--checkpoint-quota-bytes",
            "1024",
        ];
        let (auto, _) = split_args(base.into_iter().map(str::to_owned)).unwrap();
        assert_eq!(auto.kv_cache_encoding, None);
        let mut explicit_args = base.to_vec();
        explicit_args.extend(["--kv-cache-encoding", "fp16"]);
        let (explicit, _) = split_args(explicit_args.into_iter().map(str::to_owned)).unwrap();
        assert_eq!(explicit.kv_cache_encoding, Some(KvCacheEncoding::Fp16));
        let mut explicit_args = base.to_vec();
        explicit_args.extend(["--kv-cache-encoding", "kv-mxfp8-e4"]);
        let (explicit, _) = split_args(explicit_args.into_iter().map(str::to_owned)).unwrap();
        assert_eq!(explicit.kv_cache_encoding, Some(KvCacheEncoding::Mxfp8E4));
        let mut explicit_args = base.to_vec();
        explicit_args.extend(["--kv-cache-encoding", "fp8-static"]);
        let (explicit, _) = split_args(explicit_args.into_iter().map(str::to_owned)).unwrap();
        assert_eq!(
            explicit.kv_cache_encoding,
            Some(KvCacheEncoding::Fp8E4M3FnStatic)
        );

        let args = [
            "--gguf",
            "bad\0path",
            "--derived-lock",
            "l",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--checkpoint-directory",
            "c",
            "--checkpoint-quota-bytes",
            "1024",
        ];
        assert!(split_args(args.into_iter().map(str::to_owned)).is_err());
    }

    #[test]
    fn parser_omitted_derived_lock_selects_direct_ministral_defaults() {
        let args = [
            "--gguf",
            "official.gguf",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
            "--checkpoint-directory",
            "unused",
            "--checkpoint-quota-bytes",
            "1024",
        ];
        let (config, _) = split_args(args.into_iter().map(str::to_owned)).unwrap();
        assert_eq!(config.derived_lock, None);
        assert_eq!(
            config.context_length,
            MINISTRAL3_GRAPH_ORIGINAL_CONTEXT as u32
        );
    }

    #[test]
    fn parser_explicit_derived_lock_retains_qwen_default_context() {
        let args = [
            "--gguf",
            "derived.gguf",
            "--derived-lock",
            "derived.lock",
            "--device-index",
            "0",
            "--target",
            "gfx1030",
            "--checkpoint-directory",
            "unused",
            "--checkpoint-quota-bytes",
            "1024",
        ];
        let (config, _) = split_args(args.into_iter().map(str::to_owned)).unwrap();
        assert_eq!(config.derived_lock, Some("derived.lock".into()));
        assert_eq!(
            config.context_length,
            u32::try_from(QWEN35_RECOMMENDED_CONTEXT_TOKENS).unwrap()
        );
    }

    #[test]
    fn direct_ministral_rejects_context_above_official_boundary_before_open() {
        let context_length = (MINISTRAL3_CONTEXT_LENGTH + 1).to_string();
        let args = [
            "--gguf",
            "missing.gguf",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
            "--context-length",
            context_length.as_str(),
            "--checkpoint-directory",
            "unused",
            "--checkpoint-quota-bytes",
            "1024",
        ];
        let (config, _) = split_args(args.into_iter().map(str::to_owned)).unwrap();
        let error = match Ministral3CliChatBackend::open(&config) {
            Ok(_) => panic!("context boundary should reject before opening a model"),
            Err(error) => error,
        };
        assert!(error.contains("context length"));
    }

    #[test]
    fn direct_ministral_rejects_non_fp16_kv_before_open() {
        let args = [
            "--gguf",
            "missing.gguf",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
            "--kv-cache-encoding",
            "kv-mxfp8-e4",
            "--checkpoint-directory",
            "unused",
            "--checkpoint-quota-bytes",
            "1024",
        ];
        let (config, _) = split_args(args.into_iter().map(str::to_owned)).unwrap();
        let error = match Ministral3CliChatBackend::open(&config) {
            Ok(_) => panic!("non-FP16 KV should reject before opening a model"),
            Err(error) => error,
        };
        assert!(error.contains("FP16 KV"));
    }

    #[test]
    fn direct_ministral_rejects_reasoning_before_model_open() {
        let model_args = [
            "--gguf",
            "missing.gguf",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
            "--checkpoint-directory",
            "unused",
            "--checkpoint-quota-bytes",
            "1024",
        ];
        let (config, _) = split_args(model_args.into_iter().map(str::to_owned)).unwrap();
        let error = validate_direct_chat_options(
            &config,
            &[
                "--thinking".to_owned(),
                "enabled".to_owned(),
                "--prompt".to_owned(),
                "question".to_owned(),
            ],
        )
        .unwrap_err();
        assert!(error.contains("thinking/reasoning"));
    }

    #[test]
    fn parser_rejects_phase41_enablement_by_not_accepting_its_flags() {
        let args = [
            "--gguf",
            "m",
            "--derived-lock",
            "l",
            "--device-index",
            "0",
            "--target",
            "gfx1201",
            "--checkpoint-directory",
            "c",
            "--checkpoint-quota-bytes",
            "1024",
            "--prefix-cache",
            "enabled",
        ];
        assert!(split_args(args.into_iter().map(str::to_owned)).is_err());
    }

    #[test]
    fn chat_preflight_rejects_invalid_source_before_backend_open() {
        let args = [
            "--gguf",
            "m",
            "--derived-lock",
            "l",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--checkpoint-directory",
            "c",
            "--checkpoint-quota-bytes",
            "1024",
            "--prompt-file",
            "/definitely/missing/phase44-prompt",
        ];
        let (_config, chat_args) = split_args(args.into_iter().map(str::to_owned)).unwrap();
        assert!(crate::chat::preflight(chat_args.into_iter()).is_err());
    }

    #[test]
    fn chat_preflight_requires_enabled_thinking_for_reasoning_budget() {
        let args = [
            "--gguf",
            "m",
            "--derived-lock",
            "l",
            "--device-index",
            "0",
            "--target",
            "gfx942",
            "--checkpoint-directory",
            "c",
            "--checkpoint-quota-bytes",
            "1024",
            "--prompt",
            "question",
            "--reasoning-budget",
            "16",
        ];
        let (_config, chat_args) = split_args(args.into_iter().map(str::to_owned)).unwrap();
        assert!(crate::chat::preflight(chat_args.into_iter()).is_err());
    }
}
