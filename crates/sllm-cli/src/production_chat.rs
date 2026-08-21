//! Production adapter for the Phase 44 `chat` command.
//!
//! This module owns only CLI/runtime configuration and maps the typed
//! model-independent chat callback to the server's persistent Qwen owner.  It
//! intentionally admits dense BF16 text with all Phase 41 startup facilities
//! disabled; other model/runtime combinations fail before model admission.

use std::io;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use sllm_core::{KvCacheEncoding, QWEN35_RECOMMENDED_CONTEXT_TOKENS};
use sllm_frontend::{GenerationCancellationV1, ThinkingModeV1};
use sllm_server::{
    CheckpointStartupConfigV1, ContextWindowStartupConfigV1, DraftStartupConfigV1,
    Phase41ProductionConfigV1, PrefixCacheStartupConfigV1, QwenBackendConfigV1,
    QwenPersistentChatFinishReasonV1, QwenPersistentChatSessionConfigV1,
    QwenPersistentChatSessionV1, QwenPersistentChatTurnRequestV1,
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
    derived_lock: std::path::PathBuf,
    device_index: u32,
    target: String,
    context_length: u32,
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
    let derived_lock = derived_lock.ok_or_else(|| "chat requires --derived-lock".to_owned())?;
    let device_index = device_index.ok_or_else(|| "chat requires --device-index".to_owned())?;
    let target = target.ok_or_else(|| "chat requires --target".to_owned())?;
    if !matches!(target.as_str(), "gfx1030" | "gfx1201" | "gfx942") {
        return Err("chat target is unsupported".to_owned());
    }
    if target.is_empty() || !target.is_ascii() {
        return Err("chat target is invalid".to_owned());
    }
    if kv_cache_encoding.as_deref().unwrap_or("fp16") != "fp16" {
        return Err("chat supports only fp16 KV cache encoding".to_owned());
    }
    let checkpoint_directory =
        checkpoint_directory.ok_or_else(|| "chat requires --checkpoint-directory".to_owned())?;
    validate_path_argument(&gguf, "--gguf")?;
    validate_path_argument(&derived_lock, "--derived-lock")?;
    validate_path_argument(&checkpoint_directory, "--checkpoint-directory")?;
    let checkpoint_quota_bytes = checkpoint_quota_bytes
        .ok_or_else(|| "chat requires --checkpoint-quota-bytes".to_owned())?;
    Ok((
        ProductionChatConfigV1 {
            gguf,
            derived_lock,
            device_index,
            target,
            context_length: context_length.unwrap_or_else(|| {
                u32::try_from(QWEN35_RECOMMENDED_CONTEXT_TOKENS)
                    .expect("Qwen recommended context fits u32")
            }),
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

fn open_backend(
    config: ProductionChatConfigV1,
    cancellation_registry: CancellationRegistryV1,
) -> Result<ProductionChatBackendV1, String> {
    let backend = QwenBackendConfigV1 {
        gguf_path: config.gguf,
        derived_lock_path: config.derived_lock,
        device_index: config.device_index,
        target: config.target,
        completion_timeout: Duration::from_secs(config.completion_timeout_seconds),
        shutdown_timeout: Duration::from_secs(config.shutdown_timeout_seconds),
        context_length: config.context_length,
        kv_cache_encoding: KvCacheEncoding::Fp16,
        phase41: phase41_disabled(),
    };
    let session = QwenPersistentChatSessionV1::open(QwenPersistentChatSessionConfigV1 {
        backend,
        checkpoint_directory: config.checkpoint_directory,
        checkpoint_quota_bytes: config.checkpoint_quota_bytes,
    })
    .map_err(|_| "chat model/runtime failed to open".to_owned())?;
    Ok(ProductionChatBackendV1 {
        session: Some(session),
        cancellation_registry,
    })
}

pub(crate) struct ProductionChatBackendV1 {
    session: Option<QwenPersistentChatSessionV1>,
    cancellation_registry: CancellationRegistryV1,
}

impl Drop for ProductionChatBackendV1 {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            let _ = session.shutdown();
        }
    }
}

impl ChatBackendV1 for ProductionChatBackendV1 {
    fn generate(
        &mut self,
        request: &ChatGenerationRequestV1,
    ) -> Result<ChatGenerationResultV1, ChatBackendErrorV1> {
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
        let session = self.session.as_mut().ok_or(ChatBackendErrorV1::Failed)?;
        session
            .save_checkpoint(name, conversation)
            .map_err(|_| ChatBackendErrorV1::CheckpointUnavailable)
    }

    fn commit_turn(&mut self, conversation: &[u8]) -> Result<(), ChatBackendErrorV1> {
        let session = self.session.as_mut().ok_or(ChatBackendErrorV1::Failed)?;
        session
            .commit_turn(conversation)
            .map_err(|_| ChatBackendErrorV1::Failed)
    }

    fn abort_turn(&mut self) -> Result<(), ChatBackendErrorV1> {
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
    fn parser_requires_exact_qwen_fp16_runtime_surface() {
        let error = split_args(["--gguf", "m"].into_iter().map(str::to_owned)).unwrap_err();
        assert!(error.contains("--derived-lock"));
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
