//! Model-independent Phase 44 `chat` CLI surface.
//!
//! The command owns input selection, bounded typed transcript state, and the
//! JSONL event protocol.  It deliberately does not construct a model or
//! interpret checkpoint state planes.  A production adapter implements
//! [`ChatBackendV1`] and can therefore be added without changing this state
//! machine or the existing one-shot `generate` report.

use std::io::{self, BufRead, BufWriter, Read, Write};
use std::path::PathBuf;

use serde_json::{Value, json};
use sllm_frontend::Qwen35ChatMessageV1;

use crate::interactive::{
    InteractiveErrorV1, InteractiveTranscriptV1, MAX_INTERACTIVE_MESSAGES_V1,
    MAX_INTERACTIVE_TRANSCRIPT_BYTES_V1, MAX_PROMPT_FILE_BYTES_V1, PromptSourceKindV1,
    ReversePromptMatcherV1, read_prompt_file_v1,
};

const EVENT_SCHEMA_V1: &str = "sllm-chat-event-v1";
const MAX_NEW_TOKENS_V1: u32 = 4096;
const MAX_STOP_SEQUENCES_V1: usize = 4;
const MAX_CHECKPOINT_NAME_BYTES_V1: usize = 255;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChatBackendErrorV1 {
    Failed,
    Cancelled,
    CheckpointUnavailable,
}

/// Request passed to the model owner.  Input text is kept in typed frontend
/// messages and never formatted into an error or diagnostic string here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ChatGenerationRequestV1 {
    pub(crate) messages: Vec<Qwen35ChatMessageV1>,
    pub(crate) max_new_tokens: u32,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) reverse_prompts: Vec<String>,
    pub(crate) thinking: ChatThinkingModeV1,
    pub(crate) reasoning_budget: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChatThinkingModeV1 {
    Default,
    Enabled,
    Disabled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ChatGenerationResultV1 {
    pub(crate) text: String,
    pub(crate) reasoning: Option<String>,
    pub(crate) finish_reason: ChatFinishReasonV1,
    pub(crate) cancelled: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChatFinishReasonV1 {
    Stop,
    ReversePrompt,
    Length,
    Cancelled,
}

impl ChatFinishReasonV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::ReversePrompt => "reverse_prompt",
            Self::Length => "length",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Backend callback boundary for C2.  Checkpoint bytes are opaque to the CLI;
/// the existing Phase 41 owner validates their model/template/KV identity.
pub(crate) trait ChatBackendV1 {
    fn generate(
        &mut self,
        request: &ChatGenerationRequestV1,
    ) -> Result<ChatGenerationResultV1, ChatBackendErrorV1>;

    fn commit_turn(&mut self, _conversation: &[u8]) -> Result<(), ChatBackendErrorV1> {
        Ok(())
    }

    fn abort_turn(&mut self) -> Result<(), ChatBackendErrorV1> {
        Ok(())
    }

    fn load_checkpoint(&mut self, _name: &str) -> Result<Option<Vec<u8>>, ChatBackendErrorV1> {
        Ok(None)
    }

    fn save_checkpoint(
        &mut self,
        _name: &str,
        _conversation: &[u8],
    ) -> Result<(), ChatBackendErrorV1> {
        Err(ChatBackendErrorV1::CheckpointUnavailable)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ChatSourceV1 {
    Prompt(String),
    Messages(Vec<Qwen35ChatMessageV1>),
    PromptFile(PathBuf),
    Interactive,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ChatOptionsV1 {
    source: ChatSourceV1,
    system: Option<String>,
    reverse_prompts: Vec<String>,
    stop_sequences: Vec<String>,
    max_new_tokens: u32,
    thinking: ChatThinkingModeV1,
    reasoning_budget: Option<u32>,
    checkpoint_load: Option<String>,
    checkpoint_save: Option<String>,
    prompt_file_contents: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParseResultV1 {
    Help,
}

fn parse_value<I>(arguments: &mut I, option: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    arguments
        .next()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_message(value: &str) -> Result<Qwen35ChatMessageV1, String> {
    let (role, content) = value
        .split_once(':')
        .ok_or_else(|| "--message requires ROLE:CONTENT".to_owned())?;
    if content.is_empty() || content.contains('\0') {
        return Err("--message content is empty or invalid".to_owned());
    }
    match role {
        "system" => Ok(Qwen35ChatMessageV1::system(content)),
        "user" => Ok(Qwen35ChatMessageV1::user(content)),
        "assistant" => Ok(Qwen35ChatMessageV1::assistant(content, None)),
        _ => Err("--message role is unsupported".to_owned()),
    }
}

fn parse_u32(value: String, option: &str, min: u32, max: u32) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("{option} value is invalid"))?;
    if parsed < min || parsed > max {
        return Err(format!("{option} value is out of range"));
    }
    Ok(parsed)
}

fn validate_checkpoint_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.len() > MAX_CHECKPOINT_NAME_BYTES_V1
        || name == "."
        || name == ".."
        || name.contains('\0')
        || name.contains('/')
        || name.contains('\\')
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        return Err("checkpoint name is invalid".to_owned());
    }
    Ok(())
}

fn parse<I>(mut arguments: I) -> Result<Result<ChatOptionsV1, ParseResultV1>, String>
where
    I: Iterator<Item = String>,
{
    let mut prompt = None;
    let mut messages = Vec::new();
    let mut message_bytes = 0_usize;
    let mut prompt_file = None;
    let mut system = None;
    let mut reverse_prompts = Vec::new();
    let mut stop_sequences = Vec::new();
    let mut max_new_tokens = 256;
    let mut thinking = ChatThinkingModeV1::Default;
    let mut reasoning_budget = None;
    let mut checkpoint_load = None;
    let mut checkpoint_save = None;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--help" | "-h" => return Ok(Err(ParseResultV1::Help)),
            "--prompt" => {
                if prompt.is_some() {
                    return Err("--prompt was provided more than once".to_owned());
                }
                let value = parse_value(&mut arguments, "--prompt")?;
                if value.contains('\0') {
                    return Err("--prompt contains NUL".to_owned());
                }
                prompt = Some(value);
            }
            "--message" => {
                let value = parse_value(&mut arguments, "--message")?;
                message_bytes = message_bytes
                    .checked_add(value.len())
                    .ok_or_else(|| "chat message input is too large".to_owned())?;
                if messages.len() == MAX_INTERACTIVE_MESSAGES_V1
                    || message_bytes > MAX_INTERACTIVE_TRANSCRIPT_BYTES_V1
                {
                    return Err("chat message input is too large".to_owned());
                }
                messages.push(parse_message(&value)?)
            }
            "--prompt-file" => {
                if prompt_file.is_some() {
                    return Err("--prompt-file was provided more than once".to_owned());
                }
                prompt_file = Some(PathBuf::from(parse_value(&mut arguments, "--prompt-file")?));
            }
            "--system" => {
                if system.is_some() {
                    return Err("--system was provided more than once".to_owned());
                }
                let value = parse_value(&mut arguments, "--system")?;
                if value.is_empty() || value.contains('\0') {
                    return Err("--system is empty or invalid".to_owned());
                }
                system = Some(value);
            }
            "--reverse-prompt" => {
                reverse_prompts.push(parse_value(&mut arguments, "--reverse-prompt")?)
            }
            "--stop" => stop_sequences.push(parse_value(&mut arguments, "--stop")?),
            "--max-new-tokens" => {
                max_new_tokens = parse_u32(
                    parse_value(&mut arguments, "--max-new-tokens")?,
                    "--max-new-tokens",
                    1,
                    MAX_NEW_TOKENS_V1,
                )?;
            }
            "--thinking" => {
                thinking = match parse_value(&mut arguments, "--thinking")?.as_str() {
                    "default" => ChatThinkingModeV1::Default,
                    "enabled" => ChatThinkingModeV1::Enabled,
                    "disabled" => ChatThinkingModeV1::Disabled,
                    _ => return Err("--thinking value is invalid".to_owned()),
                };
            }
            "--reasoning-budget" => {
                reasoning_budget = Some(parse_u32(
                    parse_value(&mut arguments, "--reasoning-budget")?,
                    "--reasoning-budget",
                    1,
                    4096,
                )?);
            }
            "--checkpoint-load" => {
                if checkpoint_load.is_some() {
                    return Err("--checkpoint-load was provided more than once".to_owned());
                }
                let value = parse_value(&mut arguments, "--checkpoint-load")?;
                validate_checkpoint_name(&value)?;
                checkpoint_load = Some(value);
            }
            "--checkpoint-save" => {
                if checkpoint_save.is_some() {
                    return Err("--checkpoint-save was provided more than once".to_owned());
                }
                let value = parse_value(&mut arguments, "--checkpoint-save")?;
                validate_checkpoint_name(&value)?;
                checkpoint_save = Some(value);
            }
            option if option.starts_with('-') => return Err("unknown chat option".to_owned()),
            _ => return Err("unexpected chat argument".to_owned()),
        }
    }

    if stop_sequences.len() > MAX_STOP_SEQUENCES_V1 {
        return Err("at most four --stop values are accepted".to_owned());
    }
    if stop_sequences
        .iter()
        .any(|value| value.is_empty() || value.contains('\0'))
    {
        return Err("--stop value is empty or invalid".to_owned());
    }
    let _ =
        ReversePromptMatcherV1::new(reverse_prompts.clone()).map_err(|error| error.to_string())?;
    if thinking == ChatThinkingModeV1::Disabled && reasoning_budget.is_some() {
        return Err("reasoning budget cannot be used with disabled thinking".to_owned());
    }
    if let Some(system_value) = system.as_ref() {
        if messages
            .iter()
            .any(|message| matches!(message, Qwen35ChatMessageV1::System { .. }))
        {
            return Err("--system conflicts with a system message".to_owned());
        }
        if system_value.len() > MAX_INTERACTIVE_TRANSCRIPT_BYTES_V1 {
            return Err("--system exceeds the transcript limit".to_owned());
        }
    }
    let source = PromptSourceKindV1::select(
        prompt.is_some(),
        !messages.is_empty(),
        prompt_file.is_some(),
        prompt.is_none() && messages.is_empty() && prompt_file.is_none(),
    )
    .map_err(interactive_error)?;
    let source = match source {
        PromptSourceKindV1::Prompt => ChatSourceV1::Prompt(prompt.expect("prompt source")),
        PromptSourceKindV1::Messages => ChatSourceV1::Messages(messages),
        PromptSourceKindV1::PromptFile => {
            ChatSourceV1::PromptFile(prompt_file.expect("file source"))
        }
        PromptSourceKindV1::InteractiveStdin => ChatSourceV1::Interactive,
    };
    if checkpoint_load.is_some() && system.is_some() {
        return Err("system option cannot be combined with checkpoint resume".to_owned());
    }
    match &source {
        ChatSourceV1::Prompt(value)
            if value.is_empty() || value.len() > MAX_PROMPT_FILE_BYTES_V1 =>
        {
            return Err("chat prompt is invalid".to_owned());
        }
        ChatSourceV1::Messages(values) => {
            if !matches!(values.last(), Some(Qwen35ChatMessageV1::User { .. })) {
                return Err("--message must end with a user message".to_owned());
            }
        }
        ChatSourceV1::Prompt(_) | ChatSourceV1::PromptFile(_) | ChatSourceV1::Interactive => {}
    }
    if reasoning_budget.is_some() && thinking != ChatThinkingModeV1::Enabled {
        return Err("reasoning budget requires enabled thinking".to_owned());
    }
    Ok(Ok(ChatOptionsV1 {
        source,
        system,
        reverse_prompts,
        stop_sequences,
        max_new_tokens,
        thinking,
        reasoning_budget,
        checkpoint_load,
        checkpoint_save,
        prompt_file_contents: None,
    }))
}

/// Parse and perform all source-boundary checks before a production backend is
/// opened.  A selected prompt file is read exactly once and retained in the
/// prepared request; its path is never copied into events or errors.
pub(crate) fn preflight<I>(arguments: I) -> Result<Option<ChatOptionsV1>, String>
where
    I: Iterator<Item = String>,
{
    match parse(arguments)? {
        Err(ParseResultV1::Help) => Ok(None),
        Ok(mut options) => {
            if let ChatSourceV1::PromptFile(path) = &options.source {
                options.prompt_file_contents =
                    Some(read_prompt_file_v1(path).map_err(interactive_error)?);
            }
            Ok(Some(options))
        }
    }
}

fn print_help<W: Write>(output: &mut W) -> Result<(), String> {
    writeln!(output, "Usage: sllm chat [OPTIONS]").map_err(|_| "chat output failed".to_owned())?;
    writeln!(
        output,
        "  --gguf PATH --derived-lock PATH --device-index N --target gfx1030|gfx1201|gfx942"
    )
    .map_err(|_| "chat output failed".to_owned())?;
    writeln!(
        output,
        "  [--context-length N] [--kv-cache-encoding fp16] (defaults: model recommendation, fp16)"
    )
    .map_err(|_| "chat output failed".to_owned())?;
    writeln!(
        output,
        "  --checkpoint-directory PATH --checkpoint-quota-bytes N"
    )
    .map_err(|_| "chat output failed".to_owned())?;
    writeln!(
        output,
        "  [--completion-timeout-seconds N] [--shutdown-timeout-seconds N]"
    )
    .map_err(|_| "chat output failed".to_owned())?;
    writeln!(
        output,
        "  --prompt TEXT | --message ROLE:CONTENT | --prompt-file PATH"
    )
    .map_err(|_| "chat output failed".to_owned())?;
    writeln!(
        output,
        "  (without a source, read line-oriented UTF-8 turns from stdin)"
    )
    .map_err(|_| "chat output failed".to_owned())?;
    writeln!(
        output,
        "  --reverse-prompt TEXT (repeat at most four; 1 MiB total)"
    )
    .map_err(|_| "chat output failed".to_owned())?;
    writeln!(output, "  --checkpoint-load NAME --checkpoint-save NAME")
        .map_err(|_| "chat output failed".to_owned())?;
    writeln!(
        output,
        "  --max-new-tokens N --thinking default|enabled|disabled --reasoning-budget N"
    )
    .map_err(|_| "chat output failed".to_owned())?;
    Ok(())
}

fn emit<W: Write>(output: &mut W, event: &str, fields: Value) -> Result<(), String> {
    let mut row = match fields {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    row.insert(
        "schema_version".to_owned(),
        Value::String(EVENT_SCHEMA_V1.to_owned()),
    );
    row.insert("event".to_owned(), Value::String(event.to_owned()));
    serde_json::to_writer(&mut *output, &Value::Object(row))
        .map_err(|_| "chat output failed".to_owned())?;
    output
        .write_all(b"\n")
        .map_err(|_| "chat output failed".to_owned())
}

fn backend_error(error: ChatBackendErrorV1) -> String {
    match error {
        ChatBackendErrorV1::Cancelled => "chat generation cancelled".to_owned(),
        ChatBackendErrorV1::CheckpointUnavailable => "checkpoint unavailable".to_owned(),
        ChatBackendErrorV1::Failed => "chat backend failed".to_owned(),
    }
}

fn interactive_error(_error: InteractiveErrorV1) -> String {
    // Do not include input content, paths, or backend diagnostics in CLI
    // errors.  The machine event only contains a stable error code.
    "chat input is invalid".to_owned()
}

fn source_label(source: &ChatSourceV1) -> &'static str {
    match source {
        ChatSourceV1::Prompt(_) => "prompt",
        ChatSourceV1::Messages(_) => "messages",
        ChatSourceV1::PromptFile(_) => "prompt_file",
        ChatSourceV1::Interactive => "stdin",
    }
}

fn read_source_prompt(source: &ChatSourceV1) -> Result<Option<String>, String> {
    match source {
        ChatSourceV1::Prompt(prompt) => {
            if prompt.is_empty() || prompt.len() > MAX_PROMPT_FILE_BYTES_V1 {
                return Err("chat prompt is invalid".to_owned());
            }
            Ok(Some(prompt.clone()))
        }
        ChatSourceV1::PromptFile(path) => read_prompt_file_v1(path)
            .map(Some)
            .map_err(interactive_error),
        ChatSourceV1::Messages(_) | ChatSourceV1::Interactive => Ok(None),
    }
}

fn make_initial_transcript(
    options: &ChatOptionsV1,
    loaded: Option<InteractiveTranscriptV1>,
) -> Result<(InteractiveTranscriptV1, Option<String>), String> {
    if loaded.is_some() && options.system.is_some() {
        return Err("system option cannot be combined with checkpoint resume".to_owned());
    }
    let had_loaded_checkpoint = loaded.is_some();
    let mut transcript = loaded.unwrap_or_default();
    match &options.source {
        ChatSourceV1::Interactive => Ok((transcript, None)),
        ChatSourceV1::Prompt(_) | ChatSourceV1::PromptFile(_) => {
            if let Some(system) = options.system.as_deref() {
                transcript =
                    InteractiveTranscriptV1::new(Some(system)).map_err(interactive_error)?;
            }
            let prompt = match options.prompt_file_contents.clone() {
                Some(prompt) => prompt,
                None => read_source_prompt(&options.source)?
                    .ok_or_else(|| "chat prompt is invalid".to_owned())?,
            };
            Ok((transcript, Some(prompt)))
        }
        ChatSourceV1::Messages(messages) => {
            if had_loaded_checkpoint {
                return Err("checkpoint resume cannot combine with --message".to_owned());
            }
            let mut all = Vec::new();
            if let Some(system) = options.system.as_deref() {
                all.push(Qwen35ChatMessageV1::system(system));
            }
            all.extend(messages.iter().cloned());
            let last = all
                .pop()
                .ok_or_else(|| "--message requires at least one message".to_owned())?;
            let user = match last {
                Qwen35ChatMessageV1::User { content } => content,
                _ => return Err("--message must end with a user message".to_owned()),
            };
            transcript = InteractiveTranscriptV1::from_messages(&all).map_err(interactive_error)?;
            Ok((transcript, Some(user)))
        }
    }
}

fn execute_turn<B, W>(
    backend: &mut B,
    output: &mut W,
    transcript: &mut InteractiveTranscriptV1,
    user: &str,
    options: &ChatOptionsV1,
    turn_index: usize,
) -> Result<bool, String>
where
    B: ChatBackendV1,
    W: Write,
{
    if user.is_empty() || user.contains('\0') {
        emit(
            output,
            "turn.failed",
            json!({"turn": turn_index, "code": "empty_input"}),
        )?;
        return Err("chat input is invalid".to_owned());
    }
    if transcript.messages().len().saturating_add(2) > MAX_INTERACTIVE_MESSAGES_V1 {
        emit(
            output,
            "turn.failed",
            json!({"turn": turn_index, "code": "message_limit"}),
        )?;
        return Err("chat transcript exceeds its message limit".to_owned());
    }
    let draft = transcript.begin_turn(user);
    let messages = draft.messages().map_err(interactive_error)?;
    emit(output, "turn.started", json!({"turn": turn_index}))?;
    let request = ChatGenerationRequestV1 {
        messages,
        max_new_tokens: options.max_new_tokens,
        stop_sequences: options.stop_sequences.clone(),
        reverse_prompts: options.reverse_prompts.clone(),
        thinking: options.thinking,
        reasoning_budget: options.reasoning_budget,
    };
    let result = match backend.generate(&request) {
        Ok(result) => result,
        Err(ChatBackendErrorV1::Cancelled) => {
            let _ = backend.abort_turn();
            emit(
                output,
                "turn.cancelled",
                json!({"turn": turn_index, "code": "cancelled"}),
            )?;
            return Ok(false);
        }
        Err(error) => {
            let _ = backend.abort_turn();
            emit(
                output,
                "turn.failed",
                json!({"turn": turn_index, "code": "backend_failure"}),
            )?;
            return Err(backend_error(error));
        }
    };
    drop(draft);
    if result.cancelled || result.finish_reason == ChatFinishReasonV1::Cancelled {
        let _ = backend.abort_turn();
        emit(
            output,
            "turn.cancelled",
            json!({"turn": turn_index, "code": "cancelled"}),
        )?;
        return Ok(false);
    }
    let mut matcher = match ReversePromptMatcherV1::new(options.reverse_prompts.clone()) {
        Ok(matcher) => matcher,
        Err(error) => {
            let _ = backend.abort_turn();
            return Err(interactive_error(error));
        }
    };
    let match_result = matcher.push(&result.text);
    let mut visible = match_result.visible;
    visible.push_str(&matcher.finish());
    let finish_reason = if match_result.matched.is_some()
        || result.finish_reason == ChatFinishReasonV1::ReversePrompt
    {
        "reverse_prompt"
    } else {
        result.finish_reason.as_str()
    };
    if visible.is_empty() {
        let _ = backend.abort_turn();
        emit(
            output,
            "turn.failed",
            json!({"turn": turn_index, "code": "empty_generation"}),
        )?;
        return Err("chat backend returned no visible output".to_owned());
    }
    let mut candidate = transcript.clone();
    if let Err(error) = candidate
        .begin_turn(user)
        .commit_with_reasoning(&visible, result.reasoning.as_deref())
    {
        let _ = backend.abort_turn();
        emit(
            output,
            "turn.failed",
            json!({"turn": turn_index, "code": "transcript_commit_failed"}),
        )?;
        return Err(interactive_error(error));
    }
    let candidate_bytes = match candidate.encode() {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = backend.abort_turn();
            emit(
                output,
                "turn.failed",
                json!({"turn": turn_index, "code": "transcript_encode_failed"}),
            )?;
            return Err(interactive_error(error));
        }
    };
    if let Some(name) = options.checkpoint_save.as_deref() {
        if let Err(error) = backend.save_checkpoint(name, &candidate_bytes) {
            let _ = backend.abort_turn();
            emit(
                output,
                "turn.failed",
                json!({"turn": turn_index, "code": "checkpoint_save_failed"}),
            )?;
            return Err(backend_error(error));
        }
    } else if let Err(error) = backend.commit_turn(&candidate_bytes) {
        let _ = backend.abort_turn();
        emit(
            output,
            "turn.failed",
            json!({"turn": turn_index, "code": "checkpoint_commit_failed"}),
        )?;
        return Err(backend_error(error));
    }
    *transcript = candidate;
    if let Some(reasoning) = result.reasoning.as_deref() {
        if !reasoning.is_empty() {
            emit(
                output,
                "assistant.reasoning.delta",
                json!({"turn": turn_index, "text": reasoning}),
            )?;
        }
    }
    emit(
        output,
        "assistant.delta",
        json!({"turn": turn_index, "text": visible}),
    )?;
    emit(
        output,
        "turn.completed",
        json!({"turn": turn_index, "finish_reason": finish_reason, "committed": true}),
    )?;
    Ok(true)
}

fn load_checkpoint<B: ChatBackendV1>(
    backend: &mut B,
    name: Option<&str>,
) -> Result<Option<InteractiveTranscriptV1>, String> {
    let Some(name) = name else { return Ok(None) };
    let bytes = backend
        .load_checkpoint(name)
        .map_err(backend_error)?
        .ok_or_else(|| "checkpoint unavailable".to_owned())?;
    InteractiveTranscriptV1::decode(&bytes)
        .map(Some)
        .map_err(|_| "checkpoint rejected".to_owned())
}

fn run_with_options<B, W, R>(
    options: ChatOptionsV1,
    backend: &mut B,
    output: &mut W,
    input: &mut R,
) -> Result<(), String>
where
    B: ChatBackendV1,
    W: Write,
    R: BufRead,
{
    let loaded = load_checkpoint(backend, options.checkpoint_load.as_deref())?;
    emit(
        output,
        "session.started",
        json!({"source": source_label(&options.source), "interactive": matches!(options.source, ChatSourceV1::Interactive)}),
    )?;
    let (mut transcript, initial_turn) = make_initial_transcript(&options, loaded)?;
    if let Some(user) = initial_turn {
        let committed = execute_turn(backend, output, &mut transcript, &user, &options, 0)?;
        emit(
            output,
            "session.completed",
            json!({"turns": if committed { 1 } else { 0 }}),
        )?;
        return Ok(());
    }

    let mut line = Vec::new();
    let mut turn_index = 0_usize;
    loop {
        line.clear();
        let read = (&mut *input)
            .take((MAX_INTERACTIVE_TRANSCRIPT_BYTES_V1 as u64) + 1)
            .read_until(b'\n', &mut line)
            .map_err(|_| "chat stdin read failed".to_owned())?;
        if read == 0 {
            emit(output, "session.eof", json!({"reason": "eof"}))?;
            return Ok(());
        }
        if line.len() > MAX_INTERACTIVE_TRANSCRIPT_BYTES_V1 {
            emit(
                output,
                "turn.failed",
                json!({"turn": turn_index, "code": "message_limit"}),
            )?;
            return Err("chat input is too large".to_owned());
        }
        while matches!(line.last(), Some(b'\n' | b'\r')) {
            line.pop();
        }
        let line = String::from_utf8(std::mem::take(&mut line))
            .map_err(|_| "chat stdin input is not valid UTF-8".to_owned())?;
        if line.contains('\0') {
            return Err("chat stdin input contains NUL".to_owned());
        }
        if line.is_empty() {
            emit(output, "turn.empty", json!({"turn": turn_index}))?;
            continue;
        }
        if line == "/exit" {
            emit(output, "session.eof", json!({"reason": "exit"}))?;
            return Ok(());
        }
        if line == "/cancel" {
            emit(
                output,
                "turn.cancelled",
                json!({"turn": turn_index, "code": "cancelled"}),
            )?;
            turn_index = turn_index.saturating_add(1);
            continue;
        }
        execute_turn(
            backend,
            output,
            &mut transcript,
            &line,
            &options,
            turn_index,
        )?;
        turn_index = turn_index.saturating_add(1);
    }
}

/// Execute a preflighted chat request against an already-open backend.
///
/// Production adapters call [`preflight`] before opening a model so malformed
/// CLI input and prompt files cannot trigger HIP/model startup.  Keeping this
/// small handoff separate also ensures a selected prompt file is not parsed or
/// read a second time.
pub(crate) fn run_prepared<B, W, R>(
    options: ChatOptionsV1,
    backend: &mut B,
    output: &mut W,
    input: &mut R,
) -> Result<(), String>
where
    B: ChatBackendV1,
    W: Write,
    R: BufRead,
{
    run_with_options(options, backend, output, input)
}

pub(crate) fn run<I>(arguments: I) -> Result<(), String>
where
    I: Iterator<Item = String>,
{
    let mut output = BufWriter::new(io::stdout().lock());
    let mut backend = UnavailableBackendV1;
    let stdin = io::stdin();
    let mut input = stdin.lock();
    run_with_backend_and_input(arguments, &mut backend, &mut output, &mut input)
}

#[allow(dead_code)]
pub(crate) fn run_with_backend<I, B, W>(
    arguments: I,
    backend: &mut B,
    output: &mut W,
) -> Result<(), String>
where
    I: Iterator<Item = String>,
    B: ChatBackendV1,
    W: Write,
{
    let stdin = io::stdin();
    let mut input = stdin.lock();
    run_with_backend_and_input(arguments, backend, output, &mut input)
}

#[allow(dead_code)]
pub(crate) fn run_with_backend_and_input<I, B, W, R>(
    arguments: I,
    backend: &mut B,
    output: &mut W,
    input: &mut R,
) -> Result<(), String>
where
    I: Iterator<Item = String>,
    B: ChatBackendV1,
    W: Write,
    R: BufRead,
{
    match preflight(arguments)? {
        None => print_help(output),
        Some(options) => run_prepared(options, backend, output, input),
    }
}

struct UnavailableBackendV1;

impl ChatBackendV1 for UnavailableBackendV1 {
    fn generate(
        &mut self,
        _request: &ChatGenerationRequestV1,
    ) -> Result<ChatGenerationResultV1, ChatBackendErrorV1> {
        Err(ChatBackendErrorV1::Failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct FakeBackendV1 {
        outputs: Arc<Mutex<Vec<Result<ChatGenerationResultV1, ChatBackendErrorV1>>>>,
        requests: Arc<Mutex<Vec<ChatGenerationRequestV1>>>,
        saved: Arc<Mutex<Vec<Vec<u8>>>>,
        committed: Arc<Mutex<Vec<Vec<u8>>>>,
        aborted: Arc<Mutex<usize>>,
        checkpoint: Option<Vec<u8>>,
    }

    impl FakeBackendV1 {
        fn new(outputs: Vec<Result<ChatGenerationResultV1, ChatBackendErrorV1>>) -> Self {
            Self {
                outputs: Arc::new(Mutex::new(outputs)),
                requests: Arc::new(Mutex::new(Vec::new())),
                saved: Arc::new(Mutex::new(Vec::new())),
                committed: Arc::new(Mutex::new(Vec::new())),
                aborted: Arc::new(Mutex::new(0)),
                checkpoint: None,
            }
        }
    }

    impl ChatBackendV1 for FakeBackendV1 {
        fn generate(
            &mut self,
            request: &ChatGenerationRequestV1,
        ) -> Result<ChatGenerationResultV1, ChatBackendErrorV1> {
            self.requests.lock().unwrap().push(request.clone());
            self.outputs.lock().unwrap().remove(0)
        }

        fn load_checkpoint(&mut self, _name: &str) -> Result<Option<Vec<u8>>, ChatBackendErrorV1> {
            Ok(self.checkpoint.clone())
        }

        fn save_checkpoint(
            &mut self,
            _name: &str,
            conversation: &[u8],
        ) -> Result<(), ChatBackendErrorV1> {
            self.saved.lock().unwrap().push(conversation.to_vec());
            Ok(())
        }

        fn commit_turn(&mut self, conversation: &[u8]) -> Result<(), ChatBackendErrorV1> {
            self.committed.lock().unwrap().push(conversation.to_vec());
            Ok(())
        }

        fn abort_turn(&mut self) -> Result<(), ChatBackendErrorV1> {
            *self.aborted.lock().unwrap() += 1;
            Ok(())
        }
    }

    fn completed(text: &str) -> Result<ChatGenerationResultV1, ChatBackendErrorV1> {
        Ok(ChatGenerationResultV1 {
            text: text.to_owned(),
            reasoning: None,
            finish_reason: ChatFinishReasonV1::Stop,
            cancelled: false,
        })
    }

    fn completed_with_reasoning(
        text: &str,
        reasoning: &str,
    ) -> Result<ChatGenerationResultV1, ChatBackendErrorV1> {
        Ok(ChatGenerationResultV1 {
            text: text.to_owned(),
            reasoning: Some(reasoning.to_owned()),
            finish_reason: ChatFinishReasonV1::Stop,
            cancelled: false,
        })
    }

    #[test]
    fn parser_enforces_closed_prompt_source_matrix() {
        assert!(parse(vec!["--prompt".to_owned(), "x".to_owned()].into_iter()).is_ok());
        assert!(parse(vec!["--message".to_owned(), "user:x".to_owned()].into_iter()).is_ok());
        assert!(
            parse(
                vec![
                    "--prompt".to_owned(),
                    "x".to_owned(),
                    "--prompt-file".to_owned(),
                    "x".to_owned()
                ]
                .into_iter()
            )
            .is_err()
        );
        assert!(parse(Vec::<String>::new().into_iter()).is_ok());
    }

    #[test]
    fn help_lists_production_runtime_requirements() {
        let mut output = Vec::new();
        let mut backend = FakeBackendV1::new(Vec::new());
        run_with_backend(
            vec!["--help".to_owned()].into_iter(),
            &mut backend,
            &mut output,
        )
        .unwrap();
        let help = String::from_utf8(output).unwrap();
        assert!(help.contains("--gguf PATH"));
        assert!(help.contains("--derived-lock PATH"));
        assert!(help.contains("[--context-length N]"));
        assert!(help.contains("--kv-cache-encoding fp16"));
        assert!(help.contains("--checkpoint-directory PATH"));
    }

    #[test]
    fn preflight_rejects_source_shape_and_size_before_backend_use() {
        let oversized = "x".repeat(MAX_PROMPT_FILE_BYTES_V1 + 1);
        assert!(preflight(vec!["--prompt".to_owned(), oversized].into_iter()).is_err());
        assert!(
            preflight(vec!["--message".to_owned(), "assistant:answer".to_owned()].into_iter())
                .is_err()
        );
        assert!(
            preflight(
                vec![
                    "--checkpoint-load".to_owned(),
                    "session".to_owned(),
                    "--system".to_owned(),
                    "system".to_owned(),
                ]
                .into_iter()
            )
            .is_err()
        );
    }

    #[test]
    fn fake_noninteractive_turn_commits_and_emits_jsonl_without_input_payload() {
        let mut backend = FakeBackendV1::new(vec![completed("answer")]);
        let mut output = Vec::new();
        run_with_backend(
            vec![
                "--prompt".to_owned(),
                "private prompt".to_owned(),
                "--checkpoint-save".to_owned(),
                "session".to_owned(),
            ]
            .into_iter(),
            &mut backend,
            &mut output,
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("assistant.delta"));
        assert!(!text.contains("private prompt"));
        assert_eq!(backend.requests.lock().unwrap().len(), 1);
        assert_eq!(backend.saved.lock().unwrap().len(), 1);
        let saved = InteractiveTranscriptV1::decode(&backend.saved.lock().unwrap()[0]).unwrap();
        assert_eq!(saved.messages().len(), 2);
    }

    #[test]
    fn reasoning_and_visible_output_use_distinct_committed_events() {
        let mut backend = FakeBackendV1::new(vec![completed_with_reasoning("answer", "analysis")]);
        let mut output = Vec::new();
        run_with_backend(
            vec!["--prompt".to_owned(), "question".to_owned()].into_iter(),
            &mut backend,
            &mut output,
        )
        .unwrap();
        let rows = String::from_utf8(output).unwrap();
        assert!(rows.contains("assistant.reasoning.delta"));
        assert!(rows.contains("assistant.delta"));
        assert!(rows.find("assistant.reasoning.delta") < rows.find("assistant.delta"));
        let committed = backend.committed.lock().unwrap();
        let transcript = InteractiveTranscriptV1::decode(&committed[0]).unwrap();
        assert!(matches!(
            transcript.messages().as_slice(),
            [
                Qwen35ChatMessageV1::User { content },
                Qwen35ChatMessageV1::Assistant {
                    content: answer,
                    reasoning_content: Some(reasoning),
                }
            ] if content == "question" && answer == "answer" && reasoning == "analysis"
        ));
    }

    #[test]
    fn failed_or_cancelled_turn_does_not_publish_transcript() {
        let mut backend = FakeBackendV1::new(vec![Err(ChatBackendErrorV1::Cancelled)]);
        let mut output = Vec::new();
        run_with_backend(
            vec!["--prompt".to_owned(), "private prompt".to_owned()].into_iter(),
            &mut backend,
            &mut output,
        )
        .unwrap();
        assert!(backend.saved.lock().unwrap().is_empty());
        assert_eq!(*backend.aborted.lock().unwrap(), 1);
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("turn.cancelled"));
        assert!(!text.contains("private prompt"));
    }

    #[test]
    fn fake_scripted_interactive_input_handles_empty_cancel_and_eof() {
        let mut backend = FakeBackendV1::new(vec![completed("answer")]);
        let mut output = Vec::new();
        let mut input = Cursor::new(b"\nhello\n/cancel\n/exit\n".to_vec());
        run_with_backend_and_input(
            Vec::<String>::new().into_iter(),
            &mut backend,
            &mut output,
            &mut input,
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("turn.empty"));
        assert!(text.contains("turn.cancelled"));
        assert!(text.contains("session.eof"));
        assert_eq!(backend.requests.lock().unwrap().len(), 1);
        assert_eq!(backend.requests.lock().unwrap()[0].messages.len(), 1);
    }

    #[test]
    fn interactive_line_reader_stops_at_the_byte_limit() {
        let mut backend = FakeBackendV1::new(Vec::new());
        let mut output = Vec::new();
        let mut input = Cursor::new(vec![b'x'; MAX_INTERACTIVE_TRANSCRIPT_BYTES_V1 + 1]);
        let error = run_with_backend_and_input(
            Vec::<String>::new().into_iter(),
            &mut backend,
            &mut output,
            &mut input,
        )
        .unwrap_err();
        assert_eq!(error, "chat input is too large");
        assert!(backend.requests.lock().unwrap().is_empty());
    }

    #[test]
    fn reverse_prompt_is_a_turn_boundary_and_not_stop_semantics() {
        let mut backend = FakeBackendV1::new(vec![completed("answer<next>")]);
        let mut output = Vec::new();
        run_with_backend(
            vec![
                "--prompt".to_owned(),
                "x".to_owned(),
                "--reverse-prompt".to_owned(),
                "<next>".to_owned(),
            ]
            .into_iter(),
            &mut backend,
            &mut output,
        )
        .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("reverse_prompt"));
        assert!(text.contains("answer"));
        assert!(!text.contains("<next>"));
    }

    #[test]
    fn checkpoint_resume_uses_opaque_transcript_and_rejects_invalid_payload() {
        let transcript = InteractiveTranscriptV1::new(Some("system"))
            .unwrap()
            .encode()
            .unwrap();
        let mut backend = FakeBackendV1::new(vec![completed("answer")]);
        backend.checkpoint = Some(transcript);
        let mut output = Vec::new();
        run_with_backend(
            vec![
                "--checkpoint-load".to_owned(),
                "session".to_owned(),
                "--prompt".to_owned(),
                "next".to_owned(),
            ]
            .into_iter(),
            &mut backend,
            &mut output,
        )
        .unwrap();
        assert_eq!(backend.requests.lock().unwrap()[0].messages.len(), 2);

        let mut invalid = FakeBackendV1::new(vec![]);
        invalid.checkpoint = Some(b"not-json".to_vec());
        let mut output = Vec::new();
        let error = run_with_backend(
            vec!["--checkpoint-load".to_owned(), "session".to_owned()].into_iter(),
            &mut invalid,
            &mut output,
        )
        .expect_err("invalid checkpoint must fail closed");
        assert_eq!(error, "checkpoint rejected");
    }
}
