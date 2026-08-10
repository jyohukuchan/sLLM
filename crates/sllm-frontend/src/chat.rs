use core::fmt;

use sha2::{Digest, Sha256};
use sllm_core::{FrontendAssetKind, ModelLock, VerifiedCache};

pub const QWEN35_CHAT_RENDERER_VERSION: u8 = 1;
pub const QWEN35_CHAT_TEMPLATE_FILENAME: &str = "chat_template.jinja";
pub const QWEN35_CHAT_TEMPLATE_SIZE_BYTES: u64 = 7_756;
pub const QWEN35_CHAT_TEMPLATE_SHA256: &str =
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715";

const QWEN35_REPO_ID: &str = "Qwen/Qwen3.5-4B";
const QWEN35_RESOLVED_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";

/// Hard host-side output cap for one rendered prompt (16 MiB).
///
/// This is deliberately independent of tokenizer/model context limits: the
/// renderer accepts UTF-8 bytes, while a later tokenizer enforces token limits.
pub const QWEN35_CHAT_MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;

const IM_START: &str = "<|im_start|>";
const IM_END_LINE: &str = "<|im_end|>\n";
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const GENERATION_THINKING: &str = "<|im_start|>assistant\n<think>\n";
const GENERATION_DISABLED: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen35ChatMessageV1 {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
        reasoning_content: Option<String>,
    },
}

impl Qwen35ChatMessageV1 {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>, reasoning_content: Option<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            reasoning_content,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThinkingModeV1 {
    TemplateDefault,
    Enabled,
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35RenderOptionsV1 {
    pub add_generation_prompt: bool,
    pub thinking: ThinkingModeV1,
}

impl Default for Qwen35RenderOptionsV1 {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            thinking: ThinkingModeV1::TemplateDefault,
        }
    }
}

/// Closed value categories for validating data before it enters the typed
/// renderer. String bytes are validated as UTF-8 without a lossy conversion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UntrustedChatValueV1 {
    Missing,
    Null,
    Boolean(bool),
    Number(String),
    StringBytes(Vec<u8>),
    Array,
    Object,
}

impl UntrustedChatValueV1 {
    pub fn string(value: impl AsRef<str>) -> Self {
        Self::StringBytes(value.as_ref().as_bytes().to_vec())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UntrustedChatMessageV1 {
    pub role: UntrustedChatValueV1,
    pub content: UntrustedChatValueV1,
    pub reasoning_content: Option<UntrustedChatValueV1>,
    pub tool_call: Option<UntrustedChatValueV1>,
    pub tool_calls: Option<UntrustedChatValueV1>,
    pub image: Option<UntrustedChatValueV1>,
    pub image_url: Option<UntrustedChatValueV1>,
    pub video: Option<UntrustedChatValueV1>,
    pub unknown_fields: Vec<String>,
}

impl UntrustedChatMessageV1 {
    pub fn text(role: &str, content: &str) -> Self {
        Self {
            role: UntrustedChatValueV1::string(role),
            content: UntrustedChatValueV1::string(content),
            reasoning_content: None,
            tool_call: None,
            tool_calls: None,
            image: None,
            image_url: None,
            video: None,
            unknown_fields: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UntrustedChatRequestV1 {
    pub renderer_version: u8,
    pub messages: Vec<UntrustedChatMessageV1>,
    pub add_generation_prompt: UntrustedChatValueV1,
    pub enable_thinking: Option<UntrustedChatValueV1>,
    pub tools: Option<UntrustedChatValueV1>,
    pub tool_choice: Option<UntrustedChatValueV1>,
    pub unknown_fields: Vec<String>,
}

impl UntrustedChatRequestV1 {
    pub fn text(messages: Vec<UntrustedChatMessageV1>) -> Self {
        Self {
            renderer_version: QWEN35_CHAT_RENDERER_VERSION,
            messages,
            add_generation_prompt: UntrustedChatValueV1::Boolean(true),
            enable_thinking: None,
            tools: None,
            tool_choice: None,
            unknown_fields: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatFieldV1 {
    Role,
    Content,
    ReasoningContent,
}

impl ChatFieldV1 {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Role => "role",
            Self::Content => "content",
            Self::ReasoningContent => "reasoning_content",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatRenderError {
    UnsupportedTemplateIdentity,
    LockCacheFingerprintMismatch,
    TemplateAssetUnavailable,
    TemplateAssetInvalidUtf8,
    UnsupportedRendererVersion { actual: u8 },
    UnknownRequestField,
    InvalidAddGenerationPrompt,
    InvalidThinkingMode,
    EmptyMessages,
    MissingMessageField { index: usize, field: ChatFieldV1 },
    InvalidMessageUtf8 { index: usize, field: ChatFieldV1 },
    InvalidRoleType { index: usize },
    UnsupportedRole { index: usize },
    InvalidContentType { index: usize },
    InvalidReasoningContentType { index: usize },
    ReasoningContentOnNonAssistant { index: usize },
    UnsupportedToolInput { index: Option<usize> },
    UnsupportedImageInput { index: usize },
    UnsupportedVideoInput { index: usize },
    UnknownMessageField { index: usize },
    MultipleSystemMessages,
    MisplacedSystemMessage { index: usize },
    ToolResponseUserContent { index: usize },
    NoOrdinaryUserMessage,
    OutputLimitExceedsHostCap,
    OutputTooLarge { limit_bytes: usize },
}

impl fmt::Display for ChatRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTemplateIdentity => formatter
                .write_str("chat template identity is not the fixed Qwen3.5 renderer-v1 asset"),
            Self::LockCacheFingerprintMismatch => {
                formatter.write_str("model lock and verified cache fingerprints differ")
            }
            Self::TemplateAssetUnavailable => {
                formatter.write_str("verified chat template asset is unavailable")
            }
            Self::TemplateAssetInvalidUtf8 => {
                formatter.write_str("verified chat template asset is not valid UTF-8")
            }
            Self::UnsupportedRendererVersion { actual } => {
                write!(
                    formatter,
                    "unsupported Qwen3.5 chat renderer version {actual}"
                )
            }
            Self::UnknownRequestField => formatter.write_str("request contains an unknown field"),
            Self::InvalidAddGenerationPrompt => {
                formatter.write_str("add_generation_prompt must be a boolean")
            }
            Self::InvalidThinkingMode => {
                formatter.write_str("enable_thinking must be a boolean when present")
            }
            Self::EmptyMessages => formatter.write_str("chat messages must not be empty"),
            Self::MissingMessageField { index, field } => write!(
                formatter,
                "message {index} is missing required {}",
                field.as_str()
            ),
            Self::InvalidMessageUtf8 { index, field } => write!(
                formatter,
                "message {index} {} is not valid UTF-8",
                field.as_str()
            ),
            Self::InvalidRoleType { index } => {
                write!(formatter, "message {index} role must be a string")
            }
            Self::UnsupportedRole { index } => {
                write!(formatter, "message {index} role is unsupported")
            }
            Self::InvalidContentType { index } => {
                write!(formatter, "message {index} content must be a string")
            }
            Self::InvalidReasoningContentType { index } => write!(
                formatter,
                "message {index} reasoning_content must be a string when present"
            ),
            Self::ReasoningContentOnNonAssistant { index } => write!(
                formatter,
                "message {index} reasoning_content is only valid for assistant"
            ),
            Self::UnsupportedToolInput { index: Some(index) } => {
                write!(formatter, "message {index} contains unsupported tool input")
            }
            Self::UnsupportedToolInput { index: None } => {
                formatter.write_str("request contains unsupported tool input")
            }
            Self::UnsupportedImageInput { index } => {
                write!(
                    formatter,
                    "message {index} contains unsupported image input"
                )
            }
            Self::UnsupportedVideoInput { index } => {
                write!(
                    formatter,
                    "message {index} contains unsupported video input"
                )
            }
            Self::UnknownMessageField { index } => {
                write!(formatter, "message {index} contains an unknown field")
            }
            Self::MultipleSystemMessages => {
                formatter.write_str("multiple system messages are unsupported")
            }
            Self::MisplacedSystemMessage { index } => {
                write!(
                    formatter,
                    "system message must be at index 0, found at {index}"
                )
            }
            Self::ToolResponseUserContent { index } => write!(
                formatter,
                "message {index} user content is an unsupported tool response"
            ),
            Self::NoOrdinaryUserMessage => {
                formatter.write_str("chat requires at least one ordinary user message")
            }
            Self::OutputLimitExceedsHostCap => {
                formatter.write_str("requested output limit exceeds the renderer host cap")
            }
            Self::OutputTooLarge { limit_bytes } => write!(
                formatter,
                "rendered chat exceeds the {limit_bytes}-byte output limit"
            ),
        }
    }
}

impl std::error::Error for ChatRenderError {}

/// Versioned renderer for the one reviewed Qwen3.5 text-only template.
///
/// Construction requires the fixed metadata identity and independently hashes
/// the exact bytes returned by the bounded `ChatTemplateJinja` read. The
/// retained lock fingerprint remains a consistency label after construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35ChatTemplateV1 {
    consistency_label: String,
}

impl Qwen35ChatTemplateV1 {
    pub fn from_verified_cache(
        lock: &ModelLock,
        cache: &VerifiedCache,
    ) -> Result<Self, ChatRenderError> {
        Self::from_verified_cache_impl(lock, cache, QWEN35_CHAT_TEMPLATE_SHA256)
    }

    /// Test-only construction still performs the production metadata, bounded
    /// read, UTF-8, and success-path checks.  It supplies an explicit digest
    /// for a synthetic same-size asset because the real locked template is
    /// intentionally not a CI fixture.  This seam is not present in normal
    /// builds and cannot construct a renderer without a verified cache read.
    #[cfg(test)]
    fn from_verified_cache_with_test_digest(
        lock: &ModelLock,
        cache: &VerifiedCache,
        expected_sha256: &str,
    ) -> Result<Self, ChatRenderError> {
        Self::from_verified_cache_impl(lock, cache, expected_sha256)
    }

    fn from_verified_cache_impl(
        lock: &ModelLock,
        cache: &VerifiedCache,
        expected_sha256: &str,
    ) -> Result<Self, ChatRenderError> {
        if lock.model.repo_id != QWEN35_REPO_ID
            || lock.model.resolved_revision != QWEN35_RESOLVED_REVISION
        {
            return Err(ChatRenderError::UnsupportedTemplateIdentity);
        }
        if lock.model.tokenizer_contract.chat_template_path != QWEN35_CHAT_TEMPLATE_FILENAME {
            return Err(ChatRenderError::UnsupportedTemplateIdentity);
        }
        let mut locked = lock
            .model
            .files
            .iter()
            .filter(|file| file.path == QWEN35_CHAT_TEMPLATE_FILENAME);
        let Some(locked_file) = locked.next() else {
            return Err(ChatRenderError::UnsupportedTemplateIdentity);
        };
        if locked.next().is_some()
            || locked_file.size_bytes != QWEN35_CHAT_TEMPLATE_SIZE_BYTES
            || locked_file.sha256 != QWEN35_CHAT_TEMPLATE_SHA256
        {
            return Err(ChatRenderError::UnsupportedTemplateIdentity);
        }

        let mut verified = cache
            .files
            .iter()
            .filter(|file| file.path == QWEN35_CHAT_TEMPLATE_FILENAME);
        let Some(verified_file) = verified.next() else {
            return Err(ChatRenderError::UnsupportedTemplateIdentity);
        };
        if verified.next().is_some()
            || verified_file.size_bytes != QWEN35_CHAT_TEMPLATE_SIZE_BYTES
            || verified_file.sha256 != QWEN35_CHAT_TEMPLATE_SHA256
        {
            return Err(ChatRenderError::UnsupportedTemplateIdentity);
        }
        if cache.lock_fingerprint != lock.fingerprint() {
            return Err(ChatRenderError::LockCacheFingerprintMismatch);
        }

        let bytes = cache
            .read_frontend_asset(FrontendAssetKind::ChatTemplateJinja)
            .map_err(|_| ChatRenderError::TemplateAssetUnavailable)?;
        if bytes.len() != QWEN35_CHAT_TEMPLATE_SIZE_BYTES as usize {
            return Err(ChatRenderError::UnsupportedTemplateIdentity);
        }
        validate_template_bytes(&bytes, expected_sha256)?;

        Ok(Self {
            consistency_label: lock.fingerprint().to_owned(),
        })
    }

    pub const fn version(&self) -> u8 {
        QWEN35_CHAT_RENDERER_VERSION
    }

    pub fn consistency_label(&self) -> &str {
        &self.consistency_label
    }

    pub fn render(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<String, ChatRenderError> {
        self.render_with_output_limit(messages, options, QWEN35_CHAT_MAX_OUTPUT_BYTES)
    }

    pub fn render_with_output_limit(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
        output_limit_bytes: usize,
    ) -> Result<String, ChatRenderError> {
        if output_limit_bytes > QWEN35_CHAT_MAX_OUTPUT_BYTES {
            return Err(ChatRenderError::OutputLimitExceedsHostCap);
        }
        let last_user = validate_typed_messages(messages)?;
        let planned = plan_output(messages, options, last_user, output_limit_bytes)?;
        let mut output = String::with_capacity(planned);
        write_output(&mut output, messages, options, last_user);
        debug_assert_eq!(output.len(), planned);
        Ok(output)
    }

    pub fn render_untrusted(
        &self,
        request: UntrustedChatRequestV1,
    ) -> Result<String, ChatRenderError> {
        let (messages, options) = validate_untrusted_request(request)?;
        self.render(&messages, options)
    }
}

fn validate_template_bytes(bytes: &[u8], expected_sha256: &str) -> Result<(), ChatRenderError> {
    if !template_digest_matches(bytes, expected_sha256) {
        return Err(ChatRenderError::UnsupportedTemplateIdentity);
    }
    core::str::from_utf8(bytes).map_err(|_| ChatRenderError::TemplateAssetInvalidUtf8)?;
    Ok(())
}

fn template_digest_matches(bytes: &[u8], expected_sha256: &str) -> bool {
    format!("{:x}", Sha256::digest(bytes)) == expected_sha256
}

fn validate_untrusted_request(
    request: UntrustedChatRequestV1,
) -> Result<(Vec<Qwen35ChatMessageV1>, Qwen35RenderOptionsV1), ChatRenderError> {
    if request.renderer_version != QWEN35_CHAT_RENDERER_VERSION {
        return Err(ChatRenderError::UnsupportedRendererVersion {
            actual: request.renderer_version,
        });
    }
    if request.tools.is_some() || request.tool_choice.is_some() {
        return Err(ChatRenderError::UnsupportedToolInput { index: None });
    }
    if !request.unknown_fields.is_empty() {
        return Err(ChatRenderError::UnknownRequestField);
    }
    if request.messages.is_empty() {
        return Err(ChatRenderError::EmptyMessages);
    }

    let add_generation_prompt = match request.add_generation_prompt {
        UntrustedChatValueV1::Boolean(value) => value,
        _ => return Err(ChatRenderError::InvalidAddGenerationPrompt),
    };
    let thinking = match request.enable_thinking {
        None => ThinkingModeV1::TemplateDefault,
        Some(UntrustedChatValueV1::Boolean(true)) => ThinkingModeV1::Enabled,
        Some(UntrustedChatValueV1::Boolean(false)) => ThinkingModeV1::Disabled,
        Some(_) => return Err(ChatRenderError::InvalidThinkingMode),
    };

    let mut messages = Vec::with_capacity(request.messages.len());
    for (index, message) in request.messages.into_iter().enumerate() {
        if message.tool_call.is_some() || message.tool_calls.is_some() {
            return Err(ChatRenderError::UnsupportedToolInput { index: Some(index) });
        }
        if message.image.is_some() || message.image_url.is_some() {
            return Err(ChatRenderError::UnsupportedImageInput { index });
        }
        if message.video.is_some() {
            return Err(ChatRenderError::UnsupportedVideoInput { index });
        }
        if !message.unknown_fields.is_empty() {
            return Err(ChatRenderError::UnknownMessageField { index });
        }

        let role = value_string(message.role, index, ChatFieldV1::Role)?;
        let content = value_string(message.content, index, ChatFieldV1::Content)?;
        let reasoning_content = message
            .reasoning_content
            .map(|value| value_string(value, index, ChatFieldV1::ReasoningContent))
            .transpose()?;
        let typed = match role.as_str() {
            "system" => {
                if reasoning_content.is_some() {
                    return Err(ChatRenderError::ReasoningContentOnNonAssistant { index });
                }
                Qwen35ChatMessageV1::System { content }
            }
            "user" => {
                if reasoning_content.is_some() {
                    return Err(ChatRenderError::ReasoningContentOnNonAssistant { index });
                }
                Qwen35ChatMessageV1::User { content }
            }
            "assistant" => Qwen35ChatMessageV1::Assistant {
                content,
                reasoning_content,
            },
            _ => return Err(ChatRenderError::UnsupportedRole { index }),
        };
        messages.push(typed);
    }

    validate_typed_messages(&messages)?;
    Ok((
        messages,
        Qwen35RenderOptionsV1 {
            add_generation_prompt,
            thinking,
        },
    ))
}

fn value_string(
    value: UntrustedChatValueV1,
    index: usize,
    field: ChatFieldV1,
) -> Result<String, ChatRenderError> {
    match value {
        UntrustedChatValueV1::Missing => Err(ChatRenderError::MissingMessageField { index, field }),
        UntrustedChatValueV1::StringBytes(bytes) => String::from_utf8(bytes)
            .map_err(|_| ChatRenderError::InvalidMessageUtf8 { index, field }),
        _ if field == ChatFieldV1::Role => Err(ChatRenderError::InvalidRoleType { index }),
        _ if field == ChatFieldV1::Content => Err(ChatRenderError::InvalidContentType { index }),
        _ => Err(ChatRenderError::InvalidReasoningContentType { index }),
    }
}

fn validate_typed_messages(messages: &[Qwen35ChatMessageV1]) -> Result<usize, ChatRenderError> {
    if messages.is_empty() {
        return Err(ChatRenderError::EmptyMessages);
    }

    let mut system_indices = Vec::new();
    let mut last_user = None;
    for (index, message) in messages.iter().enumerate() {
        match message {
            Qwen35ChatMessageV1::System { .. } => system_indices.push(index),
            Qwen35ChatMessageV1::User { content } => {
                if is_tool_response_shaped(trim_qwen(content)) {
                    return Err(ChatRenderError::ToolResponseUserContent { index });
                }
                last_user = Some(index);
            }
            Qwen35ChatMessageV1::Assistant { .. } => {}
        }
    }
    if system_indices.len() > 1 {
        return Err(ChatRenderError::MultipleSystemMessages);
    }
    if let Some(&index) = system_indices.first() {
        if index != 0 {
            return Err(ChatRenderError::MisplacedSystemMessage { index });
        }
    }
    last_user.ok_or(ChatRenderError::NoOrdinaryUserMessage)
}

fn is_tool_response_shaped(content: &str) -> bool {
    content.starts_with("<tool_response>") && content.ends_with("</tool_response>")
}

fn trim_qwen(value: &str) -> &str {
    value.trim_matches(is_qwen_trim_character)
}

fn is_qwen_trim_character(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'..='\u{000d}'
            | '\u{001c}'..='\u{0020}'
            | '\u{0085}'
            | '\u{00a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

fn assistant_parts<'a>(content: &'a str, reasoning_content: Option<&'a str>) -> (&'a str, &'a str) {
    let content = trim_qwen(content);
    if let Some(reasoning) = reasoning_content {
        return (trim_qwen(reasoning), content);
    }
    let Some(first_close) = content.find(THINK_CLOSE) else {
        return ("", content);
    };
    let prefix = &content[..first_close];
    let reasoning = prefix
        .rfind(THINK_OPEN)
        .map_or(prefix, |open| &prefix[open + THINK_OPEN.len()..]);
    let last_close = content
        .rfind(THINK_CLOSE)
        .expect("first closing tag guarantees a last closing tag");
    let answer = content[last_close + THINK_CLOSE.len()..].trim_start_matches('\n');
    (trim_qwen(reasoning), answer)
}

fn checked_fragment(
    total: &mut usize,
    fragment: &str,
    limit: usize,
) -> Result<(), ChatRenderError> {
    *total = total
        .checked_add(fragment.len())
        .ok_or(ChatRenderError::OutputTooLarge { limit_bytes: limit })?;
    if *total > limit {
        return Err(ChatRenderError::OutputTooLarge { limit_bytes: limit });
    }
    Ok(())
}

fn visit_fragments(
    messages: &[Qwen35ChatMessageV1],
    options: Qwen35RenderOptionsV1,
    last_user: usize,
    mut visit: impl FnMut(&str),
) {
    for (index, message) in messages.iter().enumerate() {
        visit(IM_START);
        match message {
            Qwen35ChatMessageV1::System { content } => {
                visit("system\n");
                visit(trim_qwen(content));
            }
            Qwen35ChatMessageV1::User { content } => {
                visit("user\n");
                visit(trim_qwen(content));
            }
            Qwen35ChatMessageV1::Assistant {
                content,
                reasoning_content,
            } => {
                visit("assistant\n");
                let (reasoning, answer) = assistant_parts(content, reasoning_content.as_deref());
                if index < last_user {
                    visit(answer);
                } else {
                    visit("<think>\n");
                    visit(reasoning);
                    visit("\n</think>\n\n");
                    visit(answer);
                }
            }
        }
        visit(IM_END_LINE);
    }

    if options.add_generation_prompt {
        match options.thinking {
            ThinkingModeV1::TemplateDefault | ThinkingModeV1::Enabled => {
                visit(GENERATION_THINKING);
            }
            ThinkingModeV1::Disabled => visit(GENERATION_DISABLED),
        }
    }
}

fn plan_output(
    messages: &[Qwen35ChatMessageV1],
    options: Qwen35RenderOptionsV1,
    last_user: usize,
    limit: usize,
) -> Result<usize, ChatRenderError> {
    let mut total = 0usize;
    let mut result = Ok(());
    visit_fragments(messages, options, last_user, |fragment| {
        if result.is_ok() {
            result = checked_fragment(&mut total, fragment, limit);
        }
    });
    result.map(|()| total)
}

fn write_output(
    output: &mut String,
    messages: &[Qwen35ChatMessageV1],
    options: Qwen35RenderOptionsV1,
    last_user: usize,
) {
    visit_fragments(messages, options, last_user, |fragment| {
        output.push_str(fragment);
    });
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use sllm_core::{
        ModelLock, VerifiedCache, fingerprint_for_json, parse_model_lock, verify_model_cache,
    };
    use tokenizers::Tokenizer;

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct Fixture {
        cache: VerifiedCache,
        lock: ModelLock,
        directory: TestDirectory,
    }

    fn repository_path(relative: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative)
    }

    fn test_directory() -> TestDirectory {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sllm-chat-unit-positive-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create chat unit test directory");
        TestDirectory(path)
    }

    fn replace_once(source: String, old: &str, new: &str) -> String {
        assert_eq!(source.matches(old).count(), 1, "replacement must be unique");
        source.replacen(old, new, 1)
    }

    fn synthetic_verified_fixture() -> Fixture {
        let template_bytes = vec![b' '; QWEN35_CHAT_TEMPLATE_SIZE_BYTES as usize];
        let directory = test_directory();
        let base_cache = repository_path("ci/fixtures/model-lock-v1/cache");
        for entry in fs::read_dir(base_cache).expect("read base cache") {
            let entry = entry.expect("read base cache entry");
            fs::copy(entry.path(), directory.0.join(entry.file_name())).expect("copy cache file");
        }
        fs::write(
            directory.0.join(QWEN35_CHAT_TEMPLATE_FILENAME),
            &template_bytes,
        )
        .expect("write synthetic bounded template carrier");

        let source = String::from_utf8(
            fs::read(repository_path("ci/fixtures/model-lock-v1/lock.json"))
                .expect("base lock exists"),
        )
        .expect("base lock is UTF-8");
        let source = replace_once(
            source,
            r#"        "path": "chat_template.jinja",
        "size_bytes": 44,
        "sha256": "00458c8b559de6bbd4c15a4d6ca59b56015d25f95ca5ff29e7f5eae1d8dee31f","#,
            &format!(
                "        \"path\": \"chat_template.jinja\",\n        \"size_bytes\": {},\n        \"sha256\": \"{}\",",
                template_bytes.len(),
                sha256_hex(&template_bytes)
            ),
        );
        let fingerprint =
            fingerprint_for_json(source.as_bytes()).expect("recompute fixture fingerprint");
        let source = replace_once(
            source,
            "  \"fingerprint\": \"sha256:1065d7427922cf9f2e37c18e18b7434b7c3e63cda0dc75236273050170292415\",",
            &format!("  \"fingerprint\": \"{fingerprint}\","),
        );
        let mut lock = parse_model_lock(source.as_bytes()).expect("synthetic lock parses");
        let mut cache = verify_model_cache(&lock, &directory.0).expect("synthetic cache verifies");

        // Keep production identity and metadata fixed while allowing the
        // private test seam to prove the same bounded-read success path with
        // synthetic bytes.
        lock.model.repo_id = QWEN35_REPO_ID.to_owned();
        lock.model.resolved_revision = QWEN35_RESOLVED_REVISION.to_owned();
        lock.model
            .files
            .iter_mut()
            .find(|file| file.path == QWEN35_CHAT_TEMPLATE_FILENAME)
            .expect("chat lock entry")
            .sha256 = QWEN35_CHAT_TEMPLATE_SHA256.to_owned();
        cache
            .files
            .iter_mut()
            .find(|file| file.path == QWEN35_CHAT_TEMPLATE_FILENAME)
            .expect("chat cache entry")
            .sha256 = QWEN35_CHAT_TEMPLATE_SHA256.to_owned();

        Fixture {
            cache,
            lock,
            directory,
        }
    }

    fn renderer() -> Qwen35ChatTemplateV1 {
        let fixture = synthetic_verified_fixture();
        let template = fs::read(fixture.directory.0.join(QWEN35_CHAT_TEMPLATE_FILENAME))
            .expect("read synthetic template");
        Qwen35ChatTemplateV1::from_verified_cache_with_test_digest(
            &fixture.lock,
            &fixture.cache,
            &sha256_hex(&template),
        )
        .expect("synthetic fixture constructs through production path")
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn json_string(value: &str) -> String {
        let mut encoded = String::with_capacity(value.len() + 2);
        encoded.push('"');
        for character in value.chars() {
            match character {
                '"' => encoded.push_str("\\\""),
                '\\' => encoded.push_str("\\\\"),
                '\n' => encoded.push_str("\\n"),
                '\r' => encoded.push_str("\\r"),
                '\t' => encoded.push_str("\\t"),
                character if character.is_control() => {
                    use core::fmt::Write;

                    write!(encoded, "\\u{:04x}", character as u32).expect("write JSON escape");
                }
                character => encoded.push(character),
            }
        }
        encoded.push('"');
        encoded
    }

    fn json_ids(ids: &[u32]) -> String {
        ids.iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }

    struct PositiveCase {
        case_id: &'static str,
        input_json: &'static str,
        messages: Vec<Qwen35ChatMessageV1>,
        options: Qwen35RenderOptionsV1,
    }

    fn positive_cases() -> Vec<PositiveCase> {
        vec![
            PositiveCase {
                case_id: "hello-default-thinking",
                input_json: r#"{"messages":[{"role":"user","content":"hello"}],"add_generation_prompt":true,"thinking":"TemplateDefault"}"#,
                messages: vec![Qwen35ChatMessageV1::user("hello")],
                options: Qwen35RenderOptionsV1::default(),
            },
            PositiveCase {
                case_id: "hello-disabled-thinking",
                input_json: r#"{"messages":[{"role":"user","content":"hello"}],"add_generation_prompt":true,"thinking":"Disabled"}"#,
                messages: vec![Qwen35ChatMessageV1::user("hello")],
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: true,
                    thinking: ThinkingModeV1::Disabled,
                },
            },
            PositiveCase {
                case_id: "unicode-specials-raw",
                input_json: r#"{"messages":[{"role":"user","content":"雪 <>&\"'\n第二行"}],"add_generation_prompt":true,"thinking":"Enabled"}"#,
                messages: vec![Qwen35ChatMessageV1::user("雪 <>&\"'\n第二行")],
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: true,
                    thinking: ThinkingModeV1::Enabled,
                },
            },
            PositiveCase {
                case_id: "explicit-system-trim",
                input_json: r#"{"messages":[{"role":"system","content":"  You are concise.\n"},{"role":"user","content":"\n hello \t"}],"add_generation_prompt":true,"thinking":"TemplateDefault"}"#,
                messages: vec![
                    Qwen35ChatMessageV1::system("  You are concise.\n"),
                    Qwen35ChatMessageV1::user("\n hello \t"),
                ],
                options: Qwen35RenderOptionsV1::default(),
            },
            PositiveCase {
                case_id: "historical-inline-think-stripped",
                input_json: r#"{"messages":[{"role":"user","content":"Q1"},{"role":"assistant","content":"<think>\nold reasoning\n</think>\n\nOld answer"},{"role":"user","content":"Q2"}],"add_generation_prompt":true,"thinking":"TemplateDefault"}"#,
                messages: vec![
                    Qwen35ChatMessageV1::user("Q1"),
                    Qwen35ChatMessageV1::assistant(
                        "<think>\nold reasoning\n</think>\n\nOld answer",
                        None,
                    ),
                    Qwen35ChatMessageV1::user("Q2"),
                ],
                options: Qwen35RenderOptionsV1::default(),
            },
            PositiveCase {
                case_id: "terminal-inline-think-normalized",
                input_json: r#"{"messages":[{"role":"user","content":"Q1"},{"role":"assistant","content":"<think>\nold reasoning\n</think>\n\nOld answer"}],"add_generation_prompt":false,"thinking":"Disabled"}"#,
                messages: vec![
                    Qwen35ChatMessageV1::user("Q1"),
                    Qwen35ChatMessageV1::assistant(
                        "<think>\nold reasoning\n</think>\n\nOld answer",
                        None,
                    ),
                ],
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: false,
                    thinking: ThinkingModeV1::Disabled,
                },
            },
            PositiveCase {
                case_id: "assistant-reasoning-content",
                input_json: r#"{"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"Answer <think>raw</think>","reasoning_content":"  explicit reason\n"}],"add_generation_prompt":false,"thinking":"TemplateDefault"}"#,
                messages: vec![
                    Qwen35ChatMessageV1::user("Q"),
                    Qwen35ChatMessageV1::assistant(
                        "Answer <think>raw</think>",
                        Some("  explicit reason\n".to_owned()),
                    ),
                ],
                options: Qwen35RenderOptionsV1 {
                    add_generation_prompt: false,
                    thinking: ThinkingModeV1::TemplateDefault,
                },
            },
            PositiveCase {
                case_id: "consecutive-users-supported",
                input_json: r#"{"messages":[{"role":"user","content":"A"},{"role":"user","content":"B"}],"add_generation_prompt":true,"thinking":"TemplateDefault"}"#,
                messages: vec![
                    Qwen35ChatMessageV1::user("A"),
                    Qwen35ChatMessageV1::user("B"),
                ],
                options: Qwen35RenderOptionsV1::default(),
            },
        ]
    }

    fn positive_manifest() -> String {
        let renderer = renderer();
        let tokenizer = Tokenizer::from_bytes(include_bytes!(
            "../../../ci/fixtures/chat-template-v1/tokenizer.json"
        ))
        .expect("B3-derived tiny tokenizer loads");
        let cases = positive_cases()
            .into_iter()
            .map(|case| {
                let output = renderer
                    .render(&case.messages, case.options)
                    .expect("positive case renders");
                let token_ids = tokenizer
                    .encode(output.clone(), false)
                    .expect("positive output tokenizes")
                    .get_ids()
                    .to_vec();
                format!(
                    "    {{\n      \"case_id\": {},\n      \"renderer_version\": {},\n      \"template\": {{\"kind\": \"ChatTemplateJinja\", \"filename\": {}, \"size_bytes\": {}, \"sha256\": {}}},\n      \"input\": {},\n      \"output\": {},\n      \"output_sha256\": {},\n      \"token_ids\": [{}]\n    }}",
                    json_string(case.case_id),
                    QWEN35_CHAT_RENDERER_VERSION,
                    json_string(QWEN35_CHAT_TEMPLATE_FILENAME),
                    QWEN35_CHAT_TEMPLATE_SIZE_BYTES,
                    json_string(QWEN35_CHAT_TEMPLATE_SHA256),
                    case.input_json,
                    json_string(&output),
                    json_string(&sha256_hex(output.as_bytes())),
                    json_ids(&token_ids),
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            "{{\n  \"fixture_version\": \"chat-template-v1\",\n  \"tokenizer_fixture\": \"B3 tokenizer-v1 byte-identical copy\",\n  \"positive_cases\": [\n{cases}\n  ]\n}}\n"
        )
    }

    #[test]
    fn digest_comparison_accepts_a_synthetic_known_digest() {
        assert!(template_digest_matches(
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        ));
        assert!(!template_digest_matches(
            b"abd",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        ));
    }

    #[test]
    fn digest_validation_precedes_utf8_validation() {
        let invalid_utf8 = [0xff];
        assert_eq!(
            validate_template_bytes(&invalid_utf8, &sha256_hex(&invalid_utf8)),
            Err(ChatRenderError::TemplateAssetInvalidUtf8)
        );
        assert_eq!(
            validate_template_bytes(&invalid_utf8, &"0".repeat(64)),
            Err(ChatRenderError::UnsupportedTemplateIdentity)
        );
    }

    #[test]
    fn authoritative_positive_manifest_is_exact() {
        assert_eq!(
            include_bytes!("../../../ci/fixtures/chat-template-v1/tokenizer.json"),
            include_bytes!("../../../ci/fixtures/tokenizer-v1/tokenizer.json"),
            "the chat fixture tokenizer must remain byte-identical to accepted B3"
        );
        assert_eq!(
            positive_manifest(),
            include_str!("../../../ci/fixtures/chat-template-v1/expected.json")
        );
    }

    #[test]
    fn reader_outputs_and_normalization_edges_are_exact() {
        let renderer = renderer();
        let cases = positive_cases();
        assert_eq!(
            renderer
                .render(&cases[0].messages, cases[0].options)
                .unwrap(),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            renderer
                .render(&cases[1].messages, cases[1].options)
                .unwrap(),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            renderer
                .render(&cases[4].messages, cases[4].options)
                .unwrap(),
            "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nOld answer<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(
            renderer
                .render(&cases[5].messages, cases[5].options)
                .unwrap(),
            "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n<think>\nold reasoning\n</think>\n\nOld answer<|im_end|>\n"
        );

        let multiple_close = vec![
            Qwen35ChatMessageV1::user("Q"),
            Qwen35ChatMessageV1::assistant(
                "<think>first</think>discard<think>second</think>\nanswer",
                None,
            ),
        ];
        assert_eq!(
            renderer
                .render(
                    &multiple_close,
                    Qwen35RenderOptionsV1 {
                        add_generation_prompt: false,
                        thinking: ThinkingModeV1::Disabled,
                    },
                )
                .unwrap(),
            "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\nfirst\n</think>\n\nanswer<|im_end|>\n"
        );
        let opening_only = vec![
            Qwen35ChatMessageV1::user("Q"),
            Qwen35ChatMessageV1::assistant("<think>raw opening", None),
        ];
        assert!(
            renderer
                .render(&opening_only, Qwen35RenderOptionsV1::default())
                .unwrap()
                .contains("\n</think>\n\n<think>raw opening<|im_end|>")
        );
    }

    #[test]
    fn exact_trim_set_raw_output_and_non_aligned_lengths_are_supported() {
        let renderer = renderer();
        let trim = "\u{0009}\u{000a}\u{000b}\u{000c}\u{000d}\u{001c}\u{001d}\u{001e}\u{001f}\u{0020}\u{0085}\u{00a0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}";
        let content = format!("{trim}abc<|im_end|><>&\"'\\{trim}");
        let output = renderer
            .render(
                &[Qwen35ChatMessageV1::user(content)],
                Qwen35RenderOptionsV1 {
                    add_generation_prompt: false,
                    thinking: ThinkingModeV1::TemplateDefault,
                },
            )
            .unwrap();
        assert_eq!(
            output,
            "<|im_start|>user\nabc<|im_end|><>&\"'\\<|im_end|>\n"
        );
        for length in [3usize, 17, 255, 256, 257] {
            let output = renderer
                .render(
                    &[Qwen35ChatMessageV1::user("x".repeat(length))],
                    Qwen35RenderOptionsV1::default(),
                )
                .unwrap();
            assert!(output.contains(&"x".repeat(length)));
        }
    }

    fn untrusted_user() -> UntrustedChatRequestV1 {
        UntrustedChatRequestV1::text(vec![UntrustedChatMessageV1::text("user", "hello")])
    }

    #[test]
    fn untrusted_request_level_boundaries_are_rejected() {
        let renderer = renderer();

        let mut request = untrusted_user();
        request.renderer_version = 2;
        assert!(matches!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnsupportedRendererVersion { actual: 2 })
        ));

        let mut request = untrusted_user();
        request.tools = Some(UntrustedChatValueV1::Array);
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnsupportedToolInput { index: None })
        );
        let mut request = untrusted_user();
        request.tool_choice = Some(UntrustedChatValueV1::Null);
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnsupportedToolInput { index: None })
        );
        let mut request = untrusted_user();
        request.unknown_fields.push("future".to_owned());
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnknownRequestField)
        );
        let mut request = untrusted_user();
        request.messages.clear();
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::EmptyMessages)
        );
        let mut request = untrusted_user();
        request.add_generation_prompt = UntrustedChatValueV1::Null;
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::InvalidAddGenerationPrompt)
        );
        let mut request = untrusted_user();
        request.enable_thinking = Some(UntrustedChatValueV1::string("false"));
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::InvalidThinkingMode)
        );
    }

    #[test]
    fn untrusted_role_content_and_field_boundaries_are_rejected() {
        let renderer = renderer();

        let mut cases = Vec::new();
        let mut request = untrusted_user();
        request.messages[0].role = UntrustedChatValueV1::Null;
        cases.push((request, ChatRenderError::InvalidRoleType { index: 0 }));
        let mut request = untrusted_user();
        request.messages[0].role = UntrustedChatValueV1::StringBytes(vec![0xff]);
        cases.push((
            request,
            ChatRenderError::InvalidMessageUtf8 {
                index: 0,
                field: ChatFieldV1::Role,
            },
        ));
        let mut request = untrusted_user();
        request.messages[0].role = UntrustedChatValueV1::string("developer");
        cases.push((request, ChatRenderError::UnsupportedRole { index: 0 }));
        let mut request = untrusted_user();
        request.messages[0].role = UntrustedChatValueV1::string("tool");
        cases.push((request, ChatRenderError::UnsupportedRole { index: 0 }));
        let mut request = untrusted_user();
        request.messages[0].role = UntrustedChatValueV1::Missing;
        cases.push((
            request,
            ChatRenderError::MissingMessageField {
                index: 0,
                field: ChatFieldV1::Role,
            },
        ));
        let mut request = untrusted_user();
        request.messages[0].content = UntrustedChatValueV1::Missing;
        cases.push((
            request,
            ChatRenderError::MissingMessageField {
                index: 0,
                field: ChatFieldV1::Content,
            },
        ));
        let mut request = untrusted_user();
        request.messages[0].content = UntrustedChatValueV1::StringBytes(vec![0xff]);
        cases.push((
            request,
            ChatRenderError::InvalidMessageUtf8 {
                index: 0,
                field: ChatFieldV1::Content,
            },
        ));
        for value in [
            UntrustedChatValueV1::Null,
            UntrustedChatValueV1::Boolean(false),
            UntrustedChatValueV1::Number("1".to_owned()),
            UntrustedChatValueV1::Array,
            UntrustedChatValueV1::Object,
        ] {
            let mut request = untrusted_user();
            request.messages[0].content = value;
            cases.push((request, ChatRenderError::InvalidContentType { index: 0 }));
        }
        for (request, expected) in cases {
            assert_eq!(renderer.render_untrusted(request), Err(expected));
        }

        let mut request = untrusted_user();
        request.messages[0].reasoning_content = Some(UntrustedChatValueV1::Null);
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::InvalidReasoningContentType { index: 0 })
        );
        let mut request = untrusted_user();
        request.messages[0].reasoning_content = Some(UntrustedChatValueV1::StringBytes(vec![0xff]));
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::InvalidMessageUtf8 {
                index: 0,
                field: ChatFieldV1::ReasoningContent,
            })
        );
        let mut request = untrusted_user();
        request.messages[0].reasoning_content = Some(UntrustedChatValueV1::string("reason"));
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::ReasoningContentOnNonAssistant { index: 0 })
        );
        let mut request = untrusted_user();
        request.messages[0].tool_call = Some(UntrustedChatValueV1::Null);
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnsupportedToolInput { index: Some(0) })
        );
        let mut request = untrusted_user();
        request.messages[0].tool_calls = Some(UntrustedChatValueV1::Array);
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnsupportedToolInput { index: Some(0) })
        );
        let mut request = untrusted_user();
        request.messages[0].image = Some(UntrustedChatValueV1::Object);
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnsupportedImageInput { index: 0 })
        );
        let mut request = untrusted_user();
        request.messages[0].image_url = Some(UntrustedChatValueV1::string("x"));
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnsupportedImageInput { index: 0 })
        );
        let mut request = untrusted_user();
        request.messages[0].video = Some(UntrustedChatValueV1::Object);
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnsupportedVideoInput { index: 0 })
        );
        let mut request = untrusted_user();
        request.messages[0].unknown_fields.push("future".to_owned());
        assert_eq!(
            renderer.render_untrusted(request),
            Err(ChatRenderError::UnknownMessageField { index: 0 })
        );
    }

    #[test]
    fn placement_tool_response_and_user_presence_fail_closed() {
        let renderer = renderer();
        assert_eq!(
            renderer.render(&[], Qwen35RenderOptionsV1::default()),
            Err(ChatRenderError::EmptyMessages)
        );
        assert_eq!(
            renderer.render(
                &[
                    Qwen35ChatMessageV1::user("Q"),
                    Qwen35ChatMessageV1::system("late"),
                ],
                Qwen35RenderOptionsV1::default(),
            ),
            Err(ChatRenderError::MisplacedSystemMessage { index: 1 })
        );
        assert_eq!(
            renderer.render(
                &[
                    Qwen35ChatMessageV1::system("one"),
                    Qwen35ChatMessageV1::system("two"),
                    Qwen35ChatMessageV1::user("Q"),
                ],
                Qwen35RenderOptionsV1::default(),
            ),
            Err(ChatRenderError::MultipleSystemMessages)
        );
        assert_eq!(
            renderer.render(
                &[Qwen35ChatMessageV1::user(
                    "  <tool_response>data</tool_response>\n",
                )],
                Qwen35RenderOptionsV1::default(),
            ),
            Err(ChatRenderError::ToolResponseUserContent { index: 0 })
        );
        assert_eq!(
            renderer.render(
                &[Qwen35ChatMessageV1::assistant("answer", None)],
                Qwen35RenderOptionsV1::default(),
            ),
            Err(ChatRenderError::NoOrdinaryUserMessage)
        );
    }

    #[test]
    fn output_cap_is_checked_before_any_output_is_returned() {
        let renderer = renderer();
        let messages = [Qwen35ChatMessageV1::user("abc")];
        let exact = renderer
            .render_with_output_limit(&messages, Qwen35RenderOptionsV1::default(), 128)
            .unwrap();
        assert_eq!(
            renderer
                .render_with_output_limit(&messages, Qwen35RenderOptionsV1::default(), exact.len(),)
                .unwrap(),
            exact
        );
        assert_eq!(
            renderer.render_with_output_limit(
                &messages,
                Qwen35RenderOptionsV1::default(),
                exact.len() - 1,
            ),
            Err(ChatRenderError::OutputTooLarge {
                limit_bytes: exact.len() - 1,
            })
        );
        assert_eq!(
            renderer.render_with_output_limit(
                &messages,
                Qwen35RenderOptionsV1::default(),
                QWEN35_CHAT_MAX_OUTPUT_BYTES + 1,
            ),
            Err(ChatRenderError::OutputLimitExceedsHostCap)
        );
    }

    #[test]
    fn output_cap_boundaries_use_actual_rendered_lengths() {
        let renderer = renderer();
        let options = Qwen35RenderOptionsV1::default();
        let framing_bytes = renderer
            .render(&[Qwen35ChatMessageV1::user("")], options)
            .expect("empty user framing renders")
            .len();
        assert!(framing_bytes < QWEN35_CHAT_MAX_OUTPUT_BYTES);
        let exact_content_bytes = QWEN35_CHAT_MAX_OUTPUT_BYTES - framing_bytes;

        let below = {
            let messages = [Qwen35ChatMessageV1::user(
                "x".repeat(exact_content_bytes - 1),
            )];
            renderer
                .render_with_output_limit(&messages, options, QWEN35_CHAT_MAX_OUTPUT_BYTES)
                .expect("max-minus-one rendered bytes fit")
        };
        assert_eq!(below.len(), QWEN35_CHAT_MAX_OUTPUT_BYTES - 1);
        drop(below);

        let exact = {
            let messages = [Qwen35ChatMessageV1::user("x".repeat(exact_content_bytes))];
            renderer
                .render_with_output_limit(&messages, options, QWEN35_CHAT_MAX_OUTPUT_BYTES)
                .expect("exact maximum rendered bytes fit")
        };
        assert_eq!(exact.len(), QWEN35_CHAT_MAX_OUTPUT_BYTES);
        drop(exact);

        let too_large = {
            let messages = [Qwen35ChatMessageV1::user(
                "x".repeat(exact_content_bytes + 1),
            )];
            renderer.render_with_output_limit(&messages, options, QWEN35_CHAT_MAX_OUTPUT_BYTES)
        };
        assert_eq!(
            too_large,
            Err(ChatRenderError::OutputTooLarge {
                limit_bytes: QWEN35_CHAT_MAX_OUTPUT_BYTES,
            })
        );
    }
}
