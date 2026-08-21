//! Bounded, transport-independent state for the Phase 44 interactive CLI.
//!
//! This module deliberately does not own model execution or checkpoint state
//! planes.  It validates terminal/file input, tracks a typed transcript, and
//! detects reverse prompts.  The production adapter commits a turn only after
//! generation succeeds and stores [`InteractiveTranscriptV1::encode`] in the
//! existing Phase 41 checkpoint conversation section.

use std::fmt;
use std::fs::OpenOptions;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;
use serde_json::json;
use sllm_frontend::Qwen35ChatMessageV1;

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

pub(crate) const MAX_PROMPT_FILE_BYTES_V1: usize = 16 * 1024 * 1024;
pub(crate) const MAX_INTERACTIVE_MESSAGES_V1: usize = 1_024;
pub(crate) const MAX_INTERACTIVE_TRANSCRIPT_BYTES_V1: usize = 16 * 1024 * 1024;
pub(crate) const MAX_REVERSE_PROMPTS_V1: usize = 4;
pub(crate) const MAX_REVERSE_PROMPT_BYTES_V1: usize = 1024 * 1024;
const TRANSCRIPT_SCHEMA_V1: &str = "sllm-interactive-transcript-v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum InteractiveErrorV1 {
    PromptSourceConflict,
    PromptFileUnavailable,
    PromptFileNotRegular,
    PromptFileChanged,
    PromptFileTooLarge,
    PromptFileInvalidUtf8,
    PromptFileContainsNul,
    EmptyPrompt,
    InvalidTranscript,
    TranscriptTooLarge,
    TooManyMessages,
    InvalidRole,
    EmptyMessage,
    MessageContainsNul,
    ReasoningContentOnNonAssistant,
    ReasoningContentContainsNul,
    TooManyReversePrompts,
    EmptyReversePrompt,
    DuplicateReversePrompt,
    ReversePromptContainsNul,
    ReversePromptsTooLarge,
}

impl fmt::Display for InteractiveErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PromptSourceConflict => "interactive prompt sources are mutually exclusive",
            Self::PromptFileUnavailable => "prompt file is unavailable",
            Self::PromptFileNotRegular => "prompt file must be a regular non-symlink file",
            Self::PromptFileChanged => "prompt file changed while it was being read",
            Self::PromptFileTooLarge => "prompt file exceeds the 16 MiB limit",
            Self::PromptFileInvalidUtf8 => "prompt file is not valid UTF-8",
            Self::PromptFileContainsNul => "prompt file must not contain NUL bytes",
            Self::EmptyPrompt => "prompt input must not be empty",
            Self::InvalidTranscript => "interactive transcript is malformed",
            Self::TranscriptTooLarge => "interactive transcript exceeds the 16 MiB limit",
            Self::TooManyMessages => "interactive transcript exceeds 1024 messages",
            Self::InvalidRole => "interactive message role is unsupported",
            Self::EmptyMessage => "interactive message content must not be empty",
            Self::MessageContainsNul => "interactive message content must not contain NUL bytes",
            Self::ReasoningContentOnNonAssistant => {
                "reasoning_content is only valid for assistant messages"
            }
            Self::ReasoningContentContainsNul => {
                "interactive reasoning_content must not contain NUL bytes"
            }
            Self::TooManyReversePrompts => "at most four reverse prompts are accepted",
            Self::EmptyReversePrompt => "reverse prompt must not be empty",
            Self::DuplicateReversePrompt => "reverse prompts must be unique",
            Self::ReversePromptContainsNul => "reverse prompt must not contain NUL bytes",
            Self::ReversePromptsTooLarge => "reverse prompts exceed the 1 MiB aggregate limit",
        })
    }
}

impl std::error::Error for InteractiveErrorV1 {}

/// Closed prompt source selection.  File contents and terminal input are never
/// implicitly concatenated with command-line prompt/message input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PromptSourceKindV1 {
    Prompt,
    Messages,
    PromptFile,
    InteractiveStdin,
}

impl PromptSourceKindV1 {
    pub(crate) fn select(
        prompt: bool,
        messages: bool,
        prompt_file: bool,
        interactive_stdin: bool,
    ) -> Result<Self, InteractiveErrorV1> {
        match (prompt, messages, prompt_file, interactive_stdin) {
            (true, false, false, false) => Ok(Self::Prompt),
            (false, true, false, false) => Ok(Self::Messages),
            (false, false, true, false) => Ok(Self::PromptFile),
            (false, false, false, true) => Ok(Self::InteractiveStdin),
            _ => Err(InteractiveErrorV1::PromptSourceConflict),
        }
    }
}

/// Read one explicitly selected prompt file without following a symlink or
/// accepting a special file.  The second metadata check closes replacement and
/// truncation races before bytes are returned to the tokenizer.
pub(crate) fn read_prompt_file_v1(path: &Path) -> Result<String, InteractiveErrorV1> {
    let path_metadata = path
        .symlink_metadata()
        .map_err(|_| InteractiveErrorV1::PromptFileUnavailable)?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(InteractiveErrorV1::PromptFileNotRegular);
    }
    if path_metadata.len() > MAX_PROMPT_FILE_BYTES_V1 as u64 {
        return Err(InteractiveErrorV1::PromptFileTooLarge);
    }

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let mut file = options
        .open(path)
        .map_err(|_| InteractiveErrorV1::PromptFileUnavailable)?;
    let opened_metadata = file
        .metadata()
        .map_err(|_| InteractiveErrorV1::PromptFileUnavailable)?;
    if !opened_metadata.is_file() || !same_file(&path_metadata, &opened_metadata) {
        return Err(InteractiveErrorV1::PromptFileChanged);
    }

    let mut bytes = Vec::with_capacity(
        usize::try_from(opened_metadata.len())
            .unwrap_or(MAX_PROMPT_FILE_BYTES_V1)
            .min(MAX_PROMPT_FILE_BYTES_V1),
    );
    file.by_ref()
        .take((MAX_PROMPT_FILE_BYTES_V1 as u64) + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| InteractiveErrorV1::PromptFileUnavailable)?;
    if bytes.len() > MAX_PROMPT_FILE_BYTES_V1 {
        return Err(InteractiveErrorV1::PromptFileTooLarge);
    }
    let final_metadata = file
        .metadata()
        .map_err(|_| InteractiveErrorV1::PromptFileUnavailable)?;
    if !same_file(&opened_metadata, &final_metadata)
        || final_metadata.len() != opened_metadata.len()
        || final_metadata.len() != bytes.len() as u64
    {
        return Err(InteractiveErrorV1::PromptFileChanged);
    }
    let prompt = String::from_utf8(bytes).map_err(|_| InteractiveErrorV1::PromptFileInvalidUtf8)?;
    if prompt.contains('\0') {
        return Err(InteractiveErrorV1::PromptFileContainsNul);
    }
    if prompt.is_empty() {
        return Err(InteractiveErrorV1::EmptyPrompt);
    }
    Ok(prompt)
}

#[cfg(unix)]
fn same_file(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    left.len() == right.len()
        && left.permissions().readonly() == right.permissions().readonly()
        && left.modified().ok() == right.modified().ok()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InteractiveRoleV1 {
    System,
    User,
    Assistant,
}

impl InteractiveRoleV1 {
    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    fn parse(value: &str) -> Result<Self, InteractiveErrorV1> {
        match value {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            _ => Err(InteractiveErrorV1::InvalidRole),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InteractiveMessageV1 {
    role: InteractiveRoleV1,
    content: String,
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTranscriptV1 {
    schema_version: String,
    messages: Vec<WireTranscriptMessageV1>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTranscriptMessageV1 {
    role: String,
    content: String,
    /// The field was added after the original transcript wire shape.  A
    /// missing field is intentionally decoded as `None` for old checkpoints.
    #[serde(default)]
    reasoning_content: Option<String>,
}

/// Canonical typed conversation persisted in the Phase 41 checkpoint's
/// bounded `conversation` bytes.  No native state plane is interpreted here.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct InteractiveTranscriptV1 {
    messages: Vec<InteractiveMessageV1>,
}

impl InteractiveTranscriptV1 {
    pub(crate) fn new(system: Option<&str>) -> Result<Self, InteractiveErrorV1> {
        let mut transcript = Self::default();
        if let Some(system) = system {
            transcript.push(InteractiveRoleV1::System, system)?;
        }
        Ok(transcript)
    }

    pub(crate) fn begin_turn<'a>(&'a mut self, user: &str) -> InteractiveTurnV1<'a> {
        InteractiveTurnV1 {
            transcript: self,
            user: user.to_owned(),
        }
    }

    pub(crate) fn messages(&self) -> Vec<Qwen35ChatMessageV1> {
        self.messages
            .iter()
            .map(|message| match message.role {
                InteractiveRoleV1::System => Qwen35ChatMessageV1::system(message.content.clone()),
                InteractiveRoleV1::User => Qwen35ChatMessageV1::user(message.content.clone()),
                InteractiveRoleV1::Assistant => Qwen35ChatMessageV1::assistant(
                    message.content.clone(),
                    message.reasoning_content.clone(),
                ),
            })
            .collect()
    }

    /// Build a transcript from the public typed chat messages without exposing
    /// transcript storage or checkpoint bytes to the CLI adapter.
    pub(crate) fn from_messages(
        messages: &[Qwen35ChatMessageV1],
    ) -> Result<Self, InteractiveErrorV1> {
        let mut transcript = Self::default();
        for message in messages {
            match message {
                Qwen35ChatMessageV1::System { content } => {
                    transcript.push(InteractiveRoleV1::System, content)?;
                }
                Qwen35ChatMessageV1::User { content } => {
                    transcript.push(InteractiveRoleV1::User, content)?;
                }
                Qwen35ChatMessageV1::Assistant {
                    content,
                    reasoning_content,
                } => {
                    transcript.push_with_reasoning(
                        InteractiveRoleV1::Assistant,
                        content,
                        reasoning_content.as_deref(),
                    )?;
                }
            }
        }
        Ok(transcript)
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, InteractiveErrorV1> {
        let messages = self
            .messages
            .iter()
            .map(|message| {
                let mut row = json!({
                    "role": message.role.as_str(),
                    "content": message.content,
                });
                // Keep the original wire shape for assistant entries without
                // reasoning.  The decoder accepts this omitted field as None;
                // new reasoning-bearing entries carry the optional string.
                if let Some(reasoning) = message.reasoning_content.as_deref() {
                    row.as_object_mut()
                        .expect("transcript row object")
                        .insert("reasoning_content".to_owned(), json!(reasoning));
                }
                row
            })
            .collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&json!({
            "schema_version": TRANSCRIPT_SCHEMA_V1,
            "messages": messages,
        }))
        .map_err(|_| InteractiveErrorV1::InvalidTranscript)?;
        if bytes.len() > MAX_INTERACTIVE_TRANSCRIPT_BYTES_V1 {
            return Err(InteractiveErrorV1::TranscriptTooLarge);
        }
        Ok(bytes)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, InteractiveErrorV1> {
        if bytes.len() > MAX_INTERACTIVE_TRANSCRIPT_BYTES_V1 {
            return Err(InteractiveErrorV1::TranscriptTooLarge);
        }
        let wire: WireTranscriptV1 =
            serde_json::from_slice(bytes).map_err(|_| InteractiveErrorV1::InvalidTranscript)?;
        if wire.schema_version != TRANSCRIPT_SCHEMA_V1 {
            return Err(InteractiveErrorV1::InvalidTranscript);
        }
        if wire.messages.len() > MAX_INTERACTIVE_MESSAGES_V1 {
            return Err(InteractiveErrorV1::TooManyMessages);
        }
        let mut transcript = Self::default();
        for row in wire.messages {
            let role = InteractiveRoleV1::parse(&row.role)?;
            transcript.push_with_reasoning(role, &row.content, row.reasoning_content.as_deref())?;
        }
        // Re-encoding applies the exact aggregate bound to the canonical form.
        let _ = transcript.encode()?;
        Ok(transcript)
    }

    fn push(&mut self, role: InteractiveRoleV1, content: &str) -> Result<(), InteractiveErrorV1> {
        self.push_with_reasoning(role, content, None)
    }

    fn push_with_reasoning(
        &mut self,
        role: InteractiveRoleV1,
        content: &str,
        reasoning_content: Option<&str>,
    ) -> Result<(), InteractiveErrorV1> {
        if content.is_empty() {
            return Err(InteractiveErrorV1::EmptyMessage);
        }
        if content.contains('\0') {
            return Err(InteractiveErrorV1::MessageContainsNul);
        }
        if reasoning_content.is_some() && role != InteractiveRoleV1::Assistant {
            return Err(InteractiveErrorV1::ReasoningContentOnNonAssistant);
        }
        if reasoning_content.is_some_and(|value| value.contains('\0')) {
            return Err(InteractiveErrorV1::ReasoningContentContainsNul);
        }
        if self.messages.len() == MAX_INTERACTIVE_MESSAGES_V1 {
            return Err(InteractiveErrorV1::TooManyMessages);
        }
        self.messages.push(InteractiveMessageV1 {
            role,
            content: content.to_owned(),
            reasoning_content: reasoning_content.map(str::to_owned),
        });
        if self.encode().is_err() {
            self.messages.pop();
            return Err(InteractiveErrorV1::TranscriptTooLarge);
        }
        Ok(())
    }
}

/// A draft turn. Dropping it after a generation error/cancel leaves the
/// transcript unchanged; only `commit` publishes both user and assistant.
pub(crate) struct InteractiveTurnV1<'a> {
    transcript: &'a mut InteractiveTranscriptV1,
    user: String,
}

impl InteractiveTurnV1<'_> {
    pub(crate) fn messages(&self) -> Result<Vec<Qwen35ChatMessageV1>, InteractiveErrorV1> {
        if self.user.is_empty() {
            return Err(InteractiveErrorV1::EmptyMessage);
        }
        let mut messages = self.transcript.messages();
        messages.push(Qwen35ChatMessageV1::user(self.user.clone()));
        Ok(messages)
    }

    #[allow(dead_code)]
    pub(crate) fn commit(self, assistant: &str) -> Result<(), InteractiveErrorV1> {
        self.commit_with_reasoning(assistant, None)
    }

    pub(crate) fn commit_with_reasoning(
        self,
        assistant: &str,
        reasoning_content: Option<&str>,
    ) -> Result<(), InteractiveErrorV1> {
        if self.user.is_empty() || assistant.is_empty() {
            return Err(InteractiveErrorV1::EmptyMessage);
        }
        let before = self.transcript.messages.len();
        self.transcript.push(InteractiveRoleV1::User, &self.user)?;
        if let Err(error) = self.transcript.push_with_reasoning(
            InteractiveRoleV1::Assistant,
            assistant,
            reasoning_content,
        ) {
            self.transcript.messages.truncate(before);
            return Err(error);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReversePromptMatchV1 {
    pub(crate) visible: String,
    pub(crate) matched: Option<String>,
}

/// Incremental reverse-prompt detector. It retains only the longest suffix
/// that can still begin a marker and never publishes the matched marker.
#[derive(Clone, Debug)]
pub(crate) struct ReversePromptMatcherV1 {
    prompts: Vec<String>,
    pending: String,
    matched: bool,
}

impl ReversePromptMatcherV1 {
    pub(crate) fn new(prompts: Vec<String>) -> Result<Self, InteractiveErrorV1> {
        if prompts.len() > MAX_REVERSE_PROMPTS_V1 {
            return Err(InteractiveErrorV1::TooManyReversePrompts);
        }
        let mut total = 0_usize;
        for (index, prompt) in prompts.iter().enumerate() {
            if prompt.is_empty() {
                return Err(InteractiveErrorV1::EmptyReversePrompt);
            }
            if prompt.contains('\0') {
                return Err(InteractiveErrorV1::ReversePromptContainsNul);
            }
            total = total
                .checked_add(prompt.len())
                .ok_or(InteractiveErrorV1::ReversePromptsTooLarge)?;
            if total > MAX_REVERSE_PROMPT_BYTES_V1 {
                return Err(InteractiveErrorV1::ReversePromptsTooLarge);
            }
            if prompts[..index].contains(prompt) {
                return Err(InteractiveErrorV1::DuplicateReversePrompt);
            }
        }
        Ok(Self {
            prompts,
            pending: String::new(),
            matched: false,
        })
    }

    pub(crate) fn push(&mut self, delta: &str) -> ReversePromptMatchV1 {
        if self.matched {
            return ReversePromptMatchV1 {
                visible: String::new(),
                matched: None,
            };
        }
        self.pending.push_str(delta);
        if let Some((position, prompt)) = earliest_match(&self.pending, &self.prompts) {
            let visible = self.pending[..position].to_owned();
            self.pending.clear();
            self.matched = true;
            return ReversePromptMatchV1 {
                visible,
                matched: Some(prompt.to_owned()),
            };
        }
        let keep = longest_possible_suffix(&self.pending, &self.prompts);
        let visible_end = self.pending.len().saturating_sub(keep);
        let visible = self.pending[..visible_end].to_owned();
        self.pending.drain(..visible_end);
        ReversePromptMatchV1 {
            visible,
            matched: None,
        }
    }

    pub(crate) fn finish(&mut self) -> String {
        if self.matched {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}

fn earliest_match<'a>(text: &str, prompts: &'a [String]) -> Option<(usize, &'a str)> {
    prompts
        .iter()
        .filter_map(|prompt| {
            text.find(prompt)
                .map(|position| (position, prompt.as_str()))
        })
        .min_by_key(|(position, prompt)| (*position, prompt.len()))
}

fn longest_possible_suffix(text: &str, prompts: &[String]) -> usize {
    prompts
        .iter()
        .flat_map(|prompt| {
            prompt
                .char_indices()
                .map(move |(index, _)| &prompt[..index])
        })
        .filter(|prefix| !prefix.is_empty() && text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(1);

    fn test_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "sllm-phase44-{label}-{}-{}",
            std::process::id(),
            TEST_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn prompt_sources_are_exactly_one() {
        assert_eq!(
            PromptSourceKindV1::select(false, false, true, false).unwrap(),
            PromptSourceKindV1::PromptFile
        );
        assert!(PromptSourceKindV1::select(false, false, false, false).is_err());
        assert!(PromptSourceKindV1::select(true, false, true, false).is_err());
    }

    #[test]
    fn prompt_file_is_regular_bounded_and_utf8() {
        let path = test_path("prompt");
        fs::write(&path, "こんにちは").unwrap();
        assert_eq!(read_prompt_file_v1(&path).unwrap(), "こんにちは");
        fs::write(&path, b"embedded\0nul").unwrap();
        assert_eq!(
            read_prompt_file_v1(&path),
            Err(InteractiveErrorV1::PromptFileContainsNul)
        );
        fs::write(&path, [0xff]).unwrap();
        assert_eq!(
            read_prompt_file_v1(&path),
            Err(InteractiveErrorV1::PromptFileInvalidUtf8)
        );
        fs::remove_file(&path).unwrap();

        #[cfg(unix)]
        {
            let target = test_path("target");
            let link = test_path("link");
            fs::write(&target, "safe").unwrap();
            std::os::unix::fs::symlink(&target, &link).unwrap();
            assert_eq!(
                read_prompt_file_v1(&link),
                Err(InteractiveErrorV1::PromptFileNotRegular)
            );
            fs::remove_file(link).unwrap();
            fs::remove_file(target).unwrap();
        }
    }

    #[test]
    fn transcript_roundtrip_and_failed_turn_are_transactional() {
        let mut transcript = InteractiveTranscriptV1::new(Some("system")).unwrap();
        {
            let turn = transcript.begin_turn("cancelled");
            assert_eq!(turn.messages().unwrap().len(), 2);
        }
        assert_eq!(transcript.messages().len(), 1);
        transcript.begin_turn("hello").commit("world").unwrap();
        let encoded = transcript.encode().unwrap();
        let restored = InteractiveTranscriptV1::decode(&encoded).unwrap();
        assert_eq!(restored, transcript);
        assert_eq!(restored.messages().len(), 3);
        for malformed in [
            br#"{"schema_version":"sllm-interactive-transcript-v1","schema_version":"sllm-interactive-transcript-v1","messages":[]}"#.as_slice(),
            br#"{"schema_version":"sllm-interactive-transcript-v1","messages":[{"role":"user","role":"user","content":"x"}]}"#.as_slice(),
            br#"{"schema_version":"sllm-interactive-transcript-v1","messages":[],"extra":true}"#.as_slice(),
        ] {
            assert_eq!(
                InteractiveTranscriptV1::decode(malformed),
                Err(InteractiveErrorV1::InvalidTranscript)
            );
        }
        assert_eq!(
            InteractiveTranscriptV1::decode(
                br#"{"schema_version":"sllm-interactive-transcript-v1","messages":[{"role":"user","content":"x\u0000y"}]}"#
            ),
            Err(InteractiveErrorV1::MessageContainsNul)
        );
    }

    #[test]
    fn reasoning_content_roundtrips_and_is_restricted_to_assistant() {
        let mut transcript = InteractiveTranscriptV1::default();
        transcript
            .begin_turn("question")
            .commit_with_reasoning("answer", Some("analysis"))
            .unwrap();
        let encoded = transcript.encode().unwrap();
        let restored = InteractiveTranscriptV1::decode(&encoded).unwrap();
        assert_eq!(restored, transcript);
        assert!(matches!(
            restored.messages().as_slice(),
            [
                Qwen35ChatMessageV1::User { content },
                Qwen35ChatMessageV1::Assistant {
                    content: answer,
                    reasoning_content: Some(reasoning),
                }
            ] if content == "question" && answer == "answer" && reasoning == "analysis"
        ));

        let old_wire =
            br#"{"schema_version":"sllm-interactive-transcript-v1","messages":[{"role":"assistant","content":"old"}]}"#;
        assert!(matches!(
            InteractiveTranscriptV1::decode(old_wire)
                .unwrap()
                .messages()
                .as_slice(),
            [Qwen35ChatMessageV1::Assistant {
                reasoning_content: None,
                ..
            }]
        ));
        assert_eq!(
            InteractiveTranscriptV1::decode(
                br#"{"schema_version":"sllm-interactive-transcript-v1","messages":[{"role":"user","content":"u","reasoning_content":"invalid"}]}"#
            ),
            Err(InteractiveErrorV1::ReasoningContentOnNonAssistant)
        );
        assert_eq!(
            InteractiveTranscriptV1::decode(
                br#"{"schema_version":"sllm-interactive-transcript-v1","messages":[{"role":"assistant","content":"a","reasoning_content":"bad\u0000reason"}]}"#
            ),
            Err(InteractiveErrorV1::ReasoningContentContainsNul)
        );
    }

    #[test]
    fn reverse_prompt_is_detected_across_unicode_deltas_and_not_published() {
        let mut matcher = ReversePromptMatcherV1::new(vec!["ユーザー:".to_owned()]).unwrap();
        let first = matcher.push("answer\nユー");
        assert_eq!(first.visible, "answer\n");
        assert_eq!(first.matched, None);
        let second = matcher.push("ザー:ignored");
        assert_eq!(second.visible, "");
        assert_eq!(second.matched.as_deref(), Some("ユーザー:"));
        assert_eq!(matcher.finish(), "");
    }

    #[test]
    fn reverse_prompt_limits_cover_both_sides() {
        assert!(ReversePromptMatcherV1::new(vec!["x".to_owned(); 5]).is_err());
        assert!(ReversePromptMatcherV1::new(vec![String::new()]).is_err());
        assert!(matches!(
            ReversePromptMatcherV1::new(vec!["bad\0marker".to_owned()]),
            Err(InteractiveErrorV1::ReversePromptContainsNul)
        ));
        assert!(ReversePromptMatcherV1::new(vec!["x".to_owned(), "x".to_owned()]).is_err());
        let exact = "x".repeat(MAX_REVERSE_PROMPT_BYTES_V1);
        assert!(ReversePromptMatcherV1::new(vec![exact]).is_ok());
        let over = "x".repeat(MAX_REVERSE_PROMPT_BYTES_V1 + 1);
        assert!(ReversePromptMatcherV1::new(vec![over]).is_err());
    }
}
