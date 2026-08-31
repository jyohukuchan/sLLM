//! Strict text-only frontend contract for the reviewed Ministral 3 3B model.
//!
//! The upstream template also contains multimodal and tool branches.  Those
//! branches are intentionally not represented here: the production boundary
//! accepts only UTF-8 system, user, and assistant messages and emits the
//! exact text markers used by the official template.  Keeping this adapter
//! separate from the generic Jinja renderer makes it impossible for a caller
//! to accidentally enable an unreviewed tool or image path.

use core::fmt;
use std::collections::HashSet;

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{
    ChatMessageV1, DecodeModeV1, GenerationExecutorV1, GenerationServiceError, GenerationStepV1,
    GenerationStopPolicyV1, GenerationTextFrontendV1, TokenByteTableV1, TokenIdsV1,
    validate_generation_stop_policy,
};
use sllm_core::{
    BudgetBoundary, MaxNewTokensZero, PromptEvaluation, StopEvaluation, StopTokenHandling,
};

/// Version of the fixed Ministral 3 text renderer/tokenizer boundary.
pub const MINISTRAL3_FRONTEND_VERSION_V1: u8 = 1;

/// The official tokenizer asset identity.  The tokenizer is kept as a GGUF
/// frontend asset rather than copied into the repository.
pub const MINISTRAL3_TOKENIZER_FILENAME: &str = "tokenizer.json";
pub const MINISTRAL3_TOKENIZER_SIZE_BYTES: usize = 17_078_128;
pub const MINISTRAL3_TOKENIZER_SHA256: &str =
    "d5f6046775b112f0e2d456ee9dba450684ab964fe5c4e231599bdc6773028135";

/// The upstream chat template asset identity.  The template itself is not
/// evaluated at runtime: its reviewed text-only branch is rendered by the
/// bounded implementation below.
pub const MINISTRAL3_CHAT_TEMPLATE_FILENAME: &str = "chat_template.jinja";
pub const MINISTRAL3_CHAT_TEMPLATE_SIZE_BYTES: usize = 11_912;
pub const MINISTRAL3_CHAT_TEMPLATE_SHA256: &str =
    "0701cfbdc2b7d44fdbad104dff604faee4b0543e8247624568777fe465746f9b";

/// Identity of the copy embedded in the official GGUF.  It is a normalized
/// template distinct from the source repository's 11,912-byte file above.
pub const MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SIZE_BYTES: usize = 7_753;
pub const MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SHA256: &str =
    "d28d7df94f0fd7e8d0075a22c473333d6e7dd2bc4c36c83e8b975300a0fb94bc";

/// Compact official tokenizer/rendering fixtures.  They intentionally stay
/// as a small contract in source; the 17 MiB tokenizer asset remains an
/// external verified GGUF frontend asset.
pub const MINISTRAL3_SYSTEM_USER_FIXTURE_TOKEN_IDS_V1: &[u32] = &[
    1, 17, 31_106, 27_457, 1_046, 18, 3, 7_493, 1_395, 1_032, 1_050, 1_043, 1_050, 1_063, 4,
];
pub const MINISTRAL3_SYSTEM_USER_FIXTURE_RENDERED_SHA256: &str =
    "b7c584efba50f90f88c91af466b3e26ba589f29a00667f30042c5c98a3c78e39";
pub const MINISTRAL3_SYSTEM_USER_FIXTURE_TOKEN_IDS_SHA256: &str =
    "afca9380c9716b9ff46c9b1cd3a1d9add7f863dddacb3d137f3c3b2fef3ea77a";
pub const MINISTRAL3_HISTORY_FIXTURE_TOKEN_IDS_V1: &[u32] = &[
    1, 17, 31_106, 27_457, 1_046, 18, 3, 67_935, 1_349, 1_046, 4, 1_065, 1_046, 2, 3, 12_082,
    1_398, 1_046, 4,
];
pub const MINISTRAL3_HISTORY_FIXTURE_RENDERED_SHA256: &str =
    "94f253a9d0ff735852147c883661ba3f5f8db6a2f7cbc94847f1a9aadfb81b5b";
pub const MINISTRAL3_HISTORY_FIXTURE_TOKEN_IDS_SHA256: &str =
    "1303dde6c30cdbea9a69859c4de8052737d9cccc874f4f5665ba72ac11ed8c0c";

/// Official GGUF-embedded template's default system prompt.  It is
/// deliberately retained as literal text: this normalized artifact leaves
/// `{today}` and `{yesterday}` as literal placeholders in this fixed revision.
pub const MINISTRAL3_DEFAULT_SYSTEM_PROMPT: &str = "You are Ministral-3-3B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is {today}.

When you're not sure about some information or when the user's request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don't have the information and avoid making up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \"What are some good restaurants around me?\" => \"Where are you?\" or \"When is the next flight to Tokyo\" => \"Where do you travel from?\").
You are always very attentive to dates, in particular you try to resolve dates (e.g. \"yesterday\" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.
You cannot read nor transcribe audio files or videos.

# TOOL CALLING INSTRUCTIONS

You may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:

1. When the request requires up-to-date information.
2. When the request requires specific data that you do not have in your knowledge base.
3. When the request involves actions that you cannot perform without tools.

Always prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment.";

pub const MINISTRAL3_CHAT_MAX_OUTPUT_BYTES_V1: usize = 16 * 1024 * 1024;
pub const MINISTRAL3_MAX_MESSAGES_V1: usize = 128;

const BOS: &str = "<s>";
const EOS: &str = "</s>";
const SYSTEM_OPEN: &str = "[SYSTEM_PROMPT]";
const SYSTEM_CLOSE: &str = "[/SYSTEM_PROMPT]";
const USER_OPEN: &str = "[INST]";
const USER_CLOSE: &str = "[/INST]";
const MINISTRAL3_GGUF_TOKEN_COUNT: usize = 131_072;
const MINISTRAL3_GGUF_MERGE_COUNT: usize = 269_443;
const MINISTRAL3_GGUF_SPECIAL_TOKEN_COUNT: usize = 1_000;
const MINISTRAL3_GGUF_PRETOKENIZER_REGEX: &str = r#"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

const GGUF_TOKENIZER_KEYS: &[&str] = &[
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.chat_template",
];

/// Errors from the bounded Ministral 3 text frontend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ministral3FrontendErrorV1 {
    UnsupportedIdentity,
    TokenizerAssetUnavailable,
    InvalidTokenizerMetadata {
        key: String,
    },
    InvalidTokenizer,
    InvalidTokenizerIdentity,
    InvalidChatTemplateIdentity,
    EmptyMessages,
    TooManyMessages {
        limit: usize,
    },
    SystemMessageNotFirst {
        index: usize,
    },
    MultipleSystemMessages,
    NoUserMessage,
    InvalidMessageOrdering {
        index: usize,
        previous: &'static str,
        current: &'static str,
    },
    EmptyAssistantMessage {
        index: usize,
    },
    UnsupportedReasoningContent {
        index: usize,
    },
    OutputLimitExceedsHostCap,
    OutputTooLarge {
        limit_bytes: usize,
    },
    Encode,
    Decode,
    UnknownTokenId {
        id: u32,
    },
}

impl fmt::Display for Ministral3FrontendErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedIdentity => formatter
                .write_str("the source is not the fixed Ministral 3 3B Instruct-2512 identity"),
            Self::TokenizerAssetUnavailable => {
                formatter.write_str("the verified GGUF has no tokenizer.json frontend asset")
            }
            Self::InvalidTokenizerMetadata { key } => {
                write!(formatter, "GGUF tokenizer metadata is invalid at {key}")
            }
            Self::InvalidTokenizer => formatter.write_str("tokenizer.json is invalid"),
            Self::InvalidTokenizerIdentity => {
                formatter.write_str("tokenizer.json differs from the fixed Ministral 3 asset")
            }
            Self::InvalidChatTemplateIdentity => {
                formatter.write_str("chat_template.jinja differs from the fixed Ministral 3 asset")
            }
            Self::EmptyMessages => formatter.write_str("chat messages must not be empty"),
            Self::TooManyMessages { limit } => {
                write!(
                    formatter,
                    "chat has more than the {limit}-message host limit"
                )
            }
            Self::SystemMessageNotFirst { index } => {
                write!(
                    formatter,
                    "system message must be at index 0, found at {index}"
                )
            }
            Self::MultipleSystemMessages => {
                formatter.write_str("multiple system messages are unsupported")
            }
            Self::NoUserMessage => formatter.write_str("chat requires at least one user message"),
            Self::InvalidMessageOrdering {
                index,
                previous,
                current,
            } => write!(
                formatter,
                "message {index} has invalid role ordering: {previous} followed by {current}"
            ),
            Self::EmptyAssistantMessage { index } => {
                write!(formatter, "assistant message {index} must not be empty")
            }
            Self::UnsupportedReasoningContent { index } => write!(
                formatter,
                "assistant message {index} has unsupported reasoning_content"
            ),
            Self::OutputLimitExceedsHostCap => {
                formatter.write_str("requested output limit exceeds the renderer host cap")
            }
            Self::OutputTooLarge { limit_bytes } => {
                write!(
                    formatter,
                    "rendered chat exceeds the {limit_bytes}-byte output limit"
                )
            }
            Self::Encode => formatter.write_str("tokenizer could not encode text"),
            Self::Decode => formatter.write_str("tokenizer could not decode token IDs"),
            Self::UnknownTokenId { id } => write!(formatter, "unknown token ID {id}"),
        }
    }
}

impl std::error::Error for Ministral3FrontendErrorV1 {}

fn invalid_metadata(key: &str) -> Ministral3FrontendErrorV1 {
    Ministral3FrontendErrorV1::InvalidTokenizerMetadata {
        key: key.to_owned(),
    }
}

fn metadata_string<'a>(
    gguf: &'a sllm_core::VerifiedGguf,
    key: &str,
) -> Result<&'a str, Ministral3FrontendErrorV1> {
    match gguf.metadata_value(key) {
        Some(sllm_core::GgufValue::String(value)) => Ok(value),
        _ => Err(invalid_metadata(key)),
    }
}

fn metadata_u32(
    gguf: &sllm_core::VerifiedGguf,
    key: &str,
) -> Result<u32, Ministral3FrontendErrorV1> {
    match gguf.metadata_value(key) {
        Some(sllm_core::GgufValue::U32(value)) => Ok(*value),
        _ => Err(invalid_metadata(key)),
    }
}

fn metadata_bool(
    gguf: &sllm_core::VerifiedGguf,
    key: &str,
) -> Result<bool, Ministral3FrontendErrorV1> {
    match gguf.metadata_value(key) {
        Some(sllm_core::GgufValue::Bool(value)) => Ok(*value),
        _ => Err(invalid_metadata(key)),
    }
}

fn metadata_string_array<'a>(
    gguf: &'a sllm_core::VerifiedGguf,
    key: &str,
    expected_len: usize,
) -> Result<&'a [String], Ministral3FrontendErrorV1> {
    match gguf.metadata_value(key) {
        Some(sllm_core::GgufValue::Array(sllm_core::GgufArray::String(values)))
            if values.len() == expected_len =>
        {
            Ok(values)
        }
        _ => Err(invalid_metadata(key)),
    }
}

fn validate_gguf_tokenizer_metadata(
    gguf: &sllm_core::VerifiedGguf,
) -> Result<(), Ministral3FrontendErrorV1> {
    for key in gguf
        .metadata()
        .keys()
        .filter(|key| key.starts_with("tokenizer."))
    {
        if !GGUF_TOKENIZER_KEYS.contains(&key.as_str()) {
            return Err(invalid_metadata(key));
        }
    }
    if metadata_string(gguf, "tokenizer.ggml.model")? != "gpt2"
        || metadata_string(gguf, "tokenizer.ggml.pre")? != "tekken"
    {
        return Err(invalid_metadata("tokenizer.ggml.model/pre"));
    }
    if metadata_u32(gguf, "tokenizer.ggml.bos_token_id")? != 1
        || metadata_u32(gguf, "tokenizer.ggml.eos_token_id")? != 2
        || metadata_u32(gguf, "tokenizer.ggml.unknown_token_id")? != 0
        || metadata_u32(gguf, "tokenizer.ggml.padding_token_id")? != 11
        || !metadata_bool(gguf, "tokenizer.ggml.add_bos_token")?
        || metadata_bool(gguf, "tokenizer.ggml.add_eos_token")?
    {
        return Err(invalid_metadata("tokenizer.ggml.special_token_ids"));
    }

    let tokens = metadata_string_array(gguf, "tokenizer.ggml.tokens", MINISTRAL3_GGUF_TOKEN_COUNT)?;
    let merges = metadata_string_array(gguf, "tokenizer.ggml.merges", MINISTRAL3_GGUF_MERGE_COUNT)?;
    match gguf.metadata_value("tokenizer.ggml.scores") {
        Some(sllm_core::GgufValue::Array(sllm_core::GgufArray::I32(scores)))
            if scores.len() == MINISTRAL3_GGUF_TOKEN_COUNT
                && scores.iter().all(|score| *score == 0) => {}
        _ => return Err(invalid_metadata("tokenizer.ggml.scores")),
    }
    match gguf.metadata_value("tokenizer.ggml.token_type") {
        Some(sllm_core::GgufValue::Array(sllm_core::GgufArray::I32(types)))
            if types.len() == MINISTRAL3_GGUF_TOKEN_COUNT
                && types.iter().enumerate().all(|(id, token_type)| {
                    *token_type
                        == if id < MINISTRAL3_GGUF_SPECIAL_TOKEN_COUNT {
                            3
                        } else {
                            1
                        }
                }) => {}
        _ => return Err(invalid_metadata("tokenizer.ggml.token_type")),
    }
    let mut seen_tokens = HashSet::with_capacity(tokens.len());
    if tokens
        .iter()
        .any(|token| !seen_tokens.insert(token.as_str()))
    {
        return Err(invalid_metadata("tokenizer.ggml.tokens.duplicates"));
    }
    let mut seen_merges = HashSet::with_capacity(merges.len());
    if merges.iter().any(|merge| {
        !seen_merges.insert(merge.as_str())
            || merge.split_once(' ').is_none_or(|(left, right)| {
                left.is_empty() || right.is_empty() || right.contains(' ')
            })
    }) {
        return Err(invalid_metadata("tokenizer.ggml.merges"));
    }
    let template = metadata_string(gguf, "tokenizer.chat_template")?;
    if template.len() != MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SIZE_BYTES
        || format!("{:x}", Sha256::digest(template.as_bytes()))
            != MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SHA256
    {
        return Err(invalid_metadata("tokenizer.chat_template"));
    }
    Ok(())
}

fn tokenizer_json_from_gguf_metadata(
    gguf: &sllm_core::VerifiedGguf,
) -> Result<Vec<u8>, Ministral3FrontendErrorV1> {
    validate_gguf_tokenizer_metadata(gguf)?;
    let tokens = metadata_string_array(gguf, "tokenizer.ggml.tokens", MINISTRAL3_GGUF_TOKEN_COUNT)?;
    let merges = metadata_string_array(gguf, "tokenizer.ggml.merges", MINISTRAL3_GGUF_MERGE_COUNT)?;

    let mut vocab = Map::with_capacity(tokens.len());
    for (id, token) in tokens.iter().enumerate() {
        vocab.insert(token.clone(), Value::from(id as u64));
    }
    let added_tokens = tokens[..MINISTRAL3_GGUF_SPECIAL_TOKEN_COUNT]
        .iter()
        .enumerate()
        .map(|(id, token)| {
            json!({
                "id": id,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect::<Vec<_>>();
    let merges = merges
        .iter()
        .map(|merge| {
            let (left, right) = merge
                .split_once(' ')
                .expect("validated GGUF merge has two pieces");
            json!([left, right])
        })
        .collect::<Vec<_>>();
    let pretokenizer_regex = Value::Object(Map::from_iter([(
        "Regex".to_owned(),
        Value::String(MINISTRAL3_GGUF_PRETOKENIZER_REGEX.to_owned()),
    )]));
    let root = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": pretokenizer_regex,
                    "behavior": "Isolated",
                    "invert": false,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": true,
                    "use_regex": false,
                },
            ],
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<s>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
            },
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true,
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": true,
            "vocab": vocab,
            "merges": merges,
        },
    });
    serde_json::to_vec(&root).map_err(|_| Ministral3FrontendErrorV1::InvalidTokenizer)
}

/// A fixed, identity-checked tokenizer for the official GGUF frontend asset.
#[derive(Clone, Debug)]
pub struct Ministral3TokenizerV1 {
    tokenizer: Tokenizer,
    token_byte_table: TokenByteTableV1,
}

impl Ministral3TokenizerV1 {
    /// Load the tokenizer embedded in a verified official Ministral 3 GGUF.
    pub fn from_verified_gguf(
        source: &sllm_core::VerifiedOfficialMinistral3Gguf,
    ) -> Result<Self, Ministral3FrontendErrorV1> {
        if source.repository() != sllm_core::MINISTRAL3_OFFICIAL_GGUF_REPOSITORY
            || source.revision() != sllm_core::MINISTRAL3_OFFICIAL_GGUF_REVISION
        {
            return Err(Ministral3FrontendErrorV1::UnsupportedIdentity);
        }
        let tokenizer_json = tokenizer_json_from_gguf_metadata(source.gguf())?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_json)
            .map_err(|_| Ministral3FrontendErrorV1::InvalidTokenizer)?;
        validate_tokenizer_identity(&tokenizer)?;
        let token_byte_table = TokenByteTableV1::from_tokenizer_json(
            &tokenizer_json,
            u64::from(sllm_core::MINISTRAL3_VOCAB_SIZE),
        )
        .map_err(|_| Ministral3FrontendErrorV1::InvalidTokenizer)?;
        Ok(Self {
            tokenizer,
            token_byte_table,
        })
    }

    /// Construct from an exact tokenizer asset.  This is public so a cache
    /// provider can validate its own asset before creating the frontend, while
    /// still enforcing the same size, digest, vocabulary, and special-token
    /// checks as the GGUF path.
    pub fn from_verified_bytes(bytes: &[u8]) -> Result<Self, Ministral3FrontendErrorV1> {
        if bytes.len() != MINISTRAL3_TOKENIZER_SIZE_BYTES {
            return Err(Ministral3FrontendErrorV1::InvalidTokenizerIdentity);
        }
        if format!("{:x}", Sha256::digest(bytes)) != MINISTRAL3_TOKENIZER_SHA256 {
            return Err(Ministral3FrontendErrorV1::InvalidTokenizerIdentity);
        }
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|_| Ministral3FrontendErrorV1::InvalidTokenizer)?;
        validate_tokenizer_identity(&tokenizer)?;
        let token_byte_table = TokenByteTableV1::from_tokenizer_json(
            bytes,
            u64::from(sllm_core::MINISTRAL3_VOCAB_SIZE),
        )
        .map_err(|_| Ministral3FrontendErrorV1::InvalidTokenizer)?;
        Ok(Self {
            tokenizer,
            token_byte_table,
        })
    }

    pub fn version(&self) -> u8 {
        MINISTRAL3_FRONTEND_VERSION_V1
    }

    pub fn encode(&self, text: &str) -> Result<TokenIdsV1, Ministral3FrontendErrorV1> {
        // The rendered prompt already includes its explicit BOS marker.  The
        // official tokenizer must therefore not inject a second BOS token.
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|_| Ministral3FrontendErrorV1::Encode)?;
        Ok(TokenIdsV1::from_slice(encoding.get_ids()))
    }

    pub fn decode(
        &self,
        token_ids: &TokenIdsV1,
        mode: DecodeModeV1,
    ) -> Result<String, Ministral3FrontendErrorV1> {
        for id in token_ids.as_slice() {
            if self.tokenizer.id_to_token(*id).is_none() {
                return Err(Ministral3FrontendErrorV1::UnknownTokenId { id: *id });
            }
        }
        self.tokenizer
            .decode(
                token_ids.as_slice(),
                matches!(mode, DecodeModeV1::SkipSpecialTokens),
            )
            .map_err(|_| Ministral3FrontendErrorV1::Decode)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Returns the bounded decoder-aware raw piece table used by constrained
    /// generation.  Every model-vocabulary ID has an explicit row; IDs above
    /// the fixed 131,072-entry capacity remain fail-closed through the
    /// generation frontend.
    pub fn token_byte_table(&self) -> &TokenByteTableV1 {
        &self.token_byte_table
    }
}

impl GenerationTextFrontendV1 for Ministral3TokenizerV1 {
    fn encode_generation(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
        self.encode(text)
            .map(|ids| ids.as_slice().to_vec())
            .map_err(|_| GenerationServiceError::Tokenize)
    }

    fn encode_assistant_prefill(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
        // Like the regular path, this deliberately does not inject special
        // tokens: the caller owns the rendered BOS and assistant boundary.
        self.encode(text)
            .map(|ids| ids.as_slice().to_vec())
            .map_err(|_| GenerationServiceError::Tokenize)
    }

    fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError> {
        self.decode(
            &TokenIdsV1::from_slice(token_ids),
            DecodeModeV1::PreserveSpecialTokens,
        )
        .map_err(|_| GenerationServiceError::Decode)
    }

    fn token_byte_table(&self) -> Result<&TokenByteTableV1, GenerationServiceError> {
        Ok(self.token_byte_table())
    }
}

/// Return the reviewed text-generation stop policy for Ministral 3.
///
/// Ministral 3 has one EOS token (`</s>`, ID 2).  EOS is evaluated only after
/// a newly generated argmax, is hidden from visible output, and is not fed to
/// a subsequent decode step.  The policy intentionally never evaluates EOS
/// in the prompt, so an EOS present in history does not suppress generation.
pub fn ministral3_generation_stop_policy() -> Result<GenerationStopPolicyV1, GenerationServiceError>
{
    let policy = GenerationStopPolicyV1 {
        version: 1,
        stop_token_ids: vec![2],
        evaluation: StopEvaluation::NewlyGeneratedAfterArgmax,
        prompt_evaluation: PromptEvaluation::NeverStop,
        stop_token: StopTokenHandling {
            visible_output: false,
            subsequent_decode_input: false,
        },
        budget_boundary: BudgetBoundary::StopTokenWins,
        max_new_tokens_zero: MaxNewTokensZero::MaxNewTokensBeforeDecode,
        reason_version: 1,
    };
    validate_generation_stop_policy(&policy)
        .map_err(|_| GenerationServiceError::InvalidStopPolicy)?;
    Ok(policy)
}

fn validate_tokenizer_identity(tokenizer: &Tokenizer) -> Result<(), Ministral3FrontendErrorV1> {
    if tokenizer.get_vocab_size(true) != sllm_core::MINISTRAL3_VOCAB_SIZE as usize {
        return Err(Ministral3FrontendErrorV1::InvalidTokenizerIdentity);
    }
    let vocab = tokenizer.get_vocab(true);
    if vocab.len() != sllm_core::MINISTRAL3_VOCAB_SIZE as usize
        || vocab.values().copied().max() != Some(sllm_core::MINISTRAL3_VOCAB_SIZE - 1)
    {
        return Err(Ministral3FrontendErrorV1::InvalidTokenizerIdentity);
    }
    const SPECIALS: &[(&str, u32)] = &[
        ("<unk>", 0),
        ("<s>", 1),
        ("</s>", 2),
        ("[INST]", 3),
        ("[/INST]", 4),
        ("<pad>", 11),
        ("[SYSTEM_PROMPT]", 17),
        ("[/SYSTEM_PROMPT]", 18),
    ];
    let added = tokenizer.get_added_tokens_decoder();
    for &(content, id) in SPECIALS {
        if tokenizer.token_to_id(content) != Some(id)
            || tokenizer.id_to_token(id).as_deref() != Some(content)
            || !added.get(&id).is_some_and(|token| token.special)
        {
            return Err(Ministral3FrontendErrorV1::InvalidTokenizerIdentity);
        }
    }
    Ok(())
}

/// Rendering controls.  Ministral's official template has no generation
/// marker or thinking switch; the only accepted control is a bounded host
/// output limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ministral3RenderOptionsV1 {
    output_limit_bytes: usize,
}

impl Ministral3RenderOptionsV1 {
    pub const fn new() -> Self {
        Self {
            output_limit_bytes: MINISTRAL3_CHAT_MAX_OUTPUT_BYTES_V1,
        }
    }

    pub const fn with_output_limit_bytes(output_limit_bytes: usize) -> Self {
        Self { output_limit_bytes }
    }

    pub const fn output_limit_bytes(self) -> usize {
        self.output_limit_bytes
    }
}

impl Default for Ministral3RenderOptionsV1 {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrowed renderer for the fixed text-only Ministral 3 template.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ministral3ChatRendererV1;

impl Ministral3ChatRendererV1 {
    pub const fn new() -> Self {
        Self
    }

    pub const fn version(self) -> u8 {
        MINISTRAL3_FRONTEND_VERSION_V1
    }

    /// Bind the renderer to a verified official GGUF identity.  The official
    /// GGUF verifier owns the container metadata/catalog checks; this method
    /// keeps the frontend-side embedded-template check explicit at the call
    /// site.
    pub fn from_verified_gguf(
        source: &sllm_core::VerifiedOfficialMinistral3Gguf,
    ) -> Result<Self, Ministral3FrontendErrorV1> {
        if source.repository() != sllm_core::MINISTRAL3_OFFICIAL_GGUF_REPOSITORY
            || source.revision() != sllm_core::MINISTRAL3_OFFICIAL_GGUF_REVISION
        {
            return Err(Ministral3FrontendErrorV1::UnsupportedIdentity);
        }
        let template = metadata_string(source.gguf(), "tokenizer.chat_template")?;
        if template.len() != MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SIZE_BYTES
            || format!("{:x}", Sha256::digest(template.as_bytes()))
                != MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SHA256
        {
            return Err(Ministral3FrontendErrorV1::InvalidChatTemplateIdentity);
        }
        Ok(Self::new())
    }

    /// Validate the standalone upstream Jinja asset before constructing the
    /// fixed text renderer.  Runtime rendering remains handwritten and does
    /// not execute arbitrary template source.
    pub fn from_verified_template_bytes(bytes: &[u8]) -> Result<Self, Ministral3FrontendErrorV1> {
        if bytes.len() != MINISTRAL3_CHAT_TEMPLATE_SIZE_BYTES
            || format!("{:x}", Sha256::digest(bytes)) != MINISTRAL3_CHAT_TEMPLATE_SHA256
            || core::str::from_utf8(bytes).is_err()
        {
            return Err(Ministral3FrontendErrorV1::InvalidChatTemplateIdentity);
        }
        Ok(Self::new())
    }

    /// Validate the normalized copy carried by an official GGUF.  This is
    /// separate from [`Self::from_verified_template_bytes`] because the two
    /// artifacts have different byte identity and digest.
    pub fn from_verified_embedded_template_bytes(
        bytes: &[u8],
    ) -> Result<Self, Ministral3FrontendErrorV1> {
        if bytes.len() != MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SIZE_BYTES
            || format!("{:x}", Sha256::digest(bytes)) != MINISTRAL3_EMBEDDED_CHAT_TEMPLATE_SHA256
            || core::str::from_utf8(bytes).is_err()
        {
            return Err(Ministral3FrontendErrorV1::InvalidChatTemplateIdentity);
        }
        Ok(Self::new())
    }

    pub fn render(self, messages: &[ChatMessageV1]) -> Result<String, Ministral3FrontendErrorV1> {
        self.render_with_options(messages, Ministral3RenderOptionsV1::default())
    }

    pub fn render_with_options(
        self,
        messages: &[ChatMessageV1],
        options: Ministral3RenderOptionsV1,
    ) -> Result<String, Ministral3FrontendErrorV1> {
        validate_messages(messages)?;
        if options.output_limit_bytes > MINISTRAL3_CHAT_MAX_OUTPUT_BYTES_V1 {
            return Err(Ministral3FrontendErrorV1::OutputLimitExceedsHostCap);
        }

        let mut output = String::new();
        output.push_str(BOS);
        let has_system = matches!(messages.first(), Some(ChatMessageV1::System { .. }));
        if !has_system {
            append_checked(&mut output, SYSTEM_OPEN, options.output_limit_bytes)?;
            append_checked(
                &mut output,
                MINISTRAL3_DEFAULT_SYSTEM_PROMPT,
                options.output_limit_bytes,
            )?;
            append_checked(&mut output, SYSTEM_CLOSE, options.output_limit_bytes)?;
        }
        for (index, message) in messages.iter().enumerate() {
            match message {
                ChatMessageV1::System { content } => {
                    append_checked(&mut output, SYSTEM_OPEN, options.output_limit_bytes)?;
                    append_checked(&mut output, content, options.output_limit_bytes)?;
                    append_checked(&mut output, SYSTEM_CLOSE, options.output_limit_bytes)?;
                }
                ChatMessageV1::User { content } => {
                    append_checked(&mut output, USER_OPEN, options.output_limit_bytes)?;
                    append_checked(&mut output, content, options.output_limit_bytes)?;
                    append_checked(&mut output, USER_CLOSE, options.output_limit_bytes)?;
                }
                ChatMessageV1::Assistant {
                    content,
                    reasoning_content,
                } => {
                    if reasoning_content.is_some() {
                        // The upstream template ignores reasoning_content;
                        // rejecting it avoids silently losing a caller field.
                        return Err(Ministral3FrontendErrorV1::UnsupportedReasoningContent {
                            index,
                        });
                    }
                    append_checked(&mut output, content, options.output_limit_bytes)?;
                    append_checked(&mut output, EOS, options.output_limit_bytes)?;
                }
            }
        }
        Ok(output)
    }

    pub fn render_history_prefix(
        self,
        messages: &[ChatMessageV1],
    ) -> Result<String, Ministral3FrontendErrorV1> {
        self.render(messages)
    }

    pub fn render_with_assistant_prefill(
        self,
        messages: &[ChatMessageV1],
        assistant_prefill: &str,
    ) -> Result<String, Ministral3FrontendErrorV1> {
        validate_messages(messages)?;
        if !matches!(messages.last(), Some(ChatMessageV1::User { .. })) {
            return Err(Ministral3FrontendErrorV1::InvalidMessageOrdering {
                index: messages.len(),
                previous: "assistant",
                current: "prefill",
            });
        }
        let mut output = self.render(messages)?;
        append_checked(
            &mut output,
            assistant_prefill,
            MINISTRAL3_CHAT_MAX_OUTPUT_BYTES_V1,
        )?;
        Ok(output)
    }
}

/// Naming parallel to the existing Qwen/Gemma template owners.
pub type Ministral3ChatTemplateV1 = Ministral3ChatRendererV1;

fn append_checked(
    output: &mut String,
    fragment: &str,
    limit_bytes: usize,
) -> Result<(), Ministral3FrontendErrorV1> {
    let next = output
        .len()
        .checked_add(fragment.len())
        .ok_or(Ministral3FrontendErrorV1::OutputTooLarge { limit_bytes })?;
    if next > limit_bytes {
        return Err(Ministral3FrontendErrorV1::OutputTooLarge { limit_bytes });
    }
    output.push_str(fragment);
    Ok(())
}

fn role_name(message: &ChatMessageV1) -> &'static str {
    match message {
        ChatMessageV1::System { .. } => "system",
        ChatMessageV1::User { .. } => "user",
        ChatMessageV1::Assistant { .. } => "assistant",
    }
}

fn validate_messages(messages: &[ChatMessageV1]) -> Result<(), Ministral3FrontendErrorV1> {
    if messages.is_empty() {
        return Err(Ministral3FrontendErrorV1::EmptyMessages);
    }
    if messages.len() > MINISTRAL3_MAX_MESSAGES_V1 {
        return Err(Ministral3FrontendErrorV1::TooManyMessages {
            limit: MINISTRAL3_MAX_MESSAGES_V1,
        });
    }
    let mut users = 0usize;
    for (index, message) in messages.iter().enumerate() {
        let role = role_name(message);
        if role == "system" {
            if index != 0 {
                return Err(Ministral3FrontendErrorV1::SystemMessageNotFirst { index });
            }
            if index > 0 {
                return Err(Ministral3FrontendErrorV1::MultipleSystemMessages);
            }
        }
        if index > 0 {
            let previous = role_name(&messages[index - 1]);
            if previous == role || (previous == "system" && role != "user") {
                return Err(Ministral3FrontendErrorV1::InvalidMessageOrdering {
                    index,
                    previous,
                    current: role,
                });
            }
        }
        match message {
            ChatMessageV1::User { .. } => users += 1,
            ChatMessageV1::Assistant {
                content,
                reasoning_content,
            } => {
                if content.is_empty() {
                    return Err(Ministral3FrontendErrorV1::EmptyAssistantMessage { index });
                }
                if reasoning_content.is_some() {
                    return Err(Ministral3FrontendErrorV1::UnsupportedReasoningContent { index });
                }
            }
            ChatMessageV1::System { .. } => {}
        }
    }
    if users == 0 {
        return Err(Ministral3FrontendErrorV1::NoUserMessage);
    }
    Ok(())
}

/// A convenient resident pair for CLI/server owners.
#[derive(Clone, Debug)]
pub struct Ministral3TextFrontendV1 {
    tokenizer: Ministral3TokenizerV1,
    renderer: Ministral3ChatRendererV1,
}

impl Ministral3TextFrontendV1 {
    pub fn from_verified_gguf(
        source: &sllm_core::VerifiedOfficialMinistral3Gguf,
    ) -> Result<Self, Ministral3FrontendErrorV1> {
        Ok(Self {
            tokenizer: Ministral3TokenizerV1::from_verified_gguf(source)?,
            renderer: Ministral3ChatRendererV1::from_verified_gguf(source)?,
        })
    }

    pub fn tokenizer(&self) -> &Ministral3TokenizerV1 {
        &self.tokenizer
    }

    pub const fn renderer(&self) -> Ministral3ChatRendererV1 {
        self.renderer
    }
}

impl GenerationTextFrontendV1 for Ministral3TextFrontendV1 {
    fn encode_generation(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
        self.tokenizer.encode_generation(text)
    }

    fn encode_assistant_prefill(&self, text: &str) -> Result<Vec<u32>, GenerationServiceError> {
        self.tokenizer.encode_assistant_prefill(text)
    }

    fn decode_generation(&self, token_ids: &[u32]) -> Result<String, GenerationServiceError> {
        self.tokenizer.decode_generation(token_ids)
    }

    fn token_byte_table(&self) -> Result<&TokenByteTableV1, GenerationServiceError> {
        Ok(self.tokenizer.token_byte_table())
    }
}

impl GenerationExecutorV1 for sllm_core::Ministral3ExecutionRequest {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if include_last_logits {
            return Err(GenerationServiceError::Execution(
                "Ministral 3 exposes reviewed device argmax only".to_owned(),
            ));
        }
        let input = input_token_ids
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = self
            .prefill(&input)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        ministral3_generation_step(output.token_ids())
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        if include_last_logits {
            return Err(GenerationServiceError::Execution(
                "Ministral 3 exposes reviewed device argmax only".to_owned(),
            ));
        }
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = self
            .decode(token)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        ministral3_generation_step(output.token_ids())
    }

    fn cancel(&mut self) {
        // Cancellation is observed by the shared generation service between
        // synchronous HIP transitions. No in-flight native abort ABI exists.
    }
}

fn ministral3_generation_step(
    token_ids: &[i32],
) -> Result<GenerationStepV1, GenerationServiceError> {
    let [token_id] = token_ids else {
        return Err(GenerationServiceError::MissingDeviceArgmax);
    };
    Ok(GenerationStepV1::new(
        u32::try_from(*token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn digest(value: &str) -> String {
        format!("{:x}", Sha256::digest(value.as_bytes()))
    }

    fn digest_u32_le(values: &[u32]) -> String {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn system_user_fixture_is_byte_exact() {
        let messages = [
            ChatMessageV1::system("Answer briefly."),
            ChatMessageV1::user("What is 2+2?"),
        ];
        let rendered = Ministral3ChatRendererV1::new()
            .render(&messages)
            .expect("fixture renders");
        assert_eq!(
            rendered,
            "<s>[SYSTEM_PROMPT]Answer briefly.[/SYSTEM_PROMPT][INST]What is 2+2?[/INST]"
        );
        assert_eq!(
            digest(&rendered),
            MINISTRAL3_SYSTEM_USER_FIXTURE_RENDERED_SHA256
        );
        assert_eq!(
            digest_u32_le(MINISTRAL3_SYSTEM_USER_FIXTURE_TOKEN_IDS_V1),
            MINISTRAL3_SYSTEM_USER_FIXTURE_TOKEN_IDS_SHA256
        );
    }

    #[test]
    fn history_fixture_is_byte_exact() {
        let messages = [
            ChatMessageV1::system("Answer briefly."),
            ChatMessageV1::user("Say A."),
            ChatMessageV1::assistant("A.", None),
            ChatMessageV1::user("Now B."),
        ];
        let rendered = Ministral3ChatRendererV1::new()
            .render_history_prefix(&messages)
            .expect("history renders");
        assert_eq!(
            rendered,
            "<s>[SYSTEM_PROMPT]Answer briefly.[/SYSTEM_PROMPT][INST]Say A.[/INST]A.</s>[INST]Now B.[/INST]"
        );
        assert_eq!(
            digest(&rendered),
            MINISTRAL3_HISTORY_FIXTURE_RENDERED_SHA256
        );
        assert_eq!(
            digest_u32_le(MINISTRAL3_HISTORY_FIXTURE_TOKEN_IDS_V1),
            MINISTRAL3_HISTORY_FIXTURE_TOKEN_IDS_SHA256
        );
    }

    #[test]
    fn default_system_prompt_matches_official_shape() {
        let rendered = Ministral3ChatRendererV1::new()
            .render(&[ChatMessageV1::user("Hello")])
            .expect("default system prompt renders");
        assert!(rendered.starts_with("<s>[SYSTEM_PROMPT]You are Ministral-3-3B-Instruct-2512"));
        assert!(rendered.ends_with("[/SYSTEM_PROMPT][INST]Hello[/INST]"));
        assert_eq!(MINISTRAL3_DEFAULT_SYSTEM_PROMPT.len(), 2_406);
    }

    #[test]
    fn malformed_ordering_and_control_fields_are_rejected() {
        let renderer = Ministral3ChatRendererV1::new();
        assert!(matches!(
            renderer.render(&[ChatMessageV1::assistant("answer", None)]),
            Err(Ministral3FrontendErrorV1::NoUserMessage)
        ));
        assert!(matches!(
            renderer.render(&[ChatMessageV1::user("a"), ChatMessageV1::user("b"),]),
            Err(Ministral3FrontendErrorV1::InvalidMessageOrdering { .. })
        ));
        assert!(matches!(
            renderer.render(&[
                ChatMessageV1::user("a"),
                ChatMessageV1::assistant("b", Some("hidden".to_owned())),
            ]),
            Err(Ministral3FrontendErrorV1::UnsupportedReasoningContent { index: 1 })
        ));
    }

    #[test]
    fn host_output_limit_is_fail_closed() {
        let renderer = Ministral3ChatRendererV1::new();
        assert!(matches!(
            renderer.render_with_options(
                &[ChatMessageV1::user("hello")],
                Ministral3RenderOptionsV1::with_output_limit_bytes(4),
            ),
            Err(Ministral3FrontendErrorV1::OutputTooLarge { limit_bytes: 4 })
        ));
        assert!(matches!(
            renderer.render_with_options(
                &[ChatMessageV1::user("hello")],
                Ministral3RenderOptionsV1::with_output_limit_bytes(
                    MINISTRAL3_CHAT_MAX_OUTPUT_BYTES_V1 + 1
                ),
            ),
            Err(Ministral3FrontendErrorV1::OutputLimitExceedsHostCap)
        ));
    }

    #[test]
    fn tokenizer_asset_identity_rejects_untrusted_bytes() {
        assert!(matches!(
            Ministral3TokenizerV1::from_verified_bytes(b"not a tokenizer"),
            Err(Ministral3FrontendErrorV1::InvalidTokenizerIdentity)
        ));
    }

    #[test]
    fn generation_stop_policy_is_fixed_eos_and_reviewed() {
        let policy = ministral3_generation_stop_policy().expect("reviewed policy validates");
        assert_eq!(policy.stop_token_ids, [2]);
        assert_eq!(policy.evaluation, StopEvaluation::NewlyGeneratedAfterArgmax);
        assert_eq!(policy.prompt_evaluation, PromptEvaluation::NeverStop);
        assert_eq!(
            policy.stop_token,
            StopTokenHandling {
                visible_output: false,
                subsequent_decode_input: false,
            }
        );
        assert_eq!(policy.budget_boundary, BudgetBoundary::StopTokenWins);
        assert_eq!(
            policy.max_new_tokens_zero,
            MaxNewTokensZero::MaxNewTokensBeforeDecode
        );
    }

    #[test]
    #[ignore = "requires the downloaded official 6.4 GiB GGUF in /tmp"]
    fn official_gguf_metadata_reconstructs_tokenizer_and_fixtures() {
        let path = "/tmp/sllm-phase60.2pDfxs/Ministral-3-3B-Instruct-2512-BF16.gguf";
        let gguf = sllm_core::VerifiedGguf::open(path).expect("official GGUF opens");
        let source = sllm_core::verify_official_ministral3_gguf(gguf)
            .expect("official GGUF identity verifies");
        let tokenizer = Ministral3TokenizerV1::from_verified_gguf(&source)
            .expect("GGUF tokenizer metadata reconstructs");
        let renderer = Ministral3ChatRendererV1::from_verified_gguf(&source)
            .expect("embedded chat template verifies");

        let system_user = renderer
            .render(&[
                ChatMessageV1::system("Answer briefly."),
                ChatMessageV1::user("What is 2+2?"),
            ])
            .expect("system/user fixture renders");
        assert_eq!(
            tokenizer
                .encode(&system_user)
                .expect("system/user fixture tokenizes")
                .as_slice(),
            MINISTRAL3_SYSTEM_USER_FIXTURE_TOKEN_IDS_V1
        );

        let history = renderer
            .render_history_prefix(&[
                ChatMessageV1::system("Answer briefly."),
                ChatMessageV1::user("Say A."),
                ChatMessageV1::assistant("A.", None),
                ChatMessageV1::user("Now B."),
            ])
            .expect("history fixture renders");
        assert_eq!(
            tokenizer
                .encode(&history)
                .expect("history fixture tokenizes")
                .as_slice(),
            MINISTRAL3_HISTORY_FIXTURE_TOKEN_IDS_V1
        );

        let source_template = std::fs::read("/tmp/sllm-phase60.2pDfxs/chat_template.jinja")
            .expect("source template exists");
        Ministral3ChatRendererV1::from_verified_template_bytes(&source_template)
            .expect("source template identity verifies");
        let embedded_template = match source.gguf().metadata_value("tokenizer.chat_template") {
            Some(sllm_core::GgufValue::String(value)) => value.as_bytes(),
            _ => panic!("embedded template metadata is absent"),
        };
        Ministral3ChatRendererV1::from_verified_embedded_template_bytes(embedded_template)
            .expect("embedded template identity verifies");
    }
}
