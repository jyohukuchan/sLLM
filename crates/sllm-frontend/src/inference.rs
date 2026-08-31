//! Transport-independent tokenizer and reviewed-template utilities.
//!
//! This module is deliberately small: it owns no model execution state and
//! does not introduce another rendering or tokenization implementation.  A
//! service borrows the already verified tokenizer and, when the model has one,
//! the already verified Qwen renderer.  CLI and HTTP adapters can therefore
//! use exactly the same special-token, byte-piece, and template semantics.

use core::fmt;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::{
    ChatRenderError, DecodeModeV1, GENERIC_TEMPLATE_PROFILE_VERSION_V1, GenericTemplateContextV1,
    GenericTemplateErrorV1, GenericTemplateIdentityV1, GenericTemplateProviderV1,
    QWEN35_CHAT_RENDERER_VERSION, QWEN35_CHAT_TEMPLATE_SHA256, QWEN35_CHAT_TEMPLATE_SIZE_BYTES,
    Qwen35ChatMessageV1, Qwen35ChatTemplateV1, Qwen35RenderOptionsV1, TokenIdsV1, TokenizerError,
    TokenizerFrontendV1,
};

/// Version of the transport-independent tokenizer utility contract.
pub const TOKENIZER_UTILITY_VERSION_V1: u8 = 1;

/// Host-side bound shared by the CLI and HTTP utility adapters.
pub const MAX_TOKENIZER_UTILITY_INPUT_BYTES_V1: usize = 16 * 1024 * 1024;

/// Version of the capability-bound FIM marker ordering.
pub const FIM_TEMPLATE_VERSION_V1: u8 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FimTemplateErrorV1 {
    DuplicateMarker,
    EmptyContent,
    TokenCountOverflow,
}

impl fmt::Display for FimTemplateErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateMarker => formatter.write_str("FIM marker token IDs must be distinct"),
            Self::EmptyContent => {
                formatter.write_str("FIM requires a nonempty prefix, suffix, or prompt")
            }
            Self::TokenCountOverflow => formatter.write_str("FIM token count overflowed usize"),
        }
    }
}

impl std::error::Error for FimTemplateErrorV1 {}

/// Verified prefix/suffix/middle marker identity for one model capability.
///
/// Rendering order is fixed as prefix-marker, optional instruction prompt,
/// prefix content, suffix-marker, suffix content, middle-marker. The digest is
/// bound by a model registry capability; arbitrary client templates are not
/// accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FimTemplateV1 {
    prefix_token_id: u32,
    suffix_token_id: u32,
    middle_token_id: u32,
    digest: String,
}

impl FimTemplateV1 {
    pub fn new(
        prefix_token_id: u32,
        suffix_token_id: u32,
        middle_token_id: u32,
    ) -> Result<Self, FimTemplateErrorV1> {
        if prefix_token_id == suffix_token_id
            || prefix_token_id == middle_token_id
            || suffix_token_id == middle_token_id
        {
            return Err(FimTemplateErrorV1::DuplicateMarker);
        }
        let mut hasher = Sha256::new();
        hasher.update(b"sllm-fim-template-v1\0prefix-prompt-prefix-suffix-suffix-middle\0");
        hasher.update(prefix_token_id.to_le_bytes());
        hasher.update(suffix_token_id.to_le_bytes());
        hasher.update(middle_token_id.to_le_bytes());
        let digest = format!("sha256:{:x}", hasher.finalize());
        Ok(Self {
            prefix_token_id,
            suffix_token_id,
            middle_token_id,
            digest,
        })
    }

    pub const fn version(&self) -> u8 {
        FIM_TEMPLATE_VERSION_V1
    }

    pub const fn prefix_token_id(&self) -> u32 {
        self.prefix_token_id
    }

    pub const fn suffix_token_id(&self) -> u32 {
        self.suffix_token_id
    }

    pub const fn middle_token_id(&self) -> u32 {
        self.middle_token_id
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn render(
        &self,
        prefix: &[u32],
        suffix: &[u32],
        prompt: Option<&[u32]>,
    ) -> Result<TokenIdsV1, FimTemplateErrorV1> {
        if prefix.is_empty() && suffix.is_empty() && prompt.is_none_or(<[u32]>::is_empty) {
            return Err(FimTemplateErrorV1::EmptyContent);
        }
        let prompt_len = prompt.map_or(0, <[u32]>::len);
        let capacity = 3_usize
            .checked_add(prefix.len())
            .and_then(|value| value.checked_add(suffix.len()))
            .and_then(|value| value.checked_add(prompt_len))
            .ok_or(FimTemplateErrorV1::TokenCountOverflow)?;
        let mut tokens = Vec::with_capacity(capacity);
        tokens.push(self.prefix_token_id);
        if let Some(prompt) = prompt {
            tokens.extend_from_slice(prompt);
        }
        tokens.extend_from_slice(prefix);
        tokens.push(self.suffix_token_id);
        tokens.extend_from_slice(suffix);
        tokens.push(self.middle_token_id);
        Ok(TokenIdsV1::from_slice(&tokens))
    }
}

/// How a token's decoder-aware piece is represented in a utility response.
///
/// A byte fallback is not converted through replacement characters.  Valid
/// UTF-8 is represented as [`Self::Utf8`], while arbitrary bytes are retained
/// losslessly in [`Self::Bytes`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenPieceV1 {
    Utf8(String),
    Bytes(Vec<u8>),
}

impl TokenPieceV1 {
    pub fn as_utf8(&self) -> Option<&str> {
        match self {
            Self::Utf8(value) => Some(value),
            Self::Bytes(_) => None,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Utf8(value) => value.as_bytes(),
            Self::Bytes(value) => value,
        }
    }
}

/// Optional detail requested from [`TokenizerUtilityServiceV1::tokenize`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenizeOptionsV1 {
    include_pieces: bool,
}

impl TokenizeOptionsV1 {
    pub const fn new() -> Self {
        Self {
            include_pieces: false,
        }
    }

    pub const fn with_pieces() -> Self {
        Self {
            include_pieces: true,
        }
    }

    pub const fn include_pieces(self) -> bool {
        self.include_pieces
    }

    pub const fn without_pieces(self) -> Self {
        Self {
            include_pieces: false,
        }
    }
}

/// Result of model-default tokenization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizeResultV1 {
    version: u8,
    token_ids: TokenIdsV1,
    pieces: Option<Vec<TokenPieceV1>>,
}

impl TokenizeResultV1 {
    fn new(token_ids: TokenIdsV1, pieces: Option<Vec<TokenPieceV1>>) -> Self {
        Self {
            version: TOKENIZER_UTILITY_VERSION_V1,
            token_ids,
            pieces,
        }
    }

    /// Constructs a utility result from a tokenizer whose model identity and
    /// decoder table were already verified by a trusted backend.
    pub fn from_verified_parts(
        token_ids: TokenIdsV1,
        pieces: Option<Vec<TokenPieceV1>>,
    ) -> Result<Self, TokenizerUtilityErrorV1> {
        if pieces
            .as_ref()
            .is_some_and(|pieces| pieces.len() != token_ids.len())
        {
            return Err(TokenizerUtilityErrorV1::InvalidTokenPieceCount);
        }
        Ok(Self::new(token_ids, pieces))
    }

    pub const fn version(&self) -> u8 {
        self.version
    }

    pub fn token_ids(&self) -> &TokenIdsV1 {
        &self.token_ids
    }

    pub fn ids(&self) -> &TokenIdsV1 {
        self.token_ids()
    }

    pub fn count(&self) -> usize {
        self.token_ids.len()
    }

    pub fn token_count(&self) -> usize {
        self.count()
    }

    pub fn pieces(&self) -> Option<&[TokenPieceV1]> {
        self.pieces.as_deref()
    }
}

/// Identity of the reviewed renderer used for an applied template.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemplateIdentityV1 {
    kind: String,
    version: u8,
    consistency_label: String,
    digest: String,
    size_bytes: u64,
}

impl TemplateIdentityV1 {
    fn qwen35(renderer: &Qwen35ChatTemplateV1) -> Self {
        Self {
            kind: "qwen35-chat-template-v1".to_owned(),
            version: renderer.version(),
            consistency_label: renderer.consistency_label().to_owned(),
            digest: QWEN35_CHAT_TEMPLATE_SHA256.to_owned(),
            size_bytes: QWEN35_CHAT_TEMPLATE_SIZE_BYTES,
        }
    }

    /// Constructs an identity already verified by a trusted model backend.
    /// The structural checks prevent malformed or floating identities from
    /// crossing the public utility boundary.
    pub fn from_verified_parts(
        kind: impl Into<String>,
        version: u8,
        consistency_label: impl Into<String>,
        digest: impl Into<String>,
        size_bytes: u64,
    ) -> Result<Self, TokenizerUtilityErrorV1> {
        let kind = kind.into();
        let consistency_label = consistency_label.into();
        let digest = digest.into();
        if kind.is_empty()
            || kind.len() > 128
            || !kind
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            || version == 0
            || consistency_label.is_empty()
            || consistency_label.len() > 256
            || size_bytes == 0
            || digest.len() != 71
            || !digest.starts_with("sha256:")
            || !digest[7..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(TokenizerUtilityErrorV1::InvalidTemplateIdentity);
        }
        Ok(Self {
            kind,
            version,
            consistency_label,
            digest,
            size_bytes,
        })
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub const fn version(&self) -> u8 {
        self.version
    }

    pub const fn renderer_version(&self) -> u8 {
        self.version()
    }

    pub fn consistency_label(&self) -> &str {
        &self.consistency_label
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn template_digest(&self) -> &str {
        self.digest()
    }

    pub fn template_sha256(&self) -> &str {
        self.digest()
    }

    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

/// Result of applying a reviewed chat template and tokenizing its output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ApplyTemplateResultV1 {
    version: u8,
    rendered: String,
    token_ids: TokenIdsV1,
    identity: TemplateIdentityV1,
    generic_identity: Option<GenericTemplateIdentityV1>,
}

impl ApplyTemplateResultV1 {
    fn new(rendered: String, token_ids: TokenIdsV1, identity: TemplateIdentityV1) -> Self {
        Self {
            version: TOKENIZER_UTILITY_VERSION_V1,
            rendered,
            token_ids,
            identity,
            generic_identity: None,
        }
    }

    fn new_generic(
        rendered: String,
        token_ids: TokenIdsV1,
        identity: TemplateIdentityV1,
        generic_identity: GenericTemplateIdentityV1,
    ) -> Self {
        Self {
            version: TOKENIZER_UTILITY_VERSION_V1,
            rendered,
            token_ids,
            identity,
            generic_identity: Some(generic_identity),
        }
    }

    /// Builds a result from a backend-verified rendered prompt and identity.
    /// This is the adapter seam for a reviewed non-Qwen template provider.
    pub fn from_verified_parts(
        rendered: String,
        token_ids: TokenIdsV1,
        identity: TemplateIdentityV1,
    ) -> Result<Self, TokenizerUtilityErrorV1> {
        if rendered.is_empty() || token_ids.is_empty() {
            return Err(TokenizerUtilityErrorV1::InvalidTemplateResult);
        }
        Ok(Self::new(rendered, token_ids, identity))
    }

    pub const fn version(&self) -> u8 {
        self.version
    }

    pub fn rendered(&self) -> &str {
        &self.rendered
    }

    pub fn text(&self) -> &str {
        self.rendered()
    }

    pub fn token_ids(&self) -> &TokenIdsV1 {
        &self.token_ids
    }

    pub fn ids(&self) -> &TokenIdsV1 {
        self.token_ids()
    }

    pub fn count(&self) -> usize {
        self.token_ids.len()
    }

    pub fn token_count(&self) -> usize {
        self.count()
    }

    pub fn identity(&self) -> &TemplateIdentityV1 {
        &self.identity
    }

    /// Returns the generic provider's full source/kwargs/render identity when
    /// this result was produced through the explicit generic adapter.
    pub fn generic_identity(&self) -> Option<&GenericTemplateIdentityV1> {
        self.generic_identity.as_ref()
    }

    pub fn kwargs_digest(&self) -> Option<&str> {
        self.generic_identity
            .as_ref()
            .map(GenericTemplateIdentityV1::kwargs_digest)
    }
}

/// Typed generic-template message context.  The provider receives only this
/// JSON-like context; raw prompts are intentionally a different rejected
/// input variant below.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenericTemplateMessagesInputV1 {
    context: GenericTemplateContextV1,
}

impl GenericTemplateMessagesInputV1 {
    pub fn new(messages: Vec<Value>) -> Result<Self, TokenizerUtilityErrorV1> {
        Self::from_parts(messages, Map::new(), Map::new(), true, false, None)
    }

    pub fn from_parts(
        messages: Vec<Value>,
        kwargs: Map<String, Value>,
        special_tokens: Map<String, Value>,
        add_generation_prompt: bool,
        enable_thinking: bool,
        reasoning_effort: Option<String>,
    ) -> Result<Self, TokenizerUtilityErrorV1> {
        validate_generic_messages(&messages)?;
        validate_special_tokens(&special_tokens)?;
        validate_reasoning_effort(reasoning_effort.as_deref())?;
        let mut object = Map::new();
        object.insert("messages".to_owned(), Value::Array(messages));
        let kwargs = Value::Object(kwargs);
        object.insert("custom_kwargs".to_owned(), kwargs.clone());
        object.insert("kwargs".to_owned(), kwargs);
        object.insert("special_tokens".to_owned(), Value::Object(special_tokens));
        object.insert(
            "add_generation_prompt".to_owned(),
            Value::Bool(add_generation_prompt),
        );
        object.insert("enable_thinking".to_owned(), Value::Bool(enable_thinking));
        if let Some(reasoning_effort) = reasoning_effort {
            object.insert(
                "reasoning_effort".to_owned(),
                Value::String(reasoning_effort),
            );
        }
        let context = GenericTemplateContextV1::new(Value::Object(object))
            .map_err(TokenizerUtilityErrorV1::GenericTemplate)?;
        Ok(Self { context })
    }

    pub fn from_context(
        context: GenericTemplateContextV1,
    ) -> Result<Self, TokenizerUtilityErrorV1> {
        let object =
            context
                .value()
                .as_object()
                .ok_or(TokenizerUtilityErrorV1::GenericTemplate(
                    GenericTemplateErrorV1::ContextNotObject,
                ))?;
        let messages = object.get("messages").and_then(Value::as_array).ok_or(
            TokenizerUtilityErrorV1::GenericTemplate(GenericTemplateErrorV1::MessagesNotArray),
        )?;
        validate_generic_messages(messages)?;
        if let Some(special_tokens) = object.get("special_tokens") {
            let special_tokens =
                special_tokens
                    .as_object()
                    .ok_or(TokenizerUtilityErrorV1::GenericTemplate(
                        GenericTemplateErrorV1::InvalidContext,
                    ))?;
            validate_special_tokens(special_tokens)?;
        }
        if let Some(reasoning_effort) = object.get("reasoning_effort").and_then(Value::as_str) {
            validate_reasoning_effort(Some(reasoning_effort))?;
        }
        Ok(Self { context })
    }

    pub fn context(&self) -> &GenericTemplateContextV1 {
        &self.context
    }
}

/// Explicit input mode for generic template application.  `RawText` and
/// `GemmaRawText` are represented so adapters can reject them before calling
/// the tokenizer, rather than silently changing prompt semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenericTemplateInputV1 {
    Json(GenericTemplateContextV1),
    Messages(GenericTemplateMessagesInputV1),
    RawText(String),
    GemmaRawText(String),
}

impl GenericTemplateInputV1 {
    pub fn json(value: Value) -> Result<Self, TokenizerUtilityErrorV1> {
        let context = GenericTemplateContextV1::new(value)
            .map_err(TokenizerUtilityErrorV1::GenericTemplate)?;
        Ok(Self::Json(context))
    }

    pub fn messages(input: GenericTemplateMessagesInputV1) -> Self {
        Self::Messages(input)
    }

    pub fn raw_text(text: impl Into<String>) -> Self {
        Self::RawText(text.into())
    }

    pub fn gemma_raw_text(text: impl Into<String>) -> Self {
        Self::GemmaRawText(text.into())
    }
}

pub type GenericTemplateApplyInputV1 = GenericTemplateInputV1;
pub type GenericTemplateMessagesV1 = GenericTemplateMessagesInputV1;

/// Input accepted by the shared input-token-count operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InputTokenCountInputV1<'a> {
    RawText(&'a str),
    Messages {
        messages: &'a [Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    },
}

/// Error surface for tokenizer/template utility adapters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerUtilityErrorV1 {
    InputTooLarge { bytes: usize, max_bytes: usize },
    Tokenize(TokenizerError),
    Detokenize(TokenizerError),
    TemplateUnavailable,
    ApplyTemplate(ChatRenderError),
    TokenPieceUnavailable { id: u32 },
    InvalidTokenPieceCount,
    InvalidTemplateIdentity,
    InvalidTemplateResult,
    GenericTemplate(GenericTemplateErrorV1),
    UnsupportedGenericTemplateInput { kind: GenericTemplateInputKindV1 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenericTemplateInputKindV1 {
    RawText,
    GemmaRawText,
}

impl fmt::Display for TokenizerUtilityErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLarge { bytes, max_bytes } => {
                write!(
                    formatter,
                    "utility input is {bytes} bytes, maximum is {max_bytes}"
                )
            }
            Self::Tokenize(error) => write!(formatter, "tokenization failed: {error}"),
            Self::Detokenize(error) => write!(formatter, "detokenization failed: {error}"),
            Self::TemplateUnavailable => {
                formatter.write_str("reviewed chat template is unavailable")
            }
            Self::ApplyTemplate(error) => write!(formatter, "template application failed: {error}"),
            Self::TokenPieceUnavailable { id } => {
                write!(formatter, "token {id} has no decoder-aware piece")
            }
            Self::InvalidTokenPieceCount => {
                formatter.write_str("token piece count differs from token ID count")
            }
            Self::InvalidTemplateIdentity => {
                formatter.write_str("verified template identity is malformed")
            }
            Self::InvalidTemplateResult => formatter.write_str("verified template result is empty"),
            Self::GenericTemplate(error) => write!(formatter, "generic template failed: {error}"),
            Self::UnsupportedGenericTemplateInput { kind } => match kind {
                GenericTemplateInputKindV1::RawText => {
                    formatter.write_str("generic template does not accept raw-text input")
                }
                GenericTemplateInputKindV1::GemmaRawText => {
                    formatter.write_str("generic template is unsupported for Gemma raw-text input")
                }
            },
        }
    }
}

impl std::error::Error for TokenizerUtilityErrorV1 {}

/// Shared, transport-independent tokenizer and reviewed-template service.
///
/// The service intentionally borrows both objects.  Production model owners
/// retain ownership and can hand the same resident instances to CLI/HTTP
/// adapters without loading a second tokenizer or renderer.
pub struct TokenizerUtilityServiceV1<'a> {
    tokenizer: &'a TokenizerFrontendV1,
    renderer: Option<&'a Qwen35ChatTemplateV1>,
}

impl<'a> TokenizerUtilityServiceV1<'a> {
    pub fn new(
        tokenizer: &'a TokenizerFrontendV1,
        renderer: Option<&'a Qwen35ChatTemplateV1>,
    ) -> Self {
        Self {
            tokenizer,
            renderer,
        }
    }

    pub fn tokenizer(&self) -> &'a TokenizerFrontendV1 {
        self.tokenizer
    }

    pub fn renderer(&self) -> Option<&'a Qwen35ChatTemplateV1> {
        self.renderer
    }

    /// Tokenizes using the model's verified default special-token policy.
    pub fn tokenize(
        &self,
        text: &str,
        options: TokenizeOptionsV1,
    ) -> Result<TokenizeResultV1, TokenizerUtilityErrorV1> {
        ensure_input_size(text)?;
        let token_ids = self
            .tokenizer
            .encode(text)
            .map_err(TokenizerUtilityErrorV1::Tokenize)?;
        let pieces = options
            .include_pieces()
            .then(|| self.decoder_pieces(&token_ids))
            .transpose()?;
        Ok(TokenizeResultV1::new(token_ids, pieces))
    }

    pub fn tokenize_default(
        &self,
        text: &str,
    ) -> Result<TokenizeResultV1, TokenizerUtilityErrorV1> {
        self.tokenize(text, TokenizeOptionsV1::default())
    }

    pub fn tokenize_with_pieces(
        &self,
        text: &str,
    ) -> Result<TokenizeResultV1, TokenizerUtilityErrorV1> {
        self.tokenize(text, TokenizeOptionsV1::with_pieces())
    }

    /// Decodes IDs with explicit preserve/skip-special semantics.
    pub fn detokenize(
        &self,
        token_ids: &TokenIdsV1,
        mode: DecodeModeV1,
    ) -> Result<String, TokenizerUtilityErrorV1> {
        self.tokenizer
            .decode(token_ids, mode)
            .map_err(TokenizerUtilityErrorV1::Detokenize)
    }

    pub fn detokenize_ids(
        &self,
        token_ids: &[u32],
        mode: DecodeModeV1,
    ) -> Result<String, TokenizerUtilityErrorV1> {
        self.detokenize(&TokenIdsV1::from_slice(token_ids), mode)
    }

    /// Applies the reviewed Qwen template, then tokenizes its exact output.
    pub fn apply_template(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<ApplyTemplateResultV1, TokenizerUtilityErrorV1> {
        let renderer = self
            .renderer
            .ok_or(TokenizerUtilityErrorV1::TemplateUnavailable)?;
        if renderer.version() != QWEN35_CHAT_RENDERER_VERSION {
            return Err(TokenizerUtilityErrorV1::ApplyTemplate(
                ChatRenderError::UnsupportedRendererVersion {
                    actual: renderer.version(),
                },
            ));
        }
        let rendered = renderer
            .render(messages, options)
            .map_err(TokenizerUtilityErrorV1::ApplyTemplate)?;
        ensure_input_size(&rendered)?;
        let token_ids = self
            .tokenizer
            .encode(&rendered)
            .map_err(TokenizerUtilityErrorV1::Tokenize)?;
        Ok(ApplyTemplateResultV1::new(
            rendered,
            token_ids,
            TemplateIdentityV1::qwen35(renderer),
        ))
    }

    /// Counts tokens for raw text without rendering or allocating GPU state.
    pub fn input_token_count(
        &self,
        input: InputTokenCountInputV1<'_>,
    ) -> Result<usize, TokenizerUtilityErrorV1> {
        match input {
            InputTokenCountInputV1::RawText(text) => Ok(self.tokenize_default(text)?.count()),
            InputTokenCountInputV1::Messages { messages, options } => {
                Ok(self.apply_template(messages, options)?.count())
            }
        }
    }

    pub fn input_token_count_raw(&self, text: &str) -> Result<usize, TokenizerUtilityErrorV1> {
        self.input_token_count(InputTokenCountInputV1::RawText(text))
    }

    pub fn input_token_count_messages(
        &self,
        messages: &[Qwen35ChatMessageV1],
        options: Qwen35RenderOptionsV1,
    ) -> Result<usize, TokenizerUtilityErrorV1> {
        self.input_token_count(InputTokenCountInputV1::Messages { messages, options })
    }

    /// Applies an explicitly opted-in generic provider.  The reviewed Qwen
    /// path above remains the default and retains its byte/token identity.
    pub fn apply_generic_template(
        &self,
        provider: &GenericTemplateProviderV1,
        input: GenericTemplateInputV1,
    ) -> Result<ApplyTemplateResultV1, TokenizerUtilityErrorV1> {
        let context = match input {
            GenericTemplateInputV1::Json(context) => context,
            GenericTemplateInputV1::Messages(messages) => messages.context.clone(),
            GenericTemplateInputV1::RawText(_) => {
                return Err(TokenizerUtilityErrorV1::UnsupportedGenericTemplateInput {
                    kind: GenericTemplateInputKindV1::RawText,
                });
            }
            GenericTemplateInputV1::GemmaRawText(_) => {
                return Err(TokenizerUtilityErrorV1::UnsupportedGenericTemplateInput {
                    kind: GenericTemplateInputKindV1::GemmaRawText,
                });
            }
        };
        let rendered = provider
            .render_context(&context)
            .map_err(TokenizerUtilityErrorV1::GenericTemplate)?;
        ensure_input_size(rendered.rendered())?;
        let token_ids = self
            .tokenizer
            .encode(rendered.rendered())
            .map_err(TokenizerUtilityErrorV1::Tokenize)?;
        if rendered.rendered().is_empty() || token_ids.is_empty() {
            return Err(TokenizerUtilityErrorV1::InvalidTemplateResult);
        }
        let identity = TemplateIdentityV1::from_verified_parts(
            "generic-jinja-v1",
            GENERIC_TEMPLATE_PROFILE_VERSION_V1,
            "minijinja-2.24.0",
            provider.digest().to_owned(),
            provider.source_size_bytes() as u64,
        )?;
        Ok(ApplyTemplateResultV1::new_generic(
            rendered.rendered().to_owned(),
            token_ids,
            identity,
            rendered.identity().clone(),
        ))
    }

    pub fn apply_template_generic(
        &self,
        provider: &GenericTemplateProviderV1,
        input: GenericTemplateInputV1,
    ) -> Result<ApplyTemplateResultV1, TokenizerUtilityErrorV1> {
        self.apply_generic_template(provider, input)
    }

    pub fn input_token_count_generic(
        &self,
        provider: &GenericTemplateProviderV1,
        input: GenericTemplateInputV1,
    ) -> Result<usize, TokenizerUtilityErrorV1> {
        Ok(self.apply_generic_template(provider, input)?.count())
    }

    fn decoder_pieces(
        &self,
        token_ids: &TokenIdsV1,
    ) -> Result<Vec<TokenPieceV1>, TokenizerUtilityErrorV1> {
        token_ids
            .as_slice()
            .iter()
            .map(|&id| {
                let entry = self
                    .tokenizer
                    .token_byte_table()
                    .get(id)
                    .map_err(|_| TokenizerUtilityErrorV1::TokenPieceUnavailable { id })?;
                if let Some(bytes) = entry.bytes() {
                    if let Ok(text) = core::str::from_utf8(bytes) {
                        Ok(TokenPieceV1::Utf8(text.to_owned()))
                    } else {
                        Ok(TokenPieceV1::Bytes(bytes.to_vec()))
                    }
                } else if let Some(piece) = entry.piece() {
                    Ok(TokenPieceV1::Utf8(piece.to_owned()))
                } else {
                    Err(TokenizerUtilityErrorV1::TokenPieceUnavailable { id })
                }
            })
            .collect()
    }
}

fn validate_generic_messages(messages: &[Value]) -> Result<(), TokenizerUtilityErrorV1> {
    if messages.len() > crate::GENERIC_TEMPLATE_MAX_MESSAGES_V1 {
        return Err(TokenizerUtilityErrorV1::GenericTemplate(
            GenericTemplateErrorV1::TooManyMessages {
                count: messages.len(),
                max_messages: crate::GENERIC_TEMPLATE_MAX_MESSAGES_V1,
            },
        ));
    }
    for message in messages {
        let Some(message) = message.as_object() else {
            return Err(TokenizerUtilityErrorV1::GenericTemplate(
                GenericTemplateErrorV1::InvalidContext,
            ));
        };
        if message.get("role").and_then(Value::as_str).is_none() || !message.contains_key("content")
        {
            return Err(TokenizerUtilityErrorV1::GenericTemplate(
                GenericTemplateErrorV1::InvalidContext,
            ));
        }
    }
    Ok(())
}

fn validate_special_tokens(tokens: &Map<String, Value>) -> Result<(), TokenizerUtilityErrorV1> {
    for value in tokens.values() {
        if !value.is_string() {
            return Err(TokenizerUtilityErrorV1::GenericTemplate(
                GenericTemplateErrorV1::InvalidContext,
            ));
        }
    }
    Ok(())
}

fn validate_reasoning_effort(value: Option<&str>) -> Result<(), TokenizerUtilityErrorV1> {
    if let Some(value) = value {
        if value.is_empty() || value.len() > 32 || !value.is_ascii() {
            return Err(TokenizerUtilityErrorV1::GenericTemplate(
                GenericTemplateErrorV1::InvalidContext,
            ));
        }
    }
    Ok(())
}

fn ensure_input_size(text: &str) -> Result<(), TokenizerUtilityErrorV1> {
    let bytes = text.len();
    if bytes > MAX_TOKENIZER_UTILITY_INPUT_BYTES_V1 {
        return Err(TokenizerUtilityErrorV1::InputTooLarge {
            bytes,
            max_bytes: MAX_TOKENIZER_UTILITY_INPUT_BYTES_V1,
        });
    }
    Ok(())
}
