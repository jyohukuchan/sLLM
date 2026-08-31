use core::fmt;
use std::collections::{BTreeMap, HashMap, HashSet};

use sllm_core::{
    GEMMA4_MOE_MODEL_FINGERPRINT, Gemma4ModelLock, Gemma4TokenizerContract, ModelLock,
    StopIdentity, TokenizerContract, VerifiedCache, VerifiedGguf,
};
use tokenizers::{AddedToken, Tokenizer};

use crate::{StopPolicyError, validate_generation_stop_policy};

const QWEN35_MOE_TOKENIZER_CONTRACT: &str = r#"{
  "files":["chat_template.jinja","merges.txt","tokenizer.json","tokenizer_config.json","vocab.json"],
  "chat_template_path":"chat_template.jinja",
  "vocab_size":248320,
  "eos_token_id":248044,
  "special_token_ids":{"vision_start":248053,"vision_end":248054,"vision_pad":248055,"image_pad":248056,"video_pad":248057},
  "stop_identity":{
    "config_eos":{"token":"<|endoftext|>","token_id":248044,"source_file":"config.json"},
    "tokenizer_eos":{"token":"<|im_end|>","token_id":248046,"source_files":["tokenizer_config.json","tokenizer.json"]}
  },
  "generation_stop_policy":{
    "version":1,"stop_token_ids":[248046,248044],"evaluation":"newly_generated_after_argmax",
    "prompt_evaluation":"never_stop","stop_token":{"visible_output":false,"subsequent_decode_input":false},
    "budget_boundary":"stop_token_wins","max_new_tokens_zero":"max_new_tokens_before_decode","reason_version":1
  }
}"#;

const GEMMA4_MOE_SEMANTIC_PREFIX: &str = "gemma4moe:";

pub(crate) fn has_reviewed_gemma4_moe_gguf_identity(gguf: &VerifiedGguf) -> bool {
    let Some(extension) = gguf.extension() else {
        return false;
    };
    has_reviewed_gemma4_moe_identity_parts(
        gguf.architecture(),
        &extension.recipe.semantic_model_id,
        &extension.recipe.source_lock_fingerprints,
    )
}

fn has_reviewed_gemma4_moe_identity_parts(
    architecture: &str,
    semantic_model_id: &str,
    source_lock_fingerprints: &[String],
) -> bool {
    architecture == "gemma4moe"
        && semantic_model_id.strip_prefix(GEMMA4_MOE_SEMANTIC_PREFIX)
            == Some(GEMMA4_MOE_MODEL_FINGERPRINT)
        && source_lock_fingerprints == [GEMMA4_MOE_MODEL_FINGERPRINT]
}

fn gemma4_moe_semantic_model_id() -> String {
    format!("{GEMMA4_MOE_SEMANTIC_PREFIX}{GEMMA4_MOE_MODEL_FINGERPRINT}")
}

fn gemma4_moe_tokenizer_contract() -> Gemma4TokenizerContract {
    let special_token_ids = BTreeMap::from([
        ("audio".to_owned(), 258_881),
        ("audio_begin".to_owned(), 256_000),
        ("audio_end".to_owned(), 258_883),
        ("bos".to_owned(), 2),
        ("channel_begin".to_owned(), 100),
        ("channel_end".to_owned(), 101),
        ("eos".to_owned(), 1),
        ("image".to_owned(), 258_880),
        ("image_begin".to_owned(), 255_999),
        ("image_end".to_owned(), 258_882),
        ("mask".to_owned(), 4),
        ("pad".to_owned(), 0),
        ("think".to_owned(), 98),
        ("tool_call_begin".to_owned(), 48),
        ("tool_call_end".to_owned(), 49),
        ("tool_response_begin".to_owned(), 50),
        ("tool_response_end".to_owned(), 51),
        ("turn_begin".to_owned(), 105),
        ("turn_end".to_owned(), 106),
        ("unk".to_owned(), 3),
        ("video".to_owned(), 258_884),
    ]);
    Gemma4TokenizerContract {
        files: vec![
            "chat_template.jinja".to_owned(),
            "tokenizer.json".to_owned(),
            "tokenizer_config.json".to_owned(),
        ],
        tokenizer_class: "GemmaTokenizer".to_owned(),
        vocab_size: 262_144,
        chat_template_path: Some("chat_template.jinja".to_owned()),
        prompt_mode: "chat-template".to_owned(),
        special_token_ids,
        stop_token_ids: sllm_core::gemma4_moe_generation_stop_policy()
            .stop_token_ids
            .into_iter()
            .map(u64::from)
            .collect(),
    }
}

/// Immutable token IDs returned by the versioned tokenizer frontend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenIdsV1 {
    ids: Vec<u32>,
}

impl TokenIdsV1 {
    pub fn from_slice(ids: &[u32]) -> Self {
        Self { ids: ids.to_vec() }
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl AsRef<[u32]> for TokenIdsV1 {
    fn as_ref(&self) -> &[u32] {
        self.as_slice()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeModeV1 {
    SkipSpecialTokens,
    PreserveSpecialTokens,
}

impl DecodeModeV1 {
    const fn skip_special_tokens(self) -> bool {
        matches!(self, Self::SkipSpecialTokens)
    }
}

/// An observed special-token role retained by the validated tokenizer
/// snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpecialTokenSnapshotV1 {
    role: String,
    token_id: u32,
    content: String,
}

impl SpecialTokenSnapshotV1 {
    pub fn role(&self) -> &str {
        &self.role
    }

    pub const fn token_id(&self) -> u32 {
        self.token_id
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

/// A validated EOS identity and the content observed at its tokenizer ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EosIdentitySnapshotV1 {
    token: String,
    token_id: u32,
    observed_content: String,
}

impl EosIdentitySnapshotV1 {
    pub fn token(&self) -> &str {
        &self.token
    }

    pub const fn token_id(&self) -> u32 {
        self.token_id
    }

    pub fn observed_content(&self) -> &str {
        &self.observed_content
    }
}

/// The immutable tokenizer facts validated during frontend construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerSnapshotV1 {
    fingerprint: String,
    /// Model vocabulary capacity. The tokenizer may intentionally occupy only
    /// a prefix because the LM head retains reserved output rows.
    vocab_size: u64,
    special_roles: Vec<SpecialTokenSnapshotV1>,
    config_eos: EosIdentitySnapshotV1,
    tokenizer_eos: EosIdentitySnapshotV1,
    stop_token_ids: Vec<u32>,
}

impl TokenizerSnapshotV1 {
    /// This fingerprint is a consistency label, not a cryptographic lock
    /// binding. Core fingerprint opacity remains outside this frontend.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub const fn vocab_size(&self) -> u64 {
        self.vocab_size
    }

    pub fn special_roles(&self) -> &[SpecialTokenSnapshotV1] {
        &self.special_roles
    }

    pub const fn config_eos(&self) -> &EosIdentitySnapshotV1 {
        &self.config_eos
    }

    pub const fn tokenizer_eos(&self) -> &EosIdentitySnapshotV1 {
        &self.tokenizer_eos
    }

    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenIdContextV1 {
    ContractEos,
    ConfigEos,
    TokenizerEos,
    SpecialRole,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EosIdentityV1 {
    Config,
    Tokenizer,
}

/// Stable construction and use errors for the immutable tokenizer boundary.
///
/// The variants deliberately do not retain errors from the core file reader or
/// the tokenizer parser: those errors may contain local paths or unstable
/// implementation details.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerError {
    LockFingerprintMismatch {
        lock: String,
        cache: String,
    },
    FrontendAssetRead,
    InvalidTokenizer,
    InvalidGenerationStopPolicy,
    TokenIdOverflow {
        context: TokenIdContextV1,
        value: u64,
    },
    VocabSizeOverflow {
        value: usize,
    },
    VocabSizeMismatch {
        lock: u64,
        tokenizer: u64,
    },
    SpecialTokenIdMissing {
        role: String,
        id: u32,
    },
    SpecialTokenDecoderMissing {
        role: String,
        id: u32,
    },
    SpecialTokenNotMarkedSpecial {
        role: String,
        id: u32,
    },
    SpecialTokenContentMismatch {
        role: String,
        id: u32,
    },
    DuplicateSpecialId {
        first_role: String,
        second_role: String,
        id: u32,
    },
    DuplicateSpecialContent {
        first_role: String,
        second_role: String,
    },
    EosContractMismatch {
        contract_id: u32,
        config_id: u32,
    },
    EosTokenToIdMismatch {
        identity: EosIdentityV1,
        id: u32,
    },
    EosIdToTokenMismatch {
        identity: EosIdentityV1,
        id: u32,
    },
    EosAddedTokenMissing {
        identity: EosIdentityV1,
        id: u32,
    },
    EosAddedTokenContentMismatch {
        identity: EosIdentityV1,
        id: u32,
    },
    EosAddedTokenNotMarkedSpecial {
        identity: EosIdentityV1,
        id: u32,
    },
    StopPolicyMismatch {
        expected: Vec<u32>,
        actual: Vec<u32>,
    },
    Encode,
    UnknownTokenId {
        id: u32,
    },
    Decode,
    TokenByteDecoderUnsupported {
        decoder: String,
    },
    TokenByteUnsupported {
        id: u32,
    },
    TokenBytePieceTooLong {
        id: u32,
        len: usize,
    },
    TokenByteTableVocabMismatch {
        id: u32,
        vocab_size: u64,
    },
    TokenByteTableCapacityOverflow {
        value: u64,
    },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LockFingerprintMismatch { .. } => {
                formatter.write_str("model lock and verified cache fingerprints differ")
            }
            Self::FrontendAssetRead => {
                formatter.write_str("verified tokenizer asset could not be read")
            }
            Self::InvalidTokenizer => formatter.write_str("tokenizer bytes are invalid"),
            Self::InvalidGenerationStopPolicy => {
                formatter.write_str("generation stop policy is invalid")
            }
            Self::TokenIdOverflow { .. } => formatter.write_str("token ID does not fit u32"),
            Self::VocabSizeOverflow { .. } => {
                formatter.write_str("tokenizer vocabulary size does not fit u64")
            }
            Self::VocabSizeMismatch { .. } => formatter
                .write_str("tokenizer ID span exceeds the locked model vocabulary capacity"),
            Self::SpecialTokenIdMissing { .. } => {
                formatter.write_str("typed special token ID is unknown")
            }
            Self::SpecialTokenDecoderMissing { .. } => {
                formatter.write_str("typed special token lacks an added-token entry")
            }
            Self::SpecialTokenNotMarkedSpecial { .. } => {
                formatter.write_str("typed special token is not marked special")
            }
            Self::SpecialTokenContentMismatch { .. } => {
                formatter.write_str("typed special token content differs")
            }
            Self::DuplicateSpecialId { .. } => {
                formatter.write_str("typed special token IDs are duplicated")
            }
            Self::DuplicateSpecialContent { .. } => {
                formatter.write_str("typed special token contents are duplicated")
            }
            Self::EosContractMismatch { .. } => formatter.write_str("lock EOS contract IDs differ"),
            Self::EosTokenToIdMismatch { .. } => {
                formatter.write_str("EOS content does not map to its locked ID")
            }
            Self::EosIdToTokenMismatch { .. } => {
                formatter.write_str("locked EOS ID does not map to its content")
            }
            Self::EosAddedTokenMissing { .. } => {
                formatter.write_str("locked EOS lacks an added-token entry")
            }
            Self::EosAddedTokenContentMismatch { .. } => {
                formatter.write_str("locked EOS added-token content differs")
            }
            Self::EosAddedTokenNotMarkedSpecial { .. } => {
                formatter.write_str("locked EOS is not marked special")
            }
            Self::StopPolicyMismatch { .. } => {
                formatter.write_str("generation stop IDs differ from EOS identities")
            }
            Self::Encode => formatter.write_str("tokenizer could not encode text"),
            Self::UnknownTokenId { .. } => {
                formatter.write_str("token sequence contains an unknown ID")
            }
            Self::Decode => formatter.write_str("tokenizer could not decode token IDs"),
            Self::TokenByteDecoderUnsupported { .. } => {
                formatter.write_str("tokenizer decoder cannot be represented as token bytes")
            }
            Self::TokenByteUnsupported { .. } => {
                formatter.write_str("token piece cannot be represented as raw bytes")
            }
            Self::TokenBytePieceTooLong { .. } => {
                formatter.write_str("token piece exceeds the bounded raw-byte length")
            }
            Self::TokenByteTableVocabMismatch { .. } => {
                formatter.write_str("token ID is outside the model vocabulary capacity")
            }
            Self::TokenByteTableCapacityOverflow { .. } => {
                formatter.write_str("model vocabulary capacity does not fit the host index type")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<StopPolicyError> for TokenizerError {
    fn from(_: StopPolicyError) -> Self {
        Self::InvalidGenerationStopPolicy
    }
}

/// Maximum raw-byte length retained for one token piece.  Grammar and
/// structured-output consumers can therefore bound their trie transition
/// work independently of the model vocabulary size.
pub const MAX_TOKEN_PIECE_BYTES_V1: usize = 128;

/// Classification of one immutable vocabulary row.
///
/// `Reserved` rows are in the LM-head capacity but are absent from the
/// tokenizer vocabulary.  They are never treated as an empty token.  A
/// `Special` row is retained for identity/diagnostics but is not eligible for
/// grammar transitions.  `ByteFallback` is a single raw byte emitted by
/// tokenizers' byte-fallback convention; `Ordinary` is a regular piece whose
/// bytes were derived using the tokenizer's decoder metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenPieceClassV1 {
    Ordinary,
    ByteFallback,
    Special,
    Reserved,
}

/// One row in the immutable, token-ID-ordered raw piece table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenByteEntryV1 {
    class: TokenPieceClassV1,
    piece: Option<Box<str>>,
    bytes: Option<Box<[u8]>>,
}

impl TokenByteEntryV1 {
    fn reserved() -> Self {
        Self {
            class: TokenPieceClassV1::Reserved,
            piece: None,
            bytes: None,
        }
    }

    pub const fn class(&self) -> TokenPieceClassV1 {
        self.class
    }

    /// The model vocabulary piece, when this row is present in the tokenizer.
    pub fn piece(&self) -> Option<&str> {
        self.piece.as_deref()
    }

    /// Decoder-aware raw bytes.  Special and reserved rows deliberately return
    /// `None`; callers must branch on [`Self::class`] instead of interpreting
    /// an absent value as an empty token.
    pub fn bytes(&self) -> Option<&[u8]> {
        self.bytes.as_deref()
    }

    pub const fn is_grammar_eligible(&self) -> bool {
        matches!(
            self.class,
            TokenPieceClassV1::Ordinary | TokenPieceClassV1::ByteFallback
        )
    }
}

/// Immutable token-ID-ordered table used by grammar/token-trie builders.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenByteTableV1 {
    entries: Box<[TokenByteEntryV1]>,
}

impl TokenByteTableV1 {
    /// Build a table directly from a tokenizer JSON asset.  Production callers
    /// should obtain the table from [`TokenizerFrontendV1::token_byte_table`]
    /// so the capacity comes from the verified model lock.
    pub fn from_tokenizer_json(
        bytes: &[u8],
        model_vocab_size: u64,
    ) -> Result<Self, TokenizerError> {
        let tokenizer =
            Tokenizer::from_bytes(bytes).map_err(|_| TokenizerError::InvalidTokenizer)?;
        build_token_byte_table(bytes, &tokenizer, model_vocab_size)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn as_slice(&self) -> &[TokenByteEntryV1] {
        &self.entries
    }

    /// Returns `None` only for an ID outside the model capacity.  An ID inside
    /// the capacity always has an explicit ordinary, special, or reserved row.
    pub fn entry(&self, id: u32) -> Option<&TokenByteEntryV1> {
        self.entries.get(usize::try_from(id).ok()?)
    }

    pub fn get(&self, id: u32) -> Result<&TokenByteEntryV1, TokenizerError> {
        self.entry(id).ok_or(TokenizerError::UnknownTokenId { id })
    }
}

/// An immutable tokenizer loaded only from a core-verified model-cache asset.
/// The retained fingerprint is a consistency label, not a cryptographic lock
/// binding; core's mutable public fingerprint fields remain separate debt.
#[derive(Debug)]
pub struct TokenizerFrontendV1 {
    tokenizer: Tokenizer,
    snapshot: TokenizerSnapshotV1,
    token_byte_table: TokenByteTableV1,
    encode_add_special_tokens: bool,
}

impl TokenizerFrontendV1 {
    pub fn from_verified_cache(
        lock: &ModelLock,
        cache: &VerifiedCache,
    ) -> Result<Self, TokenizerError> {
        if cache.lock_fingerprint != lock.fingerprint() {
            return Err(TokenizerError::LockFingerprintMismatch {
                lock: lock.fingerprint().to_owned(),
                cache: cache.lock_fingerprint.clone(),
            });
        }

        let bytes = cache
            .read_frontend_asset(sllm_core::FrontendAssetKind::TokenizerJson)
            .map_err(|_| TokenizerError::FrontendAssetRead)?;
        Self::from_qwen35_bytes(bytes, &lock.model().tokenizer_contract, lock.fingerprint())
    }

    pub fn from_qwen35_gguf(lock: &ModelLock, gguf: &VerifiedGguf) -> Result<Self, TokenizerError> {
        let extension = gguf.extension().ok_or(TokenizerError::FrontendAssetRead)?;
        if gguf.architecture() != "qwen35"
            || !extension
                .recipe
                .source_lock_fingerprints
                .iter()
                .any(|fingerprint| fingerprint == lock.fingerprint())
        {
            return Err(TokenizerError::LockFingerprintMismatch {
                lock: lock.fingerprint().to_owned(),
                cache: extension.recipe.semantic_model_id.clone(),
            });
        }
        let bytes = gguf
            .frontend_asset("tokenizer.json")
            .ok_or(TokenizerError::FrontendAssetRead)?
            .to_vec();
        Self::from_qwen35_bytes(bytes, &lock.model().tokenizer_contract, lock.fingerprint())
    }

    /// Constructs the same Qwen3.5 tokenizer contract from the exact reviewed
    /// MoE artifact, without requiring a Dense-model lock or cache facade.
    pub fn from_qwen35_moe_artifact(
        artifact: &sllm_core::VerifiedQwen35Moe,
    ) -> Result<Self, TokenizerError> {
        let bytes = artifact
            .read_support_file("tokenizer.json")
            .map_err(|_| TokenizerError::FrontendAssetRead)?;
        let contract: TokenizerContract = serde_json::from_str(QWEN35_MOE_TOKENIZER_CONTRACT)
            .map_err(|_| TokenizerError::InvalidTokenizer)?;
        Self::from_qwen35_bytes(bytes, &contract, sllm_core::QWEN35_MOE_MODEL_FINGERPRINT)
    }

    pub fn from_qwen35_moe_gguf(
        source: &sllm_core::VerifiedGgufQwen35Moe,
    ) -> Result<Self, TokenizerError> {
        let bytes = source
            .gguf()
            .frontend_asset("tokenizer.json")
            .ok_or(TokenizerError::FrontendAssetRead)?
            .to_vec();
        let contract: TokenizerContract = serde_json::from_str(QWEN35_MOE_TOKENIZER_CONTRACT)
            .map_err(|_| TokenizerError::InvalidTokenizer)?;
        Self::from_qwen35_bytes(bytes, &contract, sllm_core::QWEN35_MOE_MODEL_FINGERPRINT)
    }

    fn from_qwen35_bytes(
        bytes: Vec<u8>,
        contract: &TokenizerContract,
        fingerprint: &str,
    ) -> Result<Self, TokenizerError> {
        let tokenizer =
            Tokenizer::from_bytes(bytes.clone()).map_err(|_| TokenizerError::InvalidTokenizer)?;

        validate_generation_stop_policy(contract.generation_stop_policy())?;
        let checked = CheckedContract::new(contract)?;

        // `get_vocab_size(false)` excludes added special tokens and therefore
        // is not the model's LM-head vocabulary. Qwen also keeps reserved,
        // currently unassigned output rows above every tokenizer ID. Validate
        // the complete tokenizer ID span against that model capacity instead
        // of requiring either count to be equal to the LM-head width.
        let tokenizer_vocab = tokenizer.get_vocab(true);
        let tokenizer_vocab_size = tokenizer.get_vocab_size(true);
        if tokenizer_vocab.len() != tokenizer_vocab_size {
            return Err(TokenizerError::InvalidTokenizer);
        }
        let tokenizer_vocab_span = tokenizer_vocab
            .values()
            .copied()
            .max()
            .map_or(0_u64, |id| u64::from(id) + 1);
        if tokenizer_vocab_span > contract.vocab_size {
            return Err(TokenizerError::VocabSizeMismatch {
                lock: contract.vocab_size,
                tokenizer: tokenizer_vocab_span,
            });
        }

        // Keep this as the sole decoder-map read.  The map is a snapshot used
        // for every typed identity check below and is never exposed.
        let added_tokens = tokenizer.get_added_tokens_decoder();
        let special_roles =
            validate_special_roles(&tokenizer, |id| added_tokens.get(&id), &checked.special_ids)?;
        let config_eos = validate_eos_identity(
            EosIdentityV1::Config,
            &contract.stop_identity,
            &tokenizer,
            |id| added_tokens.get(&id),
            checked.config_eos,
        )?;
        let tokenizer_eos = validate_eos_identity(
            EosIdentityV1::Tokenizer,
            &contract.stop_identity,
            &tokenizer,
            |id| added_tokens.get(&id),
            checked.tokenizer_eos,
        )?;

        if checked.contract_eos != checked.config_eos {
            return Err(TokenizerError::EosContractMismatch {
                contract_id: checked.contract_eos,
                config_id: checked.config_eos,
            });
        }

        let mut expected_stop_ids = vec![checked.tokenizer_eos, checked.config_eos];
        let identities_equal = contract.stop_identity.config_eos.token
            == contract.stop_identity.tokenizer_eos.token
            && checked.config_eos == checked.tokenizer_eos;
        if identities_equal {
            expected_stop_ids.dedup();
        }
        let actual_stop_ids = contract.generation_stop_policy().stop_token_ids.clone();
        if actual_stop_ids != expected_stop_ids {
            return Err(TokenizerError::StopPolicyMismatch {
                expected: expected_stop_ids,
                actual: actual_stop_ids,
            });
        }

        let token_byte_table = build_token_byte_table(&bytes, &tokenizer, contract.vocab_size)?;

        let snapshot = TokenizerSnapshotV1 {
            // This is retained as a consistency label only. It does not make
            // the mutable core label a cryptographic lock binding.
            fingerprint: fingerprint.to_owned(),
            vocab_size: contract.vocab_size,
            special_roles,
            config_eos,
            tokenizer_eos,
            stop_token_ids: actual_stop_ids,
        };

        Ok(Self {
            tokenizer,
            snapshot,
            token_byte_table,
            encode_add_special_tokens: false,
        })
    }

    /// Construct a Gemma 4 frontend from the regular verified cache.
    pub fn from_gemma4_verified_cache(
        lock: &Gemma4ModelLock,
        cache: &VerifiedCache,
    ) -> Result<Self, TokenizerError> {
        if cache.lock_fingerprint != lock.fingerprint() {
            return Err(TokenizerError::LockFingerprintMismatch {
                lock: lock.fingerprint().to_owned(),
                cache: cache.lock_fingerprint.clone(),
            });
        }
        let bytes = cache
            .read_frontend_asset(sllm_core::FrontendAssetKind::TokenizerJson)
            .map_err(|_| TokenizerError::FrontendAssetRead)?;
        Self::from_gemma4_contract_bytes(&lock.model.tokenizer_contract, bytes, lock.fingerprint())
    }

    /// Construct the identical Gemma tokenizer contract from a first-class
    /// provider artifact. The importer revalidates the asset hash on read.
    pub fn from_gemma4_quantized_model(
        lock: &Gemma4ModelLock,
        artifact: &sllm_core::VerifiedUnslothGemma4Nvfp4,
    ) -> Result<Self, TokenizerError> {
        let bytes = artifact
            .read_frontend_asset(sllm_core::FrontendAssetKind::TokenizerJson)
            .map_err(|_| TokenizerError::FrontendAssetRead)?;
        Self::from_gemma4_contract_bytes(&lock.model.tokenizer_contract, bytes, lock.fingerprint())
    }

    pub fn from_gemma4_gguf(
        lock: &Gemma4ModelLock,
        gguf: &VerifiedGguf,
    ) -> Result<Self, TokenizerError> {
        let extension = gguf.extension().ok_or(TokenizerError::FrontendAssetRead)?;
        if gguf.architecture() != "gemma4"
            || !extension
                .recipe
                .source_lock_fingerprints
                .iter()
                .any(|fingerprint| fingerprint == lock.fingerprint())
        {
            return Err(TokenizerError::LockFingerprintMismatch {
                lock: lock.fingerprint().to_owned(),
                cache: extension.recipe.semantic_model_id.clone(),
            });
        }
        let bytes = gguf
            .frontend_asset("tokenizer.json")
            .ok_or(TokenizerError::FrontendAssetRead)?
            .to_vec();
        Self::from_gemma4_contract_bytes(&lock.model.tokenizer_contract, bytes, lock.fingerprint())
    }

    /// Constructs the reviewed Gemma 4 MoE tokenizer directly from the exact
    /// source artifact, without borrowing the unrelated Dense 12B lock.
    pub fn from_gemma4_moe_artifact(
        artifact: &sllm_core::VerifiedGemma4Moe,
    ) -> Result<Self, TokenizerError> {
        let bytes = artifact
            .read_support_file("tokenizer.json")
            .map_err(|_| TokenizerError::FrontendAssetRead)?;
        Self::from_gemma4_contract_bytes(
            &gemma4_moe_tokenizer_contract(),
            bytes,
            GEMMA4_MOE_MODEL_FINGERPRINT,
        )
    }

    /// Constructs the reviewed Gemma 4 MoE tokenizer from its canonical
    /// derived GGUF while retaining the container-neutral semantic identity.
    pub fn from_gemma4_moe_gguf(gguf: &VerifiedGguf) -> Result<Self, TokenizerError> {
        if !has_reviewed_gemma4_moe_gguf_identity(gguf) {
            return Err(TokenizerError::LockFingerprintMismatch {
                lock: gemma4_moe_semantic_model_id(),
                cache: gguf
                    .extension()
                    .map(|extension| extension.recipe.semantic_model_id.clone())
                    .unwrap_or_else(|| gguf.architecture().to_owned()),
            });
        }
        let bytes = gguf
            .frontend_asset("tokenizer.json")
            .ok_or(TokenizerError::FrontendAssetRead)?
            .to_vec();
        let semantic_model_id = gemma4_moe_semantic_model_id();
        Self::from_gemma4_contract_bytes(
            &gemma4_moe_tokenizer_contract(),
            bytes,
            &semantic_model_id,
        )
    }

    fn from_gemma4_contract_bytes(
        contract: &Gemma4TokenizerContract,
        bytes: Vec<u8>,
        fingerprint: &str,
    ) -> Result<Self, TokenizerError> {
        let tokenizer =
            Tokenizer::from_bytes(bytes.clone()).map_err(|_| TokenizerError::InvalidTokenizer)?;
        let tokenizer_vocab = tokenizer.get_vocab(true);
        let tokenizer_vocab_size = tokenizer.get_vocab_size(true);
        let expected_vocab_size =
            usize::try_from(contract.vocab_size).map_err(|_| TokenizerError::InvalidTokenizer)?;
        let tokenizer_vocab_span = tokenizer_vocab
            .values()
            .copied()
            .max()
            .map_or(0_u64, |id| u64::from(id) + 1);
        if tokenizer_vocab.len() != tokenizer_vocab_size
            || tokenizer_vocab_size != expected_vocab_size
            || tokenizer_vocab_span != contract.vocab_size
        {
            return Err(TokenizerError::VocabSizeMismatch {
                lock: contract.vocab_size,
                tokenizer: tokenizer_vocab_span,
            });
        }

        let special_ids = contract
            .special_token_ids
            .iter()
            .map(|(role, id)| {
                checked_token_id(*id, TokenIdContextV1::SpecialRole).map(|id| (role.clone(), id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let added_tokens = tokenizer.get_added_tokens_decoder();
        let special_roles =
            validate_special_roles(&tokenizer, |id| added_tokens.get(&id), &special_ids)?;
        let expected_contents = gemma4_special_token_contents();
        for token in &special_roles {
            if expected_contents.get(token.role.as_str()).copied() != Some(token.content.as_str()) {
                return Err(TokenizerError::SpecialTokenContentMismatch {
                    role: token.role.clone(),
                    id: token.token_id,
                });
            }
        }
        if special_roles.len() != contract.special_token_ids.len() {
            return Err(TokenizerError::InvalidTokenizer);
        }
        let eos_id = checked_token_id(
            *contract
                .special_token_ids
                .get("eos")
                .ok_or(TokenizerError::InvalidTokenizer)?,
            TokenIdContextV1::TokenizerEos,
        )?;
        let stop_token_ids = contract
            .stop_token_ids
            .iter()
            .map(|id| checked_token_id(*id, TokenIdContextV1::ContractEos))
            .collect::<Result<Vec<_>, _>>()?;
        let expected_stop_ids = contract
            .stop_token_ids
            .iter()
            .map(|id| checked_token_id(*id, TokenIdContextV1::ContractEos))
            .collect::<Result<Vec<_>, _>>()?;
        if stop_token_ids != expected_stop_ids || stop_token_ids.first() != Some(&eos_id) {
            return Err(TokenizerError::StopPolicyMismatch {
                expected: expected_stop_ids,
                actual: stop_token_ids,
            });
        }
        let token_byte_table = build_token_byte_table(&bytes, &tokenizer, contract.vocab_size)?;
        let eos = EosIdentitySnapshotV1 {
            token: "<eos>".to_owned(),
            token_id: eos_id,
            observed_content: tokenizer.id_to_token(eos_id).ok_or(
                TokenizerError::EosIdToTokenMismatch {
                    identity: EosIdentityV1::Tokenizer,
                    id: eos_id,
                },
            )?,
        };
        let snapshot = TokenizerSnapshotV1 {
            fingerprint: fingerprint.to_owned(),
            vocab_size: contract.vocab_size,
            special_roles,
            config_eos: eos.clone(),
            tokenizer_eos: eos,
            stop_token_ids,
        };
        Ok(Self {
            tokenizer,
            snapshot,
            token_byte_table,
            encode_add_special_tokens: true,
        })
    }

    pub fn snapshot(&self) -> &TokenizerSnapshotV1 {
        &self.snapshot
    }

    /// Returns the decoder-aware immutable raw piece table in token-ID order.
    pub fn token_byte_table(&self) -> &TokenByteTableV1 {
        &self.token_byte_table
    }

    pub fn encode(&self, text: &str) -> Result<TokenIdsV1, TokenizerError> {
        let encoding = self
            .tokenizer
            .encode(text, self.encode_add_special_tokens)
            .map_err(|_| TokenizerError::Encode)?;
        Ok(TokenIdsV1::from_slice(encoding.get_ids()))
    }

    /// Tokenizes an auxiliary generation control string without injecting
    /// model BOS/EOS tokens.  This is used for stop and DRY sequence-breaker
    /// configuration, never for the user prompt path.
    pub fn encode_without_special_tokens(&self, text: &str) -> Result<TokenIdsV1, TokenizerError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|_| TokenizerError::Encode)?;
        Ok(TokenIdsV1::from_slice(encoding.get_ids()))
    }

    pub fn decode(
        &self,
        token_ids: &TokenIdsV1,
        mode: DecodeModeV1,
    ) -> Result<String, TokenizerError> {
        // Do this complete pass before calling the third-party decoder.  It
        // prevents its unknown-ID filtering behavior from becoming frontend
        // semantics.
        for id in token_ids.as_slice() {
            if self.tokenizer.id_to_token(*id).is_none() {
                return Err(TokenizerError::UnknownTokenId { id: *id });
            }
        }
        self.tokenizer
            .decode(token_ids.as_slice(), mode.skip_special_tokens())
            .map_err(|_| TokenizerError::Decode)
    }
}

#[derive(Clone, Debug, Default)]
struct TokenDecoderFeatures {
    byte_level: bool,
    byte_fallback: bool,
    metaspace_marker: Option<char>,
    unsupported_decoder: Option<String>,
    byte_level_inverse: Option<HashMap<char, u8>>,
}

fn build_token_byte_table(
    bytes: &[u8],
    tokenizer: &Tokenizer,
    model_vocab_size: u64,
) -> Result<TokenByteTableV1, TokenizerError> {
    let root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| TokenizerError::InvalidTokenizer)?;
    let features = decoder_features(&root)?;
    if let Some(decoder) = features.unsupported_decoder {
        return Err(TokenizerError::TokenByteDecoderUnsupported { decoder });
    }

    let capacity = usize::try_from(model_vocab_size).map_err(|_| {
        TokenizerError::TokenByteTableCapacityOverflow {
            value: model_vocab_size,
        }
    })?;
    let vocabulary = tokenizer.get_vocab(true);
    if vocabulary.len() != tokenizer.get_vocab_size(true) {
        return Err(TokenizerError::InvalidTokenizer);
    }
    let max_id = vocabulary.values().copied().max();
    if max_id.is_some_and(|id| u64::from(id) >= model_vocab_size) {
        return Err(TokenizerError::TokenByteTableVocabMismatch {
            id: max_id.expect("max_id is present"),
            vocab_size: model_vocab_size,
        });
    }
    let mut entries = vec![TokenByteEntryV1::reserved(); capacity];
    let added_tokens = tokenizer.get_added_tokens_decoder();
    for (piece, id) in vocabulary {
        let index =
            usize::try_from(id).map_err(|_| TokenizerError::TokenByteTableVocabMismatch {
                id,
                vocab_size: model_vocab_size,
            })?;
        let observed = tokenizer
            .id_to_token(id)
            .ok_or(TokenizerError::UnknownTokenId { id })?;
        if observed != piece {
            return Err(TokenizerError::InvalidTokenizer);
        }
        let is_special = added_tokens.get(&id).is_some_and(|token| token.special);
        let entry = if is_special {
            TokenByteEntryV1 {
                class: TokenPieceClassV1::Special,
                piece: Some(piece.into_boxed_str()),
                bytes: None,
            }
        } else {
            let (class, raw_bytes) = encode_piece_bytes(&piece, id, &features)?;
            TokenByteEntryV1 {
                class,
                piece: Some(piece.into_boxed_str()),
                bytes: Some(raw_bytes.into_boxed_slice()),
            }
        };
        entries[index] = entry;
    }
    Ok(TokenByteTableV1 {
        entries: entries.into_boxed_slice(),
    })
}

fn decoder_features(root: &serde_json::Value) -> Result<TokenDecoderFeatures, TokenizerError> {
    let mut features = TokenDecoderFeatures::default();
    if root
        .get("model")
        .and_then(|model| model.get("byte_fallback"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        features.byte_fallback = true;
    }
    if let Some(pre_tokenizer) = root.get("pre_tokenizer") {
        visit_decoder_value(pre_tokenizer, &mut features, false)?;
    }
    if let Some(decoder) = root.get("decoder") {
        visit_decoder_value(decoder, &mut features, true)?;
    }
    if features.byte_level {
        features.byte_level_inverse = Some(byte_level_inverse());
    }
    Ok(features)
}

fn visit_decoder_value(
    value: &serde_json::Value,
    features: &mut TokenDecoderFeatures,
    decoder_position: bool,
) -> Result<(), TokenizerError> {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                visit_decoder_value(value, features, decoder_position)?;
            }
        }
        serde_json::Value::Object(object) => {
            if let Some(kind) = object.get("type").and_then(serde_json::Value::as_str) {
                match kind {
                    "ByteLevel" => features.byte_level = true,
                    "ByteFallback" => features.byte_fallback = true,
                    "Metaspace" => {
                        if let Some(marker) = object
                            .get("replacement")
                            .and_then(serde_json::Value::as_str)
                            .and_then(|value| value.chars().next())
                        {
                            features.metaspace_marker = Some(marker);
                        }
                    }
                    "Sequence" | "Fuse" => {}
                    "Replace" => {}
                    "Strip" | "Precompiled" => {
                        features.unsupported_decoder = Some(kind.to_owned());
                    }
                    // A tokenizer can have an arbitrary pre-tokenizer plugin,
                    // but an unknown *decoder* would make byte derivation
                    // ambiguous.  Keep the contract fail-closed there.
                    other if decoder_position => {
                        features.unsupported_decoder = Some(other.to_owned());
                    }
                    _ => {}
                }
            }
            let mut recognized_replace = false;
            if let Some(pattern) = object.get("pattern") {
                if let Some(replacement) = object
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|value| value.chars().next())
                {
                    if pattern
                        .get("String")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|value| value == "▁")
                    {
                        features.metaspace_marker = Some('▁');
                        recognized_replace = replacement == ' ';
                        if replacement != ' ' {
                            features.unsupported_decoder = Some("Replace".to_owned());
                        }
                    }
                }
            }
            if object
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|kind| kind == "Replace")
                && !recognized_replace
            {
                features.unsupported_decoder = Some("Replace".to_owned());
            }
            for child in object.values() {
                if child.is_object() || child.is_array() {
                    visit_decoder_value(child, features, decoder_position)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn encode_piece_bytes(
    piece: &str,
    id: u32,
    features: &TokenDecoderFeatures,
) -> Result<(TokenPieceClassV1, Vec<u8>), TokenizerError> {
    if let Some(byte) = parse_byte_fallback_piece(piece) {
        if features.byte_fallback {
            return bounded_piece_bytes(id, TokenPieceClassV1::ByteFallback, vec![byte]);
        }
        return bounded_piece_bytes(id, TokenPieceClassV1::Ordinary, piece.as_bytes().to_vec());
    }
    if piece.starts_with("<0x") && piece.ends_with('>') {
        return Err(TokenizerError::TokenByteUnsupported { id });
    }

    let raw = if features.byte_level {
        let inverse = features
            .byte_level_inverse
            .as_ref()
            .expect("byte-level feature carries its inverse map");
        let mut raw = Vec::with_capacity(piece.len());
        for character in piece.chars() {
            let byte = inverse
                .get(&character)
                .copied()
                .ok_or(TokenizerError::TokenByteUnsupported { id })?;
            raw.push(byte);
        }
        raw
    } else if let Some(marker) = features.metaspace_marker {
        let mut raw = Vec::with_capacity(piece.len());
        for character in piece.chars() {
            if character == marker {
                raw.push(b' ');
            } else {
                let mut encoded = [0_u8; 4];
                raw.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
            }
        }
        raw
    } else {
        piece.as_bytes().to_vec()
    };
    bounded_piece_bytes(id, TokenPieceClassV1::Ordinary, raw)
}

fn bounded_piece_bytes(
    id: u32,
    class: TokenPieceClassV1,
    bytes: Vec<u8>,
) -> Result<(TokenPieceClassV1, Vec<u8>), TokenizerError> {
    if bytes.is_empty() || bytes.len() > MAX_TOKEN_PIECE_BYTES_V1 {
        return if bytes.len() > MAX_TOKEN_PIECE_BYTES_V1 {
            Err(TokenizerError::TokenBytePieceTooLong {
                id,
                len: bytes.len(),
            })
        } else {
            Err(TokenizerError::TokenByteUnsupported { id })
        };
    }
    Ok((class, bytes))
}

fn parse_byte_fallback_piece(piece: &str) -> Option<u8> {
    if piece.len() != 6 || !piece.starts_with("<0x") || !piece.ends_with('>') {
        return None;
    }
    let digits = piece.as_bytes();
    let high = hex_digit(digits[3])?;
    let low = hex_digit(digits[4])?;
    Some((high << 4) | low)
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn byte_level_inverse() -> HashMap<char, u8> {
    let mut map = HashMap::with_capacity(256);
    let mut used = HashSet::with_capacity(256);
    for byte in 33_u8..=126 {
        map.insert(char::from(byte), byte);
        used.insert(byte);
    }
    for byte in 161_u8..=172 {
        map.insert(char::from(byte), byte);
        used.insert(byte);
    }
    for byte in 174_u8..=255 {
        map.insert(char::from(byte), byte);
        used.insert(byte);
    }
    let mut next_codepoint = 256_u32;
    for byte in 0_u8..=255 {
        if used.contains(&byte) {
            continue;
        }
        let character =
            char::from_u32(next_codepoint).expect("GPT-2 byte-level Unicode mapping is valid");
        map.insert(character, byte);
        next_codepoint += 1;
    }
    map
}

fn gemma4_special_token_contents() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("pad", "<pad>"),
        ("eos", "<eos>"),
        ("bos", "<bos>"),
        ("unk", "<unk>"),
        ("mask", "<mask>"),
        ("think", "<|think|>"),
        ("tool_call_begin", "<|tool_call>"),
        ("tool_call_end", "<tool_call|>"),
        ("tool_response_begin", "<|tool_response>"),
        ("tool_response_end", "<tool_response|>"),
        ("channel_begin", "<|channel>"),
        ("channel_end", "<channel|>"),
        ("turn_begin", "<|turn>"),
        ("turn_end", "<turn|>"),
        ("image_begin", "<|image>"),
        ("audio_begin", "<|audio>"),
        ("image", "<|image|>"),
        ("audio", "<|audio|>"),
        ("image_end", "<image|>"),
        ("audio_end", "<audio|>"),
        ("video", "<|video|>"),
    ])
}

#[derive(Debug)]
struct CheckedContract {
    contract_eos: u32,
    config_eos: u32,
    tokenizer_eos: u32,
    special_ids: Vec<(String, u32)>,
}

impl CheckedContract {
    fn new(contract: &TokenizerContract) -> Result<Self, TokenizerError> {
        let mut special_ids = Vec::with_capacity(contract.special_token_ids.len());
        for (role, id) in &contract.special_token_ids {
            special_ids.push((
                role.clone(),
                checked_token_id(*id, TokenIdContextV1::SpecialRole)?,
            ));
        }
        Ok(Self {
            contract_eos: checked_token_id(contract.eos_token_id, TokenIdContextV1::ContractEos)?,
            config_eos: checked_token_id(
                contract.stop_identity.config_eos.token_id,
                TokenIdContextV1::ConfigEos,
            )?,
            tokenizer_eos: checked_token_id(
                contract.stop_identity.tokenizer_eos.token_id,
                TokenIdContextV1::TokenizerEos,
            )?,
            special_ids,
        })
    }
}

fn checked_token_id(value: u64, context: TokenIdContextV1) -> Result<u32, TokenizerError> {
    u32::try_from(value).map_err(|_| TokenizerError::TokenIdOverflow { context, value })
}

fn validate_special_roles<'a, F>(
    tokenizer: &Tokenizer,
    added_token: F,
    special_ids: &[(String, u32)],
) -> Result<Vec<SpecialTokenSnapshotV1>, TokenizerError>
where
    F: Fn(u32) -> Option<&'a AddedToken>,
{
    let mut ids = HashMap::<u32, &str>::with_capacity(special_ids.len());
    let mut observed_contents = Vec::with_capacity(special_ids.len());
    for (role, id) in special_ids {
        if let Some(first_role) = ids.insert(*id, role.as_str()) {
            return Err(TokenizerError::DuplicateSpecialId {
                first_role: first_role.to_owned(),
                second_role: role.clone(),
                id: *id,
            });
        }
        let content =
            tokenizer
                .id_to_token(*id)
                .ok_or_else(|| TokenizerError::SpecialTokenIdMissing {
                    role: role.clone(),
                    id: *id,
                })?;
        let added = added_token(*id).ok_or_else(|| TokenizerError::SpecialTokenDecoderMissing {
            role: role.clone(),
            id: *id,
        })?;
        if !added.special {
            return Err(TokenizerError::SpecialTokenNotMarkedSpecial {
                role: role.clone(),
                id: *id,
            });
        }
        observed_contents.push(SpecialTokenSnapshotV1 {
            role: role.clone(),
            token_id: *id,
            content: content.to_owned(),
        });
    }
    reject_duplicate_special_contents(&observed_contents)?;
    Ok(observed_contents)
}

fn reject_duplicate_special_contents(
    observed_contents: &[SpecialTokenSnapshotV1],
) -> Result<(), TokenizerError> {
    let mut contents = HashMap::<&str, &str>::with_capacity(observed_contents.len());
    for token in observed_contents {
        if let Some(first_role) = contents.insert(token.content.as_str(), token.role.as_str()) {
            return Err(TokenizerError::DuplicateSpecialContent {
                first_role: first_role.to_owned(),
                second_role: token.role.clone(),
            });
        }
    }
    Ok(())
}

fn validate_eos_identity<'a, F>(
    identity: EosIdentityV1,
    stop_identity: &StopIdentity,
    tokenizer: &Tokenizer,
    added_token: F,
    id: u32,
) -> Result<EosIdentitySnapshotV1, TokenizerError>
where
    F: Fn(u32) -> Option<&'a AddedToken>,
{
    let content = match identity {
        EosIdentityV1::Config => stop_identity.config_eos.token.as_str(),
        EosIdentityV1::Tokenizer => stop_identity.tokenizer_eos.token.as_str(),
    };
    if tokenizer.token_to_id(content) != Some(id) {
        return Err(TokenizerError::EosTokenToIdMismatch { identity, id });
    }
    let observed_content = tokenizer
        .id_to_token(id)
        .ok_or(TokenizerError::EosIdToTokenMismatch { identity, id })?;
    if observed_content != content {
        return Err(TokenizerError::EosIdToTokenMismatch { identity, id });
    }
    let added = added_token(id).ok_or(TokenizerError::EosAddedTokenMissing { identity, id })?;
    if added.content != content {
        return Err(TokenizerError::EosAddedTokenContentMismatch { identity, id });
    }
    if !added.special {
        return Err(TokenizerError::EosAddedTokenNotMarkedSpecial { identity, id });
    }
    Ok(EosIdentitySnapshotV1 {
        token: content.to_owned(),
        token_id: id,
        observed_content,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sllm_core::{ConfigEos, TokenizerEos};

    #[test]
    fn duplicate_special_content_is_rejected_for_distinct_roles() {
        let observed = vec![
            SpecialTokenSnapshotV1 {
                role: "first".to_owned(),
                token_id: 1,
                content: "same".to_owned(),
            },
            SpecialTokenSnapshotV1 {
                role: "second".to_owned(),
                token_id: 2,
                content: "same".to_owned(),
            },
        ];
        assert_eq!(
            reject_duplicate_special_contents(&observed),
            Err(TokenizerError::DuplicateSpecialContent {
                first_role: "first".to_owned(),
                second_role: "second".to_owned(),
            })
        );
    }

    #[test]
    fn loaded_tokenizer_with_missing_eos_added_entry_is_rejected() {
        // The integration fixture verifies the raw asset through
        // verify_model_cache. Here the already-loaded tokenizer's decoder map
        // is the post-load semantic input, with only the EOS entry omitted.
        let tokenizer = Tokenizer::from_bytes(include_bytes!(
            "../../../ci/fixtures/tokenizer-v1/tokenizer.json"
        ))
        .expect("fixture tokenizer loads");
        let mut added_tokens = tokenizer.get_added_tokens_decoder();
        assert!(added_tokens.remove(&9).is_some());
        let stop_identity = StopIdentity {
            config_eos: ConfigEos {
                token: "<|endoftext|>".to_owned(),
                token_id: 8,
                source_file: "config.json".to_owned(),
            },
            tokenizer_eos: TokenizerEos {
                token: "<|im_end|>".to_owned(),
                token_id: 9,
                source_files: vec!["tokenizer.json".to_owned()],
            },
        };
        assert_eq!(
            validate_eos_identity(
                EosIdentityV1::Tokenizer,
                &stop_identity,
                &tokenizer,
                |id| added_tokens.get(&id),
                9,
            ),
            Err(TokenizerError::EosAddedTokenMissing {
                identity: EosIdentityV1::Tokenizer,
                id: 9
            })
        );
    }

    #[test]
    fn byte_level_piece_uses_decoder_byte_mapping_without_decode_round_trip() {
        let bytes = br#"{
          "version": "1.0",
          "added_tokens": [],
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
          "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
          "model": {
            "type": "WordLevel",
            "vocab": {"\u0120": 0, "\u00c3\u00a9": 1},
            "unk_token": "\u0120"
          }
        }"#;
        let table =
            TokenByteTableV1::from_tokenizer_json(bytes, 2).expect("byte-level table constructs");
        assert_eq!(
            table.entry(0).and_then(TokenByteEntryV1::bytes),
            Some(&b" "[..])
        );
        assert_eq!(
            table.entry(1).and_then(TokenByteEntryV1::bytes),
            Some(&[0xc3, 0xa9][..])
        );
    }

    #[test]
    fn byte_fallback_piece_is_one_raw_byte() {
        let bytes = br#"{
          "version": "1.0",
          "added_tokens": [],
          "decoder": {"type": "ByteFallback"},
          "model": {
            "type": "WordLevel",
            "byte_fallback": true,
            "vocab": {"<0x00>": 0, "<0xFF>": 1},
            "unk_token": "<0x00>"
          }
        }"#;
        let table = TokenByteTableV1::from_tokenizer_json(bytes, 2)
            .expect("byte-fallback table constructs");
        assert_eq!(
            table.entry(0).and_then(TokenByteEntryV1::bytes),
            Some(&[0][..])
        );
        assert_eq!(
            table.entry(0).map(TokenByteEntryV1::class),
            Some(TokenPieceClassV1::ByteFallback)
        );
        assert_eq!(
            table.entry(1).and_then(TokenByteEntryV1::bytes),
            Some(&[0xff][..])
        );
    }

    #[test]
    fn malformed_byte_fallback_piece_fails_closed() {
        let bytes = br#"{
          "version": "1.0",
          "added_tokens": [],
          "decoder": {"type": "ByteFallback"},
          "model": {
            "type": "WordLevel",
            "byte_fallback": true,
            "vocab": {"<0xGG>": 0},
            "unk_token": "<0xGG>"
          }
        }"#;
        assert_eq!(
            TokenByteTableV1::from_tokenizer_json(bytes, 1),
            Err(TokenizerError::TokenByteUnsupported { id: 0 })
        );
    }

    #[test]
    fn gemma4_moe_contract_keeps_exact_special_and_stop_identity() {
        let contract = gemma4_moe_tokenizer_contract();
        assert_eq!(contract.vocab_size, 262_144);
        assert_eq!(contract.special_token_ids.get("bos"), Some(&2));
        assert_eq!(contract.special_token_ids.get("eos"), Some(&1));
        assert_eq!(contract.special_token_ids.get("turn_begin"), Some(&105));
        assert_eq!(contract.special_token_ids.get("turn_end"), Some(&106));
        assert_eq!(
            contract.special_token_ids.get("tool_response_begin"),
            Some(&50)
        );
        assert_eq!(contract.stop_token_ids, [1, 106, 50]);
        assert_eq!(
            sllm_core::gemma4_moe_generation_stop_policy().stop_token_ids,
            [1, 106, 50]
        );
    }

    #[test]
    fn gemma4_moe_gguf_identity_is_exact_and_rejects_adjacent_values() {
        let semantic = gemma4_moe_semantic_model_id();
        let sources = vec![GEMMA4_MOE_MODEL_FINGERPRINT.to_owned()];
        assert!(has_reviewed_gemma4_moe_identity_parts(
            "gemma4moe",
            &semantic,
            &sources,
        ));
        assert!(!has_reviewed_gemma4_moe_identity_parts(
            "gemma4", &semantic, &sources,
        ));
        assert!(!has_reviewed_gemma4_moe_identity_parts(
            "gemma4moe",
            GEMMA4_MOE_MODEL_FINGERPRINT,
            &sources,
        ));
        let extra = vec![
            GEMMA4_MOE_MODEL_FINGERPRINT.to_owned(),
            "sha256:0000000000000000000000000000000000000000000000000000000000000000".to_owned(),
        ];
        assert!(!has_reviewed_gemma4_moe_identity_parts(
            "gemma4moe",
            &semantic,
            &extra,
        ));
    }
}
