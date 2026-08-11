use core::fmt;
use std::collections::HashMap;

use sllm_core::{ModelLock, StopIdentity, TokenizerContract, VerifiedCache};
use tokenizers::{AddedToken, Tokenizer};

use crate::{StopPolicyError, validate_generation_stop_policy};

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
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<StopPolicyError> for TokenizerError {
    fn from(_: StopPolicyError) -> Self {
        Self::InvalidGenerationStopPolicy
    }
}

/// An immutable tokenizer loaded only from a core-verified model-cache asset.
/// The retained fingerprint is a consistency label, not a cryptographic lock
/// binding; core's mutable public fingerprint fields remain separate debt.
#[derive(Debug)]
pub struct TokenizerFrontendV1 {
    tokenizer: Tokenizer,
    snapshot: TokenizerSnapshotV1,
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
        let tokenizer =
            Tokenizer::from_bytes(bytes).map_err(|_| TokenizerError::InvalidTokenizer)?;

        let contract = &lock.model().tokenizer_contract;
        validate_generation_stop_policy(lock.generation_stop_policy())?;
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
        let actual_stop_ids = lock.generation_stop_policy().stop_token_ids.clone();
        if actual_stop_ids != expected_stop_ids {
            return Err(TokenizerError::StopPolicyMismatch {
                expected: expected_stop_ids,
                actual: actual_stop_ids,
            });
        }

        let snapshot = TokenizerSnapshotV1 {
            // This is retained as a consistency label only. It does not make
            // the mutable core label a cryptographic lock binding.
            fingerprint: lock.fingerprint().to_owned(),
            vocab_size: contract.vocab_size,
            special_roles,
            config_eos,
            tokenizer_eos,
            stop_token_ids: actual_stop_ids,
        };

        Ok(Self {
            tokenizer,
            snapshot,
        })
    }

    pub fn snapshot(&self) -> &TokenizerSnapshotV1 {
        &self.snapshot
    }

    pub fn encode(&self, text: &str) -> Result<TokenIdsV1, TokenizerError> {
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
}
