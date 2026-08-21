//! Strict, bounded session checkpoint envelopes.
//!
//! The checkpoint format is deliberately backend-independent.  It contains
//! model/frontend identity, token history, and opaque state planes, but never
//! native device handles, pointers, or page tables.  The bytes are intended
//! to be treated as sensitive application data by callers.

use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::KvCacheEncoding;

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};

/// The fixed magic at the beginning of every checkpoint.
pub const CHECKPOINT_MAGIC: [u8; 8] = *b"SLLMCKP1";
/// The only envelope schema currently accepted by this module.
pub const CHECKPOINT_SCHEMA_VERSION: u16 = 1;
/// Stable descriptive identifier for the v1 little-endian envelope.
pub const CHECKPOINT_SCHEMA_ID: &str = "sllm-session-checkpoint-v1";
/// Maximum encoded header size, including identity and the section table.
pub const MAX_CHECKPOINT_HEADER_BYTES: usize = 4096;
/// Maximum number of logical sections/opaque planes represented by v1.
pub const MAX_CHECKPOINT_SECTIONS: usize = 4096;
/// Maximum bytes accepted for a decoded checkpoint in host memory.
pub const MAX_CHECKPOINT_BYTES: u64 = 64 * 1024 * 1024 * 1024;
/// Maximum bytes accepted for a single opaque state section.
pub const MAX_SECTION_BYTES: u64 = MAX_CHECKPOINT_BYTES;
/// Maximum number of bytes in an identity string.
pub const MAX_IDENTITY_FIELD_BYTES: usize = 1024;
/// Maximum number of token IDs in a checkpoint.
pub const MAX_TOKEN_HISTORY: usize = 1_048_576;
/// Maximum number of native state layers represented by one checkpoint.
pub const MAX_STATE_LAYERS: usize = MAX_CHECKPOINT_SECTIONS;
/// Maximum number of opaque native state planes.
pub const MAX_STATE_PLANES: usize = MAX_CHECKPOINT_SECTIONS;
/// Compatibility alias for the original Phase 41 draft name.
pub const MAX_KV_PLANES: usize = MAX_STATE_PLANES;
/// Maximum bytes in the serialized conversation transcript.
pub const MAX_CONVERSATION_BYTES: usize = 16 * 1024 * 1024;

const FIXED_HEADER_BYTES: usize = 96;
const SECTION_ENTRY_BYTES: usize = 56;
const SECTION_COUNT: usize = 7;
const CHECKSUM_OFFSET: usize = 28;
const CHECKSUM_END: usize = CHECKSUM_OFFSET + 32;
const MAX_TEMP_NAME_BYTES: usize = 255;

const SECTION_TOKEN_HISTORY: u16 = 1;
const SECTION_CONVERSATION: u16 = 2;
const SECTION_STATE_LAYERS: u16 = 3;
const SECTION_STATE_PLANES: u16 = 4;
const SECTION_SAMPLER_STATE: u16 = 5;
const SECTION_GRAMMAR_STATE: u16 = 6;
const SECTION_STOP_STATE: u16 = 7;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Errors returned while validating, encoding, or atomically storing a
/// checkpoint.  Error messages intentionally do not include payload bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointError {
    Invalid(String),
    Bounds(String),
    Corrupt(String),
    Truncated,
    TrailingBytes,
    UnsupportedVersion(u16),
    IdentityMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    PathViolation(String),
    Security(String),
    QuotaExceeded {
        requested: u64,
        available: u64,
    },
    Io {
        message: String,
    },
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid checkpoint: {message}"),
            Self::Bounds(message) => write!(formatter, "checkpoint bounds error: {message}"),
            Self::Corrupt(message) => write!(formatter, "corrupt checkpoint: {message}"),
            Self::Truncated => write!(formatter, "truncated checkpoint"),
            Self::TrailingBytes => write!(formatter, "checkpoint has trailing bytes"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported checkpoint schema version {version}")
            }
            Self::IdentityMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "checkpoint identity mismatch for {field}: expected {expected}, got {actual}"
            ),
            Self::PathViolation(message) => {
                write!(formatter, "checkpoint path violation: {message}")
            }
            Self::Security(message) => {
                write!(formatter, "checkpoint security violation: {message}")
            }
            Self::QuotaExceeded {
                requested,
                available,
            } => write!(
                formatter,
                "checkpoint quota exceeded: requested {requested} bytes, available {available}"
            ),
            Self::Io { message } => write!(formatter, "checkpoint I/O error: {message}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

fn io_error(_path: &Path, error: impl fmt::Display) -> CheckpointError {
    CheckpointError::Io {
        message: error.to_string(),
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest.finalize().into()
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Compute a canonical digest for a token sequence.  The length prefix keeps
/// `[1, 2]` distinct from malformed concatenation schemes.
pub fn token_sequence_digest(tokens: &[u32]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(4 + tokens.len().saturating_mul(4));
    bytes.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    sha256(&bytes)
}

/// Identity that must match before opaque state is restored or reused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointIdentity {
    pub model_lock_fingerprint: String,
    pub derived_artifact_identity: String,
    pub adapter_identity: String,
    pub renderer_identity: String,
    pub tokenizer_identity: String,
    pub target_semantics: String,
    pub plan_digest: String,
    pub token_sequence_digest: [u8; 32],
    pub kv_encoding: KvCacheEncoding,
    pub kv_descriptor_digest: [u8; 32],
    pub context_policy_digest: [u8; 32],
}

impl CheckpointIdentity {
    /// Construct identity from the exact token sequence used by the prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn for_tokens(
        model_lock_fingerprint: impl Into<String>,
        derived_artifact_identity: impl Into<String>,
        adapter_identity: impl Into<String>,
        renderer_identity: impl Into<String>,
        tokenizer_identity: impl Into<String>,
        target_semantics: impl Into<String>,
        plan_digest: impl Into<String>,
        tokens: &[u32],
        kv_encoding: KvCacheEncoding,
        kv_descriptor_digest: [u8; 32],
        context_policy_digest: [u8; 32],
    ) -> Result<Self, CheckpointError> {
        let identity = Self {
            model_lock_fingerprint: model_lock_fingerprint.into(),
            derived_artifact_identity: derived_artifact_identity.into(),
            adapter_identity: adapter_identity.into(),
            renderer_identity: renderer_identity.into(),
            tokenizer_identity: tokenizer_identity.into(),
            target_semantics: target_semantics.into(),
            plan_digest: plan_digest.into(),
            token_sequence_digest: token_sequence_digest(tokens),
            kv_encoding,
            kv_descriptor_digest,
            context_policy_digest,
        };
        identity.validate()?;
        Ok(identity)
    }

    /// Revalidate every bounded identity field.
    pub fn validate(&self) -> Result<(), CheckpointError> {
        validate_lock_fingerprint(&self.model_lock_fingerprint)?;
        for (name, value) in [
            ("derived_artifact_identity", &self.derived_artifact_identity),
            ("adapter_identity", &self.adapter_identity),
            ("renderer_identity", &self.renderer_identity),
            ("tokenizer_identity", &self.tokenizer_identity),
            ("target_semantics", &self.target_semantics),
            ("plan_digest", &self.plan_digest),
        ] {
            if value.is_empty() || value.len() > MAX_IDENTITY_FIELD_BYTES {
                return Err(CheckpointError::Bounds(format!(
                    "{name} must be 1..={MAX_IDENTITY_FIELD_BYTES} bytes"
                )));
            }
            if value.as_bytes().contains(&0) {
                return Err(CheckpointError::Invalid(format!("{name} contains NUL")));
            }
        }
        Ok(())
    }
}

fn validate_lock_fingerprint(value: &str) -> Result<(), CheckpointError> {
    let valid = value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
    if !valid {
        return Err(CheckpointError::Invalid(
            "model_lock_fingerprint must be sha256:<64 lowercase hex characters>".into(),
        ));
    }
    Ok(())
}

/// Backend-neutral owner class for an opaque native state image.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum StateOwnerKindV1 {
    Kv = 1,
    LinearAttention = 2,
}

/// Semantic plane identity. Numeric tags intentionally match the additive
/// HIP state-image ABI within each owner class, while remaining independent
/// of native pointers, handles, and struct layout.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum StatePlaneKindV1 {
    KvKey = 1,
    KvValue = 2,
    KvKeyScale = 3,
    KvValueScale = 4,
    KvKeyOuterScale = 5,
    KvValueOuterScale = 6,
    LinearConvSlot0 = 17,
    LinearConvSlot1 = 18,
    LinearRecurrentSlot0 = 19,
    LinearRecurrentSlot1 = 20,
    LinearScratch = 21,
}

impl StatePlaneKindV1 {
    const fn owner(self) -> StateOwnerKindV1 {
        match self {
            Self::KvKey
            | Self::KvValue
            | Self::KvKeyScale
            | Self::KvValueScale
            | Self::KvKeyOuterScale
            | Self::KvValueOuterScale => StateOwnerKindV1::Kv,
            Self::LinearConvSlot0
            | Self::LinearConvSlot1
            | Self::LinearRecurrentSlot0
            | Self::LinearRecurrentSlot1
            | Self::LinearScratch => StateOwnerKindV1::LinearAttention,
        }
    }
}

/// Publication metadata restored only after every plane for the layer has
/// been imported successfully.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateLayerMetadataV1 {
    pub owner: StateOwnerKindV1,
    pub layer_id: u32,
    pub published_length: u64,
    pub generation: u64,
    /// Linear-attention/GDN uses double-buffered active state. KV must use
    /// `None`; linear-attention must use `Some(0 | 1)`.
    pub active_slot: Option<u8>,
}

/// One exact encoded backend state plane, identified by owner and layer.
/// Native device handles, pointers, page tables, and struct padding are never
/// persisted.
#[derive(Clone, Eq, PartialEq)]
pub struct OpaqueStatePlane {
    pub owner: StateOwnerKindV1,
    pub layer_id: u32,
    pub plane: StatePlaneKindV1,
    pub bytes: Vec<u8>,
}

impl fmt::Debug for OpaqueStatePlane {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpaqueStatePlane")
            .field("owner", &self.owner)
            .field("layer_id", &self.layer_id)
            .field("plane", &self.plane)
            .field("byte_len", &self.bytes.len())
            .finish()
    }
}

/// State and conversation payload carried by a checkpoint.
#[derive(Clone, Eq, PartialEq, Default)]
pub struct CheckpointPayload {
    pub token_history: Vec<u32>,
    pub conversation: Vec<u8>,
    pub state_layers: Vec<StateLayerMetadataV1>,
    pub state_planes: Vec<OpaqueStatePlane>,
    pub sampler_state: Vec<u8>,
    pub grammar_state: Vec<u8>,
    pub stop_state: Vec<u8>,
}

impl fmt::Debug for CheckpointPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckpointPayload")
            .field("token_count", &self.token_history.len())
            .field("conversation_bytes", &self.conversation.len())
            .field("state_layers", &self.state_layers)
            .field("state_planes", &self.state_planes)
            .field("sampler_state_bytes", &self.sampler_state.len())
            .field("grammar_state_bytes", &self.grammar_state.len())
            .field("stop_state_bytes", &self.stop_state.len())
            .finish()
    }
}

impl CheckpointPayload {
    fn validate(&self, kv_encoding: KvCacheEncoding) -> Result<(), CheckpointError> {
        if self.token_history.len() > MAX_TOKEN_HISTORY {
            return Err(CheckpointError::Bounds(
                "token history exceeds limit".into(),
            ));
        }
        if self.conversation.len() > MAX_CONVERSATION_BYTES {
            return Err(CheckpointError::Bounds(format!(
                "conversation exceeds {MAX_CONVERSATION_BYTES} bytes"
            )));
        }
        for (name, bytes) in [
            ("sampler_state", &self.sampler_state),
            ("grammar_state", &self.grammar_state),
            ("stop_state", &self.stop_state),
        ] {
            if bytes.len() as u64 > MAX_SECTION_BYTES {
                return Err(CheckpointError::Bounds(format!(
                    "{name} exceeds {MAX_SECTION_BYTES} bytes"
                )));
            }
        }
        if self.state_layers.len() > MAX_STATE_LAYERS {
            return Err(CheckpointError::Bounds("too many state layers".into()));
        }
        let mut layers = std::collections::BTreeSet::new();
        for layer in &self.state_layers {
            if !layers.insert((layer.owner, layer.layer_id)) {
                return Err(CheckpointError::Invalid(
                    "duplicate state-layer metadata".into(),
                ));
            }
            match (layer.owner, layer.active_slot) {
                (StateOwnerKindV1::Kv, None) => {}
                (StateOwnerKindV1::LinearAttention, Some(0 | 1)) => {}
                (StateOwnerKindV1::Kv, Some(_)) => {
                    return Err(CheckpointError::Invalid(
                        "KV state must not define an active slot".into(),
                    ));
                }
                (StateOwnerKindV1::LinearAttention, _) => {
                    return Err(CheckpointError::Invalid(
                        "linear-attention active slot must be 0 or 1".into(),
                    ));
                }
            }
        }
        if self.state_planes.len() > MAX_STATE_PLANES {
            return Err(CheckpointError::Bounds("too many state planes".into()));
        }
        let mut seen = std::collections::BTreeSet::new();
        for plane in &self.state_planes {
            if plane.plane.owner() != plane.owner {
                return Err(CheckpointError::Invalid(
                    "state plane kind does not match its owner".into(),
                ));
            }
            if !layers.contains(&(plane.owner, plane.layer_id)) {
                return Err(CheckpointError::Invalid(
                    "state plane has no layer metadata".into(),
                ));
            }
            if !seen.insert((plane.owner, plane.layer_id, plane.plane)) {
                return Err(CheckpointError::Invalid("duplicate state plane".into()));
            }
            if plane.bytes.len() as u64 > MAX_SECTION_BYTES {
                return Err(CheckpointError::Bounds(
                    "state plane exceeds section limit".into(),
                ));
            }
        }
        for layer in &self.state_layers {
            let actual = seen
                .iter()
                .filter_map(|(owner, layer_id, plane)| {
                    (*owner == layer.owner && *layer_id == layer.layer_id).then_some(*plane)
                })
                .collect::<std::collections::BTreeSet<_>>();
            let expected = required_state_planes(layer.owner, kv_encoding);
            if actual != expected {
                return Err(CheckpointError::Invalid(
                    "state layer has missing or unexpected encoding-native planes".into(),
                ));
            }
        }
        Ok(())
    }
}

fn required_state_planes(
    owner: StateOwnerKindV1,
    encoding: KvCacheEncoding,
) -> std::collections::BTreeSet<StatePlaneKindV1> {
    use StatePlaneKindV1::*;
    match owner {
        StateOwnerKindV1::Kv => match encoding {
            KvCacheEncoding::Fp16 | KvCacheEncoding::Fp8E4M3FnStatic => {
                [KvKey, KvValue].into_iter().collect()
            }
            KvCacheEncoding::Fp8E4M3Fn => [KvKey, KvValue, KvKeyScale, KvValueScale]
                .into_iter()
                .collect(),
            KvCacheEncoding::Nvfp4 => [
                KvKey,
                KvValue,
                KvKeyScale,
                KvValueScale,
                KvKeyOuterScale,
                KvValueOuterScale,
            ]
            .into_iter()
            .collect(),
        },
        StateOwnerKindV1::LinearAttention => [
            LinearConvSlot0,
            LinearConvSlot1,
            LinearRecurrentSlot0,
            LinearRecurrentSlot1,
            LinearScratch,
        ]
        .into_iter()
        .collect(),
    }
}

/// Metadata in the envelope header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionCheckpointHeader {
    pub identity: CheckpointIdentity,
    pub token_count: u64,
    pub absolute_position: u64,
    pub logical_position: u64,
    pub generation_state_version: u32,
}

/// Phase-wide name for the versioned backend-neutral state header.
pub type SessionStateHeaderV1 = SessionCheckpointHeader;

/// A complete, validated checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionCheckpoint {
    pub header: SessionCheckpointHeader,
    pub payload: CheckpointPayload,
}

impl SessionCheckpoint {
    pub fn new(
        identity: CheckpointIdentity,
        absolute_position: u64,
        logical_position: u64,
        generation_state_version: u32,
        payload: CheckpointPayload,
    ) -> Result<Self, CheckpointError> {
        identity.validate()?;
        payload.validate(identity.kv_encoding)?;
        let token_count = payload.token_history.len() as u64;
        if identity.token_sequence_digest != token_sequence_digest(&payload.token_history) {
            return Err(CheckpointError::Invalid(
                "identity token sequence digest does not match token history".into(),
            ));
        }
        let checkpoint = Self {
            header: SessionCheckpointHeader {
                identity,
                token_count,
                absolute_position,
                logical_position,
                generation_state_version,
            },
            payload,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Encode as a strict binary envelope with a section table and SHA-256
    /// checksums for both the envelope and each section.
    pub fn encode(&self) -> Result<Vec<u8>, CheckpointError> {
        self.validate()?;
        let sections = self.encode_sections()?;
        let identity = encode_identity(&self.header.identity)?;
        let table_bytes = SECTION_COUNT
            .checked_mul(SECTION_ENTRY_BYTES)
            .ok_or_else(|| CheckpointError::Bounds("section table overflow".into()))?;
        let header_len = FIXED_HEADER_BYTES
            .checked_add(identity.len())
            .and_then(|length| length.checked_add(table_bytes))
            .ok_or_else(|| CheckpointError::Bounds("header length overflow".into()))?;
        if header_len > MAX_CHECKPOINT_HEADER_BYTES {
            return Err(CheckpointError::Bounds(
                "checkpoint header exceeds maximum size".into(),
            ));
        }
        let payload_len = sections.iter().try_fold(0usize, |sum, section| {
            sum.checked_add(section.1.len())
                .ok_or_else(|| CheckpointError::Bounds("payload length overflow".into()))
        })?;
        let total_len = header_len
            .checked_add(payload_len)
            .ok_or_else(|| CheckpointError::Bounds("checkpoint length overflow".into()))?;
        if total_len as u64 > MAX_CHECKPOINT_BYTES {
            return Err(CheckpointError::Bounds(
                "checkpoint exceeds maximum size".into(),
            ));
        }
        let mut output = Vec::with_capacity(total_len);
        output.extend_from_slice(&CHECKPOINT_MAGIC);
        output.extend_from_slice(&CHECKPOINT_SCHEMA_VERSION.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(&(header_len as u32).to_le_bytes());
        output.extend_from_slice(&(SECTION_COUNT as u16).to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(&(total_len as u64).to_le_bytes());
        output.extend_from_slice(&[0u8; 32]);
        output.extend_from_slice(&(identity.len() as u32).to_le_bytes());
        output.extend_from_slice(&self.header.token_count.to_le_bytes());
        output.extend_from_slice(&self.header.absolute_position.to_le_bytes());
        output.extend_from_slice(&self.header.logical_position.to_le_bytes());
        output.extend_from_slice(&self.header.generation_state_version.to_le_bytes());
        output.extend_from_slice(&0u32.to_le_bytes());
        debug_assert_eq!(output.len(), FIXED_HEADER_BYTES);
        output.extend_from_slice(&identity);

        let mut offset = header_len as u64;
        for (kind, bytes) in &sections {
            output.extend_from_slice(&kind.to_le_bytes());
            output.extend_from_slice(&0u16.to_le_bytes());
            output.extend_from_slice(&0u32.to_le_bytes());
            output.extend_from_slice(&offset.to_le_bytes());
            output.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            output.extend_from_slice(&sha256(bytes));
            offset = offset
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| CheckpointError::Bounds("section offset overflow".into()))?;
        }
        debug_assert_eq!(output.len(), header_len);
        for (_, bytes) in sections {
            output.extend_from_slice(&bytes);
        }
        let checksum = sha256_with_zeroed_checksum(&output)?;
        output[CHECKSUM_OFFSET..CHECKSUM_END].copy_from_slice(&checksum);
        Ok(output)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, CheckpointError> {
        Self::decode_with_identity(bytes, None)
    }

    pub fn decode_with_identity(
        bytes: &[u8],
        expected_identity: Option<&CheckpointIdentity>,
    ) -> Result<Self, CheckpointError> {
        if bytes.len() < FIXED_HEADER_BYTES {
            return Err(CheckpointError::Truncated);
        }
        if bytes.len() as u64 > MAX_CHECKPOINT_BYTES {
            return Err(CheckpointError::Bounds(
                "checkpoint exceeds maximum size".into(),
            ));
        }
        if bytes[..8] != CHECKPOINT_MAGIC {
            return Err(CheckpointError::Corrupt("invalid magic".into()));
        }
        let schema = read_u16(bytes, 8)?;
        if schema != CHECKPOINT_SCHEMA_VERSION {
            return Err(CheckpointError::UnsupportedVersion(schema));
        }
        if read_u16(bytes, 10)? != 0 || read_u16(bytes, 18)? != 0 || read_u32(bytes, 92)? != 0 {
            return Err(CheckpointError::Corrupt(
                "reserved header field is nonzero".into(),
            ));
        }
        let header_len = read_u32(bytes, 12)? as usize;
        let section_count = read_u16(bytes, 16)? as usize;
        let total_len = read_u64(bytes, 20)?;
        if section_count != SECTION_COUNT {
            return Err(CheckpointError::Corrupt("unexpected section count".into()));
        }
        if total_len != bytes.len() as u64 {
            return if total_len > bytes.len() as u64 {
                Err(CheckpointError::Truncated)
            } else {
                Err(CheckpointError::TrailingBytes)
            };
        }
        let identity_len = read_u32(bytes, 60)? as usize;
        let minimum_header = FIXED_HEADER_BYTES
            .checked_add(identity_len)
            .and_then(|value| value.checked_add(section_count * SECTION_ENTRY_BYTES))
            .ok_or_else(|| CheckpointError::Bounds("header length overflow".into()))?;
        if identity_len == 0
            || identity_len > MAX_IDENTITY_FIELD_BYTES * 8
            || header_len != minimum_header
            || header_len > MAX_CHECKPOINT_HEADER_BYTES
        {
            return Err(CheckpointError::Corrupt("invalid header length".into()));
        }
        if header_len > bytes.len() {
            return Err(CheckpointError::Truncated);
        }
        let actual_checksum = &bytes[CHECKSUM_OFFSET..CHECKSUM_END];
        let expected_checksum = sha256_with_zeroed_checksum(bytes)?;
        if actual_checksum != expected_checksum {
            return Err(CheckpointError::Corrupt(
                "envelope checksum mismatch".into(),
            ));
        }
        let identity =
            decode_identity(&bytes[FIXED_HEADER_BYTES..FIXED_HEADER_BYTES + identity_len])?;
        if let Some(expected) = expected_identity {
            compare_identity(expected, &identity)?;
        }
        let token_count = read_u64(bytes, 64)?;
        let absolute_position = read_u64(bytes, 72)?;
        let logical_position = read_u64(bytes, 80)?;
        let generation_state_version = read_u32(bytes, 88)?;

        let table_start = FIXED_HEADER_BYTES + identity_len;
        let mut entries = Vec::with_capacity(section_count);
        let mut seen = std::collections::BTreeSet::new();
        for index in 0..section_count {
            let start = table_start + index * SECTION_ENTRY_BYTES;
            let kind = read_u16(bytes, start)?;
            if !known_section(kind) || !seen.insert(kind) {
                return Err(CheckpointError::Corrupt(
                    "unknown or duplicate section".into(),
                ));
            }
            if read_u16(bytes, start + 2)? != 0 || read_u32(bytes, start + 4)? != 0 {
                return Err(CheckpointError::Corrupt(
                    "reserved section field is nonzero".into(),
                ));
            }
            let offset = read_u64(bytes, start + 8)?;
            let length = read_u64(bytes, start + 16)?;
            let end = offset
                .checked_add(length)
                .ok_or_else(|| CheckpointError::Bounds("section range overflow".into()))?;
            if offset < header_len as u64 || end > total_len || length > MAX_SECTION_BYTES {
                return Err(CheckpointError::Corrupt(
                    "section range is outside envelope".into(),
                ));
            }
            let digest_start = start + 24;
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&bytes[digest_start..digest_start + 32]);
            entries.push((kind, offset, length, digest));
        }
        if seen.len() != SECTION_COUNT {
            return Err(CheckpointError::Corrupt("missing section".into()));
        }
        entries.sort_by_key(|entry| entry.1);
        let mut next_offset = header_len as u64;
        for (_, offset, length, digest) in &entries {
            if *offset != next_offset {
                return if *offset < next_offset {
                    Err(CheckpointError::Corrupt("overlapping sections".into()))
                } else {
                    Err(CheckpointError::Corrupt(
                        "gap or trailing section bytes".into(),
                    ))
                };
            }
            let start = *offset as usize;
            let end = start + *length as usize;
            if sha256(&bytes[start..end]) != *digest {
                return Err(CheckpointError::Corrupt("section checksum mismatch".into()));
            }
            next_offset = next_offset
                .checked_add(*length)
                .ok_or_else(|| CheckpointError::Bounds("section offset overflow".into()))?;
        }
        if next_offset != total_len {
            return Err(CheckpointError::TrailingBytes);
        }

        let mut payload = CheckpointPayload::default();
        for (kind, offset, length, _) in entries {
            let data = &bytes[offset as usize..(offset + length) as usize];
            match kind {
                SECTION_TOKEN_HISTORY => payload.token_history = decode_tokens(data)?,
                SECTION_CONVERSATION => {
                    payload.conversation =
                        bounded_copy(data, "conversation", MAX_CONVERSATION_BYTES as u64)?
                }
                SECTION_STATE_LAYERS => payload.state_layers = decode_state_layers(data)?,
                SECTION_STATE_PLANES => payload.state_planes = decode_state_planes(data)?,
                SECTION_SAMPLER_STATE => {
                    payload.sampler_state = bounded_copy(data, "sampler state", MAX_SECTION_BYTES)?
                }
                SECTION_GRAMMAR_STATE => {
                    payload.grammar_state = bounded_copy(data, "grammar state", MAX_SECTION_BYTES)?
                }
                SECTION_STOP_STATE => {
                    payload.stop_state = bounded_copy(data, "stop state", MAX_SECTION_BYTES)?
                }
                _ => unreachable!(),
            }
        }
        if token_count != payload.token_history.len() as u64 {
            return Err(CheckpointError::Corrupt(
                "token count does not match history".into(),
            ));
        }
        if identity.token_sequence_digest != token_sequence_digest(&payload.token_history) {
            return Err(CheckpointError::Corrupt(
                "token sequence digest mismatch".into(),
            ));
        }
        let checkpoint = Self {
            header: SessionCheckpointHeader {
                identity,
                token_count,
                absolute_position,
                logical_position,
                generation_state_version,
            },
            payload,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Revalidate an in-memory checkpoint before a backend consumes any
    /// opaque state. This performs the same bounded structural and identity
    /// checks as encoding without allocating a second copy of the payload.
    pub fn validate(&self) -> Result<(), CheckpointError> {
        self.header.identity.validate()?;
        self.payload.validate(self.header.identity.kv_encoding)?;
        if self.header.token_count != self.payload.token_history.len() as u64 {
            return Err(CheckpointError::Invalid(
                "token count does not match history".into(),
            ));
        }
        if self.header.identity.token_sequence_digest
            != token_sequence_digest(&self.payload.token_history)
        {
            return Err(CheckpointError::Invalid(
                "token sequence digest mismatch".into(),
            ));
        }
        let position_delta = self
            .header
            .absolute_position
            .checked_sub(self.header.logical_position)
            .ok_or_else(|| {
                CheckpointError::Invalid("absolute position precedes logical position".into())
            })?;
        i64::try_from(position_delta).map_err(|_| {
            CheckpointError::Bounds("absolute/logical position delta exceeds i64".into())
        })?;
        Ok(())
    }

    fn encode_sections(&self) -> Result<Vec<(u16, Vec<u8>)>, CheckpointError> {
        let mut tokens = Vec::with_capacity(4 + self.payload.token_history.len() * 4);
        tokens.extend_from_slice(&(self.payload.token_history.len() as u32).to_le_bytes());
        for token in &self.payload.token_history {
            tokens.extend_from_slice(&token.to_le_bytes());
        }
        let layers = encode_state_layers(&self.payload.state_layers)?;
        let planes = encode_state_planes(&self.payload.state_planes)?;
        let sections = vec![
            (SECTION_TOKEN_HISTORY, tokens),
            (SECTION_CONVERSATION, self.payload.conversation.clone()),
            (SECTION_STATE_LAYERS, layers),
            (SECTION_STATE_PLANES, planes),
            (SECTION_SAMPLER_STATE, self.payload.sampler_state.clone()),
            (SECTION_GRAMMAR_STATE, self.payload.grammar_state.clone()),
            (SECTION_STOP_STATE, self.payload.stop_state.clone()),
        ];
        for (_, bytes) in &sections {
            if bytes.len() as u64 > MAX_SECTION_BYTES {
                return Err(CheckpointError::Bounds(
                    "section exceeds maximum size".into(),
                ));
            }
        }
        Ok(sections)
    }
}

fn known_section(kind: u16) -> bool {
    (SECTION_TOKEN_HISTORY..=SECTION_STOP_STATE).contains(&kind)
}

fn encode_identity(identity: &CheckpointIdentity) -> Result<Vec<u8>, CheckpointError> {
    identity.validate()?;
    let mut bytes = Vec::new();
    for value in [
        &identity.model_lock_fingerprint,
        &identity.derived_artifact_identity,
        &identity.adapter_identity,
        &identity.renderer_identity,
        &identity.tokenizer_identity,
        &identity.target_semantics,
        &identity.plan_digest,
    ] {
        bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    bytes.extend_from_slice(&identity.token_sequence_digest);
    bytes.push(kv_encoding_tag(identity.kv_encoding));
    bytes.extend_from_slice(&[0u8; 3]);
    bytes.extend_from_slice(&identity.kv_descriptor_digest);
    bytes.extend_from_slice(&identity.context_policy_digest);
    Ok(bytes)
}

fn decode_identity(bytes: &[u8]) -> Result<CheckpointIdentity, CheckpointError> {
    let mut cursor = Cursor::new(bytes);
    let model_lock_fingerprint = cursor.string("model lock fingerprint")?;
    let derived_artifact_identity = cursor.string("derived artifact identity")?;
    let adapter_identity = cursor.string("adapter identity")?;
    let renderer_identity = cursor.string("renderer identity")?;
    let tokenizer_identity = cursor.string("tokenizer identity")?;
    let target_semantics = cursor.string("target semantics")?;
    let plan_digest = cursor.string("plan digest")?;
    let token_sequence_digest = cursor.array::<32>("token sequence digest")?;
    let kv_encoding = decode_kv_encoding(cursor.byte("KV encoding")?)?;
    if cursor.byte("identity reserved")? != 0
        || cursor.byte("identity reserved")? != 0
        || cursor.byte("identity reserved")? != 0
    {
        return Err(CheckpointError::Corrupt(
            "identity reserved field is nonzero".into(),
        ));
    }
    let kv_descriptor_digest = cursor.array::<32>("KV descriptor digest")?;
    let context_policy_digest = cursor.array::<32>("context policy digest")?;
    if !cursor.is_empty() {
        return Err(CheckpointError::TrailingBytes);
    }
    let identity = CheckpointIdentity {
        model_lock_fingerprint,
        derived_artifact_identity,
        adapter_identity,
        renderer_identity,
        tokenizer_identity,
        target_semantics,
        plan_digest,
        token_sequence_digest,
        kv_encoding,
        kv_descriptor_digest,
        context_policy_digest,
    };
    identity.validate()?;
    Ok(identity)
}

fn compare_identity(
    expected: &CheckpointIdentity,
    actual: &CheckpointIdentity,
) -> Result<(), CheckpointError> {
    expected.validate()?;
    actual.validate()?;
    let checks = [
        (
            "model_lock_fingerprint",
            expected.model_lock_fingerprint.as_str(),
            actual.model_lock_fingerprint.as_str(),
        ),
        (
            "derived_artifact_identity",
            expected.derived_artifact_identity.as_str(),
            actual.derived_artifact_identity.as_str(),
        ),
        (
            "adapter_identity",
            expected.adapter_identity.as_str(),
            actual.adapter_identity.as_str(),
        ),
        (
            "renderer_identity",
            expected.renderer_identity.as_str(),
            actual.renderer_identity.as_str(),
        ),
        (
            "tokenizer_identity",
            expected.tokenizer_identity.as_str(),
            actual.tokenizer_identity.as_str(),
        ),
        (
            "target_semantics",
            expected.target_semantics.as_str(),
            actual.target_semantics.as_str(),
        ),
        (
            "plan_digest",
            expected.plan_digest.as_str(),
            actual.plan_digest.as_str(),
        ),
    ];
    for (field, expected_value, actual_value) in checks {
        if expected_value != actual_value {
            return Err(CheckpointError::IdentityMismatch {
                field,
                expected: expected_value.to_owned(),
                actual: actual_value.to_owned(),
            });
        }
    }
    if expected.token_sequence_digest != actual.token_sequence_digest {
        return Err(CheckpointError::IdentityMismatch {
            field: "token_sequence_digest",
            expected: hex_digest(&expected.token_sequence_digest),
            actual: hex_digest(&actual.token_sequence_digest),
        });
    }
    if expected.kv_encoding != actual.kv_encoding {
        return Err(CheckpointError::IdentityMismatch {
            field: "kv_encoding",
            expected: kv_encoding_name(expected.kv_encoding).to_owned(),
            actual: kv_encoding_name(actual.kv_encoding).to_owned(),
        });
    }
    if expected.kv_descriptor_digest != actual.kv_descriptor_digest {
        return Err(CheckpointError::IdentityMismatch {
            field: "kv_descriptor_digest",
            expected: hex_digest(&expected.kv_descriptor_digest),
            actual: hex_digest(&actual.kv_descriptor_digest),
        });
    }
    if expected.context_policy_digest != actual.context_policy_digest {
        return Err(CheckpointError::IdentityMismatch {
            field: "context_policy_digest",
            expected: hex_digest(&expected.context_policy_digest),
            actual: hex_digest(&actual.context_policy_digest),
        });
    }
    Ok(())
}

const fn kv_encoding_name(encoding: KvCacheEncoding) -> &'static str {
    match encoding {
        KvCacheEncoding::Fp16 => "fp16",
        KvCacheEncoding::Fp8E4M3Fn => "fp8-e4m3fn-dynamic",
        KvCacheEncoding::Fp8E4M3FnStatic => "fp8-e4m3fn-static",
        KvCacheEncoding::Nvfp4 => "nvfp4",
    }
}

const fn kv_encoding_tag(encoding: KvCacheEncoding) -> u8 {
    match encoding {
        KvCacheEncoding::Fp16 => 0,
        KvCacheEncoding::Fp8E4M3Fn => 1,
        KvCacheEncoding::Fp8E4M3FnStatic => 2,
        KvCacheEncoding::Nvfp4 => 3,
    }
}

fn decode_kv_encoding(tag: u8) -> Result<KvCacheEncoding, CheckpointError> {
    match tag {
        0 => Ok(KvCacheEncoding::Fp16),
        1 => Ok(KvCacheEncoding::Fp8E4M3Fn),
        2 => Ok(KvCacheEncoding::Fp8E4M3FnStatic),
        3 => Ok(KvCacheEncoding::Nvfp4),
        _ => Err(CheckpointError::Corrupt("unknown KV encoding".into())),
    }
}

fn decode_tokens(bytes: &[u8]) -> Result<Vec<u32>, CheckpointError> {
    if bytes.len() < 4 {
        return Err(CheckpointError::Truncated);
    }
    let count = u32::from_le_bytes(bytes[..4].try_into().expect("four bytes")) as usize;
    if count > MAX_TOKEN_HISTORY {
        return Err(CheckpointError::Bounds(
            "token history exceeds limit".into(),
        ));
    }
    let expected = 4usize
        .checked_add(
            count
                .checked_mul(4)
                .ok_or_else(|| CheckpointError::Bounds("token bytes overflow".into()))?,
        )
        .ok_or_else(|| CheckpointError::Bounds("token bytes overflow".into()))?;
    if expected != bytes.len() {
        return if expected > bytes.len() {
            Err(CheckpointError::Truncated)
        } else {
            Err(CheckpointError::TrailingBytes)
        };
    }
    let mut tokens = Vec::with_capacity(count);
    for chunk in bytes[4..].chunks_exact(4) {
        tokens.push(u32::from_le_bytes(chunk.try_into().expect("four bytes")));
    }
    Ok(tokens)
}

fn encode_state_layers(layers: &[StateLayerMetadataV1]) -> Result<Vec<u8>, CheckpointError> {
    let count = u16::try_from(layers.len())
        .map_err(|_| CheckpointError::Bounds("too many state layers".into()))?;
    let mut bytes = Vec::with_capacity(4usize.saturating_add(layers.len().saturating_mul(32)));
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    for layer in layers {
        bytes.push(state_owner_tag(layer.owner));
        bytes.push(layer.active_slot.unwrap_or(u8::MAX));
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&layer.layer_id.to_le_bytes());
        bytes.extend_from_slice(&layer.published_length.to_le_bytes());
        bytes.extend_from_slice(&layer.generation.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_state_layers(bytes: &[u8]) -> Result<Vec<StateLayerMetadataV1>, CheckpointError> {
    if bytes.len() < 4 {
        return Err(CheckpointError::Truncated);
    }
    let count = u16::from_le_bytes(bytes[..2].try_into().expect("two bytes")) as usize;
    if count > MAX_STATE_LAYERS {
        return Err(CheckpointError::Bounds("too many state layers".into()));
    }
    if bytes[2..4] != [0, 0] {
        return Err(CheckpointError::Corrupt(
            "state-layer table reserved field is nonzero".into(),
        ));
    }
    let mut cursor = Cursor::new(&bytes[4..]);
    let mut layers = Vec::with_capacity(count);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..count {
        let owner = decode_state_owner(cursor.byte("state owner")?)?;
        let active_slot = match cursor.byte("active slot")? {
            u8::MAX => None,
            slot => Some(slot),
        };
        if cursor.u16("state-layer reserved")? != 0 {
            return Err(CheckpointError::Corrupt(
                "state-layer reserved field is nonzero".into(),
            ));
        }
        let layer_id = cursor.u32("state layer ID")?;
        let published_length = cursor.u64("state published length")?;
        let generation = cursor.u64("state generation")?;
        if cursor.u64("state-layer reserved")? != 0 {
            return Err(CheckpointError::Corrupt(
                "state-layer reserved field is nonzero".into(),
            ));
        }
        if !seen.insert((owner, layer_id)) {
            return Err(CheckpointError::Corrupt(
                "duplicate state-layer metadata".into(),
            ));
        }
        layers.push(StateLayerMetadataV1 {
            owner,
            layer_id,
            published_length,
            generation,
            active_slot,
        });
    }
    if !cursor.is_empty() {
        return Err(CheckpointError::TrailingBytes);
    }
    Ok(layers)
}

fn encode_state_planes(planes: &[OpaqueStatePlane]) -> Result<Vec<u8>, CheckpointError> {
    let count = u16::try_from(planes.len())
        .map_err(|_| CheckpointError::Bounds("too many state planes".into()))?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    for plane in planes {
        bytes.push(state_owner_tag(plane.owner));
        bytes.push(state_plane_tag(plane.plane));
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&plane.layer_id.to_le_bytes());
        bytes.extend_from_slice(&(plane.bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&plane.bytes);
    }
    Ok(bytes)
}

fn decode_state_planes(bytes: &[u8]) -> Result<Vec<OpaqueStatePlane>, CheckpointError> {
    if bytes.len() < 4 {
        return Err(CheckpointError::Truncated);
    }
    let count = u16::from_le_bytes(bytes[..2].try_into().expect("two bytes")) as usize;
    if count > MAX_STATE_PLANES {
        return Err(CheckpointError::Bounds("too many state planes".into()));
    }
    if bytes[2..4] != [0, 0] {
        return Err(CheckpointError::Corrupt(
            "state-plane table reserved field is nonzero".into(),
        ));
    }
    let mut cursor = Cursor::new(&bytes[4..]);
    let mut planes = Vec::with_capacity(count);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..count {
        let owner = decode_state_owner(cursor.byte("state owner")?)?;
        let plane = decode_state_plane(cursor.byte("state plane")?)?;
        if plane.owner() != owner {
            return Err(CheckpointError::Corrupt(
                "state plane kind does not match its owner".into(),
            ));
        }
        if cursor.u16("state-plane reserved")? != 0 {
            return Err(CheckpointError::Corrupt(
                "state-plane reserved field is nonzero".into(),
            ));
        }
        let layer_id = cursor.u32("state layer ID")?;
        if !seen.insert((owner, layer_id, plane)) {
            return Err(CheckpointError::Corrupt("duplicate state plane".into()));
        }
        let length = cursor.u64("plane length")?;
        if length > MAX_SECTION_BYTES {
            return Err(CheckpointError::Bounds(
                "state plane exceeds section limit".into(),
            ));
        }
        let bytes = cursor.take(length as usize, "state plane")?.to_vec();
        planes.push(OpaqueStatePlane {
            owner,
            layer_id,
            plane,
            bytes,
        });
    }
    if !cursor.is_empty() {
        return Err(CheckpointError::TrailingBytes);
    }
    Ok(planes)
}

const fn state_owner_tag(owner: StateOwnerKindV1) -> u8 {
    owner as u8
}

fn decode_state_owner(tag: u8) -> Result<StateOwnerKindV1, CheckpointError> {
    match tag {
        1 => Ok(StateOwnerKindV1::Kv),
        2 => Ok(StateOwnerKindV1::LinearAttention),
        _ => Err(CheckpointError::Corrupt("unknown state owner".into())),
    }
}

const fn state_plane_tag(plane: StatePlaneKindV1) -> u8 {
    plane as u8
}

fn decode_state_plane(tag: u8) -> Result<StatePlaneKindV1, CheckpointError> {
    match tag {
        1 => Ok(StatePlaneKindV1::KvKey),
        2 => Ok(StatePlaneKindV1::KvValue),
        3 => Ok(StatePlaneKindV1::KvKeyScale),
        4 => Ok(StatePlaneKindV1::KvValueScale),
        5 => Ok(StatePlaneKindV1::KvKeyOuterScale),
        6 => Ok(StatePlaneKindV1::KvValueOuterScale),
        17 => Ok(StatePlaneKindV1::LinearConvSlot0),
        18 => Ok(StatePlaneKindV1::LinearConvSlot1),
        19 => Ok(StatePlaneKindV1::LinearRecurrentSlot0),
        20 => Ok(StatePlaneKindV1::LinearRecurrentSlot1),
        21 => Ok(StatePlaneKindV1::LinearScratch),
        _ => Err(CheckpointError::Corrupt("unknown state plane".into())),
    }
}

fn bounded_copy(bytes: &[u8], name: &str, limit: u64) -> Result<Vec<u8>, CheckpointError> {
    if bytes.len() as u64 > limit {
        return Err(CheckpointError::Bounds(format!(
            "{name} exceeds {limit} bytes"
        )));
    }
    Ok(bytes.to_vec())
}

fn sha256_with_zeroed_checksum(bytes: &[u8]) -> Result<[u8; 32], CheckpointError> {
    if bytes.len() < CHECKSUM_END {
        return Err(CheckpointError::Truncated);
    }
    let mut copy = bytes.to_vec();
    copy[CHECKSUM_OFFSET..CHECKSUM_END].fill(0);
    Ok(sha256(&copy))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, CheckpointError> {
    let end = offset.checked_add(2).ok_or(CheckpointError::Truncated)?;
    bytes
        .get(offset..end)
        .map(|value| u16::from_le_bytes(value.try_into().expect("two bytes")))
        .ok_or(CheckpointError::Truncated)
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, CheckpointError> {
    let end = offset.checked_add(4).ok_or(CheckpointError::Truncated)?;
    bytes
        .get(offset..end)
        .map(|value| u32::from_le_bytes(value.try_into().expect("four bytes")))
        .ok_or(CheckpointError::Truncated)
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, CheckpointError> {
    let end = offset.checked_add(8).ok_or(CheckpointError::Truncated)?;
    bytes
        .get(offset..end)
        .map(|value| u64::from_le_bytes(value.try_into().expect("eight bytes")))
        .ok_or(CheckpointError::Truncated)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize, name: &str) -> Result<&'a [u8], CheckpointError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| CheckpointError::Bounds(format!("{name} length overflow")))?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or(CheckpointError::Truncated)?;
        self.offset = end;
        Ok(result)
    }

    fn byte(&mut self, name: &str) -> Result<u8, CheckpointError> {
        Ok(*self.take(1, name)?.first().expect("one byte"))
    }

    fn u16(&mut self, name: &str) -> Result<u16, CheckpointError> {
        Ok(u16::from_le_bytes(
            self.take(2, name)?.try_into().expect("two bytes"),
        ))
    }

    fn u64(&mut self, name: &str) -> Result<u64, CheckpointError> {
        Ok(u64::from_le_bytes(
            self.take(8, name)?.try_into().expect("eight bytes"),
        ))
    }

    fn array<const N: usize>(&mut self, name: &str) -> Result<[u8; N], CheckpointError> {
        self.take(N, name)?
            .try_into()
            .map_err(|_| CheckpointError::Truncated)
    }

    fn string(&mut self, name: &str) -> Result<String, CheckpointError> {
        let length = self.u32(name)? as usize;
        if length == 0 || length > MAX_IDENTITY_FIELD_BYTES {
            return Err(CheckpointError::Bounds(format!(
                "{name} has invalid length"
            )));
        }
        let bytes = self.take(length, name)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| CheckpointError::Corrupt(format!("{name} is not UTF-8")))
    }

    fn u32(&mut self, name: &str) -> Result<u32, CheckpointError> {
        Ok(u32::from_le_bytes(
            self.take(4, name)?.try_into().expect("four bytes"),
        ))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// A bounded, owner-checked, atomically replaced checkpoint directory.
#[derive(Clone)]
pub struct CheckpointStore {
    root: PathBuf,
    max_bytes: u64,
    max_file_bytes: u64,
    lock: Arc<Mutex<()>>,
    root_owner: u32,
}

/// Descriptive alias for callers that prefer the full name.
pub type SessionCheckpointStore = CheckpointStore;

impl CheckpointStore {
    pub fn new(root: impl AsRef<Path>, max_bytes: u64) -> Result<Self, CheckpointError> {
        Self::with_limits(root, max_bytes, max_bytes.min(MAX_CHECKPOINT_BYTES))
    }

    pub fn with_limits(
        root: impl AsRef<Path>,
        max_bytes: u64,
        max_file_bytes: u64,
    ) -> Result<Self, CheckpointError> {
        let root = root.as_ref().to_path_buf();
        if max_bytes == 0 || max_file_bytes == 0 || max_file_bytes > MAX_CHECKPOINT_BYTES {
            return Err(CheckpointError::Bounds(
                "invalid checkpoint store limits".into(),
            ));
        }
        if let Ok(metadata) = fs::symlink_metadata(&root) {
            if metadata.file_type().is_symlink() {
                return Err(CheckpointError::Security(
                    "checkpoint root is a symlink".into(),
                ));
            }
            if !metadata.is_dir() {
                return Err(CheckpointError::Security(
                    "checkpoint root is not a directory".into(),
                ));
            }
            validate_directory_permissions(&root, &metadata)?;
        } else {
            fs::create_dir_all(&root).map_err(|error| io_error(&root, error))?;
            let metadata = fs::symlink_metadata(&root).map_err(|error| io_error(&root, error))?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(CheckpointError::Security(
                    "checkpoint root is not a directory".into(),
                ));
            }
            #[cfg(unix)]
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
                .map_err(|error| io_error(&root, error))?;
            let metadata = fs::symlink_metadata(&root).map_err(|error| io_error(&root, error))?;
            validate_directory_permissions(&root, &metadata)?;
        }
        let metadata = fs::symlink_metadata(&root).map_err(|error| io_error(&root, error))?;
        let root_owner = owner_id(&metadata);
        Ok(Self {
            root,
            max_bytes,
            max_file_bytes,
            lock: Arc::new(Mutex::new(())),
            root_owner,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn save(
        &self,
        id: &str,
        checkpoint: &SessionCheckpoint,
    ) -> Result<PathBuf, CheckpointError> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| CheckpointError::Security("store lock poisoned".into()))?;
        self.validate_root()?;
        let filename = checkpoint_filename(id)?;
        let target = self.root.join(&filename);
        validate_existing_target(&target, self.root_owner)?;
        let bytes = checkpoint.encode()?;
        let byte_len = bytes.len() as u64;
        if byte_len > self.max_file_bytes {
            return Err(CheckpointError::QuotaExceeded {
                requested: byte_len,
                available: self.max_file_bytes,
            });
        }
        let current = self.scan_usage()?;
        let old_size = fs::symlink_metadata(&target)
            .ok()
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let retained = current.saturating_sub(old_size);
        let available = self.max_bytes.saturating_sub(retained);
        if byte_len > available {
            return Err(CheckpointError::QuotaExceeded {
                requested: byte_len,
                available,
            });
        }

        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp_name = format!(".{filename}.{counter:016x}.tmp");
        if temp_name.len() > MAX_TEMP_NAME_BYTES {
            return Err(CheckpointError::Bounds(
                "temporary filename too long".into(),
            ));
        }
        let temp = self.root.join(temp_name);
        let mut file = OpenOptions::new();
        file.write(true).create_new(true);
        #[cfg(unix)]
        file.mode(0o600);
        let mut file = file.open(&temp).map_err(|error| io_error(&temp, error))?;
        #[cfg(unix)]
        fs::set_permissions(&temp, fs::Permissions::from_mode(0o600))
            .map_err(|error| io_error(&temp, error))?;
        let result = (|| {
            file.write_all(&bytes)
                .map_err(|error| io_error(&temp, error))?;
            file.sync_all().map_err(|error| io_error(&temp, error))?;
            drop(file);
            fs::rename(&temp, &target).map_err(|error| io_error(&target, error))?;
            let directory = File::open(&self.root).map_err(|error| io_error(&self.root, error))?;
            directory
                .sync_all()
                .map_err(|error| io_error(&self.root, error))?;
            Ok::<(), CheckpointError>(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result.map(|()| target)
    }

    pub fn load(
        &self,
        id: &str,
        expected_identity: &CheckpointIdentity,
    ) -> Result<SessionCheckpoint, CheckpointError> {
        let checkpoint = self.load_validated(id)?;
        compare_identity(expected_identity, &checkpoint.header.identity)?;
        Ok(checkpoint)
    }

    /// Loads and fully validates a checkpoint envelope without binding it to
    /// a caller-provided identity. Filesystem ownership, permissions,
    /// hard-link count, configured size limits, schema, section bounds, and
    /// every checksum are still verified. Callers must compare the returned
    /// header identity with their exact runtime identity before importing any
    /// opaque state bytes.
    pub fn load_validated(&self, id: &str) -> Result<SessionCheckpoint, CheckpointError> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| CheckpointError::Security("store lock poisoned".into()))?;
        self.validate_root()?;
        let filename = checkpoint_filename(id)?;
        let usage = self.scan_usage()?;
        if usage > self.max_bytes {
            return Err(CheckpointError::QuotaExceeded {
                requested: usage,
                available: self.max_bytes,
            });
        }
        let path = self.root.join(filename);
        let metadata = validate_existing_target(&path, self.root_owner)?
            .ok_or_else(|| io_error(&path, "checkpoint does not exist"))?;
        if metadata.len() > self.max_file_bytes || metadata.len() > MAX_CHECKPOINT_BYTES {
            return Err(CheckpointError::Bounds(
                "checkpoint file exceeds size limit".into(),
            ));
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(o_nofollow());
        let mut file = options
            .open(&path)
            .map_err(|error| io_error(&path, error))?;
        let fd_metadata = file.metadata().map_err(|error| io_error(&path, error))?;
        validate_file_metadata(&fd_metadata, self.root_owner)?;
        if fd_metadata.len() != metadata.len() {
            return Err(CheckpointError::Security(
                "checkpoint changed while opening".into(),
            ));
        }
        let mut bytes = Vec::with_capacity(fd_metadata.len() as usize);
        let read_limit = self
            .max_file_bytes
            .min(MAX_CHECKPOINT_BYTES)
            .saturating_add(1);
        Read::by_ref(&mut file)
            .take(read_limit)
            .read_to_end(&mut bytes)
            .map_err(|error| io_error(&path, error))?;
        if bytes.len() as u64 > self.max_file_bytes || bytes.len() as u64 > MAX_CHECKPOINT_BYTES {
            return Err(CheckpointError::Bounds(
                "checkpoint file exceeds size limit".into(),
            ));
        }
        SessionCheckpoint::decode(&bytes)
    }

    pub fn usage_bytes(&self) -> Result<u64, CheckpointError> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| CheckpointError::Security("store lock poisoned".into()))?;
        self.validate_root()?;
        self.scan_usage()
    }

    fn validate_root(&self) -> Result<(), CheckpointError> {
        let metadata =
            fs::symlink_metadata(&self.root).map_err(|error| io_error(&self.root, error))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(CheckpointError::Security("checkpoint root changed".into()));
        }
        if owner_id(&metadata) != self.root_owner {
            return Err(CheckpointError::Security(
                "checkpoint root owner changed".into(),
            ));
        }
        validate_directory_permissions(&self.root, &metadata)
    }

    fn scan_usage(&self) -> Result<u64, CheckpointError> {
        let entries = fs::read_dir(&self.root).map_err(|error| io_error(&self.root, error))?;
        let mut total = 0u64;
        for entry in entries {
            let entry = entry.map_err(|error| io_error(&self.root, error))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).map_err(|error| io_error(&path, error))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(CheckpointError::Security(
                    "checkpoint directory contains non-regular entry".into(),
                ));
            }
            validate_file_metadata(&metadata, self.root_owner)?;
            total = total
                .checked_add(metadata.len())
                .ok_or_else(|| CheckpointError::Bounds("checkpoint quota overflow".into()))?;
            if total > self.max_bytes {
                return Ok(total);
            }
        }
        Ok(total)
    }
}

fn checkpoint_filename(id: &str) -> Result<String, CheckpointError> {
    if id.is_empty() || id.len() > 128 || id == "." || id == ".." {
        return Err(CheckpointError::PathViolation(
            "invalid checkpoint ID".into(),
        ));
    }
    if !id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        return Err(CheckpointError::PathViolation(
            "checkpoint ID contains path characters".into(),
        ));
    }
    Ok(format!("{id}.ckpt"))
}

fn validate_directory_permissions(
    _path: &Path,
    metadata: &Metadata,
) -> Result<(), CheckpointError> {
    #[cfg(unix)]
    {
        if metadata.uid() != effective_user_id() {
            return Err(CheckpointError::Security(
                "checkpoint directory owner mismatch".into(),
            ));
        }
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(CheckpointError::Security(
                "checkpoint directory is group/world accessible".into(),
            ));
        }
    }
    Ok(())
}

fn validate_existing_target(path: &Path, owner: u32) -> Result<Option<Metadata>, CheckpointError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(CheckpointError::Security(
                    "checkpoint target is a symlink".into(),
                ));
            }
            validate_file_metadata(&metadata, owner)?;
            Ok(Some(metadata))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(io_error(path, error)),
    }
}

fn validate_file_metadata(metadata: &Metadata, owner: u32) -> Result<(), CheckpointError> {
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(CheckpointError::Security(
            "checkpoint target is not a regular file".into(),
        ));
    }
    if owner_id(metadata) != owner {
        return Err(CheckpointError::Security(
            "checkpoint owner mismatch".into(),
        ));
    }
    #[cfg(unix)]
    {
        if metadata.nlink() != 1 {
            return Err(CheckpointError::Security(
                "checkpoint hard-link count is not one".into(),
            ));
        }
        if metadata.permissions().mode() & 0o077 != 0
            || metadata.permissions().mode() & 0o600 != 0o600
        {
            return Err(CheckpointError::Security(
                "checkpoint permissions are not 0600".into(),
            ));
        }
    }
    Ok(())
}

fn owner_id(metadata: &Metadata) -> u32 {
    #[cfg(unix)]
    {
        metadata.uid()
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        0
    }
}

#[cfg(unix)]
fn effective_user_id() -> u32 {
    // SAFETY: `geteuid` has no preconditions and does not dereference pointers.
    unsafe { libc::geteuid() }
}

#[cfg(unix)]
fn o_nofollow() -> i32 {
    // Linux is the supported host.  Keep this local to avoid a new libc
    // dependency solely for a safe read-open operation.
    0o400000
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(1);

    fn test_directory(label: &str) -> PathBuf {
        let id = TEST_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "sllm-checkpoint-load-validated-{label}-{}-{id}",
            std::process::id()
        ))
    }

    fn test_checkpoint() -> SessionCheckpoint {
        let tokens = [3, 5, 8];
        let identity = CheckpointIdentity::for_tokens(
            format!("sha256:{}", "a".repeat(64)),
            "artifact-v1",
            "adapter-none-v1",
            "renderer-v1",
            "tokenizer-v1",
            "gfx1201",
            "sha256:plan-v1",
            &tokens,
            KvCacheEncoding::Fp16,
            [2; 32],
            [3; 32],
        )
        .expect("test identity");
        SessionCheckpoint::new(
            identity,
            3,
            3,
            1,
            CheckpointPayload {
                token_history: tokens.to_vec(),
                conversation: b"private-conversation-marker".to_vec(),
                ..CheckpointPayload::default()
            },
        )
        .expect("test checkpoint")
    }

    fn write_secure(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).expect("write checkpoint fixture");
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .expect("secure checkpoint fixture");
    }

    #[test]
    fn load_validated_defers_only_runtime_identity_binding() {
        let root = test_directory("identity-secret-root");
        let checkpoint = test_checkpoint();
        let store = CheckpointStore::new(&root, MAX_CHECKPOINT_BYTES).expect("checkpoint store");
        store
            .save("identity-secret-name", &checkpoint)
            .expect("save checkpoint");

        assert_eq!(
            store
                .load_validated("identity-secret-name")
                .expect("unbound strict load"),
            checkpoint
        );
        let mut wrong_identity = checkpoint.header.identity.clone();
        wrong_identity.adapter_identity = "different-adapter".to_owned();
        assert!(matches!(
            store.load("identity-secret-name", &wrong_identity),
            Err(CheckpointError::IdentityMismatch {
                field: "adapter_identity",
                ..
            })
        ));

        fs::remove_dir_all(root).expect("remove checkpoint fixture");
    }

    #[test]
    fn load_validated_rejects_malformed_schema_and_checksum_without_disclosure() {
        let root = test_directory("malformed-secret-root");
        let checkpoint = test_checkpoint();
        let bytes = checkpoint.encode().expect("encode checkpoint");
        let store = CheckpointStore::new(&root, MAX_CHECKPOINT_BYTES).expect("checkpoint store");
        let path = root.join("malformed-secret-name.ckpt");

        for (id, error) in [
            (
                "missing-secret-name",
                store
                    .load_validated("missing-secret-name")
                    .expect_err("missing checkpoint must fail"),
            ),
            (
                "../traversal-secret-name",
                store
                    .load_validated("../traversal-secret-name")
                    .expect_err("path traversal must fail"),
            ),
        ] {
            let rendered = error.to_string();
            assert!(!rendered.contains(id));
            assert!(!rendered.contains(root.to_string_lossy().as_ref()));
        }

        let mut wrong_schema = bytes.clone();
        wrong_schema[8..10].copy_from_slice(&u16::MAX.to_le_bytes());
        write_secure(&path, &wrong_schema);
        let schema_error = store
            .load_validated("malformed-secret-name")
            .expect_err("schema mismatch must fail");
        assert!(matches!(
            schema_error,
            CheckpointError::UnsupportedVersion(u16::MAX)
        ));

        let mut corrupt = bytes;
        let final_byte = corrupt.last_mut().expect("encoded checkpoint is nonempty");
        *final_byte ^= 0x80;
        write_secure(&path, &corrupt);
        let checksum_error = store
            .load_validated("malformed-secret-name")
            .expect_err("checksum mismatch must fail");
        assert!(matches!(checksum_error, CheckpointError::Corrupt(_)));
        let rendered = checksum_error.to_string();
        for secret in [
            "malformed-secret-name",
            root.to_string_lossy().as_ref(),
            "private-conversation-marker",
        ] {
            assert!(!rendered.contains(secret), "error exposed {secret}");
        }

        fs::remove_dir_all(root).expect("remove checkpoint fixture");
    }

    #[test]
    fn load_validated_enforces_configured_file_quota_without_disclosure() {
        let root = test_directory("quota-secret-root");
        let bytes = test_checkpoint().encode().expect("encode checkpoint");
        let store =
            CheckpointStore::with_limits(&root, bytes.len() as u64 * 2, bytes.len() as u64 - 1)
                .expect("checkpoint store");
        write_secure(&root.join("quota-secret-name.ckpt"), &bytes);

        let error = store
            .load_validated("quota-secret-name")
            .expect_err("oversized checkpoint must fail");
        assert!(matches!(error, CheckpointError::Bounds(_)));
        let rendered = error.to_string();
        assert!(!rendered.contains("quota-secret-name"));
        assert!(!rendered.contains(root.to_string_lossy().as_ref()));

        fs::remove_dir_all(root).expect("remove checkpoint fixture");

        let root = test_directory("aggregate-quota-secret-root");
        let store = CheckpointStore::with_limits(&root, bytes.len() as u64 - 1, bytes.len() as u64)
            .expect("checkpoint store");
        write_secure(&root.join("aggregate-secret-name.ckpt"), &bytes);
        let error = store
            .load_validated("aggregate-secret-name")
            .expect_err("aggregate quota overflow must fail");
        assert!(matches!(error, CheckpointError::QuotaExceeded { .. }));
        let rendered = error.to_string();
        assert!(!rendered.contains("aggregate-secret-name"));
        assert!(!rendered.contains(root.to_string_lossy().as_ref()));

        fs::remove_dir_all(root).expect("remove aggregate quota fixture");
    }

    #[cfg(unix)]
    #[test]
    fn load_validated_enforces_symlink_hardlink_and_mode_security() {
        use std::os::unix::fs::symlink;

        let root = test_directory("security-secret-root");
        let checkpoint = test_checkpoint();
        let store = CheckpointStore::new(&root, MAX_CHECKPOINT_BYTES).expect("checkpoint store");
        let path = store
            .save("security-secret-name", &checkpoint)
            .expect("save checkpoint");

        fs::hard_link(&path, root.join("hardlink.ckpt")).expect("create hard link");
        assert!(matches!(
            store.load_validated("security-secret-name"),
            Err(CheckpointError::Security(_))
        ));
        fs::remove_file(root.join("hardlink.ckpt")).expect("remove hard link");

        symlink(&path, root.join("symlink.ckpt")).expect("create symlink");
        assert!(matches!(
            store.load_validated("symlink"),
            Err(CheckpointError::Security(_))
        ));
        fs::remove_file(root.join("symlink.ckpt")).expect("remove symlink");

        fs::set_permissions(&path, fs::Permissions::from_mode(0o644))
            .expect("make checkpoint mode insecure");
        let error = store
            .load_validated("security-secret-name")
            .expect_err("insecure mode must fail");
        assert!(matches!(error, CheckpointError::Security(_)));
        let rendered = error.to_string();
        assert!(!rendered.contains("security-secret-name"));
        assert!(!rendered.contains(root.to_string_lossy().as_ref()));

        fs::remove_dir_all(root).expect("remove checkpoint fixture");
    }
}
