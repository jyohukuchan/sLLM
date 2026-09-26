//! Typed, backend-neutral request-local full-attention KV state contracts.
//!
//! The state is deliberately not a semantic operation.  Its storage and
//! length are owned by the backend, while these types make the C3a2 geometry,
//! versioned physical encoding, and append admission rules explicit at the
//! core boundary.

use std::fmt;
use std::num::NonZeroU64;

use crate::execution::{ExecutionSessionId, KvStateId};
use crate::{DType, Encoding, Fp8ResidentRepresentation, Fp8ScaleGranularity};

pub const KV_STATIC_FP8_SLIDING_WINDOW: u64 = 1024;
pub const KV_STATIC_FP8_SLIDING_MAX_CAPACITY: u64 = 262_144;

/// Versioned physical encoding selected for a request-local KV state.
///
/// This is backend metadata, not a user-facing generation option. The model
/// runtime chooses it from the loaded model recipe and target capabilities.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum KvCacheEncoding {
    Fp16,
    Fp8E4M3Fn,
    /// Provider-supplied layer-static E4M3 decode scales. Scale values live on
    /// [`KvStateDescriptor`], not in this encoding tag.
    Fp8E4M3FnStatic,
    Nvfp4,
    /// Logical E4M3 KV values with one E8M0 scale for each consecutive 16
    /// head-dimension lanes. The physical OCP/FNUZ variant is descriptor
    /// metadata and must never be inferred by reinterpreting resident bytes.
    Fp8E4M3Block16,
    /// Logical E5M2 KV values with one E8M0 scale for each consecutive 16
    /// head-dimension lanes.
    Fp8E5M2Block16,
    /// Standard OCP MXFP8 using E4M3FN values and one E8M0 scale per 32
    /// consecutive head-dimension lanes.
    #[default]
    Mxfp8E4,
    /// Standard OCP MXFP8 using E5M2 values and one E8M0 scale per 32
    /// consecutive head-dimension lanes.
    Mxfp8E5,
}

impl KvCacheEncoding {
    /// Canonical public spelling. Existing spellings and meanings are stable.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp8E4M3Fn => "fp8",
            Self::Fp8E4M3FnStatic => "fp8-static",
            Self::Nvfp4 => "nvfp4",
            Self::Fp8E4M3Block16 => "kv-fp8-e4-block16",
            Self::Fp8E5M2Block16 => "kv-fp8-e5-block16",
            Self::Mxfp8E4 => "kv-mxfp8-e4",
            Self::Mxfp8E5 => "kv-mxfp8-e5",
        }
    }

    pub const fn dtype(self) -> DType {
        match self {
            Self::Fp16 => DType::F16,
            Self::Fp8E4M3Fn | Self::Fp8E4M3FnStatic => DType::F8E4M3Fn,
            Self::Nvfp4 => DType::U8,
            Self::Fp8E4M3Block16 => DType::F8E4M3Fn,
            Self::Fp8E5M2Block16 => DType::F8E5M2,
            Self::Mxfp8E4 => DType::F8E4M3Fn,
            Self::Mxfp8E5 => DType::F8E5M2,
        }
    }

    pub const fn encoding(self) -> Encoding {
        match self {
            Self::Fp16 => Encoding::Unquantized,
            Self::Fp8E4M3Fn | Self::Fp8E4M3FnStatic => Encoding::Fp8Scaled {
                granularity: Fp8ScaleGranularity::OuterDimension,
                scale_dtype: DType::F32,
                resident: Fp8ResidentRepresentation::PackedBytes,
            },
            Self::Nvfp4 => Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F8E4M3Fn,
            },
            Self::Fp8E4M3Block16 | Self::Fp8E5M2Block16 => Encoding::Fp8Scaled {
                granularity: Fp8ScaleGranularity::KBlock {
                    block_size: KV_FP8_BLOCK_SIZE,
                },
                scale_dtype: DType::U8,
                resident: Fp8ResidentRepresentation::PackedBytes,
            },
            Self::Mxfp8E4 | Self::Mxfp8E5 => Encoding::Fp8Scaled {
                granularity: Fp8ScaleGranularity::KBlock {
                    block_size: KV_MXFP8_BLOCK_SIZE,
                },
                scale_dtype: DType::U8,
                resident: Fp8ResidentRepresentation::PackedBytes,
            },
        }
    }

    pub const fn is_kv_fp8_block16(self) -> bool {
        matches!(self, Self::Fp8E4M3Block16 | Self::Fp8E5M2Block16)
    }

    pub const fn is_kv_mxfp8(self) -> bool {
        matches!(self, Self::Mxfp8E4 | Self::Mxfp8E5)
    }
}

/// Number of consecutive head-dimension values sharing one E8M0 scale.
pub const KV_FP8_BLOCK_SIZE: usize = 16;
/// Standard OCP MXFP8 block size along the head-dimension axis.
pub const KV_MXFP8_BLOCK_SIZE: usize = 32;

/// Exact resident byte encoding for a logical KV FP8 block16 format.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KvFp8PhysicalVariant {
    OcpE4M3Fn,
    E4M3FnuZ,
    OcpE5M2,
}

/// Scale byte encoding for KV FP8 block16. It is distinct from arithmetic
/// `u8` even though both occupy one resident byte.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KvFp8ScaleEncoding {
    E8M0,
}

impl KvFp8PhysicalVariant {
    pub const fn dtype(self) -> DType {
        match self {
            Self::OcpE4M3Fn => DType::F8E4M3Fn,
            Self::E4M3FnuZ => DType::F8E4M3FnuZ,
            Self::OcpE5M2 => DType::F8E5M2,
        }
    }

    pub const fn identity_tag(self) -> u8 {
        match self {
            Self::OcpE4M3Fn => 1,
            Self::E4M3FnuZ => 2,
            Self::OcpE5M2 => 3,
        }
    }
}

/// Additive versioned descriptor for the two logical KV FP8 block16 formats.
/// E8M0 is represented as raw `u8` because it is a scale encoding rather than
/// an arithmetic scalar dtype.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvFp8Block16Descriptor {
    encoding: KvCacheEncoding,
    physical_variant: KvFp8PhysicalVariant,
    scale_recipe_tag: u8,
}

/// Versioned descriptor for standard OCP MXFP8 KV storage. FNUZ is excluded:
/// a target-native FNUZ byte stream is not standard OCP MXFP8 and cannot be
/// reinterpreted through this descriptor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvMxfp8Descriptor {
    encoding: KvCacheEncoding,
    physical_variant: KvFp8PhysicalVariant,
}

impl KvMxfp8Descriptor {
    pub const FORMAT_VERSION: u8 = 1;

    pub const fn new(
        encoding: KvCacheEncoding,
        physical_variant: KvFp8PhysicalVariant,
    ) -> Result<Self, KvStateError> {
        let compatible = matches!(
            (encoding, physical_variant),
            (KvCacheEncoding::Mxfp8E4, KvFp8PhysicalVariant::OcpE4M3Fn)
                | (KvCacheEncoding::Mxfp8E5, KvFp8PhysicalVariant::OcpE5M2)
        );
        if !compatible {
            return Err(KvStateError::InvalidMxfp8Variant);
        }
        Ok(Self {
            encoding,
            physical_variant,
        })
    }

    pub const fn canonical_for_encoding(encoding: KvCacheEncoding) -> Option<Self> {
        match encoding {
            KvCacheEncoding::Mxfp8E4 => Some(Self {
                encoding,
                physical_variant: KvFp8PhysicalVariant::OcpE4M3Fn,
            }),
            KvCacheEncoding::Mxfp8E5 => Some(Self {
                encoding,
                physical_variant: KvFp8PhysicalVariant::OcpE5M2,
            }),
            _ => None,
        }
    }

    pub const fn format_version(self) -> u8 {
        Self::FORMAT_VERSION
    }

    pub const fn encoding(self) -> KvCacheEncoding {
        self.encoding
    }

    pub const fn physical_variant(self) -> KvFp8PhysicalVariant {
        self.physical_variant
    }

    pub const fn block_size(self) -> usize {
        KV_MXFP8_BLOCK_SIZE
    }

    pub const fn scale_dtype(self) -> DType {
        DType::U8
    }

    pub const fn scale_encoding(self) -> KvFp8ScaleEncoding {
        KvFp8ScaleEncoding::E8M0
    }

    pub const fn blocks_per_head(self, head_dim: usize) -> usize {
        head_dim.div_ceil(KV_MXFP8_BLOCK_SIZE)
    }

    pub const fn padded_head_dim(self, head_dim: usize) -> usize {
        self.blocks_per_head(head_dim) * KV_MXFP8_BLOCK_SIZE
    }
}

impl KvFp8Block16Descriptor {
    /// Version 2 replaces the original minimal-non-overflow scale selection
    /// with the standard MX floor-power E8M0 rule. Public encoding names stay
    /// stable, while state/checkpoint identities must not alias v1 payloads.
    pub const FORMAT_VERSION: u8 = 2;
    /// Stable identity tag for the v2 `floor_log2(amax) - element_power`
    /// E8M0 scale recipe shared with standard OCP MXFP8.
    pub const STANDARD_MX_FLOOR_POWER_V1_SCALE_RECIPE_TAG: u8 = 1;
    /// Human-readable recipe identity paired with the numeric identity tag.
    pub const STANDARD_MX_FLOOR_POWER_V1_SCALE_RECIPE: &'static str = "StandardMxFloorPowerV1";

    pub const fn new(
        encoding: KvCacheEncoding,
        physical_variant: KvFp8PhysicalVariant,
    ) -> Result<Self, KvStateError> {
        let compatible = matches!(
            (encoding, physical_variant),
            (
                KvCacheEncoding::Fp8E4M3Block16,
                KvFp8PhysicalVariant::OcpE4M3Fn | KvFp8PhysicalVariant::E4M3FnuZ
            ) | (
                KvCacheEncoding::Fp8E5M2Block16,
                KvFp8PhysicalVariant::OcpE5M2
            )
        );
        if !compatible {
            return Err(KvStateError::InvalidFp8Block16Variant);
        }
        Ok(Self {
            encoding,
            physical_variant,
            scale_recipe_tag: Self::STANDARD_MX_FLOOR_POWER_V1_SCALE_RECIPE_TAG,
        })
    }

    pub const fn canonical_for_encoding(encoding: KvCacheEncoding) -> Option<Self> {
        match encoding {
            KvCacheEncoding::Fp8E4M3Block16 => Some(Self {
                encoding,
                physical_variant: KvFp8PhysicalVariant::OcpE4M3Fn,
                scale_recipe_tag: Self::STANDARD_MX_FLOOR_POWER_V1_SCALE_RECIPE_TAG,
            }),
            KvCacheEncoding::Fp8E5M2Block16 => Some(Self {
                encoding,
                physical_variant: KvFp8PhysicalVariant::OcpE5M2,
                scale_recipe_tag: Self::STANDARD_MX_FLOOR_POWER_V1_SCALE_RECIPE_TAG,
            }),
            _ => None,
        }
    }

    pub const fn format_version(self) -> u8 {
        Self::FORMAT_VERSION
    }

    pub const fn encoding(self) -> KvCacheEncoding {
        self.encoding
    }

    pub const fn physical_variant(self) -> KvFp8PhysicalVariant {
        self.physical_variant
    }

    pub const fn scale_recipe(self) -> &'static str {
        Self::STANDARD_MX_FLOOR_POWER_V1_SCALE_RECIPE
    }

    pub const fn scale_recipe_identity_tag(self) -> u8 {
        self.scale_recipe_tag
    }

    pub const fn block_size(self) -> usize {
        KV_FP8_BLOCK_SIZE
    }

    pub const fn scale_dtype(self) -> DType {
        DType::U8
    }

    pub const fn scale_encoding(self) -> KvFp8ScaleEncoding {
        KvFp8ScaleEncoding::E8M0
    }

    pub const fn blocks_per_head(self, head_dim: usize) -> usize {
        head_dim.div_ceil(KV_FP8_BLOCK_SIZE)
    }

    pub const fn padded_head_dim(self, head_dim: usize) -> usize {
        self.blocks_per_head(head_dim) * KV_FP8_BLOCK_SIZE
    }
}

/// Physical ownership selected by a quiescent opaque-state fork.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateForkModeV1 {
    /// Immutable VMM pages are mapped into both owners. A later append must
    /// privately copy every shared tail page before it becomes writable.
    SharedReadOnlyPages,
    /// Paged KV physical token blocks are shared through the native pool.
    /// This accounting is intentionally separate from legacy VMM pages:
    /// `shared_blocks` is a count of 128-token physical blocks and must never
    /// be derived from the legacy `page_bytes` field.
    SharedPagedBlocks,
    /// The destination owns an exact device-side byte copy. This is used for
    /// contiguous-resident providers and mutable linear/GDN state.
    DeviceCopy,
}

/// Redacted accounting returned by a backend after a state fork. It contains
/// no pointer, allocation handle, token ID, or state payload.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StateForkAuditV1 {
    mode: StateForkModeV1,
    published_length: u64,
    shared_pages: u64,
    shared_blocks: u64,
    shared_bytes: u64,
    copied_bytes: u64,
    destination_owned_bytes: u64,
}

impl StateForkAuditV1 {
    pub fn new(
        mode: StateForkModeV1,
        published_length: u64,
        shared_pages: u64,
        copied_bytes: u64,
        destination_owned_bytes: u64,
    ) -> Result<Self, KvStateError> {
        Self::new_with_shared_blocks(
            mode,
            published_length,
            shared_pages,
            0,
            0,
            copied_bytes,
            destination_owned_bytes,
        )
    }

    /// Constructs an audit with explicit paged-block accounting. Paged state
    /// owns token blocks from a shared pool, so its resident bytes are carried
    /// independently from the legacy VMM page count.
    pub fn new_paged(
        published_length: u64,
        shared_blocks: u64,
        shared_bytes: u64,
        copied_bytes: u64,
        destination_owned_bytes: u64,
    ) -> Result<Self, KvStateError> {
        Self::new_with_shared_blocks(
            StateForkModeV1::SharedPagedBlocks,
            published_length,
            0,
            shared_blocks,
            shared_bytes,
            copied_bytes,
            destination_owned_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_shared_blocks(
        mode: StateForkModeV1,
        published_length: u64,
        shared_pages: u64,
        shared_blocks: u64,
        shared_bytes: u64,
        copied_bytes: u64,
        destination_owned_bytes: u64,
    ) -> Result<Self, KvStateError> {
        if published_length == 0
            || (mode == StateForkModeV1::SharedReadOnlyPages
                && (copied_bytes != 0
                    || shared_pages == 0
                    || shared_blocks != 0
                    || shared_bytes != 0))
            || (mode == StateForkModeV1::SharedPagedBlocks
                && (shared_pages != 0
                    || (shared_blocks == 0) != (shared_bytes == 0)
                    || (shared_blocks == 0 && destination_owned_bytes == 0)))
            || (mode == StateForkModeV1::DeviceCopy
                && (shared_pages != 0 || shared_blocks != 0 || shared_bytes != 0))
        {
            return Err(KvStateError::InvalidForkAudit);
        }
        Ok(Self {
            mode,
            published_length,
            shared_pages,
            shared_blocks,
            shared_bytes,
            copied_bytes,
            destination_owned_bytes,
        })
    }

    pub const fn mode(self) -> StateForkModeV1 {
        self.mode
    }

    pub const fn published_length(self) -> u64 {
        self.published_length
    }

    pub const fn shared_pages(self) -> u64 {
        self.shared_pages
    }

    pub const fn shared_blocks(self) -> u64 {
        self.shared_blocks
    }

    /// Bytes occupied by the shared paged blocks in the native pool. Legacy
    /// audits leave this at zero and use VMM physical metadata instead.
    pub const fn shared_bytes(self) -> u64 {
        self.shared_bytes
    }

    pub const fn copied_bytes(self) -> u64 {
        self.copied_bytes
    }

    pub const fn destination_owned_bytes(self) -> u64 {
        self.destination_owned_bytes
    }
}

/// Backend-neutral KV geometry. Physical dtype and quantization encoding live
/// on [`KvStateDescriptor`], so geometry-only callers cannot accidentally
/// infer FP16 storage for a low-bit state.
///
/// Each of K and V has a separate contiguous-address token-major value plane
/// with logical shape `[capacity, heads, head_dim]`. A descriptor selects FP16,
/// FP8 plus an outer scale plane, or packed NVFP4 plus block/outer scale planes.
/// Physical ownership may use virtual-contiguous VMM pages or resident
/// allocations. Query-head repetition is performed by attention, not stored.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvStateLayout {
    heads: usize,
    head_dim: usize,
}

impl Default for KvStateLayout {
    fn default() -> Self {
        Self {
            heads: Self::HEADS,
            head_dim: Self::HEAD_DIM,
        }
    }
}

impl KvStateLayout {
    pub const HEADS: usize = 4;
    pub const HEAD_DIM: usize = 256;

    pub fn new(heads: usize, head_dim: usize) -> Result<Self, KvStateError> {
        if heads == 0 || head_dim == 0 {
            return Err(KvStateError::InvalidLayout);
        }
        Ok(Self { heads, head_dim })
    }

    pub const fn heads(self) -> usize {
        self.heads
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn storage_shape(self, capacity: u64) -> [u64; 3] {
        [capacity, self.heads as u64, self.head_dim as u64]
    }
}

/// The fixed C3b causal GQA request shape and immutable snapshot contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CausalAttentionDescriptor {
    start_position: u64,
    query_count: u64,
    expected_kv_length: u64,
    sliding_window: Option<NonZeroU64>,
    score_scale_bits: Option<u32>,
}

impl CausalAttentionDescriptor {
    pub fn new(
        start_position: u64,
        query_count: u64,
        expected_kv_length: u64,
    ) -> Result<Self, KvStateError> {
        if query_count == 0 {
            return Err(KvStateError::ZeroQueryCount);
        }
        let end = start_position
            .checked_add(query_count)
            .ok_or(KvStateError::LengthOverflow)?;
        if end != expected_kv_length {
            return Err(KvStateError::LengthMismatch {
                expected: end,
                actual: expected_kv_length,
            });
        }
        Ok(Self {
            start_position,
            query_count,
            expected_kv_length,
            sliding_window: None,
            score_scale_bits: None,
        })
    }

    /// Constructs an explicitly windowed causal-attention request. Each query
    /// at logical position `p` may read exactly
    /// `p.saturating_add(1).saturating_sub(sliding_window)..=p`.
    pub fn new_sliding(
        start_position: u64,
        query_count: u64,
        expected_kv_length: u64,
        sliding_window: u64,
    ) -> Result<Self, KvStateError> {
        if sliding_window != KV_STATIC_FP8_SLIDING_WINDOW {
            return Err(KvStateError::InvalidLayout);
        }
        let mut descriptor = Self::new(start_position, query_count, expected_kv_length)?;
        descriptor.sliding_window =
            Some(NonZeroU64::new(sliding_window).ok_or(KvStateError::ZeroSlidingWindow)?);
        Ok(descriptor)
    }

    pub fn new_scaled(
        start_position: u64,
        query_count: u64,
        expected_kv_length: u64,
        score_scale: f32,
    ) -> Result<Self, KvStateError> {
        let mut descriptor = Self::new(start_position, query_count, expected_kv_length)?;
        descriptor.set_score_scale(score_scale)?;
        Ok(descriptor)
    }

    pub fn new_sliding_scaled(
        start_position: u64,
        query_count: u64,
        expected_kv_length: u64,
        sliding_window: u64,
        score_scale: f32,
    ) -> Result<Self, KvStateError> {
        let mut descriptor = Self::new_sliding(
            start_position,
            query_count,
            expected_kv_length,
            sliding_window,
        )?;
        descriptor.set_score_scale(score_scale)?;
        Ok(descriptor)
    }

    fn set_score_scale(&mut self, score_scale: f32) -> Result<(), KvStateError> {
        if !score_scale.is_finite() || score_scale <= 0.0 {
            return Err(KvStateError::InvalidScoreScale);
        }
        self.score_scale_bits = Some(score_scale.to_bits());
        Ok(())
    }

    pub const fn start_position(self) -> u64 {
        self.start_position
    }

    pub const fn query_count(self) -> u64 {
        self.query_count
    }

    pub const fn expected_kv_length(self) -> u64 {
        self.expected_kv_length
    }

    pub const fn sliding_window(self) -> Option<u64> {
        match self.sliding_window {
            Some(window) => Some(window.get()),
            None => None,
        }
    }

    pub fn score_scale(self) -> Option<f32> {
        self.score_scale_bits.map(f32::from_bits)
    }
}

/// Errors found while constructing typed KV metadata.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KvStateError {
    ZeroCapacity,
    InvalidLayout,
    ZeroQueryCount,
    ZeroSlidingWindow,
    InvalidScoreScale,
    LengthOverflow,
    LengthMismatch { expected: u64, actual: u64 },
    LengthOutOfBounds { length: u64, capacity: u64 },
    InvalidPhysicalMemory,
    InvalidPagedPhysicalMemory,
    InvalidPagedImageMetadata,
    InvalidPagedImageTopology,
    PagedImageMetadataOverflow,
    InvalidForkAudit,
    InvalidFp8Block16Variant,
    InvalidMxfp8Variant,
}

impl fmt::Display for KvStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => formatter.write_str("KV state capacity must be non-zero"),
            Self::InvalidLayout => {
                formatter.write_str("KV state layout dimensions must be non-zero")
            }
            Self::ZeroQueryCount => formatter.write_str("attention query count must be non-zero"),
            Self::ZeroSlidingWindow => {
                formatter.write_str("sliding-attention window must be non-zero")
            }
            Self::InvalidScoreScale => {
                formatter.write_str("attention score scale must be finite and positive")
            }
            Self::LengthOverflow => formatter.write_str("attention length overflowed u64"),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "attention expected KV length {expected}, got {actual}"
                )
            }
            Self::LengthOutOfBounds { length, capacity } => {
                write!(
                    formatter,
                    "KV state length {length} exceeds capacity {capacity}"
                )
            }
            Self::InvalidPhysicalMemory => formatter.write_str(
                "KV physical-memory metadata must be page-aligned and within logical capacity",
            ),
            Self::InvalidPagedPhysicalMemory => formatter.write_str(
                "paged KV physical metadata has invalid block, table, pool, or plane accounting",
            ),
            Self::InvalidPagedImageMetadata => {
                formatter.write_str("paged KV image metadata is inconsistent with its descriptor")
            }
            Self::InvalidPagedImageTopology => {
                formatter.write_str("paged KV image topology has invalid IDs, duplicates, or tags")
            }
            Self::PagedImageMetadataOverflow => {
                formatter.write_str("paged KV image metadata arithmetic overflowed")
            }
            Self::InvalidForkAudit => formatter.write_str("invalid opaque-state fork audit"),
            Self::InvalidFp8Block16Variant => formatter.write_str(
                "KV FP8 block16 physical variant is incompatible with its logical encoding",
            ),
            Self::InvalidMxfp8Variant => formatter
                .write_str("KV MXFP8 physical variant is incompatible with standard OCP MXFP8"),
        }
    }
}

impl std::error::Error for KvStateError {}

/// Identity and fixed layout metadata for one request-local layer state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvStateDescriptor {
    layer_id: u32,
    capacity: NonZeroU64,
    layout: KvStateLayout,
    cache_encoding: KvCacheEncoding,
    static_key_scale_bits: u32,
    static_value_scale_bits: u32,
    kv_fp8_block16: Option<KvFp8Block16Descriptor>,
    kv_mxfp8: Option<KvMxfp8Descriptor>,
    sliding_window: Option<NonZeroU64>,
}

impl KvStateDescriptor {
    pub fn new(layer_id: u32, capacity: u64) -> Result<Self, KvStateError> {
        let capacity = NonZeroU64::new(capacity).ok_or(KvStateError::ZeroCapacity)?;
        Ok(Self {
            layer_id,
            capacity,
            layout: KvStateLayout::default(),
            cache_encoding: KvCacheEncoding::Fp16,
            static_key_scale_bits: 0,
            static_value_scale_bits: 0,
            kv_fp8_block16: None,
            kv_mxfp8: None,
            sliding_window: None,
        })
    }

    pub fn new_with_layout(
        layer_id: u32,
        capacity: u64,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self, KvStateError> {
        let capacity = NonZeroU64::new(capacity).ok_or(KvStateError::ZeroCapacity)?;
        Ok(Self {
            layer_id,
            capacity,
            layout: KvStateLayout::new(heads, head_dim)?,
            cache_encoding: KvCacheEncoding::Fp16,
            static_key_scale_bits: 0,
            static_value_scale_bits: 0,
            kv_fp8_block16: None,
            kv_mxfp8: None,
            sliding_window: None,
        })
    }

    pub fn new_with_storage(
        layer_id: u32,
        capacity: u64,
        heads: usize,
        head_dim: usize,
        cache_encoding: KvCacheEncoding,
    ) -> Result<Self, KvStateError> {
        let capacity = NonZeroU64::new(capacity).ok_or(KvStateError::ZeroCapacity)?;
        Ok(Self {
            layer_id,
            capacity,
            layout: KvStateLayout::new(heads, head_dim)?,
            cache_encoding,
            static_key_scale_bits: 0,
            static_value_scale_bits: 0,
            kv_fp8_block16: KvFp8Block16Descriptor::canonical_for_encoding(cache_encoding),
            kv_mxfp8: KvMxfp8Descriptor::canonical_for_encoding(cache_encoding),
            sliding_window: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_kv_fp8_block16(
        layer_id: u32,
        capacity: u64,
        heads: usize,
        head_dim: usize,
        cache_encoding: KvCacheEncoding,
        physical_variant: KvFp8PhysicalVariant,
    ) -> Result<Self, KvStateError> {
        let mut descriptor =
            Self::new_with_storage(layer_id, capacity, heads, head_dim, cache_encoding)?;
        descriptor.kv_fp8_block16 = Some(KvFp8Block16Descriptor::new(
            cache_encoding,
            physical_variant,
        )?);
        Ok(descriptor)
    }

    /// Adds a physical retained-window contract without changing the logical
    /// capacity or length domain. The current native sliding provider is
    /// intentionally limited to unit-scale static E4M3 and fails closed for
    /// every other encoding.
    pub fn with_sliding_window(mut self, sliding_window: u64) -> Result<Self, KvStateError> {
        let window = NonZeroU64::new(sliding_window).ok_or(KvStateError::ZeroSlidingWindow)?;
        if sliding_window != KV_STATIC_FP8_SLIDING_WINDOW
            || self.capacity() > KV_STATIC_FP8_SLIDING_MAX_CAPACITY
            || sliding_window > self.capacity()
            || self.cache_encoding != KvCacheEncoding::Fp8E4M3FnStatic
            || self.static_fp8_scales() != Some((1.0, 1.0))
        {
            return Err(KvStateError::InvalidLayout);
        }
        self.sliding_window = Some(window);
        Ok(self)
    }

    pub fn new_with_static_fp8_sliding(
        layer_id: u32,
        capacity: u64,
        heads: usize,
        head_dim: usize,
        sliding_window: u64,
    ) -> Result<Self, KvStateError> {
        Self::new_with_static_fp8(layer_id, capacity, heads, head_dim, 1.0, 1.0)?
            .with_sliding_window(sliding_window)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_kv_mxfp8(
        layer_id: u32,
        capacity: u64,
        heads: usize,
        head_dim: usize,
        cache_encoding: KvCacheEncoding,
        physical_variant: KvFp8PhysicalVariant,
    ) -> Result<Self, KvStateError> {
        let mut descriptor =
            Self::new_with_storage(layer_id, capacity, heads, head_dim, cache_encoding)?;
        descriptor.kv_mxfp8 = Some(KvMxfp8Descriptor::new(cache_encoding, physical_variant)?);
        Ok(descriptor)
    }

    pub fn new_with_static_fp8(
        layer_id: u32,
        capacity: u64,
        heads: usize,
        head_dim: usize,
        key_decode_scale: f32,
        value_decode_scale: f32,
    ) -> Result<Self, KvStateError> {
        if !key_decode_scale.is_finite()
            || key_decode_scale <= 0.0
            || !value_decode_scale.is_finite()
            || value_decode_scale <= 0.0
        {
            return Err(KvStateError::InvalidLayout);
        }
        let mut descriptor = Self::new_with_storage(
            layer_id,
            capacity,
            heads,
            head_dim,
            KvCacheEncoding::Fp8E4M3FnStatic,
        )?;
        descriptor.static_key_scale_bits = key_decode_scale.to_bits();
        descriptor.static_value_scale_bits = value_decode_scale.to_bits();
        Ok(descriptor)
    }

    pub const fn layer_id(self) -> u32 {
        self.layer_id
    }

    pub const fn capacity(self) -> u64 {
        self.capacity.get()
    }

    pub const fn layout(self) -> KvStateLayout {
        self.layout
    }

    pub const fn storage_shape(self) -> [u64; 3] {
        self.layout().storage_shape(self.physical_capacity_tokens())
    }

    pub const fn dtype(self) -> DType {
        match self.kv_fp8_block16 {
            Some(descriptor) => descriptor.physical_variant().dtype(),
            None => match self.kv_mxfp8 {
                Some(descriptor) => descriptor.physical_variant().dtype(),
                None => self.cache_encoding.dtype(),
            },
        }
    }

    pub const fn encoding(self) -> Encoding {
        self.cache_encoding.encoding()
    }

    pub const fn cache_encoding(self) -> KvCacheEncoding {
        self.cache_encoding
    }

    pub const fn kv_fp8_block16_descriptor(self) -> Option<KvFp8Block16Descriptor> {
        self.kv_fp8_block16
    }

    pub const fn kv_mxfp8_descriptor(self) -> Option<KvMxfp8Descriptor> {
        self.kv_mxfp8
    }

    pub const fn sliding_window(self) -> Option<u64> {
        match self.sliding_window {
            Some(window) => Some(window.get()),
            None => None,
        }
    }

    /// Maximum number of token rows physically owned by each K/V plane.
    /// Sliding state keeps one spare row so a canceled saturated append cannot
    /// overwrite the oldest still-published row.
    pub const fn physical_capacity_tokens(self) -> u64 {
        match self.sliding_window {
            Some(window) => {
                let ring_capacity = window.get().saturating_add(1);
                if self.capacity.get() < ring_capacity {
                    self.capacity.get()
                } else {
                    ring_capacity
                }
            }
            None => self.capacity.get(),
        }
    }

    pub fn static_fp8_scales(self) -> Option<(f32, f32)> {
        (self.cache_encoding == KvCacheEncoding::Fp8E4M3FnStatic).then(|| {
            (
                f32::from_bits(self.static_key_scale_bits),
                f32::from_bits(self.static_value_scale_bits),
            )
        })
    }

    /// Resident bytes for K or V, including separately owned dynamic scale
    /// planes. Static FP8 scales are descriptor scalars and own no device
    /// scale plane. The complete state owns two such composites.
    pub fn resident_bytes_per_plane(self) -> Option<u64> {
        let capacity = self.physical_capacity_tokens();
        let heads = u64::try_from(self.layout.heads()).ok()?;
        let head_dim = u64::try_from(self.layout.head_dim()).ok()?;
        let bytes_per_token = match self.cache_encoding {
            KvCacheEncoding::Fp16 => heads.checked_mul(head_dim)?.checked_mul(2)?,
            KvCacheEncoding::Fp8E4M3Fn => heads
                .checked_mul(head_dim)?
                .checked_add(heads.checked_mul(4)?)?,
            KvCacheEncoding::Fp8E4M3FnStatic => heads.checked_mul(head_dim)?,
            KvCacheEncoding::Nvfp4 => heads
                .checked_mul(head_dim.div_ceil(2))?
                .checked_add(heads.checked_mul(head_dim.div_ceil(16))?)?
                .checked_add(heads.checked_mul(4)?)?,
            KvCacheEncoding::Fp8E4M3Block16 | KvCacheEncoding::Fp8E5M2Block16 => heads
                .checked_mul(head_dim.div_ceil(KV_FP8_BLOCK_SIZE as u64))?
                .checked_mul((KV_FP8_BLOCK_SIZE + 1) as u64)?,
            KvCacheEncoding::Mxfp8E4 | KvCacheEncoding::Mxfp8E5 => heads
                .checked_mul(head_dim.div_ceil(KV_MXFP8_BLOCK_SIZE as u64))?
                .checked_mul((KV_MXFP8_BLOCK_SIZE + 1) as u64)?,
        };
        capacity.checked_mul(bytes_per_token)
    }
}

/// Backend-selected physical backing for an opaque contiguous KV plane.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KvMemoryKind {
    VirtualContiguous,
    ContiguousResident,
}

/// Physical layout used by a KV state snapshot.
///
/// Legacy VMM counters and paged-pool counters have different meanings. They
/// remain tagged so a paged state cannot be interpreted through the legacy
/// `physical_page_bytes` or `tokens_per_page` fields.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KvPhysicalMemoryMetadata {
    Vmm(KvPhysicalMemorySnapshot),
    Paged(KvPagedPhysicalMemorySnapshot),
}

pub const KV_PAGED_TOKEN_BLOCK_SIZE: u32 = 128;
pub const KV_PAGED_PHYSICAL_LAYOUT_VERSION: u32 = 1;
pub const KV_PAGED_PLANE_COUNT: usize = 6;

/// Backend-reported physical backing for a segmented paged KV pool.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvPagedPhysicalMemorySnapshot {
    token_block_size: u32,
    physical_layout_version: u32,
    logical_table_capacity: u64,
    max_physical_blocks: u64,
    allocated_physical_blocks: u64,
    committed_bytes_per_plane: [u64; KV_PAGED_PLANE_COUNT],
    committed_bytes_total: u64,
}

impl KvPagedPhysicalMemorySnapshot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        capacity_tokens: u64,
        observed_length: u64,
        token_block_size: u32,
        physical_layout_version: u32,
        logical_table_capacity: u64,
        max_physical_blocks: u64,
        allocated_physical_blocks: u64,
        committed_bytes_per_plane: [u64; KV_PAGED_PLANE_COUNT],
        committed_bytes_total: u64,
    ) -> Result<Self, KvStateError> {
        let required_logical_blocks = paged_block_count(capacity_tokens, token_block_size)?;
        let required_allocated_blocks = paged_block_count(observed_length, token_block_size)?;
        let sum = committed_bytes_per_plane
            .iter()
            .try_fold(0_u64, |sum, bytes| sum.checked_add(*bytes))
            .ok_or(KvStateError::InvalidPagedPhysicalMemory)?;
        if capacity_tokens == 0
            || observed_length > capacity_tokens
            || token_block_size != KV_PAGED_TOKEN_BLOCK_SIZE
            || physical_layout_version != KV_PAGED_PHYSICAL_LAYOUT_VERSION
            || logical_table_capacity < required_logical_blocks
            || max_physical_blocks < required_logical_blocks
            || allocated_physical_blocks < required_allocated_blocks
            || allocated_physical_blocks > max_physical_blocks
            || committed_bytes_total != sum
        {
            return Err(KvStateError::InvalidPagedPhysicalMemory);
        }
        Ok(Self {
            token_block_size,
            physical_layout_version,
            logical_table_capacity,
            max_physical_blocks,
            allocated_physical_blocks,
            committed_bytes_per_plane,
            committed_bytes_total,
        })
    }

    pub const fn token_block_size(self) -> u32 {
        self.token_block_size
    }

    pub const fn physical_layout_version(self) -> u32 {
        self.physical_layout_version
    }

    pub const fn logical_table_capacity(self) -> u64 {
        self.logical_table_capacity
    }

    pub const fn max_physical_blocks(self) -> u64 {
        self.max_physical_blocks
    }

    pub const fn allocated_physical_blocks(self) -> u64 {
        self.allocated_physical_blocks
    }

    pub const fn committed_bytes_per_plane(self) -> [u64; KV_PAGED_PLANE_COUNT] {
        self.committed_bytes_per_plane
    }

    pub const fn committed_bytes_total(self) -> u64 {
        self.committed_bytes_total
    }
}

/// Version of the host-only Paged KV image topology contract.
#[allow(dead_code)]
pub const KV_PAGED_IMAGE_METADATA_VERSION: u32 = 1;
/// A Paged physical block always contains 128 logical token rows.
#[allow(dead_code)]
pub const KV_PAGED_IMAGE_TOKEN_BLOCK_SIZE: u32 = KV_PAGED_TOKEN_BLOCK_SIZE;
/// Sliding Paged KV uses nine ring slots so one spare slot can be used while
/// a saturated tail is being prepared.
#[allow(dead_code)]
pub const KV_PAGED_RING_SLOT_COUNT: usize = 9;
/// Sentinel used by native logical and ring tables for an unassigned block.
#[allow(dead_code)]
pub const KV_PAGED_INVALID_BLOCK_ID: u32 = u32::MAX;
/// Sentinel used by native ring tags for an unassigned slot.
#[allow(dead_code)]
pub const KV_PAGED_INVALID_TAG: u64 = u64::MAX;

/// Compact absolute-block topology for a sliding Paged KV image.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[allow(dead_code)]
pub struct KvPagedRingTableV1 {
    block_ids: [u32; KV_PAGED_RING_SLOT_COUNT],
    absolute_tags: [u64; KV_PAGED_RING_SLOT_COUNT],
}

#[allow(dead_code)]
impl KvPagedRingTableV1 {
    pub const fn new(
        block_ids: [u32; KV_PAGED_RING_SLOT_COUNT],
        absolute_tags: [u64; KV_PAGED_RING_SLOT_COUNT],
    ) -> Self {
        Self {
            block_ids,
            absolute_tags,
        }
    }

    pub const fn block_ids(self) -> [u32; KV_PAGED_RING_SLOT_COUNT] {
        self.block_ids
    }

    pub const fn absolute_tags(self) -> [u64; KV_PAGED_RING_SLOT_COUNT] {
        self.absolute_tags
    }
}

/// Host checkpoint topology for one Paged KV image.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(dead_code)]
pub enum KvPagedImageTopologyV1 {
    /// Logical block index to physical block ID. Unassigned tail entries use
    /// [`KV_PAGED_INVALID_BLOCK_ID`].
    LogicalTable(Vec<u32>),
    /// Physical block IDs indexed by nine ring slots, paired with absolute
    /// logical block tags. Unassigned slots use both sentinels.
    SlidingRing(KvPagedRingTableV1),
}

/// Versioned, host-only Paged KV image metadata.
///
/// This type describes checkpoint topology and arithmetic only. It contains
/// no device pointer, allocation handle, or serialized payload. Plane strides
/// are the six byte sizes of one physical 128-token block in the order
/// key/value/key-scale/value-scale/key-outer-scale/value-outer-scale.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(dead_code)]
pub struct KvPagedImageMetadataV1 {
    descriptor: KvStateDescriptor,
    observed_length: u64,
    generation: u64,
    retained_start: u64,
    sliding_window: u64,
    physical_block_capacity: u64,
    plane_strides: [u64; KV_PAGED_PLANE_COUNT],
    topology: KvPagedImageTopologyV1,
}

#[allow(dead_code)]
impl KvPagedImageMetadataV1 {
    pub const FORMAT_VERSION: u32 = KV_PAGED_IMAGE_METADATA_VERSION;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        descriptor: KvStateDescriptor,
        observed_length: u64,
        generation: u64,
        retained_start: u64,
        sliding_window: u64,
        physical_block_capacity: u64,
        plane_strides: [u64; KV_PAGED_PLANE_COUNT],
        topology: KvPagedImageTopologyV1,
    ) -> Result<Self, KvStateError> {
        validate_paged_image_metadata(
            descriptor,
            observed_length,
            retained_start,
            sliding_window,
            physical_block_capacity,
            plane_strides,
            &topology,
        )?;
        Ok(Self {
            descriptor,
            observed_length,
            generation,
            retained_start,
            sliding_window,
            physical_block_capacity,
            plane_strides,
            topology,
        })
    }

    pub const fn format_version(&self) -> u32 {
        Self::FORMAT_VERSION
    }

    pub const fn descriptor(&self) -> KvStateDescriptor {
        self.descriptor
    }

    pub const fn observed_length(&self) -> u64 {
        self.observed_length
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn retained_start(&self) -> u64 {
        self.retained_start
    }

    pub const fn sliding_window(&self) -> u64 {
        self.sliding_window
    }

    pub const fn physical_block_capacity(&self) -> u64 {
        self.physical_block_capacity
    }

    pub const fn plane_strides(&self) -> [u64; KV_PAGED_PLANE_COUNT] {
        self.plane_strides
    }

    pub const fn topology(&self) -> &KvPagedImageTopologyV1 {
        &self.topology
    }
}

#[allow(dead_code)]
fn checked_paged_image_mul(lhs: u64, rhs: u64) -> Result<u64, KvStateError> {
    lhs.checked_mul(rhs)
        .ok_or(KvStateError::PagedImageMetadataOverflow)
}

#[allow(dead_code)]
fn expected_paged_image_plane_strides(
    descriptor: KvStateDescriptor,
) -> Result<[u64; KV_PAGED_PLANE_COUNT], KvStateError> {
    let heads = u64::try_from(descriptor.layout().heads())
        .map_err(|_| KvStateError::PagedImageMetadataOverflow)?;
    let head_dim = u64::try_from(descriptor.layout().head_dim())
        .map_err(|_| KvStateError::PagedImageMetadataOverflow)?;
    let block_tokens = u64::from(KV_PAGED_IMAGE_TOKEN_BLOCK_SIZE);
    let (value_bytes, scale_bytes, outer_scale_bytes) = match descriptor.cache_encoding() {
        KvCacheEncoding::Fp16 => (
            checked_paged_image_mul(heads, head_dim)?
                .checked_mul(2)
                .ok_or(KvStateError::PagedImageMetadataOverflow)?,
            0,
            0,
        ),
        KvCacheEncoding::Fp8E4M3Fn => (
            checked_paged_image_mul(heads, head_dim)?,
            checked_paged_image_mul(heads, 4)?,
            0,
        ),
        KvCacheEncoding::Fp8E4M3FnStatic => (checked_paged_image_mul(heads, head_dim)?, 0, 0),
        KvCacheEncoding::Nvfp4 => (
            checked_paged_image_mul(heads, head_dim.div_ceil(2))?,
            checked_paged_image_mul(heads, head_dim.div_ceil(16))?,
            checked_paged_image_mul(heads, 4)?,
        ),
        KvCacheEncoding::Fp8E4M3Block16 | KvCacheEncoding::Fp8E5M2Block16 => {
            let blocks = head_dim.div_ceil(KV_FP8_BLOCK_SIZE as u64);
            (
                checked_paged_image_mul(heads, checked_paged_image_mul(blocks, 16)?)?,
                checked_paged_image_mul(heads, blocks)?,
                0,
            )
        }
        KvCacheEncoding::Mxfp8E4 | KvCacheEncoding::Mxfp8E5 => {
            let blocks = head_dim.div_ceil(KV_MXFP8_BLOCK_SIZE as u64);
            (
                checked_paged_image_mul(heads, checked_paged_image_mul(blocks, 32)?)?,
                checked_paged_image_mul(heads, blocks)?,
                0,
            )
        }
    };
    Ok([
        checked_paged_image_mul(value_bytes, block_tokens)?,
        checked_paged_image_mul(value_bytes, block_tokens)?,
        checked_paged_image_mul(scale_bytes, block_tokens)?,
        checked_paged_image_mul(scale_bytes, block_tokens)?,
        checked_paged_image_mul(outer_scale_bytes, block_tokens)?,
        checked_paged_image_mul(outer_scale_bytes, block_tokens)?,
    ])
}

#[allow(dead_code)]
fn validate_paged_image_metadata(
    descriptor: KvStateDescriptor,
    observed_length: u64,
    retained_start: u64,
    sliding_window: u64,
    physical_block_capacity: u64,
    plane_strides: [u64; KV_PAGED_PLANE_COUNT],
    topology: &KvPagedImageTopologyV1,
) -> Result<(), KvStateError> {
    if observed_length > descriptor.capacity() || retained_start > observed_length {
        return Err(KvStateError::InvalidPagedImageMetadata);
    }
    let descriptor_window = descriptor.sliding_window().unwrap_or(0);
    if sliding_window != descriptor_window {
        return Err(KvStateError::InvalidPagedImageMetadata);
    }
    let expected_retained_start = if sliding_window == 0 {
        0
    } else {
        observed_length.saturating_sub(sliding_window)
    };
    if retained_start != expected_retained_start
        || physical_block_capacity == 0
        || physical_block_capacity >= u64::from(u32::MAX)
        || plane_strides != expected_paged_image_plane_strides(descriptor)?
    {
        return Err(KvStateError::InvalidPagedImageMetadata);
    }
    let required_blocks =
        paged_block_count(descriptor.capacity(), KV_PAGED_IMAGE_TOKEN_BLOCK_SIZE)?;
    let observed_blocks = paged_block_count(observed_length, KV_PAGED_IMAGE_TOKEN_BLOCK_SIZE)?;
    match (descriptor_window == 0, topology) {
        (true, KvPagedImageTopologyV1::LogicalTable(table)) => {
            if table.len() as u64 != required_blocks {
                return Err(KvStateError::InvalidPagedImageTopology);
            }
            validate_paged_logical_table(table, observed_blocks, physical_block_capacity)
        }
        (false, KvPagedImageTopologyV1::SlidingRing(ring)) => validate_paged_ring_table(
            ring,
            retained_start,
            observed_blocks,
            physical_block_capacity,
        ),
        _ => Err(KvStateError::InvalidPagedImageTopology),
    }
}

#[allow(dead_code)]
fn validate_paged_logical_table(
    table: &[u32],
    observed_blocks: u64,
    physical_block_capacity: u64,
) -> Result<(), KvStateError> {
    let observed_blocks =
        usize::try_from(observed_blocks).map_err(|_| KvStateError::PagedImageMetadataOverflow)?;
    let mut seen = Vec::with_capacity(table.len());
    for (index, &block) in table.iter().enumerate() {
        if index < observed_blocks && block == KV_PAGED_INVALID_BLOCK_ID {
            return Err(KvStateError::InvalidPagedImageTopology);
        }
        if block != KV_PAGED_INVALID_BLOCK_ID {
            if u64::from(block) >= physical_block_capacity || seen.contains(&block) {
                return Err(KvStateError::InvalidPagedImageTopology);
            }
            seen.push(block);
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn validate_paged_ring_table(
    ring: &KvPagedRingTableV1,
    retained_start: u64,
    observed_blocks: u64,
    physical_block_capacity: u64,
) -> Result<(), KvStateError> {
    let first_block = retained_start / u64::from(KV_PAGED_IMAGE_TOKEN_BLOCK_SIZE);
    let mut expected_tags = Vec::new();
    let mut tag = first_block;
    while tag < observed_blocks {
        expected_tags.push(tag);
        tag = tag
            .checked_add(1)
            .ok_or(KvStateError::PagedImageMetadataOverflow)?;
    }
    let mut seen_blocks = Vec::new();
    let mut seen_tags = Vec::new();
    for (&block, &absolute_tag) in ring.block_ids.iter().zip(ring.absolute_tags.iter()) {
        let block_invalid = block == KV_PAGED_INVALID_BLOCK_ID;
        let tag_invalid = absolute_tag == KV_PAGED_INVALID_TAG;
        if block_invalid != tag_invalid {
            return Err(KvStateError::InvalidPagedImageTopology);
        }
        if !block_invalid {
            if u64::from(block) >= physical_block_capacity
                || !expected_tags.contains(&absolute_tag)
                || seen_blocks.contains(&block)
                || seen_tags.contains(&absolute_tag)
            {
                return Err(KvStateError::InvalidPagedImageTopology);
            }
            seen_blocks.push(block);
            seen_tags.push(absolute_tag);
        }
    }
    if seen_tags.len() != expected_tags.len()
        || expected_tags.iter().any(|tag| !seen_tags.contains(tag))
    {
        return Err(KvStateError::InvalidPagedImageTopology);
    }
    Ok(())
}

fn paged_block_count(tokens: u64, token_block_size: u32) -> Result<u64, KvStateError> {
    if token_block_size == 0 {
        return Err(KvStateError::InvalidPagedPhysicalMemory);
    }
    tokens
        .checked_add(u64::from(token_block_size) - 1)
        .map(|value| value / u64::from(token_block_size))
        .ok_or(KvStateError::InvalidPagedPhysicalMemory)
}

/// Backend-reported physical backing for a KV plane.
///
/// This is evidence metadata only: allocation and mapping remain owned by the
/// backend. `committed_bytes_per_plane` describes K or V, not their sum.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvPhysicalMemorySnapshot {
    memory_kind: KvMemoryKind,
    physical_page_bytes: u64,
    tokens_per_page: u64,
    mapped_token_capacity: u64,
    committed_bytes_per_plane: u64,
    retained_start: u64,
    retained_length: u64,
}

impl KvPhysicalMemorySnapshot {
    pub fn new(
        logical_capacity: u64,
        observed_length: u64,
        physical_page_bytes: u64,
        tokens_per_page: u64,
        mapped_token_capacity: u64,
        committed_bytes_per_plane: u64,
    ) -> Result<Self, KvStateError> {
        Self::new_with_kind(
            KvMemoryKind::VirtualContiguous,
            logical_capacity,
            observed_length,
            physical_page_bytes,
            tokens_per_page,
            mapped_token_capacity,
            committed_bytes_per_plane,
        )
    }

    pub fn new_with_kind(
        memory_kind: KvMemoryKind,
        logical_capacity: u64,
        observed_length: u64,
        physical_page_bytes: u64,
        tokens_per_page: u64,
        mapped_token_capacity: u64,
        committed_bytes_per_plane: u64,
    ) -> Result<Self, KvStateError> {
        if physical_page_bytes == 0
            || tokens_per_page == 0
            || mapped_token_capacity > logical_capacity
            || observed_length > mapped_token_capacity
            || committed_bytes_per_plane % physical_page_bytes != 0
        {
            return Err(KvStateError::InvalidPhysicalMemory);
        }
        Ok(Self {
            memory_kind,
            physical_page_bytes,
            tokens_per_page,
            mapped_token_capacity,
            committed_bytes_per_plane,
            retained_start: 0,
            retained_length: observed_length,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_retention(
        memory_kind: KvMemoryKind,
        logical_capacity: u64,
        observed_length: u64,
        physical_page_bytes: u64,
        tokens_per_page: u64,
        mapped_token_capacity: u64,
        committed_bytes_per_plane: u64,
        retained_start: u64,
        retained_length: u64,
    ) -> Result<Self, KvStateError> {
        if physical_page_bytes == 0
            || tokens_per_page == 0
            || mapped_token_capacity > logical_capacity
            || retained_length > mapped_token_capacity
            || retained_start > observed_length
            || retained_start.checked_add(retained_length) != Some(observed_length)
            || committed_bytes_per_plane % physical_page_bytes != 0
        {
            return Err(KvStateError::InvalidPhysicalMemory);
        }
        Ok(Self {
            memory_kind,
            physical_page_bytes,
            tokens_per_page,
            mapped_token_capacity,
            committed_bytes_per_plane,
            retained_start,
            retained_length,
        })
    }

    pub const fn memory_kind(self) -> KvMemoryKind {
        self.memory_kind
    }

    pub const fn physical_page_bytes(self) -> u64 {
        self.physical_page_bytes
    }

    pub const fn tokens_per_page(self) -> u64 {
        self.tokens_per_page
    }

    pub const fn mapped_token_capacity(self) -> u64 {
        self.mapped_token_capacity
    }

    pub const fn committed_bytes_per_plane(self) -> u64 {
        self.committed_bytes_per_plane
    }

    pub const fn retained_start(self) -> u64 {
        self.retained_start
    }

    pub const fn retained_length(self) -> u64 {
        self.retained_length
    }
}

/// Backend-reported authoritative state metadata at one observation point.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvStateSnapshot {
    session_id: ExecutionSessionId,
    state_id: KvStateId,
    descriptor: KvStateDescriptor,
    length: u64,
    physical_memory: Option<KvPhysicalMemorySnapshot>,
    physical_metadata: Option<KvPhysicalMemoryMetadata>,
}

impl KvStateSnapshot {
    /// Constructs a snapshot for a backend adapter response.  Core validates
    /// all identity and descriptor fields again before exposing it to callers.
    pub fn new(
        session_id: ExecutionSessionId,
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
        length: u64,
    ) -> Result<Self, KvStateError> {
        if length > descriptor.capacity() {
            return Err(KvStateError::LengthOutOfBounds {
                length,
                capacity: descriptor.capacity(),
            });
        }
        Ok(Self {
            session_id,
            state_id,
            descriptor,
            length,
            physical_memory: None,
            physical_metadata: None,
        })
    }

    /// Constructs a snapshot that includes authoritative physical backing
    /// metadata. Backends without virtual-memory reporting continue to use
    /// [`Self::new`] and expose `None`.
    pub fn new_with_physical_memory(
        session_id: ExecutionSessionId,
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
        length: u64,
        physical_memory: KvPhysicalMemorySnapshot,
    ) -> Result<Self, KvStateError> {
        let physical_covers_length = match descriptor.sliding_window() {
            Some(window) => {
                physical_memory.retained_length() == length.min(window)
                    && physical_memory.retained_start() == length.saturating_sub(window)
            }
            None => length <= physical_memory.mapped_token_capacity(),
        };
        if length > descriptor.capacity() || !physical_covers_length {
            return Err(KvStateError::LengthOutOfBounds {
                length,
                capacity: descriptor.capacity(),
            });
        }
        Ok(Self {
            session_id,
            state_id,
            descriptor,
            length,
            physical_memory: Some(physical_memory),
            physical_metadata: Some(KvPhysicalMemoryMetadata::Vmm(physical_memory)),
        })
    }

    /// Constructs a snapshot with paged-pool accounting. The paged metadata
    /// remains tagged and is not exposed through legacy VMM accessors.
    pub fn new_with_paged_physical_memory(
        session_id: ExecutionSessionId,
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
        length: u64,
        physical_memory: KvPagedPhysicalMemorySnapshot,
    ) -> Result<Self, KvStateError> {
        if length > descriptor.capacity()
            || physical_memory.allocated_physical_blocks()
                < paged_block_count(length, physical_memory.token_block_size())?
        {
            return Err(KvStateError::LengthOutOfBounds {
                length,
                capacity: descriptor.capacity(),
            });
        }
        Ok(Self {
            session_id,
            state_id,
            descriptor,
            length,
            physical_memory: None,
            physical_metadata: Some(KvPhysicalMemoryMetadata::Paged(physical_memory)),
        })
    }

    pub const fn session_id(self) -> ExecutionSessionId {
        self.session_id
    }

    pub const fn state_id(self) -> KvStateId {
        self.state_id
    }

    pub const fn descriptor(self) -> KvStateDescriptor {
        self.descriptor
    }

    pub const fn layer_id(self) -> u32 {
        self.descriptor.layer_id()
    }

    pub const fn capacity(self) -> u64 {
        self.descriptor.capacity()
    }

    pub const fn length(self) -> u64 {
        self.length
    }

    pub const fn layout(self) -> KvStateLayout {
        self.descriptor.layout()
    }

    pub const fn physical_memory(self) -> Option<KvPhysicalMemorySnapshot> {
        self.physical_memory
    }

    pub const fn physical_metadata(self) -> Option<KvPhysicalMemoryMetadata> {
        self.physical_metadata
    }

    pub const fn paged_physical_memory(self) -> Option<KvPagedPhysicalMemorySnapshot> {
        match self.physical_metadata {
            Some(KvPhysicalMemoryMetadata::Paged(metadata)) => Some(metadata),
            _ => None,
        }
    }
}

/// Metadata passed to a backend for one admitted append.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvStateAppendRequest {
    state_id: KvStateId,
    descriptor: KvStateDescriptor,
    token_count: u64,
    expected_length: u64,
    start_position: u64,
}

impl KvStateAppendRequest {
    pub(crate) const fn new(
        state_id: KvStateId,
        descriptor: KvStateDescriptor,
        token_count: u64,
        expected_length: u64,
        start_position: u64,
    ) -> Self {
        Self {
            state_id,
            descriptor,
            token_count,
            expected_length,
            start_position,
        }
    }

    pub const fn state_id(self) -> KvStateId {
        self.state_id
    }

    pub const fn descriptor(self) -> KvStateDescriptor {
        self.descriptor
    }

    pub const fn token_count(self) -> u64 {
        self.token_count
    }

    pub const fn expected_length(self) -> u64 {
        self.expected_length
    }

    pub const fn start_position(self) -> u64 {
        self.start_position
    }

    pub const fn end_position(self) -> u64 {
        self.start_position + self.token_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c3a2_layout_is_typed_fixed_and_does_not_repeat_query_heads() {
        let descriptor = KvStateDescriptor::new(7, 257).expect("non-zero capacity");
        let layout = descriptor.layout();

        assert_eq!(layout.heads(), 4);
        assert_eq!(layout.head_dim(), 256);
        assert_eq!(descriptor.dtype(), DType::F16);
        assert_eq!(descriptor.encoding(), Encoding::Unquantized);
        assert_eq!(descriptor.storage_shape(), [257, 4, 256]);
        assert_eq!(descriptor.layer_id(), 7);
        assert_eq!(descriptor.capacity(), 257);
        assert_ne!(descriptor, KvStateDescriptor::new(8, 257).unwrap());
        assert_ne!(descriptor, KvStateDescriptor::new(7, 256).unwrap());
        assert_eq!(KvStateLayout::HEADS, 4);
    }

    #[test]
    fn descriptor_and_snapshot_reject_zero_or_overflowed_capacity() {
        assert_eq!(
            KvStateDescriptor::new(0, 0),
            Err(KvStateError::ZeroCapacity)
        );
        let descriptor = KvStateDescriptor::new(0, 1).unwrap();
        assert_eq!(
            KvStateSnapshot::new(ExecutionSessionId::new(1), KvStateId::new(2), descriptor, 2,),
            Err(KvStateError::LengthOutOfBounds {
                length: 2,
                capacity: 1,
            })
        );
    }

    #[test]
    fn lowbit_descriptors_include_scale_planes_in_resident_bytes() {
        let fp16 =
            KvStateDescriptor::new_with_storage(0, 257, 4, 256, KvCacheEncoding::Fp16).unwrap();
        let fp8 = KvStateDescriptor::new_with_storage(0, 257, 4, 256, KvCacheEncoding::Fp8E4M3Fn)
            .unwrap();
        let fp8_static = KvStateDescriptor::new_with_static_fp8(0, 257, 4, 256, 0.5, 0.25).unwrap();
        let nvfp4 =
            KvStateDescriptor::new_with_storage(0, 257, 4, 257, KvCacheEncoding::Nvfp4).unwrap();
        assert_eq!(fp16.resident_bytes_per_plane(), Some(257 * 2048));
        assert_eq!(fp8.resident_bytes_per_plane(), Some(257 * 1040));
        assert_eq!(fp8_static.resident_bytes_per_plane(), Some(257 * 1024));
        assert_eq!(nvfp4.resident_bytes_per_plane(), Some(257 * 600));
        assert_eq!(fp8.dtype(), DType::F8E4M3Fn);
        assert_eq!(nvfp4.dtype(), DType::U8);
        assert_ne!(fp16, fp8);
    }

    #[test]
    fn block16_descriptor_accounts_for_tail_padding_and_physical_variant() {
        for head_dim in [15_usize, 16, 17, 255, 256, 257] {
            let descriptor = KvStateDescriptor::new_with_storage(
                0,
                3,
                2,
                head_dim,
                KvCacheEncoding::Fp8E4M3Block16,
            )
            .unwrap();
            let blocks = head_dim.div_ceil(KV_FP8_BLOCK_SIZE) as u64;
            assert_eq!(
                descriptor.resident_bytes_per_plane(),
                Some(3 * 2 * blocks * 17)
            );
            assert_eq!(descriptor.dtype(), DType::F8E4M3Fn);
            let block16 = descriptor.kv_fp8_block16_descriptor().unwrap();
            assert_eq!(block16.scale_dtype(), DType::U8);
            assert_eq!(block16.scale_encoding(), KvFp8ScaleEncoding::E8M0);
            assert_eq!(block16.format_version(), 2);
            assert_eq!(block16.scale_recipe(), "StandardMxFloorPowerV1");
            assert_eq!(block16.scale_recipe_identity_tag(), 1);
        }

        let fnuz = KvStateDescriptor::new_with_kv_fp8_block16(
            0,
            257,
            4,
            256,
            KvCacheEncoding::Fp8E4M3Block16,
            KvFp8PhysicalVariant::E4M3FnuZ,
        )
        .unwrap();
        let e5 =
            KvStateDescriptor::new_with_storage(0, 257, 4, 256, KvCacheEncoding::Fp8E5M2Block16)
                .unwrap();
        assert_eq!(fnuz.dtype(), DType::F8E4M3FnuZ);
        assert_eq!(e5.dtype(), DType::F8E5M2);
        for descriptor in [fnuz, e5] {
            let block16 = descriptor.kv_fp8_block16_descriptor().unwrap();
            assert_eq!(block16.format_version(), 2);
            assert_eq!(
                block16.scale_recipe(),
                KvFp8Block16Descriptor::STANDARD_MX_FLOOR_POWER_V1_SCALE_RECIPE
            );
            assert_eq!(
                block16.scale_recipe_identity_tag(),
                KvFp8Block16Descriptor::STANDARD_MX_FLOOR_POWER_V1_SCALE_RECIPE_TAG
            );
        }
        assert_eq!(fnuz.resident_bytes_per_plane(), Some(257 * 1088));
        assert_eq!(e5.resident_bytes_per_plane(), Some(257 * 1088));
        assert_eq!(
            KvStateDescriptor::new_with_kv_fp8_block16(
                0,
                1,
                1,
                16,
                KvCacheEncoding::Fp8E5M2Block16,
                KvFp8PhysicalVariant::E4M3FnuZ,
            ),
            Err(KvStateError::InvalidFp8Block16Variant)
        );
    }

    #[test]
    fn mxfp8_descriptor_accounts_for_block32_tails_and_excludes_fnuz() {
        for head_dim in [15_usize, 16, 17, 31, 32, 33, 255, 256, 257] {
            for encoding in [KvCacheEncoding::Mxfp8E4, KvCacheEncoding::Mxfp8E5] {
                let descriptor =
                    KvStateDescriptor::new_with_storage(0, 3, 2, head_dim, encoding).unwrap();
                let mx = descriptor.kv_mxfp8_descriptor().unwrap();
                let blocks = head_dim.div_ceil(KV_MXFP8_BLOCK_SIZE) as u64;
                assert_eq!(mx.blocks_per_head(head_dim), blocks as usize);
                assert_eq!(mx.padded_head_dim(head_dim), blocks as usize * 32);
                assert_eq!(mx.block_size(), 32);
                assert_eq!(mx.scale_dtype(), DType::U8);
                assert_eq!(mx.scale_encoding(), KvFp8ScaleEncoding::E8M0);
                assert_eq!(descriptor.kv_fp8_block16_descriptor(), None);
                assert_eq!(
                    descriptor.resident_bytes_per_plane(),
                    Some(3 * 2 * blocks * 33)
                );
            }
        }

        let e4 =
            KvStateDescriptor::new_with_storage(0, 257, 4, 256, KvCacheEncoding::Mxfp8E4).unwrap();
        let e5 =
            KvStateDescriptor::new_with_storage(0, 257, 4, 256, KvCacheEncoding::Mxfp8E5).unwrap();
        assert_eq!(e4.dtype(), DType::F8E4M3Fn);
        assert_eq!(e5.dtype(), DType::F8E5M2);
        assert_eq!(e4.resident_bytes_per_plane(), Some(257 * 1056));
        assert_eq!(e5.resident_bytes_per_plane(), Some(257 * 1056));
        assert_eq!(
            KvMxfp8Descriptor::new(KvCacheEncoding::Mxfp8E4, KvFp8PhysicalVariant::E4M3FnuZ,),
            Err(KvStateError::InvalidMxfp8Variant)
        );
    }

    #[test]
    fn canonical_names_and_existing_meanings_remain_stable() {
        assert_eq!(KvCacheEncoding::Fp16.canonical_name(), "fp16");
        assert_eq!(KvCacheEncoding::Fp8E4M3Fn.canonical_name(), "fp8");
        assert_eq!(
            KvCacheEncoding::Fp8E4M3FnStatic.canonical_name(),
            "fp8-static"
        );
        assert_eq!(KvCacheEncoding::Nvfp4.canonical_name(), "nvfp4");
        assert_eq!(
            KvCacheEncoding::Fp8E4M3Block16.canonical_name(),
            "kv-fp8-e4-block16"
        );
        assert_eq!(
            KvCacheEncoding::Fp8E5M2Block16.canonical_name(),
            "kv-fp8-e5-block16"
        );
        assert_eq!(KvCacheEncoding::Mxfp8E4.canonical_name(), "kv-mxfp8-e4");
        assert_eq!(KvCacheEncoding::Mxfp8E5.canonical_name(), "kv-mxfp8-e5");
    }

    #[test]
    fn causal_attention_descriptor_covers_prefill_and_decode_boundaries() {
        for query_count in [1_u64, 3, 17, 255, 256, 257] {
            let descriptor = CausalAttentionDescriptor::new(0, query_count, query_count)
                .expect("valid prefill range");
            assert_eq!(descriptor.query_count(), query_count);
            assert_eq!(descriptor.expected_kv_length(), query_count);
        }
        for (start_position, query_count) in [(0_u64, 1_u64), (3, 3), (255, 1), (256, 1), (257, 1)]
        {
            let expected = start_position + query_count;
            let descriptor = CausalAttentionDescriptor::new(start_position, query_count, expected)
                .expect("valid decode prefix");
            assert_eq!(descriptor.expected_kv_length(), expected);
        }
    }

    #[test]
    fn static_fp8_sliding_descriptor_and_score_scale_are_fail_closed() {
        let descriptor = KvStateDescriptor::new_with_static_fp8_sliding(
            17,
            KV_STATIC_FP8_SLIDING_MAX_CAPACITY,
            4,
            512,
            KV_STATIC_FP8_SLIDING_WINDOW,
        )
        .unwrap();
        assert_eq!(
            descriptor.cache_encoding(),
            KvCacheEncoding::Fp8E4M3FnStatic
        );
        assert_eq!(descriptor.static_fp8_scales(), Some((1.0, 1.0)));
        assert_eq!(descriptor.sliding_window(), Some(1024));
        assert_eq!(descriptor.capacity(), 262_144);
        assert_eq!(descriptor.resident_bytes_per_plane(), Some(1025 * 4 * 512));
        assert_eq!(descriptor.physical_capacity_tokens(), 1025);
        assert_eq!(descriptor.storage_shape(), [1025, 4, 512]);

        assert_eq!(
            KvStateDescriptor::new_with_static_fp8(0, 2048, 4, 256, 0.5, 1.0)
                .unwrap()
                .with_sliding_window(1024),
            Err(KvStateError::InvalidLayout)
        );
        assert_eq!(
            KvStateDescriptor::new_with_static_fp8_sliding(0, 2048, 4, 256, 1023),
            Err(KvStateError::InvalidLayout)
        );
        assert_eq!(
            KvStateDescriptor::new_with_static_fp8_sliding(0, 262_145, 4, 256, 1024),
            Err(KvStateError::InvalidLayout)
        );

        for (start, count, end) in [(0, 1023, 1023), (0, 1024, 1024), (1024, 1, 1025)] {
            let attention =
                CausalAttentionDescriptor::new_sliding_scaled(start, count, end, 1024, 1.0)
                    .unwrap();
            assert_eq!(attention.sliding_window(), Some(1024));
            assert_eq!(attention.score_scale(), Some(1.0));
        }
        for scale in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(
                CausalAttentionDescriptor::new_scaled(0, 1, 1, scale),
                Err(KvStateError::InvalidScoreScale)
            );
        }
    }

    #[test]
    fn sliding_snapshot_tracks_logical_length_and_only_retained_physical_rows() {
        let descriptor =
            KvStateDescriptor::new_with_static_fp8_sliding(2, 262_144, 4, 256, 1024).unwrap();
        for (length, retained_start, retained_length) in
            [(1023, 0, 1023), (1024, 0, 1024), (1025, 1, 1024)]
        {
            let physical = KvPhysicalMemorySnapshot::new_with_retention(
                KvMemoryKind::VirtualContiguous,
                descriptor.capacity(),
                length,
                4096,
                4,
                1025,
                1_052_672,
                retained_start,
                retained_length,
            )
            .unwrap();
            let snapshot = KvStateSnapshot::new_with_physical_memory(
                ExecutionSessionId::new(7),
                KvStateId::new(11),
                descriptor,
                length,
                physical,
            )
            .unwrap();
            assert_eq!(snapshot.length(), length);
            assert_eq!(
                snapshot.physical_memory().unwrap().retained_start(),
                retained_start
            );
            assert_eq!(
                snapshot.physical_memory().unwrap().retained_length(),
                retained_length
            );
        }
    }

    #[test]
    fn paged_image_metadata_accepts_logical_boundaries_and_sliding_ring_boundaries() {
        let descriptor =
            KvStateDescriptor::new_with_storage(4, 1152, 4, 256, KvCacheEncoding::Mxfp8E4).unwrap();
        let strides = expected_paged_image_plane_strides(descriptor).unwrap();
        for length in [127_u64, 128, 129] {
            let mut table = vec![KV_PAGED_INVALID_BLOCK_ID; 9];
            for block in 0..length.div_ceil(u64::from(KV_PAGED_IMAGE_TOKEN_BLOCK_SIZE)) {
                table[block as usize] = block as u32;
            }
            let metadata = KvPagedImageMetadataV1::new(
                descriptor,
                length,
                7,
                0,
                0,
                16,
                strides,
                KvPagedImageTopologyV1::LogicalTable(table),
            )
            .unwrap();
            assert_eq!(metadata.format_version(), KV_PAGED_IMAGE_METADATA_VERSION);
            assert_eq!(metadata.observed_length(), length);
            assert_eq!(metadata.generation(), 7);
            assert_eq!(metadata.plane_strides(), strides);
        }

        let sliding =
            KvStateDescriptor::new_with_static_fp8_sliding(4, 1152, 4, 256, 1024).unwrap();
        let sliding_strides = expected_paged_image_plane_strides(sliding).unwrap();
        for length in [1023_u64, 1024, 1152] {
            let retained_start = length.saturating_sub(1024);
            let first = retained_start / 128;
            let blocks = length.div_ceil(128);
            let mut ids = [KV_PAGED_INVALID_BLOCK_ID; KV_PAGED_RING_SLOT_COUNT];
            let mut tags = [KV_PAGED_INVALID_TAG; KV_PAGED_RING_SLOT_COUNT];
            for (slot, tag) in (first..blocks).enumerate() {
                ids[slot] = slot as u32;
                tags[slot] = tag;
            }
            let metadata = KvPagedImageMetadataV1::new(
                sliding,
                length,
                9,
                retained_start,
                1024,
                16,
                sliding_strides,
                KvPagedImageTopologyV1::SlidingRing(KvPagedRingTableV1::new(ids, tags)),
            )
            .unwrap();
            assert_eq!(metadata.retained_start(), retained_start);
            assert_eq!(metadata.sliding_window(), 1024);
        }
    }

    #[test]
    fn paged_image_metadata_rejects_invalid_ids_duplicate_or_stale_tags_and_overflow() {
        let descriptor =
            KvStateDescriptor::new_with_storage(4, 129, 4, 256, KvCacheEncoding::Mxfp8E4).unwrap();
        let strides = expected_paged_image_plane_strides(descriptor).unwrap();
        let mut duplicate = vec![KV_PAGED_INVALID_BLOCK_ID; 2];
        duplicate[0] = 1;
        duplicate[1] = 1;
        assert_eq!(
            KvPagedImageMetadataV1::new(
                descriptor,
                129,
                0,
                0,
                0,
                4,
                strides,
                KvPagedImageTopologyV1::LogicalTable(duplicate),
            )
            .unwrap_err(),
            KvStateError::InvalidPagedImageTopology
        );

        let mut invalid_id = vec![KV_PAGED_INVALID_BLOCK_ID; 2];
        invalid_id[0] = KV_PAGED_INVALID_BLOCK_ID;
        assert_eq!(
            KvPagedImageMetadataV1::new(
                descriptor,
                1,
                0,
                0,
                0,
                4,
                strides,
                KvPagedImageTopologyV1::LogicalTable(invalid_id),
            )
            .unwrap_err(),
            KvStateError::InvalidPagedImageTopology
        );

        let sliding =
            KvStateDescriptor::new_with_static_fp8_sliding(4, 1152, 4, 256, 1024).unwrap();
        let sliding_strides = expected_paged_image_plane_strides(sliding).unwrap();
        let mut stale_tags = [KV_PAGED_INVALID_TAG; KV_PAGED_RING_SLOT_COUNT];
        let mut stale_ids = [KV_PAGED_INVALID_BLOCK_ID; KV_PAGED_RING_SLOT_COUNT];
        for slot in 0..8 {
            stale_tags[slot] = slot as u64;
            stale_ids[slot] = slot as u32;
        }
        stale_tags[8] = 99;
        stale_ids[8] = 8;
        assert_eq!(
            KvPagedImageMetadataV1::new(
                sliding,
                1024,
                0,
                0,
                1024,
                16,
                sliding_strides,
                KvPagedImageTopologyV1::SlidingRing(
                    KvPagedRingTableV1::new(stale_ids, stale_tags,)
                ),
            )
            .unwrap_err(),
            KvStateError::InvalidPagedImageTopology
        );

        let overflow = KvStateDescriptor::new_with_layout(4, 1, usize::MAX, usize::MAX).unwrap();
        assert_eq!(
            KvPagedImageMetadataV1::new(
                overflow,
                0,
                0,
                0,
                0,
                1,
                [0; KV_PAGED_PLANE_COUNT],
                KvPagedImageTopologyV1::LogicalTable(vec![KV_PAGED_INVALID_BLOCK_ID]),
            )
            .unwrap_err(),
            KvStateError::PagedImageMetadataOverflow
        );
    }

    #[test]
    fn paged_snapshot_keeps_pool_counts_tagged_and_checks_boundaries() {
        let descriptor =
            KvStateDescriptor::new_with_storage(4, 257, 4, 256, KvCacheEncoding::Mxfp8E4).unwrap();
        for length in [0_u64, 127, 128, 129, 256, 257] {
            let allocated = length.div_ceil(u64::from(KV_PAGED_TOKEN_BLOCK_SIZE));
            let per_plane = [allocated * 33; KV_PAGED_PLANE_COUNT];
            let total = per_plane.iter().sum();
            let physical = KvPagedPhysicalMemorySnapshot::new(
                descriptor.capacity(),
                length,
                KV_PAGED_TOKEN_BLOCK_SIZE,
                KV_PAGED_PHYSICAL_LAYOUT_VERSION,
                3,
                4,
                allocated,
                per_plane,
                total,
            )
            .unwrap();
            let snapshot = KvStateSnapshot::new_with_paged_physical_memory(
                ExecutionSessionId::new(8),
                KvStateId::new(12),
                descriptor,
                length,
                physical,
            )
            .unwrap();
            assert_eq!(snapshot.physical_memory(), None);
            assert_eq!(snapshot.paged_physical_memory(), Some(physical));
            assert_eq!(
                snapshot.physical_metadata(),
                Some(KvPhysicalMemoryMetadata::Paged(physical))
            );
        }
        let invalid_total = KvPagedPhysicalMemorySnapshot::new(
            descriptor.capacity(),
            129,
            KV_PAGED_TOKEN_BLOCK_SIZE,
            KV_PAGED_PHYSICAL_LAYOUT_VERSION,
            3,
            4,
            2,
            [1; KV_PAGED_PLANE_COUNT],
            5,
        );
        assert_eq!(invalid_total, Err(KvStateError::InvalidPagedPhysicalMemory));
        assert_eq!(
            KvPagedPhysicalMemorySnapshot::new(
                descriptor.capacity(),
                129,
                64,
                KV_PAGED_PHYSICAL_LAYOUT_VERSION,
                5,
                5,
                3,
                [0; KV_PAGED_PLANE_COUNT],
                0,
            ),
            Err(KvStateError::InvalidPagedPhysicalMemory)
        );
    }

    #[test]
    fn paged_fork_audit_keeps_shared_blocks_out_of_vmm_page_counts() {
        let audit = StateForkAuditV1::new_paged(129, 2, 2_112, 0, 0).unwrap();
        assert_eq!(audit.mode(), StateForkModeV1::SharedPagedBlocks);
        assert_eq!(audit.shared_pages(), 0);
        assert_eq!(audit.shared_blocks(), 2);
        assert_eq!(audit.shared_bytes(), 2_112);
        let diverged = StateForkAuditV1::new_paged(130, 0, 0, 592, 2_112).unwrap();
        assert_eq!(diverged.shared_blocks(), 0);
        assert_eq!(diverged.shared_pages(), 0);
        assert_eq!(diverged.copied_bytes(), 592);
        assert!(StateForkAuditV1::new_paged(130, 0, 0, 0, 0).is_err());
        assert!(
            StateForkAuditV1::new_with_shared_blocks(
                StateForkModeV1::SharedPagedBlocks,
                129,
                1,
                0,
                1,
                0,
                0,
            )
            .is_err()
        );
        assert!(
            StateForkAuditV1::new_with_shared_blocks(
                StateForkModeV1::SharedReadOnlyPages,
                129,
                1,
                1,
                0,
                0,
                0,
            )
            .is_err()
        );
    }
}
