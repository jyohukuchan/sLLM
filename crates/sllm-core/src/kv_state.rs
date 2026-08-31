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
        if published_length == 0
            || (mode == StateForkModeV1::SharedReadOnlyPages
                && (copied_bytes != 0 || shared_pages == 0))
            || (mode == StateForkModeV1::DeviceCopy && shared_pages != 0)
        {
            return Err(KvStateError::InvalidForkAudit);
        }
        Ok(Self {
            mode,
            published_length,
            shared_pages,
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
}
