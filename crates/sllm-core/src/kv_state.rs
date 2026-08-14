//! Typed, backend-neutral request-local full-attention KV state contracts.
//!
//! The state is deliberately not a semantic operation.  Its storage and
//! length are owned by the backend, while these types make the fixed C3a2
//! layout and append admission rules explicit at the core boundary.

use std::fmt;
use std::num::NonZeroU64;

use crate::execution::{ExecutionSessionId, KvStateId};
use crate::{DType, Encoding};

/// The only C3a2 KV storage layout.
///
/// Each of K and V is a separate virtual-contiguous unquantized FP16 buffer
/// with token-major shape `[capacity, 4, 256]`. Query-head repetition is performed by
/// attention, not materialized in this state.
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
    pub const DTYPE: DType = DType::F16;
    pub const ENCODING: Encoding = Encoding::Unquantized;

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

    pub const fn dtype(self) -> DType {
        Self::DTYPE
    }

    pub const fn encoding(self) -> Encoding {
        Self::ENCODING
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
        })
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
}

/// Errors found while constructing typed KV metadata.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KvStateError {
    ZeroCapacity,
    InvalidLayout,
    ZeroQueryCount,
    LengthOverflow,
    LengthMismatch { expected: u64, actual: u64 },
    LengthOutOfBounds { length: u64, capacity: u64 },
    InvalidPhysicalMemory,
}

impl fmt::Display for KvStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => formatter.write_str("KV state capacity must be non-zero"),
            Self::InvalidLayout => {
                formatter.write_str("KV state layout dimensions must be non-zero")
            }
            Self::ZeroQueryCount => formatter.write_str("attention query count must be non-zero"),
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
}

impl KvStateDescriptor {
    pub fn new(layer_id: u32, capacity: u64) -> Result<Self, KvStateError> {
        let capacity = NonZeroU64::new(capacity).ok_or(KvStateError::ZeroCapacity)?;
        Ok(Self {
            layer_id,
            capacity,
            layout: KvStateLayout::default(),
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
        })
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
        self.layout().storage_shape(self.capacity())
    }

    pub const fn dtype(self) -> DType {
        KvStateLayout::DTYPE
    }

    pub const fn encoding(self) -> Encoding {
        KvStateLayout::ENCODING
    }
}

/// Backend-reported physical backing for a virtual-contiguous KV plane.
///
/// This is evidence metadata only: allocation and mapping remain owned by the
/// backend. `committed_bytes_per_plane` describes K or V, not their sum.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvPhysicalMemorySnapshot {
    physical_page_bytes: u64,
    tokens_per_page: u64,
    mapped_token_capacity: u64,
    committed_bytes_per_plane: u64,
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
        if physical_page_bytes == 0
            || tokens_per_page == 0
            || mapped_token_capacity > logical_capacity
            || observed_length > mapped_token_capacity
            || committed_bytes_per_plane % physical_page_bytes != 0
        {
            return Err(KvStateError::InvalidPhysicalMemory);
        }
        Ok(Self {
            physical_page_bytes,
            tokens_per_page,
            mapped_token_capacity,
            committed_bytes_per_plane,
        })
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
        if length > descriptor.capacity() || length > physical_memory.mapped_token_capacity() {
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
        assert_eq!(layout.dtype(), DType::F16);
        assert_eq!(layout.encoding(), Encoding::Unquantized);
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
}
