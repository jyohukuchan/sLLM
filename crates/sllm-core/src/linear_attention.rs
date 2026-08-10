//! Typed, backend-neutral request-local linear-attention state contracts.
//!
//! Projection and output projection remain ordinary matmul operations. This
//! module fixes the Qwen3.5 C4 state layout and the transactional execution
//! metadata for the convolution/recurrent portion without exposing backend
//! storage.

use std::fmt;
use std::num::NonZeroU64;

use crate::execution::{ExecutionSessionId, LinearAttentionStateId};
use crate::{DType, Encoding};

/// The only Phase 3 Qwen3.5 linear-attention layout.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LinearAttentionLayout;

impl LinearAttentionLayout {
    pub const QK_HEADS: usize = 16;
    pub const VALUE_HEADS: usize = 32;
    pub const HEAD_DIM: usize = 128;
    pub const QKV_WIDTH: usize = 8_192;
    pub const OUTPUT_WIDTH: usize = 4_096;
    pub const CONV_KERNEL_SIZE: usize = 4;
    pub const CONV_HISTORY: usize = Self::CONV_KERNEL_SIZE - 1;
    pub const EPSILON_BITS: u32 = 1.0e-6_f32.to_bits();
    pub const QK_REPEAT_FACTOR: usize = Self::VALUE_HEADS / Self::QK_HEADS;
    pub const ACTIVATION_DTYPE: DType = DType::Bf16;
    pub const CONV_STATE_DTYPE: DType = DType::Bf16;
    pub const RECURRENT_STATE_DTYPE: DType = DType::F32;
    pub const ENCODING: Encoding = Encoding::Unquantized;

    pub const fn conv_state_shape(self) -> [u64; 2] {
        [Self::CONV_HISTORY as u64, Self::QKV_WIDTH as u64]
    }

    pub const fn recurrent_state_shape(self) -> [u64; 3] {
        [
            Self::VALUE_HEADS as u64,
            Self::HEAD_DIM as u64,
            Self::HEAD_DIM as u64,
        ]
    }

    pub const fn qkv_shape(self, token_count: u64) -> [u64; 2] {
        [token_count, Self::QKV_WIDTH as u64]
    }

    pub const fn output_shape(self, token_count: u64) -> [u64; 2] {
        [token_count, Self::OUTPUT_WIDTH as u64]
    }

    pub const fn scalar_head_shape(self, token_count: u64) -> [u64; 2] {
        [token_count, Self::VALUE_HEADS as u64]
    }
}

/// Identity and capacity for one request-local linear-attention layer state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LinearAttentionStateDescriptor {
    layer_id: u32,
    capacity: NonZeroU64,
}

impl LinearAttentionStateDescriptor {
    pub fn new(layer_id: u32, capacity: u64) -> Result<Self, LinearAttentionError> {
        let capacity = NonZeroU64::new(capacity).ok_or(LinearAttentionError::ZeroCapacity)?;
        Ok(Self { layer_id, capacity })
    }

    pub const fn layer_id(self) -> u32 {
        self.layer_id
    }

    pub const fn capacity(self) -> u64 {
        self.capacity.get()
    }

    pub const fn layout(self) -> LinearAttentionLayout {
        LinearAttentionLayout
    }
}

/// One ordered state transition over a contiguous token interval.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LinearAttentionDescriptor {
    start_position: u64,
    token_count: u64,
    expected_length: u64,
}

impl LinearAttentionDescriptor {
    pub fn new(
        start_position: u64,
        token_count: u64,
        expected_length: u64,
    ) -> Result<Self, LinearAttentionError> {
        if token_count == 0 {
            return Err(LinearAttentionError::ZeroTokenCount);
        }
        let end = start_position
            .checked_add(token_count)
            .ok_or(LinearAttentionError::LengthOverflow)?;
        if end != expected_length {
            return Err(LinearAttentionError::LengthMismatch {
                expected: end,
                actual: expected_length,
            });
        }
        Ok(Self {
            start_position,
            token_count,
            expected_length,
        })
    }

    pub const fn start_position(self) -> u64 {
        self.start_position
    }

    pub const fn token_count(self) -> u64 {
        self.token_count
    }

    pub const fn expected_length(self) -> u64 {
        self.expected_length
    }
}

/// Authoritative backend state metadata at one observation point.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LinearAttentionStateSnapshot {
    session_id: ExecutionSessionId,
    state_id: LinearAttentionStateId,
    descriptor: LinearAttentionStateDescriptor,
    length: u64,
}

impl LinearAttentionStateSnapshot {
    pub fn new(
        session_id: ExecutionSessionId,
        state_id: LinearAttentionStateId,
        descriptor: LinearAttentionStateDescriptor,
        length: u64,
    ) -> Result<Self, LinearAttentionError> {
        if length > descriptor.capacity() {
            return Err(LinearAttentionError::LengthOutOfBounds {
                length,
                capacity: descriptor.capacity(),
            });
        }
        Ok(Self {
            session_id,
            state_id,
            descriptor,
            length,
        })
    }

    pub const fn session_id(self) -> ExecutionSessionId {
        self.session_id
    }

    pub const fn state_id(self) -> LinearAttentionStateId {
        self.state_id
    }

    pub const fn descriptor(self) -> LinearAttentionStateDescriptor {
        self.descriptor
    }

    pub const fn length(self) -> u64 {
        self.length
    }
}

/// Metadata passed to a backend for one admitted transactional execution.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LinearAttentionRequest {
    state_id: LinearAttentionStateId,
    state_descriptor: LinearAttentionStateDescriptor,
    descriptor: LinearAttentionDescriptor,
}

impl LinearAttentionRequest {
    pub(crate) const fn new(
        state_id: LinearAttentionStateId,
        state_descriptor: LinearAttentionStateDescriptor,
        descriptor: LinearAttentionDescriptor,
    ) -> Self {
        Self {
            state_id,
            state_descriptor,
            descriptor,
        }
    }

    pub const fn state_id(self) -> LinearAttentionStateId {
        self.state_id
    }

    pub const fn state_descriptor(self) -> LinearAttentionStateDescriptor {
        self.state_descriptor
    }

    pub const fn descriptor(self) -> LinearAttentionDescriptor {
        self.descriptor
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LinearAttentionError {
    ZeroCapacity,
    ZeroTokenCount,
    LengthOverflow,
    LengthMismatch { expected: u64, actual: u64 },
    LengthOutOfBounds { length: u64, capacity: u64 },
}

impl fmt::Display for LinearAttentionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => formatter.write_str("linear-attention capacity must be non-zero"),
            Self::ZeroTokenCount => {
                formatter.write_str("linear-attention token count must be non-zero")
            }
            Self::LengthOverflow => formatter.write_str("linear-attention length overflowed u64"),
            Self::LengthMismatch { expected, actual } => write!(
                formatter,
                "linear-attention expected committed length {expected}, got {actual}"
            ),
            Self::LengthOutOfBounds { length, capacity } => write!(
                formatter,
                "linear-attention state length {length} exceeds capacity {capacity}"
            ),
        }
    }
}

impl std::error::Error for LinearAttentionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_layout_matches_qwen35_contract() {
        let layout = LinearAttentionLayout;
        assert_eq!(layout.conv_state_shape(), [3, 8_192]);
        assert_eq!(layout.recurrent_state_shape(), [32, 128, 128]);
        assert_eq!(layout.qkv_shape(257), [257, 8_192]);
        assert_eq!(layout.output_shape(257), [257, 4_096]);
        assert_eq!(layout.scalar_head_shape(257), [257, 32]);
        assert_eq!(LinearAttentionLayout::QK_REPEAT_FACTOR, 2);
        assert_eq!(LinearAttentionLayout::EPSILON_BITS, 1.0e-6_f32.to_bits());
    }

    #[test]
    fn descriptor_accepts_boundaries_and_rejects_invalid_lengths() {
        for token_count in [1_u64, 3, 17, 255, 256, 257, 511, 512, 513] {
            let descriptor = LinearAttentionDescriptor::new(7, token_count, 7 + token_count)
                .expect("valid ordered transition");
            assert_eq!(descriptor.token_count(), token_count);
        }
        assert_eq!(
            LinearAttentionDescriptor::new(0, 0, 0),
            Err(LinearAttentionError::ZeroTokenCount)
        );
        assert_eq!(
            LinearAttentionDescriptor::new(u64::MAX, 1, 0),
            Err(LinearAttentionError::LengthOverflow)
        );
        assert_eq!(
            LinearAttentionDescriptor::new(2, 3, 4),
            Err(LinearAttentionError::LengthMismatch {
                expected: 5,
                actual: 4,
            })
        );
    }

    #[test]
    fn state_descriptor_and_snapshot_are_bounded() {
        assert_eq!(
            LinearAttentionStateDescriptor::new(0, 0),
            Err(LinearAttentionError::ZeroCapacity)
        );
        let descriptor = LinearAttentionStateDescriptor::new(9, 257).unwrap();
        let session_id = ExecutionSessionId::new(11);
        let state_id = LinearAttentionStateId::new(13);
        let snapshot = LinearAttentionStateSnapshot::new(session_id, state_id, descriptor, 257)
            .expect("capacity boundary is valid");
        assert_eq!(snapshot.session_id(), session_id);
        assert_eq!(snapshot.state_id(), state_id);
        assert_eq!(snapshot.descriptor(), descriptor);
        assert_eq!(snapshot.length(), 257);
        assert_eq!(
            LinearAttentionStateSnapshot::new(session_id, state_id, descriptor, 258),
            Err(LinearAttentionError::LengthOutOfBounds {
                length: 258,
                capacity: 257,
            })
        );
    }
}
