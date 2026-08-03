use std::fmt;

use crate::{DType, Encoding, EncodingError};

/// A non-owning tensor descriptor. Strides are measured in logical elements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorView {
    dtype: DType,
    encoding: Encoding,
    shape: Vec<usize>,
    strides: Vec<usize>,
    byte_offset: u64,
    span_bytes: u64,
}

impl TensorView {
    /// Creates a contiguous, zero-offset view using unquantized storage.
    pub fn contiguous(dtype: DType, shape: &[usize]) -> Result<Self, TensorError> {
        Self::with_encoding(dtype, Encoding::Unquantized, shape)
    }

    /// Creates a contiguous, zero-offset view with an explicit encoding.
    pub fn with_encoding(
        dtype: DType,
        encoding: Encoding,
        shape: &[usize],
    ) -> Result<Self, TensorError> {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1usize;
        for (dimension, current_stride) in shape.iter().zip(strides.iter_mut()).rev() {
            *current_stride = stride;
            stride = stride
                .checked_mul(*dimension)
                .ok_or(TensorError::ShapeOverflow)?;
        }
        Self::new(dtype, encoding, shape, &strides, 0)
    }

    /// Creates a view with element strides and a byte offset into its backing buffer.
    pub fn new(
        dtype: DType,
        encoding: Encoding,
        shape: &[usize],
        strides: &[usize],
        byte_offset: u64,
    ) -> Result<Self, TensorError> {
        if shape.len() != strides.len() {
            return Err(TensorError::RankMismatch {
                shape_rank: shape.len(),
                stride_rank: strides.len(),
            });
        }
        encoding.validate(dtype)?;
        let alignment = encoding.offset_alignment(dtype);
        if byte_offset % alignment != 0 {
            return Err(TensorError::MisalignedOffset {
                offset: byte_offset,
                alignment,
            });
        }

        let mut element_count = 1u64;
        let mut max_element_index = 0u64;
        let mut has_zero_extent = false;
        for (&dimension, &stride) in shape.iter().zip(strides) {
            let dimension = u64::try_from(dimension).map_err(|_| TensorError::SizeOverflow)?;
            let stride = u64::try_from(stride).map_err(|_| TensorError::SizeOverflow)?;
            if dimension == 0 {
                has_zero_extent = true;
            }
            element_count = element_count
                .checked_mul(dimension)
                .ok_or(TensorError::ShapeOverflow)?;
            if dimension > 0 {
                max_element_index = max_element_index
                    .checked_add(
                        (dimension - 1)
                            .checked_mul(stride)
                            .ok_or(TensorError::ShapeOverflow)?,
                    )
                    .ok_or(TensorError::ShapeOverflow)?;
            }
        }

        let payload_bytes = if has_zero_extent {
            0
        } else {
            let addressed_elements = max_element_index
                .checked_add(1)
                .ok_or(TensorError::ShapeOverflow)?;
            encoding.storage_bytes(dtype, addressed_elements)?
        };
        let span_bytes = byte_offset
            .checked_add(payload_bytes)
            .ok_or(TensorError::SizeOverflow)?;

        Ok(Self {
            dtype,
            encoding,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            byte_offset,
            span_bytes,
        })
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn encoding(&self) -> Encoding {
        self.encoding
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Number of logical elements, including the scalar case of rank zero.
    pub fn element_count(&self) -> u64 {
        self.shape
            .iter()
            .try_fold(1u64, |count, &dimension| {
                count.checked_mul(dimension as u64)
            })
            .expect("TensorView validates element count at construction")
    }

    /// Bytes covered from the beginning of the backing buffer through this view.
    pub const fn span_bytes(&self) -> u64 {
        self.span_bytes
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1usize;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if dimension != 0 && stride != expected {
                return false;
            }
            expected = match expected.checked_mul(dimension) {
                Some(value) => value,
                None => return false,
            };
        }
        self.byte_offset == 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorError {
    RankMismatch {
        shape_rank: usize,
        stride_rank: usize,
    },
    ShapeOverflow,
    SizeOverflow,
    MisalignedOffset {
        offset: u64,
        alignment: u64,
    },
    InvalidEncoding(EncodingError),
}

impl fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankMismatch {
                shape_rank,
                stride_rank,
            } => write!(
                formatter,
                "shape rank {shape_rank} differs from stride rank {stride_rank}"
            ),
            Self::ShapeOverflow => formatter.write_str("tensor shape or stride overflowed"),
            Self::SizeOverflow => formatter.write_str("tensor byte size overflowed u64"),
            Self::MisalignedOffset { offset, alignment } => write!(
                formatter,
                "tensor byte offset {offset} is not aligned to {alignment} bytes"
            ),
            Self::InvalidEncoding(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TensorError {}

impl From<EncodingError> for TensorError {
    fn from(error: EncodingError) -> Self {
        Self::InvalidEncoding(error)
    }
}
