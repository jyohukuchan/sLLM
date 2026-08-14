use std::fmt;

/// Physical scalar storage formats. Quantization policy is represented by [`Encoding`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum DType {
    Bf16,
    F16,
    F32,
    F8E4M3Fn,
    F8E4M3FnuZ,
    F8E5M2,
    F8E5M2FnuZ,
    I32,
    I8,
    U8,
}

impl DType {
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F8E4M3Fn
            | Self::F8E4M3FnuZ
            | Self::F8E5M2
            | Self::F8E5M2FnuZ
            | Self::I8
            | Self::U8 => 1,
        }
    }

    pub const fn is_float(self) -> bool {
        !matches!(self, Self::I32 | Self::I8 | Self::U8)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F8E4M3Fn => "f8e4m3fn",
            Self::F8E4M3FnuZ => "f8e4m3fnuz",
            Self::F8E5M2 => "f8e5m2",
            Self::F8E5M2FnuZ => "f8e5m2fnuz",
            Self::I32 => "i32",
            Self::I8 => "i8",
            Self::U8 => "u8",
        };
        formatter.write_str(name)
    }
}

/// Storage encoding independent of the physical scalar [`DType`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Encoding {
    /// One scalar is stored in the selected physical dtype.
    Unquantized,
    /// Packed four-bit values with one scale per block.
    ///
    /// This is a descriptor only in Phase 1. No backend in this phase accepts it.
    Nvfp4 {
        block_size: usize,
        scale_dtype: DType,
    },
    /// One byte per FP8 value with scales stored in a distinct resident region.
    ///
    /// Keeping scales out of the value payload makes the safetensors layout,
    /// provider bindings, and resident memory accounting explicit.  OCP and
    /// FNUZ are selected by the physical [`DType`], never by reinterpreting the
    /// same byte stream.
    Fp8Scaled {
        granularity: Fp8ScaleGranularity,
        scale_dtype: DType,
        resident: Fp8ResidentRepresentation,
    },
}

/// How separately stored FP8 scale data maps to the value tensor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Fp8ScaleGranularity {
    /// One scale for the complete tensor.
    Tensor,
    /// One scale for every row of a row-major matrix.  For GEMM this is the
    /// hipBLASLt outer-vector contract: M activation scales and N weight
    /// scales multiply each MxN output element.
    OuterDimension,
    /// One scale per consecutive K-axis block in every row.
    KBlock { block_size: usize },
}

/// Physical ownership of FP8 values in model-resident memory.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Fp8ResidentRepresentation {
    /// Canonical one-byte FP8 values; scales occupy a separate resident region.
    PackedBytes,
    /// Explicitly converted BF16 resident storage used only by a provider that
    /// reports conversion rather than native or emulated FP8 execution.
    ConvertedBf16,
}

impl Encoding {
    /// Byte alignment required for the beginning of encoded storage.
    pub const fn offset_alignment(self, dtype: DType) -> u64 {
        match self {
            Self::Unquantized => dtype.size_bytes(),
            // NVFP4 values are packed into bytes.  Scale alignment is an
            // internal encoding detail and does not change the view offset.
            Self::Nvfp4 { .. } => 1,
            Self::Fp8Scaled { resident, .. } => match resident {
                Fp8ResidentRepresentation::PackedBytes => 1,
                Fp8ResidentRepresentation::ConvertedBf16 => DType::Bf16.size_bytes(),
            },
        }
    }

    pub fn validate(self, dtype: DType) -> Result<(), EncodingError> {
        match self {
            Self::Unquantized => Ok(()),
            Self::Nvfp4 {
                block_size,
                scale_dtype,
            } => {
                if block_size == 0 {
                    return Err(EncodingError::ZeroBlockSize);
                }
                if dtype != DType::U8 {
                    return Err(EncodingError::PackedStorageMustBeU8 { dtype });
                }
                if !matches!(scale_dtype, DType::Bf16 | DType::F16 | DType::F32) {
                    return Err(EncodingError::InvalidScaleDType { dtype: scale_dtype });
                }
                Ok(())
            }
            Self::Fp8Scaled {
                granularity,
                scale_dtype,
                resident,
            } => {
                if let Fp8ScaleGranularity::KBlock { block_size } = granularity {
                    if block_size == 0 {
                        return Err(EncodingError::ZeroBlockSize);
                    }
                }
                if !matches!(dtype, DType::F8E4M3Fn | DType::F8E4M3FnuZ) {
                    return Err(EncodingError::Fp8StorageRequired { dtype });
                }
                if scale_dtype != DType::F32 {
                    return Err(EncodingError::InvalidScaleDType { dtype: scale_dtype });
                }
                if matches!(resident, Fp8ResidentRepresentation::ConvertedBf16)
                    && !matches!(granularity, Fp8ScaleGranularity::Tensor)
                {
                    return Err(EncodingError::ConvertedBf16MustUseTensorScale);
                }
                Ok(())
            }
        }
    }

    pub fn storage_bytes(self, dtype: DType, elements: u64) -> Result<u64, EncodingError> {
        self.validate(dtype)?;
        match self {
            Self::Unquantized => elements
                .checked_mul(dtype.size_bytes())
                .ok_or(EncodingError::SizeOverflow),
            Self::Nvfp4 {
                block_size,
                scale_dtype,
            } => {
                let packed_values = elements / 2 + u64::from(elements % 2 != 0);
                let block_size =
                    u64::try_from(block_size).map_err(|_| EncodingError::SizeOverflow)?;
                let blocks = elements / block_size + u64::from(elements % block_size != 0);
                let scales = blocks
                    .checked_mul(scale_dtype.size_bytes())
                    .ok_or(EncodingError::SizeOverflow)?;
                packed_values
                    .checked_add(scales)
                    .ok_or(EncodingError::SizeOverflow)
            }
            Self::Fp8Scaled { resident, .. } => elements
                .checked_mul(match resident {
                    Fp8ResidentRepresentation::PackedBytes => 1,
                    Fp8ResidentRepresentation::ConvertedBf16 => DType::Bf16.size_bytes(),
                })
                .ok_or(EncodingError::SizeOverflow),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodingError {
    ZeroBlockSize,
    PackedStorageMustBeU8 { dtype: DType },
    Fp8StorageRequired { dtype: DType },
    InvalidScaleDType { dtype: DType },
    ConvertedBf16MustUseTensorScale,
    SizeOverflow,
}

impl fmt::Display for EncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlockSize => formatter.write_str("encoding block size must be non-zero"),
            Self::PackedStorageMustBeU8 { dtype } => {
                write!(formatter, "packed NVFP4 storage must use u8, got {dtype}")
            }
            Self::Fp8StorageRequired { dtype } => {
                write!(
                    formatter,
                    "scaled FP8 storage must use an E4M3 dtype, got {dtype}"
                )
            }
            Self::InvalidScaleDType { dtype } => {
                write!(formatter, "invalid NVFP4 scale dtype: {dtype}")
            }
            Self::ConvertedBf16MustUseTensorScale => formatter.write_str(
                "converted BF16 FP8 residency must use the explicit tensor-scale marker",
            ),
            Self::SizeOverflow => formatter.write_str("encoding byte size overflowed u64"),
        }
    }
}

impl std::error::Error for EncodingError {}
