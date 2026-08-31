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
    /// `scale_dtype` is the separately resident block-scale dtype. Weight
    /// NVFP4 v1 additionally requires one separately resident FP32 tensor
    /// scale. The logical view covers packed values only; resident layout owns
    /// and validates both scale regions explicitly.
    Nvfp4 {
        block_size: usize,
        scale_dtype: DType,
    },
    /// Packed NVFP4 weights consumed through a W4A4 matmul contract.
    ///
    /// The resident weight region contains block scales followed by separate
    /// FP32 decode-global scales for the weight and dynamically quantized
    /// activation.  Keeping this distinct from [`Self::Nvfp4`] prevents the
    /// weight-only W4A16 path from being selected accidentally.
    Nvfp4W4A4 {
        block_size: usize,
        scale_dtype: DType,
    },
    /// OCP MXFP4 weights consumed through a dynamic-activation W4A4 matmul.
    ///
    /// Values use packed E2M1 nibbles and each consecutive K-axis block of 32
    /// owns one E8M0 scale byte. E8M0 is represented physically as [`DType::U8`]
    /// because it is a scale encoding rather than an arithmetic scalar dtype.
    Mxfp4W4A4 {
        block_size: usize,
        scale_dtype: DType,
    },
    /// OCP MXFP8 E4M3 weights consumed through dynamic MXFP8 activation
    /// quantization. Values use one byte each and every consecutive K-axis
    /// block of 32 owns one separately resident E8M0 scale byte.
    Mxfp8W8A8 {
        block_size: usize,
        scale_dtype: DType,
    },
    /// OCP MXFP6 E3M2 weights consumed through dynamic MXFP6 activation
    /// quantization. Four six-bit values are packed into three bytes and every
    /// consecutive K-axis block of 32 owns one E8M0 scale byte.
    Mxfp6W6A6 {
        block_size: usize,
        scale_dtype: DType,
    },
    /// One byte per FP8 value with scales stored in a distinct resident region.
    ///
    /// Keeping scales out of the value payload makes the safetensors layout,
    /// provider bindings, and resident memory accounting explicit.  OCP and
    /// FNUZ are selected by the physical [`DType`], never by reinterpreting the
    /// same byte stream. Outer/tensor scales are FP32. KV block16 uses
    /// `KBlock { block_size: 16 }` with raw-U8 E8M0 scales.
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
            Self::Nvfp4 { .. }
            | Self::Nvfp4W4A4 { .. }
            | Self::Mxfp4W4A4 { .. }
            | Self::Mxfp8W8A8 { .. }
            | Self::Mxfp6W6A6 { .. } => 1,
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
            }
            | Self::Nvfp4W4A4 {
                block_size,
                scale_dtype,
            } => {
                if block_size != 16 {
                    return Err(EncodingError::InvalidNvfp4BlockSize { block_size });
                }
                if dtype != DType::U8 {
                    return Err(EncodingError::PackedStorageMustBeU8 { dtype });
                }
                if scale_dtype != DType::F8E4M3Fn {
                    return Err(EncodingError::Nvfp4BlockScaleMustBeE4M3Fn { dtype: scale_dtype });
                }
                Ok(())
            }
            Self::Mxfp4W4A4 {
                block_size,
                scale_dtype,
            } => {
                if block_size != 32 {
                    return Err(EncodingError::InvalidMxfp4BlockSize { block_size });
                }
                if dtype != DType::U8 {
                    return Err(EncodingError::PackedStorageMustBeU8 { dtype });
                }
                if scale_dtype != DType::U8 {
                    return Err(EncodingError::Mxfp4BlockScaleMustBeE8M0 { dtype: scale_dtype });
                }
                Ok(())
            }
            Self::Mxfp8W8A8 {
                block_size,
                scale_dtype,
            } => {
                if block_size != 32 {
                    return Err(EncodingError::InvalidMxfp8BlockSize { block_size });
                }
                if dtype != DType::F8E4M3Fn {
                    return Err(EncodingError::Mxfp8StorageMustBeE4M3Fn { dtype });
                }
                if scale_dtype != DType::U8 {
                    return Err(EncodingError::Mxfp8BlockScaleMustBeE8M0 { dtype: scale_dtype });
                }
                Ok(())
            }
            Self::Mxfp6W6A6 {
                block_size,
                scale_dtype,
            } => {
                if block_size != 32 {
                    return Err(EncodingError::InvalidMxfp6BlockSize { block_size });
                }
                if dtype != DType::U8 {
                    return Err(EncodingError::PackedStorageMustBeU8 { dtype });
                }
                if scale_dtype != DType::U8 {
                    return Err(EncodingError::Mxfp6BlockScaleMustBeE8M0 { dtype: scale_dtype });
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
                if !matches!(dtype, DType::F8E4M3Fn | DType::F8E4M3FnuZ | DType::F8E5M2) {
                    return Err(EncodingError::Fp8StorageRequired { dtype });
                }
                let valid_scale = match granularity {
                    Fp8ScaleGranularity::KBlock {
                        block_size: 16 | 32,
                    } => {
                        matches!(scale_dtype, DType::F32 | DType::U8)
                    }
                    _ => scale_dtype == DType::F32,
                };
                if !valid_scale {
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
            Self::Nvfp4 { .. } | Self::Nvfp4W4A4 { .. } | Self::Mxfp4W4A4 { .. } => {
                Ok(elements / 2 + u64::from(elements % 2 != 0))
            }
            Self::Mxfp8W8A8 { .. } => Ok(elements),
            Self::Mxfp6W6A6 { .. } => elements
                .checked_mul(6)
                .and_then(|bits| bits.checked_add(7))
                .map(|bits| bits / 8)
                .ok_or(EncodingError::SizeOverflow),
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
    InvalidNvfp4BlockSize { block_size: usize },
    InvalidMxfp4BlockSize { block_size: usize },
    InvalidMxfp8BlockSize { block_size: usize },
    InvalidMxfp6BlockSize { block_size: usize },
    PackedStorageMustBeU8 { dtype: DType },
    Nvfp4BlockScaleMustBeE4M3Fn { dtype: DType },
    Mxfp4BlockScaleMustBeE8M0 { dtype: DType },
    Mxfp8StorageMustBeE4M3Fn { dtype: DType },
    Mxfp8BlockScaleMustBeE8M0 { dtype: DType },
    Mxfp6BlockScaleMustBeE8M0 { dtype: DType },
    Fp8StorageRequired { dtype: DType },
    InvalidScaleDType { dtype: DType },
    ConvertedBf16MustUseTensorScale,
    SizeOverflow,
}

impl fmt::Display for EncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlockSize => formatter.write_str("encoding block size must be non-zero"),
            Self::InvalidNvfp4BlockSize { block_size } => {
                write!(formatter, "NVFP4 block size must be 16, got {block_size}")
            }
            Self::InvalidMxfp4BlockSize { block_size } => {
                write!(formatter, "MXFP4 block size must be 32, got {block_size}")
            }
            Self::InvalidMxfp8BlockSize { block_size } => {
                write!(formatter, "MXFP8 block size must be 32, got {block_size}")
            }
            Self::InvalidMxfp6BlockSize { block_size } => {
                write!(formatter, "MXFP6 block size must be 32, got {block_size}")
            }
            Self::PackedStorageMustBeU8 { dtype } => {
                write!(formatter, "packed NVFP4 storage must use u8, got {dtype}")
            }
            Self::Fp8StorageRequired { dtype } => {
                write!(
                    formatter,
                    "scaled FP8 storage must use an E4M3 or E5M2 dtype, got {dtype}"
                )
            }
            Self::Nvfp4BlockScaleMustBeE4M3Fn { dtype } => {
                write!(
                    formatter,
                    "NVFP4 block scale must use OCP E4M3FN, got {dtype}"
                )
            }
            Self::Mxfp4BlockScaleMustBeE8M0 { dtype } => {
                write!(
                    formatter,
                    "MXFP4 block scale must use E8M0 u8 storage, got {dtype}"
                )
            }
            Self::Mxfp8StorageMustBeE4M3Fn { dtype } => {
                write!(
                    formatter,
                    "MXFP8 E4M3 storage must use f8e4m3fn, got {dtype}"
                )
            }
            Self::Mxfp8BlockScaleMustBeE8M0 { dtype } => {
                write!(
                    formatter,
                    "MXFP8 block scale must use E8M0 u8 storage, got {dtype}"
                )
            }
            Self::Mxfp6BlockScaleMustBeE8M0 { dtype } => {
                write!(
                    formatter,
                    "MXFP6 block scale must use E8M0 u8 storage, got {dtype}"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_fp8_block16_accepts_e8m0_without_broadening_outer_scale_contract() {
        let block16 = Encoding::Fp8Scaled {
            granularity: Fp8ScaleGranularity::KBlock { block_size: 16 },
            scale_dtype: DType::U8,
            resident: Fp8ResidentRepresentation::PackedBytes,
        };
        for dtype in [DType::F8E4M3Fn, DType::F8E4M3FnuZ, DType::F8E5M2] {
            assert_eq!(block16.validate(dtype), Ok(()));
        }
        let block32 = Encoding::Fp8Scaled {
            granularity: Fp8ScaleGranularity::KBlock { block_size: 32 },
            scale_dtype: DType::U8,
            resident: Fp8ResidentRepresentation::PackedBytes,
        };
        assert_eq!(block32.validate(DType::F8E4M3Fn), Ok(()));
        assert_eq!(block32.validate(DType::F8E5M2), Ok(()));
        let outer_e8m0 = Encoding::Fp8Scaled {
            granularity: Fp8ScaleGranularity::OuterDimension,
            scale_dtype: DType::U8,
            resident: Fp8ResidentRepresentation::PackedBytes,
        };
        assert!(matches!(
            outer_e8m0.validate(DType::F8E4M3Fn),
            Err(EncodingError::InvalidScaleDType { .. })
        ));
    }

    #[test]
    fn ocp_mx_weight_activation_encodings_fix_types_block_and_value_bytes() {
        let mxfp8 = Encoding::Mxfp8W8A8 {
            block_size: 32,
            scale_dtype: DType::U8,
        };
        assert_eq!(mxfp8.validate(DType::F8E4M3Fn), Ok(()));
        assert_eq!(mxfp8.storage_bytes(DType::F8E4M3Fn, 192), Ok(192));
        assert!(mxfp8.validate(DType::F8E4M3FnuZ).is_err());
        assert!(
            Encoding::Mxfp8W8A8 {
                block_size: 16,
                scale_dtype: DType::U8,
            }
            .validate(DType::F8E4M3Fn)
            .is_err()
        );

        let mxfp6 = Encoding::Mxfp6W6A6 {
            block_size: 32,
            scale_dtype: DType::U8,
        };
        assert_eq!(mxfp6.validate(DType::U8), Ok(()));
        assert_eq!(mxfp6.storage_bytes(DType::U8, 192), Ok(144));
        assert!(mxfp6.validate(DType::F8E4M3Fn).is_err());
        assert!(
            Encoding::Mxfp6W6A6 {
                block_size: 32,
                scale_dtype: DType::Bf16,
            }
            .validate(DType::U8)
            .is_err()
        );
    }
}
