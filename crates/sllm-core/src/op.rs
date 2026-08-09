use std::fmt;

use crate::{DType, Encoding, TensorView};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SemanticOpKind {
    Copy,
    Add,
    Matmul,
    RmsNorm,
}

impl SemanticOpKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Add => "add",
            Self::Matmul => "matmul",
            Self::RmsNorm => "rms_norm",
        }
    }

    pub const fn arity(self) -> (usize, usize) {
        match self {
            Self::Copy => (1, 1),
            Self::Add => (2, 1),
            Self::Matmul => (2, 1),
            Self::RmsNorm => (2, 1),
        }
    }
}

/// The scale interpretation used by the Qwen3.5 RMSNorm contract.
///
/// This is intentionally an explicit enum rather than an implicit default:
/// the raw checkpoint scale is not a conventional RMSNorm scale.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RmsNormScaleMode {
    OffsetOne,
}

/// Tensor roles used in RMSNorm validation errors.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RmsNormTensor {
    Activation,
    RawScale,
    Output,
}

impl fmt::Display for RmsNormTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Activation => "activation",
            Self::RawScale => "raw scale",
            Self::Output => "output",
        })
    }
}

/// A positive finite FP32 epsilon stored by bits so the contract remains Eq.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RmsNormEpsilon(u32);

impl RmsNormEpsilon {
    pub fn new(value: f32) -> Result<Self, OpError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(OpError::RmsNormInvalidEpsilon {
                bits: value.to_bits(),
            });
        }
        Ok(Self(value.to_bits()))
    }

    pub const fn value(self) -> f32 {
        f32::from_bits(self.0)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }
}

/// Fixed numeric and aliasing promises made by an RMSNorm descriptor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RmsNormContract {
    epsilon: RmsNormEpsilon,
    scale_mode: RmsNormScaleMode,
    accumulation_dtype: DType,
    output_dtype: DType,
    alias_policy: RmsNormAliasPolicy,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RmsNormAliasPolicy {
    Unsupported,
}

impl RmsNormContract {
    /// Creates the explicit baseline contract. There is no implicit epsilon or
    /// scale-mode default for RMSNorm.
    pub fn new(epsilon: f32, scale_mode: RmsNormScaleMode) -> Result<Self, OpError> {
        Ok(Self {
            epsilon: RmsNormEpsilon::new(epsilon)?,
            scale_mode,
            accumulation_dtype: DType::F32,
            output_dtype: DType::Bf16,
            alias_policy: RmsNormAliasPolicy::Unsupported,
        })
    }

    pub const fn epsilon(self) -> RmsNormEpsilon {
        self.epsilon
    }

    pub const fn scale_mode(self) -> RmsNormScaleMode {
        self.scale_mode
    }

    pub const fn accumulation_dtype(self) -> DType {
        self.accumulation_dtype
    }

    pub const fn output_dtype(self) -> DType {
        self.output_dtype
    }

    pub const fn alias_policy(self) -> RmsNormAliasPolicy {
        self.alias_policy
    }

    /// Returns the effective FP32 scale for one raw BF16 scale value.
    pub fn effective_scale(self, raw_scale: f32) -> f32 {
        match self.scale_mode {
            RmsNormScaleMode::OffsetOne => 1.0_f32 + raw_scale,
        }
    }
}

/// A backend-independent operation contract containing only tensor metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticOpDescriptor {
    kind: SemanticOpKind,
    inputs: Vec<TensorView>,
    outputs: Vec<TensorView>,
    rms_norm_contract: Option<RmsNormContract>,
}

/// Short name for the semantic operation descriptor used by backend traits.
pub type SemanticOp = SemanticOpDescriptor;

impl SemanticOpDescriptor {
    pub fn new(
        kind: SemanticOpKind,
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind,
            inputs,
            outputs,
            rms_norm_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Creates an RMSNorm descriptor with every RMSNorm-only parameter
    /// explicit. The generic constructor intentionally cannot supply defaults.
    pub fn new_rms_norm(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        epsilon: f32,
        scale_mode: RmsNormScaleMode,
    ) -> Result<Self, OpError> {
        let contract = RmsNormContract::new(epsilon, scale_mode)?;
        Self::new_rms_norm_with_contract(inputs, outputs, contract)
    }

    pub fn new_rms_norm_with_contract(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: RmsNormContract,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::RmsNorm,
            inputs,
            outputs,
            rms_norm_contract: Some(contract),
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub const fn kind(&self) -> SemanticOpKind {
        self.kind
    }

    pub const fn arity(&self) -> (usize, usize) {
        self.kind.arity()
    }

    pub fn inputs(&self) -> &[TensorView] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[TensorView] {
        &self.outputs
    }

    pub const fn rms_norm_contract(&self) -> Option<RmsNormContract> {
        self.rms_norm_contract
    }

    pub fn validate(&self) -> Result<(), OpError> {
        let (expected_inputs, expected_outputs) = self.arity();
        if self.inputs.len() != expected_inputs || self.outputs.len() != expected_outputs {
            return Err(OpError::Arity {
                kind: self.kind,
                expected_inputs,
                actual_inputs: self.inputs.len(),
                expected_outputs,
                actual_outputs: self.outputs.len(),
            });
        }

        match self.kind {
            SemanticOpKind::Copy => {
                if !same_metadata(&self.inputs[0], &self.outputs[0]) {
                    return Err(OpError::CopyMetadataMismatch);
                }
            }
            SemanticOpKind::Add => {
                if !same_metadata(&self.inputs[0], &self.inputs[1])
                    || !same_metadata(&self.inputs[0], &self.outputs[0])
                {
                    return Err(OpError::ElementwiseMetadataMismatch);
                }
            }
            SemanticOpKind::Matmul => {
                let left = self.inputs[0].shape();
                let right = self.inputs[1].shape();
                let output = self.outputs[0].shape();
                if left.len() != 2
                    || right.len() != 2
                    || output.len() != 2
                    || left[1] != right[0]
                    || output != [left[0], right[1]]
                    || self.inputs[0].dtype() != self.inputs[1].dtype()
                    || self.inputs[0].dtype() != self.outputs[0].dtype()
                    || self.inputs[0].encoding() != self.inputs[1].encoding()
                    || self.inputs[0].encoding() != self.outputs[0].encoding()
                {
                    return Err(OpError::MatmulShapeMismatch);
                }
            }
            SemanticOpKind::RmsNorm => {
                let contract = self
                    .rms_norm_contract
                    .ok_or(OpError::RmsNormContractRequired)?;
                validate_rms_norm(&self.inputs, &self.outputs, contract)?;
            }
        }
        Ok(())
    }
}

fn validate_rms_norm(
    inputs: &[TensorView],
    outputs: &[TensorView],
    _contract: RmsNormContract,
) -> Result<(), OpError> {
    let activation = &inputs[0];
    let raw_scale = &inputs[1];
    let output = &outputs[0];

    for (tensor, role) in [
        (activation, RmsNormTensor::Activation),
        (raw_scale, RmsNormTensor::RawScale),
        (output, RmsNormTensor::Output),
    ] {
        if tensor.shape().is_empty() {
            return Err(OpError::RmsNormRankZero { tensor: role });
        }
        if tensor.shape().contains(&0) {
            return Err(OpError::RmsNormZeroExtent { tensor: role });
        }
        if !tensor.is_contiguous() {
            return Err(OpError::RmsNormNonContiguous { tensor: role });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::RmsNormUnsupportedEncoding {
                tensor: role,
                actual: tensor.encoding(),
            });
        }
        if tensor.dtype() != DType::Bf16 {
            return Err(OpError::RmsNormUnsupportedDType {
                tensor: role,
                actual: tensor.dtype(),
            });
        }
    }

    if activation.shape() != output.shape() {
        return Err(OpError::RmsNormOutputShapeMismatch);
    }
    if raw_scale.shape().len() != 1 {
        return Err(OpError::RmsNormScaleRankMismatch);
    }
    if raw_scale.shape()[0] != activation.shape()[activation.shape().len() - 1] {
        return Err(OpError::RmsNormScaleShapeMismatch);
    }

    // TensorView deliberately has no backing-buffer identity. Equal offsets
    // therefore neither prove nor disprove aliasing. The binding layer must
    // reject input/output aliasing according to the contract getter.
    Ok(())
}

fn same_metadata(left: &TensorView, right: &TensorView) -> bool {
    left.dtype() == right.dtype()
        && left.encoding() == right.encoding()
        && left.shape() == right.shape()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpError {
    Arity {
        kind: SemanticOpKind,
        expected_inputs: usize,
        actual_inputs: usize,
        expected_outputs: usize,
        actual_outputs: usize,
    },
    CopyMetadataMismatch,
    ElementwiseMetadataMismatch,
    MatmulShapeMismatch,
    RmsNormContractRequired,
    RmsNormRankZero {
        tensor: RmsNormTensor,
    },
    RmsNormZeroExtent {
        tensor: RmsNormTensor,
    },
    RmsNormNonContiguous {
        tensor: RmsNormTensor,
    },
    RmsNormUnsupportedDType {
        tensor: RmsNormTensor,
        actual: DType,
    },
    RmsNormUnsupportedEncoding {
        tensor: RmsNormTensor,
        actual: Encoding,
    },
    RmsNormOutputShapeMismatch,
    RmsNormScaleRankMismatch,
    RmsNormScaleShapeMismatch,
    RmsNormInvalidEpsilon {
        bits: u32,
    },
}

impl fmt::Display for OpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arity {
                kind,
                expected_inputs,
                actual_inputs,
                expected_outputs,
                actual_outputs,
            } => write!(
                formatter,
                "{} expects {expected_inputs} input(s) and {expected_outputs} output(s), got {actual_inputs} and {actual_outputs}",
                kind.name()
            ),
            Self::CopyMetadataMismatch => {
                formatter.write_str("copy input and output metadata differ")
            }
            Self::ElementwiseMetadataMismatch => {
                formatter.write_str("elementwise operand metadata differ")
            }
            Self::MatmulShapeMismatch => {
                formatter.write_str("matmul shapes or dtypes are incompatible")
            }
            Self::RmsNormContractRequired => {
                formatter.write_str("rms_norm requires an explicit contract")
            }
            Self::RmsNormRankZero { tensor } => {
                write!(formatter, "rms_norm {tensor} must have rank at least one")
            }
            Self::RmsNormZeroExtent { tensor } => {
                write!(formatter, "rms_norm {tensor} must not have a zero extent")
            }
            Self::RmsNormNonContiguous { tensor } => {
                write!(formatter, "rms_norm {tensor} must be row-major contiguous")
            }
            Self::RmsNormUnsupportedDType { tensor, actual } => {
                write!(formatter, "rms_norm {tensor} must be bf16, got {actual}")
            }
            Self::RmsNormUnsupportedEncoding { tensor, actual } => {
                write!(
                    formatter,
                    "rms_norm {tensor} must use unquantized encoding, got {actual:?}"
                )
            }
            Self::RmsNormOutputShapeMismatch => {
                formatter.write_str("rms_norm activation and output shapes differ")
            }
            Self::RmsNormScaleRankMismatch => {
                formatter.write_str("rms_norm raw scale must have rank one")
            }
            Self::RmsNormScaleShapeMismatch => {
                formatter.write_str("rms_norm raw scale length differs from the last dimension")
            }
            Self::RmsNormInvalidEpsilon { bits } => {
                write!(
                    formatter,
                    "rms_norm epsilon is not finite and positive (bits 0x{bits:08x})"
                )
            }
        }
    }
}

impl std::error::Error for OpError {}
