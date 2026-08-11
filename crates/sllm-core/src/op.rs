use std::fmt;

use crate::{DType, Encoding, TensorView};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SemanticOpKind {
    Copy,
    Add,
    Embedding,
    Matmul,
    SiluMul,
    SigmoidMul,
    RmsNorm,
    AttentionPreprocess,
    Argmax,
}

/// The only accepted C3a1 Q/gate storage layout. The packed tensor is
/// `[M, 16, 512]`, with `[Q256, gate256]` in the final axis of every head.
/// A flat `[Q4096, gate4096]` interpretation is intentionally not representable
/// by this contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttentionPreprocessPacking {
    HeadInterleavedQGate,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttentionPreprocessPositionMode {
    Prefill,
    DecodeContinuation,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttentionPreprocessTensor {
    PackedQGate,
    K,
    QRawScale,
    KRawScale,
    Positions,
    QOutput,
    GateOutput,
    KOutput,
}

impl fmt::Display for AttentionPreprocessTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PackedQGate => "packed Q/gate",
            Self::K => "K",
            Self::QRawScale => "Q raw scale",
            Self::KRawScale => "K raw scale",
            Self::Positions => "positions",
            Self::QOutput => "Q output",
            Self::GateOutput => "gate output",
            Self::KOutput => "K output",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ElementwiseTensor {
    Input0,
    Input1,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ArgmaxTensor {
    Logits,
    Output,
}

impl fmt::Display for ArgmaxTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Logits => "logits",
            Self::Output => "output",
        })
    }
}

impl fmt::Display for ElementwiseTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Input0 => "input 0",
            Self::Input1 => "input 1",
            Self::Output => "output",
        })
    }
}

impl SemanticOpKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Add => "add",
            Self::Embedding => "embedding",
            Self::Matmul => "matmul",
            Self::SiluMul => "silu_mul",
            Self::SigmoidMul => "sigmoid_mul",
            Self::RmsNorm => "rms_norm",
            Self::AttentionPreprocess => "attention_preprocess",
            Self::Argmax => "argmax",
        }
    }

    pub const fn arity(self) -> (usize, usize) {
        match self {
            Self::Copy => (1, 1),
            Self::Add => (2, 1),
            Self::Embedding => (2, 1),
            Self::Matmul => (2, 1),
            Self::SiluMul => (2, 1),
            Self::SigmoidMul => (2, 1),
            Self::RmsNorm => (2, 1),
            Self::AttentionPreprocess => (5, 3),
            Self::Argmax => (1, 1),
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

/// The complete backend-neutral C3a1 semantic contract.
///
/// This descriptor carries the expected absolute position sequence as
/// metadata. The later host-side dispatch boundary must compare the actual
/// I32 position bytes with `start_position .. start_position + token_count`.
/// Keeping the sequence here makes prefill reset and decode continuation
/// distinct without pretending that a `TensorView` contains payload values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AttentionPreprocessContract {
    packing: AttentionPreprocessPacking,
    position_mode: AttentionPreprocessPositionMode,
    start_position: u32,
    token_count: u32,
    epsilon: RmsNormEpsilon,
    scale_mode: RmsNormScaleMode,
    accumulation_dtype: DType,
    output_dtype: DType,
    rotary_dim: u32,
    rope_theta_bits: u32,
    mrope_interleaved: bool,
    mrope_sections: [u32; 3],
    max_position_embeddings: u32,
}

impl AttentionPreprocessContract {
    pub const Q_HEADS: usize = 16;
    pub const KV_HEADS: usize = 4;
    pub const HEAD_DIM: usize = 256;
    pub const PACKED_Q_GATE_WIDTH: usize = 512;
    pub const ROTARY_DIM: u32 = 64;
    pub const MAX_POSITION_EMBEDDINGS: u32 = 262_144;
    pub const MROPE_SECTIONS: [u32; 3] = [11, 11, 10];

    /// Constructs the exact Qwen3.5 C3a1 contract, rejecting any changed
    /// numeric, layout, RoPE, or position configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        packing: AttentionPreprocessPacking,
        position_mode: AttentionPreprocessPositionMode,
        start_position: i64,
        token_count: u64,
        epsilon: f32,
        scale_mode: RmsNormScaleMode,
        accumulation_dtype: DType,
        output_dtype: DType,
        rotary_dim: u32,
        rope_theta: f32,
        mrope_interleaved: bool,
        mrope_sections: [u32; 3],
        max_position_embeddings: u32,
    ) -> Result<Self, OpError> {
        if !matches!(packing, AttentionPreprocessPacking::HeadInterleavedQGate) {
            return Err(OpError::AttentionPreprocessInvalidConfig { field: "packing" });
        }
        if accumulation_dtype != DType::F32 {
            return Err(OpError::AttentionPreprocessInvalidConfig {
                field: "accumulation dtype",
            });
        }
        if output_dtype != DType::Bf16 {
            return Err(OpError::AttentionPreprocessInvalidConfig {
                field: "output dtype",
            });
        }
        if rotary_dim != Self::ROTARY_DIM {
            return Err(OpError::AttentionPreprocessInvalidConfig {
                field: "rotary dimension",
            });
        }
        if rope_theta.to_bits() != 10_000_000.0_f32.to_bits() {
            return Err(OpError::AttentionPreprocessInvalidConfig {
                field: "RoPE theta",
            });
        }
        if !mrope_interleaved {
            return Err(OpError::AttentionPreprocessInvalidConfig {
                field: "mrope_interleaved",
            });
        }
        if mrope_sections != Self::MROPE_SECTIONS {
            return Err(OpError::AttentionPreprocessInvalidConfig {
                field: "mRoPE sections",
            });
        }
        if max_position_embeddings != Self::MAX_POSITION_EMBEDDINGS {
            return Err(OpError::AttentionPreprocessInvalidConfig {
                field: "max position embeddings",
            });
        }
        let epsilon = RmsNormEpsilon::new(epsilon)?;
        if epsilon.bits() != 1.0e-6_f32.to_bits() {
            return Err(OpError::AttentionPreprocessInvalidConfig { field: "epsilon" });
        }
        if !matches!(scale_mode, RmsNormScaleMode::OffsetOne) {
            return Err(OpError::AttentionPreprocessInvalidConfig {
                field: "scale mode",
            });
        }
        if token_count == 0 {
            return Err(OpError::AttentionPreprocessZeroTokenCount);
        }
        if start_position < 0 {
            return Err(OpError::AttentionPreprocessNegativePosition { start_position });
        }
        match position_mode {
            AttentionPreprocessPositionMode::Prefill if start_position != 0 => {
                return Err(OpError::AttentionPreprocessPositionReset {
                    mode: position_mode,
                    start_position,
                });
            }
            AttentionPreprocessPositionMode::DecodeContinuation if start_position == 0 => {
                return Err(OpError::AttentionPreprocessPositionReset {
                    mode: position_mode,
                    start_position,
                });
            }
            _ => {}
        }
        let start_position_u64 = u64::try_from(start_position)
            .map_err(|_| OpError::AttentionPreprocessNegativePosition { start_position })?;
        let end_position = start_position_u64
            .checked_add(token_count)
            .ok_or(OpError::AttentionPreprocessPositionOverflow)?;
        if end_position > u64::from(max_position_embeddings) {
            return Err(OpError::AttentionPreprocessPositionOutOfRange {
                last_position: end_position - 1,
                max_position_embeddings,
            });
        }
        let start_position = u32::try_from(start_position_u64).map_err(|_| {
            OpError::AttentionPreprocessPositionOutOfRange {
                last_position: start_position_u64,
                max_position_embeddings,
            }
        })?;
        let token_count = u32::try_from(token_count)
            .map_err(|_| OpError::AttentionPreprocessTokenCountOverflow { token_count })?;

        Ok(Self {
            packing,
            position_mode,
            start_position,
            token_count,
            epsilon,
            scale_mode,
            accumulation_dtype,
            output_dtype,
            rotary_dim,
            rope_theta_bits: rope_theta.to_bits(),
            mrope_interleaved,
            mrope_sections,
            max_position_embeddings,
        })
    }

    pub fn new_qwen3_5(
        position_mode: AttentionPreprocessPositionMode,
        start_position: i64,
        token_count: u64,
    ) -> Result<Self, OpError> {
        Self::new(
            AttentionPreprocessPacking::HeadInterleavedQGate,
            position_mode,
            start_position,
            token_count,
            1.0e-6,
            RmsNormScaleMode::OffsetOne,
            DType::F32,
            DType::Bf16,
            Self::ROTARY_DIM,
            10_000_000.0,
            true,
            Self::MROPE_SECTIONS,
            Self::MAX_POSITION_EMBEDDINGS,
        )
    }

    pub const fn packing(self) -> AttentionPreprocessPacking {
        self.packing
    }

    pub const fn position_mode(self) -> AttentionPreprocessPositionMode {
        self.position_mode
    }

    pub const fn start_position(self) -> u32 {
        self.start_position
    }

    pub const fn token_count(self) -> u32 {
        self.token_count
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

    pub const fn rotary_dim(self) -> u32 {
        self.rotary_dim
    }

    pub const fn rope_theta_bits(self) -> u32 {
        self.rope_theta_bits
    }

    pub const fn rope_theta(self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }

    pub const fn mrope_interleaved(self) -> bool {
        self.mrope_interleaved
    }

    pub const fn mrope_sections(self) -> [u32; 3] {
        self.mrope_sections
    }

    pub const fn max_position_embeddings(self) -> u32 {
        self.max_position_embeddings
    }
}

/// A backend-independent operation contract containing only tensor metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticOpDescriptor {
    kind: SemanticOpKind,
    inputs: Vec<TensorView>,
    outputs: Vec<TensorView>,
    rms_norm_contract: Option<RmsNormContract>,
    attention_preprocess_contract: Option<AttentionPreprocessContract>,
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
            attention_preprocess_contract: None,
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
            attention_preprocess_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn new_attention_preprocess(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: AttentionPreprocessContract,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::AttentionPreprocess,
            inputs,
            outputs,
            rms_norm_contract: None,
            attention_preprocess_contract: Some(contract),
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

    pub const fn attention_preprocess_contract(&self) -> Option<AttentionPreprocessContract> {
        self.attention_preprocess_contract
    }

    /// Returns the zero-copy rank-2 view consumed by the existing `o_proj`
    /// matmul path. Only the validated C3c sigmoid output gate has this
    /// handoff: `[M, 16, 256]` is the same contiguous storage as `[M, 4096]`.
    pub fn sigmoid_mul_o_proj_input_view(&self) -> Option<TensorView> {
        if self.kind != SemanticOpKind::SigmoidMul {
            return None;
        }
        let output = &self.outputs[0];
        let m = output.shape()[0];
        Some(
            TensorView::new(
                DType::Bf16,
                Encoding::Unquantized,
                &[m, 16 * 256],
                &[16 * 256, 1],
                output.byte_offset(),
            )
            .expect("validated sigmoid_mul output always has a representable o_proj view"),
        )
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
                validate_baseline_elementwise(self.kind, &self.inputs, &self.outputs)?;
            }
            SemanticOpKind::Add => {
                validate_baseline_elementwise(self.kind, &self.inputs, &self.outputs)?;
            }
            SemanticOpKind::Embedding => {
                validate_embedding(&self.inputs, &self.outputs)?;
            }
            SemanticOpKind::Matmul => {
                validate_matmul(&self.inputs, &self.outputs)?;
            }
            SemanticOpKind::SiluMul => {
                validate_baseline_elementwise(self.kind, &self.inputs, &self.outputs)?;
            }
            SemanticOpKind::SigmoidMul => {
                validate_baseline_elementwise(self.kind, &self.inputs, &self.outputs)?;
            }
            SemanticOpKind::RmsNorm => {
                let contract = self
                    .rms_norm_contract
                    .ok_or(OpError::RmsNormContractRequired)?;
                validate_rms_norm(&self.inputs, &self.outputs, contract)?;
            }
            SemanticOpKind::AttentionPreprocess => {
                let contract = self
                    .attention_preprocess_contract
                    .ok_or(OpError::AttentionPreprocessContractRequired)?;
                validate_attention_preprocess(&self.inputs, &self.outputs, contract)?;
            }
            SemanticOpKind::Argmax => {
                validate_argmax(&self.inputs, &self.outputs)?;
            }
        }
        Ok(())
    }
}

fn validate_argmax(inputs: &[TensorView], outputs: &[TensorView]) -> Result<(), OpError> {
    let logits = &inputs[0];
    let output = &outputs[0];
    if logits.shape().len() != 2 {
        return Err(OpError::ArgmaxRankMismatch {
            tensor: ArgmaxTensor::Logits,
        });
    }
    if output.shape().len() != 1 {
        return Err(OpError::ArgmaxRankMismatch {
            tensor: ArgmaxTensor::Output,
        });
    }
    for (tensor, role) in [
        (logits, ArgmaxTensor::Logits),
        (output, ArgmaxTensor::Output),
    ] {
        if tensor.shape().contains(&0) {
            return Err(OpError::ArgmaxZeroExtent { tensor: role });
        }
        if !tensor.is_contiguous() {
            return Err(OpError::ArgmaxNonContiguous { tensor: role });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::ArgmaxUnsupportedEncoding {
                tensor: role,
                actual: tensor.encoding(),
            });
        }
    }
    if logits.dtype() != DType::Bf16 {
        return Err(OpError::ArgmaxUnsupportedDType {
            tensor: ArgmaxTensor::Logits,
            expected: DType::Bf16,
            actual: logits.dtype(),
        });
    }
    if output.dtype() != DType::I32 {
        return Err(OpError::ArgmaxUnsupportedDType {
            tensor: ArgmaxTensor::Output,
            expected: DType::I32,
            actual: output.dtype(),
        });
    }
    if output.shape()[0] != logits.shape()[0] {
        return Err(OpError::ArgmaxShapeMismatch);
    }
    if logits.shape()[1] > 1_048_576 {
        return Err(OpError::ArgmaxVocabTooLarge {
            vocab: logits.shape()[1],
        });
    }
    Ok(())
}

fn validate_embedding(inputs: &[TensorView], outputs: &[TensorView]) -> Result<(), OpError> {
    let weight = &inputs[0];
    let token_ids = &inputs[1];
    let output = &outputs[0];
    for tensor in [weight, token_ids, output] {
        if tensor.shape().contains(&0) {
            return Err(OpError::EmbeddingZeroExtent);
        }
        if !tensor.is_contiguous() {
            return Err(OpError::EmbeddingNonContiguous);
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::EmbeddingUnsupportedEncoding);
        }
    }
    if weight.shape().len() != 2
        || weight.dtype() != DType::Bf16
        || token_ids.shape().len() != 1
        || token_ids.dtype() != DType::I32
        || output.shape().len() != 2
        || output.dtype() != DType::Bf16
    {
        return Err(OpError::EmbeddingTensorContractMismatch);
    }
    if output.shape() != [token_ids.shape()[0], weight.shape()[1]] {
        return Err(OpError::EmbeddingOutputShapeMismatch);
    }
    Ok(())
}

fn validate_matmul(inputs: &[TensorView], outputs: &[TensorView]) -> Result<(), OpError> {
    let activation = &inputs[0];
    let weight = &inputs[1];
    let output = &outputs[0];
    for tensor in [activation, weight, output] {
        if tensor.shape().contains(&0) {
            return Err(OpError::MatmulZeroExtent);
        }
    }
    let activation_shape = activation.shape();
    let weight_shape = weight.shape();
    let output_shape = output.shape();
    if activation_shape.len() != 2
        || weight_shape.len() != 2
        || output_shape.len() != 2
        || activation_shape[1] != weight_shape[1]
        || output_shape != [activation_shape[0], weight_shape[0]]
    {
        return Err(OpError::MatmulShapeMismatch);
    }
    for tensor in [activation, weight, output] {
        if !tensor.is_contiguous() {
            return Err(OpError::MatmulNonContiguous);
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::MatmulUnsupportedEncoding);
        }
        if tensor.dtype() != DType::Bf16 {
            return Err(OpError::MatmulUnsupportedDType {
                actual: tensor.dtype(),
            });
        }
    }
    Ok(())
}

fn validate_baseline_elementwise(
    kind: SemanticOpKind,
    inputs: &[TensorView],
    outputs: &[TensorView],
) -> Result<(), OpError> {
    let metadata_matches = match kind {
        SemanticOpKind::Copy => same_metadata(&inputs[0], &outputs[0]),
        SemanticOpKind::Add | SemanticOpKind::SiluMul | SemanticOpKind::SigmoidMul => {
            same_metadata(&inputs[0], &inputs[1]) && same_metadata(&inputs[0], &outputs[0])
        }
        _ => unreachable!("elementwise validation is only used by copy/add"),
    };
    if !metadata_matches {
        return Err(match kind {
            SemanticOpKind::Copy => OpError::CopyMetadataMismatch,
            SemanticOpKind::Add | SemanticOpKind::SiluMul | SemanticOpKind::SigmoidMul => {
                OpError::ElementwiseMetadataMismatch
            }
            _ => unreachable!("elementwise validation is only used by copy/add"),
        });
    }

    let mut tensors = Vec::with_capacity(inputs.len() + outputs.len());
    tensors.push((&inputs[0], ElementwiseTensor::Input0));
    if matches!(
        kind,
        SemanticOpKind::Add | SemanticOpKind::SiluMul | SemanticOpKind::SigmoidMul
    ) {
        tensors.push((&inputs[1], ElementwiseTensor::Input1));
    }
    tensors.push((&outputs[0], ElementwiseTensor::Output));
    for (tensor, role) in tensors {
        if tensor.shape().is_empty() {
            return Err(OpError::ElementwiseRankZero { kind, tensor: role });
        }
        if tensor.shape().contains(&0) {
            return Err(OpError::ElementwiseZeroExtent { kind, tensor: role });
        }
        if !tensor.is_contiguous() {
            return Err(OpError::ElementwiseNonContiguous { kind, tensor: role });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::ElementwiseUnsupportedEncoding {
                kind,
                tensor: role,
                actual: tensor.encoding(),
            });
        }
        if tensor.dtype() != DType::Bf16 {
            return Err(OpError::ElementwiseUnsupportedDType {
                kind,
                tensor: role,
                actual: tensor.dtype(),
            });
        }
    }
    if kind == SemanticOpKind::SigmoidMul {
        let shape = inputs[0].shape();
        if shape.len() != 3 || shape[1] != 16 || shape[2] != 256 {
            return Err(OpError::SigmoidMulShapeMismatch);
        }
    }
    Ok(())
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

fn validate_attention_preprocess(
    inputs: &[TensorView],
    outputs: &[TensorView],
    contract: AttentionPreprocessContract,
) -> Result<(), OpError> {
    let tensors = [
        (&inputs[0], AttentionPreprocessTensor::PackedQGate),
        (&inputs[1], AttentionPreprocessTensor::K),
        (&inputs[2], AttentionPreprocessTensor::QRawScale),
        (&inputs[3], AttentionPreprocessTensor::KRawScale),
        (&inputs[4], AttentionPreprocessTensor::Positions),
        (&outputs[0], AttentionPreprocessTensor::QOutput),
        (&outputs[1], AttentionPreprocessTensor::GateOutput),
        (&outputs[2], AttentionPreprocessTensor::KOutput),
    ];

    for (tensor, role) in tensors {
        if tensor.shape().contains(&0) {
            return Err(OpError::AttentionPreprocessZeroExtent { tensor: role });
        }
        if !tensor.is_contiguous() {
            return Err(OpError::AttentionPreprocessNonContiguous { tensor: role });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::AttentionPreprocessUnsupportedEncoding {
                tensor: role,
                actual: tensor.encoding(),
            });
        }
        let expected_dtype = if matches!(role, AttentionPreprocessTensor::Positions) {
            DType::I32
        } else {
            DType::Bf16
        };
        if tensor.dtype() != expected_dtype {
            return Err(OpError::AttentionPreprocessUnsupportedDType {
                tensor: role,
                expected: expected_dtype,
                actual: tensor.dtype(),
            });
        }
    }

    let token_count = usize::try_from(contract.token_count()).map_err(|_| {
        OpError::AttentionPreprocessTokenCountOverflow {
            token_count: u64::from(contract.token_count()),
        }
    })?;
    let expected_shapes: [&[usize]; 8] = [
        &[
            token_count,
            AttentionPreprocessContract::Q_HEADS,
            AttentionPreprocessContract::PACKED_Q_GATE_WIDTH,
        ],
        &[
            token_count,
            AttentionPreprocessContract::KV_HEADS,
            AttentionPreprocessContract::HEAD_DIM,
        ],
        &[
            AttentionPreprocessContract::Q_HEADS,
            AttentionPreprocessContract::HEAD_DIM,
        ],
        &[
            AttentionPreprocessContract::KV_HEADS,
            AttentionPreprocessContract::HEAD_DIM,
        ],
        &[token_count],
        &[
            token_count,
            AttentionPreprocessContract::Q_HEADS,
            AttentionPreprocessContract::HEAD_DIM,
        ],
        &[
            token_count,
            AttentionPreprocessContract::Q_HEADS,
            AttentionPreprocessContract::HEAD_DIM,
        ],
        &[
            token_count,
            AttentionPreprocessContract::KV_HEADS,
            AttentionPreprocessContract::HEAD_DIM,
        ],
    ];
    for (index, (tensor, role)) in tensors.into_iter().enumerate() {
        if tensor.shape() != expected_shapes[index] {
            return Err(OpError::AttentionPreprocessShapeMismatch { tensor: role });
        }
    }
    debug_assert_eq!(
        contract.packing(),
        AttentionPreprocessPacking::HeadInterleavedQGate
    );
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
    ElementwiseRankZero {
        kind: SemanticOpKind,
        tensor: ElementwiseTensor,
    },
    ElementwiseZeroExtent {
        kind: SemanticOpKind,
        tensor: ElementwiseTensor,
    },
    ElementwiseNonContiguous {
        kind: SemanticOpKind,
        tensor: ElementwiseTensor,
    },
    ElementwiseUnsupportedDType {
        kind: SemanticOpKind,
        tensor: ElementwiseTensor,
        actual: DType,
    },
    ElementwiseUnsupportedEncoding {
        kind: SemanticOpKind,
        tensor: ElementwiseTensor,
        actual: Encoding,
    },
    SigmoidMulShapeMismatch,
    EmbeddingZeroExtent,
    EmbeddingNonContiguous,
    EmbeddingUnsupportedEncoding,
    EmbeddingTensorContractMismatch,
    EmbeddingOutputShapeMismatch,
    MatmulShapeMismatch,
    MatmulZeroExtent,
    MatmulNonContiguous,
    MatmulUnsupportedEncoding,
    MatmulUnsupportedDType {
        actual: DType,
    },
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
    AttentionPreprocessContractRequired,
    AttentionPreprocessInvalidConfig {
        field: &'static str,
    },
    AttentionPreprocessZeroTokenCount,
    AttentionPreprocessNegativePosition {
        start_position: i64,
    },
    AttentionPreprocessPositionReset {
        mode: AttentionPreprocessPositionMode,
        start_position: i64,
    },
    AttentionPreprocessPositionOverflow,
    AttentionPreprocessPositionOutOfRange {
        last_position: u64,
        max_position_embeddings: u32,
    },
    AttentionPreprocessTokenCountOverflow {
        token_count: u64,
    },
    AttentionPreprocessZeroExtent {
        tensor: AttentionPreprocessTensor,
    },
    AttentionPreprocessNonContiguous {
        tensor: AttentionPreprocessTensor,
    },
    AttentionPreprocessUnsupportedDType {
        tensor: AttentionPreprocessTensor,
        expected: DType,
        actual: DType,
    },
    AttentionPreprocessUnsupportedEncoding {
        tensor: AttentionPreprocessTensor,
        actual: Encoding,
    },
    AttentionPreprocessShapeMismatch {
        tensor: AttentionPreprocessTensor,
    },
    ArgmaxRankMismatch {
        tensor: ArgmaxTensor,
    },
    ArgmaxZeroExtent {
        tensor: ArgmaxTensor,
    },
    ArgmaxNonContiguous {
        tensor: ArgmaxTensor,
    },
    ArgmaxUnsupportedDType {
        tensor: ArgmaxTensor,
        expected: DType,
        actual: DType,
    },
    ArgmaxUnsupportedEncoding {
        tensor: ArgmaxTensor,
        actual: Encoding,
    },
    ArgmaxShapeMismatch,
    ArgmaxVocabTooLarge {
        vocab: usize,
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
            Self::ElementwiseRankZero { kind, tensor } => {
                write!(
                    formatter,
                    "{} {tensor} must have rank at least one",
                    kind.name()
                )
            }
            Self::ElementwiseZeroExtent { kind, tensor } => {
                write!(
                    formatter,
                    "{} {tensor} must not have a zero extent",
                    kind.name()
                )
            }
            Self::ElementwiseNonContiguous { kind, tensor } => {
                write!(
                    formatter,
                    "{} {tensor} must be row-major contiguous",
                    kind.name()
                )
            }
            Self::ElementwiseUnsupportedDType {
                kind,
                tensor,
                actual,
            } => write!(
                formatter,
                "{} {tensor} must be bf16, got {actual}",
                kind.name()
            ),
            Self::ElementwiseUnsupportedEncoding {
                kind,
                tensor,
                actual,
            } => write!(
                formatter,
                "{} {tensor} must use unquantized encoding, got {actual:?}",
                kind.name()
            ),
            Self::SigmoidMulShapeMismatch => formatter.write_str(
                "sigmoid_mul requires identical contiguous BF16 [M,16,256] gate, attention value, and output",
            ),
            Self::EmbeddingZeroExtent => {
                formatter.write_str("embedding tensors must have non-zero extents")
            }
            Self::EmbeddingNonContiguous => {
                formatter.write_str("embedding tensors must be row-major contiguous")
            }
            Self::EmbeddingUnsupportedEncoding => {
                formatter.write_str("embedding tensors must be unquantized")
            }
            Self::EmbeddingTensorContractMismatch => formatter.write_str(
                "embedding requires BF16 [vocab, hidden], I32 [tokens], and BF16 [tokens, hidden]",
            ),
            Self::EmbeddingOutputShapeMismatch => {
                formatter.write_str("embedding output shape does not match token and hidden sizes")
            }
            Self::MatmulShapeMismatch => formatter
                .write_str("matmul requires BF16 activation [M,K], weight [N,K], and output [M,N]"),
            Self::MatmulZeroExtent => {
                formatter.write_str("matmul tensors must have non-zero extents")
            }
            Self::MatmulNonContiguous => {
                formatter.write_str("matmul tensors must be row-major contiguous")
            }
            Self::MatmulUnsupportedEncoding => {
                formatter.write_str("matmul tensors must be unquantized")
            }
            Self::MatmulUnsupportedDType { actual } => {
                write!(formatter, "matmul tensors must be bf16, got {actual}")
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
            Self::AttentionPreprocessContractRequired => {
                formatter.write_str("attention_preprocess requires an explicit contract")
            }
            Self::AttentionPreprocessInvalidConfig { field } => {
                write!(
                    formatter,
                    "attention_preprocess has an invalid {field} configuration"
                )
            }
            Self::AttentionPreprocessZeroTokenCount => {
                formatter.write_str("attention_preprocess token count must be non-zero")
            }
            Self::AttentionPreprocessNegativePosition { start_position } => write!(
                formatter,
                "attention_preprocess start position {start_position} must be non-negative"
            ),
            Self::AttentionPreprocessPositionReset {
                mode,
                start_position,
            } => write!(
                formatter,
                "attention_preprocess {mode:?} cannot start at position {start_position}"
            ),
            Self::AttentionPreprocessPositionOverflow => {
                formatter.write_str("attention_preprocess position range overflowed")
            }
            Self::AttentionPreprocessPositionOutOfRange {
                last_position,
                max_position_embeddings,
            } => write!(
                formatter,
                "attention_preprocess position {last_position} is not below max position {max_position_embeddings}"
            ),
            Self::AttentionPreprocessTokenCountOverflow { token_count } => write!(
                formatter,
                "attention_preprocess token count {token_count} does not fit the contract"
            ),
            Self::AttentionPreprocessZeroExtent { tensor } => {
                write!(
                    formatter,
                    "attention_preprocess {tensor} must not have a zero extent"
                )
            }
            Self::AttentionPreprocessNonContiguous { tensor } => {
                write!(
                    formatter,
                    "attention_preprocess {tensor} must be row-major contiguous"
                )
            }
            Self::AttentionPreprocessUnsupportedDType {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "attention_preprocess {tensor} must use {expected}, got {actual}"
            ),
            Self::AttentionPreprocessUnsupportedEncoding { tensor, actual } => write!(
                formatter,
                "attention_preprocess {tensor} must use unquantized encoding, got {actual:?}"
            ),
            Self::AttentionPreprocessShapeMismatch { tensor } => {
                write!(
                    formatter,
                    "attention_preprocess {tensor} has the wrong rank or shape"
                )
            }
            Self::ArgmaxRankMismatch { tensor } => {
                write!(formatter, "argmax {tensor} has the wrong rank")
            }
            Self::ArgmaxZeroExtent { tensor } => {
                write!(formatter, "argmax {tensor} must have non-zero extents")
            }
            Self::ArgmaxNonContiguous { tensor } => {
                write!(formatter, "argmax {tensor} must be row-major contiguous")
            }
            Self::ArgmaxUnsupportedDType {
                tensor,
                expected,
                actual,
            } => write!(formatter, "argmax {tensor} must use {expected}, got {actual}"),
            Self::ArgmaxUnsupportedEncoding { tensor, actual } => write!(
                formatter,
                "argmax {tensor} must use unquantized encoding, got {actual:?}"
            ),
            Self::ArgmaxShapeMismatch => {
                formatter.write_str("argmax output shape must be [M] for logits shape [M,V]")
            }
            Self::ArgmaxVocabTooLarge { vocab } => {
                write!(formatter, "argmax vocabulary size {vocab} exceeds 1048576")
            }
        }
    }
}

impl std::error::Error for OpError {}
