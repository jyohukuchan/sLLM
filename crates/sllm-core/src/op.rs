use std::fmt;

use crate::{DType, Encoding, Fp8ResidentRepresentation, Fp8ScaleGranularity, TensorView};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SemanticOpKind {
    Copy,
    Add,
    ScalarMul,
    Embedding,
    Matmul,
    /// Fixed Qwen3.5 GDN projection bundle. This is intentionally not a
    /// generic multi-column matmul: four independent BF16 outputs and weight
    /// roles are part of the contract.
    GdnProjectionBundle,
    /// Fixed Qwen3.5 MLP gate/up projection followed by SiLU multiplication.
    /// The two projection outputs remain explicit so the backend may preserve
    /// the baseline graph's tensor boundaries while submitting one M=1 bundle.
    MlpGateUpSiluBundle,
    SiluMul,
    GeluTanhMul,
    SigmoidMul,
    TanhSoftcap,
    RmsNorm,
    /// F32 residual add with BF16-RNE intermediate followed by RMSNorm.
    /// Inputs are residual, addend, and raw scale; outputs retain the BF16
    /// add intermediate and the normalized BF16 result.
    ResidualRmsNorm,
    Rotary,
    CausalAttention,
    AttentionPreprocess,
    Argmax,
    TokenSelect,
    /// Gemma 4 26B-A4B router output selection. The input is already the
    /// separately normalized/root-scaled router projection logits `[M,128]`.
    /// This operation applies stable full softmax, stable top-8 selection, and
    /// top-k renormalization into the canonical route-metadata byte layout.
    MoeRoute,
    /// DeepSeek V4's model-specific top-6 router. Score routing and the first
    /// three main layers' hash routing share one stable descriptor arity while
    /// keeping their distinct numerical contracts explicit.
    DeepSeekV4MoeRoute,
    /// MiniMax M3's fixed sigmoid top-4 router. The F32 selection bias changes
    /// only expert ranking; emitted weights are the selected unbiased sigmoid
    /// values, renormalized and multiplied by the model-fixed scale 2.0.
    MiniMaxM3MoeRoute,
    /// Gemma 4 26B-A4B NVFP4 routed experts. The separately normalized expert
    /// activation and canonical `MoeRoute` metadata are explicit inputs; the
    /// dense shared MLP remains an external graph branch.
    MoeExpert,
    SparseMoe,
    BroadcastAdd,
    BroadcastMul,
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

/// Position payload interpretation for Qwen C3a1 attention preprocessing.
/// The transition mode (prefill/decode) remains separate from this payload
/// mode so explicit absolute positions can be used after logical compaction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttentionPreprocessPositionPayloadModeV1 {
    Contiguous,
    /// Derive `start_position + row` in the device kernel. The position
    /// tensor remains bound for ABI/layout compatibility but is not read.
    DerivedContiguous,
    Explicit,
}

/// Position payload interpretation for split-half rotary.
///
/// `Contiguous` preserves the original contract: the backend validates that
/// the position tensor is exactly `start_position + [0..token_count)`.  The
/// `Explicit` variant is used by context-window compaction; the position
/// tensor carries the original absolute position for each retained token and
/// is validated by the backend before dispatch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RotaryPositionModeV1 {
    Contiguous,
    Explicit,
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

/// Tensor roles in the baseline categorical token-selection operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TokenSelectorTensor {
    Logits,
    AdditiveLogits,
    ValidMask,
    Output,
}

impl fmt::Display for TokenSelectorTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Logits => "logits",
            Self::AdditiveLogits => "additive logits",
            Self::ValidMask => "valid mask",
            Self::Output => "output",
        })
    }
}

/// Fixed numerical and RNG contract for one baseline categorical selection.
///
/// The temperature is stored as bits so this contract remains `Eq`/`Hash` and
/// can be used as a prepared-operation cache key.  The seed and counter are
/// explicit to make replay and multi-choice derivation backend-independent.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TokenSelectorContractV1 {
    vocab_size: u64,
    temperature_bits: u32,
    seed: u64,
    counter: u64,
}

impl TokenSelectorContractV1 {
    pub const MAX_VOCAB_SIZE: u64 = 1_048_576;

    pub fn new(
        vocab_size: u64,
        temperature: f32,
        seed: u64,
        counter: u64,
    ) -> Result<Self, OpError> {
        if vocab_size == 0 || vocab_size > Self::MAX_VOCAB_SIZE {
            return Err(OpError::TokenSelectorVocabOutOfRange { vocab: vocab_size });
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(OpError::TokenSelectorInvalidTemperature {
                bits: temperature.to_bits(),
            });
        }
        Ok(Self {
            vocab_size,
            temperature_bits: temperature.to_bits(),
            seed,
            counter,
        })
    }

    pub const fn vocab_size(self) -> u64 {
        self.vocab_size
    }

    pub const fn vocab(self) -> u64 {
        self.vocab_size
    }

    pub const fn temperature(self) -> f32 {
        f32::from_bits(self.temperature_bits)
    }

    pub const fn temperature_bits(self) -> u32 {
        self.temperature_bits
    }

    pub const fn seed(self) -> u64 {
        self.seed
    }

    pub const fn counter(self) -> u64 {
        self.counter
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RotaryTensor {
    Query,
    Key,
    Positions,
    QueryOutput,
    KeyOutput,
}

impl fmt::Display for RotaryTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Positions => "positions",
            Self::QueryOutput => "query output",
            Self::KeyOutput => "key output",
        })
    }
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
            Self::BroadcastAdd => "broadcast_add",
            Self::BroadcastMul => "broadcast_mul",
            Self::ScalarMul => "scalar_mul",
            Self::Embedding => "embedding",
            Self::Matmul => "matmul",
            Self::GdnProjectionBundle => "gdn_projection_bundle",
            Self::MlpGateUpSiluBundle => "mlp_gate_up_silu_bundle",
            Self::SiluMul => "silu_mul",
            Self::GeluTanhMul => "gelu_tanh_mul",
            Self::SigmoidMul => "sigmoid_mul",
            Self::TanhSoftcap => "tanh_softcap",
            Self::RmsNorm => "rms_norm",
            Self::ResidualRmsNorm => "residual_rms_norm",
            Self::Rotary => "rotary",
            Self::CausalAttention => "causal_attention",
            Self::AttentionPreprocess => "attention_preprocess",
            Self::Argmax => "argmax",
            Self::TokenSelect => "token_select",
            Self::MoeRoute => "moe_route",
            Self::DeepSeekV4MoeRoute => "deepseek_v4_moe_route",
            Self::MiniMaxM3MoeRoute => "minimax_m3_moe_route",
            Self::MoeExpert => "moe_expert",
            Self::SparseMoe => "sparse_moe",
        }
    }

    pub const fn arity(self) -> (usize, usize) {
        match self {
            Self::Copy => (1, 1),
            Self::Add => (2, 1),
            Self::BroadcastAdd => (2, 1),
            Self::BroadcastMul => (2, 1),
            Self::ScalarMul => (2, 1),
            Self::Embedding => (2, 1),
            Self::Matmul => (2, 1),
            Self::GdnProjectionBundle => (5, 4),
            Self::MlpGateUpSiluBundle => (3, 3),
            Self::SiluMul => (2, 1),
            Self::GeluTanhMul => (2, 1),
            Self::SigmoidMul => (2, 1),
            Self::TanhSoftcap => (2, 1),
            Self::RmsNorm => (2, 1),
            Self::ResidualRmsNorm => (3, 2),
            Self::Rotary => (3, 2),
            Self::CausalAttention => (3, 1),
            Self::AttentionPreprocess => (5, 3),
            Self::Argmax => (1, 1),
            Self::TokenSelect => (3, 1),
            Self::MoeRoute => (1, 1),
            Self::DeepSeekV4MoeRoute => (3, 1),
            Self::MiniMaxM3MoeRoute => (2, 1),
            Self::MoeExpert => (3, 1),
            Self::SparseMoe => (3, 1),
        }
    }
}

/// Routing source for the fixed DeepSeek V4 top-6 MoE boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeepSeekV4MoeRouteMode {
    /// Compute `sqrt(softplus(logit))`, use the F32 bias for selection only,
    /// then emit unbiased selected scores as route weights.
    Score,
    /// Consume the fixed six expert IDs supplied by the reviewed hash table.
    Hash,
}

/// Fixed DeepSeek V4 routing contract.
///
/// The routed scale is stored as bits so the descriptor remains an `Eq` and
/// `Hash` cache key. Its constructor rejects non-finite and non-positive values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeepSeekV4MoeRouteContractV1 {
    mode: DeepSeekV4MoeRouteMode,
    renormalize_selected_weights: bool,
    routed_scale_bits: u32,
}

impl DeepSeekV4MoeRouteContractV1 {
    pub const EXPERT_COUNT: usize = 256;
    pub const SELECTED_EXPERT_COUNT: usize = 6;
    pub const MAX_TOKEN_COUNT: usize = 65_536;

    pub fn new(
        mode: DeepSeekV4MoeRouteMode,
        renormalize_selected_weights: bool,
        routed_scale: f32,
    ) -> Result<Self, OpError> {
        if !routed_scale.is_finite() || routed_scale <= 0.0 {
            return Err(OpError::DeepSeekV4MoeRouteInvalidRoutedScale {
                bits: routed_scale.to_bits(),
            });
        }
        Ok(Self {
            mode,
            renormalize_selected_weights,
            routed_scale_bits: routed_scale.to_bits(),
        })
    }

    pub const fn mode(self) -> DeepSeekV4MoeRouteMode {
        self.mode
    }

    pub const fn renormalize_selected_weights(self) -> bool {
        self.renormalize_selected_weights
    }

    pub const fn routed_scale(self) -> f32 {
        f32::from_bits(self.routed_scale_bits)
    }

    pub const fn routed_scale_bits(self) -> u32 {
        self.routed_scale_bits
    }

    /// Returns the canonical byte length for:
    /// `i32 ids[M,6] + f32 weights[M,6] + i32 counts[256] +
    /// i32 offsets[257] + i32 grouped_token[M,6] +
    /// i32 grouped_slot[M,6] + i32 status`.
    pub fn metadata_bytes(token_count: usize) -> Option<usize> {
        deepseek_v4_moe_route_metadata_bytes(token_count)
    }
}

/// Fixed MiniMax M3 score-routing contract.
///
/// There are deliberately no configurable fields: the reviewed model fixes
/// sigmoid routing, 128 experts, stable top-4 selection, selected-weight
/// renormalization, and routed scale 2.0.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MiniMaxM3MoeRouteContractV1;

impl MiniMaxM3MoeRouteContractV1 {
    pub const EXPERT_COUNT: usize = 128;
    pub const SELECTED_EXPERT_COUNT: usize = 4;
    pub const MAX_TOKEN_COUNT: usize = 65_536;
    pub const ROUTED_SCALE: f32 = 2.0;

    pub const fn new() -> Self {
        Self
    }

    pub fn metadata_bytes(token_count: usize) -> Option<usize> {
        minimax_m3_moe_route_metadata_bytes(token_count)
    }
}

/// Model-neutral sparse-MoE semantic boundary. Resident router/expert weights
/// are bound by the graph adapter; this contract fixes only numerical shape
/// and routing semantics shared by all backends.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SparseMoeContract {
    hidden_size: u32,
    expert_count: u32,
    selected_expert_count: u32,
    expert_intermediate_size: u32,
    shared_expert_intermediate_size: u32,
    renormalize_selected_weights: bool,
}

impl SparseMoeContract {
    pub fn new(
        hidden_size: u32,
        expert_count: u32,
        selected_expert_count: u32,
        expert_intermediate_size: u32,
        shared_expert_intermediate_size: u32,
        renormalize_selected_weights: bool,
    ) -> Result<Self, OpError> {
        if hidden_size == 0
            || expert_count == 0
            || expert_count > 256
            || selected_expert_count == 0
            || selected_expert_count > expert_count
            || selected_expert_count > 16
            || expert_intermediate_size == 0
            || shared_expert_intermediate_size == 0
        {
            return Err(OpError::SparseMoeInvalidContract);
        }
        Ok(Self {
            hidden_size,
            expert_count,
            selected_expert_count,
            expert_intermediate_size,
            shared_expert_intermediate_size,
            renormalize_selected_weights,
        })
    }

    pub const fn hidden_size(self) -> u32 {
        self.hidden_size
    }
    pub const fn expert_count(self) -> u32 {
        self.expert_count
    }
    pub const fn selected_expert_count(self) -> u32 {
        self.selected_expert_count
    }
    pub const fn expert_intermediate_size(self) -> u32 {
        self.expert_intermediate_size
    }
    pub const fn shared_expert_intermediate_size(self) -> u32 {
        self.shared_expert_intermediate_size
    }
    pub const fn renormalize_selected_weights(self) -> bool {
        self.renormalize_selected_weights
    }
}

/// The explicit checkpoint scale interpretation used by RMSNorm.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RmsNormScaleMode {
    OffsetOne,
    Direct,
}

/// Tensor roles used in RMSNorm validation errors.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RmsNormTensor {
    Activation,
    RawScale,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResidualRmsNormTensor {
    Residual,
    Addend,
    RawScale,
    ResidualOutput,
    NormalizedOutput,
}

impl fmt::Display for ResidualRmsNormTensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Residual => "residual",
            Self::Addend => "addend",
            Self::RawScale => "raw scale",
            Self::ResidualOutput => "residual output",
            Self::NormalizedOutput => "normalized output",
        })
    }
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
            RmsNormScaleMode::Direct => raw_scale,
        }
    }
}

/// Fixed numerical and aliasing promises made by the fused residual-add and
/// RMSNorm operation. The operation is intentionally narrower than a generic
/// elementwise fusion: all activations and the retained intermediate are BF16
/// storage, while both the add and RMS statistics are accumulated in F32.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResidualRmsNormContract {
    rms_norm: RmsNormContract,
}

impl ResidualRmsNormContract {
    /// Creates the explicit fused contract. No epsilon or scale-mode default
    /// is inferred; they are carried by the nested RMSNorm contract.
    pub fn new(epsilon: f32, scale_mode: RmsNormScaleMode) -> Result<Self, OpError> {
        Ok(Self {
            rms_norm: RmsNormContract::new(epsilon, scale_mode)?,
        })
    }

    pub const fn from_rms_norm(contract: RmsNormContract) -> Self {
        Self { rms_norm: contract }
    }

    pub const fn rms_norm(self) -> RmsNormContract {
        self.rms_norm
    }

    pub const fn epsilon(self) -> RmsNormEpsilon {
        self.rms_norm.epsilon()
    }

    pub const fn scale_mode(self) -> RmsNormScaleMode {
        self.rms_norm.scale_mode()
    }

    pub const fn accumulation_dtype(self) -> DType {
        self.rms_norm.accumulation_dtype()
    }

    pub const fn output_dtype(self) -> DType {
        self.rms_norm.output_dtype()
    }

    pub const fn alias_policy(self) -> RmsNormAliasPolicy {
        self.rms_norm.alias_policy()
    }
}

/// The complete backend-neutral C3a1 semantic contract.
///
/// This descriptor carries the expected absolute position sequence as
/// metadata. `Contiguous` compares the actual I32 position bytes with
/// `start_position .. start_position + token_count`; `DerivedContiguous`
/// creates that sequence in the device kernel and avoids a host readback.
/// Keeping the sequence here makes prefill reset and decode continuation
/// distinct without pretending that a `TensorView` contains payload values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AttentionPreprocessContract {
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    packing: AttentionPreprocessPacking,
    position_mode: AttentionPreprocessPositionMode,
    position_payload_mode: AttentionPreprocessPositionPayloadModeV1,
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
        Self::new_with_position_payload_mode(
            packing,
            position_mode,
            start_position,
            token_count,
            epsilon,
            scale_mode,
            accumulation_dtype,
            output_dtype,
            rotary_dim,
            rope_theta,
            mrope_interleaved,
            mrope_sections,
            max_position_embeddings,
            AttentionPreprocessPositionPayloadModeV1::Contiguous,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_position_payload_mode(
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
        position_payload_mode: AttentionPreprocessPositionPayloadModeV1,
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
        if max_position_embeddings == 0 {
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
            q_heads: Self::Q_HEADS as u32,
            kv_heads: Self::KV_HEADS as u32,
            head_dim: Self::HEAD_DIM as u32,
            packing,
            position_mode,
            position_payload_mode,
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
        Self::new_qwen3_5_with_layout(
            position_mode,
            start_position,
            token_count,
            Self::Q_HEADS as u32,
            Self::KV_HEADS as u32,
            Self::HEAD_DIM as u32,
        )
    }

    pub fn new_qwen3_5_with_layout(
        position_mode: AttentionPreprocessPositionMode,
        start_position: i64,
        token_count: u64,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> Result<Self, OpError> {
        Self::new_qwen3_5_with_layout_and_context(
            position_mode,
            start_position,
            token_count,
            q_heads,
            kv_heads,
            head_dim,
            Self::MAX_POSITION_EMBEDDINGS,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_qwen3_5_with_layout_and_context(
        position_mode: AttentionPreprocessPositionMode,
        start_position: i64,
        token_count: u64,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        runtime_context_tokens: u32,
    ) -> Result<Self, OpError> {
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 || head_dim != 256 {
            return Err(OpError::AttentionPreprocessInvalidConfig {
                field: "head layout",
            });
        }
        let mut contract = Self::new(
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
            runtime_context_tokens,
        )?;
        contract.q_heads = q_heads;
        contract.kv_heads = kv_heads;
        contract.head_dim = head_dim;
        Ok(contract)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_qwen3_5_with_layout_and_context_and_position_payload_mode(
        position_mode: AttentionPreprocessPositionMode,
        start_position: i64,
        token_count: u64,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        runtime_context_tokens: u32,
        position_payload_mode: AttentionPreprocessPositionPayloadModeV1,
    ) -> Result<Self, OpError> {
        let mut contract = Self::new_qwen3_5_with_layout_and_context(
            position_mode,
            start_position,
            token_count,
            q_heads,
            kv_heads,
            head_dim,
            runtime_context_tokens,
        )?;
        contract.position_payload_mode = position_payload_mode;
        Ok(contract)
    }

    pub const fn packing(self) -> AttentionPreprocessPacking {
        self.packing
    }

    pub const fn q_heads(self) -> u32 {
        self.q_heads
    }

    pub const fn kv_heads(self) -> u32 {
        self.kv_heads
    }

    pub const fn head_dim(self) -> u32 {
        self.head_dim
    }

    pub const fn position_mode(self) -> AttentionPreprocessPositionMode {
        self.position_mode
    }

    pub const fn position_payload_mode(self) -> AttentionPreprocessPositionPayloadModeV1 {
        self.position_payload_mode
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

/// A backend-neutral split-half RoPE contract.
///
/// Frequencies are `theta^(-2*i/head_dim)`. The first `rotary_dim / 2`
/// dimensions are paired with the dimensions beginning at `head_dim / 2`;
/// dimensions outside those two active ranges are copied unchanged. This
/// represents both Gemma 4 sliding RoPE (`256/256`) and proportional full
/// RoPE (`512/128`) without treating either as Qwen mRoPE.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SplitHalfRotaryContract {
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta_bits: u32,
    start_position: u32,
    token_count: u32,
    max_position_embeddings: u32,
    position_mode: RotaryPositionModeV1,
    accumulation_dtype: DType,
    output_dtype: DType,
}

impl SplitHalfRotaryContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        theta: f32,
        start_position: u64,
        token_count: u64,
        max_position_embeddings: u32,
    ) -> Result<Self, OpError> {
        Self::new_with_position_mode(
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            theta,
            start_position,
            token_count,
            max_position_embeddings,
            RotaryPositionModeV1::Contiguous,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_position_mode(
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        theta: f32,
        start_position: u64,
        token_count: u64,
        max_position_embeddings: u32,
        position_mode: RotaryPositionModeV1,
    ) -> Result<Self, OpError> {
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(OpError::RotaryInvalidConfig {
                field: "head count",
            });
        }
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(OpError::RotaryInvalidConfig {
                field: "head dimension",
            });
        }
        if rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > head_dim {
            return Err(OpError::RotaryInvalidConfig {
                field: "rotary dimension",
            });
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(OpError::RotaryInvalidConfig { field: "theta" });
        }
        if token_count == 0 {
            return Err(OpError::RotaryInvalidConfig {
                field: "token count",
            });
        }
        if max_position_embeddings == 0 {
            return Err(OpError::RotaryInvalidConfig {
                field: "max position embeddings",
            });
        }
        let end_position = start_position
            .checked_add(token_count)
            .ok_or(OpError::RotaryPositionOverflow)?;
        if end_position > u64::from(max_position_embeddings) {
            return Err(OpError::RotaryPositionOutOfRange {
                last_position: end_position - 1,
                max_position_embeddings,
            });
        }
        let start_position =
            u32::try_from(start_position).map_err(|_| OpError::RotaryPositionOutOfRange {
                last_position: start_position,
                max_position_embeddings,
            })?;
        let token_count = u32::try_from(token_count).map_err(|_| OpError::RotaryInvalidConfig {
            field: "token count",
        })?;
        Ok(Self {
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            theta_bits: theta.to_bits(),
            start_position,
            token_count,
            max_position_embeddings,
            position_mode,
            accumulation_dtype: DType::F32,
            output_dtype: DType::Bf16,
        })
    }

    pub const fn q_heads(self) -> u32 {
        self.q_heads
    }

    pub const fn kv_heads(self) -> u32 {
        self.kv_heads
    }

    pub const fn head_dim(self) -> u32 {
        self.head_dim
    }

    pub const fn rotary_dim(self) -> u32 {
        self.rotary_dim
    }

    pub const fn theta(self) -> f32 {
        f32::from_bits(self.theta_bits)
    }

    pub const fn theta_bits(self) -> u32 {
        self.theta_bits
    }

    pub const fn start_position(self) -> u32 {
        self.start_position
    }

    pub const fn token_count(self) -> u32 {
        self.token_count
    }

    pub const fn max_position_embeddings(self) -> u32 {
        self.max_position_embeddings
    }

    pub const fn position_mode(self) -> RotaryPositionModeV1 {
        self.position_mode
    }

    pub const fn accumulation_dtype(self) -> DType {
        self.accumulation_dtype
    }

    pub const fn output_dtype(self) -> DType {
        self.output_dtype
    }
}

/// Model-neutral BF16 GQA causal-attention semantics used by Gemma 4.
///
/// A zero `sliding_window()` means full causal attention. A non-zero value is
/// the inclusive token count ending at the current query, so the first key is
/// `max(0, query_position + 1 - sliding_window)`. Scores use the explicit
/// multiplicative scale before FP32 softmax.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WindowedCausalAttentionContract {
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    start_position: u64,
    query_count: u32,
    expected_kv_length: u64,
    sliding_window: u64,
    scaling_bits: u32,
    accumulation_dtype: DType,
    output_dtype: DType,
}

impl WindowedCausalAttentionContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        start_position: u64,
        query_count: u64,
        expected_kv_length: u64,
        sliding_window: Option<u64>,
        scaling: f32,
    ) -> Result<Self, OpError> {
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(OpError::CausalAttentionInvalidConfig {
                field: "head count",
            });
        }
        if head_dim == 0 {
            return Err(OpError::CausalAttentionInvalidConfig {
                field: "head dimension",
            });
        }
        if query_count == 0 {
            return Err(OpError::CausalAttentionInvalidConfig {
                field: "query count",
            });
        }
        if matches!(sliding_window, Some(0)) {
            return Err(OpError::CausalAttentionInvalidConfig {
                field: "sliding window",
            });
        }
        if !scaling.is_finite() || scaling <= 0.0 {
            return Err(OpError::CausalAttentionInvalidConfig { field: "scaling" });
        }
        let end = start_position
            .checked_add(query_count)
            .ok_or(OpError::CausalAttentionLengthOverflow)?;
        if end != expected_kv_length {
            return Err(OpError::CausalAttentionLengthMismatch {
                expected: end,
                actual: expected_kv_length,
            });
        }
        let query_count =
            u32::try_from(query_count).map_err(|_| OpError::CausalAttentionInvalidConfig {
                field: "query count",
            })?;
        Ok(Self {
            q_heads,
            kv_heads,
            head_dim,
            start_position,
            query_count,
            expected_kv_length,
            sliding_window: sliding_window.unwrap_or(0),
            scaling_bits: scaling.to_bits(),
            accumulation_dtype: DType::F32,
            output_dtype: DType::Bf16,
        })
    }

    pub const fn q_heads(self) -> u32 {
        self.q_heads
    }

    pub const fn kv_heads(self) -> u32 {
        self.kv_heads
    }

    pub const fn head_dim(self) -> u32 {
        self.head_dim
    }

    pub const fn start_position(self) -> u64 {
        self.start_position
    }

    pub const fn query_count(self) -> u32 {
        self.query_count
    }

    pub const fn expected_kv_length(self) -> u64 {
        self.expected_kv_length
    }

    pub const fn sliding_window(self) -> Option<u64> {
        if self.sliding_window == 0 {
            None
        } else {
            Some(self.sliding_window)
        }
    }

    pub const fn scaling(self) -> f32 {
        f32::from_bits(self.scaling_bits)
    }

    pub const fn scaling_bits(self) -> u32 {
        self.scaling_bits
    }

    pub const fn accumulation_dtype(self) -> DType {
        self.accumulation_dtype
    }

    pub const fn output_dtype(self) -> DType {
        self.output_dtype
    }
}

/// Fixed numerical contract for the Qwen3.5 GDN projection bundle. Inputs are
/// activation, qkv, z, b and a weights; outputs are independent qkv/z/b/a
/// BF16 tensors. The role widths intentionally remain fixed to prevent this
/// boundary from becoming a generic multi-column matmul.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GdnProjectionBundleContractV1 {
    hidden_size: u32,
    qkv_width: u32,
    z_width: u32,
    gate_width: u32,
}

impl GdnProjectionBundleContractV1 {
    pub const HIDDEN_SIZE: u32 = 2_560;
    pub const QKV_WIDTH: u32 = 8_192;
    pub const Z_WIDTH: u32 = 4_096;
    pub const GATE_WIDTH: u32 = 32;

    pub fn qwen35() -> Self {
        Self {
            hidden_size: Self::HIDDEN_SIZE,
            qkv_width: Self::QKV_WIDTH,
            z_width: Self::Z_WIDTH,
            gate_width: Self::GATE_WIDTH,
        }
    }

    pub const fn hidden_size(self) -> u32 {
        self.hidden_size
    }
    pub const fn qkv_width(self) -> u32 {
        self.qkv_width
    }
    pub const fn z_width(self) -> u32 {
        self.z_width
    }
    pub const fn gate_width(self) -> u32 {
        self.gate_width
    }
}

/// Fixed numerical contract for the Qwen3.5 dense MLP gate/up/SiLU bundle.
/// Inputs are activation, gate weight, and up weight; outputs retain the gate,
/// up, and SiLU-multiplication tensors from the baseline graph.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MlpGateUpSiluBundleContractV1 {
    hidden_size: u32,
    intermediate_size: u32,
}

impl MlpGateUpSiluBundleContractV1 {
    pub const HIDDEN_SIZE: u32 = 2_560;
    pub const INTERMEDIATE_SIZE: u32 = 9_216;

    pub fn qwen35() -> Self {
        Self {
            hidden_size: Self::HIDDEN_SIZE,
            intermediate_size: Self::INTERMEDIATE_SIZE,
        }
    }

    pub const fn hidden_size(self) -> u32 {
        self.hidden_size
    }

    pub const fn intermediate_size(self) -> u32 {
        self.intermediate_size
    }
}

/// A backend-independent operation contract containing only tensor metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SemanticOpDescriptor {
    kind: SemanticOpKind,
    inputs: Vec<TensorView>,
    outputs: Vec<TensorView>,
    rms_norm_contract: Option<RmsNormContract>,
    residual_rms_norm_contract: Option<ResidualRmsNormContract>,
    rotary_contract: Option<SplitHalfRotaryContract>,
    causal_attention_contract: Option<WindowedCausalAttentionContract>,
    attention_preprocess_contract: Option<AttentionPreprocessContract>,
    token_selector_contract: Option<TokenSelectorContractV1>,
    sparse_moe_contract: Option<SparseMoeContract>,
    deepseek_v4_moe_route_contract: Option<DeepSeekV4MoeRouteContractV1>,
    minimax_m3_moe_route_contract: Option<MiniMaxM3MoeRouteContractV1>,
    gdn_projection_bundle_contract: Option<GdnProjectionBundleContractV1>,
    mlp_gate_up_silu_bundle_contract: Option<MlpGateUpSiluBundleContractV1>,
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
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
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
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Creates the fused residual-add/RMSNorm descriptor. Output zero is the
    /// BF16-RNE residual intermediate and output one is the normalized BF16
    /// result. Keeping both outputs explicit preserves the baseline graph's
    /// tensor and access contracts.
    pub fn new_residual_rms_norm(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        epsilon: f32,
        scale_mode: RmsNormScaleMode,
    ) -> Result<Self, OpError> {
        let contract = ResidualRmsNormContract::new(epsilon, scale_mode)?;
        Self::new_residual_rms_norm_with_contract(inputs, outputs, contract)
    }

    pub fn new_residual_rms_norm_with_contract(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: ResidualRmsNormContract,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::ResidualRmsNorm,
            inputs,
            outputs,
            rms_norm_contract: None,
            residual_rms_norm_contract: Some(contract),
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
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
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: Some(contract),
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn new_rotary(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: SplitHalfRotaryContract,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::Rotary,
            inputs,
            outputs,
            rms_norm_contract: None,
            residual_rms_norm_contract: None,
            rotary_contract: Some(contract),
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn new_causal_attention(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: WindowedCausalAttentionContract,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::CausalAttention,
            inputs,
            outputs,
            rms_norm_contract: None,
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: Some(contract),
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Creates a categorical token-selection descriptor with an explicit
    /// vocabulary, temperature, seed, and counter contract.
    pub fn new_token_select(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: TokenSelectorContractV1,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::TokenSelect,
            inputs,
            outputs,
            rms_norm_contract: None,
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: Some(contract),
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Alias retaining the noun used by the HIP public API.
    pub fn new_token_selector(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: TokenSelectorContractV1,
    ) -> Result<Self, OpError> {
        Self::new_token_select(inputs, outputs, contract)
    }

    pub fn new_sparse_moe(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: SparseMoeContract,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::SparseMoe,
            inputs,
            outputs,
            rms_norm_contract: None,
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: Some(contract),
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Creates the DeepSeek V4 model-specific top-6 routing descriptor.
    ///
    /// `TensorView` supports zero extents, so this operation preserves one
    /// stable three-input arity with typed, storage-free placeholders instead
    /// of making backend binding depend on the route mode. Score mode binds
    /// `[logits, bias, I32[0,6]]`; hash mode binds
    /// `[logits, F32[0], hash_ids]`.
    pub fn new_deepseek_v4_moe_route(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: DeepSeekV4MoeRouteContractV1,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::DeepSeekV4MoeRoute,
            inputs,
            outputs,
            rms_norm_contract: None,
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: Some(contract),
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Creates the fixed MiniMax M3 top-4 sigmoid routing descriptor.
    pub fn new_minimax_m3_moe_route(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: MiniMaxM3MoeRouteContractV1,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::MiniMaxM3MoeRoute,
            inputs,
            outputs,
            rms_norm_contract: None,
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: Some(contract),
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn new_gdn_projection_bundle(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: GdnProjectionBundleContractV1,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::GdnProjectionBundle,
            inputs,
            outputs,
            rms_norm_contract: None,
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: Some(contract),
            mlp_gate_up_silu_bundle_contract: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn new_mlp_gate_up_silu_bundle(
        inputs: Vec<TensorView>,
        outputs: Vec<TensorView>,
        contract: MlpGateUpSiluBundleContractV1,
    ) -> Result<Self, OpError> {
        let descriptor = Self {
            kind: SemanticOpKind::MlpGateUpSiluBundle,
            inputs,
            outputs,
            rms_norm_contract: None,
            residual_rms_norm_contract: None,
            rotary_contract: None,
            causal_attention_contract: None,
            attention_preprocess_contract: None,
            token_selector_contract: None,
            sparse_moe_contract: None,
            deepseek_v4_moe_route_contract: None,
            minimax_m3_moe_route_contract: None,
            gdn_projection_bundle_contract: None,
            mlp_gate_up_silu_bundle_contract: Some(contract),
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

    pub const fn residual_rms_norm_contract(&self) -> Option<ResidualRmsNormContract> {
        self.residual_rms_norm_contract
    }

    pub const fn rotary_contract(&self) -> Option<SplitHalfRotaryContract> {
        self.rotary_contract
    }

    pub const fn causal_attention_contract(&self) -> Option<WindowedCausalAttentionContract> {
        self.causal_attention_contract
    }

    pub const fn attention_preprocess_contract(&self) -> Option<AttentionPreprocessContract> {
        self.attention_preprocess_contract
    }

    pub const fn token_selector_contract(&self) -> Option<TokenSelectorContractV1> {
        self.token_selector_contract
    }

    pub const fn sparse_moe_contract(&self) -> Option<SparseMoeContract> {
        self.sparse_moe_contract
    }

    pub const fn deepseek_v4_moe_route_contract(&self) -> Option<DeepSeekV4MoeRouteContractV1> {
        self.deepseek_v4_moe_route_contract
    }

    pub const fn minimax_m3_moe_route_contract(&self) -> Option<MiniMaxM3MoeRouteContractV1> {
        self.minimax_m3_moe_route_contract
    }

    pub const fn gdn_projection_bundle_contract(&self) -> Option<GdnProjectionBundleContractV1> {
        self.gdn_projection_bundle_contract
    }

    pub const fn mlp_gate_up_silu_bundle_contract(&self) -> Option<MlpGateUpSiluBundleContractV1> {
        self.mlp_gate_up_silu_bundle_contract
    }

    /// Returns the zero-copy rank-2 view consumed by the existing `o_proj`
    /// matmul path. Only the validated C3c sigmoid output gate has this
    /// handoff: `[M, H, 256]` is the same contiguous storage as `[M, H * 256]`.
    pub fn sigmoid_mul_o_proj_input_view(&self) -> Option<TensorView> {
        if self.kind != SemanticOpKind::SigmoidMul {
            return None;
        }
        let output = &self.outputs[0];
        let m = output.shape()[0];
        let width = output.shape()[1].checked_mul(output.shape()[2])?;
        Some(
            TensorView::new(
                DType::Bf16,
                Encoding::Unquantized,
                &[m, width],
                &[width, 1],
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
            SemanticOpKind::BroadcastAdd => {
                validate_broadcast_add(&self.inputs, &self.outputs)?;
            }
            SemanticOpKind::BroadcastMul => {
                validate_broadcast_mul(&self.inputs, &self.outputs)?;
            }
            SemanticOpKind::ScalarMul => {
                validate_baseline_elementwise(self.kind, &self.inputs, &self.outputs)?;
            }
            SemanticOpKind::Embedding => {
                validate_embedding(&self.inputs, &self.outputs)?;
            }
            SemanticOpKind::Matmul => {
                validate_matmul(&self.inputs, &self.outputs)?;
            }
            SemanticOpKind::GdnProjectionBundle => {
                let contract = self
                    .gdn_projection_bundle_contract
                    .ok_or(OpError::GdnProjectionBundleContractRequired)?;
                validate_gdn_projection_bundle(&self.inputs, &self.outputs, contract)?;
            }
            SemanticOpKind::MlpGateUpSiluBundle => {
                let contract = self
                    .mlp_gate_up_silu_bundle_contract
                    .ok_or(OpError::MlpGateUpSiluBundleContractRequired)?;
                validate_mlp_gate_up_silu_bundle(&self.inputs, &self.outputs, contract)?;
            }
            SemanticOpKind::SiluMul => {
                validate_baseline_elementwise(self.kind, &self.inputs, &self.outputs)?;
            }
            SemanticOpKind::GeluTanhMul => {
                validate_baseline_elementwise(self.kind, &self.inputs, &self.outputs)?;
            }
            SemanticOpKind::SigmoidMul => {
                validate_baseline_elementwise(self.kind, &self.inputs, &self.outputs)?;
            }
            SemanticOpKind::TanhSoftcap => {
                validate_baseline_elementwise(self.kind, &self.inputs, &self.outputs)?;
            }
            SemanticOpKind::RmsNorm => {
                let contract = self
                    .rms_norm_contract
                    .ok_or(OpError::RmsNormContractRequired)?;
                validate_rms_norm(&self.inputs, &self.outputs, contract)?;
            }
            SemanticOpKind::ResidualRmsNorm => {
                let contract = self
                    .residual_rms_norm_contract
                    .ok_or(OpError::ResidualRmsNormContractRequired)?;
                validate_residual_rms_norm(&self.inputs, &self.outputs, contract)?;
            }
            SemanticOpKind::Rotary => {
                let contract = self
                    .rotary_contract
                    .ok_or(OpError::RotaryContractRequired)?;
                validate_rotary(&self.inputs, &self.outputs, contract)?;
            }
            SemanticOpKind::CausalAttention => {
                let contract = self
                    .causal_attention_contract
                    .ok_or(OpError::CausalAttentionContractRequired)?;
                validate_causal_attention(&self.inputs, &self.outputs, contract)?;
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
            SemanticOpKind::TokenSelect => {
                let contract = self
                    .token_selector_contract
                    .ok_or(OpError::TokenSelectorContractRequired)?;
                validate_token_selector(&self.inputs, &self.outputs, contract)?;
            }
            SemanticOpKind::MoeRoute => {
                validate_gemma4_moe_route(&self.inputs, &self.outputs)?;
            }
            SemanticOpKind::DeepSeekV4MoeRoute => {
                let contract = self
                    .deepseek_v4_moe_route_contract
                    .ok_or(OpError::DeepSeekV4MoeRouteContractRequired)?;
                validate_deepseek_v4_moe_route(&self.inputs, &self.outputs, contract)?;
            }
            SemanticOpKind::MiniMaxM3MoeRoute => {
                self.minimax_m3_moe_route_contract
                    .ok_or(OpError::MiniMaxM3MoeRouteContractRequired)?;
                validate_minimax_m3_moe_route(&self.inputs, &self.outputs)?;
            }
            SemanticOpKind::MoeExpert => {
                validate_gemma4_moe_expert(&self.inputs, &self.outputs)?;
            }
            SemanticOpKind::SparseMoe => {
                let contract = self
                    .sparse_moe_contract
                    .ok_or(OpError::SparseMoeContractRequired)?;
                validate_sparse_moe(&self.inputs, &self.outputs, contract)?;
            }
        }
        Ok(())
    }
}

fn validate_gdn_projection_bundle(
    inputs: &[TensorView],
    outputs: &[TensorView],
    contract: GdnProjectionBundleContractV1,
) -> Result<(), OpError> {
    let expected = [
        contract.qkv_width as usize,
        contract.z_width as usize,
        contract.gate_width as usize,
        contract.gate_width as usize,
    ];
    let mut tensors = Vec::with_capacity(9);
    tensors.extend(inputs.iter());
    tensors.extend(outputs.iter());
    for tensor in tensors {
        if tensor.shape().is_empty() || tensor.shape().contains(&0) {
            return Err(OpError::GdnProjectionBundleZeroExtent);
        }
        if !tensor.is_contiguous() {
            return Err(OpError::GdnProjectionBundleNonContiguous);
        }
        if tensor.dtype() != DType::Bf16 {
            return Err(OpError::GdnProjectionBundleUnsupportedDType {
                actual: tensor.dtype(),
            });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::GdnProjectionBundleUnsupportedEncoding {
                actual: tensor.encoding(),
            });
        }
    }
    let activation = &inputs[0];
    if activation.shape().len() != 2 || activation.shape()[1] != contract.hidden_size as usize {
        return Err(OpError::GdnProjectionBundleShapeMismatch);
    }
    let m = activation.shape()[0];
    for (output, width) in outputs.iter().zip(expected) {
        if output.shape() != [m, width] {
            return Err(OpError::GdnProjectionBundleShapeMismatch);
        }
    }
    for (weight, width) in inputs[1..].iter().zip(expected) {
        if weight.shape() != [width, contract.hidden_size as usize] {
            return Err(OpError::GdnProjectionBundleShapeMismatch);
        }
    }
    Ok(())
}

fn validate_mlp_gate_up_silu_bundle(
    inputs: &[TensorView],
    outputs: &[TensorView],
    contract: MlpGateUpSiluBundleContractV1,
) -> Result<(), OpError> {
    let mut tensors = Vec::with_capacity(6);
    tensors.extend(inputs.iter());
    tensors.extend(outputs.iter());
    for tensor in tensors {
        if tensor.shape().is_empty() || tensor.shape().contains(&0) {
            return Err(OpError::MlpGateUpSiluBundleZeroExtent);
        }
        if !tensor.is_contiguous() {
            return Err(OpError::MlpGateUpSiluBundleNonContiguous);
        }
        if tensor.dtype() != DType::Bf16 {
            return Err(OpError::MlpGateUpSiluBundleUnsupportedDType {
                actual: tensor.dtype(),
            });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::MlpGateUpSiluBundleUnsupportedEncoding {
                actual: tensor.encoding(),
            });
        }
    }
    let activation = &inputs[0];
    if activation.shape().len() != 2 || activation.shape()[1] != contract.hidden_size as usize {
        return Err(OpError::MlpGateUpSiluBundleShapeMismatch);
    }
    let m = activation.shape()[0];
    for output in outputs {
        if output.shape() != [m, contract.intermediate_size as usize] {
            return Err(OpError::MlpGateUpSiluBundleShapeMismatch);
        }
    }
    for weight in &inputs[1..] {
        if weight.shape()
            != [
                contract.intermediate_size as usize,
                contract.hidden_size as usize,
            ]
        {
            return Err(OpError::MlpGateUpSiluBundleShapeMismatch);
        }
    }
    Ok(())
}

fn validate_sparse_moe(
    inputs: &[TensorView],
    outputs: &[TensorView],
    contract: SparseMoeContract,
) -> Result<(), OpError> {
    let input = &inputs[0];
    let router = &inputs[1];
    let layer_blob = &inputs[2];
    let output = &outputs[0];
    if input.shape().len() != 2
        || input.shape() != output.shape()
        || input.shape()[1] != contract.hidden_size as usize
        || input.shape().contains(&0)
        || !input.is_contiguous()
        || !output.is_contiguous()
        || input.dtype() != DType::Bf16
        || output.dtype() != DType::Bf16
        || input.encoding() != Encoding::Unquantized
        || output.encoding() != Encoding::Unquantized
        || router.shape()
            != [
                contract.expert_count as usize,
                contract.hidden_size as usize,
            ]
        || router.dtype() != DType::Bf16
        || router.encoding() != Encoding::Unquantized
        || !router.is_contiguous()
        || layer_blob.shape() != [434_114_560]
        || layer_blob.dtype() != DType::U8
        || layer_blob.encoding() != Encoding::Unquantized
        || !layer_blob.is_contiguous()
    {
        return Err(OpError::SparseMoeTensorContractMismatch);
    }
    Ok(())
}

fn deepseek_v4_moe_route_metadata_bytes(token_count: usize) -> Option<usize> {
    let routed_elements =
        token_count.checked_mul(DeepSeekV4MoeRouteContractV1::SELECTED_EXPERT_COUNT)?;
    let ids_bytes = routed_elements.checked_mul(std::mem::size_of::<i32>())?;
    let weights_bytes = routed_elements.checked_mul(std::mem::size_of::<f32>())?;
    let counts_bytes =
        DeepSeekV4MoeRouteContractV1::EXPERT_COUNT.checked_mul(std::mem::size_of::<i32>())?;
    let offsets_bytes = DeepSeekV4MoeRouteContractV1::EXPERT_COUNT
        .checked_add(1)?
        .checked_mul(std::mem::size_of::<i32>())?;
    let grouped_token_bytes = routed_elements.checked_mul(std::mem::size_of::<i32>())?;
    let grouped_slot_bytes = routed_elements.checked_mul(std::mem::size_of::<i32>())?;
    ids_bytes
        .checked_add(weights_bytes)?
        .checked_add(counts_bytes)?
        .checked_add(offsets_bytes)?
        .checked_add(grouped_token_bytes)?
        .checked_add(grouped_slot_bytes)?
        .checked_add(std::mem::size_of::<i32>())
}

fn minimax_m3_moe_route_metadata_bytes(token_count: usize) -> Option<usize> {
    let routed_elements =
        token_count.checked_mul(MiniMaxM3MoeRouteContractV1::SELECTED_EXPERT_COUNT)?;
    let routed_plane_bytes = routed_elements.checked_mul(std::mem::size_of::<i32>())?;
    let counts_bytes =
        MiniMaxM3MoeRouteContractV1::EXPERT_COUNT.checked_mul(std::mem::size_of::<i32>())?;
    let offsets_bytes = MiniMaxM3MoeRouteContractV1::EXPERT_COUNT
        .checked_add(1)?
        .checked_mul(std::mem::size_of::<i32>())?;
    routed_plane_bytes
        .checked_mul(4)?
        .checked_add(counts_bytes)?
        .checked_add(offsets_bytes)?
        .checked_add(std::mem::size_of::<i32>())
}

fn validate_minimax_m3_moe_route(
    inputs: &[TensorView],
    outputs: &[TensorView],
) -> Result<(), OpError> {
    let logits = &inputs[0];
    let bias = &inputs[1];
    let metadata = &outputs[0];
    if logits.shape().len() != 2
        || logits.shape()[1] != MiniMaxM3MoeRouteContractV1::EXPERT_COUNT
        || logits.dtype() != DType::F32
        || logits.encoding() != Encoding::Unquantized
        || !logits.is_contiguous()
    {
        return Err(OpError::MiniMaxM3MoeRouteLogitsContractMismatch);
    }
    let token_count = logits.shape()[0];
    if token_count == 0 || token_count > MiniMaxM3MoeRouteContractV1::MAX_TOKEN_COUNT {
        return Err(OpError::MiniMaxM3MoeRouteTokenCountOutOfRange { token_count });
    }
    if bias.shape() != [MiniMaxM3MoeRouteContractV1::EXPERT_COUNT]
        || bias.dtype() != DType::F32
        || bias.encoding() != Encoding::Unquantized
        || !bias.is_contiguous()
    {
        return Err(OpError::MiniMaxM3MoeRouteBiasContractMismatch);
    }
    let metadata_bytes = minimax_m3_moe_route_metadata_bytes(token_count)
        .ok_or(OpError::MiniMaxM3MoeRouteMetadataSizeOverflow { token_count })?;
    if metadata.shape() != [metadata_bytes]
        || metadata.dtype() != DType::U8
        || metadata.encoding() != Encoding::Unquantized
        || !metadata.is_contiguous()
    {
        return Err(OpError::MiniMaxM3MoeRouteOutputContractMismatch);
    }
    Ok(())
}

fn validate_deepseek_v4_moe_route(
    inputs: &[TensorView],
    outputs: &[TensorView],
    contract: DeepSeekV4MoeRouteContractV1,
) -> Result<(), OpError> {
    let logits = &inputs[0];
    let bias = &inputs[1];
    let hash_ids = &inputs[2];
    let metadata = &outputs[0];

    if !contract.routed_scale().is_finite() || contract.routed_scale() <= 0.0 {
        return Err(OpError::DeepSeekV4MoeRouteInvalidRoutedScale {
            bits: contract.routed_scale_bits(),
        });
    }
    if logits.shape().len() != 2
        || logits.shape()[1] != DeepSeekV4MoeRouteContractV1::EXPERT_COUNT
        || logits.dtype() != DType::Bf16
        || logits.encoding() != Encoding::Unquantized
        || !logits.is_contiguous()
    {
        return Err(OpError::DeepSeekV4MoeRouteLogitsContractMismatch);
    }

    let token_count = logits.shape()[0];
    if token_count == 0 || token_count > DeepSeekV4MoeRouteContractV1::MAX_TOKEN_COUNT {
        return Err(OpError::DeepSeekV4MoeRouteTokenCountOutOfRange { token_count });
    }

    match contract.mode() {
        DeepSeekV4MoeRouteMode::Score => {
            if bias.shape() != [DeepSeekV4MoeRouteContractV1::EXPERT_COUNT]
                || bias.dtype() != DType::F32
                || bias.encoding() != Encoding::Unquantized
                || !bias.is_contiguous()
            {
                return Err(OpError::DeepSeekV4MoeRouteBiasContractMismatch {
                    mode: contract.mode(),
                });
            }
            // This typed zero-extent view consumes no backing storage and
            // makes an accidentally live hash table fail validation.
            if hash_ids.shape() != [0, DeepSeekV4MoeRouteContractV1::SELECTED_EXPERT_COUNT]
                || hash_ids.dtype() != DType::I32
                || hash_ids.encoding() != Encoding::Unquantized
                || !hash_ids.is_contiguous()
            {
                return Err(OpError::DeepSeekV4MoeRouteHashContractMismatch {
                    mode: contract.mode(),
                });
            }
        }
        DeepSeekV4MoeRouteMode::Hash => {
            // The inactive F32 bias role is a canonical storage-free [0]
            // placeholder, preserving a mode-independent descriptor arity.
            if bias.shape() != [0]
                || bias.dtype() != DType::F32
                || bias.encoding() != Encoding::Unquantized
                || !bias.is_contiguous()
            {
                return Err(OpError::DeepSeekV4MoeRouteBiasContractMismatch {
                    mode: contract.mode(),
                });
            }
            if hash_ids.shape()
                != [
                    token_count,
                    DeepSeekV4MoeRouteContractV1::SELECTED_EXPERT_COUNT,
                ]
                || hash_ids.dtype() != DType::I32
                || hash_ids.encoding() != Encoding::Unquantized
                || !hash_ids.is_contiguous()
            {
                return Err(OpError::DeepSeekV4MoeRouteHashContractMismatch {
                    mode: contract.mode(),
                });
            }
        }
    }

    let metadata_bytes = deepseek_v4_moe_route_metadata_bytes(token_count)
        .ok_or(OpError::DeepSeekV4MoeRouteMetadataSizeOverflow { token_count })?;
    if metadata.shape() != [metadata_bytes]
        || metadata.dtype() != DType::U8
        || metadata.encoding() != Encoding::Unquantized
        || !metadata.is_contiguous()
    {
        return Err(OpError::DeepSeekV4MoeRouteOutputContractMismatch);
    }
    Ok(())
}

const GEMMA4_MOE_EXPERT_COUNT: usize = 128;
const GEMMA4_MOE_SELECTED_EXPERT_COUNT: usize = 8;
const GEMMA4_MOE_HIDDEN_SIZE: usize = 2_816;
const GEMMA4_MOE_LAYER_BLOB_BYTES: usize = 428_215_552;
const GEMMA4_MOE_MAX_TOKENS: usize = 65_536;

fn gemma4_moe_route_metadata_bytes(token_count: usize) -> Option<usize> {
    token_count
        .checked_mul(GEMMA4_MOE_SELECTED_EXPERT_COUNT)?
        .checked_mul(16)?
        .checked_add(GEMMA4_MOE_EXPERT_COUNT.checked_mul(4)?)?
        .checked_add((GEMMA4_MOE_EXPERT_COUNT + 1).checked_mul(4)?)?
        .checked_add(4)
}

fn validate_gemma4_moe_route(inputs: &[TensorView], outputs: &[TensorView]) -> Result<(), OpError> {
    let logits = &inputs[0];
    let metadata = &outputs[0];
    let token_count = logits.shape().first().copied().unwrap_or(0);
    let metadata_bytes = gemma4_moe_route_metadata_bytes(token_count)
        .ok_or(OpError::MoeRouteTensorContractMismatch)?;
    if logits.shape() != [token_count, GEMMA4_MOE_EXPERT_COUNT]
        || token_count == 0
        || token_count > GEMMA4_MOE_MAX_TOKENS
        || logits.dtype() != DType::Bf16
        || logits.encoding() != Encoding::Unquantized
        || !logits.is_contiguous()
        || metadata.shape() != [metadata_bytes]
        || metadata.dtype() != DType::U8
        || metadata.encoding() != Encoding::Unquantized
        || !metadata.is_contiguous()
    {
        return Err(OpError::MoeRouteTensorContractMismatch);
    }
    Ok(())
}

fn validate_gemma4_moe_expert(
    inputs: &[TensorView],
    outputs: &[TensorView],
) -> Result<(), OpError> {
    let hidden = &inputs[0];
    let metadata = &inputs[1];
    let layer_blob = &inputs[2];
    let output = &outputs[0];
    let token_count = hidden.shape().first().copied().unwrap_or(0);
    let metadata_bytes = gemma4_moe_route_metadata_bytes(token_count)
        .ok_or(OpError::MoeExpertTensorContractMismatch)?;
    if hidden.shape() != [token_count, GEMMA4_MOE_HIDDEN_SIZE]
        || token_count == 0
        || token_count > GEMMA4_MOE_MAX_TOKENS
        || hidden.dtype() != DType::Bf16
        || hidden.encoding() != Encoding::Unquantized
        || !hidden.is_contiguous()
        || output.shape() != hidden.shape()
        || output.dtype() != DType::Bf16
        || output.encoding() != Encoding::Unquantized
        || !output.is_contiguous()
        || metadata.shape() != [metadata_bytes]
        || metadata.dtype() != DType::U8
        || metadata.encoding() != Encoding::Unquantized
        || !metadata.is_contiguous()
        || layer_blob.shape() != [GEMMA4_MOE_LAYER_BLOB_BYTES]
        || layer_blob.dtype() != DType::U8
        || layer_blob.encoding() != Encoding::Unquantized
        || !layer_blob.is_contiguous()
    {
        return Err(OpError::MoeExpertTensorContractMismatch);
    }
    Ok(())
}

fn validate_causal_attention(
    inputs: &[TensorView],
    outputs: &[TensorView],
    contract: WindowedCausalAttentionContract,
) -> Result<(), OpError> {
    for tensor in inputs.iter().chain(outputs) {
        if tensor.shape().contains(&0) {
            return Err(OpError::CausalAttentionZeroExtent);
        }
        if !tensor.is_contiguous() {
            return Err(OpError::CausalAttentionNonContiguous);
        }
        if tensor.dtype() != DType::Bf16 || tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::CausalAttentionTensorContractMismatch);
        }
    }
    let m = contract.query_count() as usize;
    let length = usize::try_from(contract.expected_kv_length())
        .map_err(|_| OpError::CausalAttentionShapeMismatch)?;
    let q_shape = [m, contract.q_heads() as usize, contract.head_dim() as usize];
    let kv_shape = [
        length,
        contract.kv_heads() as usize,
        contract.head_dim() as usize,
    ];
    if inputs[0].shape() != q_shape
        || inputs[1].shape() != kv_shape
        || inputs[2].shape() != kv_shape
        || outputs[0].shape() != q_shape
    {
        return Err(OpError::CausalAttentionShapeMismatch);
    }
    Ok(())
}

fn validate_rotary(
    inputs: &[TensorView],
    outputs: &[TensorView],
    contract: SplitHalfRotaryContract,
) -> Result<(), OpError> {
    let tensors = [
        (&inputs[0], RotaryTensor::Query),
        (&inputs[1], RotaryTensor::Key),
        (&inputs[2], RotaryTensor::Positions),
        (&outputs[0], RotaryTensor::QueryOutput),
        (&outputs[1], RotaryTensor::KeyOutput),
    ];
    for (tensor, role) in tensors {
        if tensor.shape().contains(&0) {
            return Err(OpError::RotaryZeroExtent { tensor: role });
        }
        if !tensor.is_contiguous() {
            return Err(OpError::RotaryNonContiguous { tensor: role });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::RotaryUnsupportedEncoding {
                tensor: role,
                actual: tensor.encoding(),
            });
        }
        let expected_dtype = if role == RotaryTensor::Positions {
            DType::I32
        } else {
            DType::Bf16
        };
        if tensor.dtype() != expected_dtype {
            return Err(OpError::RotaryUnsupportedDType {
                tensor: role,
                expected: expected_dtype,
                actual: tensor.dtype(),
            });
        }
    }
    let m = contract.token_count() as usize;
    let q_shape = [m, contract.q_heads() as usize, contract.head_dim() as usize];
    let k_shape = [
        m,
        contract.kv_heads() as usize,
        contract.head_dim() as usize,
    ];
    if inputs[0].shape() != q_shape
        || outputs[0].shape() != q_shape
        || inputs[1].shape() != k_shape
        || outputs[1].shape() != k_shape
        || inputs[2].shape() != [m]
    {
        return Err(OpError::RotaryShapeMismatch);
    }
    Ok(())
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

fn validate_token_selector(
    inputs: &[TensorView],
    outputs: &[TensorView],
    contract: TokenSelectorContractV1,
) -> Result<(), OpError> {
    let roles = [
        (&inputs[0], TokenSelectorTensor::Logits),
        (&inputs[1], TokenSelectorTensor::AdditiveLogits),
        (&inputs[2], TokenSelectorTensor::ValidMask),
        (&outputs[0], TokenSelectorTensor::Output),
    ];
    for (tensor, role) in roles {
        if tensor.shape().contains(&0) {
            return Err(OpError::TokenSelectorZeroExtent { tensor: role });
        }
        if !tensor.is_contiguous() {
            return Err(OpError::TokenSelectorNonContiguous { tensor: role });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::TokenSelectorUnsupportedEncoding {
                tensor: role,
                actual: tensor.encoding(),
            });
        }
    }
    let vocab = usize::try_from(contract.vocab_size()).map_err(|_| {
        OpError::TokenSelectorVocabOutOfRange {
            vocab: contract.vocab_size(),
        }
    })?;
    if inputs[0].shape() != [1, vocab]
        || inputs[1].shape() != [1, vocab]
        || inputs[2].shape() != [1, vocab]
    {
        return Err(OpError::TokenSelectorShapeMismatch);
    }
    if outputs[0].shape() != [16] || outputs[0].payload_bytes() != 16 {
        return Err(OpError::TokenSelectorOutputShapeMismatch);
    }
    let expected = [
        (TokenSelectorTensor::Logits, DType::Bf16),
        (TokenSelectorTensor::AdditiveLogits, DType::F32),
        (TokenSelectorTensor::ValidMask, DType::U8),
        (TokenSelectorTensor::Output, DType::U8),
    ];
    for ((tensor, role), (_, dtype)) in roles.into_iter().zip(expected) {
        if tensor.dtype() != dtype {
            return Err(OpError::TokenSelectorUnsupportedDType {
                tensor: role,
                expected: dtype,
                actual: tensor.dtype(),
            });
        }
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
    }
    for tensor in [activation, output] {
        if tensor.encoding() != Encoding::Unquantized || tensor.dtype() != DType::Bf16 {
            return Err(OpError::MatmulActivationOutputContract);
        }
    }
    let bf16_weight = weight.encoding() == Encoding::Unquantized && weight.dtype() == DType::Bf16;
    let fp8_weight = matches!(weight.dtype(), DType::F8E4M3Fn | DType::F8E4M3FnuZ)
        && weight.encoding()
            == Encoding::Fp8Scaled {
                granularity: Fp8ScaleGranularity::OuterDimension,
                scale_dtype: DType::F32,
                resident: Fp8ResidentRepresentation::PackedBytes,
            };
    let low_bit_weight = weight.dtype() == DType::U8
        && matches!(
            weight.encoding(),
            Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F8E4M3Fn,
            } | Encoding::Nvfp4W4A4 {
                block_size: 16,
                scale_dtype: DType::F8E4M3Fn,
            } | Encoding::Mxfp4W4A4 {
                block_size: 32,
                scale_dtype: DType::U8,
            } | Encoding::Mxfp6W6A6 {
                block_size: 32,
                scale_dtype: DType::U8,
            }
        );
    let mxfp8_weight = weight.dtype() == DType::F8E4M3Fn
        && weight.encoding()
            == Encoding::Mxfp8W8A8 {
                block_size: 32,
                scale_dtype: DType::U8,
            };
    if (mxfp8_weight || matches!(weight.encoding(), Encoding::Mxfp6W6A6 { .. }))
        && activation_shape[1] % 32 != 0
    {
        return Err(OpError::MatmulWeightContract);
    }
    if !bf16_weight && !fp8_weight && !low_bit_weight && !mxfp8_weight {
        return Err(OpError::MatmulWeightContract);
    }
    Ok(())
}

#[cfg(test)]
mod mxfp_weight_activation_matmul_tests {
    use super::*;

    fn descriptor(
        k: usize,
        dtype: DType,
        encoding: Encoding,
    ) -> Result<SemanticOpDescriptor, OpError> {
        SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![
                TensorView::contiguous(DType::Bf16, &[3, k]).unwrap(),
                TensorView::with_encoding(dtype, encoding, &[7, k]).unwrap(),
            ],
            vec![TensorView::contiguous(DType::Bf16, &[3, 7]).unwrap()],
        )
    }

    #[test]
    fn ocp_mx_weight_activation_matmul_requires_exact_block32_k() {
        for (dtype, encoding) in [
            (
                DType::F8E4M3Fn,
                Encoding::Mxfp8W8A8 {
                    block_size: 32,
                    scale_dtype: DType::U8,
                },
            ),
            (
                DType::U8,
                Encoding::Mxfp6W6A6 {
                    block_size: 32,
                    scale_dtype: DType::U8,
                },
            ),
        ] {
            descriptor(32, dtype, encoding).expect("one complete MX block");
            descriptor(64, dtype, encoding).expect("two complete MX blocks");
            assert!(matches!(
                descriptor(31, dtype, encoding),
                Err(OpError::MatmulWeightContract)
            ));
            assert!(matches!(
                descriptor(33, dtype, encoding),
                Err(OpError::MatmulWeightContract)
            ));
        }
    }
}

fn validate_baseline_elementwise(
    kind: SemanticOpKind,
    inputs: &[TensorView],
    outputs: &[TensorView],
) -> Result<(), OpError> {
    let metadata_matches = match kind {
        SemanticOpKind::Copy => same_metadata(&inputs[0], &outputs[0]),
        SemanticOpKind::Add
        | SemanticOpKind::SiluMul
        | SemanticOpKind::GeluTanhMul
        | SemanticOpKind::SigmoidMul => {
            same_metadata(&inputs[0], &inputs[1]) && same_metadata(&inputs[0], &outputs[0])
        }
        SemanticOpKind::ScalarMul | SemanticOpKind::TanhSoftcap => {
            same_metadata(&inputs[0], &outputs[0])
                && inputs[1].shape() == [1]
                && inputs[1].dtype() == inputs[0].dtype()
                && inputs[1].encoding() == inputs[0].encoding()
        }
        _ => unreachable!("elementwise validation is only used by copy/add"),
    };
    if !metadata_matches {
        return Err(match kind {
            SemanticOpKind::Copy => OpError::CopyMetadataMismatch,
            SemanticOpKind::Add
            | SemanticOpKind::SiluMul
            | SemanticOpKind::GeluTanhMul
            | SemanticOpKind::SigmoidMul => OpError::ElementwiseMetadataMismatch,
            SemanticOpKind::ScalarMul | SemanticOpKind::TanhSoftcap => {
                OpError::ScalarElementwiseShapeMismatch { kind }
            }
            _ => unreachable!("elementwise validation is only used by copy/add"),
        });
    }

    let mut tensors = Vec::with_capacity(inputs.len() + outputs.len());
    tensors.push((&inputs[0], ElementwiseTensor::Input0));
    if matches!(
        kind,
        SemanticOpKind::Add
            | SemanticOpKind::ScalarMul
            | SemanticOpKind::SiluMul
            | SemanticOpKind::GeluTanhMul
            | SemanticOpKind::SigmoidMul
            | SemanticOpKind::TanhSoftcap
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
        if shape.len() != 3 || !matches!(shape[1], 8 | 16 | 24) || shape[2] != 256 {
            return Err(OpError::SigmoidMulShapeMismatch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod qwen35_sigmoid_mul_tests {
    use super::*;

    fn descriptor(heads: usize) -> Result<SemanticOpDescriptor, OpError> {
        let view = TensorView::contiguous(DType::Bf16, &[3, heads, 256]).unwrap();
        SemanticOpDescriptor::new(
            SemanticOpKind::SigmoidMul,
            vec![view.clone(), view.clone()],
            vec![view],
        )
    }

    #[test]
    fn reviewed_qwen35_head_counts_include_27b() {
        for heads in [8, 16, 24] {
            descriptor(heads).expect("reviewed Qwen3.5 sigmoid-mul head count");
        }
        for heads in [23, 25] {
            assert!(matches!(
                descriptor(heads),
                Err(OpError::SigmoidMulShapeMismatch)
            ));
        }
    }
}

fn validate_broadcast_add(inputs: &[TensorView], outputs: &[TensorView]) -> Result<(), OpError> {
    let input = &inputs[0];
    let vector = &inputs[1];
    let output = &outputs[0];
    let tensors = [
        (input, ElementwiseTensor::Input0),
        (vector, ElementwiseTensor::Input1),
        (output, ElementwiseTensor::Output),
    ];
    for (tensor, role) in tensors {
        if tensor.shape().is_empty() {
            return Err(OpError::ElementwiseRankZero {
                kind: SemanticOpKind::BroadcastAdd,
                tensor: role,
            });
        }
        if tensor.shape().contains(&0) {
            return Err(OpError::BroadcastAddZeroExtent { tensor: role });
        }
        if !tensor.is_contiguous() {
            return Err(OpError::BroadcastAddNonContiguous { tensor: role });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::BroadcastAddUnsupportedEncoding {
                tensor: role,
                actual: tensor.encoding(),
            });
        }
        if tensor.dtype() != DType::Bf16 {
            return Err(OpError::BroadcastAddUnsupportedDType {
                tensor: role,
                actual: tensor.dtype(),
            });
        }
    }
    if input.shape().len() != 2
        || vector.shape().len() != 1
        || output.shape() != input.shape()
        || vector.shape()[0] != input.shape()[1]
    {
        return Err(OpError::BroadcastAddShapeMismatch);
    }
    Ok(())
}

fn validate_broadcast_mul(inputs: &[TensorView], outputs: &[TensorView]) -> Result<(), OpError> {
    let input = &inputs[0];
    let vector = &inputs[1];
    let output = &outputs[0];
    let tensors = [
        (input, ElementwiseTensor::Input0),
        (vector, ElementwiseTensor::Input1),
        (output, ElementwiseTensor::Output),
    ];
    for (tensor, role) in tensors {
        if tensor.shape().is_empty() {
            return Err(OpError::ElementwiseRankZero {
                kind: SemanticOpKind::BroadcastMul,
                tensor: role,
            });
        }
        if tensor.shape().contains(&0) {
            return Err(OpError::BroadcastMulZeroExtent { tensor: role });
        }
        if !tensor.is_contiguous() {
            return Err(OpError::BroadcastMulNonContiguous { tensor: role });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::BroadcastMulUnsupportedEncoding {
                tensor: role,
                actual: tensor.encoding(),
            });
        }
        if tensor.dtype() != DType::Bf16 {
            return Err(OpError::BroadcastMulUnsupportedDType {
                tensor: role,
                actual: tensor.dtype(),
            });
        }
    }
    if input.shape().len() != 2
        || vector.shape().len() != 1
        || output.shape() != input.shape()
        || vector.shape()[0] != input.shape()[1]
    {
        return Err(OpError::BroadcastMulShapeMismatch);
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

fn validate_residual_rms_norm(
    inputs: &[TensorView],
    outputs: &[TensorView],
    _contract: ResidualRmsNormContract,
) -> Result<(), OpError> {
    let tensors = [
        (&inputs[0], ResidualRmsNormTensor::Residual),
        (&inputs[1], ResidualRmsNormTensor::Addend),
        (&inputs[2], ResidualRmsNormTensor::RawScale),
        (&outputs[0], ResidualRmsNormTensor::ResidualOutput),
        (&outputs[1], ResidualRmsNormTensor::NormalizedOutput),
    ];
    for (tensor, role) in tensors {
        if tensor.shape().is_empty() {
            return Err(OpError::ResidualRmsNormRankZero { tensor: role });
        }
        if tensor.shape().contains(&0) {
            return Err(OpError::ResidualRmsNormZeroExtent { tensor: role });
        }
        if !tensor.is_contiguous() {
            return Err(OpError::ResidualRmsNormNonContiguous { tensor: role });
        }
        if tensor.encoding() != Encoding::Unquantized {
            return Err(OpError::ResidualRmsNormUnsupportedEncoding {
                tensor: role,
                actual: tensor.encoding(),
            });
        }
        if tensor.dtype() != DType::Bf16 {
            return Err(OpError::ResidualRmsNormUnsupportedDType {
                tensor: role,
                actual: tensor.dtype(),
            });
        }
    }
    if inputs[0].shape() != inputs[1].shape()
        || inputs[0].shape() != outputs[0].shape()
        || inputs[0].shape() != outputs[1].shape()
    {
        return Err(OpError::ResidualRmsNormShapeMismatch);
    }
    if inputs[2].shape().len() != 1
        || inputs[2].shape()[0] != inputs[0].shape()[inputs[0].shape().len() - 1]
    {
        return Err(OpError::ResidualRmsNormScaleShapeMismatch);
    }
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
    let position_shape = inputs[4].shape();
    if position_shape != [token_count] && position_shape != [token_count, 3] {
        return Err(OpError::AttentionPreprocessShapeMismatch {
            tensor: AttentionPreprocessTensor::Positions,
        });
    }
    let expected_shapes: [&[usize]; 7] = [
        &[
            token_count,
            contract.q_heads() as usize,
            (contract.head_dim() * 2) as usize,
        ],
        &[
            token_count,
            contract.kv_heads() as usize,
            contract.head_dim() as usize,
        ],
        &[contract.q_heads() as usize, contract.head_dim() as usize],
        &[contract.kv_heads() as usize, contract.head_dim() as usize],
        &[
            token_count,
            contract.q_heads() as usize,
            contract.head_dim() as usize,
        ],
        &[
            token_count,
            contract.q_heads() as usize,
            contract.head_dim() as usize,
        ],
        &[
            token_count,
            contract.kv_heads() as usize,
            contract.head_dim() as usize,
        ],
    ];
    for (expected, (tensor, role)) in expected_shapes.into_iter().zip(
        tensors
            .into_iter()
            .filter(|(_, role)| !matches!(role, AttentionPreprocessTensor::Positions)),
    ) {
        if tensor.shape() != expected {
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
    ScalarElementwiseShapeMismatch {
        kind: SemanticOpKind,
    },
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
    BroadcastAddShapeMismatch,
    BroadcastAddZeroExtent {
        tensor: ElementwiseTensor,
    },
    BroadcastAddNonContiguous {
        tensor: ElementwiseTensor,
    },
    BroadcastAddUnsupportedDType {
        tensor: ElementwiseTensor,
        actual: DType,
    },
    BroadcastAddUnsupportedEncoding {
        tensor: ElementwiseTensor,
        actual: Encoding,
    },
    BroadcastMulShapeMismatch,
    BroadcastMulZeroExtent {
        tensor: ElementwiseTensor,
    },
    BroadcastMulNonContiguous {
        tensor: ElementwiseTensor,
    },
    BroadcastMulUnsupportedDType {
        tensor: ElementwiseTensor,
        actual: DType,
    },
    BroadcastMulUnsupportedEncoding {
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
    MatmulActivationOutputContract,
    MatmulWeightContract,
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
    ResidualRmsNormContractRequired,
    ResidualRmsNormRankZero {
        tensor: ResidualRmsNormTensor,
    },
    ResidualRmsNormZeroExtent {
        tensor: ResidualRmsNormTensor,
    },
    ResidualRmsNormNonContiguous {
        tensor: ResidualRmsNormTensor,
    },
    ResidualRmsNormUnsupportedDType {
        tensor: ResidualRmsNormTensor,
        actual: DType,
    },
    ResidualRmsNormUnsupportedEncoding {
        tensor: ResidualRmsNormTensor,
        actual: Encoding,
    },
    ResidualRmsNormShapeMismatch,
    ResidualRmsNormScaleShapeMismatch,
    RotaryContractRequired,
    RotaryInvalidConfig {
        field: &'static str,
    },
    RotaryPositionOverflow,
    RotaryPositionOutOfRange {
        last_position: u64,
        max_position_embeddings: u32,
    },
    RotaryZeroExtent {
        tensor: RotaryTensor,
    },
    RotaryNonContiguous {
        tensor: RotaryTensor,
    },
    RotaryUnsupportedDType {
        tensor: RotaryTensor,
        expected: DType,
        actual: DType,
    },
    RotaryUnsupportedEncoding {
        tensor: RotaryTensor,
        actual: Encoding,
    },
    RotaryShapeMismatch,
    CausalAttentionContractRequired,
    CausalAttentionInvalidConfig {
        field: &'static str,
    },
    CausalAttentionLengthOverflow,
    CausalAttentionLengthMismatch {
        expected: u64,
        actual: u64,
    },
    CausalAttentionZeroExtent,
    CausalAttentionNonContiguous,
    CausalAttentionTensorContractMismatch,
    CausalAttentionShapeMismatch,
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
    TokenSelectorContractRequired,
    TokenSelectorInvalidTemperature {
        bits: u32,
    },
    TokenSelectorVocabOutOfRange {
        vocab: u64,
    },
    TokenSelectorZeroExtent {
        tensor: TokenSelectorTensor,
    },
    TokenSelectorNonContiguous {
        tensor: TokenSelectorTensor,
    },
    TokenSelectorUnsupportedDType {
        tensor: TokenSelectorTensor,
        expected: DType,
        actual: DType,
    },
    TokenSelectorUnsupportedEncoding {
        tensor: TokenSelectorTensor,
        actual: Encoding,
    },
    TokenSelectorShapeMismatch,
    TokenSelectorOutputShapeMismatch,
    GdnProjectionBundleContractRequired,
    GdnProjectionBundleZeroExtent,
    GdnProjectionBundleNonContiguous,
    GdnProjectionBundleUnsupportedDType {
        actual: DType,
    },
    GdnProjectionBundleUnsupportedEncoding {
        actual: Encoding,
    },
    GdnProjectionBundleShapeMismatch,
    MlpGateUpSiluBundleContractRequired,
    MlpGateUpSiluBundleZeroExtent,
    MlpGateUpSiluBundleNonContiguous,
    MlpGateUpSiluBundleUnsupportedDType {
        actual: DType,
    },
    MlpGateUpSiluBundleUnsupportedEncoding {
        actual: Encoding,
    },
    MlpGateUpSiluBundleShapeMismatch,
    SparseMoeContractRequired,
    SparseMoeInvalidContract,
    SparseMoeTensorContractMismatch,
    DeepSeekV4MoeRouteContractRequired,
    DeepSeekV4MoeRouteInvalidRoutedScale {
        bits: u32,
    },
    DeepSeekV4MoeRouteTokenCountOutOfRange {
        token_count: usize,
    },
    DeepSeekV4MoeRouteMetadataSizeOverflow {
        token_count: usize,
    },
    DeepSeekV4MoeRouteLogitsContractMismatch,
    DeepSeekV4MoeRouteBiasContractMismatch {
        mode: DeepSeekV4MoeRouteMode,
    },
    DeepSeekV4MoeRouteHashContractMismatch {
        mode: DeepSeekV4MoeRouteMode,
    },
    DeepSeekV4MoeRouteOutputContractMismatch,
    MiniMaxM3MoeRouteContractRequired,
    MiniMaxM3MoeRouteTokenCountOutOfRange {
        token_count: usize,
    },
    MiniMaxM3MoeRouteMetadataSizeOverflow {
        token_count: usize,
    },
    MiniMaxM3MoeRouteLogitsContractMismatch,
    MiniMaxM3MoeRouteBiasContractMismatch,
    MiniMaxM3MoeRouteOutputContractMismatch,
    MoeRouteTensorContractMismatch,
    MoeExpertTensorContractMismatch,
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
            Self::ScalarElementwiseShapeMismatch { kind } => write!(
                formatter,
                "{} requires matching input/output metadata and one BF16 scalar input",
                kind.name()
            ),
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
            Self::BroadcastAddShapeMismatch => formatter.write_str(
                "broadcast_add requires contiguous BF16 input/output [M,H] and vector [H]",
            ),
            Self::BroadcastAddZeroExtent { tensor } => {
                write!(formatter, "broadcast_add {tensor} must not have zero extents")
            }
            Self::BroadcastAddNonContiguous { tensor } => {
                write!(formatter, "broadcast_add {tensor} must be row-major contiguous")
            }
            Self::BroadcastAddUnsupportedDType { tensor, actual } => write!(
                formatter,
                "broadcast_add {tensor} must use bf16, got {actual}"
            ),
            Self::BroadcastAddUnsupportedEncoding { tensor, actual } => write!(
                formatter,
                "broadcast_add {tensor} must use unquantized encoding, got {actual:?}"
            ),
            Self::BroadcastMulShapeMismatch => formatter.write_str(
                "broadcast_mul requires contiguous BF16 input/output [M,H] and vector [H]",
            ),
            Self::BroadcastMulZeroExtent { tensor } => {
                write!(formatter, "broadcast_mul {tensor} must not have zero extents")
            }
            Self::BroadcastMulNonContiguous { tensor } => {
                write!(formatter, "broadcast_mul {tensor} must be row-major contiguous")
            }
            Self::BroadcastMulUnsupportedDType { tensor, actual } => write!(
                formatter,
                "broadcast_mul {tensor} must use bf16, got {actual}"
            ),
            Self::BroadcastMulUnsupportedEncoding { tensor, actual } => write!(
                formatter,
                "broadcast_mul {tensor} must use unquantized encoding, got {actual:?}"
            ),
            Self::SigmoidMulShapeMismatch => formatter.write_str(
                "sigmoid_mul requires identical contiguous BF16 [M,H,256] gate, attention value, and output",
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
            Self::MatmulActivationOutputContract => formatter.write_str(
                "matmul activation and output must be contiguous unquantized BF16",
            ),
            Self::MatmulWeightContract => formatter.write_str(
                "matmul weight must use a supported BF16, FP8, NVFP4, MXFP4, MXFP6, or MXFP8 resident contract",
            ),
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
            Self::ResidualRmsNormContractRequired => {
                formatter.write_str("residual_rms_norm requires an explicit contract")
            }
            Self::ResidualRmsNormRankZero { tensor } => {
                write!(formatter, "residual_rms_norm {tensor} must have rank at least one")
            }
            Self::ResidualRmsNormZeroExtent { tensor } => {
                write!(formatter, "residual_rms_norm {tensor} must not have a zero extent")
            }
            Self::ResidualRmsNormNonContiguous { tensor } => {
                write!(formatter, "residual_rms_norm {tensor} must be row-major contiguous")
            }
            Self::ResidualRmsNormUnsupportedDType { tensor, actual } => {
                write!(formatter, "residual_rms_norm {tensor} must be bf16, got {actual}")
            }
            Self::ResidualRmsNormUnsupportedEncoding { tensor, actual } => {
                write!(
                    formatter,
                    "residual_rms_norm {tensor} must use unquantized encoding, got {actual:?}"
                )
            }
            Self::ResidualRmsNormShapeMismatch => formatter.write_str(
                "residual_rms_norm residual, addend, intermediate, and output shapes must match",
            ),
            Self::ResidualRmsNormScaleShapeMismatch => formatter.write_str(
                "residual_rms_norm raw scale must be rank one and match the final dimension",
            ),
            Self::RotaryContractRequired => {
                formatter.write_str("rotary requires an explicit split-half contract")
            }
            Self::RotaryInvalidConfig { field } => {
                write!(formatter, "rotary has an invalid {field} configuration")
            }
            Self::RotaryPositionOverflow => {
                formatter.write_str("rotary position range overflowed")
            }
            Self::RotaryPositionOutOfRange {
                last_position,
                max_position_embeddings,
            } => write!(
                formatter,
                "rotary position {last_position} is not below max position {max_position_embeddings}"
            ),
            Self::RotaryZeroExtent { tensor } => {
                write!(formatter, "rotary {tensor} must not have a zero extent")
            }
            Self::RotaryNonContiguous { tensor } => {
                write!(formatter, "rotary {tensor} must be row-major contiguous")
            }
            Self::RotaryUnsupportedDType {
                tensor,
                expected,
                actual,
            } => write!(formatter, "rotary {tensor} must use {expected}, got {actual}"),
            Self::RotaryUnsupportedEncoding { tensor, actual } => write!(
                formatter,
                "rotary {tensor} must use unquantized encoding, got {actual:?}"
            ),
            Self::RotaryShapeMismatch => formatter.write_str(
                "rotary requires Q/K and output [M,H,D] tensors plus I32 positions [M] matching its contract",
            ),
            Self::CausalAttentionContractRequired => {
                formatter.write_str("causal_attention requires an explicit window contract")
            }
            Self::CausalAttentionInvalidConfig { field } => {
                write!(formatter, "causal_attention has an invalid {field} configuration")
            }
            Self::CausalAttentionLengthOverflow => {
                formatter.write_str("causal_attention length overflowed")
            }
            Self::CausalAttentionLengthMismatch { expected, actual } => write!(
                formatter,
                "causal_attention expected KV length {expected}, got {actual}"
            ),
            Self::CausalAttentionZeroExtent => {
                formatter.write_str("causal_attention tensors must not have zero extents")
            }
            Self::CausalAttentionNonContiguous => {
                formatter.write_str("causal_attention tensors must be row-major contiguous")
            }
            Self::CausalAttentionTensorContractMismatch => formatter.write_str(
                "causal_attention tensors must be unquantized BF16",
            ),
            Self::CausalAttentionShapeMismatch => formatter.write_str(
                "causal_attention requires Q/output [M,Hq,D] and K/V [L,Hkv,D] matching its contract",
            ),
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
            Self::TokenSelectorContractRequired => {
                formatter.write_str("token_select requires an explicit selection contract")
            }
            Self::TokenSelectorInvalidTemperature { bits } => write!(
                formatter,
                "token_select temperature must be finite and positive (bits 0x{bits:08x})"
            ),
            Self::TokenSelectorVocabOutOfRange { vocab } => write!(
                formatter,
                "token_select vocabulary size {vocab} is outside the supported 1..=1048576 range"
            ),
            Self::TokenSelectorZeroExtent { tensor } => {
                write!(formatter, "token_select {tensor} must have non-zero extents")
            }
            Self::TokenSelectorNonContiguous { tensor } => {
                write!(formatter, "token_select {tensor} must be row-major contiguous")
            }
            Self::TokenSelectorUnsupportedDType {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "token_select {tensor} must use {expected}, got {actual}"
            ),
            Self::TokenSelectorUnsupportedEncoding { tensor, actual } => write!(
                formatter,
                "token_select {tensor} must use unquantized encoding, got {actual:?}"
            ),
            Self::TokenSelectorShapeMismatch => formatter.write_str(
                "token_select logits, additive logits, and valid mask must be contiguous [1,V] tensors"
            ),
            Self::TokenSelectorOutputShapeMismatch => formatter.write_str(
                "token_select output must be a contiguous unquantized U8 [16] record"
            ),
            Self::GdnProjectionBundleContractRequired => {
                formatter.write_str("gdn_projection_bundle requires an explicit contract")
            }
            Self::GdnProjectionBundleZeroExtent => {
                formatter.write_str("gdn_projection_bundle tensors must have non-zero extents")
            }
            Self::GdnProjectionBundleNonContiguous => {
                formatter.write_str("gdn_projection_bundle tensors must be row-major contiguous")
            }
            Self::GdnProjectionBundleUnsupportedDType { actual } => write!(
                formatter,
                "gdn_projection_bundle tensors must use bf16, got {actual}"
            ),
            Self::GdnProjectionBundleUnsupportedEncoding { actual } => write!(
                formatter,
                "gdn_projection_bundle tensors must use unquantized encoding, got {actual:?}"
            ),
            Self::GdnProjectionBundleShapeMismatch => formatter.write_str(
                "gdn_projection_bundle requires activation [M,2560], weights [N,2560], outputs [M,N] for N=8192,4096,32,32",
            ),
            Self::MlpGateUpSiluBundleContractRequired => {
                formatter.write_str("mlp_gate_up_silu_bundle requires an explicit contract")
            },
            Self::MlpGateUpSiluBundleZeroExtent => {
                formatter.write_str("mlp_gate_up_silu_bundle tensors must have non-zero extents")
            },
            Self::MlpGateUpSiluBundleNonContiguous => formatter.write_str(
                "mlp_gate_up_silu_bundle tensors must be row-major contiguous",
            ),
            Self::MlpGateUpSiluBundleUnsupportedDType { actual } => write!(
                formatter,
                "mlp_gate_up_silu_bundle tensors must use bf16, got {actual}"
            ),
            Self::MlpGateUpSiluBundleUnsupportedEncoding { actual } => write!(
                formatter,
                "mlp_gate_up_silu_bundle tensors must use unquantized encoding, got {actual:?}"
            ),
            Self::MlpGateUpSiluBundleShapeMismatch => formatter.write_str(
                "mlp_gate_up_silu_bundle requires activation [M,2560], weights [9216,2560], outputs [M,9216]",
            ),
            Self::SparseMoeContractRequired => {
                formatter.write_str("sparse_moe requires an explicit routing/expert contract")
            }
            Self::SparseMoeInvalidContract => {
                formatter.write_str("sparse_moe contract dimensions are invalid")
            }
            Self::SparseMoeTensorContractMismatch => formatter.write_str(
                "sparse_moe input/output must be matching contiguous unquantized BF16 [M,H] tensors",
            ),
            Self::DeepSeekV4MoeRouteContractRequired => formatter
                .write_str("deepseek_v4_moe_route requires an explicit routing contract"),
            Self::DeepSeekV4MoeRouteInvalidRoutedScale { bits } => write!(
                formatter,
                "deepseek_v4_moe_route routed_scale must be finite and positive (bits 0x{bits:08x})"
            ),
            Self::DeepSeekV4MoeRouteTokenCountOutOfRange { token_count } => write!(
                formatter,
                "deepseek_v4_moe_route token count {token_count} is outside the supported 1..=65536 range"
            ),
            Self::DeepSeekV4MoeRouteMetadataSizeOverflow { token_count } => write!(
                formatter,
                "deepseek_v4_moe_route metadata size overflowed for {token_count} tokens"
            ),
            Self::DeepSeekV4MoeRouteLogitsContractMismatch => formatter.write_str(
                "deepseek_v4_moe_route logits must be contiguous unquantized BF16 [M,256]",
            ),
            Self::DeepSeekV4MoeRouteBiasContractMismatch { mode } => write!(
                formatter,
                "deepseek_v4_moe_route {mode:?} bias must be contiguous unquantized F32 [256] in score mode or the canonical empty F32 [0] placeholder in hash mode"
            ),
            Self::DeepSeekV4MoeRouteHashContractMismatch { mode } => write!(
                formatter,
                "deepseek_v4_moe_route {mode:?} hash IDs must be contiguous unquantized I32 [M,6] in hash mode or the canonical empty I32 [0,6] placeholder in score mode"
            ),
            Self::DeepSeekV4MoeRouteOutputContractMismatch => formatter.write_str(
                "deepseek_v4_moe_route output must be contiguous unquantized U8 canonical route metadata",
            ),
            Self::MiniMaxM3MoeRouteContractRequired => formatter
                .write_str("minimax_m3_moe_route requires its explicit fixed routing contract"),
            Self::MiniMaxM3MoeRouteTokenCountOutOfRange { token_count } => write!(
                formatter,
                "minimax_m3_moe_route token count {token_count} is outside the supported 1..=65536 range"
            ),
            Self::MiniMaxM3MoeRouteMetadataSizeOverflow { token_count } => write!(
                formatter,
                "minimax_m3_moe_route metadata size overflowed for {token_count} tokens"
            ),
            Self::MiniMaxM3MoeRouteLogitsContractMismatch => formatter.write_str(
                "minimax_m3_moe_route logits must be contiguous unquantized F32 [M,128]",
            ),
            Self::MiniMaxM3MoeRouteBiasContractMismatch => formatter.write_str(
                "minimax_m3_moe_route selection bias must be contiguous unquantized F32 [128]",
            ),
            Self::MiniMaxM3MoeRouteOutputContractMismatch => formatter.write_str(
                "minimax_m3_moe_route output must be contiguous unquantized U8 canonical route metadata",
            ),
            Self::MoeRouteTensorContractMismatch => formatter.write_str(
                "moe_route requires contiguous BF16 logits [M,128] and canonical top-8 U8 route metadata",
            ),
            Self::MoeExpertTensorContractMismatch => formatter.write_str(
                "moe_expert requires Gemma 4 v2 BF16 [M,2816], canonical top-8 route metadata, and a 428215552-byte layer blob",
            ),
        }
    }
}

impl std::error::Error for OpError {}

#[cfg(test)]
mod minimax_m3_moe_route_tests {
    use super::*;

    fn descriptor(token_count: usize) -> Result<SemanticOpDescriptor, OpError> {
        SemanticOpDescriptor::new_minimax_m3_moe_route(
            vec![
                TensorView::contiguous(
                    DType::F32,
                    &[token_count, MiniMaxM3MoeRouteContractV1::EXPERT_COUNT],
                )
                .unwrap(),
                TensorView::contiguous(DType::F32, &[MiniMaxM3MoeRouteContractV1::EXPERT_COUNT])
                    .unwrap(),
            ],
            vec![
                TensorView::contiguous(
                    DType::U8,
                    &[MiniMaxM3MoeRouteContractV1::metadata_bytes(token_count).unwrap()],
                )
                .unwrap(),
            ],
            MiniMaxM3MoeRouteContractV1::new(),
        )
    }

    #[test]
    fn fixed_m3_contract_and_layout_are_exact() {
        let descriptor = descriptor(3).unwrap();
        assert_eq!(descriptor.kind(), SemanticOpKind::MiniMaxM3MoeRoute);
        assert_eq!(MiniMaxM3MoeRouteContractV1::metadata_bytes(3), Some(1_224));
        assert_eq!(
            descriptor.minimax_m3_moe_route_contract(),
            Some(MiniMaxM3MoeRouteContractV1::new())
        );
    }

    #[test]
    fn missing_contract_and_shape_boundaries_fail_closed() {
        let valid = descriptor(3).unwrap();
        assert!(
            SemanticOpDescriptor::new(
                SemanticOpKind::MiniMaxM3MoeRoute,
                valid.inputs().to_vec(),
                valid.outputs().to_vec(),
            )
            .is_err()
        );
        assert!(matches!(
            descriptor(0),
            Err(OpError::MiniMaxM3MoeRouteTokenCountOutOfRange { token_count: 0 })
        ));
        assert!(matches!(
            descriptor(MiniMaxM3MoeRouteContractV1::MAX_TOKEN_COUNT + 1),
            Err(OpError::MiniMaxM3MoeRouteTokenCountOutOfRange { .. })
        ));

        let mut wrong_logits = valid.inputs().to_vec();
        wrong_logits[0] = TensorView::contiguous(DType::Bf16, &[3, 128]).unwrap();
        assert!(matches!(
            SemanticOpDescriptor::new_minimax_m3_moe_route(
                wrong_logits,
                valid.outputs().to_vec(),
                MiniMaxM3MoeRouteContractV1::new(),
            ),
            Err(OpError::MiniMaxM3MoeRouteLogitsContractMismatch)
        ));

        let mut wrong_bias = valid.inputs().to_vec();
        wrong_bias[1] = TensorView::contiguous(DType::F32, &[127]).unwrap();
        assert!(matches!(
            SemanticOpDescriptor::new_minimax_m3_moe_route(
                wrong_bias,
                valid.outputs().to_vec(),
                MiniMaxM3MoeRouteContractV1::new(),
            ),
            Err(OpError::MiniMaxM3MoeRouteBiasContractMismatch)
        ));
    }

    #[test]
    fn metadata_size_overflow_is_checked() {
        assert_eq!(
            MiniMaxM3MoeRouteContractV1::metadata_bytes(usize::MAX),
            None
        );
    }
}

#[cfg(test)]
mod deepseek_v4_moe_route_tests {
    use super::*;

    fn view(dtype: DType, shape: &[usize]) -> TensorView {
        TensorView::contiguous(dtype, shape).unwrap()
    }

    fn score_inputs(token_count: usize) -> Vec<TensorView> {
        vec![
            view(
                DType::Bf16,
                &[token_count, DeepSeekV4MoeRouteContractV1::EXPERT_COUNT],
            ),
            view(DType::F32, &[DeepSeekV4MoeRouteContractV1::EXPERT_COUNT]),
            view(
                DType::I32,
                &[0, DeepSeekV4MoeRouteContractV1::SELECTED_EXPERT_COUNT],
            ),
        ]
    }

    fn hash_inputs(token_count: usize) -> Vec<TensorView> {
        vec![
            view(
                DType::Bf16,
                &[token_count, DeepSeekV4MoeRouteContractV1::EXPERT_COUNT],
            ),
            view(DType::F32, &[0]),
            view(
                DType::I32,
                &[
                    token_count,
                    DeepSeekV4MoeRouteContractV1::SELECTED_EXPERT_COUNT,
                ],
            ),
        ]
    }

    fn output(token_count: usize) -> Vec<TensorView> {
        vec![view(
            DType::U8,
            &[DeepSeekV4MoeRouteContractV1::metadata_bytes(token_count).unwrap()],
        )]
    }

    #[test]
    fn deepseek_v4_moe_route_score_mode_accepts_non_aligned_m3() {
        let contract =
            DeepSeekV4MoeRouteContractV1::new(DeepSeekV4MoeRouteMode::Score, true, 1.5).unwrap();
        let descriptor =
            SemanticOpDescriptor::new_deepseek_v4_moe_route(score_inputs(3), output(3), contract)
                .unwrap();

        assert_eq!(
            SemanticOpKind::DeepSeekV4MoeRoute.name(),
            "deepseek_v4_moe_route"
        );
        assert_eq!(descriptor.arity(), (3, 1));
        assert_eq!(descriptor.kind(), SemanticOpKind::DeepSeekV4MoeRoute);
        assert_eq!(descriptor.deepseek_v4_moe_route_contract(), Some(contract));
        assert_eq!(contract.mode(), DeepSeekV4MoeRouteMode::Score);
        assert!(contract.renormalize_selected_weights());
        assert_eq!(contract.routed_scale(), 1.5);
        assert_eq!(DeepSeekV4MoeRouteContractV1::metadata_bytes(3), Some(2_344));
    }

    #[test]
    fn deepseek_v4_moe_route_hash_mode_accepts_non_aligned_m3() {
        let contract =
            DeepSeekV4MoeRouteContractV1::new(DeepSeekV4MoeRouteMode::Hash, false, 1.5).unwrap();
        let descriptor =
            SemanticOpDescriptor::new_deepseek_v4_moe_route(hash_inputs(3), output(3), contract)
                .unwrap();

        assert_eq!(descriptor.deepseek_v4_moe_route_contract(), Some(contract));
        assert_eq!(contract.mode(), DeepSeekV4MoeRouteMode::Hash);
        assert!(!contract.renormalize_selected_weights());
    }

    #[test]
    fn deepseek_v4_moe_route_rejects_zero_and_too_many_tokens() {
        let contract =
            DeepSeekV4MoeRouteContractV1::new(DeepSeekV4MoeRouteMode::Score, true, 1.5).unwrap();
        assert_eq!(
            SemanticOpDescriptor::new_deepseek_v4_moe_route(score_inputs(0), output(0), contract,),
            Err(OpError::DeepSeekV4MoeRouteTokenCountOutOfRange { token_count: 0 })
        );

        let too_many = DeepSeekV4MoeRouteContractV1::MAX_TOKEN_COUNT + 1;
        assert_eq!(
            SemanticOpDescriptor::new_deepseek_v4_moe_route(
                score_inputs(too_many),
                output(too_many),
                contract,
            ),
            Err(OpError::DeepSeekV4MoeRouteTokenCountOutOfRange {
                token_count: too_many,
            })
        );
    }

    #[test]
    fn deepseek_v4_moe_route_rejects_invalid_dtype_and_shapes() {
        let score =
            DeepSeekV4MoeRouteContractV1::new(DeepSeekV4MoeRouteMode::Score, true, 1.5).unwrap();
        let mut inputs = score_inputs(3);
        inputs[0] = view(DType::F32, &[3, 256]);
        assert_eq!(
            SemanticOpDescriptor::new_deepseek_v4_moe_route(inputs, output(3), score),
            Err(OpError::DeepSeekV4MoeRouteLogitsContractMismatch)
        );

        let mut inputs = score_inputs(3);
        inputs[1] = view(DType::F32, &[255]);
        assert_eq!(
            SemanticOpDescriptor::new_deepseek_v4_moe_route(inputs, output(3), score),
            Err(OpError::DeepSeekV4MoeRouteBiasContractMismatch {
                mode: DeepSeekV4MoeRouteMode::Score,
            })
        );

        let hash =
            DeepSeekV4MoeRouteContractV1::new(DeepSeekV4MoeRouteMode::Hash, true, 1.5).unwrap();
        let mut inputs = hash_inputs(3);
        inputs[2] = view(DType::I32, &[3, 5]);
        assert_eq!(
            SemanticOpDescriptor::new_deepseek_v4_moe_route(inputs, output(3), hash),
            Err(OpError::DeepSeekV4MoeRouteHashContractMismatch {
                mode: DeepSeekV4MoeRouteMode::Hash,
            })
        );

        assert_eq!(
            SemanticOpDescriptor::new_deepseek_v4_moe_route(
                score_inputs(3),
                vec![view(DType::F32, &[2_344])],
                score,
            ),
            Err(OpError::DeepSeekV4MoeRouteOutputContractMismatch)
        );
    }

    #[test]
    fn deepseek_v4_moe_route_rejects_non_positive_or_non_finite_scale() {
        for scale in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(
                DeepSeekV4MoeRouteContractV1::new(DeepSeekV4MoeRouteMode::Score, true, scale,),
                Err(OpError::DeepSeekV4MoeRouteInvalidRoutedScale {
                    bits: scale.to_bits(),
                })
            );
        }
    }

    #[test]
    fn deepseek_v4_moe_route_requires_contract_and_exact_placeholders() {
        assert_eq!(
            SemanticOpDescriptor::new(
                SemanticOpKind::DeepSeekV4MoeRoute,
                score_inputs(3),
                output(3),
            ),
            Err(OpError::DeepSeekV4MoeRouteContractRequired)
        );

        let contract =
            DeepSeekV4MoeRouteContractV1::new(DeepSeekV4MoeRouteMode::Score, true, 1.5).unwrap();
        let mut inputs = score_inputs(3);
        inputs[2] = view(DType::I32, &[0]);
        assert_eq!(
            SemanticOpDescriptor::new_deepseek_v4_moe_route(inputs, output(3), contract),
            Err(OpError::DeepSeekV4MoeRouteHashContractMismatch {
                mode: DeepSeekV4MoeRouteMode::Score,
            })
        );
    }

    #[test]
    fn deepseek_v4_moe_route_metadata_size_is_checked() {
        assert_eq!(
            DeepSeekV4MoeRouteContractV1::metadata_bytes(usize::MAX),
            None
        );
        assert!(
            DeepSeekV4MoeRouteContractV1::metadata_bytes(
                DeepSeekV4MoeRouteContractV1::MAX_TOKEN_COUNT
            )
            .is_some()
        );
    }
}
