// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Backend-independent ModelGraph declarations and validation.
//!
//! The declarations intentionally do not import scheduler, runtime, backend, or
//! model-family modules. They are the validated input to later planning layers.

use std::collections::{BTreeMap, BTreeSet};

use crate::format_id::{FORMAT_AQ4_0, FORMAT_SQ8_0, canonical_format_id};

/// Maximum ASCII byte length of stable graph identifiers.
pub const MAX_IDENTIFIER_BYTES: usize = 128;
/// Maximum logical tensor rank.
pub const MAX_TENSOR_RANK: usize = 16;
/// Maximum logical elements in one tensor declaration.
///
/// This is a format-independent structural limit, not a byte estimate or an
/// allocation admission decision. Quantized and custom formats can have
/// physical metadata that makes a nominal element-to-byte conversion unsound.
pub const MAX_TENSOR_LOGICAL_ELEMENTS: u64 = 4_294_967_296;
/// Maximum count for graph values, weights, or nodes.
pub const MAX_GRAPH_DECLARATIONS: usize = 65_536;
/// Maximum graph inputs or final outputs.
pub const MAX_GRAPH_ENDPOINTS: usize = 1_024;
/// Maximum value references on one graph node.
pub const MAX_NODE_VALUES: usize = 64;
/// Maximum weight references on one graph node.
pub const MAX_NODE_WEIGHTS: usize = 128;
/// Maximum state references on one graph node.
pub const MAX_NODE_STATES: usize = 64;

/// Validates a bounded stable identifier.
pub(crate) fn validate_identifier(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{label} must be nonempty"));
    }
    if value.len() > MAX_IDENTIFIER_BYTES {
        return Err(format!(
            "{label} exceeds the {MAX_IDENTIFIER_BYTES}-byte limit"
        ));
    }
    if !value.is_ascii() {
        return Err(format!("{label} must be ASCII"));
    }
    if value
        .bytes()
        .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control())
    {
        return Err(format!(
            "{label} must not contain whitespace or C0 controls"
        ));
    }
    Ok(())
}

macro_rules! stable_id {
    ($name:ident, $label:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub String);

        impl $name {
            /// Creates a validated stable identifier.
            pub fn new(value: impl Into<String>) -> Result<Self, String> {
                let value = value.into();
                validate_identifier(&value, $label)?;
                Ok(Self(value))
            }

            /// Returns the validated identifier text.
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Revalidates this public newtype.
            pub fn validate(&self) -> Result<(), String> {
                validate_identifier(&self.0, $label)
            }
        }
    };
}

stable_id!(NodeId, "node ID");
stable_id!(ValueId, "value ID");
stable_id!(WeightId, "weight ID");
stable_id!(StateId, "state ID");

/// Storage or compute format, kept separate from model topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumericalFormat {
    /// IEEE binary32.
    F32,
    /// Brain floating point 16-bit.
    Bf16,
    /// IEEE binary16.
    Fp16,
    /// Unsigned 32-bit integer for typed counters and indices.
    U32,
    /// Unsigned 64-bit integer for typed counters and indices.
    U64,
    /// uLLM AQ4_0.
    Aq4_0,
    /// uLLM SQ8_0.
    Sq8_0,
    /// A future validated format identifier.
    Custom(String),
}

impl NumericalFormat {
    /// Creates a validated future format identifier.
    pub fn custom(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        let format = Self::Custom(value);
        format.validate()?;
        Ok(format)
    }

    /// Returns the canonical or custom format identifier.
    pub fn as_str(&self) -> &str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
            Self::Fp16 => "FP16",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::Aq4_0 => FORMAT_AQ4_0,
            Self::Sq8_0 => FORMAT_SQ8_0,
            Self::Custom(value) => value,
        }
    }

    /// Validates a public custom format value.
    pub fn validate(&self) -> Result<(), String> {
        if let Self::Custom(value) = self {
            validate_identifier(value, "custom numerical format")?;
            if is_builtin_numerical_format_id(value) {
                return Err(format!(
                    "custom numerical format {value} conflicts with a built-in format ID"
                ));
            }
        }
        Ok(())
    }
}

/// Logical tensor layout, independent of numerical format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorLayout {
    /// Conventional contiguous row-major layout.
    RowMajor,
    /// Logical tokens by hidden width layout.
    TokensHidden,
    /// Packed values whose offsets are supplied by an execution batch.
    PackedRagged,
    /// A future validated layout identifier.
    Custom(String),
}

impl TensorLayout {
    /// Creates a validated future layout identifier.
    pub fn custom(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        validate_identifier(&value, "custom tensor layout")?;
        Ok(Self::Custom(value))
    }

    /// Validates a public custom layout value.
    pub fn validate(&self) -> Result<(), String> {
        if let Self::Custom(value) = self {
            validate_identifier(value, "custom tensor layout")?;
        }
        Ok(())
    }
}

/// Logical shape, format, and layout of a graph value or weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Tensor dimensions with rank between one and sixteen.
    pub shape: Vec<usize>,
    /// Storage or compute format.
    pub format: NumericalFormat,
    /// Logical layout.
    pub layout: TensorLayout,
}

impl TensorSpec {
    /// Builds a validated tensor specification.
    pub fn new(
        shape: Vec<usize>,
        format: NumericalFormat,
        layout: TensorLayout,
    ) -> Result<Self, String> {
        let spec = Self {
            shape,
            format,
            layout,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Validates shape, format, layout, and checked element count.
    pub fn validate(&self) -> Result<(), String> {
        if self.shape.is_empty() || self.shape.len() > MAX_TENSOR_RANK {
            return Err(format!(
                "tensor rank must be in 1..={MAX_TENSOR_RANK}, got {}",
                self.shape.len()
            ));
        }
        if self.shape.contains(&0) {
            return Err("tensor dimensions must be greater than zero".into());
        }
        self.format.validate()?;
        self.layout.validate()?;
        self.element_count().map(|_| ())
    }

    /// Returns the checked logical element count.
    pub fn element_count(&self) -> Result<usize, String> {
        let count = self.shape.iter().try_fold(1_usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| "tensor element count overflows usize".to_string())
        })?;
        if (count as u128) > u128::from(MAX_TENSOR_LOGICAL_ELEMENTS) {
            return Err(format!(
                "tensor logical element count {count} exceeds limit {MAX_TENSOR_LOGICAL_ELEMENTS}"
            ));
        }
        Ok(count)
    }
}

/// A declared graph value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphValue {
    /// Stable graph-local value ID.
    pub id: ValueId,
    /// Logical tensor metadata.
    pub tensor: TensorSpec,
}

impl GraphValue {
    /// Validates this declaration.
    pub fn validate(&self) -> Result<(), String> {
        self.id.validate()?;
        self.tensor.validate()
    }
}

/// Logical weight declaration used by graph nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSpec {
    /// Stable logical weight ID.
    pub id: WeightId,
    /// Expected physical shape, storage format, and layout.
    pub tensor: TensorSpec,
}

impl WeightSpec {
    /// Validates this logical declaration.
    pub fn validate(&self) -> Result<(), String> {
        self.id.validate()?;
        self.tensor.validate()
    }
}

/// Physical package or artifact binding for one logical weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBinding {
    /// Referenced logical weight.
    pub logical_id: WeightId,
    /// Physical package or artifact tensor name.
    pub physical_tensor_name: String,
    /// Physical tensor metadata.
    pub tensor: TensorSpec,
    /// Optional lowercase SHA-256 of the physical payload.
    pub content_sha256: Option<String>,
}

impl WeightBinding {
    /// Validates binding-local metadata.
    pub fn validate(&self) -> Result<(), String> {
        self.logical_id.validate()?;
        validate_identifier(&self.physical_tensor_name, "physical tensor name")?;
        self.tensor.validate()?;
        if let Some(sha256) = &self.content_sha256 {
            validate_sha256(sha256)?;
        }
        Ok(())
    }
}

/// Physical bindings for every logical graph weight.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WeightBindings {
    /// One binding per logical graph weight.
    pub bindings: Vec<WeightBinding>,
}

impl WeightBindings {
    /// Validates duplicate logical binding IDs and local physical metadata.
    pub fn validate(&self) -> Result<(), String> {
        if self.bindings.len() > MAX_GRAPH_DECLARATIONS {
            return Err("weight binding count exceeds limit".into());
        }
        let mut logical_ids = BTreeSet::new();
        for binding in &self.bindings {
            binding.validate()?;
            if !logical_ids.insert(binding.logical_id.clone()) {
                return Err(format!(
                    "logical weight binding {} is duplicated",
                    binding.logical_id.as_str()
                ));
            }
        }
        Ok(())
    }

    /// Validates this binding set against declared logical weights.
    pub fn validate_against(&self, weights: &[WeightSpec]) -> Result<(), String> {
        self.validate()?;
        if weights.len() > MAX_GRAPH_DECLARATIONS {
            return Err("logical weight declaration count exceeds limit".into());
        }
        let mut declarations = BTreeMap::new();
        for weight in weights {
            weight.validate()?;
            if declarations
                .insert(weight.id.clone(), &weight.tensor)
                .is_some()
            {
                return Err(format!(
                    "logical weight declaration {} is duplicated",
                    weight.id.as_str()
                ));
            }
        }
        if declarations.len() != self.bindings.len() {
            return Err(format!(
                "logical weight declaration count {} does not match binding count {}",
                declarations.len(),
                self.bindings.len()
            ));
        }
        for binding in &self.bindings {
            let Some(expected) = declarations.get(&binding.logical_id) else {
                return Err(format!(
                    "binding references unknown logical weight {}",
                    binding.logical_id.as_str()
                ));
            };
            if *expected != &binding.tensor {
                return Err(format!(
                    "binding for weight {} has mismatched shape, format, or layout",
                    binding.logical_id.as_str()
                ));
            }
        }
        Ok(())
    }
}

/// A finite scalar greater than zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositiveF32(f32);

impl Eq for PositiveF32 {}

impl PositiveF32 {
    /// Creates a finite positive scalar.
    pub fn new(value: f32, label: &str) -> Result<Self, String> {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{label} must be finite and greater than zero"));
        }
        Ok(Self(value))
    }

    /// Returns the validated value.
    pub fn get(self) -> f32 {
        self.0
    }

    /// Revalidates this public scalar.
    pub fn validate(self, label: &str) -> Result<(), String> {
        Self::new(self.0, label).map(|_| ())
    }
}

/// Mathematical normalization family.
///
/// These variants fix the real-valued mathematical operation. Accumulation
/// precision and reduction order (for example F32 or F64 accumulation) are
/// executor or backend implementation attributes and must be fixed by
/// validation evidence; they do not change this graph semantic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationKind {
    /// Root-mean-square normalization without mean subtraction.
    ///
    /// For each vector normalized along the configured axis, this computes
    /// `inv = 1 / sqrt(mean(x^2) + epsilon)` and applies `x * inv` before the
    /// configured affine parameters. Epsilon is inside the square root.
    Rms,
    /// Layer normalization with mean subtraction and population variance.
    ///
    /// For each vector normalized along the configured axis, this computes
    /// `mean = mean(x)`, `variance = mean((x - mean)^2)`, and
    /// `inv = 1 / sqrt(variance + epsilon)`, then applies `(x - mean) * inv`
    /// before the configured affine parameters.
    Layer,
}

/// Learned affine parameters applied after normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationAffine {
    /// Multiplies normalized values by the sole learned scale vector.
    ///
    /// `GraphNode::weights` is `[scale]`.
    Scale,
    /// Multiplies normalized values by one plus the learned scale vector.
    ///
    /// This represents a unit-offset scale parameterization without tying the
    /// graph contract to a model family or checkpoint tensor name.
    /// `GraphNode::weights` is `[scale]`.
    UnitOffsetScale,
    /// Multiplies normalized values by a learned scale vector and adds a
    /// learned bias vector.
    ///
    /// `GraphNode::weights` is `[scale, bias]` in that order.
    ScaleAndBias,
}

impl NormalizationAffine {
    fn weight_count(self) -> usize {
        match self {
            Self::Scale | Self::UnitOffsetScale => 1,
            Self::ScaleAndBias => 2,
        }
    }
}

/// Logical tensor axis normalized by a normalization operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationAxis {
    /// Normalizes independently across the final logical tensor dimension.
    Last,
}

impl NormalizationAxis {
    fn validate(self) -> Result<(), String> {
        match self {
            Self::Last => Ok(()),
        }
    }
}

/// Typed activation semantic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationKind {
    /// Sigmoid linear unit.
    Silu,
    /// Gaussian error linear unit.
    Gelu,
    /// Rectified linear unit.
    Relu,
    /// Future validated activation semantic.
    Custom(String),
}

impl ActivationKind {
    /// Validates a future custom activation semantic.
    pub fn validate(&self) -> Result<(), String> {
        if let Self::Custom(value) = self {
            validate_identifier(value, "custom activation kind")?;
        }
        Ok(())
    }
}

/// Pairing of rotary coordinates within the configured prefix.
///
/// For pair index `i`, the rotation angle is
/// `theta = position / base^(2i / rotary_dim)`. For the selected coordinate
/// pair `(a, b)`, the output is `y_a = x_a cos(theta) - x_b sin(theta)` and
/// `y_b = x_a sin(theta) + x_b cos(theta)`. Coordinates after the rotary
/// prefix are unchanged. The final feature axis is head-major: it contains
/// `heads` contiguous blocks of `head_dim` coordinates. Pair index `i` and
/// frequency index `i` reset to zero for each head. Every element of the
/// explicit `positions` tensor is the position for its corresponding token
/// and is shared by all heads of that token. The angle `theta` is in radians,
/// and each head's suffix `[rotary_dim, head_dim)` is unchanged. This is a
/// mathematical graph semantic; calculation precision, reduction order, and
/// other implementation details are backend evidence concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryPairing {
    /// Pairs the rotary prefix's first half with its second half at the same
    /// index: `(a, b) = (i, rotary_dim / 2 + i)` for
    /// `0 <= i < rotary_dim / 2`.
    SplitHalf,
    /// Pairs adjacent rotary coordinates: `(a, b) = (2i, 2i + 1)` for
    /// `0 <= i < rotary_dim / 2`.
    Interleaved,
}

/// Typed graph-node semantics without a model-name or generic attribute bag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphNodeKind {
    /// Embedding lookup.
    Embedding {
        vocab_size: usize,
        hidden_size: usize,
    },
    /// Normalization with an explicit mathematical and affine contract.
    ///
    /// `GraphNode::weights` is `[scale]` for [`NormalizationAffine::Scale`]
    /// and [`NormalizationAffine::UnitOffsetScale`], or `[scale, bias]` for
    /// [`NormalizationAffine::ScaleAndBias`]. Scale and bias have the same
    /// required shape, so adapters must bind them in this order; graph
    /// validation cannot infer a reversed binding from tensor shapes.
    Norm {
        epsilon: PositiveF32,
        kind: NormalizationKind,
        affine: NormalizationAffine,
        axis: NormalizationAxis,
    },
    /// One linear projection.
    Linear { has_bias: bool },
    /// A group of projections with compatible input semantics.
    ///
    /// `GraphNode::weights` is paired with `GraphNode::outputs` by index.
    FusedLinearGroup { output_count: usize },
    /// Rotary position operation.
    ///
    /// `GraphNode::inputs` is `[values, positions]`. `values` has rank at
    /// least two and its final feature axis is `heads * head_dim`; `positions`
    /// has the same shape as `values` without that final axis. The output
    /// preserves the values tensor's shape, format, and layout. The final
    /// feature axis is head-major, with contiguous `head_dim` blocks for each
    /// head; each head resets pair and frequency indices to zero, and its
    /// suffix `[rotary_dim, head_dim)` is unchanged. Each position applies to
    /// the corresponding token across all heads and is interpreted in radians.
    RotaryPosition {
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        base: PositiveF32,
        pairing: RotaryPairing,
    },
    /// Causal grouped-query attention over prepared query, key, and value tensors.
    ///
    /// `GraphNode::inputs` is ordered `[query, key, value]` and the sole output is
    /// `context`; this operator owns no projections, normalization, RoPE, gate, or
    /// output projection. Let `L` denote all leading dimensions, `T` the
    /// penultimate token dimension, and the final axis the feature dimension:
    /// query is `L + [T, q_heads * head_dim]`, key is
    /// `L + [T, kv_heads * head_dim]`, value is
    /// `L + [T, kv_heads * value_dim]`, and context is
    /// `L + [T, q_heads * value_dim]`.
    ///
    /// Query head `h` maps to grouped-query key/value head
    /// `h / (q_heads / kv_heads)`. For each token, score is
    /// `dot(query, key) * softmax_scale`; causal softmax attends only to token
    /// positions at or before the query position within the same sequence,
    /// subtracts the row maximum before exponentiation, and computes context as
    /// the weighted value sum. With one state reference, a committed key/value
    /// prefix is placed first and the current key/value chunk is appended as a
    /// causal overlay for reads; transaction owners define commit semantics.
    ///
    /// `RowMajor` and `TokensHidden` separate sequences through leading
    /// dimensions. `PackedRagged` receives sequence boundaries from execution
    /// batch offsets. These are mathematical graph semantics; accumulation
    /// precision and reduction order require backend evidence.
    CausalGqaAttentionCore {
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: PositiveF32,
    },
    /// Dense attention with explicit head geometry.
    ///
    /// `GraphNode::weights` is ordered query, key, value, output. The node has
    /// zero or one state reference; schema composition validates its role.
    DenseAttention {
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        softmax_scale: PositiveF32,
    },
    /// Recurrent or linear attention.
    ///
    /// The sole weight is a square `state_width` projection. The first state is
    /// mandatory; a second state can represent an explicit companion such as a
    /// convolution history and is validated by schema composition.
    RecurrentAttention { state_width: usize },
    /// Pointwise activation.
    Activation { kind: ActivationKind },
    /// Gated MLP.
    ///
    /// `GraphNode::weights` is ordered gate, up, down.
    GatedMlp {
        intermediate_size: usize,
        activation: ActivationKind,
    },
    /// Residual addition.
    Residual,
    /// Final normalization with an explicit mathematical and affine contract.
    ///
    /// `GraphNode::weights` is `[scale]` for [`NormalizationAffine::Scale`]
    /// and [`NormalizationAffine::UnitOffsetScale`], or `[scale, bias]` for
    /// [`NormalizationAffine::ScaleAndBias`]. Scale and bias have the same
    /// required shape, so adapters must bind them in this order; graph
    /// validation cannot infer a reversed binding from tensor shapes.
    FinalNorm {
        epsilon: PositiveF32,
        kind: NormalizationKind,
        affine: NormalizationAffine,
        axis: NormalizationAxis,
    },
    /// Language-model head.
    LmHead { vocab_size: usize },
    /// Sampling operation.
    Sampling { top_k: usize },
}

impl GraphNodeKind {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Embedding {
                vocab_size,
                hidden_size,
            } => {
                ensure_nonzero(*vocab_size, "embedding vocab_size")?;
                ensure_nonzero(*hidden_size, "embedding hidden_size")
            }
            Self::Norm { epsilon, axis, .. } | Self::FinalNorm { epsilon, axis, .. } => {
                epsilon.validate("normalization epsilon")?;
                axis.validate()
            }
            Self::Linear { .. } | Self::Residual => Ok(()),
            Self::FusedLinearGroup { output_count } => {
                ensure_nonzero(*output_count, "fused linear output_count")
            }
            Self::RotaryPosition {
                heads,
                head_dim,
                rotary_dim,
                base,
                ..
            } => {
                ensure_nonzero(*heads, "rotary heads")?;
                ensure_nonzero(*head_dim, "rotary head_dim")?;
                ensure_nonzero(*rotary_dim, "rotary_dim")?;
                if rotary_dim % 2 != 0 {
                    return Err("rotary_dim must be even".into());
                }
                if rotary_dim > head_dim {
                    return Err("rotary_dim must not exceed rotary head_dim".into());
                }
                checked_dimension_product(*heads, *head_dim, "rotary heads * head_dim")?;
                base.validate("rotary base")
            }
            Self::DenseAttention {
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            } => validate_attention_geometry(
                *q_heads,
                *kv_heads,
                *head_dim,
                *value_dim,
                *softmax_scale,
                "dense attention",
            ),
            Self::CausalGqaAttentionCore {
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
            } => validate_attention_geometry(
                *q_heads,
                *kv_heads,
                *head_dim,
                *value_dim,
                *softmax_scale,
                "causal GQA attention core",
            ),
            Self::RecurrentAttention { state_width } => {
                ensure_nonzero(*state_width, "recurrent attention state_width")
            }
            Self::Activation { kind } => kind.validate(),
            Self::GatedMlp {
                intermediate_size,
                activation,
            } => {
                ensure_nonzero(*intermediate_size, "gated MLP intermediate_size")?;
                activation.validate()
            }
            Self::LmHead { vocab_size } => ensure_nonzero(*vocab_size, "LM head vocab_size"),
            Self::Sampling { top_k } => ensure_nonzero(*top_k, "sampling top_k"),
        }
    }

    fn validate_arity(
        &self,
        inputs: usize,
        outputs: usize,
        weights: usize,
        states: usize,
    ) -> Result<(), String> {
        let exact = |expected_inputs, expected_outputs, expected_weights, expected_states| {
            if inputs == expected_inputs
                && outputs == expected_outputs
                && weights == expected_weights
                && states == expected_states
            {
                Ok(())
            } else {
                Err(format!(
                    "node arity inputs={inputs}, outputs={outputs}, weights={weights}, states={states} does not match expected inputs={expected_inputs}, outputs={expected_outputs}, weights={expected_weights}, states={expected_states}"
                ))
            }
        };
        match self {
            Self::Embedding { .. } => exact(1, 1, 1, 0),
            Self::Norm { affine, .. } | Self::FinalNorm { affine, .. } => {
                exact(1, 1, affine.weight_count(), 0)
            }
            Self::Linear { has_bias } => exact(1, 1, if *has_bias { 2 } else { 1 }, 0),
            Self::FusedLinearGroup { output_count } => exact(1, *output_count, *output_count, 0),
            Self::RotaryPosition { .. } => exact(2, 1, 0, 0),
            Self::Activation { .. } => exact(1, 1, 0, 0),
            Self::CausalGqaAttentionCore { .. } => {
                if inputs == 3 && outputs == 1 && weights == 0 && states <= 1 {
                    Ok(())
                } else {
                    Err(
                        "causal GQA attention core requires 3 inputs, 1 output, zero weights, and zero or one state"
                            .into(),
                    )
                }
            }
            Self::DenseAttention { .. } => {
                if inputs == 1 && outputs == 1 && weights == 4 && states <= 1 {
                    Ok(())
                } else {
                    Err(
                        "dense attention requires 1 input, 1 output, exactly 4 ordered weights, and zero or one state"
                            .into(),
                    )
                }
            }
            Self::RecurrentAttention { .. } => {
                if inputs == 1 && outputs == 1 && weights == 1 && (1..=2).contains(&states) {
                    Ok(())
                } else {
                    Err(
                        "recurrent attention requires 1 input, 1 output, exactly 1 weight, and one or two states"
                            .into(),
                    )
                }
            }
            Self::GatedMlp { .. } => exact(1, 1, 3, 0),
            Self::Residual => exact(2, 1, 0, 0),
            Self::LmHead { .. } => exact(1, 1, 1, 0),
            Self::Sampling { .. } => exact(1, 1, 0, 0),
        }
    }
}

/// One topologically ordered semantic graph node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNode {
    /// Stable node ID.
    pub id: NodeId,
    /// Values read by this node.
    pub inputs: Vec<ValueId>,
    /// Values produced by this node.
    pub outputs: Vec<ValueId>,
    /// Logical weights used by this node.
    ///
    /// Operators with multiple weights use the order documented by their
    /// semantic validator. A new ordering requires a new semantic operator or
    /// attribute; it must not be encoded in a model-name branch.
    pub weights: Vec<WeightId>,
    /// State IDs used by this node.
    ///
    /// This graph enforces operator-local cardinality. [`crate::state_schema::StateSchema`] performs
    /// existence, ownership, kind, layout, and transaction validation in the
    /// composition layer.
    pub states: Vec<StateId>,
    /// Typed operation semantic.
    pub kind: GraphNodeKind,
}

impl GraphNode {
    fn validate_local(&self) -> Result<(), String> {
        self.id.validate()?;
        if self.inputs.len() > MAX_NODE_VALUES
            || self.outputs.is_empty()
            || self.outputs.len() > MAX_NODE_VALUES
            || self.weights.len() > MAX_NODE_WEIGHTS
            || self.states.len() > MAX_NODE_STATES
        {
            return Err(format!(
                "node {} exceeds reference limits",
                self.id.as_str()
            ));
        }
        validate_unique(&self.inputs, "node input value")?;
        validate_unique(&self.outputs, "node output value")?;
        validate_unique(&self.weights, "node weight")?;
        validate_unique(&self.states, "node state")?;
        self.kind.validate()?;
        self.kind.validate_arity(
            self.inputs.len(),
            self.outputs.len(),
            self.weights.len(),
            self.states.len(),
        )
    }

    fn validate_semantics(
        &self,
        values: &BTreeMap<ValueId, &TensorSpec>,
        weights: &BTreeMap<WeightId, &TensorSpec>,
    ) -> Result<(), String> {
        let value = |id: &ValueId, role: &str| {
            values.get(id).copied().ok_or_else(|| {
                format!(
                    "node {} {role} value {} is undeclared",
                    self.id.as_str(),
                    id.as_str()
                )
            })
        };
        let weight = |id: &WeightId, role: &str| {
            weights.get(id).copied().ok_or_else(|| {
                format!(
                    "node {} {role} weight {} is undeclared",
                    self.id.as_str(),
                    id.as_str()
                )
            })
        };

        match &self.kind {
            GraphNodeKind::Embedding {
                vocab_size,
                hidden_size,
            } => {
                let tokens = value(&self.inputs[0], "embedding input")?;
                let output = value(&self.outputs[0], "embedding output")?;
                let table = weight(&self.weights[0], "embedding table")?;
                if !matches!(tokens.format, NumericalFormat::U32 | NumericalFormat::U64) {
                    return Err(format!(
                        "node {} embedding input must use U32 or U64 indices",
                        self.id.as_str()
                    ));
                }
                require_embedding_output_shape(tokens, output, *hidden_size, self.id.as_str())?;
                require_matrix_shape(
                    table,
                    *vocab_size,
                    *hidden_size,
                    "embedding table",
                    self.id.as_str(),
                )
            }
            GraphNodeKind::Norm { affine, axis, .. }
            | GraphNodeKind::FinalNorm { affine, axis, .. } => {
                let input = value(&self.inputs[0], "normalization input")?;
                let output = value(&self.outputs[0], "normalization output")?;
                let scale = weight(&self.weights[0], "normalization scale")?;
                require_same_shape(
                    input,
                    output,
                    "normalization input and output",
                    self.id.as_str(),
                )?;
                require_vector_shape(
                    scale,
                    normalization_axis_width(
                        input,
                        *axis,
                        "normalization input",
                        self.id.as_str(),
                    )?,
                    "normalization scale",
                    self.id.as_str(),
                )?;
                if *affine == NormalizationAffine::ScaleAndBias {
                    require_vector_shape(
                        weight(&self.weights[1], "normalization bias")?,
                        normalization_axis_width(
                            input,
                            *axis,
                            "normalization input",
                            self.id.as_str(),
                        )?,
                        "normalization bias",
                        self.id.as_str(),
                    )?;
                }
                Ok(())
            }
            GraphNodeKind::Linear { has_bias } => {
                let input = value(&self.inputs[0], "linear input")?;
                let output = value(&self.outputs[0], "linear output")?;
                let projection = weight(&self.weights[0], "linear projection")?;
                validate_linear_projection(
                    input,
                    output,
                    projection,
                    "linear projection",
                    self.id.as_str(),
                )?;
                if *has_bias {
                    require_vector_shape(
                        weight(&self.weights[1], "linear bias")?,
                        feature_width(output, "linear output", self.id.as_str())?,
                        "linear bias",
                        self.id.as_str(),
                    )?;
                }
                Ok(())
            }
            GraphNodeKind::FusedLinearGroup { .. } => {
                let input = value(&self.inputs[0], "fused linear input")?;
                for (index, (output_id, weight_id)) in
                    self.outputs.iter().zip(&self.weights).enumerate()
                {
                    let output = value(output_id, "fused linear output")?;
                    let projection = weight(weight_id, "fused linear projection")?;
                    validate_linear_projection(
                        input,
                        output,
                        projection,
                        &format!("fused linear projection {index}"),
                        self.id.as_str(),
                    )?;
                }
                Ok(())
            }
            GraphNodeKind::RotaryPosition {
                heads,
                head_dim,
                rotary_dim,
                ..
            } => {
                let values = value(&self.inputs[0], "rotary values input")?;
                let positions = value(&self.inputs[1], "rotary positions input")?;
                let output = value(&self.outputs[0], "rotary output")?;
                if values.shape.len() < 2 {
                    return Err(format!(
                        "node {} rotary values input must have rank at least 2",
                        self.id.as_str()
                    ));
                }
                if !matches!(
                    values.format,
                    NumericalFormat::F32 | NumericalFormat::Bf16 | NumericalFormat::Fp16
                ) {
                    return Err(format!(
                        "node {} rotary values input must use F32, BF16, or FP16",
                        self.id.as_str()
                    ));
                }
                if !matches!(
                    values.layout,
                    TensorLayout::RowMajor
                        | TensorLayout::TokensHidden
                        | TensorLayout::PackedRagged
                ) {
                    return Err(format!(
                        "node {} rotary values input must use RowMajor, TokensHidden, or PackedRagged layout",
                        self.id.as_str()
                    ));
                }
                if output.shape != values.shape {
                    return Err(format!(
                        "node {} rotary output shape must match values shape, got {:?} and {:?}",
                        self.id.as_str(),
                        values.shape,
                        output.shape
                    ));
                }
                if output.format != values.format || output.layout != values.layout {
                    return Err(format!(
                        "node {} rotary output format and layout must match values input",
                        self.id.as_str()
                    ));
                }
                if !matches!(
                    positions.format,
                    NumericalFormat::U32 | NumericalFormat::U64
                ) {
                    return Err(format!(
                        "node {} rotary positions input must use U32 or U64",
                        self.id.as_str()
                    ));
                }
                if positions.layout != TensorLayout::RowMajor {
                    return Err(format!(
                        "node {} rotary positions input must use RowMajor layout",
                        self.id.as_str()
                    ));
                }
                if positions.shape.as_slice() != &values.shape[..values.shape.len() - 1] {
                    return Err(format!(
                        "node {} rotary positions shape must match values shape without the final feature axis",
                        self.id.as_str()
                    ));
                }
                let hidden = feature_width(values, "rotary values input", self.id.as_str())?;
                let expected_hidden =
                    checked_dimension_product(*heads, *head_dim, "rotary heads * head_dim")?;
                if hidden != expected_hidden {
                    return Err(format!(
                        "node {} rotary values feature width {hidden} does not match heads * head_dim {expected_hidden}",
                        self.id.as_str(),
                    ));
                }
                if *rotary_dim > *head_dim {
                    return Err(format!(
                        "node {} rotary_dim {rotary_dim} exceeds head_dim {head_dim}",
                        self.id.as_str()
                    ));
                }
                Ok(())
            }
            GraphNodeKind::CausalGqaAttentionCore {
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                ..
            } => {
                let query = value(&self.inputs[0], "causal GQA query")?;
                let key = value(&self.inputs[1], "causal GQA key")?;
                let value_tensor = value(&self.inputs[2], "causal GQA value")?;
                let output = value(&self.outputs[0], "causal GQA context")?;
                validate_causal_gqa_attention_core(
                    query,
                    key,
                    value_tensor,
                    output,
                    *q_heads,
                    *kv_heads,
                    *head_dim,
                    *value_dim,
                    self.id.as_str(),
                )
            }
            GraphNodeKind::DenseAttention {
                q_heads,
                kv_heads,
                head_dim,
                value_dim,
                ..
            } => {
                let input = value(&self.inputs[0], "dense attention input")?;
                let output = value(&self.outputs[0], "dense attention output")?;
                require_same_shape(
                    input,
                    output,
                    "dense attention input and output",
                    self.id.as_str(),
                )?;
                let hidden = feature_width(input, "dense attention input", self.id.as_str())?;
                let query_width =
                    checked_dimension_product(*q_heads, *head_dim, "dense attention query width")?;
                if hidden != query_width {
                    return Err(format!(
                        "node {} dense attention input feature width {hidden} does not match q_heads * head_dim {query_width}",
                        self.id.as_str()
                    ));
                }
                let key_width =
                    checked_dimension_product(*kv_heads, *head_dim, "dense attention key width")?;
                let value_width = checked_dimension_product(
                    *kv_heads,
                    *value_dim,
                    "dense attention value width",
                )?;
                let attended_width = checked_dimension_product(
                    *q_heads,
                    *value_dim,
                    "dense attention attended width",
                )?;
                require_matrix_shape(
                    weight(&self.weights[0], "dense attention query projection")?,
                    query_width,
                    hidden,
                    "dense attention query projection",
                    self.id.as_str(),
                )?;
                require_matrix_shape(
                    weight(&self.weights[1], "dense attention key projection")?,
                    key_width,
                    hidden,
                    "dense attention key projection",
                    self.id.as_str(),
                )?;
                require_matrix_shape(
                    weight(&self.weights[2], "dense attention value projection")?,
                    value_width,
                    hidden,
                    "dense attention value projection",
                    self.id.as_str(),
                )?;
                require_matrix_shape(
                    weight(&self.weights[3], "dense attention output projection")?,
                    hidden,
                    attended_width,
                    "dense attention output projection",
                    self.id.as_str(),
                )
            }
            GraphNodeKind::RecurrentAttention { state_width } => {
                let input = value(&self.inputs[0], "recurrent attention input")?;
                let output = value(&self.outputs[0], "recurrent attention output")?;
                require_same_shape(
                    input,
                    output,
                    "recurrent attention input and output",
                    self.id.as_str(),
                )?;
                let hidden = feature_width(input, "recurrent attention input", self.id.as_str())?;
                if hidden != *state_width {
                    return Err(format!(
                        "node {} recurrent attention input feature width {hidden} does not match state_width {state_width}",
                        self.id.as_str()
                    ));
                }
                require_matrix_shape(
                    weight(&self.weights[0], "recurrent attention projection")?,
                    *state_width,
                    *state_width,
                    "recurrent attention projection",
                    self.id.as_str(),
                )
            }
            GraphNodeKind::Activation { .. } => require_same_shape(
                value(&self.inputs[0], "activation input")?,
                value(&self.outputs[0], "activation output")?,
                "activation input and output",
                self.id.as_str(),
            ),
            GraphNodeKind::GatedMlp {
                intermediate_size, ..
            } => {
                let input = value(&self.inputs[0], "gated MLP input")?;
                let output = value(&self.outputs[0], "gated MLP output")?;
                require_same_shape(
                    input,
                    output,
                    "gated MLP input and output",
                    self.id.as_str(),
                )?;
                let hidden = feature_width(input, "gated MLP input", self.id.as_str())?;
                require_matrix_shape(
                    weight(&self.weights[0], "gated MLP gate projection")?,
                    *intermediate_size,
                    hidden,
                    "gated MLP gate projection",
                    self.id.as_str(),
                )?;
                require_matrix_shape(
                    weight(&self.weights[1], "gated MLP up projection")?,
                    *intermediate_size,
                    hidden,
                    "gated MLP up projection",
                    self.id.as_str(),
                )?;
                require_matrix_shape(
                    weight(&self.weights[2], "gated MLP down projection")?,
                    hidden,
                    *intermediate_size,
                    "gated MLP down projection",
                    self.id.as_str(),
                )
            }
            GraphNodeKind::Residual => {
                let left = value(&self.inputs[0], "residual left input")?;
                require_same_shape(
                    left,
                    value(&self.inputs[1], "residual right input")?,
                    "residual inputs",
                    self.id.as_str(),
                )?;
                require_same_shape(
                    left,
                    value(&self.outputs[0], "residual output")?,
                    "residual input and output",
                    self.id.as_str(),
                )
            }
            GraphNodeKind::LmHead { vocab_size } => {
                let input = value(&self.inputs[0], "LM head input")?;
                let output = value(&self.outputs[0], "LM head output")?;
                require_linear_value_shapes(input, output, "LM head", self.id.as_str())?;
                if feature_width(output, "LM head output", self.id.as_str())? != *vocab_size {
                    return Err(format!(
                        "node {} LM head output feature width does not match vocab_size {vocab_size}",
                        self.id.as_str()
                    ));
                }
                require_matrix_shape(
                    weight(&self.weights[0], "LM head projection")?,
                    *vocab_size,
                    feature_width(input, "LM head input", self.id.as_str())?,
                    "LM head projection",
                    self.id.as_str(),
                )
            }
            GraphNodeKind::Sampling { top_k } => {
                let logits = value(&self.inputs[0], "sampling logits")?;
                let tokens = value(&self.outputs[0], "sampling output")?;
                let vocab_size = feature_width(logits, "sampling logits", self.id.as_str())?;
                if *top_k > vocab_size {
                    return Err(format!(
                        "node {} sampling top_k {top_k} exceeds logits feature width {vocab_size}",
                        self.id.as_str()
                    ));
                }
                if logits.shape.len() < 2
                    || tokens.shape.as_slice() != &logits.shape[..logits.shape.len() - 1]
                {
                    return Err(format!(
                        "node {} sampling output shape must equal logits shape without the final vocabulary dimension",
                        self.id.as_str()
                    ));
                }
                if !matches!(tokens.format, NumericalFormat::U32 | NumericalFormat::U64) {
                    return Err(format!(
                        "node {} sampling output must use U32 or U64 token IDs",
                        self.id.as_str()
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Immutable backend-independent model topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelGraph {
    /// Stable graph identifier.
    pub graph_id: String,
    /// Values available before graph execution.
    pub inputs: Vec<ValueId>,
    /// Values required after graph execution.
    pub outputs: Vec<ValueId>,
    /// All declared graph values.
    pub values: Vec<GraphValue>,
    /// All declared logical weights.
    pub weights: Vec<WeightSpec>,
    /// Topologically ordered graph nodes.
    pub nodes: Vec<GraphNode>,
}

impl ModelGraph {
    /// Validates graph topology and logical declarations without physical bindings.
    pub fn validate(&self) -> Result<(), String> {
        validate_identifier(&self.graph_id, "model graph ID")?;
        if self.inputs.is_empty() || self.outputs.is_empty() || self.nodes.is_empty() {
            return Err("model graph requires inputs, outputs, and nodes".into());
        }
        if self.inputs.len() > MAX_GRAPH_ENDPOINTS || self.outputs.len() > MAX_GRAPH_ENDPOINTS {
            return Err("model graph endpoint count exceeds limit".into());
        }
        if self.values.len() > MAX_GRAPH_DECLARATIONS
            || self.weights.len() > MAX_GRAPH_DECLARATIONS
            || self.nodes.len() > MAX_GRAPH_DECLARATIONS
        {
            return Err("model graph declaration count exceeds limit".into());
        }

        let mut values = BTreeMap::new();
        for value in &self.values {
            value.validate()?;
            if values.insert(value.id.clone(), &value.tensor).is_some() {
                return Err(format!("graph value {} is duplicated", value.id.as_str()));
            }
        }
        let mut weights = BTreeMap::new();
        for weight in &self.weights {
            weight.validate()?;
            if weights.insert(weight.id.clone(), &weight.tensor).is_some() {
                return Err(format!("graph weight {} is duplicated", weight.id.as_str()));
            }
        }
        validate_unique(&self.inputs, "graph input")?;
        validate_unique(&self.outputs, "graph output")?;
        for input in &self.inputs {
            if !values.contains_key(input) {
                return Err(format!("graph input {} is undeclared", input.as_str()));
            }
        }

        let mut available = self.inputs.iter().cloned().collect::<BTreeSet<_>>();
        let mut produced = BTreeSet::new();
        let mut node_ids = BTreeSet::new();
        for node in &self.nodes {
            node.validate_local()?;
            if !node_ids.insert(node.id.clone()) {
                return Err(format!("graph node {} is duplicated", node.id.as_str()));
            }
            for input in &node.inputs {
                if !values.contains_key(input) {
                    return Err(format!(
                        "node {} input {} is undeclared",
                        node.id.as_str(),
                        input.as_str()
                    ));
                }
                if !available.contains(input) {
                    return Err(format!(
                        "node {} input {} is unavailable in topological order",
                        node.id.as_str(),
                        input.as_str()
                    ));
                }
            }
            for weight in &node.weights {
                if !weights.contains_key(weight) {
                    return Err(format!(
                        "node {} references unknown weight {}",
                        node.id.as_str(),
                        weight.as_str()
                    ));
                }
            }
            for state in &node.states {
                state.validate()?;
            }
            for output in &node.outputs {
                if !values.contains_key(output) {
                    return Err(format!(
                        "node {} output {} is undeclared",
                        node.id.as_str(),
                        output.as_str()
                    ));
                }
                if available.contains(output) || !produced.insert(output.clone()) {
                    return Err(format!(
                        "graph value {} has duplicate producer or overwrites input",
                        output.as_str()
                    ));
                }
            }
            node.validate_semantics(&values, &weights)?;
            available.extend(node.outputs.iter().cloned());
        }
        for output in &self.outputs {
            if !produced.contains(output) {
                return Err(format!(
                    "final graph output {} is unavailable",
                    output.as_str()
                ));
            }
        }
        Ok(())
    }

    /// Validates both graph declarations and physical logical-weight bindings.
    pub fn validate_with_bindings(&self, bindings: &WeightBindings) -> Result<(), String> {
        self.validate()?;
        bindings.validate_against(&self.weights)
    }
}

fn validate_causal_gqa_attention_core(
    query: &TensorSpec,
    key: &TensorSpec,
    value: &TensorSpec,
    output: &TensorSpec,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    node_id: &str,
) -> Result<(), String> {
    let rank = query.shape.len();
    if rank < 2 || key.shape.len() < 2 || value.shape.len() < 2 || output.shape.len() < 2 {
        return Err(format!(
            "node {node_id} causal GQA query, key, value, and context must have rank at least 2"
        ));
    }
    if key.shape.len() != rank || value.shape.len() != rank || output.shape.len() != rank {
        return Err(format!(
            "node {node_id} causal GQA query, key, value, and context must have equal rank"
        ));
    }
    if key.shape[..rank - 1] != query.shape[..rank - 1]
        || value.shape[..rank - 1] != query.shape[..rank - 1]
        || output.shape[..rank - 1] != query.shape[..rank - 1]
    {
        return Err(format!(
            "node {node_id} causal GQA query, key, value, and context leading shapes must match"
        ));
    }
    let query_width = checked_dimension_product(q_heads, head_dim, "causal GQA query width")?;
    let key_width = checked_dimension_product(kv_heads, head_dim, "causal GQA key width")?;
    let value_width = checked_dimension_product(kv_heads, value_dim, "causal GQA value width")?;
    let context_width = checked_dimension_product(q_heads, value_dim, "causal GQA context width")?;
    if feature_width(query, "causal GQA query", node_id)? != query_width {
        return Err(format!(
            "node {node_id} causal GQA query feature width must equal q_heads * head_dim {query_width}"
        ));
    }
    if feature_width(key, "causal GQA key", node_id)? != key_width {
        return Err(format!(
            "node {node_id} causal GQA key feature width must equal kv_heads * head_dim {key_width}"
        ));
    }
    if feature_width(value, "causal GQA value", node_id)? != value_width {
        return Err(format!(
            "node {node_id} causal GQA value feature width must equal kv_heads * value_dim {value_width}"
        ));
    }
    if feature_width(output, "causal GQA context", node_id)? != context_width {
        return Err(format!(
            "node {node_id} causal GQA context feature width must equal q_heads * value_dim {context_width}"
        ));
    }
    if !matches!(
        query.format,
        NumericalFormat::F32 | NumericalFormat::Bf16 | NumericalFormat::Fp16
    ) {
        return Err(format!(
            "node {node_id} causal GQA tensors must use F32, BF16, or FP16"
        ));
    }
    if key.format != query.format || value.format != query.format || output.format != query.format {
        return Err(format!(
            "node {node_id} causal GQA query, key, value, and context formats must match"
        ));
    }
    if !matches!(
        query.layout,
        TensorLayout::RowMajor | TensorLayout::TokensHidden | TensorLayout::PackedRagged
    ) {
        return Err(format!(
            "node {node_id} causal GQA tensors use an unsupported layout"
        ));
    }
    if key.layout != query.layout || value.layout != query.layout || output.layout != query.layout {
        return Err(format!(
            "node {node_id} causal GQA query, key, value, and context layouts must match"
        ));
    }
    Ok(())
}

fn require_embedding_output_shape(
    tokens: &TensorSpec,
    output: &TensorSpec,
    hidden_size: usize,
    node_id: &str,
) -> Result<(), String> {
    let Some(expected_rank) = tokens.shape.len().checked_add(1) else {
        return Err(format!("node {node_id} embedding input rank is too large"));
    };
    if output.shape.len() != expected_rank
        || output.shape[..tokens.shape.len()] != tokens.shape
        || output.shape.last().copied() != Some(hidden_size)
    {
        return Err(format!(
            "node {node_id} embedding output shape must equal input token shape followed by hidden_size {hidden_size}"
        ));
    }
    Ok(())
}

fn require_same_shape(
    left: &TensorSpec,
    right: &TensorSpec,
    label: &str,
    node_id: &str,
) -> Result<(), String> {
    if left.shape != right.shape {
        return Err(format!(
            "node {node_id} {label} must have identical shapes, got {:?} and {:?}",
            left.shape, right.shape
        ));
    }
    Ok(())
}

fn require_linear_value_shapes(
    input: &TensorSpec,
    output: &TensorSpec,
    label: &str,
    node_id: &str,
) -> Result<(), String> {
    if input.shape.len() != output.shape.len()
        || input.shape[..input.shape.len() - 1] != output.shape[..output.shape.len() - 1]
    {
        return Err(format!(
            "node {node_id} {label} input and output must have equal leading dimensions"
        ));
    }
    Ok(())
}

fn validate_linear_projection(
    input: &TensorSpec,
    output: &TensorSpec,
    projection: &TensorSpec,
    label: &str,
    node_id: &str,
) -> Result<(), String> {
    require_linear_value_shapes(input, output, label, node_id)?;
    require_matrix_shape(
        projection,
        feature_width(output, "linear output", node_id)?,
        feature_width(input, "linear input", node_id)?,
        label,
        node_id,
    )
}

fn require_matrix_shape(
    tensor: &TensorSpec,
    rows: usize,
    columns: usize,
    label: &str,
    node_id: &str,
) -> Result<(), String> {
    if tensor.shape.len() != 2 || tensor.shape[0] != rows || tensor.shape[1] != columns {
        return Err(format!(
            "node {node_id} {label} must have logical shape [{rows}, {columns}], got {:?}",
            tensor.shape
        ));
    }
    Ok(())
}

fn require_vector_shape(
    tensor: &TensorSpec,
    width: usize,
    label: &str,
    node_id: &str,
) -> Result<(), String> {
    if tensor.shape.len() != 1 || tensor.shape[0] != width {
        return Err(format!(
            "node {node_id} {label} must have logical shape [{width}], got {:?}",
            tensor.shape
        ));
    }
    Ok(())
}

fn feature_width(tensor: &TensorSpec, label: &str, node_id: &str) -> Result<usize, String> {
    tensor
        .shape
        .last()
        .copied()
        .ok_or_else(|| format!("node {node_id} {label} must have at least one logical dimension"))
}

fn normalization_axis_width(
    tensor: &TensorSpec,
    axis: NormalizationAxis,
    label: &str,
    node_id: &str,
) -> Result<usize, String> {
    match axis {
        NormalizationAxis::Last => feature_width(tensor, label, node_id),
    }
}

fn validate_attention_geometry(
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: PositiveF32,
    label: &str,
) -> Result<(), String> {
    ensure_nonzero(q_heads, &format!("{label} q_heads"))?;
    ensure_nonzero(kv_heads, &format!("{label} kv_heads"))?;
    ensure_nonzero(head_dim, &format!("{label} head_dim"))?;
    ensure_nonzero(value_dim, &format!("{label} value_dim"))?;
    if q_heads % kv_heads != 0 {
        return Err(format!("{label} q_heads must divide evenly by kv_heads"));
    }
    checked_dimension_product(q_heads, head_dim, &format!("{label} query width"))?;
    checked_dimension_product(kv_heads, head_dim, &format!("{label} key width"))?;
    checked_dimension_product(kv_heads, value_dim, &format!("{label} value width"))?;
    checked_dimension_product(q_heads, value_dim, &format!("{label} context width"))?;
    softmax_scale.validate(&format!("{label} softmax_scale"))
}

fn checked_dimension_product(left: usize, right: usize, label: &str) -> Result<usize, String> {
    let product = left
        .checked_mul(right)
        .ok_or_else(|| format!("{label} overflows usize"))?;
    if (product as u128) > u128::from(MAX_TENSOR_LOGICAL_ELEMENTS) {
        return Err(format!(
            "{label} {product} exceeds tensor logical element limit {MAX_TENSOR_LOGICAL_ELEMENTS}"
        ));
    }
    Ok(product)
}

fn ensure_nonzero(value: usize, label: &str) -> Result<(), String> {
    if value == 0 {
        Err(format!("{label} must be greater than zero"))
    } else {
        Ok(())
    }
}

fn validate_unique<T>(values: &[T], label: &str) -> Result<(), String>
where
    T: Ord + Clone + std::fmt::Debug,
{
    let mut set = BTreeSet::new();
    for value in values {
        if !set.insert(value.clone()) {
            return Err(format!("{label} {value:?} is duplicated"));
        }
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte <= b'f'))
    {
        return Err("weight binding SHA-256 must be 64 lowercase hexadecimal bytes".into());
    }
    Ok(())
}

fn is_builtin_numerical_format_id(value: &str) -> bool {
    canonical_format_id(value).is_some()
        || ["F32", "BF16", "FP16", "U32", "U64"]
            .iter()
            .any(|builtin| value.eq_ignore_ascii_case(builtin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value_id(name: &str) -> ValueId {
        ValueId::new(name).unwrap()
    }

    fn weight_id(name: &str) -> WeightId {
        WeightId::new(name).unwrap()
    }

    fn state_id(name: &str) -> StateId {
        StateId::new(name).unwrap()
    }

    fn node_id(name: &str) -> NodeId {
        NodeId::new(name).unwrap()
    }

    fn spec(shape: &[usize]) -> TensorSpec {
        TensorSpec::new(
            shape.to_vec(),
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
        )
        .unwrap()
    }

    fn index_spec(shape: &[usize]) -> TensorSpec {
        TensorSpec::new(
            shape.to_vec(),
            NumericalFormat::U32,
            TensorLayout::PackedRagged,
        )
        .unwrap()
    }

    fn rotary_spec(shape: &[usize], format: NumericalFormat, layout: TensorLayout) -> TensorSpec {
        TensorSpec::new(shape.to_vec(), format, layout).unwrap()
    }

    fn value(name: &str, shape: &[usize]) -> GraphValue {
        GraphValue {
            id: value_id(name),
            tensor: spec(shape),
        }
    }

    fn index_value(name: &str, shape: &[usize]) -> GraphValue {
        GraphValue {
            id: value_id(name),
            tensor: index_spec(shape),
        }
    }

    fn weight(name: &str, shape: &[usize]) -> WeightSpec {
        WeightSpec {
            id: weight_id(name),
            tensor: spec(shape),
        }
    }

    fn bindings(weights: &[WeightSpec]) -> WeightBindings {
        WeightBindings {
            bindings: weights
                .iter()
                .map(|weight| WeightBinding {
                    logical_id: weight.id.clone(),
                    physical_tensor_name: format!("pkg.{}", weight.id.as_str()),
                    tensor: weight.tensor.clone(),
                    content_sha256: Some("a".repeat(64)),
                })
                .collect(),
        }
    }

    fn unary_graph(
        input: GraphValue,
        outputs: Vec<GraphValue>,
        weights: Vec<WeightSpec>,
        states: Vec<StateId>,
        kind: GraphNodeKind,
    ) -> ModelGraph {
        let input_id = input.id.clone();
        let output_ids = outputs
            .iter()
            .map(|value| value.id.clone())
            .collect::<Vec<_>>();
        let graph_output = output_ids[0].clone();
        let weight_ids = weights
            .iter()
            .map(|weight| weight.id.clone())
            .collect::<Vec<_>>();
        let mut values = vec![input];
        values.extend(outputs);
        ModelGraph {
            graph_id: "single-node".into(),
            inputs: vec![input_id.clone()],
            outputs: vec![graph_output],
            values,
            weights,
            nodes: vec![GraphNode {
                id: node_id("node"),
                inputs: vec![input_id],
                outputs: output_ids,
                weights: weight_ids,
                states,
                kind,
            }],
        }
    }

    fn dense_graph() -> ModelGraph {
        let weights = vec![
            weight("embed", &[1024, 32]),
            weight("norm", &[32]),
            weight("q", &[32, 32]),
            weight("k", &[16, 32]),
            weight("v", &[16, 32]),
            weight("o", &[32, 32]),
            weight("gate", &[64, 32]),
            weight("up", &[64, 32]),
            weight("down", &[32, 64]),
        ];
        ModelGraph {
            graph_id: "qwen3-style-dense".into(),
            inputs: vec![value_id("tokens")],
            outputs: vec![value_id("out")],
            values: vec![
                index_value("tokens", &[16]),
                value("hidden", &[16, 32]),
                value("normed", &[16, 32]),
                value("attended", &[16, 32]),
                value("out", &[16, 32]),
            ],
            weights,
            nodes: vec![
                GraphNode {
                    id: node_id("embedding"),
                    inputs: vec![value_id("tokens")],
                    outputs: vec![value_id("hidden")],
                    weights: vec![weight_id("embed")],
                    states: vec![],
                    kind: GraphNodeKind::Embedding {
                        vocab_size: 1024,
                        hidden_size: 32,
                    },
                },
                GraphNode {
                    id: node_id("norm"),
                    inputs: vec![value_id("hidden")],
                    outputs: vec![value_id("normed")],
                    weights: vec![weight_id("norm")],
                    states: vec![],
                    kind: GraphNodeKind::Norm {
                        epsilon: PositiveF32::new(1e-5, "epsilon").unwrap(),
                        kind: NormalizationKind::Rms,
                        affine: NormalizationAffine::Scale,
                        axis: NormalizationAxis::Last,
                    },
                },
                GraphNode {
                    id: node_id("attention"),
                    inputs: vec![value_id("normed")],
                    outputs: vec![value_id("attended")],
                    weights: vec![
                        weight_id("q"),
                        weight_id("k"),
                        weight_id("v"),
                        weight_id("o"),
                    ],
                    states: vec![state_id("layer0.kv")],
                    kind: GraphNodeKind::DenseAttention {
                        q_heads: 4,
                        kv_heads: 2,
                        head_dim: 8,
                        value_dim: 8,
                        softmax_scale: PositiveF32::new(0.25, "scale").unwrap(),
                    },
                },
                GraphNode {
                    id: node_id("mlp"),
                    inputs: vec![value_id("attended")],
                    outputs: vec![value_id("out")],
                    weights: vec![weight_id("gate"), weight_id("up"), weight_id("down")],
                    states: vec![],
                    kind: GraphNodeKind::GatedMlp {
                        intermediate_size: 64,
                        activation: ActivationKind::Silu,
                    },
                },
            ],
        }
    }

    fn hybrid_graph() -> ModelGraph {
        let mut graph = dense_graph();
        graph.graph_id = "qwen35-style-hybrid".into();
        graph.values.insert(3, value("recurrent", &[16, 32]));
        graph.nodes.insert(
            2,
            GraphNode {
                id: node_id("recurrent"),
                inputs: vec![value_id("normed")],
                outputs: vec![value_id("recurrent")],
                weights: vec![weight_id("q")],
                states: vec![state_id("layer0.recurrent")],
                kind: GraphNodeKind::RecurrentAttention { state_width: 32 },
            },
        );
        graph.nodes[3].inputs = vec![value_id("recurrent")];
        graph
    }

    #[test]
    fn causal_gqa_core_accepts_rank2_stateless_and_rank3_stateful_packed_graphs() {
        causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        )
        .validate()
        .unwrap();

        causal_gqa_graph(
            &[2, 3, 4],
            &[2, 3, 2],
            &[2, 3, 2],
            &[2, 3, 4],
            NumericalFormat::Bf16,
            TensorLayout::PackedRagged,
            vec![state_id("kv")],
            4,
            2,
            1,
            1,
        )
        .validate()
        .unwrap();
    }

    #[test]
    fn causal_gqa_core_accepts_distinct_value_dimension() {
        causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 3],
            &[2, 6],
            NumericalFormat::Fp16,
            TensorLayout::RowMajor,
            vec![],
            2,
            1,
            2,
            3,
        )
        .validate()
        .unwrap();
    }

    #[test]
    fn causal_gqa_core_rejects_zero_divisibility_and_product_overflow() {
        let mut zero = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        if let GraphNodeKind::CausalGqaAttentionCore { q_heads, .. } = &mut zero.nodes[0].kind {
            *q_heads = 0;
        }
        assert!(zero.validate().unwrap_err().contains("q_heads"));

        let indivisible = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            3,
            2,
            1,
            1,
        );
        assert!(
            indivisible
                .validate()
                .unwrap_err()
                .contains("divide evenly")
        );

        let overflow = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            usize::MAX,
            1,
            2,
            1,
        );
        assert!(overflow.validate().unwrap_err().contains("overflows usize"));
    }

    #[test]
    fn causal_gqa_core_rejects_arity_and_invalid_state_cardinality() {
        let mut inputs = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        inputs.nodes[0].inputs.pop();
        assert!(inputs.validate().unwrap_err().contains("3 inputs"));

        let mut outputs = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        outputs.nodes[0].outputs.push(value_id("unexpected"));
        assert!(outputs.validate().unwrap_err().contains("1 output"));

        let mut weights = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        weights.nodes[0].weights.push(weight_id("unexpected"));
        assert!(weights.validate().unwrap_err().contains("zero weights"));

        let states = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![state_id("kv"), state_id("other")],
            4,
            2,
            1,
            1,
        );
        assert!(states.validate().unwrap_err().contains("zero or one state"));
    }

    #[test]
    fn causal_gqa_core_rejects_rank_leading_token_and_feature_shape_mismatches() {
        let mut rank = causal_gqa_graph(
            &[2, 3, 4],
            &[2, 3, 2],
            &[2, 3, 2],
            &[2, 3, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        rank.values[1].tensor.shape = vec![2, 3, 2, 1];
        assert!(rank.validate().unwrap_err().contains("equal rank"));

        let mut leading = causal_gqa_graph(
            &[2, 3, 4],
            &[2, 3, 2],
            &[2, 3, 2],
            &[2, 3, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        leading.values[1].tensor.shape = vec![3, 3, 2];
        assert!(
            leading
                .validate()
                .unwrap_err()
                .contains("leading shapes must match")
        );

        let mut token = causal_gqa_graph(
            &[2, 3, 4],
            &[2, 3, 2],
            &[2, 3, 2],
            &[2, 3, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        token.values[2].tensor.shape = vec![2, 4, 2];
        assert!(
            token
                .validate()
                .unwrap_err()
                .contains("leading shapes must match")
        );

        let mut query_width = causal_gqa_graph(
            &[2, 3, 4],
            &[2, 3, 2],
            &[2, 3, 2],
            &[2, 3, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        query_width.values[0].tensor.shape = vec![2, 3, 3];
        assert!(
            query_width
                .validate()
                .unwrap_err()
                .contains("query feature width")
        );

        let mut key_width = causal_gqa_graph(
            &[2, 3, 4],
            &[2, 3, 2],
            &[2, 3, 2],
            &[2, 3, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        key_width.values[1].tensor.shape = vec![2, 3, 1];
        assert!(
            key_width
                .validate()
                .unwrap_err()
                .contains("key feature width")
        );

        let mut value_width = causal_gqa_graph(
            &[2, 3, 4],
            &[2, 3, 2],
            &[2, 3, 2],
            &[2, 3, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        value_width.values[2].tensor.shape = vec![2, 3, 1];
        assert!(
            value_width
                .validate()
                .unwrap_err()
                .contains("value feature width")
        );

        let mut context_width = causal_gqa_graph(
            &[2, 3, 4],
            &[2, 3, 2],
            &[2, 3, 2],
            &[2, 3, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        context_width.values[3].tensor.shape = vec![2, 3, 3];
        assert!(
            context_width
                .validate()
                .unwrap_err()
                .contains("context feature width")
        );
    }

    #[test]
    fn causal_gqa_core_rejects_format_and_layout_mismatches() {
        let mut format_mismatch = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        format_mismatch.values[1].tensor.format = NumericalFormat::Bf16;
        assert!(
            format_mismatch
                .validate()
                .unwrap_err()
                .contains("formats must match")
        );

        for format in [
            NumericalFormat::U32,
            NumericalFormat::Aq4_0,
            NumericalFormat::Sq8_0,
        ] {
            let mut invalid = causal_gqa_graph(
                &[2, 4],
                &[2, 2],
                &[2, 2],
                &[2, 4],
                NumericalFormat::F32,
                TensorLayout::TokensHidden,
                vec![],
                4,
                2,
                1,
                1,
            );
            invalid.values[0].tensor.format = format;
            assert!(
                invalid
                    .validate()
                    .unwrap_err()
                    .contains("F32, BF16, or FP16")
            );
        }

        let mut layout_mismatch = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        layout_mismatch.values[2].tensor.layout = TensorLayout::RowMajor;
        assert!(
            layout_mismatch
                .validate()
                .unwrap_err()
                .contains("layouts must match")
        );

        let mut output_format = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        output_format.values[3].tensor.format = NumericalFormat::Fp16;
        assert!(
            output_format
                .validate()
                .unwrap_err()
                .contains("formats must match")
        );

        let mut output_layout = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        output_layout.values[3].tensor.layout = TensorLayout::RowMajor;
        assert!(
            output_layout
                .validate()
                .unwrap_err()
                .contains("layouts must match")
        );

        let mut custom = causal_gqa_graph(
            &[2, 4],
            &[2, 2],
            &[2, 2],
            &[2, 4],
            NumericalFormat::F32,
            TensorLayout::TokensHidden,
            vec![],
            4,
            2,
            1,
            1,
        );
        custom.values[0].tensor.layout = TensorLayout::custom("strided").unwrap();
        assert!(
            custom
                .validate()
                .unwrap_err()
                .contains("unsupported layout")
        );
    }

    #[test]
    fn model_graph_dense_qwen3_style_validates() {
        let graph = dense_graph();
        graph
            .validate_with_bindings(&bindings(&graph.weights))
            .unwrap();
    }

    #[test]
    fn model_graph_hybrid_qwen35_style_uses_same_types() {
        let graph = hybrid_graph();
        graph
            .validate_with_bindings(&bindings(&graph.weights))
            .unwrap();
    }

    #[test]
    fn model_graph_rejects_duplicate_producer() {
        let mut graph = dense_graph();
        graph.nodes[3].outputs = vec![value_id("attended")];
        assert!(graph.validate().unwrap_err().contains("duplicate producer"));
    }

    #[test]
    fn model_graph_rejects_missing_or_unavailable_input() {
        let mut graph = dense_graph();
        graph.nodes[1].inputs = vec![value_id("missing")];
        assert!(graph.validate().unwrap_err().contains("undeclared"));
        graph.nodes[1].inputs = vec![value_id("attended")];
        assert!(graph.validate().unwrap_err().contains("unavailable"));
    }

    #[test]
    fn model_graph_rejects_bad_head_or_epsilon() {
        assert!(PositiveF32::new(f32::NAN, "epsilon").is_err());
        let mut graph = dense_graph();
        graph.nodes[2].kind = GraphNodeKind::DenseAttention {
            q_heads: 3,
            kv_heads: 2,
            head_dim: 8,
            value_dim: 8,
            softmax_scale: PositiveF32::new(0.25, "scale").unwrap(),
        };
        assert!(graph.validate().unwrap_err().contains("divide evenly"));
    }

    #[test]
    fn model_graph_rejects_binding_mismatch_and_bad_sha() {
        let graph = dense_graph();
        let mut bound = bindings(&graph.weights);
        bound.bindings[0].tensor.format = NumericalFormat::Bf16;
        assert!(
            graph
                .validate_with_bindings(&bound)
                .unwrap_err()
                .contains("mismatched")
        );
        let mut bound = bindings(&graph.weights);
        bound.bindings[0].content_sha256 = Some("ABC".into());
        assert!(bound.validate().unwrap_err().contains("SHA-256"));
    }

    #[test]
    fn model_graph_rejects_tensor_overflow_and_invalid_custom_ids() {
        assert!(
            TensorSpec::new(
                vec![usize::MAX, 2],
                NumericalFormat::F32,
                TensorLayout::RowMajor
            )
            .is_err()
        );
        let over_limit = usize::try_from(MAX_TENSOR_LOGICAL_ELEMENTS + 1).unwrap();
        assert!(
            TensorSpec::new(
                vec![over_limit],
                NumericalFormat::F32,
                TensorLayout::RowMajor
            )
            .unwrap_err()
            .contains("logical element count")
        );
        assert!(NumericalFormat::custom("future format").is_err());
        assert!(NumericalFormat::custom("aq4_0").is_err());
        assert!(NumericalFormat::custom("f32").is_err());
        assert!(NumericalFormat::Custom("SQ8_0".into()).validate().is_err());
        assert!(TensorLayout::custom("future-layout").is_ok());
    }

    #[test]
    fn model_graph_rejects_semantic_weight_shape_mismatches() {
        let mut graph = dense_graph();
        graph.values[0].tensor = spec(&[16]);
        assert!(graph.validate().unwrap_err().contains("U32 or U64 indices"));

        let mut graph = dense_graph();
        graph.weights[0].tensor = spec(&[32, 32]);
        assert!(graph.validate().unwrap_err().contains("embedding table"));

        let mut graph = dense_graph();
        graph.weights[1].tensor = spec(&[32, 1]);
        assert!(
            graph
                .validate()
                .unwrap_err()
                .contains("normalization scale")
        );

        let mut graph = dense_graph();
        graph.weights[2].tensor = spec(&[31, 32]);
        assert!(graph.validate().unwrap_err().contains("query projection"));

        let mut graph = dense_graph();
        graph.weights[6].tensor = spec(&[32, 32]);
        assert!(graph.validate().unwrap_err().contains("gate projection"));
    }

    #[test]
    fn model_graph_normalization_affines_validate_arity_and_bias_shape() {
        let normalization = |kind, affine| GraphNodeKind::Norm {
            epsilon: PositiveF32::new(1e-5, "epsilon").unwrap(),
            kind,
            affine,
            axis: NormalizationAxis::Last,
        };
        let graph = |kind, affine, weights| {
            unary_graph(
                value("input", &[2, 4]),
                vec![value("output", &[2, 4])],
                weights,
                vec![],
                normalization(kind, affine),
            )
        };

        graph(
            NormalizationKind::Rms,
            NormalizationAffine::Scale,
            vec![weight("scale", &[4])],
        )
        .validate()
        .unwrap();
        graph(
            NormalizationKind::Rms,
            NormalizationAffine::UnitOffsetScale,
            vec![weight("scale", &[4])],
        )
        .validate()
        .unwrap();
        graph(
            NormalizationKind::Layer,
            NormalizationAffine::ScaleAndBias,
            vec![weight("scale", &[4]), weight("bias", &[4])],
        )
        .validate()
        .unwrap();

        let scale_extra_weight = graph(
            NormalizationKind::Rms,
            NormalizationAffine::Scale,
            vec![weight("scale", &[4]), weight("unexpected-bias", &[4])],
        );
        assert!(
            scale_extra_weight
                .validate()
                .unwrap_err()
                .contains("weights=2")
        );

        let unit_offset_extra_weight = graph(
            NormalizationKind::Rms,
            NormalizationAffine::UnitOffsetScale,
            vec![weight("scale", &[4]), weight("unexpected-bias", &[4])],
        );
        assert!(
            unit_offset_extra_weight
                .validate()
                .unwrap_err()
                .contains("weights=2")
        );

        let scale_and_bias_missing_bias = graph(
            NormalizationKind::Layer,
            NormalizationAffine::ScaleAndBias,
            vec![weight("scale", &[4])],
        );
        assert!(
            scale_and_bias_missing_bias
                .validate()
                .unwrap_err()
                .contains("weights=1")
        );

        let mut scale_and_bias_bad_bias = graph(
            NormalizationKind::Layer,
            NormalizationAffine::ScaleAndBias,
            vec![weight("scale", &[4]), weight("bias", &[4])],
        );
        scale_and_bias_bad_bias.weights[1].tensor = spec(&[3]);
        assert!(
            scale_and_bias_bad_bias
                .validate()
                .unwrap_err()
                .contains("normalization bias")
        );
    }

    #[test]
    fn model_graph_final_norm_scale_and_bias_validates_bias_shape() {
        let mut graph = unary_graph(
            value("input", &[2, 4]),
            vec![value("output", &[2, 4])],
            vec![weight("scale", &[4]), weight("bias", &[4])],
            vec![],
            GraphNodeKind::FinalNorm {
                epsilon: PositiveF32::new(1e-5, "epsilon").unwrap(),
                kind: NormalizationKind::Layer,
                affine: NormalizationAffine::ScaleAndBias,
                axis: NormalizationAxis::Last,
            },
        );
        graph.validate().unwrap();

        graph.weights[1].tensor = spec(&[3]);
        assert!(graph.validate().unwrap_err().contains("normalization bias"));
    }

    #[test]
    fn model_graph_rejects_linear_fused_final_norm_and_lm_head_shape_mismatches() {
        let mut graph = unary_graph(
            value("input", &[2, 4]),
            vec![value("output", &[2, 3])],
            vec![weight("projection", &[3, 4])],
            vec![],
            GraphNodeKind::Linear { has_bias: false },
        );
        graph.weights[0].tensor = spec(&[4, 3]);
        assert!(graph.validate().unwrap_err().contains("linear projection"));

        let mut graph = unary_graph(
            value("input", &[2, 4]),
            vec![value("output", &[2, 3])],
            vec![weight("projection", &[3, 4]), weight("bias", &[3])],
            vec![],
            GraphNodeKind::Linear { has_bias: true },
        );
        graph.weights[1].tensor = spec(&[2]);
        assert!(graph.validate().unwrap_err().contains("linear bias"));

        let mut graph = unary_graph(
            value("input", &[2, 4]),
            vec![value("first", &[2, 3]), value("second", &[2, 5])],
            vec![
                weight("first-weight", &[3, 4]),
                weight("second-weight", &[5, 4]),
            ],
            vec![],
            GraphNodeKind::FusedLinearGroup { output_count: 2 },
        );
        graph.weights[1].tensor = spec(&[4, 5]);
        assert!(
            graph
                .validate()
                .unwrap_err()
                .contains("fused linear projection 1")
        );

        let mut graph = unary_graph(
            value("input", &[2, 4]),
            vec![value("output", &[2, 4])],
            vec![weight("scale", &[4])],
            vec![],
            GraphNodeKind::FinalNorm {
                epsilon: PositiveF32::new(1e-5, "epsilon").unwrap(),
                kind: NormalizationKind::Rms,
                affine: NormalizationAffine::Scale,
                axis: NormalizationAxis::Last,
            },
        );
        graph.weights[0].tensor = spec(&[2, 2]);
        assert!(
            graph
                .validate()
                .unwrap_err()
                .contains("normalization scale")
        );

        let mut graph = unary_graph(
            value("input", &[2, 4]),
            vec![value("logits", &[2, 10])],
            vec![weight("head", &[10, 4])],
            vec![],
            GraphNodeKind::LmHead { vocab_size: 10 },
        );
        graph.weights[0].tensor = spec(&[9, 4]);
        assert!(graph.validate().unwrap_err().contains("LM head projection"));
    }

    #[test]
    fn model_graph_rejects_invalid_state_cardinality_and_recurrent_shape() {
        let mut graph = dense_graph();
        graph.nodes[0].states = vec![state_id("unexpected")];
        assert!(graph.validate().unwrap_err().contains("states=1"));

        let mut graph = dense_graph();
        graph.nodes[2].states.clear();
        graph.validate().unwrap();

        let mut graph = dense_graph();
        graph.nodes[2].states.push(state_id("layer0.position"));
        assert!(graph.validate().unwrap_err().contains("zero or one state"));

        let mut graph = dense_graph();
        graph.nodes[2].states.push(state_id("layer0.kv"));
        assert!(graph.validate().unwrap_err().contains("is duplicated"));

        let mut graph = hybrid_graph();
        graph.nodes[2].states.clear();
        assert!(graph.validate().unwrap_err().contains("one or two states"));

        let mut graph = hybrid_graph();
        graph.nodes[2].states.push(state_id("layer0.conv"));
        graph.validate().unwrap();

        let mut graph = hybrid_graph();
        graph.weights[2].tensor = spec(&[31, 32]);
        assert!(
            graph
                .validate()
                .unwrap_err()
                .contains("recurrent attention projection")
        );
    }

    fn rotary_graph(pairing: RotaryPairing) -> ModelGraph {
        ModelGraph {
            graph_id: "rotary-contract".into(),
            inputs: vec![value_id("values"), value_id("positions")],
            outputs: vec![value_id("output")],
            values: vec![
                GraphValue {
                    id: value_id("values"),
                    tensor: rotary_spec(&[2, 8], NumericalFormat::F32, TensorLayout::RowMajor),
                },
                GraphValue {
                    id: value_id("positions"),
                    tensor: rotary_spec(&[2], NumericalFormat::U32, TensorLayout::RowMajor),
                },
                GraphValue {
                    id: value_id("output"),
                    tensor: rotary_spec(&[2, 8], NumericalFormat::F32, TensorLayout::RowMajor),
                },
            ],
            weights: vec![],
            nodes: vec![GraphNode {
                id: node_id("rotary"),
                inputs: vec![value_id("values"), value_id("positions")],
                outputs: vec![value_id("output")],
                weights: vec![],
                states: vec![],
                kind: GraphNodeKind::RotaryPosition {
                    heads: 2,
                    head_dim: 4,
                    rotary_dim: 4,
                    base: PositiveF32::new(10_000.0, "rotary base").unwrap(),
                    pairing,
                },
            }],
        }
    }

    fn causal_gqa_graph(
        query_shape: &[usize],
        key_shape: &[usize],
        value_shape: &[usize],
        output_shape: &[usize],
        format: NumericalFormat,
        layout: TensorLayout,
        states: Vec<StateId>,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> ModelGraph {
        ModelGraph {
            graph_id: "causal-gqa-core".into(),
            inputs: vec![value_id("query"), value_id("key"), value_id("value")],
            outputs: vec![value_id("context")],
            values: vec![
                GraphValue {
                    id: value_id("query"),
                    tensor: rotary_spec(query_shape, format.clone(), layout.clone()),
                },
                GraphValue {
                    id: value_id("key"),
                    tensor: rotary_spec(key_shape, format.clone(), layout.clone()),
                },
                GraphValue {
                    id: value_id("value"),
                    tensor: rotary_spec(value_shape, format.clone(), layout.clone()),
                },
                GraphValue {
                    id: value_id("context"),
                    tensor: rotary_spec(output_shape, format, layout),
                },
            ],
            weights: vec![],
            nodes: vec![GraphNode {
                id: node_id("causal-gqa"),
                inputs: vec![value_id("query"), value_id("key"), value_id("value")],
                outputs: vec![value_id("context")],
                weights: vec![],
                states,
                kind: GraphNodeKind::CausalGqaAttentionCore {
                    q_heads,
                    kv_heads,
                    head_dim,
                    value_dim,
                    softmax_scale: PositiveF32::new(1.0, "softmax scale").unwrap(),
                },
            }],
        }
    }

    #[test]
    fn rotary_position_accepts_both_pairing_semantics() {
        for pairing in [RotaryPairing::SplitHalf, RotaryPairing::Interleaved] {
            rotary_graph(pairing).validate().unwrap();
        }

        for format in [NumericalFormat::Bf16, NumericalFormat::Fp16] {
            let mut graph = rotary_graph(RotaryPairing::SplitHalf);
            graph.values[0].tensor =
                rotary_spec(&[2, 8], format.clone(), TensorLayout::TokensHidden);
            graph.values[2].tensor = rotary_spec(&[2, 8], format, TensorLayout::TokensHidden);
            graph.validate().unwrap();
        }

        let mut partial = rotary_graph(RotaryPairing::Interleaved);
        partial.nodes[0].kind = GraphNodeKind::RotaryPosition {
            heads: 2,
            head_dim: 4,
            rotary_dim: 2,
            base: PositiveF32::new(10_000.0, "rotary base").unwrap(),
            pairing: RotaryPairing::Interleaved,
        };
        partial.validate().unwrap();

        let mut packed = rotary_graph(RotaryPairing::SplitHalf);
        packed.values[0].tensor =
            rotary_spec(&[2, 8], NumericalFormat::F32, TensorLayout::PackedRagged);
        packed.values[2].tensor =
            rotary_spec(&[2, 8], NumericalFormat::F32, TensorLayout::PackedRagged);
        packed.validate().unwrap();

        let mut rank3 = rotary_graph(RotaryPairing::Interleaved);
        rank3.values[0].tensor =
            rotary_spec(&[2, 3, 8], NumericalFormat::F32, TensorLayout::TokensHidden);
        rank3.values[1].tensor = rotary_spec(&[2, 3], NumericalFormat::U64, TensorLayout::RowMajor);
        rank3.values[2].tensor =
            rotary_spec(&[2, 3, 8], NumericalFormat::F32, TensorLayout::TokensHidden);
        rank3.validate().unwrap();
    }

    #[test]
    fn rotary_position_rejects_bad_arity_and_feature_width() {
        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.nodes[0].inputs.pop();
        assert!(graph.validate().unwrap_err().contains("expected inputs=2"));

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.nodes[0].weights.push(weight_id("unexpected"));
        assert!(graph.validate().unwrap_err().contains("weights=1"));

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.nodes[0].states.push(state_id("unexpected"));
        assert!(graph.validate().unwrap_err().contains("states=1"));

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.values[0].tensor = rotary_spec(&[2, 7], NumericalFormat::F32, TensorLayout::RowMajor);
        graph.values[2].tensor = rotary_spec(&[2, 7], NumericalFormat::F32, TensorLayout::RowMajor);
        assert!(graph.validate().unwrap_err().contains("feature width"));
    }

    #[test]
    fn rotary_position_rejects_bad_positions_and_output_metadata() {
        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.values[1].tensor = rotary_spec(&[2, 1], NumericalFormat::U32, TensorLayout::RowMajor);
        assert!(graph.validate().unwrap_err().contains("positions shape"));

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.values[1].tensor = rotary_spec(&[2], NumericalFormat::F32, TensorLayout::RowMajor);
        assert!(graph.validate().unwrap_err().contains("U32 or U64"));

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.values[1].tensor =
            rotary_spec(&[2], NumericalFormat::U64, TensorLayout::TokensHidden);
        assert!(graph.validate().unwrap_err().contains("RowMajor"));

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.values[2].tensor =
            rotary_spec(&[2, 8], NumericalFormat::Bf16, TensorLayout::RowMajor);
        assert!(graph.validate().unwrap_err().contains("format and layout"));

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.values[2].tensor =
            rotary_spec(&[2, 8], NumericalFormat::F32, TensorLayout::TokensHidden);
        assert!(graph.validate().unwrap_err().contains("format and layout"));

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.values[2].tensor = rotary_spec(&[2, 7], NumericalFormat::F32, TensorLayout::RowMajor);
        assert!(graph.validate().unwrap_err().contains("output shape"));
    }

    #[test]
    fn rotary_position_rejects_non_real_values_and_ambiguous_layouts() {
        for format in [
            NumericalFormat::U32,
            NumericalFormat::U64,
            NumericalFormat::Aq4_0,
            NumericalFormat::Sq8_0,
            NumericalFormat::custom("future-format").unwrap(),
        ] {
            let mut graph = rotary_graph(RotaryPairing::SplitHalf);
            graph.values[0].tensor = rotary_spec(&[2, 8], format.clone(), TensorLayout::RowMajor);
            graph.values[2].tensor = rotary_spec(&[2, 8], format, TensorLayout::RowMajor);
            assert!(
                graph
                    .validate()
                    .unwrap_err()
                    .contains("must use F32, BF16, or FP16")
            );
        }

        for layout in [TensorLayout::Custom("future-layout".into())] {
            let mut graph = rotary_graph(RotaryPairing::Interleaved);
            graph.values[0].tensor = rotary_spec(&[2, 8], NumericalFormat::F32, layout.clone());
            graph.values[2].tensor = rotary_spec(&[2, 8], NumericalFormat::F32, layout);
            assert!(
                graph
                    .validate()
                    .unwrap_err()
                    .contains("must use RowMajor, TokensHidden, or PackedRagged layout")
            );
        }
    }

    #[test]
    fn rotary_position_rejects_invalid_geometry_and_overflow() {
        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.nodes[0].kind = GraphNodeKind::RotaryPosition {
            heads: 2,
            head_dim: 4,
            rotary_dim: 3,
            base: PositiveF32::new(10_000.0, "rotary base").unwrap(),
            pairing: RotaryPairing::SplitHalf,
        };
        assert!(
            graph
                .validate()
                .unwrap_err()
                .contains("rotary_dim must be even")
        );

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.nodes[0].kind = GraphNodeKind::RotaryPosition {
            heads: 2,
            head_dim: 4,
            rotary_dim: 6,
            base: PositiveF32::new(10_000.0, "rotary base").unwrap(),
            pairing: RotaryPairing::SplitHalf,
        };
        assert!(
            graph
                .validate()
                .unwrap_err()
                .contains("must not exceed rotary head_dim")
        );

        let mut graph = rotary_graph(RotaryPairing::SplitHalf);
        graph.nodes[0].kind = GraphNodeKind::RotaryPosition {
            heads: usize::MAX,
            head_dim: 2,
            rotary_dim: 2,
            base: PositiveF32::new(10_000.0, "rotary base").unwrap(),
            pairing: RotaryPairing::SplitHalf,
        };
        assert!(graph.validate().unwrap_err().contains("overflows usize"));
    }
}
