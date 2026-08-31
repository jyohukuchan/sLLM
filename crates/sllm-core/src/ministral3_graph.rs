//! Container-neutral text graph contract for the reviewed Ministral 3 3B
//! checkpoint.
//!
//! This module fixes the model topology and reuses existing model-neutral
//! semantic operators where their contracts are exact. YaRN plus the
//! position-dependent query scale, opaque KV append, and state-backed causal
//! attention remain explicitly typed graph stages. Their presence here does
//! not claim a HIP implementation, a production executor, or GPU evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{
    DType, ExecutionBoundaryKind, OpError, RmsNormScaleMode, SemanticOpDescriptor, SemanticOpKind,
    TensorError, TensorView, WeightConsumer, WeightConsumerKey, WindowedCausalAttentionContract,
};

pub const MINISTRAL3_GRAPH_LAYER_COUNT: u32 = 26;
pub const MINISTRAL3_GRAPH_HIDDEN_SIZE: usize = 3_072;
pub const MINISTRAL3_GRAPH_INTERMEDIATE_SIZE: usize = 9_216;
pub const MINISTRAL3_GRAPH_VOCAB_SIZE: usize = 131_072;
pub const MINISTRAL3_GRAPH_Q_HEADS: u32 = 32;
pub const MINISTRAL3_GRAPH_KV_HEADS: u32 = 8;
pub const MINISTRAL3_GRAPH_HEAD_DIM: u32 = 128;
pub const MINISTRAL3_GRAPH_Q_WIDTH: usize = 4_096;
pub const MINISTRAL3_GRAPH_KV_WIDTH: usize = 1_024;
pub const MINISTRAL3_GRAPH_ORIGINAL_CONTEXT: u64 = 16_384;
pub const MINISTRAL3_GRAPH_MAX_CONTEXT: u64 = 262_144;
pub const MINISTRAL3_GRAPH_ROPE_THETA: f32 = 1_000_000.0;
pub const MINISTRAL3_GRAPH_YARN_FACTOR: f32 = 16.0;
pub const MINISTRAL3_GRAPH_YARN_BETA_FAST: f64 = 32.0;
pub const MINISTRAL3_GRAPH_YARN_BETA_SLOW: f64 = 1.0;
pub const MINISTRAL3_GRAPH_LLAMA4_SCALING_BETA: f32 = 0.1;
pub const MINISTRAL3_GRAPH_RMS_EPSILON: f32 = 1.0e-5;

fn ministral3_attention_scale() -> f32 {
    (MINISTRAL3_GRAPH_HEAD_DIM as f32).sqrt().recip()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ministral3GraphError {
    InvalidRequest(&'static str),
    LengthOverflow,
    ContextExceeded { end: u64, maximum: u64 },
    StateCapacityExceeded { end: u64, capacity: u64 },
    Tensor(String),
    Semantic(String),
    InvalidTopology(&'static str),
    InvalidOrder,
    InvalidTensorWriter,
    DuplicateKvWriter { layer: u32 },
    MissingKvWriter { layer: u32 },
    InvalidBoundary,
    InvalidWeightContract,
}

impl fmt::Display for Ministral3GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(field) => {
                write!(formatter, "invalid Ministral 3 graph request: {field}")
            }
            Self::LengthOverflow => formatter.write_str("Ministral 3 context length overflows"),
            Self::ContextExceeded { end, maximum } => write!(
                formatter,
                "Ministral 3 context end {end} exceeds maximum {maximum}"
            ),
            Self::StateCapacityExceeded { end, capacity } => write!(
                formatter,
                "Ministral 3 context end {end} exceeds state capacity {capacity}"
            ),
            Self::Tensor(error) => write!(formatter, "invalid Ministral 3 tensor: {error}"),
            Self::Semantic(error) => {
                write!(formatter, "invalid Ministral 3 semantic operation: {error}")
            }
            Self::InvalidTopology(reason) => {
                write!(formatter, "invalid Ministral 3 graph topology: {reason}")
            }
            Self::InvalidOrder => {
                formatter.write_str("Ministral 3 graph dependency order is invalid")
            }
            Self::InvalidTensorWriter => {
                formatter.write_str("Ministral 3 activation does not have exactly one writer")
            }
            Self::DuplicateKvWriter { layer } => {
                write!(
                    formatter,
                    "Ministral 3 layer {layer} has duplicate KV writers"
                )
            }
            Self::MissingKvWriter { layer } => {
                write!(formatter, "Ministral 3 layer {layer} has no KV writer")
            }
            Self::InvalidBoundary => {
                formatter.write_str("Ministral 3 state/readback boundary contract is invalid")
            }
            Self::InvalidWeightContract => {
                formatter.write_str("Ministral 3 weight consumer contract is invalid")
            }
        }
    }
}

impl std::error::Error for Ministral3GraphError {}

impl From<TensorError> for Ministral3GraphError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error.to_string())
    }
}

impl From<OpError> for Ministral3GraphError {
    fn from(error: OpError) -> Self {
        Self::Semantic(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Ministral3TensorClass {
    RequestInput,
    Weight,
    Activation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3GraphTensor {
    id: usize,
    label: String,
    view: TensorView,
    class: Ministral3TensorClass,
    weight: Option<WeightConsumerKey>,
    /// The source tensor for a metadata-only, zero-copy view/reshape alias.
    ///
    /// An alias still has its own graph writer so dependency and lifetime
    /// tracking remain explicit; this field records that no payload is
    /// allocated for the alias.
    alias_of: Option<usize>,
    writer: Option<usize>,
}

impl Ministral3GraphTensor {
    pub const fn id(&self) -> usize {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn view(&self) -> &TensorView {
        &self.view
    }

    pub const fn class(&self) -> Ministral3TensorClass {
        self.class
    }

    pub const fn weight(&self) -> Option<WeightConsumerKey> {
        self.weight
    }

    /// Returns the backing tensor when this is a zero-copy view/reshape alias.
    pub const fn alias_of(&self) -> Option<usize> {
        self.alias_of
    }

    /// Whether this tensor is a metadata-only alias of another tensor.
    pub const fn is_zero_copy_alias(&self) -> bool {
        self.alias_of.is_some()
    }

    pub const fn writer(&self) -> Option<usize> {
        self.writer
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Ministral3NormRole {
    Input,
    PostAttention,
    Final,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Ministral3RotaryPairing {
    SplitHalf,
    AdjacentAfterGgufHeadPermutation,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Ministral3QueryScaleApplication {
    QueryOnlyAfterRotary,
}

/// The fixed YaRN rotation followed by the query-only Llama-4 scale.
///
/// This is deliberately not represented as [`SemanticOpKind::Rotary`]: the
/// existing rotary contract contains only a plain theta-based frequency law.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Ministral3YarnQueryScaleStage {
    start_position: u64,
    token_count: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    pairing: Ministral3RotaryPairing,
    theta_bits: u32,
    factor_bits: u32,
    beta_fast_bits: u64,
    beta_slow_bits: u64,
    original_context: u32,
    maximum_context: u32,
    query_scale_beta_bits: u32,
    query_scale_application: Ministral3QueryScaleApplication,
}

impl Ministral3YarnQueryScaleStage {
    pub fn new(start_position: u64, token_count: u64) -> Result<Self, Ministral3GraphError> {
        if token_count == 0 {
            return Err(Ministral3GraphError::InvalidRequest("token count is zero"));
        }
        let end = start_position
            .checked_add(token_count)
            .ok_or(Ministral3GraphError::LengthOverflow)?;
        if end > MINISTRAL3_GRAPH_MAX_CONTEXT {
            return Err(Ministral3GraphError::ContextExceeded {
                end,
                maximum: MINISTRAL3_GRAPH_MAX_CONTEXT,
            });
        }
        Ok(Self {
            start_position,
            token_count: u32::try_from(token_count)
                .map_err(|_| Ministral3GraphError::InvalidRequest("token count exceeds u32"))?,
            q_heads: MINISTRAL3_GRAPH_Q_HEADS,
            kv_heads: MINISTRAL3_GRAPH_KV_HEADS,
            head_dim: MINISTRAL3_GRAPH_HEAD_DIM,
            rotary_dim: MINISTRAL3_GRAPH_HEAD_DIM,
            pairing: Ministral3RotaryPairing::AdjacentAfterGgufHeadPermutation,
            theta_bits: MINISTRAL3_GRAPH_ROPE_THETA.to_bits(),
            factor_bits: MINISTRAL3_GRAPH_YARN_FACTOR.to_bits(),
            beta_fast_bits: MINISTRAL3_GRAPH_YARN_BETA_FAST.to_bits(),
            beta_slow_bits: MINISTRAL3_GRAPH_YARN_BETA_SLOW.to_bits(),
            original_context: MINISTRAL3_GRAPH_ORIGINAL_CONTEXT as u32,
            maximum_context: MINISTRAL3_GRAPH_MAX_CONTEXT as u32,
            query_scale_beta_bits: MINISTRAL3_GRAPH_LLAMA4_SCALING_BETA.to_bits(),
            query_scale_application: Ministral3QueryScaleApplication::QueryOnlyAfterRotary,
        })
    }

    pub const fn start_position(self) -> u64 {
        self.start_position
    }

    pub const fn token_count(self) -> u32 {
        self.token_count
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

    pub const fn pairing(self) -> Ministral3RotaryPairing {
        self.pairing
    }

    pub const fn theta(self) -> f32 {
        f32::from_bits(self.theta_bits)
    }

    pub const fn factor(self) -> f32 {
        f32::from_bits(self.factor_bits)
    }

    pub const fn beta_fast(self) -> f64 {
        f64::from_bits(self.beta_fast_bits)
    }

    pub const fn beta_slow(self) -> f64 {
        f64::from_bits(self.beta_slow_bits)
    }

    pub const fn original_context(self) -> u32 {
        self.original_context
    }

    pub const fn maximum_context(self) -> u32 {
        self.maximum_context
    }

    pub const fn query_scale_beta(self) -> f32 {
        f32::from_bits(self.query_scale_beta_bits)
    }

    pub const fn query_scale_application(self) -> Ministral3QueryScaleApplication {
        self.query_scale_application
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Ministral3KvAppendContract {
    layer: u32,
    heads: u32,
    head_dim: u32,
    previous_length: u64,
    append_count: u32,
    published_length: u64,
    capacity: u64,
}

impl Ministral3KvAppendContract {
    fn new(
        layer: u32,
        previous_length: u64,
        append_count: u64,
        published_length: u64,
        capacity: u64,
    ) -> Result<Self, Ministral3GraphError> {
        if layer >= MINISTRAL3_GRAPH_LAYER_COUNT
            || previous_length
                .checked_add(append_count)
                .ok_or(Ministral3GraphError::LengthOverflow)?
                != published_length
            || published_length > capacity
        {
            return Err(Ministral3GraphError::InvalidRequest("KV append geometry"));
        }
        Ok(Self {
            layer,
            heads: MINISTRAL3_GRAPH_KV_HEADS,
            head_dim: MINISTRAL3_GRAPH_HEAD_DIM,
            previous_length,
            append_count: u32::try_from(append_count)
                .map_err(|_| Ministral3GraphError::InvalidRequest("KV append count exceeds u32"))?,
            published_length,
            capacity,
        })
    }

    pub const fn layer(self) -> u32 {
        self.layer
    }

    pub const fn heads(self) -> u32 {
        self.heads
    }

    pub const fn head_dim(self) -> u32 {
        self.head_dim
    }

    pub const fn previous_length(self) -> u64 {
        self.previous_length
    }

    pub const fn append_count(self) -> u32 {
        self.append_count
    }

    pub const fn published_length(self) -> u64 {
        self.published_length
    }

    pub const fn capacity(self) -> u64 {
        self.capacity
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ministral3GraphNodeKind {
    Embedding {
        weight: WeightConsumerKey,
    },
    RmsNorm {
        role: Ministral3NormRole,
        weight: WeightConsumerKey,
        epsilon_bits: u32,
        scale_mode: RmsNormScaleMode,
    },
    Matmul {
        weight: WeightConsumerKey,
    },
    /// Metadata-only contiguous reinterpretation. This node never allocates
    /// or copies payload bytes; its output tensor aliases its sole input.
    View,
    /// Metadata-only contiguous reshape. This node never allocates or copies
    /// payload bytes; its output tensor aliases its sole input.
    Reshape,
    YarnRopeQueryScale(Ministral3YarnQueryScaleStage),
    KvAppend(Ministral3KvAppendContract),
    CausalGqa(WindowedCausalAttentionContract),
    SiluMul,
    Add,
    Argmax,
}

impl Ministral3GraphNodeKind {
    pub const fn reused_semantic_kind(&self) -> Option<SemanticOpKind> {
        match self {
            Self::Embedding { .. } => Some(SemanticOpKind::Embedding),
            Self::RmsNorm { .. } => Some(SemanticOpKind::RmsNorm),
            Self::Matmul { .. } => Some(SemanticOpKind::Matmul),
            Self::SiluMul => Some(SemanticOpKind::SiluMul),
            Self::Add => Some(SemanticOpKind::Add),
            Self::Argmax => Some(SemanticOpKind::Argmax),
            Self::View
            | Self::Reshape
            | Self::YarnRopeQueryScale(_)
            | Self::KvAppend(_)
            | Self::CausalGqa(_) => None,
        }
    }

    pub const fn weight(&self) -> Option<WeightConsumerKey> {
        match self {
            Self::Embedding { weight } | Self::RmsNorm { weight, .. } | Self::Matmul { weight } => {
                Some(*weight)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3GraphNode {
    id: usize,
    label: String,
    layer: Option<u32>,
    kind: Ministral3GraphNodeKind,
    operation: Option<SemanticOpDescriptor>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    dependencies: Vec<usize>,
    boundary_after: Option<ExecutionBoundaryKind>,
}

impl Ministral3GraphNode {
    pub const fn id(&self) -> usize {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn layer(&self) -> Option<u32> {
        self.layer
    }

    pub const fn kind(&self) -> &Ministral3GraphNodeKind {
        &self.kind
    }

    pub fn operation(&self) -> Option<&SemanticOpDescriptor> {
        self.operation.as_ref()
    }

    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }

    pub fn dependencies(&self) -> &[usize] {
        &self.dependencies
    }

    pub const fn boundary_after(&self) -> Option<ExecutionBoundaryKind> {
        self.boundary_after
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3TextGraph {
    token_count: u64,
    start_position: u64,
    expected_length: u64,
    state_capacity: u64,
    tensors: Vec<Ministral3GraphTensor>,
    nodes: Vec<Ministral3GraphNode>,
    kv_contracts: Vec<Ministral3KvAppendContract>,
}

impl Ministral3TextGraph {
    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    pub const fn start_position(&self) -> u64 {
        self.start_position
    }

    pub const fn expected_length(&self) -> u64 {
        self.expected_length
    }

    pub const fn state_capacity(&self) -> u64 {
        self.state_capacity
    }

    pub fn tensors(&self) -> &[Ministral3GraphTensor] {
        &self.tensors
    }

    pub fn nodes(&self) -> &[Ministral3GraphNode] {
        &self.nodes
    }

    pub fn kv_contracts(&self) -> &[Ministral3KvAppendContract] {
        &self.kv_contracts
    }

    /// The graph is independent of safetensors, GGUF, and resident layout.
    pub const fn is_container_neutral(&self) -> bool {
        true
    }

    /// Whether this reviewed graph contract has a bound production executor.
    /// This remains separate from per-target full-model GPU evidence.
    pub const fn has_production_executor(&self) -> bool {
        true
    }
}

/// Build the fixed text-only Ministral 3 graph for one prefill or decode
/// transition. `state_capacity` is a request-local allocation bound, not an
/// advertisement that the full context fits on a particular GPU.
pub fn build_ministral3_text_graph(
    token_count: u64,
    start_position: u64,
    state_capacity: u64,
) -> Result<Ministral3TextGraph, Ministral3GraphError> {
    if token_count == 0 {
        return Err(Ministral3GraphError::InvalidRequest("token count is zero"));
    }
    if state_capacity == 0 || state_capacity > MINISTRAL3_GRAPH_MAX_CONTEXT {
        return Err(Ministral3GraphError::InvalidRequest(
            "state capacity is outside reviewed context",
        ));
    }
    let expected_length = start_position
        .checked_add(token_count)
        .ok_or(Ministral3GraphError::LengthOverflow)?;
    if expected_length > MINISTRAL3_GRAPH_MAX_CONTEXT {
        return Err(Ministral3GraphError::ContextExceeded {
            end: expected_length,
            maximum: MINISTRAL3_GRAPH_MAX_CONTEXT,
        });
    }
    if expected_length > state_capacity {
        return Err(Ministral3GraphError::StateCapacityExceeded {
            end: expected_length,
            capacity: state_capacity,
        });
    }
    let rows = usize::try_from(token_count)
        .map_err(|_| Ministral3GraphError::InvalidRequest("token count exceeds usize"))?;
    let mut builder = GraphBuilder::new(
        rows,
        token_count,
        start_position,
        expected_length,
        state_capacity,
    )?;
    builder.build()?;
    builder.finish()
}

struct GraphBuilder {
    rows: usize,
    token_count: u64,
    start_position: u64,
    expected_length: u64,
    state_capacity: u64,
    tensors: Vec<Ministral3GraphTensor>,
    nodes: Vec<Ministral3GraphNode>,
    weights: BTreeMap<WeightConsumerKey, usize>,
    token_ids: usize,
    positions: usize,
    kv_contracts: Vec<Ministral3KvAppendContract>,
}

impl GraphBuilder {
    fn new(
        rows: usize,
        token_count: u64,
        start_position: u64,
        expected_length: u64,
        state_capacity: u64,
    ) -> Result<Self, Ministral3GraphError> {
        let mut builder = Self {
            rows,
            token_count,
            start_position,
            expected_length,
            state_capacity,
            tensors: Vec::new(),
            nodes: Vec::new(),
            weights: BTreeMap::new(),
            token_ids: usize::MAX,
            positions: usize::MAX,
            kv_contracts: Vec::with_capacity(MINISTRAL3_GRAPH_LAYER_COUNT as usize),
        };
        builder.token_ids = builder.add_source(
            "input.token_ids",
            TensorView::contiguous(DType::I32, &[rows])?,
        );
        builder.positions = builder.add_source(
            "input.positions",
            TensorView::contiguous(DType::I32, &[rows])?,
        );
        Ok(builder)
    }

    fn build(&mut self) -> Result<(), Ministral3GraphError> {
        let tied_key = WeightConsumerKey {
            layer: None,
            role: WeightConsumer::EmbeddingAndTiedOutput,
        };
        let tied_weight = self.weight(
            tied_key,
            &[MINISTRAL3_GRAPH_VOCAB_SIZE, MINISTRAL3_GRAPH_HIDDEN_SIZE],
        )?;
        let embedding = self.activation(
            "embedding.output",
            &[self.rows, MINISTRAL3_GRAPH_HIDDEN_SIZE],
        )?;
        let embedding_op = SemanticOpDescriptor::new(
            SemanticOpKind::Embedding,
            vec![self.view(tied_weight), self.view(self.token_ids)],
            vec![self.view(embedding)],
        )?;
        self.push_node(
            "embedding",
            None,
            Ministral3GraphNodeKind::Embedding { weight: tied_key },
            Some(embedding_op),
            vec![tied_weight, self.token_ids],
            vec![embedding],
            &[],
            None,
        )?;
        let mut hidden = embedding;

        for layer in 0..MINISTRAL3_GRAPH_LAYER_COUNT {
            let layer_u64 = u64::from(layer);
            let input_norm = self.rms_norm(
                layer,
                "input_norm",
                Ministral3NormRole::Input,
                WeightConsumer::InputNorm,
                hidden,
            )?;
            let q = self.matmul(
                layer,
                "q_proj",
                WeightConsumer::AttentionQ,
                input_norm,
                MINISTRAL3_GRAPH_Q_WIDTH,
            )?;
            let k = self.matmul(
                layer,
                "k_proj",
                WeightConsumer::AttentionK,
                input_norm,
                MINISTRAL3_GRAPH_KV_WIDTH,
            )?;
            let v = self.matmul(
                layer,
                "v_proj",
                WeightConsumer::AttentionV,
                input_norm,
                MINISTRAL3_GRAPH_KV_WIDTH,
            )?;
            let q_heads = self.reshape_alias(
                layer,
                "q_proj.reshape",
                q,
                &[
                    self.rows,
                    MINISTRAL3_GRAPH_Q_HEADS as usize,
                    MINISTRAL3_GRAPH_HEAD_DIM as usize,
                ],
            )?;
            let k_heads = self.reshape_alias(
                layer,
                "k_proj.reshape",
                k,
                &[
                    self.rows,
                    MINISTRAL3_GRAPH_KV_HEADS as usize,
                    MINISTRAL3_GRAPH_HEAD_DIM as usize,
                ],
            )?;
            let v_heads = self.reshape_alias(
                layer,
                "v_proj.reshape",
                v,
                &[
                    self.rows,
                    MINISTRAL3_GRAPH_KV_HEADS as usize,
                    MINISTRAL3_GRAPH_HEAD_DIM as usize,
                ],
            )?;
            let q_rotary = self.activation(
                &format!("layer.{layer}.yarn.q.output"),
                &[
                    self.rows,
                    MINISTRAL3_GRAPH_Q_HEADS as usize,
                    MINISTRAL3_GRAPH_HEAD_DIM as usize,
                ],
            )?;
            let k_rotary = self.activation(
                &format!("layer.{layer}.yarn.k.output"),
                &[
                    self.rows,
                    MINISTRAL3_GRAPH_KV_HEADS as usize,
                    MINISTRAL3_GRAPH_HEAD_DIM as usize,
                ],
            )?;
            let yarn = Ministral3YarnQueryScaleStage::new(self.start_position, self.token_count)?;
            self.push_node(
                format!("layer.{layer}.yarn_rope_query_scale"),
                Some(layer),
                Ministral3GraphNodeKind::YarnRopeQueryScale(yarn),
                None,
                vec![q_heads, k_heads, self.positions],
                vec![q_rotary, k_rotary],
                &[],
                None,
            )?;

            let kv = Ministral3KvAppendContract::new(
                layer,
                self.start_position,
                self.token_count,
                self.expected_length,
                self.state_capacity,
            )?;
            let kv_node = self.push_node(
                format!("layer.{layer}.kv_append"),
                Some(layer),
                Ministral3GraphNodeKind::KvAppend(kv),
                None,
                vec![k_rotary, v_heads],
                vec![],
                &[],
                None,
            )?;
            self.kv_contracts.push(kv);

            let attention_output = self.activation(
                &format!("layer.{layer}.attention.output"),
                &[
                    self.rows,
                    MINISTRAL3_GRAPH_Q_HEADS as usize,
                    MINISTRAL3_GRAPH_HEAD_DIM as usize,
                ],
            )?;
            let attention_contract = WindowedCausalAttentionContract::new(
                MINISTRAL3_GRAPH_Q_HEADS,
                MINISTRAL3_GRAPH_KV_HEADS,
                MINISTRAL3_GRAPH_HEAD_DIM,
                self.start_position,
                self.token_count,
                self.expected_length,
                None,
                ministral3_attention_scale(),
            )?;
            self.push_node(
                format!("layer.{layer}.causal_gqa"),
                Some(layer),
                Ministral3GraphNodeKind::CausalGqa(attention_contract),
                None,
                vec![q_rotary],
                vec![attention_output],
                &[kv_node],
                None,
            )?;
            let attention_flat = self.view_alias(
                layer,
                "attention.output.view",
                attention_output,
                &[self.rows, MINISTRAL3_GRAPH_Q_WIDTH],
            )?;
            let attention_projected = self.matmul(
                layer,
                "o_proj",
                WeightConsumer::AttentionO,
                attention_flat,
                MINISTRAL3_GRAPH_HIDDEN_SIZE,
            )?;
            let attention_residual = self.binary(
                layer,
                "attention_residual",
                SemanticOpKind::Add,
                Ministral3GraphNodeKind::Add,
                hidden,
                attention_projected,
                MINISTRAL3_GRAPH_HIDDEN_SIZE,
                None,
            )?;
            let post_norm = self.rms_norm(
                layer,
                "post_attention_norm",
                Ministral3NormRole::PostAttention,
                WeightConsumer::PostAttentionNorm,
                attention_residual,
            )?;
            let gate = self.matmul(
                layer,
                "mlp_gate",
                WeightConsumer::MlpGate,
                post_norm,
                MINISTRAL3_GRAPH_INTERMEDIATE_SIZE,
            )?;
            let up = self.matmul(
                layer,
                "mlp_up",
                WeightConsumer::MlpUp,
                post_norm,
                MINISTRAL3_GRAPH_INTERMEDIATE_SIZE,
            )?;
            let activated = self.binary(
                layer,
                "mlp_silu_mul",
                SemanticOpKind::SiluMul,
                Ministral3GraphNodeKind::SiluMul,
                gate,
                up,
                MINISTRAL3_GRAPH_INTERMEDIATE_SIZE,
                None,
            )?;
            let down = self.matmul(
                layer,
                "mlp_down",
                WeightConsumer::MlpDown,
                activated,
                MINISTRAL3_GRAPH_HIDDEN_SIZE,
            )?;
            hidden = self.binary(
                layer,
                "mlp_residual",
                SemanticOpKind::Add,
                Ministral3GraphNodeKind::Add,
                attention_residual,
                down,
                MINISTRAL3_GRAPH_HIDDEN_SIZE,
                (layer_u64 + 1 == u64::from(MINISTRAL3_GRAPH_LAYER_COUNT))
                    .then_some(ExecutionBoundaryKind::StatePublication),
            )?;
        }

        let final_norm = self.rms_norm_root(
            "final_norm",
            Ministral3NormRole::Final,
            WeightConsumer::FinalNorm,
            hidden,
        )?;
        let terminal_norm = self.terminal_row_view("final_norm.terminal", final_norm)?;
        let logits = self.activation("logits", &[1, MINISTRAL3_GRAPH_VOCAB_SIZE])?;
        let logits_op = SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![self.view(terminal_norm), self.view(tied_weight)],
            vec![self.view(logits)],
        )?;
        self.push_node(
            "tied_logits",
            None,
            Ministral3GraphNodeKind::Matmul { weight: tied_key },
            Some(logits_op),
            vec![terminal_norm, tied_weight],
            vec![logits],
            &[],
            None,
        )?;
        let selected = self.activation_typed("argmax.output", DType::I32, &[1])?;
        let argmax_op = SemanticOpDescriptor::new(
            SemanticOpKind::Argmax,
            vec![self.view(logits)],
            vec![self.view(selected)],
        )?;
        self.push_node(
            "argmax",
            None,
            Ministral3GraphNodeKind::Argmax,
            Some(argmax_op),
            vec![logits],
            vec![selected],
            &[],
            Some(ExecutionBoundaryKind::TerminalReadback),
        )?;
        Ok(())
    }

    fn rms_norm(
        &mut self,
        layer: u32,
        label: &str,
        role: Ministral3NormRole,
        consumer: WeightConsumer,
        input: usize,
    ) -> Result<usize, Ministral3GraphError> {
        let key = WeightConsumerKey {
            layer: Some(u64::from(layer)),
            role: consumer,
        };
        self.rms_norm_key(Some(layer), label, role, key, input)
    }

    fn rms_norm_root(
        &mut self,
        label: &str,
        role: Ministral3NormRole,
        consumer: WeightConsumer,
        input: usize,
    ) -> Result<usize, Ministral3GraphError> {
        let key = WeightConsumerKey {
            layer: None,
            role: consumer,
        };
        self.rms_norm_key(None, label, role, key, input)
    }

    fn rms_norm_key(
        &mut self,
        layer: Option<u32>,
        label: &str,
        role: Ministral3NormRole,
        key: WeightConsumerKey,
        input: usize,
    ) -> Result<usize, Ministral3GraphError> {
        let weight = self.weight(key, &[MINISTRAL3_GRAPH_HIDDEN_SIZE])?;
        let output_label = layer.map_or_else(
            || format!("{label}.output"),
            |layer| format!("layer.{layer}.{label}.output"),
        );
        let output = self.activation(&output_label, &[self.rows, MINISTRAL3_GRAPH_HIDDEN_SIZE])?;
        let operation = SemanticOpDescriptor::new_rms_norm(
            vec![self.view(input), self.view(weight)],
            vec![self.view(output)],
            MINISTRAL3_GRAPH_RMS_EPSILON,
            RmsNormScaleMode::Direct,
        )?;
        let node_label = layer.map_or_else(
            || label.to_owned(),
            |layer| format!("layer.{layer}.{label}"),
        );
        self.push_node(
            node_label,
            layer,
            Ministral3GraphNodeKind::RmsNorm {
                role,
                weight: key,
                epsilon_bits: MINISTRAL3_GRAPH_RMS_EPSILON.to_bits(),
                scale_mode: RmsNormScaleMode::Direct,
            },
            Some(operation),
            vec![input, weight],
            vec![output],
            &[],
            None,
        )?;
        Ok(output)
    }

    fn matmul(
        &mut self,
        layer: u32,
        label: &str,
        consumer: WeightConsumer,
        input: usize,
        output_width: usize,
    ) -> Result<usize, Ministral3GraphError> {
        let input_width = *self
            .tensors
            .get(input)
            .and_then(|tensor| tensor.view.shape().get(1))
            .ok_or(Ministral3GraphError::InvalidTopology(
                "matmul input is not rank two",
            ))?;
        let key = WeightConsumerKey {
            layer: Some(u64::from(layer)),
            role: consumer,
        };
        let weight = self.weight(key, &[output_width, input_width])?;
        let output = self.activation(
            &format!("layer.{layer}.{label}.output"),
            &[self.rows, output_width],
        )?;
        let operation = SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![self.view(input), self.view(weight)],
            vec![self.view(output)],
        )?;
        self.push_node(
            format!("layer.{layer}.{label}"),
            Some(layer),
            Ministral3GraphNodeKind::Matmul { weight: key },
            Some(operation),
            vec![input, weight],
            vec![output],
            &[],
            None,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn binary(
        &mut self,
        layer: u32,
        label: &str,
        semantic_kind: SemanticOpKind,
        node_kind: Ministral3GraphNodeKind,
        left: usize,
        right: usize,
        width: usize,
        boundary_after: Option<ExecutionBoundaryKind>,
    ) -> Result<usize, Ministral3GraphError> {
        let output = self.activation(
            &format!("layer.{layer}.{label}.output"),
            &[self.rows, width],
        )?;
        let operation = SemanticOpDescriptor::new(
            semantic_kind,
            vec![self.view(left), self.view(right)],
            vec![self.view(output)],
        )?;
        self.push_node(
            format!("layer.{layer}.{label}"),
            Some(layer),
            node_kind,
            Some(operation),
            vec![left, right],
            vec![output],
            &[],
            boundary_after,
        )?;
        Ok(output)
    }

    fn weight(
        &mut self,
        key: WeightConsumerKey,
        shape: &[usize],
    ) -> Result<usize, Ministral3GraphError> {
        if let Some(&tensor) = self.weights.get(&key) {
            if self.tensors[tensor].view.shape() != shape {
                return Err(Ministral3GraphError::InvalidWeightContract);
            }
            return Ok(tensor);
        }
        let tensor = self.add_tensor(
            weight_label(key),
            TensorView::contiguous(DType::Bf16, shape)?,
            Ministral3TensorClass::Weight,
            Some(key),
        );
        self.weights.insert(key, tensor);
        Ok(tensor)
    }

    fn activation(&mut self, label: &str, shape: &[usize]) -> Result<usize, Ministral3GraphError> {
        self.activation_typed(label, DType::Bf16, shape)
    }

    fn activation_typed(
        &mut self,
        label: &str,
        dtype: DType,
        shape: &[usize],
    ) -> Result<usize, Ministral3GraphError> {
        Ok(self.add_tensor(
            label.to_owned(),
            TensorView::contiguous(dtype, shape)?,
            Ministral3TensorClass::Activation,
            None,
        ))
    }

    fn add_source(&mut self, label: &str, view: TensorView) -> usize {
        self.add_tensor(
            label.to_owned(),
            view,
            Ministral3TensorClass::RequestInput,
            None,
        )
    }

    fn add_tensor(
        &mut self,
        label: String,
        view: TensorView,
        class: Ministral3TensorClass,
        weight: Option<WeightConsumerKey>,
    ) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Ministral3GraphTensor {
            id,
            label,
            view,
            class,
            weight,
            alias_of: None,
            writer: None,
        });
        id
    }

    /// Add a metadata-only contiguous alias and its explicit structural node.
    ///
    /// Matmuls intentionally retain rank-two outputs because that is the
    /// semantic matmul contract. Attention stages consume the aliases below,
    /// while the executor can bind them to the same payload as their source.
    fn zero_copy_alias(
        &mut self,
        layer: u32,
        label: &str,
        kind: Ministral3GraphNodeKind,
        source: usize,
        shape: &[usize],
    ) -> Result<usize, Ministral3GraphError> {
        let source_view = self
            .tensors
            .get(source)
            .ok_or(Ministral3GraphError::InvalidTopology(
                "alias source tensor is absent",
            ))?;
        if source_view.class != Ministral3TensorClass::Activation
            || !source_view.view.is_contiguous()
        {
            return Err(Ministral3GraphError::InvalidTopology(
                "alias source is not a contiguous activation",
            ));
        }
        let dtype = source_view.view.dtype();
        let encoding = source_view.view.encoding();
        let source_elements = source_view.view.element_count();
        let source_bytes = source_view.view.payload_bytes();
        let source_offset = source_view.view.byte_offset();
        let alias_view = TensorView::with_encoding(dtype, encoding, shape)?;
        if !alias_view.is_contiguous()
            || alias_view.dtype() != dtype
            || alias_view.encoding() != encoding
            || alias_view.element_count() != source_elements
            || alias_view.payload_bytes() != source_bytes
            || alias_view.byte_offset() != source_offset
        {
            return Err(Ministral3GraphError::InvalidTopology(
                "zero-copy alias requires equal contiguous dtype and bytes",
            ));
        }
        let alias = self.tensors.len();
        self.tensors.push(Ministral3GraphTensor {
            id: alias,
            label: format!("layer.{layer}.{label}"),
            view: alias_view,
            class: Ministral3TensorClass::Activation,
            weight: None,
            alias_of: Some(source),
            writer: None,
        });
        self.push_node(
            format!("layer.{layer}.{label}"),
            Some(layer),
            kind,
            None,
            vec![source],
            vec![alias],
            &[],
            None,
        )?;
        Ok(alias)
    }

    fn reshape_alias(
        &mut self,
        layer: u32,
        label: &str,
        source: usize,
        shape: &[usize],
    ) -> Result<usize, Ministral3GraphError> {
        self.zero_copy_alias(
            layer,
            label,
            Ministral3GraphNodeKind::Reshape,
            source,
            shape,
        )
    }

    fn view_alias(
        &mut self,
        layer: u32,
        label: &str,
        source: usize,
        shape: &[usize],
    ) -> Result<usize, Ministral3GraphError> {
        self.zero_copy_alias(layer, label, Ministral3GraphNodeKind::View, source, shape)
    }

    /// Selects only the final token row for the vocabulary projection. This is
    /// a metadata-only subview into the final-normalized activation, so prefill
    /// does not materialize `[tokens, vocabulary]` logits that are never read.
    fn terminal_row_view(
        &mut self,
        label: &str,
        source: usize,
    ) -> Result<usize, Ministral3GraphError> {
        let source_tensor =
            self.tensors
                .get(source)
                .ok_or(Ministral3GraphError::InvalidTopology(
                    "terminal view source tensor is absent",
                ))?;
        let source_view = &source_tensor.view;
        if source_tensor.class != Ministral3TensorClass::Activation
            || !source_view.is_contiguous()
            || source_view.shape() != [self.rows, MINISTRAL3_GRAPH_HIDDEN_SIZE]
        {
            return Err(Ministral3GraphError::InvalidTopology(
                "terminal view source is not the contiguous final activation",
            ));
        }
        let rows = u64::try_from(self.rows).map_err(|_| {
            Ministral3GraphError::InvalidTopology("terminal view row count overflowed")
        })?;
        let row_bytes = source_view.payload_bytes().checked_div(rows).ok_or(
            Ministral3GraphError::InvalidTopology("terminal view source has no rows"),
        )?;
        let byte_offset = source_view
            .byte_offset()
            .checked_add((rows - 1).checked_mul(row_bytes).ok_or(
                Ministral3GraphError::InvalidTopology("terminal view offset overflowed"),
            )?)
            .ok_or(Ministral3GraphError::InvalidTopology(
                "terminal view offset overflowed",
            ))?;
        let terminal_view = TensorView::new(
            source_view.dtype(),
            source_view.encoding(),
            &[1, MINISTRAL3_GRAPH_HIDDEN_SIZE],
            &[MINISTRAL3_GRAPH_HIDDEN_SIZE, 1],
            byte_offset,
        )?;
        if terminal_view.span_bytes() > source_view.span_bytes() {
            return Err(Ministral3GraphError::InvalidTopology(
                "terminal view exceeds its source activation",
            ));
        }
        let alias = self.tensors.len();
        self.tensors.push(Ministral3GraphTensor {
            id: alias,
            label: label.to_owned(),
            view: terminal_view,
            class: Ministral3TensorClass::Activation,
            weight: None,
            alias_of: Some(source),
            writer: None,
        });
        self.push_node(
            label,
            None,
            Ministral3GraphNodeKind::View,
            None,
            vec![source],
            vec![alias],
            &[],
            None,
        )?;
        Ok(alias)
    }

    fn view(&self, tensor: usize) -> TensorView {
        self.tensors[tensor].view.clone()
    }

    #[allow(clippy::too_many_arguments)]
    fn push_node(
        &mut self,
        label: impl Into<String>,
        layer: Option<u32>,
        kind: Ministral3GraphNodeKind,
        operation: Option<SemanticOpDescriptor>,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        extra_dependencies: &[usize],
        boundary_after: Option<ExecutionBoundaryKind>,
    ) -> Result<usize, Ministral3GraphError> {
        let id = self.nodes.len();
        let mut dependencies = BTreeSet::new();
        for &input in &inputs {
            let tensor = self
                .tensors
                .get(input)
                .ok_or(Ministral3GraphError::InvalidTopology(
                    "node input is absent",
                ))?;
            if let Some(writer) = tensor.writer {
                dependencies.insert(writer);
            }
        }
        for &dependency in extra_dependencies {
            if dependency >= id {
                return Err(Ministral3GraphError::InvalidOrder);
            }
            dependencies.insert(dependency);
        }
        for &output in &outputs {
            let tensor =
                self.tensors
                    .get_mut(output)
                    .ok_or(Ministral3GraphError::InvalidTopology(
                        "node output is absent",
                    ))?;
            if tensor.class != Ministral3TensorClass::Activation || tensor.writer.is_some() {
                return Err(Ministral3GraphError::InvalidTensorWriter);
            }
            tensor.writer = Some(id);
        }
        self.nodes.push(Ministral3GraphNode {
            id,
            label: label.into(),
            layer,
            kind,
            operation,
            inputs,
            outputs,
            dependencies: dependencies.into_iter().collect(),
            boundary_after,
        });
        Ok(id)
    }

    fn finish(self) -> Result<Ministral3TextGraph, Ministral3GraphError> {
        let graph = Ministral3TextGraph {
            token_count: self.token_count,
            start_position: self.start_position,
            expected_length: self.expected_length,
            state_capacity: self.state_capacity,
            tensors: self.tensors,
            nodes: self.nodes,
            kv_contracts: self.kv_contracts,
        };
        validate_graph(&graph)?;
        Ok(graph)
    }
}

fn weight_label(key: WeightConsumerKey) -> String {
    let scope = key
        .layer
        .map_or_else(|| "root".to_owned(), |layer| format!("layer.{layer}"));
    format!("weight.{scope}.{:?}", key.role)
}

fn expected_weight_shapes() -> BTreeMap<WeightConsumerKey, Vec<usize>> {
    let mut expected = BTreeMap::from([
        (
            WeightConsumerKey {
                layer: None,
                role: WeightConsumer::EmbeddingAndTiedOutput,
            },
            vec![MINISTRAL3_GRAPH_VOCAB_SIZE, MINISTRAL3_GRAPH_HIDDEN_SIZE],
        ),
        (
            WeightConsumerKey {
                layer: None,
                role: WeightConsumer::FinalNorm,
            },
            vec![MINISTRAL3_GRAPH_HIDDEN_SIZE],
        ),
    ]);
    for layer in 0..MINISTRAL3_GRAPH_LAYER_COUNT {
        let layer = Some(u64::from(layer));
        for (role, shape) in [
            (
                WeightConsumer::InputNorm,
                vec![MINISTRAL3_GRAPH_HIDDEN_SIZE],
            ),
            (
                WeightConsumer::PostAttentionNorm,
                vec![MINISTRAL3_GRAPH_HIDDEN_SIZE],
            ),
            (
                WeightConsumer::AttentionQ,
                vec![MINISTRAL3_GRAPH_Q_WIDTH, MINISTRAL3_GRAPH_HIDDEN_SIZE],
            ),
            (
                WeightConsumer::AttentionK,
                vec![MINISTRAL3_GRAPH_KV_WIDTH, MINISTRAL3_GRAPH_HIDDEN_SIZE],
            ),
            (
                WeightConsumer::AttentionV,
                vec![MINISTRAL3_GRAPH_KV_WIDTH, MINISTRAL3_GRAPH_HIDDEN_SIZE],
            ),
            (
                WeightConsumer::AttentionO,
                vec![MINISTRAL3_GRAPH_HIDDEN_SIZE, MINISTRAL3_GRAPH_Q_WIDTH],
            ),
            (
                WeightConsumer::MlpGate,
                vec![
                    MINISTRAL3_GRAPH_INTERMEDIATE_SIZE,
                    MINISTRAL3_GRAPH_HIDDEN_SIZE,
                ],
            ),
            (
                WeightConsumer::MlpUp,
                vec![
                    MINISTRAL3_GRAPH_INTERMEDIATE_SIZE,
                    MINISTRAL3_GRAPH_HIDDEN_SIZE,
                ],
            ),
            (
                WeightConsumer::MlpDown,
                vec![
                    MINISTRAL3_GRAPH_HIDDEN_SIZE,
                    MINISTRAL3_GRAPH_INTERMEDIATE_SIZE,
                ],
            ),
        ] {
            expected.insert(WeightConsumerKey { layer, role }, shape);
        }
    }
    expected
}

fn validate_graph(graph: &Ministral3TextGraph) -> Result<(), Ministral3GraphError> {
    if graph.nodes.len() != 499 {
        return Err(Ministral3GraphError::InvalidTopology("node count"));
    }
    let node_labels = graph
        .nodes
        .iter()
        .map(|node| node.label.as_str())
        .collect::<BTreeSet<_>>();
    let tensor_labels = graph
        .tensors
        .iter()
        .map(|tensor| tensor.label.as_str())
        .collect::<BTreeSet<_>>();
    if node_labels.len() != graph.nodes.len() || tensor_labels.len() != graph.tensors.len() {
        return Err(Ministral3GraphError::InvalidTopology("duplicate label"));
    }

    for (id, tensor) in graph.tensors.iter().enumerate() {
        if tensor.id != id
            || (tensor.class == Ministral3TensorClass::Activation) != tensor.writer.is_some()
            || (tensor.class == Ministral3TensorClass::Weight) != tensor.weight.is_some()
            || (tensor.class != Ministral3TensorClass::Activation && tensor.alias_of.is_some())
        {
            return Err(Ministral3GraphError::InvalidTensorWriter);
        }
        if tensor
            .writer
            .is_some_and(|writer| writer >= graph.nodes.len())
        {
            return Err(Ministral3GraphError::InvalidTensorWriter);
        }
        if let Some(source) = tensor.alias_of {
            let source_tensor =
                graph
                    .tensors
                    .get(source)
                    .ok_or(Ministral3GraphError::InvalidTopology(
                        "alias source tensor is absent",
                    ))?;
            let alias_kind = tensor
                .writer
                .and_then(|writer| graph.nodes.get(writer))
                .map(|node| &node.kind);
            let exact_payload = source_tensor.view.element_count() == tensor.view.element_count()
                && source_tensor.view.payload_bytes() == tensor.view.payload_bytes()
                && source_tensor.view.byte_offset() == tensor.view.byte_offset();
            let contained_view = tensor.view.byte_offset() >= source_tensor.view.byte_offset()
                && tensor.view.span_bytes() <= source_tensor.view.span_bytes();
            if source == id
                || source_tensor.class != Ministral3TensorClass::Activation
                || !source_tensor.view.is_contiguous()
                || !tensor.view.is_contiguous()
                || source_tensor.view.dtype() != tensor.view.dtype()
                || source_tensor.view.encoding() != tensor.view.encoding()
                || match alias_kind {
                    Some(Ministral3GraphNodeKind::Reshape) => !exact_payload,
                    Some(Ministral3GraphNodeKind::View) => !contained_view,
                    _ => true,
                }
            {
                return Err(Ministral3GraphError::InvalidTopology(
                    "zero-copy alias requires equal contiguous dtype and bytes",
                ));
            }
        }
    }

    for (id, node) in graph.nodes.iter().enumerate() {
        if node.id != id
            || node.dependencies.iter().any(|dependency| *dependency >= id)
            || node
                .inputs
                .iter()
                .any(|input| *input >= graph.tensors.len())
            || node
                .outputs
                .iter()
                .any(|output| *output >= graph.tensors.len())
        {
            return Err(Ministral3GraphError::InvalidOrder);
        }
        if node
            .outputs
            .iter()
            .any(|output| graph.tensors[*output].writer != Some(id))
        {
            return Err(Ministral3GraphError::InvalidTensorWriter);
        }
        if matches!(
            &node.kind,
            Ministral3GraphNodeKind::View | Ministral3GraphNodeKind::Reshape
        ) {
            if node.operation.is_some() || node.inputs.len() != 1 || node.outputs.len() != 1 {
                return Err(Ministral3GraphError::InvalidTopology(
                    "zero-copy alias node arity or semantic ownership",
                ));
            }
            let source = node.inputs[0];
            let output = node.outputs[0];
            let source_view = &graph.tensors[source];
            let output_tensor = &graph.tensors[output];
            let exact_payload = source_view.view.element_count()
                == output_tensor.view.element_count()
                && source_view.view.payload_bytes() == output_tensor.view.payload_bytes()
                && source_view.view.byte_offset() == output_tensor.view.byte_offset();
            let contained_view = output_tensor.view.byte_offset() >= source_view.view.byte_offset()
                && output_tensor.view.span_bytes() <= source_view.view.span_bytes();
            if output_tensor.alias_of != Some(source)
                || output_tensor.class != Ministral3TensorClass::Activation
                || !source_view.view.is_contiguous()
                || !output_tensor.view.is_contiguous()
                || source_view.view.dtype() != output_tensor.view.dtype()
                || source_view.view.encoding() != output_tensor.view.encoding()
                || match &node.kind {
                    Ministral3GraphNodeKind::Reshape => !exact_payload,
                    Ministral3GraphNodeKind::View => !contained_view,
                    _ => true,
                }
            {
                return Err(Ministral3GraphError::InvalidTopology(
                    "zero-copy alias node binding",
                ));
            }
        } else {
            match (node.kind.reused_semantic_kind(), node.operation.as_ref()) {
                (Some(kind), Some(operation)) if operation.kind() == kind => {
                    operation
                        .validate()
                        .map_err(|error| Ministral3GraphError::Semantic(error.to_string()))?;
                    if operation.inputs()
                        != node
                            .inputs
                            .iter()
                            .map(|input| graph.tensors[*input].view.clone())
                            .collect::<Vec<_>>()
                        || operation.outputs()
                            != node
                                .outputs
                                .iter()
                                .map(|output| graph.tensors[*output].view.clone())
                                .collect::<Vec<_>>()
                    {
                        return Err(Ministral3GraphError::InvalidTopology(
                            "semantic tensor binding",
                        ));
                    }
                }
                (None, None) => {}
                _ => {
                    return Err(Ministral3GraphError::InvalidTopology(
                        "semantic operation ownership",
                    ));
                }
            }
        }
    }

    validate_weights(graph)?;
    validate_model_specific_stages(graph)?;
    validate_boundaries(graph)?;
    Ok(())
}

fn validate_weights(graph: &Ministral3TextGraph) -> Result<(), Ministral3GraphError> {
    let expected = expected_weight_shapes();
    let observed = graph
        .tensors
        .iter()
        .filter_map(|tensor| tensor.weight.map(|key| (key, tensor.view.shape().to_vec())))
        .collect::<BTreeMap<_, _>>();
    if observed != expected || observed.len() != 236 {
        return Err(Ministral3GraphError::InvalidWeightContract);
    }
    let mut uses = BTreeMap::<WeightConsumerKey, usize>::new();
    for node in &graph.nodes {
        if let Some(weight) = node.kind.weight() {
            *uses.entry(weight).or_default() += 1;
        }
        if let Ministral3GraphNodeKind::RmsNorm {
            epsilon_bits,
            scale_mode,
            ..
        } = node.kind
        {
            if epsilon_bits != MINISTRAL3_GRAPH_RMS_EPSILON.to_bits()
                || scale_mode != RmsNormScaleMode::Direct
            {
                return Err(Ministral3GraphError::InvalidWeightContract);
            }
        }
    }
    for key in expected.keys() {
        let expected_uses = if key.role == WeightConsumer::EmbeddingAndTiedOutput {
            2
        } else {
            1
        };
        if uses.get(key).copied() != Some(expected_uses) {
            return Err(Ministral3GraphError::InvalidWeightContract);
        }
    }
    Ok(())
}

fn validate_model_specific_stages(graph: &Ministral3TextGraph) -> Result<(), Ministral3GraphError> {
    let mut kv_writers = BTreeMap::new();
    let mut yarn_layers = BTreeSet::new();
    let mut attention_layers = BTreeSet::new();
    for node in &graph.nodes {
        match node.kind {
            Ministral3GraphNodeKind::YarnRopeQueryScale(stage) => {
                let layer = node.layer.ok_or(Ministral3GraphError::InvalidTopology(
                    "YaRN stage has no layer",
                ))?;
                if !yarn_layers.insert(layer)
                    || stage.start_position() != graph.start_position
                    || u64::from(stage.token_count()) != graph.token_count
                    || stage.q_heads() != MINISTRAL3_GRAPH_Q_HEADS
                    || stage.kv_heads() != MINISTRAL3_GRAPH_KV_HEADS
                    || stage.head_dim() != MINISTRAL3_GRAPH_HEAD_DIM
                    || stage.rotary_dim() != MINISTRAL3_GRAPH_HEAD_DIM
                    || stage.pairing() != Ministral3RotaryPairing::AdjacentAfterGgufHeadPermutation
                    || stage.theta().to_bits() != MINISTRAL3_GRAPH_ROPE_THETA.to_bits()
                    || stage.factor().to_bits() != MINISTRAL3_GRAPH_YARN_FACTOR.to_bits()
                    || stage.beta_fast().to_bits() != MINISTRAL3_GRAPH_YARN_BETA_FAST.to_bits()
                    || stage.beta_slow().to_bits() != MINISTRAL3_GRAPH_YARN_BETA_SLOW.to_bits()
                    || u64::from(stage.original_context()) != MINISTRAL3_GRAPH_ORIGINAL_CONTEXT
                    || u64::from(stage.maximum_context()) != MINISTRAL3_GRAPH_MAX_CONTEXT
                    || stage.query_scale_beta().to_bits()
                        != MINISTRAL3_GRAPH_LLAMA4_SCALING_BETA.to_bits()
                    || stage.query_scale_application()
                        != Ministral3QueryScaleApplication::QueryOnlyAfterRotary
                    || node.inputs.len() != 3
                    || node.outputs.len() != 2
                    || graph.tensors[node.inputs[0]].view.shape()
                        != [
                            graph.token_count as usize,
                            MINISTRAL3_GRAPH_Q_HEADS as usize,
                            MINISTRAL3_GRAPH_HEAD_DIM as usize,
                        ]
                    || graph.tensors[node.inputs[1]].view.shape()
                        != [
                            graph.token_count as usize,
                            MINISTRAL3_GRAPH_KV_HEADS as usize,
                            MINISTRAL3_GRAPH_HEAD_DIM as usize,
                        ]
                    || graph.tensors[node.inputs[2]].view.shape() != [graph.token_count as usize]
                    || graph.tensors[node.inputs[2]].view.dtype() != DType::I32
                    || graph.tensors[node.outputs[0]].view.shape()
                        != [
                            graph.token_count as usize,
                            MINISTRAL3_GRAPH_Q_HEADS as usize,
                            MINISTRAL3_GRAPH_HEAD_DIM as usize,
                        ]
                    || graph.tensors[node.outputs[1]].view.shape()
                        != [
                            graph.token_count as usize,
                            MINISTRAL3_GRAPH_KV_HEADS as usize,
                            MINISTRAL3_GRAPH_HEAD_DIM as usize,
                        ]
                    || graph.tensors[node.inputs[0]].alias_of.is_none()
                    || graph.tensors[node.inputs[1]].alias_of.is_none()
                {
                    return Err(Ministral3GraphError::InvalidTopology("YaRN stage"));
                }
            }
            Ministral3GraphNodeKind::KvAppend(contract) => {
                let layer = contract.layer();
                if kv_writers.insert(layer, node.id).is_some() {
                    return Err(Ministral3GraphError::DuplicateKvWriter { layer });
                }
                if node.layer != Some(layer)
                    || contract.heads() != MINISTRAL3_GRAPH_KV_HEADS
                    || contract.head_dim() != MINISTRAL3_GRAPH_HEAD_DIM
                    || contract.previous_length() != graph.start_position
                    || u64::from(contract.append_count()) != graph.token_count
                    || contract.published_length() != graph.expected_length
                    || contract.capacity() != graph.state_capacity
                    || node.inputs.len() != 2
                    || !node.outputs.is_empty()
                    || graph.tensors[node.inputs[0]].view.shape()
                        != [
                            graph.token_count as usize,
                            MINISTRAL3_GRAPH_KV_HEADS as usize,
                            MINISTRAL3_GRAPH_HEAD_DIM as usize,
                        ]
                    || graph.tensors[node.inputs[1]].view.shape()
                        != [
                            graph.token_count as usize,
                            MINISTRAL3_GRAPH_KV_HEADS as usize,
                            MINISTRAL3_GRAPH_HEAD_DIM as usize,
                        ]
                    || graph.tensors[node.inputs[1]].alias_of.is_none()
                {
                    return Err(Ministral3GraphError::InvalidTopology("KV append"));
                }
            }
            Ministral3GraphNodeKind::CausalGqa(contract) => {
                let layer = node.layer.ok_or(Ministral3GraphError::InvalidTopology(
                    "causal attention has no layer",
                ))?;
                if !attention_layers.insert(layer)
                    || contract.q_heads() != MINISTRAL3_GRAPH_Q_HEADS
                    || contract.kv_heads() != MINISTRAL3_GRAPH_KV_HEADS
                    || contract.head_dim() != MINISTRAL3_GRAPH_HEAD_DIM
                    || contract.start_position() != graph.start_position
                    || u64::from(contract.query_count()) != graph.token_count
                    || contract.expected_kv_length() != graph.expected_length
                    || contract.sliding_window().is_some()
                    || contract.scaling_bits() != ministral3_attention_scale().to_bits()
                    || contract.accumulation_dtype() != DType::F32
                    || contract.output_dtype() != DType::Bf16
                    || node.inputs.len() != 1
                    || node.outputs.len() != 1
                    || graph.tensors[node.inputs[0]].view.shape()
                        != [
                            graph.token_count as usize,
                            MINISTRAL3_GRAPH_Q_HEADS as usize,
                            MINISTRAL3_GRAPH_HEAD_DIM as usize,
                        ]
                    || graph.tensors[node.outputs[0]].view.shape()
                        != [
                            graph.token_count as usize,
                            MINISTRAL3_GRAPH_Q_HEADS as usize,
                            MINISTRAL3_GRAPH_HEAD_DIM as usize,
                        ]
                {
                    return Err(Ministral3GraphError::InvalidTopology("causal GQA"));
                }
                let writer = kv_writers
                    .get(&layer)
                    .ok_or(Ministral3GraphError::MissingKvWriter { layer })?;
                if !node.dependencies.contains(writer) {
                    return Err(Ministral3GraphError::InvalidOrder);
                }
            }
            _ => {}
        }
    }
    for layer in 0..MINISTRAL3_GRAPH_LAYER_COUNT {
        if !kv_writers.contains_key(&layer) {
            return Err(Ministral3GraphError::MissingKvWriter { layer });
        }
        if !yarn_layers.contains(&layer) || !attention_layers.contains(&layer) {
            return Err(Ministral3GraphError::InvalidTopology(
                "model-specific stage coverage",
            ));
        }
    }
    if graph.kv_contracts.len() != MINISTRAL3_GRAPH_LAYER_COUNT as usize
        || graph
            .kv_contracts
            .iter()
            .enumerate()
            .any(|(layer, contract)| contract.layer() as usize != layer)
    {
        return Err(Ministral3GraphError::InvalidTopology("KV catalog"));
    }
    validate_attention_projection_layout(graph)?;
    Ok(())
}

fn validate_attention_projection_layout(
    graph: &Ministral3TextGraph,
) -> Result<(), Ministral3GraphError> {
    let q_shape = [
        graph.token_count as usize,
        MINISTRAL3_GRAPH_Q_HEADS as usize,
        MINISTRAL3_GRAPH_HEAD_DIM as usize,
    ];
    let kv_shape = [
        graph.token_count as usize,
        MINISTRAL3_GRAPH_KV_HEADS as usize,
        MINISTRAL3_GRAPH_HEAD_DIM as usize,
    ];
    for node in &graph.nodes {
        let Ministral3GraphNodeKind::Matmul { weight } = node.kind else {
            continue;
        };
        match weight.role {
            WeightConsumer::AttentionQ
            | WeightConsumer::AttentionK
            | WeightConsumer::AttentionV
            | WeightConsumer::AttentionO => {
                if weight.layer.is_none() {
                    return Err(Ministral3GraphError::InvalidTopology(
                        "attention projection has no layer",
                    ));
                }
            }
            _ => continue,
        }
        if node.inputs.len() != 2 || node.outputs.len() != 1 {
            return Err(Ministral3GraphError::InvalidTopology(
                "attention projection arity",
            ));
        }
        let input = &graph.tensors[node.inputs[0]].view;
        let output = &graph.tensors[node.outputs[0]].view;
        match weight.role {
            WeightConsumer::AttentionQ
            | WeightConsumer::AttentionK
            | WeightConsumer::AttentionV => {
                let expected_width = match weight.role {
                    WeightConsumer::AttentionQ => MINISTRAL3_GRAPH_Q_WIDTH,
                    WeightConsumer::AttentionK | WeightConsumer::AttentionV => {
                        MINISTRAL3_GRAPH_KV_WIDTH
                    }
                    _ => unreachable!("role is narrowed above"),
                };
                let expected_shape = if weight.role == WeightConsumer::AttentionQ {
                    q_shape
                } else {
                    kv_shape
                };
                if input.shape() != [graph.token_count as usize, MINISTRAL3_GRAPH_HIDDEN_SIZE]
                    || output.shape() != [graph.token_count as usize, expected_width]
                {
                    return Err(Ministral3GraphError::InvalidTopology(
                        "attention projection flat matmul shape",
                    ));
                }
                let aliases = graph
                    .tensors
                    .iter()
                    .filter_map(|tensor| {
                        (tensor.alias_of == Some(node.outputs[0]))
                            .then_some(tensor)
                            .filter(|tensor| tensor.view.shape() == expected_shape)
                    })
                    .collect::<Vec<_>>();
                if aliases.len() != 1 {
                    return Err(Ministral3GraphError::InvalidTopology(
                        "attention projection reshape alias",
                    ));
                }
                let alias_writer = aliases[0]
                    .writer
                    .ok_or(Ministral3GraphError::InvalidTensorWriter)?;
                if !matches!(
                    graph.nodes[alias_writer].kind,
                    Ministral3GraphNodeKind::Reshape
                ) {
                    return Err(Ministral3GraphError::InvalidTopology(
                        "attention projection reshape node",
                    ));
                }
            }
            WeightConsumer::AttentionO => {
                if input.shape() != [graph.token_count as usize, MINISTRAL3_GRAPH_Q_WIDTH]
                    || output.shape() != [graph.token_count as usize, MINISTRAL3_GRAPH_HIDDEN_SIZE]
                {
                    return Err(Ministral3GraphError::InvalidTopology(
                        "attention O projection flat matmul shape",
                    ));
                }
                let source = graph.tensors[node.inputs[0]].alias_of.ok_or(
                    Ministral3GraphError::InvalidTopology(
                        "attention O projection does not consume a view",
                    ),
                )?;
                if graph.tensors[source].view.shape() != q_shape
                    || graph.tensors[node.inputs[0]].view.encoding()
                        != graph.tensors[source].view.encoding()
                {
                    return Err(Ministral3GraphError::InvalidTopology(
                        "attention O projection view shape",
                    ));
                }
                let alias_writer = graph.tensors[node.inputs[0]]
                    .writer
                    .ok_or(Ministral3GraphError::InvalidTensorWriter)?;
                if !matches!(
                    graph.nodes[alias_writer].kind,
                    Ministral3GraphNodeKind::View
                ) {
                    return Err(Ministral3GraphError::InvalidTopology(
                        "attention O projection view node",
                    ));
                }
            }
            _ => unreachable!("role is narrowed above"),
        }
    }
    Ok(())
}

fn validate_boundaries(graph: &Ministral3TextGraph) -> Result<(), Ministral3GraphError> {
    let state = graph
        .nodes
        .iter()
        .filter(|node| node.boundary_after == Some(ExecutionBoundaryKind::StatePublication))
        .collect::<Vec<_>>();
    let terminal = graph
        .nodes
        .iter()
        .filter(|node| node.boundary_after == Some(ExecutionBoundaryKind::TerminalReadback))
        .collect::<Vec<_>>();
    if state.len() != 1
        || terminal.len() != 1
        || state[0].layer != Some(MINISTRAL3_GRAPH_LAYER_COUNT - 1)
        || !matches!(state[0].kind, Ministral3GraphNodeKind::Add)
        || !matches!(terminal[0].kind, Ministral3GraphNodeKind::Argmax)
        || state[0].id >= terminal[0].id
    {
        return Err(Ministral3GraphError::InvalidBoundary);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(token_count: u64, end: u64) -> Ministral3TextGraph {
        build_ministral3_text_graph(token_count, end - token_count, end)
            .expect("reviewed graph builds")
    }

    #[test]
    fn exact_text_topology_uses_only_reviewed_dense_semantics() {
        let graph = graph(3, 17);
        assert!(graph.is_container_neutral());
        assert!(graph.has_production_executor());
        assert_eq!(graph.nodes().len(), 499);
        assert_eq!(graph.kv_contracts().len(), 26);
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| matches!(
                    node.kind(),
                    Ministral3GraphNodeKind::Reshape | Ministral3GraphNodeKind::View
                ))
                .count(),
            105
        );
        assert_eq!(
            graph
                .tensors()
                .iter()
                .filter(|tensor| tensor.is_zero_copy_alias())
                .count(),
            105
        );
        assert_eq!(
            graph
                .tensors()
                .iter()
                .filter(|tensor| tensor.class() == Ministral3TensorClass::Weight)
                .count(),
            236
        );

        let semantic_kinds = graph
            .nodes()
            .iter()
            .filter_map(|node| node.operation().map(SemanticOpDescriptor::kind))
            .collect::<Vec<_>>();
        assert!(!semantic_kinds.contains(&SemanticOpKind::AttentionPreprocess));
        assert!(!semantic_kinds.contains(&SemanticOpKind::SigmoidMul));
        assert!(!semantic_kinds.contains(&SemanticOpKind::Rotary));
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| matches!(
                    node.kind(),
                    Ministral3GraphNodeKind::YarnRopeQueryScale(_)
                ))
                .count(),
            26
        );
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), Ministral3GraphNodeKind::CausalGqa(_)))
                .count(),
            26
        );

        for node in graph.nodes() {
            if let Ministral3GraphNodeKind::RmsNorm {
                epsilon_bits,
                scale_mode,
                ..
            } = node.kind()
            {
                assert_eq!(*epsilon_bits, MINISTRAL3_GRAPH_RMS_EPSILON.to_bits());
                assert_eq!(*scale_mode, RmsNormScaleMode::Direct);
            }
            assert!(!matches!(
                node.kind().weight().map(|weight| weight.role),
                Some(WeightConsumer::AttentionQNorm | WeightConsumer::AttentionKNorm)
            ));
        }

        let q = graph
            .nodes()
            .iter()
            .find(|node| node.label() == "layer.0.q_proj")
            .unwrap();
        let k = graph
            .nodes()
            .iter()
            .find(|node| node.label() == "layer.0.k_proj")
            .unwrap();
        let v = graph
            .nodes()
            .iter()
            .find(|node| node.label() == "layer.0.v_proj")
            .unwrap();
        assert_eq!(graph.tensors()[q.outputs()[0]].view().shape(), [3, 4_096]);
        assert_eq!(graph.tensors()[k.outputs()[0]].view().shape(), [3, 1_024]);
        assert_eq!(graph.tensors()[v.outputs()[0]].view().shape(), [3, 1_024]);
    }

    #[test]
    fn attention_layout_uses_zero_copy_head_views_and_flat_o_input() {
        let graph = graph(3, 17);
        for layer in [0, 25] {
            let find_node = |label: &str| {
                graph
                    .nodes()
                    .iter()
                    .find(|node| node.label() == label)
                    .unwrap_or_else(|| panic!("missing node {label}"))
            };
            let find_tensor = |label: &str| {
                graph
                    .tensors()
                    .iter()
                    .find(|tensor| tensor.label() == label)
                    .unwrap_or_else(|| panic!("missing tensor {label}"))
            };
            let q = find_node(&format!("layer.{layer}.q_proj"));
            let k = find_node(&format!("layer.{layer}.k_proj"));
            let v = find_node(&format!("layer.{layer}.v_proj"));
            for (projection, shape) in [(q, [3, 32, 128]), (k, [3, 8, 128]), (v, [3, 8, 128])] {
                assert_eq!(projection.outputs().len(), 1);
                let source = projection.outputs()[0];
                let alias = find_tensor(&format!(
                    "layer.{layer}.{}.reshape",
                    match projection.kind() {
                        Ministral3GraphNodeKind::Matmul { weight }
                            if weight.role == WeightConsumer::AttentionQ =>
                            "q_proj",
                        Ministral3GraphNodeKind::Matmul { weight }
                            if weight.role == WeightConsumer::AttentionK =>
                            "k_proj",
                        _ => "v_proj",
                    }
                ));
                assert_eq!(alias.alias_of(), Some(source));
                assert_eq!(alias.view().shape(), shape);
                assert!(alias.view().is_contiguous());
                assert_eq!(alias.view().dtype(), graph.tensors()[source].view().dtype());
                assert_eq!(
                    alias.view().payload_bytes(),
                    graph.tensors()[source].view().payload_bytes()
                );
                let writer = alias.writer().unwrap();
                assert!(matches!(
                    graph.nodes()[writer].kind(),
                    Ministral3GraphNodeKind::Reshape
                ));
                assert!(graph.nodes()[writer].operation().is_none());
            }

            let causal = find_node(&format!("layer.{layer}.causal_gqa"));
            assert_eq!(causal.inputs().len(), 1);
            assert_eq!(causal.outputs().len(), 1);
            assert_eq!(
                graph.tensors()[causal.inputs()[0]].view().shape(),
                [3, 32, 128]
            );
            assert_eq!(
                graph.tensors()[causal.outputs()[0]].view().shape(),
                [3, 32, 128]
            );
            let attention_view = find_tensor(&format!("layer.{layer}.attention.output.view"));
            assert_eq!(attention_view.alias_of(), Some(causal.outputs()[0]));
            assert_eq!(attention_view.view().shape(), [3, 4_096]);
            let attention_view_writer = attention_view.writer().unwrap();
            assert!(matches!(
                graph.nodes()[attention_view_writer].kind(),
                Ministral3GraphNodeKind::View
            ));
            let o = find_node(&format!("layer.{layer}.o_proj"));
            assert_eq!(o.inputs()[0], attention_view.id());
            assert_eq!(graph.tensors()[o.inputs()[0]].view().shape(), [3, 4_096]);
        }
    }

    #[test]
    fn zero_copy_alias_validation_rejects_byte_or_writer_drift() {
        let mut bytes = graph(3, 3);
        let alias = bytes
            .tensors
            .iter()
            .position(|tensor| tensor.label() == "layer.0.q_proj.reshape")
            .unwrap();
        bytes.tensors[alias].view = TensorView::contiguous(DType::Bf16, &[1, 4_096]).unwrap();
        assert_eq!(
            validate_graph(&bytes),
            Err(Ministral3GraphError::InvalidTopology(
                "zero-copy alias requires equal contiguous dtype and bytes"
            ))
        );

        let mut writer = graph(3, 3);
        let alias = writer
            .tensors
            .iter()
            .position(|tensor| tensor.label() == "layer.0.q_proj.reshape")
            .unwrap();
        writer.tensors[alias].writer = None;
        assert_eq!(
            validate_graph(&writer),
            Err(Ministral3GraphError::InvalidTensorWriter)
        );
    }

    #[test]
    fn token_counts_cover_non_aligned_execution_shapes() {
        for token_count in [1, 3, 17] {
            let graph = graph(token_count, token_count);
            assert_eq!(graph.token_count(), token_count);
            for tensor in graph
                .tensors()
                .iter()
                .filter(|tensor| tensor.class() != Ministral3TensorClass::Weight)
            {
                let expected_rows = if matches!(
                    tensor.label(),
                    "final_norm.terminal" | "logits" | "argmax.output"
                ) {
                    1
                } else {
                    token_count as usize
                };
                assert_eq!(tensor.view().shape()[0], expected_rows);
            }

            let final_norm = graph
                .tensors()
                .iter()
                .find(|tensor| tensor.label() == "final_norm.output")
                .expect("final norm output");
            let terminal = graph
                .tensors()
                .iter()
                .find(|tensor| tensor.label() == "final_norm.terminal")
                .expect("terminal final norm view");
            assert_eq!(terminal.alias_of(), Some(final_norm.id()));
            assert_eq!(terminal.view().shape(), [1, MINISTRAL3_GRAPH_HIDDEN_SIZE]);
            assert_eq!(
                terminal.view().byte_offset(),
                (token_count - 1) * MINISTRAL3_GRAPH_HIDDEN_SIZE as u64 * 2
            );
        }
    }

    #[test]
    fn original_and_maximum_context_boundaries_fail_closed() {
        for end in [16_383, 16_384, 16_385, 262_143, 262_144] {
            let graph = graph(1, end);
            assert_eq!(graph.expected_length(), end);
            assert_eq!(graph.state_capacity(), end);
        }
        assert!(matches!(
            build_ministral3_text_graph(1, 262_144, 262_144),
            Err(Ministral3GraphError::ContextExceeded { end: 262_145, .. })
        ));
        assert!(matches!(
            build_ministral3_text_graph(1, 262_144, 262_145),
            Err(Ministral3GraphError::InvalidRequest(_))
        ));
        assert!(matches!(
            build_ministral3_text_graph(0, 0, 1),
            Err(Ministral3GraphError::InvalidRequest(_))
        ));
    }

    #[test]
    fn request_overflow_and_capacity_are_rejected_before_shapes() {
        assert_eq!(
            build_ministral3_text_graph(1, u64::MAX, 1),
            Err(Ministral3GraphError::LengthOverflow)
        );
        assert!(matches!(
            build_ministral3_text_graph(17, 0, 16),
            Err(Ministral3GraphError::StateCapacityExceeded {
                end: 17,
                capacity: 16
            })
        ));
    }

    #[test]
    fn every_layer_has_one_ordered_kv_writer() {
        let graph = graph(17, 16_385);
        let mut writers = BTreeMap::new();
        for node in graph.nodes() {
            if let Ministral3GraphNodeKind::KvAppend(contract) = node.kind() {
                assert!(writers.insert(contract.layer(), node.id()).is_none());
            }
            if let Ministral3GraphNodeKind::CausalGqa(_) = node.kind() {
                let writer = writers[&node.layer().unwrap()];
                assert!(node.dependencies().contains(&writer));
            }
        }
        assert_eq!(writers.len(), 26);
    }

    #[test]
    fn duplicate_state_writer_and_order_drift_are_rejected() {
        let mut duplicate = graph(1, 1);
        let layer_one = duplicate
            .nodes
            .iter_mut()
            .find(|node| matches!(node.kind, Ministral3GraphNodeKind::KvAppend(contract) if contract.layer() == 1))
            .unwrap();
        if let Ministral3GraphNodeKind::KvAppend(mut contract) = layer_one.kind {
            contract.layer = 0;
            layer_one.kind = Ministral3GraphNodeKind::KvAppend(contract);
        }
        assert_eq!(
            validate_graph(&duplicate),
            Err(Ministral3GraphError::DuplicateKvWriter { layer: 0 })
        );

        let mut unordered = graph(1, 1);
        unordered.nodes[1].dependencies.push(1);
        assert_eq!(
            validate_graph(&unordered),
            Err(Ministral3GraphError::InvalidOrder)
        );
    }

    #[test]
    fn state_and_terminal_boundaries_are_single_and_ordered() {
        let graph = graph(1, 1);
        let boundaries = graph
            .nodes()
            .iter()
            .filter_map(|node| node.boundary_after().map(|boundary| (node.id(), boundary)))
            .collect::<Vec<_>>();
        assert_eq!(
            boundaries,
            vec![
                (494, ExecutionBoundaryKind::StatePublication),
                (498, ExecutionBoundaryKind::TerminalReadback)
            ]
        );

        let mut duplicate = graph;
        duplicate.nodes[374].boundary_after = Some(ExecutionBoundaryKind::StatePublication);
        assert_eq!(
            validate_graph(&duplicate),
            Err(Ministral3GraphError::InvalidBoundary)
        );
    }

    #[test]
    fn yarn_and_attention_contracts_fix_long_context_semantics() {
        let graph = graph(1, 16_384);
        let yarn = graph
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                Ministral3GraphNodeKind::YarnRopeQueryScale(stage) => Some(*stage),
                _ => None,
            })
            .unwrap();
        assert_eq!(yarn.start_position(), 16_383);
        assert_eq!(yarn.rotary_dim(), 128);
        assert_eq!(
            yarn.pairing(),
            Ministral3RotaryPairing::AdjacentAfterGgufHeadPermutation
        );
        assert_eq!(yarn.theta().to_bits(), 1_000_000.0_f32.to_bits());
        assert_eq!(yarn.factor().to_bits(), 16.0_f32.to_bits());
        assert_eq!(yarn.beta_fast().to_bits(), 32.0_f64.to_bits());
        assert_eq!(yarn.beta_slow().to_bits(), 1.0_f64.to_bits());
        assert_eq!(yarn.original_context(), 16_384);
        assert_eq!(yarn.maximum_context(), 262_144);
        assert_eq!(yarn.query_scale_beta().to_bits(), 0.1_f32.to_bits());
        assert_eq!(
            yarn.query_scale_application(),
            Ministral3QueryScaleApplication::QueryOnlyAfterRotary
        );

        let attention = graph
            .nodes()
            .iter()
            .find_map(|node| match node.kind() {
                Ministral3GraphNodeKind::CausalGqa(contract) => Some(*contract),
                _ => None,
            })
            .unwrap();
        assert_eq!(attention.q_heads(), 32);
        assert_eq!(attention.kv_heads(), 8);
        assert_eq!(attention.head_dim(), 128);
        assert_eq!(attention.sliding_window(), None);
        assert_eq!(
            attention.scaling_bits(),
            ministral3_attention_scale().to_bits()
        );
        assert_eq!(attention.accumulation_dtype(), DType::F32);
        assert_eq!(attention.output_dtype(), DType::Bf16);
    }
}
