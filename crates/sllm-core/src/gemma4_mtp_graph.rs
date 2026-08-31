//! Structural graph for the reviewed Gemma 4 12B MTP assistant.
//!
//! The assistant is deliberately not a small standalone Gemma decoder.  It
//! consumes the paired target's 3,840-wide embedding/hidden row and reads the
//! published target KV state through a read-only lease.  Consequently this
//! graph has no K/V projections, no request KV descriptors, and no assistant
//! prefill state.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::gemma4::Gemma4LayerType;
use crate::op::{
    OpError, RmsNormScaleMode, RotaryPositionModeV1, SemanticOpKind, SplitHalfRotaryContract,
    WindowedCausalAttentionContract,
};
use crate::prepared_execution::ExecutionBoundaryKind;
use crate::weights::{
    WeightClassification, WeightConsumer, WeightConsumerKey, WeightLoadPlan,
    build_verified_gemma4_mtp_weight_load_plan,
};
use crate::{Gemma4MtpModelLock, Gemma4MtpWeightSource};

pub const GEMMA4_MTP_HIDDEN_SIZE: u64 = 1_024;
pub const GEMMA4_MTP_BACKBONE_HIDDEN_SIZE: u64 = 3_840;
pub const GEMMA4_MTP_INTERMEDIATE_SIZE: u64 = 8_192;
pub const GEMMA4_MTP_VOCAB_SIZE: u64 = 262_144;
pub const GEMMA4_MTP_LAYER_COUNT: usize = 4;
pub const GEMMA4_MTP_SLIDING_WINDOW: u64 = 1_024;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MtpBindingClass {
    TargetReadOnly,
    AssistantResident,
    Workspace,
    TerminalOutput,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MtpNormRole {
    Input,
    Query,
    PostAttention,
    PreFeedforward,
    PostFeedforward,
    Final,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MtpRopeType {
    Default,
    Proportional,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Gemma4MtpRopeDescriptor {
    pub rope_type: Gemma4MtpRopeType,
    pub theta: u64,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub q_heads: u32,
    pub dummy_kv_heads: u32,
    pub absolute_position: u64,
}

impl Gemma4MtpRopeDescriptor {
    /// The existing split-half rotary semantic rotates Q and K together.  The
    /// assistant has no K projection, so lowering supplies a zeroed temporary
    /// K row and discards its output.  This is workspace, never assistant KV.
    pub fn semantic_contract(self) -> Result<SplitHalfRotaryContract, OpError> {
        SplitHalfRotaryContract::new_with_position_mode(
            self.q_heads,
            self.dummy_kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.theta as f32,
            self.absolute_position,
            1,
            u32::MAX,
            RotaryPositionModeV1::Explicit,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Gemma4MtpAttentionDescriptor {
    pub assistant_layer: u32,
    pub target_layer: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub target_kv_length: u64,
    pub sliding_window: Option<u64>,
}

impl Gemma4MtpAttentionDescriptor {
    pub fn semantic_contract(self) -> Result<WindowedCausalAttentionContract, OpError> {
        let query_position = self
            .target_kv_length
            .checked_sub(1)
            .ok_or(OpError::CausalAttentionLengthOverflow)?;
        WindowedCausalAttentionContract::new(
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            query_position,
            1,
            self.target_kv_length,
            self.sliding_window,
            1.0,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MtpGraphNodeKind {
    /// Gather from the paired target's read-only 262144x3840 embedding.
    TargetEmbedding,
    ScaleConstant {
        value_bits: u32,
    },
    /// One of two existing Copy submissions into the 7,680-wide concat row.
    CopyToConcat {
        destination_column: u32,
    },
    Matmul {
        consumer: WeightConsumer,
        weight: String,
    },
    RmsNorm {
        role: Gemma4MtpNormRole,
        scale_mode: RmsNormScaleMode,
        epsilon_bits: u32,
        weight: String,
    },
    QueryRotary(Gemma4MtpRopeDescriptor),
    SharedTargetAttention(Gemma4MtpAttentionDescriptor),
    GeluTanhMul,
    Add,
    ScaleWeight {
        weight: String,
    },
    Argmax,
}

impl Gemma4MtpGraphNodeKind {
    pub const fn semantic_kind(&self) -> SemanticOpKind {
        match self {
            Self::TargetEmbedding => SemanticOpKind::Embedding,
            Self::ScaleConstant { .. } | Self::ScaleWeight { .. } => SemanticOpKind::ScalarMul,
            Self::CopyToConcat { .. } => SemanticOpKind::Copy,
            Self::Matmul { .. } => SemanticOpKind::Matmul,
            Self::RmsNorm { .. } => SemanticOpKind::RmsNorm,
            Self::QueryRotary(_) => SemanticOpKind::Rotary,
            Self::SharedTargetAttention(_) => SemanticOpKind::CausalAttention,
            Self::GeluTanhMul => SemanticOpKind::GeluTanhMul,
            Self::Add => SemanticOpKind::Add,
            Self::Argmax => SemanticOpKind::Argmax,
        }
    }

    fn weight_name(&self) -> Option<&str> {
        match self {
            Self::Matmul { weight, .. }
            | Self::RmsNorm { weight, .. }
            | Self::ScaleWeight { weight } => Some(weight),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpGraphNode {
    id: usize,
    label: String,
    assistant_layer: Option<u32>,
    kind: Gemma4MtpGraphNodeKind,
    predecessors: Vec<usize>,
    binding_class: Gemma4MtpBindingClass,
    boundary_after: Option<ExecutionBoundaryKind>,
}

impl Gemma4MtpGraphNode {
    pub const fn id(&self) -> usize {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn assistant_layer(&self) -> Option<u32> {
        self.assistant_layer
    }

    pub const fn kind(&self) -> &Gemma4MtpGraphNodeKind {
        &self.kind
    }

    pub fn predecessors(&self) -> &[usize] {
        &self.predecessors
    }

    pub const fn binding_class(&self) -> Gemma4MtpBindingClass {
        self.binding_class
    }

    pub const fn boundary_after(&self) -> Option<ExecutionBoundaryKind> {
        self.boundary_after
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpGraph {
    assistant_fingerprint: String,
    target_fingerprint: String,
    weight_plan_digest: [u8; 32],
    target_kv_length: u64,
    absolute_query_position: u64,
    nodes: Vec<Gemma4MtpGraphNode>,
    attention: [Gemma4MtpAttentionDescriptor; GEMMA4_MTP_LAYER_COUNT],
}

impl Gemma4MtpGraph {
    pub fn assistant_fingerprint(&self) -> &str {
        &self.assistant_fingerprint
    }

    pub fn target_fingerprint(&self) -> &str {
        &self.target_fingerprint
    }

    pub const fn weight_plan_digest(&self) -> &[u8; 32] {
        &self.weight_plan_digest
    }

    pub const fn target_kv_length(&self) -> u64 {
        self.target_kv_length
    }

    pub const fn absolute_query_position(&self) -> u64 {
        self.absolute_query_position
    }

    pub fn nodes(&self) -> &[Gemma4MtpGraphNode] {
        &self.nodes
    }

    pub const fn attention_descriptors(
        &self,
    ) -> &[Gemma4MtpAttentionDescriptor; GEMMA4_MTP_LAYER_COUNT] {
        &self.attention
    }

    /// The assistant never owns or appends a KV cache.
    pub const fn assistant_kv_allocation_bytes(&self) -> u64 {
        0
    }

    /// Rebinds the immutable assistant topology to one published target tail.
    /// Every assistant layer receives the same absolute draft-round position;
    /// only the borrowed target K/V length and rotary position may change.
    pub(crate) fn with_target_snapshot(
        &self,
        target_kv_length: u64,
        absolute_query_position: u64,
    ) -> Result<Self, Gemma4MtpGraphError> {
        if target_kv_length == 0 || absolute_query_position < target_kv_length - 1 {
            return Err(Gemma4MtpGraphError::invalid(
                "assistant target snapshot position differs",
            ));
        }
        let mut graph = self.clone();
        graph.target_kv_length = target_kv_length;
        graph.absolute_query_position = absolute_query_position;
        let mut attention_index = 0_usize;
        for node in &mut graph.nodes {
            match &mut node.kind {
                Gemma4MtpGraphNodeKind::QueryRotary(descriptor) => {
                    descriptor.absolute_position = absolute_query_position;
                    descriptor
                        .semantic_contract()
                        .map_err(|error| Gemma4MtpGraphError::invalid(error.to_string()))?;
                }
                Gemma4MtpGraphNodeKind::SharedTargetAttention(descriptor) => {
                    descriptor.target_kv_length = target_kv_length;
                    descriptor
                        .semantic_contract()
                        .map_err(|error| Gemma4MtpGraphError::invalid(error.to_string()))?;
                    let slot = graph.attention.get_mut(attention_index).ok_or_else(|| {
                        Gemma4MtpGraphError::invalid("assistant attention catalog overflowed")
                    })?;
                    *slot = *descriptor;
                    attention_index += 1;
                }
                _ => {}
            }
        }
        if attention_index != GEMMA4_MTP_LAYER_COUNT {
            return Err(Gemma4MtpGraphError::invalid(
                "assistant attention catalog length differs",
            ));
        }
        Ok(graph)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MtpGraphError(String);

impl Gemma4MtpGraphError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Gemma4MtpGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Gemma 4 MTP graph: {}", self.0)
    }
}

impl std::error::Error for Gemma4MtpGraphError {}

pub fn build_gemma4_mtp_graph<S>(
    lock: &Gemma4MtpModelLock,
    source: &S,
    plan: &WeightLoadPlan,
    target_kv_length: u64,
    absolute_query_position: u64,
) -> Result<Gemma4MtpGraph, Gemma4MtpGraphError>
where
    S: Gemma4MtpWeightSource + ?Sized,
{
    if target_kv_length == 0
        || absolute_query_position < target_kv_length - 1
        || source.lock_fingerprint() != lock.fingerprint()
        || source.target_fingerprint() != lock.target_fingerprint()
    {
        return Err(Gemma4MtpGraphError::invalid(
            "assistant/target identity or published target position differs",
        ));
    }
    let expected_plan = build_verified_gemma4_mtp_weight_load_plan(lock, source)
        .map_err(|error| Gemma4MtpGraphError::invalid(error.to_string()))?;
    if plan != &expected_plan {
        return Err(Gemma4MtpGraphError::invalid(
            "assistant weight plan differs from the verified source",
        ));
    }
    let config = source.config();
    if config.hidden_size != GEMMA4_MTP_HIDDEN_SIZE as u32
        || config.backbone_hidden_size != GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as u32
        || config.intermediate_size != GEMMA4_MTP_INTERMEDIATE_SIZE as u32
        || config.layer_count != GEMMA4_MTP_LAYER_COUNT as u32
        || config.vocab_size != GEMMA4_MTP_VOCAB_SIZE as u32
        || config.draft_to_target_kv_layers != [46, 46, 46, 47]
    {
        return Err(Gemma4MtpGraphError::invalid(
            "verified assistant config differs from the graph contract",
        ));
    }

    let weights = weight_lookup(plan)?;
    let mut nodes = Vec::new();
    let target_embedding = push_node(
        &mut nodes,
        "target_embedding",
        None,
        Gemma4MtpGraphNodeKind::TargetEmbedding,
        vec![],
        Gemma4MtpBindingClass::TargetReadOnly,
        None,
    );
    let target_embedding_scaled = push_node(
        &mut nodes,
        "target_embedding_scale",
        None,
        Gemma4MtpGraphNodeKind::ScaleConstant {
            value_bits: (GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as f32).sqrt().to_bits(),
        },
        vec![target_embedding],
        Gemma4MtpBindingClass::Workspace,
        None,
    );
    let embedding_copy = push_node(
        &mut nodes,
        "concat_target_embedding",
        None,
        Gemma4MtpGraphNodeKind::CopyToConcat {
            destination_column: 0,
        },
        vec![target_embedding_scaled],
        Gemma4MtpBindingClass::Workspace,
        None,
    );
    let hidden_copy = push_node(
        &mut nodes,
        "concat_target_hidden",
        None,
        Gemma4MtpGraphNodeKind::CopyToConcat {
            destination_column: GEMMA4_MTP_BACKBONE_HIDDEN_SIZE as u32,
        },
        vec![],
        Gemma4MtpBindingClass::TargetReadOnly,
        None,
    );
    let mut hidden = matmul_node(
        &mut nodes,
        &weights,
        None,
        "pre_projection",
        WeightConsumer::Gemma4MtpPreProjection,
        vec![embedding_copy, hidden_copy],
    )?;

    let mut attention = Vec::with_capacity(GEMMA4_MTP_LAYER_COUNT);
    for (layer_index, layer_type) in config.layer_types.iter().copied().enumerate() {
        let layer = u32::try_from(layer_index)
            .map_err(|_| Gemma4MtpGraphError::invalid("assistant layer does not fit u32"))?;
        let input_norm = norm_node(
            &mut nodes,
            &weights,
            layer,
            "input_norm",
            Gemma4MtpNormRole::Input,
            WeightConsumer::InputNorm,
            hidden,
        )?;
        let q = matmul_node(
            &mut nodes,
            &weights,
            Some(layer),
            "q_proj",
            WeightConsumer::AttentionQ,
            vec![input_norm],
        )?;
        let q_norm = norm_node(
            &mut nodes,
            &weights,
            layer,
            "q_norm",
            Gemma4MtpNormRole::Query,
            WeightConsumer::AttentionQNorm,
            q,
        )?;
        let (head_dim, kv_heads, rope_type, theta, rotary_dim, window) = match layer_type {
            Gemma4LayerType::SlidingAttention => (
                256_u32,
                8_u32,
                Gemma4MtpRopeType::Default,
                10_000_u64,
                256_u32,
                Some(GEMMA4_MTP_SLIDING_WINDOW),
            ),
            Gemma4LayerType::FullAttention => (
                512_u32,
                1_u32,
                Gemma4MtpRopeType::Proportional,
                1_000_000_u64,
                128_u32,
                None,
            ),
        };
        let rotary = push_node(
            &mut nodes,
            format!("layer.{layer}.query_rotary"),
            Some(layer),
            Gemma4MtpGraphNodeKind::QueryRotary(Gemma4MtpRopeDescriptor {
                rope_type,
                theta,
                head_dim,
                rotary_dim,
                q_heads: 16,
                dummy_kv_heads: kv_heads,
                absolute_position: absolute_query_position,
            }),
            vec![q_norm],
            Gemma4MtpBindingClass::Workspace,
            None,
        );
        let descriptor = Gemma4MtpAttentionDescriptor {
            assistant_layer: layer,
            target_layer: config.draft_to_target_kv_layers[layer_index],
            q_heads: 16,
            kv_heads,
            head_dim,
            target_kv_length,
            sliding_window: window,
        };
        descriptor
            .semantic_contract()
            .map_err(|error| Gemma4MtpGraphError::invalid(error.to_string()))?;
        let attended = push_node(
            &mut nodes,
            format!("layer.{layer}.shared_target_attention"),
            Some(layer),
            Gemma4MtpGraphNodeKind::SharedTargetAttention(descriptor),
            vec![rotary],
            Gemma4MtpBindingClass::TargetReadOnly,
            None,
        );
        attention.push(descriptor);
        let attention_output = matmul_node(
            &mut nodes,
            &weights,
            Some(layer),
            "o_proj",
            WeightConsumer::AttentionO,
            vec![attended],
        )?;
        let post_attention = norm_node(
            &mut nodes,
            &weights,
            layer,
            "post_attention_norm",
            Gemma4MtpNormRole::PostAttention,
            WeightConsumer::PostAttentionNorm,
            attention_output,
        )?;
        let attention_residual = push_node(
            &mut nodes,
            format!("layer.{layer}.attention_residual"),
            Some(layer),
            Gemma4MtpGraphNodeKind::Add,
            vec![hidden, post_attention],
            Gemma4MtpBindingClass::Workspace,
            None,
        );
        let pre_feedforward = norm_node(
            &mut nodes,
            &weights,
            layer,
            "pre_feedforward_norm",
            Gemma4MtpNormRole::PreFeedforward,
            WeightConsumer::PreFeedforwardNorm,
            attention_residual,
        )?;
        let gate = matmul_node(
            &mut nodes,
            &weights,
            Some(layer),
            "mlp_gate",
            WeightConsumer::MlpGate,
            vec![pre_feedforward],
        )?;
        let up = matmul_node(
            &mut nodes,
            &weights,
            Some(layer),
            "mlp_up",
            WeightConsumer::MlpUp,
            vec![pre_feedforward],
        )?;
        let activated = push_node(
            &mut nodes,
            format!("layer.{layer}.gelu_tanh_mul"),
            Some(layer),
            Gemma4MtpGraphNodeKind::GeluTanhMul,
            vec![gate, up],
            Gemma4MtpBindingClass::Workspace,
            None,
        );
        let down = matmul_node(
            &mut nodes,
            &weights,
            Some(layer),
            "mlp_down",
            WeightConsumer::MlpDown,
            vec![activated],
        )?;
        let post_feedforward = norm_node(
            &mut nodes,
            &weights,
            layer,
            "post_feedforward_norm",
            Gemma4MtpNormRole::PostFeedforward,
            WeightConsumer::PostFeedforwardNorm,
            down,
        )?;
        let residual = push_node(
            &mut nodes,
            format!("layer.{layer}.feedforward_residual"),
            Some(layer),
            Gemma4MtpGraphNodeKind::Add,
            vec![attention_residual, post_feedforward],
            Gemma4MtpBindingClass::Workspace,
            None,
        );
        let scalar = required_weight(&weights, Some(layer), WeightConsumer::LayerScalar)?;
        hidden = push_node(
            &mut nodes,
            format!("layer.{layer}.layer_scalar"),
            Some(layer),
            Gemma4MtpGraphNodeKind::ScaleWeight { weight: scalar },
            vec![residual],
            Gemma4MtpBindingClass::Workspace,
            None,
        );
    }

    let final_norm = norm_node(
        &mut nodes,
        &weights,
        u32::MAX,
        "final_norm",
        Gemma4MtpNormRole::Final,
        WeightConsumer::FinalNorm,
        hidden,
    )?;
    matmul_node(
        &mut nodes,
        &weights,
        None,
        "post_projection",
        WeightConsumer::Gemma4MtpPostProjection,
        vec![final_norm],
    )?;
    let logits = matmul_node(
        &mut nodes,
        &weights,
        None,
        "logits",
        WeightConsumer::EmbeddingAndTiedOutput,
        vec![final_norm],
    )?;
    push_node(
        &mut nodes,
        "argmax",
        None,
        Gemma4MtpGraphNodeKind::Argmax,
        vec![logits],
        Gemma4MtpBindingClass::TerminalOutput,
        Some(ExecutionBoundaryKind::TerminalReadback),
    );
    validate_graph(&nodes, plan)?;
    let attention: [Gemma4MtpAttentionDescriptor; GEMMA4_MTP_LAYER_COUNT] = attention
        .try_into()
        .map_err(|_| Gemma4MtpGraphError::invalid("assistant attention layer count differs"))?;
    Ok(Gemma4MtpGraph {
        assistant_fingerprint: source.lock_fingerprint().to_owned(),
        target_fingerprint: source.target_fingerprint().to_owned(),
        weight_plan_digest: *plan.digest(),
        target_kv_length,
        absolute_query_position,
        nodes,
        attention,
    })
}

fn weight_lookup(
    plan: &WeightLoadPlan,
) -> Result<BTreeMap<WeightConsumerKey, String>, Gemma4MtpGraphError> {
    let mut weights = BTreeMap::new();
    for entry in &plan.entries {
        if entry.classification != WeightClassification::Required {
            return Err(Gemma4MtpGraphError::invalid(
                "assistant plan contains a non-required tensor",
            ));
        }
        let key = entry
            .consumer
            .ok_or_else(|| Gemma4MtpGraphError::invalid("assistant consumer is absent"))?;
        if weights.insert(key, entry.tensor_name.clone()).is_some() {
            return Err(Gemma4MtpGraphError::invalid(
                "assistant consumer is duplicated",
            ));
        }
    }
    Ok(weights)
}

fn required_weight(
    weights: &BTreeMap<WeightConsumerKey, String>,
    layer: Option<u32>,
    role: WeightConsumer,
) -> Result<String, Gemma4MtpGraphError> {
    weights
        .get(&WeightConsumerKey {
            layer: layer.map(u64::from),
            role,
        })
        .cloned()
        .ok_or_else(|| {
            Gemma4MtpGraphError::invalid(format!(
                "assistant weight is absent: layer={layer:?}, role={role:?}"
            ))
        })
}

fn push_node(
    nodes: &mut Vec<Gemma4MtpGraphNode>,
    label: impl Into<String>,
    assistant_layer: Option<u32>,
    kind: Gemma4MtpGraphNodeKind,
    predecessors: Vec<usize>,
    binding_class: Gemma4MtpBindingClass,
    boundary_after: Option<ExecutionBoundaryKind>,
) -> usize {
    let id = nodes.len();
    nodes.push(Gemma4MtpGraphNode {
        id,
        label: label.into(),
        assistant_layer,
        kind,
        predecessors,
        binding_class,
        boundary_after,
    });
    id
}

fn matmul_node(
    nodes: &mut Vec<Gemma4MtpGraphNode>,
    weights: &BTreeMap<WeightConsumerKey, String>,
    layer: Option<u32>,
    label: &str,
    consumer: WeightConsumer,
    predecessors: Vec<usize>,
) -> Result<usize, Gemma4MtpGraphError> {
    let weight = required_weight(weights, layer, consumer)?;
    Ok(push_node(
        nodes,
        match layer {
            Some(layer) => format!("layer.{layer}.{label}"),
            None => label.to_owned(),
        },
        layer,
        Gemma4MtpGraphNodeKind::Matmul { consumer, weight },
        predecessors,
        if matches!(
            consumer,
            WeightConsumer::EmbeddingAndTiedOutput | WeightConsumer::Gemma4MtpPostProjection
        ) {
            Gemma4MtpBindingClass::TerminalOutput
        } else {
            Gemma4MtpBindingClass::Workspace
        },
        None,
    ))
}

#[allow(clippy::too_many_arguments)]
fn norm_node(
    nodes: &mut Vec<Gemma4MtpGraphNode>,
    weights: &BTreeMap<WeightConsumerKey, String>,
    layer: u32,
    label: &str,
    role: Gemma4MtpNormRole,
    consumer: WeightConsumer,
    predecessor: usize,
) -> Result<usize, Gemma4MtpGraphError> {
    let layer_key = (layer != u32::MAX).then_some(layer);
    let weight = required_weight(weights, layer_key, consumer)?;
    Ok(push_node(
        nodes,
        match layer_key {
            Some(layer) => format!("layer.{layer}.{label}"),
            None => label.to_owned(),
        },
        layer_key,
        Gemma4MtpGraphNodeKind::RmsNorm {
            role,
            scale_mode: RmsNormScaleMode::Direct,
            epsilon_bits: 1.0e-6_f32.to_bits(),
            weight,
        },
        vec![predecessor],
        Gemma4MtpBindingClass::Workspace,
        None,
    ))
}

fn validate_graph(
    nodes: &[Gemma4MtpGraphNode],
    plan: &WeightLoadPlan,
) -> Result<(), Gemma4MtpGraphError> {
    let labels = nodes
        .iter()
        .map(|node| node.label.as_str())
        .collect::<BTreeSet<_>>();
    if labels.len() != nodes.len()
        || nodes.iter().enumerate().any(|(id, node)| {
            node.id != id
                || node
                    .predecessors
                    .iter()
                    .any(|predecessor| *predecessor >= id)
        })
        || nodes
            .iter()
            .filter(|node| node.boundary_after.is_some())
            .count()
            != 1
        || nodes.last().and_then(|node| node.boundary_after)
            != Some(ExecutionBoundaryKind::TerminalReadback)
    {
        return Err(Gemma4MtpGraphError::invalid(
            "assistant graph topology or terminal boundary differs",
        ));
    }
    let mut uses = BTreeMap::<&str, usize>::new();
    for node in nodes {
        if let Some(weight) = node.kind.weight_name() {
            *uses.entry(weight).or_default() += 1;
        }
    }
    if plan
        .entries
        .iter()
        .any(|entry| uses.get(entry.tensor_name.as_str()).copied() != Some(1))
        || uses.len() != plan.entries.len()
    {
        return Err(Gemma4MtpGraphError::invalid(
            "assistant graph does not consume every resident tensor exactly once",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_graph() -> Gemma4MtpGraph {
        let attention = std::array::from_fn(|layer| Gemma4MtpAttentionDescriptor {
            assistant_layer: layer as u32,
            target_layer: if layer == 3 { 47 } else { 46 },
            q_heads: 16,
            kv_heads: if layer == 3 { 1 } else { 8 },
            head_dim: if layer == 3 { 512 } else { 256 },
            target_kv_length: 17,
            sliding_window: (layer != 3).then_some(GEMMA4_MTP_SLIDING_WINDOW),
        });
        let mut nodes = Vec::new();
        for descriptor in attention {
            let head_dim = descriptor.head_dim;
            push_node(
                &mut nodes,
                format!("layer.{}.query_rotary", descriptor.assistant_layer),
                Some(descriptor.assistant_layer),
                Gemma4MtpGraphNodeKind::QueryRotary(Gemma4MtpRopeDescriptor {
                    rope_type: if descriptor.assistant_layer == 3 {
                        Gemma4MtpRopeType::Proportional
                    } else {
                        Gemma4MtpRopeType::Default
                    },
                    theta: if descriptor.assistant_layer == 3 {
                        1_000_000
                    } else {
                        10_000
                    },
                    head_dim,
                    rotary_dim: if descriptor.assistant_layer == 3 {
                        128
                    } else {
                        256
                    },
                    q_heads: 16,
                    dummy_kv_heads: descriptor.kv_heads,
                    absolute_position: 16,
                }),
                Vec::new(),
                Gemma4MtpBindingClass::Workspace,
                None,
            );
            push_node(
                &mut nodes,
                format!(
                    "layer.{}.shared_target_attention",
                    descriptor.assistant_layer
                ),
                Some(descriptor.assistant_layer),
                Gemma4MtpGraphNodeKind::SharedTargetAttention(descriptor),
                Vec::new(),
                Gemma4MtpBindingClass::TargetReadOnly,
                None,
            );
        }
        Gemma4MtpGraph {
            assistant_fingerprint: "assistant".to_owned(),
            target_fingerprint: "target".to_owned(),
            weight_plan_digest: [7; 32],
            target_kv_length: 17,
            absolute_query_position: 16,
            nodes,
            attention,
        }
    }

    #[test]
    fn shared_kv_mapping_is_q_only_and_allocates_no_assistant_state() {
        let graph = snapshot_graph();
        assert_eq!(graph.assistant_kv_allocation_bytes(), 0);
        assert_eq!(
            graph
                .attention_descriptors()
                .map(|descriptor| descriptor.target_layer),
            [46, 46, 46, 47]
        );
        for (layer, descriptor) in graph.attention_descriptors().iter().enumerate() {
            assert_eq!(descriptor.q_heads, 16);
            assert_eq!(descriptor.target_kv_length, 17);
            if layer < 3 {
                assert_eq!((descriptor.kv_heads, descriptor.head_dim), (8, 256));
                assert_eq!(descriptor.sliding_window, Some(1_024));
            } else {
                assert_eq!((descriptor.kv_heads, descriptor.head_dim), (1, 512));
                assert_eq!(descriptor.sliding_window, None);
            }
            let contract = descriptor.semantic_contract().unwrap();
            assert_eq!(contract.start_position(), 16);
            assert_eq!(contract.expected_kv_length(), 17);
        }
        assert!(graph.nodes().iter().all(|node| !matches!(
            node.kind(),
            Gemma4MtpGraphNodeKind::Matmul {
                consumer: WeightConsumer::AttentionK
                    | WeightConsumer::AttentionV
                    | WeightConsumer::AttentionKAndV,
                ..
            }
        )));
    }

    #[test]
    fn retarget_keeps_one_fixed_round_position_for_all_four_layers() {
        let graph = snapshot_graph()
            .with_target_snapshot(65, 91)
            .expect("retarget snapshot");
        assert_eq!(graph.target_kv_length(), 65);
        assert_eq!(graph.absolute_query_position(), 91);
        assert!(
            graph
                .attention_descriptors()
                .iter()
                .all(|descriptor| descriptor.target_kv_length == 65)
        );
        let rotary_positions = graph
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                Gemma4MtpGraphNodeKind::QueryRotary(descriptor) => {
                    Some(descriptor.absolute_position)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(rotary_positions, [91, 91, 91, 91]);
        assert!(graph.with_target_snapshot(0, 0).is_err());
        assert!(graph.with_target_snapshot(65, 63).is_err());
    }
}
