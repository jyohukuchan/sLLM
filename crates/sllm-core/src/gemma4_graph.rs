//! Host-only structural graph for the reviewed Gemma 4 text adapter.
//!
//! This graph records exact weights, numerical contracts, state layouts, and
//! publication boundaries. It intentionally does not allocate buffers;
//! every numerical node exposes its shared semantic kind. Provider
//! availability remains a backend preparation decision.

use crate::gemma4::{
    Gemma4LayerType, Gemma4ModelLock, is_reviewed_gemma4_identity, reviewed_layer_schedule,
};
use crate::op::{
    OpError, RmsNormScaleMode, SemanticOpKind, SplitHalfRotaryContract,
    WindowedCausalAttentionContract,
};
use crate::prepared_execution::{
    ExecutionBoundaryKind, ExecutionTransaction, ExecutionTransactionGuard, PreparedExecutionError,
    PreparedExecutionPlan, PreparedPlanNode, PreparedTransition,
};
use crate::weights::{
    WeightClassification, WeightConsumer, WeightConsumerKey, WeightLoadPlan,
    build_gemma4_weight_load_plan,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

pub const GEMMA4_HIDDEN_SIZE: u64 = 3_840;
pub const GEMMA4_INTERMEDIATE_SIZE: u64 = 15_360;
pub const GEMMA4_VOCAB_SIZE: u64 = 262_144;
pub const GEMMA4_LAYER_COUNT: usize = 48;
pub const GEMMA4_SLIDING_WINDOW: u64 = 1_024;
pub const GEMMA4_MAX_POSITION_EMBEDDINGS: u64 = 262_144;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4GraphBindingClass {
    ModelResident,
    TokenRows,
    PositionAndKv,
    TerminalOutput,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4NormRole {
    Input,
    Query,
    Key,
    ValueUnitScale,
    PostAttention,
    PreFeedforward,
    PostFeedforward,
    Final,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4RopeType {
    Default,
    Proportional,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Gemma4RopeDescriptor {
    pub rope_type: Gemma4RopeType,
    pub theta: u64,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
}

impl Gemma4RopeDescriptor {
    pub fn semantic_contract(
        self,
        start_position: u64,
        token_count: u64,
    ) -> Result<SplitHalfRotaryContract, OpError> {
        SplitHalfRotaryContract::new(
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.theta as f32,
            start_position,
            token_count,
            GEMMA4_MAX_POSITION_EMBEDDINGS as u32,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Gemma4AttentionDescriptor {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub scaling_bits: u32,
    pub sliding_window: Option<u64>,
    pub k_equals_v_before_norm: bool,
}

impl Gemma4AttentionDescriptor {
    pub fn semantic_contract(
        self,
        start_position: u64,
        query_count: u64,
    ) -> Result<WindowedCausalAttentionContract, OpError> {
        let expected_kv_length = start_position
            .checked_add(query_count)
            .ok_or(OpError::CausalAttentionLengthOverflow)?;
        WindowedCausalAttentionContract::new(
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            start_position,
            query_count,
            expected_kv_length,
            self.sliding_window,
            f32::from_bits(self.scaling_bits),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Gemma4KvDescriptor {
    pub layer: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub capacity: u64,
    pub retention_window: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4GraphNodeKind {
    Embedding {
        weight: String,
    },
    ScaleConstant {
        value_bits: u32,
    },
    ScaleWeight {
        weight: String,
    },
    RmsNorm {
        role: Gemma4NormRole,
        scale_mode: RmsNormScaleMode,
        epsilon_bits: u32,
        weight: Option<String>,
    },
    Matmul {
        consumer: WeightConsumer,
        weight: String,
    },
    Rotary(Gemma4RopeDescriptor),
    CausalAttention(Gemma4AttentionDescriptor),
    GeluTanhMul,
    Add,
    LogitSoftcap {
        cap_bits: u32,
    },
    Argmax,
}

impl Gemma4GraphNodeKind {
    pub const fn semantic_kind(&self) -> Option<SemanticOpKind> {
        match self {
            Self::Embedding { .. } => Some(SemanticOpKind::Embedding),
            Self::ScaleConstant { .. } | Self::ScaleWeight { .. } => {
                Some(SemanticOpKind::ScalarMul)
            }
            Self::RmsNorm { .. } => Some(SemanticOpKind::RmsNorm),
            Self::Matmul { .. } => Some(SemanticOpKind::Matmul),
            Self::Rotary(_) => Some(SemanticOpKind::Rotary),
            Self::CausalAttention(_) => Some(SemanticOpKind::CausalAttention),
            Self::GeluTanhMul => Some(SemanticOpKind::GeluTanhMul),
            Self::Add => Some(SemanticOpKind::Add),
            Self::LogitSoftcap { .. } => Some(SemanticOpKind::TanhSoftcap),
            Self::Argmax => Some(SemanticOpKind::Argmax),
        }
    }

    fn weight_name(&self) -> Option<&str> {
        match self {
            Self::Embedding { weight }
            | Self::ScaleWeight { weight }
            | Self::Matmul { weight, .. } => Some(weight),
            Self::RmsNorm {
                weight: Some(weight),
                ..
            } => Some(weight),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4GraphNode {
    id: usize,
    label: String,
    layer: Option<u32>,
    kind: Gemma4GraphNodeKind,
    predecessors: Vec<usize>,
    binding_class: Gemma4GraphBindingClass,
    boundary_after: Option<ExecutionBoundaryKind>,
}

impl Gemma4GraphNode {
    pub const fn id(&self) -> usize {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn layer(&self) -> Option<u32> {
        self.layer
    }

    pub const fn kind(&self) -> &Gemma4GraphNodeKind {
        &self.kind
    }

    pub fn predecessors(&self) -> &[usize] {
        &self.predecessors
    }

    pub const fn binding_class(&self) -> Gemma4GraphBindingClass {
        self.binding_class
    }

    pub const fn boundary_after(&self) -> Option<ExecutionBoundaryKind> {
        self.boundary_after
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4Graph {
    lock_fingerprint: String,
    weight_plan_digest: [u8; 32],
    token_count: u64,
    start_position: u64,
    expected_length: u64,
    state_capacity: u64,
    nodes: Vec<Gemma4GraphNode>,
    kv_descriptors: Vec<Gemma4KvDescriptor>,
}

impl Gemma4Graph {
    pub fn lock_fingerprint(&self) -> &str {
        &self.lock_fingerprint
    }

    pub const fn weight_plan_digest(&self) -> &[u8; 32] {
        &self.weight_plan_digest
    }

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

    pub fn nodes(&self) -> &[Gemma4GraphNode] {
        &self.nodes
    }

    /// Lowers the immutable Gemma graph into the model-neutral Phase 13
    /// execution-plan envelope. Numerical descriptor construction and state
    /// ownership remain adapter responsibilities.
    pub fn prepared_execution_plan(
        &self,
    ) -> Result<PreparedExecutionPlan<Gemma4GraphNode>, PreparedExecutionError> {
        PreparedExecutionPlan::new(
            self.nodes
                .iter()
                .cloned()
                .map(|node| {
                    let boundary = node.boundary_after();
                    PreparedPlanNode::new(node, boundary)
                })
                .collect(),
        )
    }

    /// Produces the request-local identity consumed by the shared prepared
    /// cache and transaction controller.
    pub fn prepared_transition(
        &self,
        binding_generation: u64,
        state_generation: u64,
    ) -> Result<PreparedTransition, PreparedExecutionError> {
        let transition = PreparedTransition::new(
            self.token_count,
            self.start_position,
            binding_generation,
            state_generation,
        )?;
        if transition.expected_length() != self.expected_length {
            return Err(PreparedExecutionError::InvalidTransition(
                "Gemma graph and prepared transition length differ".to_owned(),
            ));
        }
        Ok(transition)
    }

    pub fn kv_descriptors(&self) -> &[Gemma4KvDescriptor] {
        &self.kv_descriptors
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4RequestStateSnapshot {
    pub committed_length: u64,
    pub binding_generation: u64,
    pub state_generation: u64,
    pub poisoned: bool,
}

#[derive(Clone, Copy, Debug)]
struct Gemma4PublishedState {
    committed_length: u64,
    binding_generation: u64,
    state_generation: u64,
}

/// Request-local publication owner for Gemma KV buffers.
///
/// Kernels may write the uncommitted tail on their ordered queue, but no
/// subsequent request can observe that tail until the shared execution
/// boundaries have succeeded and this owner advances `committed_length`.
#[derive(Clone)]
pub struct Gemma4RequestState {
    capacity: u64,
    published: Arc<Mutex<Gemma4PublishedState>>,
    transaction: ExecutionTransaction,
}

impl Gemma4RequestState {
    pub fn new(capacity: u64) -> Result<Self, PreparedExecutionError> {
        if capacity == 0 || capacity > GEMMA4_MAX_POSITION_EMBEDDINGS {
            return Err(PreparedExecutionError::InvalidTransition(
                "Gemma request-state capacity is outside the reviewed model range".to_owned(),
            ));
        }
        Ok(Self {
            capacity,
            published: Arc::new(Mutex::new(Gemma4PublishedState {
                committed_length: 0,
                binding_generation: 0,
                state_generation: 0,
            })),
            transaction: ExecutionTransaction::new(),
        })
    }

    pub fn begin(
        &self,
        token_count: u64,
        start_position: u64,
        binding_generation: u64,
    ) -> Result<Gemma4RequestTransition, PreparedExecutionError> {
        let published = self
            .published
            .lock()
            .map_err(|_| PreparedExecutionError::Poisoned)?;
        if start_position != published.committed_length {
            return Err(PreparedExecutionError::InvalidTransition(
                "Gemma transition does not start at the published KV length".to_owned(),
            ));
        }
        let transition = PreparedTransition::new(
            token_count,
            start_position,
            binding_generation,
            published.state_generation,
        )?;
        if transition.expected_length() > self.capacity {
            return Err(PreparedExecutionError::InvalidTransition(
                "Gemma transition exceeds request-state capacity".to_owned(),
            ));
        }
        let guard = self.transaction.begin()?;
        drop(published);
        Ok(Gemma4RequestTransition {
            published: Arc::clone(&self.published),
            guard,
            transition,
            state_boundary_complete: false,
            terminal_boundary_complete: false,
        })
    }

    pub fn cancel(&self) {
        self.transaction.cancel();
    }

    pub fn snapshot(&self) -> Result<Gemma4RequestStateSnapshot, PreparedExecutionError> {
        let published = self
            .published
            .lock()
            .map_err(|_| PreparedExecutionError::Poisoned)?;
        Ok(Gemma4RequestStateSnapshot {
            committed_length: published.committed_length,
            binding_generation: published.binding_generation,
            state_generation: published.state_generation,
            poisoned: self.transaction.is_poisoned(),
        })
    }
}

pub struct Gemma4RequestTransition {
    published: Arc<Mutex<Gemma4PublishedState>>,
    guard: ExecutionTransactionGuard,
    transition: PreparedTransition,
    state_boundary_complete: bool,
    terminal_boundary_complete: bool,
}

impl Gemma4RequestTransition {
    pub const fn transition(&self) -> PreparedTransition {
        self.transition
    }

    pub fn complete_boundary(
        &mut self,
        boundary: ExecutionBoundaryKind,
    ) -> Result<(), PreparedExecutionError> {
        match boundary {
            ExecutionBoundaryKind::StatePublication if !self.state_boundary_complete => {
                self.state_boundary_complete = true;
                Ok(())
            }
            ExecutionBoundaryKind::TerminalReadback
                if self.state_boundary_complete && !self.terminal_boundary_complete =>
            {
                self.terminal_boundary_complete = true;
                Ok(())
            }
            _ => Err(PreparedExecutionError::InvalidTransition(
                "Gemma execution boundary is duplicated, missing, or out of order".to_owned(),
            )),
        }
    }

    pub fn commit(mut self) -> Result<Gemma4RequestStateSnapshot, PreparedExecutionError> {
        if !self.state_boundary_complete || !self.terminal_boundary_complete {
            return Err(PreparedExecutionError::InvalidTransition(
                "Gemma transition cannot publish before both declared boundaries".to_owned(),
            ));
        }
        let mut published = self
            .published
            .lock()
            .map_err(|_| PreparedExecutionError::Poisoned)?;
        if published.committed_length != self.transition.start_position()
            || published.state_generation
                != self
                    .transition
                    .dynamic_identity()
                    .state_generation()
                    .unwrap_or(u64::MAX)
        {
            return Err(PreparedExecutionError::InvalidTransition(
                "Gemma published state changed during an active transition".to_owned(),
            ));
        }
        let next_generation = published.state_generation.checked_add(1).ok_or_else(|| {
            PreparedExecutionError::InvalidTransition(
                "Gemma state generation overflowed u64".to_owned(),
            )
        })?;
        self.guard.commit()?;
        published.committed_length = self.transition.expected_length();
        published.binding_generation = self.transition.dynamic_identity().binding_generation();
        published.state_generation = next_generation;
        Ok(Gemma4RequestStateSnapshot {
            committed_length: published.committed_length,
            binding_generation: published.binding_generation,
            state_generation: published.state_generation,
            poisoned: false,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4GraphError {
    InvalidModel,
    InvalidWeightPlan,
    ZeroTokenCount,
    ZeroStateCapacity,
    PositionOverflow,
    LengthOutOfBounds,
    MissingWeight(WeightConsumerKey),
    DuplicateWeight(WeightConsumerKey),
    InvalidTopology,
}

impl fmt::Display for Gemma4GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModel => formatter.write_str("invalid reviewed Gemma 4 model"),
            Self::InvalidWeightPlan => formatter.write_str("invalid reviewed Gemma 4 weight plan"),
            Self::ZeroTokenCount => formatter.write_str("Gemma 4 token count must be non-zero"),
            Self::ZeroStateCapacity => {
                formatter.write_str("Gemma 4 state capacity must be non-zero")
            }
            Self::PositionOverflow => formatter.write_str("Gemma 4 position range overflowed"),
            Self::LengthOutOfBounds => {
                formatter.write_str("Gemma 4 position range exceeds state/model capacity")
            }
            Self::MissingWeight(key) => write!(formatter, "missing Gemma 4 weight: {key:?}"),
            Self::DuplicateWeight(key) => write!(formatter, "duplicate Gemma 4 weight: {key:?}"),
            Self::InvalidTopology => formatter.write_str("invalid Gemma 4 graph topology"),
        }
    }
}

impl std::error::Error for Gemma4GraphError {}

pub fn build_gemma4_graph(
    lock: &Gemma4ModelLock,
    plan: &WeightLoadPlan,
    token_count: u64,
    start_position: u64,
    state_capacity: u64,
) -> Result<Gemma4Graph, Gemma4GraphError> {
    if !is_reviewed_gemma4_identity(lock) {
        return Err(Gemma4GraphError::InvalidModel);
    }
    if token_count == 0 {
        return Err(Gemma4GraphError::ZeroTokenCount);
    }
    if state_capacity == 0 {
        return Err(Gemma4GraphError::ZeroStateCapacity);
    }
    let expected_length = start_position
        .checked_add(token_count)
        .ok_or(Gemma4GraphError::PositionOverflow)?;
    if expected_length > state_capacity || state_capacity > GEMMA4_MAX_POSITION_EMBEDDINGS {
        return Err(Gemma4GraphError::LengthOutOfBounds);
    }
    let expected_catalog = crate::gemma4::expected_gemma4_tensor_catalog()
        .map_err(|_| Gemma4GraphError::InvalidModel)?;
    let expected_plan = build_gemma4_weight_load_plan(lock, expected_catalog.values())
        .map_err(|_| Gemma4GraphError::InvalidModel)?;
    if plan != &expected_plan {
        return Err(Gemma4GraphError::InvalidWeightPlan);
    }

    let weights = weight_lookup(plan)?;
    let mut nodes = Vec::new();
    let embedding_weight = required_weight(&weights, None, WeightConsumer::EmbeddingAndTiedOutput)?;
    let embedding = push_node(
        &mut nodes,
        "embedding",
        None,
        Gemma4GraphNodeKind::Embedding {
            weight: embedding_weight.clone(),
        },
        vec![],
        Gemma4GraphBindingClass::TokenRows,
        None,
    );
    let mut hidden = push_node(
        &mut nodes,
        "embedding_scale",
        None,
        Gemma4GraphNodeKind::ScaleConstant {
            value_bits: (GEMMA4_HIDDEN_SIZE as f32).sqrt().to_bits(),
        },
        vec![embedding],
        Gemma4GraphBindingClass::TokenRows,
        None,
    );
    let mut kv_descriptors = Vec::with_capacity(GEMMA4_LAYER_COUNT);
    let schedule = reviewed_layer_schedule();
    for (layer_index, layer_type) in schedule.into_iter().enumerate() {
        let layer = u32::try_from(layer_index).expect("reviewed layer count fits u32");
        let input_norm = norm_node(
            &mut nodes,
            &weights,
            layer,
            "input_norm",
            Gemma4NormRole::Input,
            WeightConsumer::InputNorm,
            hidden,
        )?;
        let q = matmul_node(
            &mut nodes,
            &weights,
            layer,
            "q_proj",
            WeightConsumer::AttentionQ,
            input_norm,
        )?;
        let k_role = match layer_type {
            Gemma4LayerType::SlidingAttention => WeightConsumer::AttentionK,
            Gemma4LayerType::FullAttention => WeightConsumer::AttentionKAndV,
        };
        let k = matmul_node(&mut nodes, &weights, layer, "k_proj", k_role, input_norm)?;
        let v = match layer_type {
            Gemma4LayerType::SlidingAttention => matmul_node(
                &mut nodes,
                &weights,
                layer,
                "v_proj",
                WeightConsumer::AttentionV,
                input_norm,
            )?,
            Gemma4LayerType::FullAttention => k,
        };
        let q_norm = norm_node(
            &mut nodes,
            &weights,
            layer,
            "q_norm",
            Gemma4NormRole::Query,
            WeightConsumer::AttentionQNorm,
            q,
        )?;
        let k_norm = norm_node(
            &mut nodes,
            &weights,
            layer,
            "k_norm",
            Gemma4NormRole::Key,
            WeightConsumer::AttentionKNorm,
            k,
        )?;
        let v_norm = push_node(
            &mut nodes,
            format!("layer.{layer}.v_norm"),
            Some(layer),
            Gemma4GraphNodeKind::RmsNorm {
                role: Gemma4NormRole::ValueUnitScale,
                scale_mode: RmsNormScaleMode::Direct,
                epsilon_bits: 1.0e-6_f32.to_bits(),
                weight: None,
            },
            vec![v],
            Gemma4GraphBindingClass::TokenRows,
            None,
        );
        let (head_dim, kv_heads, rope_type, theta, rotary_dim, window, k_equals_v) =
            match layer_type {
                Gemma4LayerType::SlidingAttention => (
                    256_u32,
                    8_u32,
                    Gemma4RopeType::Default,
                    10_000_u64,
                    256_u32,
                    Some(GEMMA4_SLIDING_WINDOW),
                    false,
                ),
                Gemma4LayerType::FullAttention => (
                    512_u32,
                    1_u32,
                    Gemma4RopeType::Proportional,
                    1_000_000_u64,
                    128_u32,
                    None,
                    true,
                ),
            };
        let rotary = push_node(
            &mut nodes,
            format!("layer.{layer}.rotary"),
            Some(layer),
            Gemma4GraphNodeKind::Rotary(Gemma4RopeDescriptor {
                rope_type,
                theta,
                head_dim,
                rotary_dim,
                q_heads: 16,
                kv_heads,
            }),
            vec![q_norm, k_norm],
            Gemma4GraphBindingClass::PositionAndKv,
            None,
        );
        let attention = push_node(
            &mut nodes,
            format!("layer.{layer}.attention"),
            Some(layer),
            Gemma4GraphNodeKind::CausalAttention(Gemma4AttentionDescriptor {
                q_heads: 16,
                kv_heads,
                head_dim,
                scaling_bits: 1.0_f32.to_bits(),
                sliding_window: window,
                k_equals_v_before_norm: k_equals_v,
            }),
            vec![rotary, v_norm],
            Gemma4GraphBindingClass::PositionAndKv,
            None,
        );
        kv_descriptors.push(Gemma4KvDescriptor {
            layer,
            heads: kv_heads,
            head_dim,
            capacity: state_capacity,
            retention_window: window,
        });
        let attention_output = matmul_node(
            &mut nodes,
            &weights,
            layer,
            "o_proj",
            WeightConsumer::AttentionO,
            attention,
        )?;
        let post_attention = norm_node(
            &mut nodes,
            &weights,
            layer,
            "post_attention_norm",
            Gemma4NormRole::PostAttention,
            WeightConsumer::PostAttentionNorm,
            attention_output,
        )?;
        let attention_residual = push_node(
            &mut nodes,
            format!("layer.{layer}.attention_residual"),
            Some(layer),
            Gemma4GraphNodeKind::Add,
            vec![hidden, post_attention],
            Gemma4GraphBindingClass::TokenRows,
            None,
        );
        let pre_feedforward = norm_node(
            &mut nodes,
            &weights,
            layer,
            "pre_feedforward_norm",
            Gemma4NormRole::PreFeedforward,
            WeightConsumer::PreFeedforwardNorm,
            attention_residual,
        )?;
        let gate = matmul_node(
            &mut nodes,
            &weights,
            layer,
            "mlp_gate",
            WeightConsumer::MlpGate,
            pre_feedforward,
        )?;
        let up = matmul_node(
            &mut nodes,
            &weights,
            layer,
            "mlp_up",
            WeightConsumer::MlpUp,
            pre_feedforward,
        )?;
        let activated = push_node(
            &mut nodes,
            format!("layer.{layer}.gelu_tanh_mul"),
            Some(layer),
            Gemma4GraphNodeKind::GeluTanhMul,
            vec![gate, up],
            Gemma4GraphBindingClass::TokenRows,
            None,
        );
        let down = matmul_node(
            &mut nodes,
            &weights,
            layer,
            "mlp_down",
            WeightConsumer::MlpDown,
            activated,
        )?;
        let post_feedforward = norm_node(
            &mut nodes,
            &weights,
            layer,
            "post_feedforward_norm",
            Gemma4NormRole::PostFeedforward,
            WeightConsumer::PostFeedforwardNorm,
            down,
        )?;
        let feedforward_residual = push_node(
            &mut nodes,
            format!("layer.{layer}.feedforward_residual"),
            Some(layer),
            Gemma4GraphNodeKind::Add,
            vec![attention_residual, post_feedforward],
            Gemma4GraphBindingClass::TokenRows,
            None,
        );
        let scalar = required_weight(&weights, Some(layer), WeightConsumer::LayerScalar)?;
        hidden = push_node(
            &mut nodes,
            format!("layer.{layer}.layer_scalar"),
            Some(layer),
            Gemma4GraphNodeKind::ScaleWeight { weight: scalar },
            vec![feedforward_residual],
            Gemma4GraphBindingClass::TokenRows,
            (layer_index + 1 == GEMMA4_LAYER_COUNT)
                .then_some(ExecutionBoundaryKind::StatePublication),
        );
    }
    let final_norm = norm_node(
        &mut nodes,
        &weights,
        u32::MAX,
        "final_norm",
        Gemma4NormRole::Final,
        WeightConsumer::FinalNorm,
        hidden,
    )?;
    let logits = push_node(
        &mut nodes,
        "logits",
        None,
        Gemma4GraphNodeKind::Matmul {
            consumer: WeightConsumer::EmbeddingAndTiedOutput,
            weight: embedding_weight,
        },
        vec![final_norm],
        Gemma4GraphBindingClass::TerminalOutput,
        None,
    );
    let softcapped = push_node(
        &mut nodes,
        "logit_softcap",
        None,
        Gemma4GraphNodeKind::LogitSoftcap {
            cap_bits: 30.0_f32.to_bits(),
        },
        vec![logits],
        Gemma4GraphBindingClass::TerminalOutput,
        None,
    );
    push_node(
        &mut nodes,
        "argmax",
        None,
        Gemma4GraphNodeKind::Argmax,
        vec![softcapped],
        Gemma4GraphBindingClass::TerminalOutput,
        Some(ExecutionBoundaryKind::TerminalReadback),
    );
    validate_graph(&nodes, plan)?;
    Ok(Gemma4Graph {
        lock_fingerprint: lock.fingerprint().to_owned(),
        weight_plan_digest: *plan.digest(),
        token_count,
        start_position,
        expected_length,
        state_capacity,
        nodes,
        kv_descriptors,
    })
}

fn weight_lookup(
    plan: &WeightLoadPlan,
) -> Result<BTreeMap<WeightConsumerKey, String>, Gemma4GraphError> {
    let mut weights = BTreeMap::new();
    for entry in &plan.entries {
        if entry.classification == WeightClassification::KnownUnconsumed {
            continue;
        }
        let consumer = entry.consumer.ok_or(Gemma4GraphError::InvalidWeightPlan)?;
        if weights
            .insert(consumer, entry.tensor_name.clone())
            .is_some()
        {
            return Err(Gemma4GraphError::DuplicateWeight(consumer));
        }
    }
    Ok(weights)
}

fn required_weight(
    weights: &BTreeMap<WeightConsumerKey, String>,
    layer: Option<u32>,
    role: WeightConsumer,
) -> Result<String, Gemma4GraphError> {
    let key = WeightConsumerKey {
        layer: layer.map(u64::from),
        role,
    };
    weights
        .get(&key)
        .cloned()
        .ok_or(Gemma4GraphError::MissingWeight(key))
}

fn push_node(
    nodes: &mut Vec<Gemma4GraphNode>,
    label: impl Into<String>,
    layer: Option<u32>,
    kind: Gemma4GraphNodeKind,
    predecessors: Vec<usize>,
    binding_class: Gemma4GraphBindingClass,
    boundary_after: Option<ExecutionBoundaryKind>,
) -> usize {
    let id = nodes.len();
    nodes.push(Gemma4GraphNode {
        id,
        label: label.into(),
        layer,
        kind,
        predecessors,
        binding_class,
        boundary_after,
    });
    id
}

fn matmul_node(
    nodes: &mut Vec<Gemma4GraphNode>,
    weights: &BTreeMap<WeightConsumerKey, String>,
    layer: u32,
    label: &str,
    consumer: WeightConsumer,
    predecessor: usize,
) -> Result<usize, Gemma4GraphError> {
    let weight = required_weight(weights, Some(layer), consumer)?;
    Ok(push_node(
        nodes,
        format!("layer.{layer}.{label}"),
        Some(layer),
        Gemma4GraphNodeKind::Matmul { consumer, weight },
        vec![predecessor],
        Gemma4GraphBindingClass::TokenRows,
        None,
    ))
}

#[allow(clippy::too_many_arguments)]
fn norm_node(
    nodes: &mut Vec<Gemma4GraphNode>,
    weights: &BTreeMap<WeightConsumerKey, String>,
    layer: u32,
    label: &str,
    role: Gemma4NormRole,
    consumer: WeightConsumer,
    predecessor: usize,
) -> Result<usize, Gemma4GraphError> {
    let weight = if layer == u32::MAX {
        required_weight(weights, None, consumer)?
    } else {
        required_weight(weights, Some(layer), consumer)?
    };
    Ok(push_node(
        nodes,
        if layer == u32::MAX {
            label.to_owned()
        } else {
            format!("layer.{layer}.{label}")
        },
        (layer != u32::MAX).then_some(layer),
        Gemma4GraphNodeKind::RmsNorm {
            role,
            scale_mode: RmsNormScaleMode::Direct,
            epsilon_bits: 1.0e-6_f32.to_bits(),
            weight: Some(weight),
        },
        vec![predecessor],
        Gemma4GraphBindingClass::TokenRows,
        None,
    ))
}

fn validate_graph(
    nodes: &[Gemma4GraphNode],
    plan: &WeightLoadPlan,
) -> Result<(), Gemma4GraphError> {
    let labels: BTreeSet<_> = nodes.iter().map(|node| node.label.as_str()).collect();
    if labels.len() != nodes.len()
        || nodes.iter().enumerate().any(|(id, node)| {
            node.id != id
                || node
                    .predecessors
                    .iter()
                    .any(|predecessor| *predecessor >= id)
        })
    {
        return Err(Gemma4GraphError::InvalidTopology);
    }
    let mut uses = BTreeMap::<&str, usize>::new();
    for node in nodes {
        if let Some(weight) = node.kind.weight_name() {
            *uses.entry(weight).or_default() += 1;
        }
    }
    for entry in &plan.entries {
        if entry.classification == WeightClassification::KnownUnconsumed {
            if uses.contains_key(entry.tensor_name.as_str()) {
                return Err(Gemma4GraphError::InvalidTopology);
            }
            continue;
        }
        let expected_uses = usize::from(
            entry.consumer.map(|consumer| consumer.role)
                == Some(WeightConsumer::EmbeddingAndTiedOutput),
        ) + 1;
        if uses.get(entry.tensor_name.as_str()).copied() != Some(expected_uses) {
            return Err(Gemma4GraphError::InvalidTopology);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Gemma4ModelLock, WeightLoadPlan) {
        let lock = crate::parse_gemma4_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-bf16.json"
        ))
        .unwrap();
        let catalog = crate::gemma4::expected_gemma4_tensor_catalog().unwrap();
        let plan = build_gemma4_weight_load_plan(&lock, catalog.values()).unwrap();
        (lock, plan)
    }

    #[test]
    fn graph_accepts_the_exact_instruction_tuned_identity() {
        let lock = crate::parse_gemma4_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-it-bf16.json"
        ))
        .unwrap();
        let catalog = crate::gemma4::expected_gemma4_tensor_catalog().unwrap();
        let plan = build_gemma4_weight_load_plan(&lock, catalog.values()).unwrap();
        let graph = build_gemma4_graph(&lock, &plan, 3, 17, 257).unwrap();
        assert_eq!(
            graph.lock_fingerprint(),
            crate::gemma4::GEMMA4_12B_IT_FINGERPRINT
        );
    }

    #[test]
    fn graph_records_dual_attention_and_shared_execution_boundaries() {
        let (lock, plan) = fixture();
        let graph = build_gemma4_graph(&lock, &plan, 3, 17, 257).unwrap();
        assert_eq!(graph.expected_length(), 20);
        assert_eq!(graph.kv_descriptors().len(), 48);
        assert_eq!(graph.kv_descriptors()[0].retention_window, Some(1_024));
        assert_eq!(graph.kv_descriptors()[5].heads, 1);
        assert_eq!(graph.kv_descriptors()[5].head_dim, 512);
        assert_eq!(graph.kv_descriptors()[5].retention_window, None);
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| node.boundary_after().is_some())
                .count(),
            2
        );
        let full = graph
            .nodes()
            .iter()
            .find(|node| node.label() == "layer.5.attention")
            .unwrap();
        assert!(matches!(
            full.kind(),
            Gemma4GraphNodeKind::CausalAttention(Gemma4AttentionDescriptor {
                head_dim: 512,
                sliding_window: None,
                k_equals_v_before_norm: true,
                ..
            })
        ));
        let full_attention = match full.kind() {
            Gemma4GraphNodeKind::CausalAttention(descriptor) => *descriptor,
            _ => unreachable!(),
        };
        let full_contract = full_attention.semantic_contract(17, 3).unwrap();
        assert_eq!(full_contract.kv_heads(), 1);
        assert_eq!(full_contract.head_dim(), 512);
        assert_eq!(full_contract.sliding_window(), None);
        assert_eq!(full_contract.scaling(), 1.0);
        let rotary = graph
            .nodes()
            .iter()
            .find(|node| node.label() == "layer.5.rotary")
            .unwrap();
        assert!(matches!(
            rotary.kind(),
            Gemma4GraphNodeKind::Rotary(Gemma4RopeDescriptor {
                rope_type: Gemma4RopeType::Proportional,
                rotary_dim: 128,
                ..
            })
        ));
        let full_rope = match rotary.kind() {
            Gemma4GraphNodeKind::Rotary(descriptor) => *descriptor,
            _ => unreachable!(),
        };
        let full_rope_contract = full_rope.semantic_contract(17, 3).unwrap();
        assert_eq!(full_rope_contract.head_dim(), 512);
        assert_eq!(full_rope_contract.rotary_dim(), 128);
        assert_eq!(full_rope_contract.theta(), 1_000_000.0);

        for (label, expected) in [
            ("embedding_scale", SemanticOpKind::ScalarMul),
            ("layer.0.rotary", SemanticOpKind::Rotary),
            ("layer.0.attention", SemanticOpKind::CausalAttention),
            ("layer.0.gelu_tanh_mul", SemanticOpKind::GeluTanhMul),
            ("layer.0.layer_scalar", SemanticOpKind::ScalarMul),
            ("logit_softcap", SemanticOpKind::TanhSoftcap),
        ] {
            let node = graph
                .nodes()
                .iter()
                .find(|node| node.label() == label)
                .unwrap();
            assert_eq!(node.kind().semantic_kind(), Some(expected));
        }
        assert!(
            graph
                .nodes()
                .iter()
                .all(|node| node.kind().semantic_kind().is_some())
        );

        let execution_plan = graph.prepared_execution_plan().unwrap();
        assert_eq!(execution_plan.nodes().len(), graph.nodes().len());
        assert!(
            execution_plan
                .nodes()
                .iter()
                .zip(graph.nodes())
                .all(|(prepared, graph_node)| prepared.operation() == graph_node
                    && prepared.boundary_after() == graph_node.boundary_after())
        );
        assert_eq!(
            execution_plan
                .nodes()
                .iter()
                .filter_map(PreparedPlanNode::boundary_after)
                .collect::<Vec<_>>(),
            [
                ExecutionBoundaryKind::StatePublication,
                ExecutionBoundaryKind::TerminalReadback,
            ]
        );
        let transition = graph.prepared_transition(7, 11).unwrap();
        assert_eq!(transition.token_count(), graph.token_count());
        assert_eq!(transition.start_position(), graph.start_position());
        assert_eq!(transition.expected_length(), graph.expected_length());
        assert_eq!(transition.dynamic_identity().binding_generation(), 7);
        assert_eq!(transition.dynamic_identity().state_generation(), Some(11));
    }

    #[test]
    fn graph_rejects_zero_overflow_bounds_and_plan_mutation() {
        let (lock, mut plan) = fixture();
        assert_eq!(
            build_gemma4_graph(&lock, &plan, 0, 0, 1),
            Err(Gemma4GraphError::ZeroTokenCount)
        );
        assert_eq!(
            build_gemma4_graph(&lock, &plan, 1, 0, 0),
            Err(Gemma4GraphError::ZeroStateCapacity)
        );
        assert_eq!(
            build_gemma4_graph(&lock, &plan, 2, u64::MAX, 257),
            Err(Gemma4GraphError::PositionOverflow)
        );
        assert_eq!(
            build_gemma4_graph(&lock, &plan, 2, 256, 257),
            Err(Gemma4GraphError::LengthOutOfBounds)
        );
        plan.entries[0].tensor_name.push_str(".mutated");
        assert_eq!(
            build_gemma4_graph(&lock, &plan, 1, 0, 1),
            Err(Gemma4GraphError::InvalidWeightPlan)
        );
    }

    #[test]
    fn request_state_publishes_only_after_both_shared_boundaries() {
        let state = Gemma4RequestState::new(257).unwrap();
        let mut first = state.begin(3, 0, 7).unwrap();
        assert_eq!(first.transition().expected_length(), 3);
        first
            .complete_boundary(ExecutionBoundaryKind::StatePublication)
            .unwrap();
        first
            .complete_boundary(ExecutionBoundaryKind::TerminalReadback)
            .unwrap();
        assert_eq!(
            first.commit().unwrap(),
            Gemma4RequestStateSnapshot {
                committed_length: 3,
                binding_generation: 7,
                state_generation: 1,
                poisoned: false,
            }
        );

        let mut second = state.begin(17, 3, 8).unwrap();
        second
            .complete_boundary(ExecutionBoundaryKind::StatePublication)
            .unwrap();
        second
            .complete_boundary(ExecutionBoundaryKind::TerminalReadback)
            .unwrap();
        assert_eq!(second.commit().unwrap().committed_length, 20);
        assert_eq!(state.snapshot().unwrap().state_generation, 2);
    }

    #[test]
    fn request_state_rejects_stale_binding_and_capacity_without_poisoning() {
        let state = Gemma4RequestState::new(17).unwrap();
        assert!(matches!(
            state.begin(3, 1, 7),
            Err(PreparedExecutionError::InvalidTransition(_))
        ));
        assert!(matches!(
            state.begin(18, 0, 7),
            Err(PreparedExecutionError::InvalidTransition(_))
        ));
        assert!(!state.snapshot().unwrap().poisoned);
        assert!(state.begin(3, 0, 7).is_ok());
    }

    #[test]
    fn request_state_drop_failure_and_cancel_do_not_publish_partial_kv() {
        let dropped = Gemma4RequestState::new(17).unwrap();
        {
            let mut transition = dropped.begin(3, 0, 7).unwrap();
            transition
                .complete_boundary(ExecutionBoundaryKind::StatePublication)
                .unwrap();
        }
        assert_eq!(dropped.snapshot().unwrap().committed_length, 0);
        assert!(dropped.snapshot().unwrap().poisoned);
        assert!(matches!(
            dropped.begin(3, 0, 8),
            Err(PreparedExecutionError::Poisoned)
        ));

        let canceled = Gemma4RequestState::new(17).unwrap();
        let mut transition = canceled.begin(3, 0, 7).unwrap();
        transition
            .complete_boundary(ExecutionBoundaryKind::StatePublication)
            .unwrap();
        transition
            .complete_boundary(ExecutionBoundaryKind::TerminalReadback)
            .unwrap();
        canceled.cancel();
        assert!(matches!(
            transition.commit(),
            Err(PreparedExecutionError::Poisoned)
        ));
        assert_eq!(canceled.snapshot().unwrap().committed_length, 0);
        assert!(canceled.snapshot().unwrap().poisoned);
    }

    #[test]
    fn request_state_enforces_boundary_order_and_single_inflight_transition() {
        let state = Gemma4RequestState::new(17).unwrap();
        let mut transition = state.begin(3, 0, 7).unwrap();
        assert!(matches!(
            state.begin(3, 0, 8),
            Err(PreparedExecutionError::Busy)
        ));
        assert!(matches!(
            transition.complete_boundary(ExecutionBoundaryKind::TerminalReadback),
            Err(PreparedExecutionError::InvalidTransition(_))
        ));
        assert!(matches!(
            transition.commit(),
            Err(PreparedExecutionError::InvalidTransition(_))
        ));
        assert_eq!(state.snapshot().unwrap().committed_length, 0);
        assert!(state.snapshot().unwrap().poisoned);
    }
}
