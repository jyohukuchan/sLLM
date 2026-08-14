//! Host-only structural execution graph for the fixed Qwen3.5-4B text model.
//!
//! This module contains metadata only.  It does not allocate model or state
//! buffers, read tensor payloads, submit work, or provide a CPU fallback.  The
//! graph is the handoff between the validated model/load-plan contracts and a
//! later execution/state owner.

use crate::final_output::{QWEN35_EMBEDDING_TENSOR, QWEN35_VOCAB_SIZE};
use crate::kv_state::KvStateDescriptor;
use crate::linear_attention::LinearAttentionStateDescriptor;
use crate::model::{
    ClassificationStatus, LayerType, ModelLock, Qwen35ReviewedSpec, RopeType, TensorDType,
    TensorDescriptor, reviewed_qwen35_spec,
};
use crate::op::{OpError, RmsNormScaleMode, SemanticOpDescriptor, SemanticOpKind};
use crate::weights::{
    WeightClassification, WeightConsumer, WeightConsumerKey, WeightLoadPlan, build_weight_load_plan,
};
use crate::{DType, Encoding, TensorError, TensorView};
use crate::{Fp8ResidentRepresentation, Fp8ScaleGranularity, VerifiedFp8Sidecar};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const QWEN35_MAX_POSITION_EMBEDDINGS: u64 = 262_144;
pub const QWEN35_LAYER_COUNT: usize = 32;
pub const QWEN35_REQUIRED_WEIGHT_COUNT: usize = 426;
pub const QWEN35_PLAN_ENTRY_COUNT: usize = 738;

#[derive(Clone, Copy, Debug)]
struct QwenGraphDimensions {
    hidden: u64,
    intermediate: u64,
    vocab: u64,
    q_heads: u64,
    kv_heads: u64,
    head_dim: u64,
    full_q_width: u64,
    full_kv_width: u64,
    full_output_width: u64,
    linear_qk_heads: u64,
    linear_value_heads: u64,
    linear_head_dim: u64,
    linear_qkv_width: u64,
    linear_output_width: u64,
    linear_conv_kernel: u64,
    tied_embeddings: bool,
}

impl QwenGraphDimensions {
    fn from_spec(spec: Qwen35ReviewedSpec) -> Result<Self, QwenGraphError> {
        let mul = |a: u64, b: u64, field| a.checked_mul(b).ok_or(QwenGraphError::Overflow(field));
        let full_output_width = mul(spec.attention_heads, spec.head_dim, "full output width")?;
        let full_q_width = mul(2, full_output_width, "full query width")?;
        let full_kv_width = mul(spec.kv_heads, spec.head_dim, "full KV width")?;
        let linear_qk_width = mul(
            spec.linear_qk_heads,
            spec.linear_head_dim,
            "linear QK width",
        )?;
        let linear_output_width = mul(
            spec.linear_value_heads,
            spec.linear_head_dim,
            "linear output width",
        )?;
        let linear_qkv_width = mul(2, linear_qk_width, "linear QK pair width")?
            .checked_add(linear_output_width)
            .ok_or(QwenGraphError::Overflow("linear QKV width"))?;
        Ok(Self {
            hidden: spec.hidden_size,
            intermediate: spec.intermediate_size,
            vocab: QWEN35_VOCAB_SIZE as u64,
            q_heads: spec.attention_heads,
            kv_heads: spec.kv_heads,
            head_dim: spec.head_dim,
            full_q_width,
            full_kv_width,
            full_output_width,
            linear_qk_heads: spec.linear_qk_heads,
            linear_value_heads: spec.linear_value_heads,
            linear_head_dim: spec.linear_head_dim,
            linear_qkv_width,
            linear_output_width,
            linear_conv_kernel: 4,
            tied_embeddings: spec.tied_embeddings,
        })
    }
}

/// The reviewed schedule.  The interval in the model config is only a
/// consistency field; this list is the dispatch authority.
pub const QWEN35_LAYER_TYPES: [LayerType; QWEN35_LAYER_COUNT] = [
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::FullAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::FullAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::FullAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::FullAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::FullAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::FullAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::FullAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::LinearAttention,
    LayerType::FullAttention,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QwenGraphError {
    InvalidModel(String),
    InvalidPlan(String),
    ZeroTokenCount,
    ZeroStateCapacity,
    TokenCountExceedsCapacity { token_count: u64, capacity: u64 },
    CapacityExceedsMax { capacity: u64, max_position: u64 },
    Overflow(&'static str),
    UnsupportedDType(TensorDType),
    Tensor(TensorError),
    Operation(OpError),
}

impl fmt::Display for QwenGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModel(message) => write!(formatter, "invalid Qwen graph model: {message}"),
            Self::InvalidPlan(message) => {
                write!(formatter, "invalid Qwen graph weight plan: {message}")
            }
            Self::ZeroTokenCount => formatter.write_str("graph token count must be non-zero"),
            Self::ZeroStateCapacity => formatter.write_str("graph state capacity must be non-zero"),
            Self::TokenCountExceedsCapacity {
                token_count,
                capacity,
            } => write!(
                formatter,
                "graph token count {token_count} exceeds state capacity {capacity}"
            ),
            Self::CapacityExceedsMax {
                capacity,
                max_position,
            } => write!(
                formatter,
                "graph state capacity {capacity} exceeds max position {max_position}"
            ),
            Self::Overflow(field) => write!(formatter, "graph {field} overflowed"),
            Self::UnsupportedDType(dtype) => {
                write!(formatter, "graph does not support tensor dtype {dtype:?}")
            }
            Self::Tensor(error) => write!(formatter, "graph tensor metadata error: {error}"),
            Self::Operation(error) => write!(formatter, "graph operation metadata error: {error}"),
        }
    }
}

impl std::error::Error for QwenGraphError {}

impl From<TensorError> for QwenGraphError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<OpError> for QwenGraphError {
    fn from(error: OpError) -> Self {
        Self::Operation(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QwenGraphDispatchError {
    UnknownWeight(String),
    KnownUnconsumedWeight(String),
}

impl fmt::Display for QwenGraphDispatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownWeight(name) => write!(formatter, "unknown graph weight: {name}"),
            Self::KnownUnconsumedWeight(name) => {
                write!(
                    formatter,
                    "known-unconsumed weight is not dispatchable: {name}"
                )
            }
        }
    }
}

impl std::error::Error for QwenGraphDispatchError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGraphTensorBacking {
    /// The graph owns this metadata entry as an external/standalone tensor.
    Owned,
    /// This tensor is a checked view of another graph tensor's payload.
    Alias { tensor_id: usize },
}

/// A tensor in the graph's metadata namespace. It has no backing buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenGraphTensor {
    id: usize,
    name: String,
    view: TensorView,
    backing: QwenGraphTensorBacking,
}

impl QwenGraphTensor {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn view(&self) -> &TensorView {
        &self.view
    }

    pub const fn backing(&self) -> QwenGraphTensorBacking {
        self.backing
    }
}

/// Structural nodes are deliberately not represented as fake semantic op
/// kinds. They contain fixed shape/state metadata only; D0 does not resolve
/// request-local positions or execute a materialization on the CPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGraphNodeKind {
    Semantic(SemanticOpKind),
    /// Structural BF16 `[256]` -> `[heads, 256]` scale materialization. The
    /// later execution/state owner performs or orchestrates the materialize.
    AttentionScaleMaterialization {
        layer: u32,
        heads: u32,
        head_dim: u32,
    },
    /// Deferred C3 attention preprocessing. `token_count` is a tensor extent,
    /// not a prefill transition descriptor; D1 supplies runtime positions.
    AttentionPreprocess {
        layer: u32,
        token_count: u64,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
    },
    FullKvAppend {
        layer: u32,
        state: KvStateDescriptor,
    },
    FullCausalAttention {
        layer: u32,
        state: KvStateDescriptor,
        query_shape: [u64; 3],
        output_shape: [u64; 3],
    },
    LinearAttentionState {
        layer: u32,
        state: LinearAttentionStateDescriptor,
        output_shape: [u64; 2],
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenGraphNode {
    label: String,
    kind: QwenGraphNodeKind,
    operation: Option<SemanticOpDescriptor>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    dependencies: Vec<usize>,
    weights: Vec<WeightConsumerKey>,
}

impl QwenGraphNode {
    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn kind(&self) -> QwenGraphNodeKind {
        self.kind
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

    pub fn weight_consumers(&self) -> &[WeightConsumerKey] {
        &self.weights
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGraphStateKind {
    FullKey,
    FullValue,
    LinearConvolution,
    LinearRecurrent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenGraphStateDescriptor {
    Kv(KvStateDescriptor),
    Linear(LinearAttentionStateDescriptor),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenGraphState {
    layer: u32,
    kind: QwenGraphStateKind,
    descriptor: QwenGraphStateDescriptor,
    dtype: DType,
    encoding: Encoding,
    shape: Vec<u64>,
    strides: Vec<u64>,
    byte_size: u64,
}

impl QwenGraphState {
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    pub const fn kind(&self) -> QwenGraphStateKind {
        self.kind
    }

    pub const fn descriptor(&self) -> QwenGraphStateDescriptor {
        self.descriptor
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn encoding(&self) -> Encoding {
        self.encoding
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn strides(&self) -> &[u64] {
        &self.strides
    }

    pub const fn byte_size(&self) -> u64 {
        self.byte_size
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenGraphWeightBinding {
    consumer: WeightConsumerKey,
    tensor_name: String,
    classification: WeightClassification,
    dtype: TensorDType,
    shape: Vec<u64>,
    source_range: [u64; 2],
    destination_start: u64,
}

impl QwenGraphWeightBinding {
    pub const fn consumer(&self) -> WeightConsumerKey {
        self.consumer
    }

    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    pub const fn classification(&self) -> WeightClassification {
        self.classification
    }

    pub const fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub const fn source_range(&self) -> [u64; 2] {
        self.source_range
    }

    pub const fn destination_start(&self) -> u64 {
        self.destination_start
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenGraph {
    model_fingerprint: String,
    plan_digest: [u8; 32],
    token_count: u64,
    state_capacity: u64,
    layer_types: Vec<LayerType>,
    tensors: Vec<QwenGraphTensor>,
    nodes: Vec<QwenGraphNode>,
    weight_bindings: Vec<QwenGraphWeightBinding>,
    known_unconsumed: BTreeSet<String>,
    states: Vec<QwenGraphState>,
    total_state_bytes: u64,
    fp8_sidecar_fingerprint: Option<String>,
}

impl QwenGraph {
    pub fn fp8_sidecar_fingerprint(&self) -> Option<&str> {
        self.fp8_sidecar_fingerprint.as_deref()
    }
}

impl QwenGraph {
    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub const fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }

    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    pub const fn state_capacity(&self) -> u64 {
        self.state_capacity
    }

    pub fn layer_types(&self) -> &[LayerType] {
        &self.layer_types
    }

    pub fn tensor_metadata(&self) -> &[QwenGraphTensor] {
        &self.tensors
    }

    pub fn nodes(&self) -> &[QwenGraphNode] {
        &self.nodes
    }

    pub fn weight_bindings(&self) -> &[QwenGraphWeightBinding] {
        &self.weight_bindings
    }

    pub fn states(&self) -> &[QwenGraphState] {
        &self.states
    }

    pub const fn total_state_bytes(&self) -> u64 {
        self.total_state_bytes
    }

    /// Look up a weight binding in this metadata graph. The method never
    /// uploads or executes anything.
    pub fn weight_binding(
        &self,
        tensor_name: &str,
    ) -> Result<&QwenGraphWeightBinding, QwenGraphDispatchError> {
        if let Some(binding) = self
            .weight_bindings
            .iter()
            .find(|binding| binding.tensor_name == tensor_name)
        {
            return Ok(binding);
        }
        if self.known_unconsumed.contains(tensor_name) {
            return Err(QwenGraphDispatchError::KnownUnconsumedWeight(
                tensor_name.to_owned(),
            ));
        }
        Err(QwenGraphDispatchError::UnknownWeight(
            tensor_name.to_owned(),
        ))
    }
}

pub fn build_qwen35_graph(
    lock: &ModelLock,
    plan: &WeightLoadPlan,
    token_count: u64,
    state_capacity: u64,
) -> Result<QwenGraph, QwenGraphError> {
    let spec = validate_reviewed_model(lock)?;
    let dimensions = QwenGraphDimensions::from_spec(spec)?;
    if token_count == 0 {
        return Err(QwenGraphError::ZeroTokenCount);
    }
    if state_capacity == 0 {
        return Err(QwenGraphError::ZeroStateCapacity);
    }
    if token_count > state_capacity {
        return Err(QwenGraphError::TokenCountExceedsCapacity {
            token_count,
            capacity: state_capacity,
        });
    }
    let max_position = lock.model.architecture.text_config.max_position_embeddings;
    if state_capacity > max_position {
        return Err(QwenGraphError::CapacityExceedsMax {
            capacity: state_capacity,
            max_position,
        });
    }

    let (bindings, known_unconsumed) = validate_plan(lock, plan, dimensions)?;
    let builder = GraphBuilder::new(GraphBuilderConfig {
        layer_types: lock.model.architecture.text_config.layer_types.clone(),
        dimensions,
        token_count,
        state_capacity,
        bindings,
        known_unconsumed,
        model_fingerprint: lock.fingerprint().to_owned(),
        plan_digest: *plan.digest(),
        fp8_tensor_names: BTreeSet::new(),
        fp8_dtype: None,
        fp8_sidecar_fingerprint: None,
    })?;
    builder.build()
}

/// Build the same production Qwen3.5 graph with every text-linear weight that
/// is present in a verified Phase 10 sidecar represented as resident OCP
/// E4M3FN plus outer-dimension FP32 scales.
pub fn build_qwen35_fp8_graph(
    lock: &ModelLock,
    plan: &WeightLoadPlan,
    sidecar: &VerifiedFp8Sidecar,
    token_count: u64,
    state_capacity: u64,
) -> Result<QwenGraph, QwenGraphError> {
    build_qwen35_fp8_graph_with_dtype(
        lock,
        plan,
        sidecar,
        token_count,
        state_capacity,
        DType::F8E4M3Fn,
    )
}

/// Build the CDNA3 resident form of the Phase 10 sidecar. Values are expected
/// to be numerically converted to FNUZ by the provisioning source.
pub fn build_qwen35_fp8_fnuz_graph(
    lock: &ModelLock,
    plan: &WeightLoadPlan,
    sidecar: &VerifiedFp8Sidecar,
    token_count: u64,
    state_capacity: u64,
) -> Result<QwenGraph, QwenGraphError> {
    build_qwen35_fp8_graph_with_dtype(
        lock,
        plan,
        sidecar,
        token_count,
        state_capacity,
        DType::F8E4M3FnuZ,
    )
}

fn build_qwen35_fp8_graph_with_dtype(
    lock: &ModelLock,
    plan: &WeightLoadPlan,
    sidecar: &VerifiedFp8Sidecar,
    token_count: u64,
    state_capacity: u64,
    fp8_dtype: DType,
) -> Result<QwenGraph, QwenGraphError> {
    if sidecar.source_lock_fingerprint() != lock.fingerprint() {
        return Err(QwenGraphError::InvalidModel(
            "FP8 sidecar source identity differs from the model lock".to_owned(),
        ));
    }
    let spec = validate_reviewed_model(lock)?;
    let dimensions = QwenGraphDimensions::from_spec(spec)?;
    if token_count == 0 {
        return Err(QwenGraphError::ZeroTokenCount);
    }
    if state_capacity == 0 {
        return Err(QwenGraphError::ZeroStateCapacity);
    }
    if token_count > state_capacity {
        return Err(QwenGraphError::TokenCountExceedsCapacity {
            token_count,
            capacity: state_capacity,
        });
    }
    let max_position = lock.model.architecture.text_config.max_position_embeddings;
    if state_capacity > max_position {
        return Err(QwenGraphError::CapacityExceedsMax {
            capacity: state_capacity,
            max_position,
        });
    }
    let (bindings, known_unconsumed) = validate_plan(lock, plan, dimensions)?;
    let by_name: BTreeMap<_, _> = bindings
        .iter()
        .map(|binding| (binding.tensor_name.as_str(), binding))
        .collect();
    let mut fp8_tensor_names = BTreeSet::new();
    for tensor in sidecar.tensors() {
        let binding = by_name.get(tensor.name.as_str()).ok_or_else(|| {
            QwenGraphError::InvalidPlan(format!(
                "FP8 sidecar tensor is not a required Qwen weight: {}",
                tensor.name
            ))
        })?;
        if tensor.shape.as_slice() != binding.shape.as_slice()
            || !is_fp8_linear_consumer(binding.consumer.role)
            || !fp8_tensor_names.insert(tensor.name.clone())
        {
            return Err(QwenGraphError::InvalidPlan(format!(
                "FP8 sidecar tensor differs from its graph binding: {}",
                tensor.name
            )));
        }
    }
    let expected_fp8: BTreeSet<_> = bindings
        .iter()
        .filter(|binding| is_fp8_linear_consumer(binding.consumer.role))
        .map(|binding| binding.tensor_name.clone())
        .collect();
    if fp8_tensor_names != expected_fp8 {
        return Err(QwenGraphError::InvalidPlan(
            "FP8 sidecar does not cover the exact text-linear weight set".to_owned(),
        ));
    }
    GraphBuilder::new(GraphBuilderConfig {
        layer_types: lock.model.architecture.text_config.layer_types.clone(),
        dimensions,
        token_count,
        state_capacity,
        bindings,
        known_unconsumed,
        model_fingerprint: lock.fingerprint().to_owned(),
        plan_digest: *plan.digest(),
        fp8_tensor_names,
        fp8_dtype: Some(fp8_dtype),
        fp8_sidecar_fingerprint: Some(sidecar.manifest_fingerprint().to_owned()),
    })?
    .build()
}

fn is_fp8_linear_consumer(consumer: WeightConsumer) -> bool {
    matches!(
        consumer,
        WeightConsumer::MlpGate
            | WeightConsumer::MlpUp
            | WeightConsumer::MlpDown
            | WeightConsumer::GdnInProjQkv
            | WeightConsumer::GdnInProjZ
            | WeightConsumer::GdnInProjB
            | WeightConsumer::GdnInProjA
            | WeightConsumer::GdnOutProj
            | WeightConsumer::AttentionQ
            | WeightConsumer::AttentionK
            | WeightConsumer::AttentionV
            | WeightConsumer::AttentionO
    )
}

fn validate_reviewed_model(lock: &ModelLock) -> Result<Qwen35ReviewedSpec, QwenGraphError> {
    let model = &lock.model;
    let architecture = &model.architecture;
    let config = &architecture.text_config;
    let spec = reviewed_qwen35_spec(lock).ok_or_else(|| {
        QwenGraphError::InvalidModel(
            "model lock identity is not a reviewed Qwen3.5 dense revision".to_owned(),
        )
    })?;
    if lock.schema_version != "model-lock-v1"
        || model.repo_type != "model"
        || model.requested_revision != "main"
    {
        return Err(QwenGraphError::InvalidModel(
            "model lock envelope differs from the reviewed Qwen3.5 contract".to_owned(),
        ));
    }
    if architecture.architectures != ["Qwen3_5ForConditionalGeneration"]
        || architecture.top_level_architecture != "Qwen3_5ForConditionalGeneration"
        || architecture.model_type != "qwen3_5"
        || architecture.text_model_type != "qwen3_5_text"
        || architecture.phase_scope != "text-only"
        || architecture.custom_code
        || architecture.converted
        || architecture.moe
        || !architecture.vision.present
        || architecture.vision.tensor_prefix != "model.visual."
        || architecture.vision.tensor_count != spec.vision_tensor_count
        || architecture.vision.phase3_status != crate::model::ComponentStatus::KnownUnconsumed
        || !architecture.mtp.present
        || architecture.mtp.tensor_prefix != "mtp."
        || architecture.mtp.tensor_count != 15
        || architecture.mtp.phase3_status != crate::model::ComponentStatus::KnownUnconsumed
    {
        return Err(QwenGraphError::InvalidModel(
            "model lock is not the fixed text-only component contract".to_owned(),
        ));
    }
    let expected_schedule: Vec<_> = (0..spec.layer_count)
        .map(|layer| {
            if (layer + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    if config.hidden_size != spec.hidden_size
        || config.num_hidden_layers != spec.layer_count
        || config.num_attention_heads != spec.attention_heads
        || config.num_key_value_heads != spec.kv_heads
        || config.head_dim != spec.head_dim
        || config.intermediate_size != spec.intermediate_size
        || config.dtype != TensorDType::Bf16
        || config.rms_norm_eps != "1e-6"
        || config.attention_bias
        || config.attention_dropout != "0"
        || !config.attn_output_gate
        || config.full_attention_interval != 4
        || config.layer_types != expected_schedule
        || config.max_position_embeddings != QWEN35_MAX_POSITION_EMBEDDINGS
        || config.rope_parameters.rope_type != RopeType::Default
        || config.rope_parameters.rope_theta != 10_000_000
        || config.rope_parameters.partial_rotary_factor != "0.25"
        || !config.rope_parameters.mrope_interleaved
        || config.rope_parameters.mrope_section != [11, 11, 10]
        || config.tie_word_embeddings != spec.tied_embeddings
        || !config.use_cache
        || config.vocab_size != QWEN35_VOCAB_SIZE as u64
        || config.mtp_num_hidden_layers != 1
    {
        return Err(QwenGraphError::InvalidModel(
            "Qwen text configuration differs from the fixed graph contract".to_owned(),
        ));
    }
    let schedule = &architecture.layer_schedule;
    if schedule.kind != "explicit"
        || schedule.num_hidden_layers != spec.layer_count
        || schedule.full_attention_interval != 4
        || schedule.layer_types != expected_schedule
        || schedule.allowed_types != [LayerType::LinearAttention, LayerType::FullAttention]
    {
        return Err(QwenGraphError::InvalidModel(
            "layer schedule is not the explicit reviewed schedule".to_owned(),
        ));
    }
    let classifications = &model.tensor_contract.classifications;
    if model.tensor_contract.index_path != "model.safetensors.index.json"
        || model.tensor_contract.shards.len() != spec.shard_count
        || model.tensor_contract.unknown_policy != "reject"
        || model.tensor_contract.duplicate_policy != "reject"
        || model.tensor_contract.index_policy != "exact-weight-map-and-shard-metadata"
        || model.tensor_contract.indexed_tensor_count != spec.indexed_tensor_count
    {
        return Err(QwenGraphError::InvalidModel(
            "tensor contract is not the fixed text/vision/MTP catalog".to_owned(),
        ));
    }
    let text = classifications.iter().find(|entry| entry.id == "text");
    let vision = classifications.iter().find(|entry| entry.id == "vision");
    let mtp = classifications.iter().find(|entry| entry.id == "mtp");
    let output = classifications.iter().find(|entry| entry.id == "output");
    if text.is_none_or(|entry| {
        entry.prefix != "model.language_model."
            || entry.tensor_count
                != if spec.tied_embeddings {
                    spec.text_tensor_count
                } else {
                    spec.text_tensor_count - 1
                }
            || entry.phase3_status != ClassificationStatus::PartiallyConsumed
    }) || vision.is_none_or(|entry| {
        entry.prefix != "model.visual."
            || entry.tensor_count != spec.vision_tensor_count
            || entry.phase3_status != ClassificationStatus::KnownUnconsumed
    }) || mtp.is_none_or(|entry| {
        entry.prefix != "mtp."
            || entry.tensor_count != 15
            || entry.phase3_status != ClassificationStatus::KnownUnconsumed
    }) || (spec.tied_embeddings && output.is_some())
        || (!spec.tied_embeddings
            && output.is_none_or(|entry| {
                entry.prefix != "lm_head."
                    || entry.tensor_count != 1
                    || entry.phase3_status != ClassificationStatus::Consumed
            }))
    {
        return Err(QwenGraphError::InvalidModel(
            "tensor classifications differ from the reviewed family contract".to_owned(),
        ));
    }
    Ok(spec)
}

fn validate_plan(
    lock: &ModelLock,
    plan: &WeightLoadPlan,
    dimensions: QwenGraphDimensions,
) -> Result<(Vec<QwenGraphWeightBinding>, BTreeSet<String>), QwenGraphError> {
    if plan.schema_version != lock.schema_version
        || plan.repo_id != lock.model.repo_id
        || plan.resolved_revision != lock.model.resolved_revision
        || plan.lock_fingerprint != lock.fingerprint()
        || plan.tied_embeddings != dimensions.tied_embeddings
    {
        return Err(QwenGraphError::InvalidPlan(
            "plan identity or tied-embedding condition differs from the lock".to_owned(),
        ));
    }
    let expected_entries = usize::try_from(lock.model.tensor_contract.indexed_tensor_count)
        .map_err(|_| QwenGraphError::Overflow("plan entry count"))?;
    if plan.entries.len() != expected_entries {
        return Err(QwenGraphError::InvalidPlan(format!(
            "expected {} entries, got {}",
            expected_entries,
            plan.entries.len()
        )));
    }

    validate_non_overlapping_source_ranges(plan)?;

    // Rebuild the digest from public entry metadata. WeightLoadPlan's own
    // digest is private, so equality with this canonical rebuild is the
    // fail-closed check for both entry content and stored digest. This is
    // canonical load-plan metadata validation, not proof of a verified cache
    // catalog. The later owner, `upload_verified_weight`, rebinds every entry
    // to `VerifiedCache` and enforces verified-cache descriptor equality
    // before upload.
    let descriptors: Vec<TensorDescriptor> = plan
        .entries
        .iter()
        .map(|entry| {
            let byte_size = entry.source_range[1].saturating_sub(entry.source_range[0]);
            TensorDescriptor {
                tensor_name: entry.tensor_name.clone(),
                source_file: entry.source_file.clone(),
                dtype: entry.dtype,
                shape: entry.shape.clone(),
                header_length_field_bytes: 8,
                header_length_bytes: 0,
                data_buffer_start: entry.source_range[0],
                data_offset_basis: "safetensors-data-buffer".to_owned(),
                data_offsets: [0, byte_size],
                absolute_byte_range: entry.source_range,
                byte_size,
            }
        })
        .collect();
    let canonical = build_weight_load_plan(lock, descriptors.iter())
        .map_err(|error| QwenGraphError::InvalidPlan(error.to_string()))?;
    if &canonical != plan {
        return Err(QwenGraphError::InvalidPlan(
            "plan entries or digest are not canonical for the supplied lock".to_owned(),
        ));
    }

    let expected_consumers = expected_consumers(
        &lock.model.architecture.text_config.layer_types,
        dimensions.tied_embeddings,
    );
    let mut required = Vec::with_capacity(expected_consumers.len());
    let mut known_unconsumed = BTreeSet::new();
    let mut vision_count = 0_u64;
    let mut mtp_count = 0_u64;
    for entry in &plan.entries {
        match entry.classification {
            WeightClassification::Required => {
                let consumer = entry.consumer.ok_or_else(|| {
                    QwenGraphError::InvalidPlan(format!(
                        "required tensor has no consumer: {}",
                        entry.tensor_name
                    ))
                })?;
                let (expected_name, expected_dtype, expected_shape) = expected_weight(
                    consumer,
                    &lock.model.architecture.text_config.layer_types,
                    dimensions,
                )?;
                if entry.tensor_name != expected_name
                    || entry.dtype != expected_dtype
                    || entry.shape != expected_shape
                    || entry.destination_start.is_none()
                    || entry.chunks.is_empty()
                {
                    return Err(QwenGraphError::InvalidPlan(format!(
                        "required tensor metadata differs from its consumer: {}",
                        entry.tensor_name
                    )));
                }
                let destination_start = entry.destination_start.ok_or_else(|| {
                    QwenGraphError::InvalidPlan("required tensor has no destination".to_owned())
                })?;
                required.push(QwenGraphWeightBinding {
                    consumer,
                    tensor_name: entry.tensor_name.clone(),
                    classification: entry.classification,
                    dtype: entry.dtype,
                    shape: entry.shape.clone(),
                    source_range: entry.source_range,
                    destination_start,
                });
            }
            WeightClassification::KnownUnconsumed => {
                if entry.consumer.is_some()
                    || entry.destination_start.is_some()
                    || !entry.chunks.is_empty()
                    || (!entry.tensor_name.starts_with("model.visual.")
                        && !entry.tensor_name.starts_with("mtp."))
                {
                    return Err(QwenGraphError::InvalidPlan(format!(
                        "known-unconsumed tensor is malformed: {}",
                        entry.tensor_name
                    )));
                }
                if entry.tensor_name.starts_with("model.visual.") {
                    vision_count = vision_count
                        .checked_add(1)
                        .ok_or(QwenGraphError::Overflow("vision tensor count"))?;
                } else if entry.tensor_name.starts_with("mtp.") {
                    mtp_count = mtp_count
                        .checked_add(1)
                        .ok_or(QwenGraphError::Overflow("MTP tensor count"))?;
                }
                known_unconsumed.insert(entry.tensor_name.clone());
            }
            WeightClassification::ConfigConditional => {
                return Err(QwenGraphError::InvalidPlan(format!(
                    "unexpected config-conditional tensor: {}",
                    entry.tensor_name
                )));
            }
        }
    }
    if required.len() != expected_consumers.len()
        || vision_count != lock.model.architecture.vision.tensor_count
        || mtp_count != lock.model.architecture.mtp.tensor_count
        || known_unconsumed.len()
            != usize::try_from(
                lock.model.architecture.vision.tensor_count
                    + lock.model.architecture.mtp.tensor_count,
            )
            .map_err(|_| QwenGraphError::Overflow("known-unconsumed count"))?
    {
        return Err(QwenGraphError::InvalidPlan(format!(
            "consumer/component coverage differs: required={}, vision={vision_count}/{}, mtp={mtp_count}/{}, known-unconsumed={}",
            required.len(),
            lock.model.architecture.vision.tensor_count,
            lock.model.architecture.mtp.tensor_count,
            known_unconsumed.len(),
        )));
    }
    required.sort_by_key(|binding| binding.consumer);
    let observed: BTreeSet<_> = required.iter().map(|binding| binding.consumer).collect();
    if observed != expected_consumers {
        return Err(QwenGraphError::InvalidPlan(
            "required consumer coverage is not one-to-one and complete".to_owned(),
        ));
    }
    Ok((required, known_unconsumed))
}

fn validate_non_overlapping_source_ranges(plan: &WeightLoadPlan) -> Result<(), QwenGraphError> {
    let mut ranges: BTreeMap<&str, Vec<[u64; 2]>> = BTreeMap::new();
    for entry in &plan.entries {
        let [start, end] = entry.source_range;
        if start >= end {
            return Err(QwenGraphError::InvalidPlan(format!(
                "source range is empty or reversed: {}",
                entry.tensor_name
            )));
        }
        ranges
            .entry(entry.source_file.as_str())
            .or_default()
            .push([start, end]);
    }
    for (source_file, mut source_ranges) in ranges {
        source_ranges.sort_unstable_by_key(|range| (range[0], range[1]));
        for pair in source_ranges.windows(2) {
            if pair[1][0] < pair[0][1] {
                return Err(QwenGraphError::InvalidPlan(format!(
                    "source ranges overlap in {source_file}: [{}, {}) and [{}, {})",
                    pair[0][0], pair[0][1], pair[1][0], pair[1][1]
                )));
            }
        }
    }
    Ok(())
}

fn expected_consumers(
    layer_types: &[LayerType],
    tied_embeddings: bool,
) -> BTreeSet<WeightConsumerKey> {
    let embedding_role = if tied_embeddings {
        WeightConsumer::EmbeddingAndTiedOutput
    } else {
        WeightConsumer::Embedding
    };
    let mut result = BTreeSet::from([
        WeightConsumerKey {
            layer: None,
            role: embedding_role,
        },
        WeightConsumerKey {
            layer: None,
            role: WeightConsumer::FinalNorm,
        },
    ]);
    if !tied_embeddings {
        result.insert(WeightConsumerKey {
            layer: None,
            role: WeightConsumer::OutputProjection,
        });
    }
    for (layer, layer_type) in layer_types.iter().copied().enumerate() {
        let layer = layer as u64;
        result.extend([
            WeightConsumerKey {
                layer: Some(layer),
                role: WeightConsumer::InputNorm,
            },
            WeightConsumerKey {
                layer: Some(layer),
                role: WeightConsumer::PostAttentionNorm,
            },
            WeightConsumerKey {
                layer: Some(layer),
                role: WeightConsumer::MlpGate,
            },
            WeightConsumerKey {
                layer: Some(layer),
                role: WeightConsumer::MlpUp,
            },
            WeightConsumerKey {
                layer: Some(layer),
                role: WeightConsumer::MlpDown,
            },
        ]);
        let attention_roles: &[WeightConsumer] = match layer_type {
            LayerType::LinearAttention => &[
                WeightConsumer::GdnInProjQkv,
                WeightConsumer::GdnInProjZ,
                WeightConsumer::GdnInProjB,
                WeightConsumer::GdnInProjA,
                WeightConsumer::GdnConv1d,
                WeightConsumer::GdnALog,
                WeightConsumer::GdnDtBias,
                WeightConsumer::GdnNorm,
                WeightConsumer::GdnOutProj,
            ],
            LayerType::FullAttention => &[
                WeightConsumer::AttentionQ,
                WeightConsumer::AttentionK,
                WeightConsumer::AttentionV,
                WeightConsumer::AttentionO,
                WeightConsumer::AttentionQNorm,
                WeightConsumer::AttentionKNorm,
            ],
        };
        for &role in attention_roles {
            result.insert(WeightConsumerKey {
                layer: Some(layer),
                role,
            });
        }
    }
    result
}

fn expected_weight(
    consumer: WeightConsumerKey,
    layer_types: &[LayerType],
    dimensions: QwenGraphDimensions,
) -> Result<(String, TensorDType, Vec<u64>), QwenGraphError> {
    let layer = consumer.layer;
    let role = consumer.role;
    let (name, dtype, shape) = match (layer, role) {
        (None, WeightConsumer::EmbeddingAndTiedOutput) => (
            QWEN35_EMBEDDING_TENSOR.to_owned(),
            TensorDType::Bf16,
            vec![dimensions.vocab, dimensions.hidden],
        ),
        (None, WeightConsumer::Embedding) => (
            QWEN35_EMBEDDING_TENSOR.to_owned(),
            TensorDType::Bf16,
            vec![dimensions.vocab, dimensions.hidden],
        ),
        (None, WeightConsumer::OutputProjection) => (
            "lm_head.weight".to_owned(),
            TensorDType::Bf16,
            vec![dimensions.vocab, dimensions.hidden],
        ),
        (None, WeightConsumer::FinalNorm) => (
            "model.language_model.norm.weight".to_owned(),
            TensorDType::Bf16,
            vec![dimensions.hidden],
        ),
        (Some(layer), role)
            if usize::try_from(layer)
                .ok()
                .is_some_and(|layer| layer < layer_types.len()) =>
        {
            let prefix = format!("model.language_model.layers.{layer}.");
            let common = match role {
                WeightConsumer::InputNorm => Some((
                    "input_layernorm.weight",
                    TensorDType::Bf16,
                    vec![dimensions.hidden],
                )),
                WeightConsumer::PostAttentionNorm => Some((
                    "post_attention_layernorm.weight",
                    TensorDType::Bf16,
                    vec![dimensions.hidden],
                )),
                WeightConsumer::MlpGate => Some((
                    "mlp.gate_proj.weight",
                    TensorDType::Bf16,
                    vec![dimensions.intermediate, dimensions.hidden],
                )),
                WeightConsumer::MlpUp => Some((
                    "mlp.up_proj.weight",
                    TensorDType::Bf16,
                    vec![dimensions.intermediate, dimensions.hidden],
                )),
                WeightConsumer::MlpDown => Some((
                    "mlp.down_proj.weight",
                    TensorDType::Bf16,
                    vec![dimensions.hidden, dimensions.intermediate],
                )),
                _ => None,
            };
            if let Some((suffix, dtype, shape)) = common {
                (format!("{prefix}{suffix}"), dtype, shape)
            } else {
                let full = matches!(
                    role,
                    WeightConsumer::AttentionQ
                        | WeightConsumer::AttentionK
                        | WeightConsumer::AttentionV
                        | WeightConsumer::AttentionO
                        | WeightConsumer::AttentionQNorm
                        | WeightConsumer::AttentionKNorm
                );
                let linear = matches!(
                    role,
                    WeightConsumer::GdnInProjQkv
                        | WeightConsumer::GdnInProjZ
                        | WeightConsumer::GdnInProjB
                        | WeightConsumer::GdnInProjA
                        | WeightConsumer::GdnConv1d
                        | WeightConsumer::GdnALog
                        | WeightConsumer::GdnDtBias
                        | WeightConsumer::GdnNorm
                        | WeightConsumer::GdnOutProj
                );
                if !full && !linear {
                    return Err(QwenGraphError::InvalidPlan(
                        "consumer has an invalid layer binding".to_owned(),
                    ));
                }
                let (suffix, dtype, shape) = if full {
                    match role {
                        WeightConsumer::AttentionQ => (
                            "self_attn.q_proj.weight",
                            TensorDType::Bf16,
                            vec![dimensions.full_q_width, dimensions.hidden],
                        ),
                        WeightConsumer::AttentionK => (
                            "self_attn.k_proj.weight",
                            TensorDType::Bf16,
                            vec![dimensions.full_kv_width, dimensions.hidden],
                        ),
                        WeightConsumer::AttentionV => (
                            "self_attn.v_proj.weight",
                            TensorDType::Bf16,
                            vec![dimensions.full_kv_width, dimensions.hidden],
                        ),
                        WeightConsumer::AttentionO => (
                            "self_attn.o_proj.weight",
                            TensorDType::Bf16,
                            vec![dimensions.hidden, dimensions.full_output_width],
                        ),
                        WeightConsumer::AttentionQNorm => (
                            "self_attn.q_norm.weight",
                            TensorDType::Bf16,
                            vec![dimensions.head_dim],
                        ),
                        WeightConsumer::AttentionKNorm => (
                            "self_attn.k_norm.weight",
                            TensorDType::Bf16,
                            vec![dimensions.head_dim],
                        ),
                        _ => unreachable!(),
                    }
                } else {
                    match role {
                        WeightConsumer::GdnInProjQkv => (
                            "linear_attn.in_proj_qkv.weight",
                            TensorDType::Bf16,
                            vec![dimensions.linear_qkv_width, dimensions.hidden],
                        ),
                        WeightConsumer::GdnInProjZ => (
                            "linear_attn.in_proj_z.weight",
                            TensorDType::Bf16,
                            vec![dimensions.linear_output_width, dimensions.hidden],
                        ),
                        WeightConsumer::GdnInProjB => (
                            "linear_attn.in_proj_b.weight",
                            TensorDType::Bf16,
                            vec![dimensions.linear_value_heads, dimensions.hidden],
                        ),
                        WeightConsumer::GdnInProjA => (
                            "linear_attn.in_proj_a.weight",
                            TensorDType::Bf16,
                            vec![dimensions.linear_value_heads, dimensions.hidden],
                        ),
                        WeightConsumer::GdnConv1d => (
                            "linear_attn.conv1d.weight",
                            TensorDType::Bf16,
                            vec![
                                dimensions.linear_qkv_width,
                                1,
                                dimensions.linear_conv_kernel,
                            ],
                        ),
                        WeightConsumer::GdnALog => (
                            "linear_attn.A_log",
                            TensorDType::F32,
                            vec![dimensions.linear_value_heads],
                        ),
                        WeightConsumer::GdnDtBias => (
                            "linear_attn.dt_bias",
                            TensorDType::Bf16,
                            vec![dimensions.linear_value_heads],
                        ),
                        WeightConsumer::GdnNorm => (
                            "linear_attn.norm.weight",
                            TensorDType::F32,
                            vec![dimensions.linear_head_dim],
                        ),
                        WeightConsumer::GdnOutProj => (
                            "linear_attn.out_proj.weight",
                            TensorDType::Bf16,
                            vec![dimensions.hidden, dimensions.linear_output_width],
                        ),
                        _ => unreachable!(),
                    }
                };
                (format!("{prefix}{suffix}"), dtype, shape)
            }
        }
        _ => {
            return Err(QwenGraphError::InvalidPlan(
                "consumer layer is outside the fixed graph".to_owned(),
            ));
        }
    };
    Ok((name, dtype, shape))
}

struct GraphBuilderConfig {
    layer_types: Vec<LayerType>,
    dimensions: QwenGraphDimensions,
    token_count: u64,
    state_capacity: u64,
    bindings: Vec<QwenGraphWeightBinding>,
    known_unconsumed: BTreeSet<String>,
    model_fingerprint: String,
    plan_digest: [u8; 32],
    fp8_tensor_names: BTreeSet<String>,
    fp8_dtype: Option<DType>,
    fp8_sidecar_fingerprint: Option<String>,
}

struct GraphBuilder {
    layer_types: Vec<LayerType>,
    dimensions: QwenGraphDimensions,
    token_count: u64,
    state_capacity: u64,
    bindings: BTreeMap<WeightConsumerKey, QwenGraphWeightBinding>,
    known_unconsumed: BTreeSet<String>,
    model_fingerprint: String,
    plan_digest: [u8; 32],
    fp8_tensor_names: BTreeSet<String>,
    fp8_dtype: Option<DType>,
    fp8_sidecar_fingerprint: Option<String>,
    tensors: Vec<QwenGraphTensor>,
    producers: Vec<Option<usize>>,
    nodes: Vec<QwenGraphNode>,
    states: Vec<QwenGraphState>,
    total_state_bytes: u64,
    weight_tensors: BTreeMap<WeightConsumerKey, usize>,
}

impl GraphBuilder {
    fn new(config: GraphBuilderConfig) -> Result<Self, QwenGraphError> {
        let GraphBuilderConfig {
            layer_types,
            dimensions,
            token_count,
            state_capacity,
            bindings,
            known_unconsumed,
            model_fingerprint,
            plan_digest,
            fp8_tensor_names,
            fp8_dtype,
            fp8_sidecar_fingerprint,
        } = config;
        let bindings = bindings
            .into_iter()
            .map(|binding| (binding.consumer, binding))
            .collect::<BTreeMap<_, _>>();
        if bindings.len() != expected_consumers(&layer_types, dimensions.tied_embeddings).len() {
            return Err(QwenGraphError::InvalidPlan(
                "graph binding map is not one-to-one".to_owned(),
            ));
        }
        Ok(Self {
            layer_types,
            dimensions,
            token_count,
            state_capacity,
            bindings,
            known_unconsumed,
            model_fingerprint,
            plan_digest,
            fp8_tensor_names,
            fp8_dtype,
            fp8_sidecar_fingerprint,
            tensors: Vec::new(),
            producers: Vec::new(),
            nodes: Vec::new(),
            states: Vec::new(),
            total_state_bytes: 0,
            weight_tensors: BTreeMap::new(),
        })
    }

    fn build(mut self) -> Result<QwenGraph, QwenGraphError> {
        self.build_states()?;
        self.build_graph()?;
        Ok(QwenGraph {
            model_fingerprint: self.model_fingerprint,
            plan_digest: self.plan_digest,
            token_count: self.token_count,
            state_capacity: self.state_capacity,
            layer_types: self.layer_types,
            tensors: self.tensors,
            nodes: self.nodes,
            weight_bindings: self.bindings.into_values().collect(),
            known_unconsumed: self.known_unconsumed,
            states: self.states,
            total_state_bytes: self.total_state_bytes,
            fp8_sidecar_fingerprint: self.fp8_sidecar_fingerprint,
        })
    }

    fn build_states(&mut self) -> Result<(), QwenGraphError> {
        for (layer, layer_type) in self.layer_types.clone().into_iter().enumerate() {
            let layer = layer as u32;
            match layer_type {
                LayerType::FullAttention => {
                    let descriptor = KvStateDescriptor::new_with_layout(
                        layer,
                        self.state_capacity,
                        usize::try_from(self.dimensions.kv_heads)
                            .map_err(|_| QwenGraphError::Overflow("KV heads"))?,
                        usize::try_from(self.dimensions.head_dim)
                            .map_err(|_| QwenGraphError::Overflow("KV head dimension"))?,
                    )
                    .map_err(|error| QwenGraphError::InvalidPlan(error.to_string()))?;
                    self.add_state(
                        layer,
                        QwenGraphStateKind::FullKey,
                        QwenGraphStateDescriptor::Kv(descriptor),
                        descriptor.dtype(),
                        descriptor.encoding(),
                        descriptor.storage_shape().to_vec(),
                    )?;
                    self.add_state(
                        layer,
                        QwenGraphStateKind::FullValue,
                        QwenGraphStateDescriptor::Kv(descriptor),
                        descriptor.dtype(),
                        descriptor.encoding(),
                        descriptor.storage_shape().to_vec(),
                    )?;
                }
                LayerType::LinearAttention => {
                    let descriptor = LinearAttentionStateDescriptor::new_with_layout(
                        layer,
                        self.state_capacity,
                        usize::try_from(self.dimensions.linear_qk_heads)
                            .map_err(|_| QwenGraphError::Overflow("linear QK heads"))?,
                        usize::try_from(self.dimensions.linear_value_heads)
                            .map_err(|_| QwenGraphError::Overflow("linear value heads"))?,
                        usize::try_from(self.dimensions.linear_head_dim)
                            .map_err(|_| QwenGraphError::Overflow("linear head dimension"))?,
                        usize::try_from(self.dimensions.linear_conv_kernel)
                            .map_err(|_| QwenGraphError::Overflow("linear convolution kernel"))?,
                    )
                    .map_err(|error| QwenGraphError::InvalidPlan(error.to_string()))?;
                    let layout = descriptor.layout();
                    self.add_state(
                        layer,
                        QwenGraphStateKind::LinearConvolution,
                        QwenGraphStateDescriptor::Linear(descriptor),
                        crate::linear_attention::LinearAttentionLayout::CONV_STATE_DTYPE,
                        crate::linear_attention::LinearAttentionLayout::ENCODING,
                        layout.conv_state_shape().to_vec(),
                    )?;
                    self.add_state(
                        layer,
                        QwenGraphStateKind::LinearRecurrent,
                        QwenGraphStateDescriptor::Linear(descriptor),
                        crate::linear_attention::LinearAttentionLayout::RECURRENT_STATE_DTYPE,
                        crate::linear_attention::LinearAttentionLayout::ENCODING,
                        layout.recurrent_state_shape().to_vec(),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn add_state(
        &mut self,
        layer: u32,
        kind: QwenGraphStateKind,
        descriptor: QwenGraphStateDescriptor,
        dtype: DType,
        encoding: Encoding,
        shape: Vec<u64>,
    ) -> Result<(), QwenGraphError> {
        let (strides, byte_size) = checked_layout(dtype, encoding, &shape)?;
        self.total_state_bytes = self
            .total_state_bytes
            .checked_add(byte_size)
            .ok_or(QwenGraphError::Overflow("total state bytes"))?;
        self.states.push(QwenGraphState {
            layer,
            kind,
            descriptor,
            dtype,
            encoding,
            shape,
            strides,
            byte_size,
        });
        Ok(())
    }

    fn build_graph(&mut self) -> Result<(), QwenGraphError> {
        let token_ids = self.add_tensor("input.token_ids", view(DType::I32, &[self.token_count])?);
        let positions = self.add_tensor("input.positions", view(DType::I32, &[self.token_count])?);
        let embedding_role = if self.dimensions.tied_embeddings {
            WeightConsumer::EmbeddingAndTiedOutput
        } else {
            WeightConsumer::Embedding
        };
        let embedding_weight = self.weight_tensor(WeightConsumerKey {
            layer: None,
            role: embedding_role,
        })?;
        let hidden_view = view(DType::Bf16, &[self.token_count, self.dimensions.hidden])?;
        let mut hidden = self.add_tensor("embedding.output", hidden_view.clone());
        let embedding = SemanticOpDescriptor::new(
            SemanticOpKind::Embedding,
            vec![
                self.tensors[embedding_weight].view.clone(),
                self.tensors[token_ids].view.clone(),
            ],
            vec![hidden_view],
        )?;
        self.add_semantic(
            "embedding",
            embedding,
            vec![embedding_weight, token_ids],
            vec![hidden],
            vec![],
            vec![WeightConsumerKey {
                layer: None,
                role: embedding_role,
            }],
        )?;

        for (layer_index, layer_type) in self.layer_types.clone().into_iter().enumerate() {
            let layer = layer_index as u32;
            let layer_input = hidden;
            let norm_weight = self.weight_tensor(WeightConsumerKey {
                layer: Some(layer_index as u64),
                role: WeightConsumer::InputNorm,
            })?;
            let normed = self.activation(
                layer,
                "input_rmsnorm.output",
                &[self.token_count, self.dimensions.hidden],
            )?;
            let norm_op = SemanticOpDescriptor::new_rms_norm(
                vec![
                    self.tensors[layer_input].view.clone(),
                    self.tensors[norm_weight].view.clone(),
                ],
                vec![self.tensors[normed].view.clone()],
                1.0e-6,
                RmsNormScaleMode::OffsetOne,
            )?;
            self.add_semantic(
                &format!("layer.{layer}.input_rmsnorm"),
                norm_op,
                vec![layer_input, norm_weight],
                vec![normed],
                vec![],
                vec![WeightConsumerKey {
                    layer: Some(layer_index as u64),
                    role: WeightConsumer::InputNorm,
                }],
            )?;

            let attention_residual = match layer_type {
                LayerType::LinearAttention => self.build_linear_attention(layer, normed)?,
                LayerType::FullAttention => self.build_full_attention(layer, normed, positions)?,
            };

            let post_weight = self.weight_tensor(WeightConsumerKey {
                layer: Some(layer_index as u64),
                role: WeightConsumer::PostAttentionNorm,
            })?;
            let post_normed = self.activation(
                layer,
                "post_attention_rmsnorm.output",
                &[self.token_count, self.dimensions.hidden],
            )?;
            let post_op = SemanticOpDescriptor::new_rms_norm(
                vec![
                    self.tensors[attention_residual].view.clone(),
                    self.tensors[post_weight].view.clone(),
                ],
                vec![self.tensors[post_normed].view.clone()],
                1.0e-6,
                RmsNormScaleMode::OffsetOne,
            )?;
            self.add_semantic(
                &format!("layer.{layer}.post_attention_rmsnorm"),
                post_op,
                vec![attention_residual, post_weight],
                vec![post_normed],
                vec![],
                vec![WeightConsumerKey {
                    layer: Some(layer_index as u64),
                    role: WeightConsumer::PostAttentionNorm,
                }],
            )?;

            let gate_weight = self.weight_tensor(WeightConsumerKey {
                layer: Some(layer_index as u64),
                role: WeightConsumer::MlpGate,
            })?;
            let up_weight = self.weight_tensor(WeightConsumerKey {
                layer: Some(layer_index as u64),
                role: WeightConsumer::MlpUp,
            })?;
            let gate = self.activation(
                layer,
                "mlp.gate.output",
                &[self.token_count, self.dimensions.intermediate],
            )?;
            let up = self.activation(
                layer,
                "mlp.up.output",
                &[self.token_count, self.dimensions.intermediate],
            )?;
            self.add_matmul(
                &format!("layer.{layer}.mlp_gate_matmul"),
                post_normed,
                gate_weight,
                gate,
                WeightConsumer::MlpGate,
                layer,
            )?;
            self.add_matmul(
                &format!("layer.{layer}.mlp_up_matmul"),
                post_normed,
                up_weight,
                up,
                WeightConsumer::MlpUp,
                layer,
            )?;
            let silu = self.activation(
                layer,
                "mlp.silu_mul.output",
                &[self.token_count, self.dimensions.intermediate],
            )?;
            let silu_op = SemanticOpDescriptor::new(
                SemanticOpKind::SiluMul,
                vec![
                    self.tensors[gate].view.clone(),
                    self.tensors[up].view.clone(),
                ],
                vec![self.tensors[silu].view.clone()],
            )?;
            self.add_semantic(
                &format!("layer.{layer}.mlp_silu_mul"),
                silu_op,
                vec![gate, up],
                vec![silu],
                vec![],
                vec![],
            )?;
            let down_weight = self.weight_tensor(WeightConsumerKey {
                layer: Some(layer_index as u64),
                role: WeightConsumer::MlpDown,
            })?;
            let down = self.activation(
                layer,
                "mlp.down.output",
                &[self.token_count, self.dimensions.hidden],
            )?;
            self.add_matmul(
                &format!("layer.{layer}.mlp_down_matmul"),
                silu,
                down_weight,
                down,
                WeightConsumer::MlpDown,
                layer,
            )?;
            let mlp_residual = self.activation(
                layer,
                "mlp.residual.output",
                &[self.token_count, self.dimensions.hidden],
            )?;
            let add_op = SemanticOpDescriptor::new(
                SemanticOpKind::Add,
                vec![
                    self.tensors[attention_residual].view.clone(),
                    self.tensors[down].view.clone(),
                ],
                vec![self.tensors[mlp_residual].view.clone()],
            )?;
            self.add_semantic(
                &format!("layer.{layer}.mlp_residual_add"),
                add_op,
                vec![attention_residual, down],
                vec![mlp_residual],
                vec![],
                vec![],
            )?;
            hidden = mlp_residual;
        }

        let final_weight = self.weight_tensor(WeightConsumerKey {
            layer: None,
            role: WeightConsumer::FinalNorm,
        })?;
        let final_norm = self.add_tensor(
            "final_rmsnorm.output",
            view(DType::Bf16, &[self.token_count, self.dimensions.hidden])?,
        );
        let final_op = SemanticOpDescriptor::new_rms_norm(
            vec![
                self.tensors[hidden].view.clone(),
                self.tensors[final_weight].view.clone(),
            ],
            vec![self.tensors[final_norm].view.clone()],
            1.0e-6,
            RmsNormScaleMode::OffsetOne,
        )?;
        self.add_semantic(
            "final_rmsnorm",
            final_op,
            vec![hidden, final_weight],
            vec![final_norm],
            vec![],
            vec![WeightConsumerKey {
                layer: None,
                role: WeightConsumer::FinalNorm,
            }],
        )?;

        let output_role = if self.dimensions.tied_embeddings {
            WeightConsumer::EmbeddingAndTiedOutput
        } else {
            WeightConsumer::OutputProjection
        };
        let tied_weight = self.weight_tensor(WeightConsumerKey {
            layer: None,
            role: output_role,
        })?;
        let logits = self.add_tensor(
            if self.dimensions.tied_embeddings {
                "tied_lm_head.logits"
            } else {
                "lm_head.logits"
            },
            view(DType::Bf16, &[self.token_count, self.dimensions.vocab])?,
        );
        self.add_matmul(
            if self.dimensions.tied_embeddings {
                "tied_lm_head_matmul"
            } else {
                "lm_head_matmul"
            },
            final_norm,
            tied_weight,
            logits,
            output_role,
            u32::MAX,
        )?;
        let output_tokens =
            self.add_tensor("argmax.output", view(DType::I32, &[self.token_count])?);
        let argmax_op = SemanticOpDescriptor::new(
            SemanticOpKind::Argmax,
            vec![self.tensors[logits].view.clone()],
            vec![self.tensors[output_tokens].view.clone()],
        )?;
        self.add_semantic(
            "argmax",
            argmax_op,
            vec![logits],
            vec![output_tokens],
            vec![],
            vec![],
        )?;
        Ok(())
    }

    fn build_linear_attention(
        &mut self,
        layer: u32,
        normed: usize,
    ) -> Result<usize, QwenGraphError> {
        let layer_key = layer as u64;
        let qkv_weight = self.weight_tensor(WeightConsumerKey {
            layer: Some(layer_key),
            role: WeightConsumer::GdnInProjQkv,
        })?;
        let z_weight = self.weight_tensor(WeightConsumerKey {
            layer: Some(layer_key),
            role: WeightConsumer::GdnInProjZ,
        })?;
        let b_weight = self.weight_tensor(WeightConsumerKey {
            layer: Some(layer_key),
            role: WeightConsumer::GdnInProjB,
        })?;
        let a_weight = self.weight_tensor(WeightConsumerKey {
            layer: Some(layer_key),
            role: WeightConsumer::GdnInProjA,
        })?;
        let qkv = self.activation(
            layer,
            "linear.qkv.output",
            &[self.token_count, self.dimensions.linear_qkv_width],
        )?;
        let z = self.activation(
            layer,
            "linear.z.output",
            &[self.token_count, self.dimensions.linear_output_width],
        )?;
        let b = self.activation(
            layer,
            "linear.b.output",
            &[self.token_count, self.dimensions.linear_value_heads],
        )?;
        let a = self.activation(
            layer,
            "linear.a.output",
            &[self.token_count, self.dimensions.linear_value_heads],
        )?;
        self.add_matmul(
            &format!("layer.{layer}.linear.qkv_matmul"),
            normed,
            qkv_weight,
            qkv,
            WeightConsumer::GdnInProjQkv,
            layer,
        )?;
        self.add_matmul(
            &format!("layer.{layer}.linear.z_matmul"),
            normed,
            z_weight,
            z,
            WeightConsumer::GdnInProjZ,
            layer,
        )?;
        self.add_matmul(
            &format!("layer.{layer}.linear.b_matmul"),
            normed,
            b_weight,
            b,
            WeightConsumer::GdnInProjB,
            layer,
        )?;
        self.add_matmul(
            &format!("layer.{layer}.linear.a_matmul"),
            normed,
            a_weight,
            a,
            WeightConsumer::GdnInProjA,
            layer,
        )?;
        let conv = self.weight_tensor(WeightConsumerKey {
            layer: Some(layer_key),
            role: WeightConsumer::GdnConv1d,
        })?;
        let a_log = self.weight_tensor(WeightConsumerKey {
            layer: Some(layer_key),
            role: WeightConsumer::GdnALog,
        })?;
        let dt_bias = self.weight_tensor(WeightConsumerKey {
            layer: Some(layer_key),
            role: WeightConsumer::GdnDtBias,
        })?;
        let gdn_norm = self.weight_tensor(WeightConsumerKey {
            layer: Some(layer_key),
            role: WeightConsumer::GdnNorm,
        })?;
        let state_output = self.activation(
            layer,
            "linear.state.output",
            &[self.token_count, self.dimensions.linear_output_width],
        )?;
        let descriptor = LinearAttentionStateDescriptor::new_with_layout(
            layer,
            self.state_capacity,
            usize::try_from(self.dimensions.linear_qk_heads)
                .map_err(|_| QwenGraphError::Overflow("linear QK heads"))?,
            usize::try_from(self.dimensions.linear_value_heads)
                .map_err(|_| QwenGraphError::Overflow("linear value heads"))?,
            usize::try_from(self.dimensions.linear_head_dim)
                .map_err(|_| QwenGraphError::Overflow("linear head dimension"))?,
            usize::try_from(self.dimensions.linear_conv_kernel)
                .map_err(|_| QwenGraphError::Overflow("linear convolution kernel"))?,
        )
        .map_err(|error| QwenGraphError::InvalidPlan(error.to_string()))?;
        self.add_typed(
            &format!("layer.{layer}.linear_attention_state"),
            QwenGraphNodeKind::LinearAttentionState {
                layer,
                state: descriptor,
                output_shape: [self.token_count, self.dimensions.linear_output_width],
            },
            // This is the existing Stage C binding order. The projection
            // weights remain owned by their matmul nodes; these four are the
            // stateful node's actual direct weight inputs.
            vec![qkv, z, b, a, conv, a_log, dt_bias, gdn_norm],
            vec![state_output],
            vec![],
            vec![
                key(layer, WeightConsumer::GdnConv1d),
                key(layer, WeightConsumer::GdnALog),
                key(layer, WeightConsumer::GdnDtBias),
                key(layer, WeightConsumer::GdnNorm),
            ],
        )?;
        let out_weight = self.weight_tensor(WeightConsumerKey {
            layer: Some(layer_key),
            role: WeightConsumer::GdnOutProj,
        })?;
        let out = self.activation(
            layer,
            "linear.out.output",
            &[self.token_count, self.dimensions.hidden],
        )?;
        self.add_matmul(
            &format!("layer.{layer}.linear.out_matmul"),
            state_output,
            out_weight,
            out,
            WeightConsumer::GdnOutProj,
            layer,
        )?;
        let residual = self.activation(
            layer,
            "attention.residual.output",
            &[self.token_count, self.dimensions.hidden],
        )?;
        self.add_add(
            &format!("layer.{layer}.attention_residual_add"),
            self.layer_input_for_residual(layer)?,
            out,
            residual,
        )?;
        Ok(residual)
    }

    fn build_full_attention(
        &mut self,
        layer: u32,
        normed: usize,
        positions: usize,
    ) -> Result<usize, QwenGraphError> {
        let q_weight = self.weight_tensor(key(layer, WeightConsumer::AttentionQ))?;
        let k_weight = self.weight_tensor(key(layer, WeightConsumer::AttentionK))?;
        let v_weight = self.weight_tensor(key(layer, WeightConsumer::AttentionV))?;
        let q = self.activation(
            layer,
            "full.q.output",
            &[self.token_count, self.dimensions.full_q_width],
        )?;
        let k = self.activation(
            layer,
            "full.k.output",
            &[self.token_count, self.dimensions.full_kv_width],
        )?;
        let v = self.activation(
            layer,
            "full.v.output",
            &[self.token_count, self.dimensions.full_kv_width],
        )?;
        let q_node = self.add_matmul(
            &format!("layer.{layer}.full.q_matmul"),
            normed,
            q_weight,
            q,
            WeightConsumer::AttentionQ,
            layer,
        )?;
        let k_node = self.add_matmul(
            &format!("layer.{layer}.full.k_matmul"),
            normed,
            k_weight,
            k,
            WeightConsumer::AttentionK,
            layer,
        )?;
        let v_node = self.add_matmul(
            &format!("layer.{layer}.full.v_matmul"),
            normed,
            v_weight,
            v,
            WeightConsumer::AttentionV,
            layer,
        )?;
        let packed_q = self.add_alias(
            &format!("layer.{layer}.full.q_gate.packed"),
            q,
            view(
                DType::Bf16,
                &[
                    self.token_count,
                    self.dimensions.q_heads,
                    self.dimensions.head_dim * 2,
                ],
            )?,
            q_node,
        )?;
        let k_reshaped = self.add_alias(
            &format!("layer.{layer}.full.k.reshaped"),
            k,
            view(
                DType::Bf16,
                &[
                    self.token_count,
                    self.dimensions.kv_heads,
                    self.dimensions.head_dim,
                ],
            )?,
            k_node,
        )?;
        let v_reshaped = self.add_alias(
            &format!("layer.{layer}.full.v.reshaped"),
            v,
            view(
                DType::Bf16,
                &[
                    self.token_count,
                    self.dimensions.kv_heads,
                    self.dimensions.head_dim,
                ],
            )?,
            v_node,
        )?;
        let q_norm = self.weight_tensor(key(layer, WeightConsumer::AttentionQNorm))?;
        let k_norm = self.weight_tensor(key(layer, WeightConsumer::AttentionKNorm))?;
        let q_norm_expanded = self.add_tensor(
            &format!("layer.{layer}.full.q_norm.expanded"),
            view(
                DType::Bf16,
                &[self.dimensions.q_heads, self.dimensions.head_dim],
            )?,
        );
        let k_norm_expanded = self.add_tensor(
            &format!("layer.{layer}.full.k_norm.expanded"),
            view(
                DType::Bf16,
                &[self.dimensions.kv_heads, self.dimensions.head_dim],
            )?,
        );
        self.add_scale_materialization(
            &format!("layer.{layer}.full.q_norm.broadcast"),
            layer,
            q_norm,
            q_norm_expanded,
            u32::try_from(self.dimensions.q_heads)
                .map_err(|_| QwenGraphError::Overflow("query heads"))?,
            key(layer, WeightConsumer::AttentionQNorm),
        )?;
        self.add_scale_materialization(
            &format!("layer.{layer}.full.k_norm.broadcast"),
            layer,
            k_norm,
            k_norm_expanded,
            u32::try_from(self.dimensions.kv_heads)
                .map_err(|_| QwenGraphError::Overflow("KV heads"))?,
            key(layer, WeightConsumer::AttentionKNorm),
        )?;
        let q_output = self.activation(
            layer,
            "full.q.preprocessed",
            &[
                self.token_count,
                self.dimensions.q_heads,
                self.dimensions.head_dim,
            ],
        )?;
        let gate = self.activation(
            layer,
            "full.gate.preprocessed",
            &[
                self.token_count,
                self.dimensions.q_heads,
                self.dimensions.head_dim,
            ],
        )?;
        let k_output = self.activation(
            layer,
            "full.k.preprocessed",
            &[
                self.token_count,
                self.dimensions.kv_heads,
                self.dimensions.head_dim,
            ],
        )?;
        self.add_typed(
            &format!("layer.{layer}.attention_preprocess"),
            QwenGraphNodeKind::AttentionPreprocess {
                layer,
                token_count: self.token_count,
                q_heads: u32::try_from(self.dimensions.q_heads)
                    .map_err(|_| QwenGraphError::Overflow("query heads"))?,
                kv_heads: u32::try_from(self.dimensions.kv_heads)
                    .map_err(|_| QwenGraphError::Overflow("KV heads"))?,
                head_dim: u32::try_from(self.dimensions.head_dim)
                    .map_err(|_| QwenGraphError::Overflow("head dimension"))?,
            },
            vec![
                packed_q,
                k_reshaped,
                q_norm_expanded,
                k_norm_expanded,
                positions,
            ],
            vec![q_output, gate, k_output],
            vec![],
            vec![],
        )?;
        let kv = KvStateDescriptor::new_with_layout(
            layer,
            self.state_capacity,
            usize::try_from(self.dimensions.kv_heads)
                .map_err(|_| QwenGraphError::Overflow("KV heads"))?,
            usize::try_from(self.dimensions.head_dim)
                .map_err(|_| QwenGraphError::Overflow("KV head dimension"))?,
        )
        .map_err(|error| QwenGraphError::InvalidPlan(error.to_string()))?;
        let kv_node = self.add_typed(
            &format!("layer.{layer}.kv_append"),
            QwenGraphNodeKind::FullKvAppend { layer, state: kv },
            vec![k_output, v_reshaped],
            vec![],
            vec![],
            vec![],
        )?;
        let context = self.activation(
            layer,
            "full.causal_attention.output",
            &[
                self.token_count,
                self.dimensions.q_heads,
                self.dimensions.head_dim,
            ],
        )?;
        self.add_typed(
            &format!("layer.{layer}.causal_attention"),
            QwenGraphNodeKind::FullCausalAttention {
                layer,
                state: kv,
                query_shape: [
                    self.token_count,
                    self.dimensions.q_heads,
                    self.dimensions.head_dim,
                ],
                output_shape: [
                    self.token_count,
                    self.dimensions.q_heads,
                    self.dimensions.head_dim,
                ],
            },
            vec![q_output],
            vec![context],
            vec![kv_node],
            vec![],
        )?;
        let sigmoid = self.activation(
            layer,
            "full.sigmoid_mul.output",
            &[
                self.token_count,
                self.dimensions.q_heads,
                self.dimensions.head_dim,
            ],
        )?;
        let sigmoid_op = SemanticOpDescriptor::new(
            SemanticOpKind::SigmoidMul,
            vec![
                self.tensors[gate].view.clone(),
                self.tensors[context].view.clone(),
            ],
            vec![self.tensors[sigmoid].view.clone()],
        )?;
        self.add_semantic(
            &format!("layer.{layer}.sigmoid_gate"),
            sigmoid_op,
            vec![gate, context],
            vec![sigmoid],
            vec![],
            vec![],
        )?;
        let sigmoid_node = self.nodes.len() - 1;
        let o_input = self.add_alias(
            &format!("layer.{layer}.full.o.input"),
            sigmoid,
            view(
                DType::Bf16,
                &[self.token_count, self.dimensions.full_output_width],
            )?,
            sigmoid_node,
        )?;
        let o_weight = self.weight_tensor(key(layer, WeightConsumer::AttentionO))?;
        let o = self.activation(
            layer,
            "full.o.output",
            &[self.token_count, self.dimensions.hidden],
        )?;
        self.add_matmul(
            &format!("layer.{layer}.full.o_matmul"),
            o_input,
            o_weight,
            o,
            WeightConsumer::AttentionO,
            layer,
        )?;
        let residual = self.activation(
            layer,
            "attention.residual.output",
            &[self.token_count, self.dimensions.hidden],
        )?;
        self.add_add(
            &format!("layer.{layer}.attention_residual_add"),
            self.layer_input_for_residual(layer)?,
            o,
            residual,
        )?;
        Ok(residual)
    }

    fn layer_input_for_residual(&self, layer: u32) -> Result<usize, QwenGraphError> {
        let input_label = if layer == 0 {
            "embedding.output".to_owned()
        } else {
            format!("layer.{}.mlp.residual.output", layer - 1)
        };
        self.tensors
            .iter()
            .find(|tensor| tensor.name == input_label)
            .map(QwenGraphTensor::id)
            .ok_or_else(|| QwenGraphError::InvalidPlan("layer residual input is absent".to_owned()))
    }

    fn add_add(
        &mut self,
        label: &str,
        left: usize,
        right: usize,
        output: usize,
    ) -> Result<usize, QwenGraphError> {
        let operation = SemanticOpDescriptor::new(
            SemanticOpKind::Add,
            vec![
                self.tensors[left].view.clone(),
                self.tensors[right].view.clone(),
            ],
            vec![self.tensors[output].view.clone()],
        )?;
        self.add_semantic(
            label,
            operation,
            vec![left, right],
            vec![output],
            vec![],
            vec![],
        )
    }

    fn add_matmul(
        &mut self,
        label: &str,
        activation: usize,
        weight: usize,
        output: usize,
        role: WeightConsumer,
        layer: u32,
    ) -> Result<usize, QwenGraphError> {
        let operation = SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![
                self.tensors[activation].view.clone(),
                self.tensors[weight].view.clone(),
            ],
            vec![self.tensors[output].view.clone()],
        )?;
        let consumer = if layer == u32::MAX {
            WeightConsumerKey { layer: None, role }
        } else {
            key(layer, role)
        };
        self.add_semantic(
            label,
            operation,
            vec![activation, weight],
            vec![output],
            vec![],
            vec![consumer],
        )
    }

    fn add_semantic(
        &mut self,
        label: &str,
        operation: SemanticOpDescriptor,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        extra_dependencies: Vec<usize>,
        weights: Vec<WeightConsumerKey>,
    ) -> Result<usize, QwenGraphError> {
        self.add_node(
            label,
            QwenGraphNodeKind::Semantic(operation.kind()),
            Some(operation),
            inputs,
            outputs,
            extra_dependencies,
            weights,
        )
    }

    fn add_typed(
        &mut self,
        label: &str,
        kind: QwenGraphNodeKind,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        extra_dependencies: Vec<usize>,
        weights: Vec<WeightConsumerKey>,
    ) -> Result<usize, QwenGraphError> {
        self.add_node(
            label,
            kind,
            None,
            inputs,
            outputs,
            extra_dependencies,
            weights,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn add_node(
        &mut self,
        label: &str,
        kind: QwenGraphNodeKind,
        operation: Option<SemanticOpDescriptor>,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        extra_dependencies: Vec<usize>,
        weights: Vec<WeightConsumerKey>,
    ) -> Result<usize, QwenGraphError> {
        let mut dependencies = Vec::new();
        for input in &inputs {
            if let Some(producer) = self.producers[*input] {
                if !dependencies.contains(&producer) {
                    dependencies.push(producer);
                }
            }
        }
        for dependency in extra_dependencies {
            if !dependencies.contains(&dependency) {
                dependencies.push(dependency);
            }
        }
        let index = self.nodes.len();
        for &output in &outputs {
            if self.producers[output].is_some() {
                return Err(QwenGraphError::InvalidPlan(format!(
                    "tensor {} has multiple producers",
                    self.tensors[output].name
                )));
            }
        }
        self.nodes.push(QwenGraphNode {
            label: label.to_owned(),
            kind,
            operation,
            inputs,
            outputs: outputs.clone(),
            dependencies,
            weights,
        });
        for output in outputs {
            self.producers[output] = Some(index);
        }
        Ok(index)
    }

    fn add_tensor(&mut self, name: &str, view: TensorView) -> usize {
        let id = self.tensors.len();
        self.tensors.push(QwenGraphTensor {
            id,
            name: name.to_owned(),
            view,
            backing: QwenGraphTensorBacking::Owned,
        });
        self.producers.push(None);
        id
    }

    fn activation(
        &mut self,
        layer: u32,
        suffix: &str,
        shape: &[u64],
    ) -> Result<usize, QwenGraphError> {
        Ok(self.add_tensor(
            &format!("layer.{layer}.{suffix}"),
            view(DType::Bf16, shape)?,
        ))
    }

    fn add_scale_materialization(
        &mut self,
        label: &str,
        layer: u32,
        input: usize,
        output: usize,
        heads: u32,
        consumer: WeightConsumerKey,
    ) -> Result<(), QwenGraphError> {
        let input_view = self.tensors.get(input).ok_or_else(|| {
            QwenGraphError::InvalidPlan("scale materialization input is absent".to_owned())
        })?;
        let output_view = self.tensors.get(output).ok_or_else(|| {
            QwenGraphError::InvalidPlan("scale materialization output is absent".to_owned())
        })?;
        let head_count = usize::try_from(heads)
            .map_err(|_| QwenGraphError::Overflow("scale materialization heads"))?;
        if input_view.view.dtype() != DType::Bf16
            || output_view.view.dtype() != DType::Bf16
            || input_view.view.encoding() != Encoding::Unquantized
            || output_view.view.encoding() != Encoding::Unquantized
            || input_view.view.shape() != [256]
            || output_view.view.shape() != [head_count, 256]
        {
            return Err(QwenGraphError::InvalidPlan(
                "attention scale materialization requires BF16 unquantized [256] to [heads,256]"
                    .to_owned(),
            ));
        }
        let expected_elements = input_view
            .view
            .element_count()
            .checked_mul(u64::from(heads))
            .ok_or(QwenGraphError::Overflow("scale materialization elements"))?;
        let expected_bytes = input_view
            .view
            .payload_bytes()
            .checked_mul(u64::from(heads))
            .ok_or(QwenGraphError::Overflow("scale materialization bytes"))?;
        if output_view.view.element_count() != expected_elements
            || output_view.view.payload_bytes() != expected_bytes
        {
            return Err(QwenGraphError::InvalidPlan(
                "attention scale materialization has an invalid element or byte relationship"
                    .to_owned(),
            ));
        }
        if self.producers[output].is_some() {
            return Err(QwenGraphError::InvalidPlan(
                "scale materialization output already has a producer".to_owned(),
            ));
        }
        self.add_typed(
            label,
            QwenGraphNodeKind::AttentionScaleMaterialization {
                layer,
                heads,
                head_dim: 256,
            },
            vec![input],
            vec![output],
            vec![],
            vec![consumer],
        )?;
        Ok(())
    }

    /// Add a metadata-only alias/view. No payload is allocated or copied.
    /// Dtype, encoding, and payload byte length are checked before the alias
    /// relation and producer edge are recorded.
    fn add_alias(
        &mut self,
        name: &str,
        source: usize,
        alias_view: TensorView,
        producer: usize,
    ) -> Result<usize, QwenGraphError> {
        let source_view = self.tensors.get(source).ok_or_else(|| {
            QwenGraphError::InvalidPlan("alias source tensor is absent".to_owned())
        })?;
        let source_producer = self.producers[source].ok_or_else(|| {
            QwenGraphError::InvalidPlan("alias source has no producer".to_owned())
        })?;
        if producer >= self.nodes.len() {
            return Err(QwenGraphError::InvalidPlan(
                "alias producer node is absent".to_owned(),
            ));
        }
        if source_view.view.dtype() != alias_view.dtype()
            || source_view.view.encoding() != alias_view.encoding()
            || source_view.view.payload_bytes() != alias_view.payload_bytes()
            || source_producer != producer
        {
            return Err(QwenGraphError::InvalidPlan(
                "alias/view requires matching dtype, encoding, and payload bytes".to_owned(),
            ));
        }
        let id = self.tensors.len();
        self.tensors.push(QwenGraphTensor {
            id,
            name: name.to_owned(),
            view: alias_view,
            backing: QwenGraphTensorBacking::Alias { tensor_id: source },
        });
        self.producers.push(Some(producer));
        Ok(id)
    }

    fn weight_tensor(&mut self, consumer: WeightConsumerKey) -> Result<usize, QwenGraphError> {
        if let Some(&id) = self.weight_tensors.get(&consumer) {
            return Ok(id);
        }
        let binding = self.bindings.get(&consumer).ok_or_else(|| {
            QwenGraphError::InvalidPlan(format!("missing graph binding: {consumer:?}"))
        })?;
        let shape = binding.shape.clone();
        let name = binding.tensor_name.clone();
        let view = if self.fp8_tensor_names.contains(&name) {
            fp8_weight_view(
                &shape,
                self.fp8_dtype.ok_or_else(|| {
                    QwenGraphError::InvalidPlan("FP8 tensor set has no resident dtype".to_owned())
                })?,
            )?
        } else {
            view(to_dtype(binding.dtype)?, &shape)?
        };
        let id = self.add_tensor(&name, view);
        self.weight_tensors.insert(consumer, id);
        Ok(id)
    }
}

fn fp8_weight_view(shape: &[u64], dtype: DType) -> Result<TensorView, QwenGraphError> {
    let shape: Vec<usize> = shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| QwenGraphError::Overflow("FP8 tensor shape"))
        })
        .collect::<Result<_, _>>()?;
    Ok(TensorView::with_encoding(
        dtype,
        Encoding::Fp8Scaled {
            granularity: Fp8ScaleGranularity::OuterDimension,
            scale_dtype: DType::F32,
            resident: Fp8ResidentRepresentation::PackedBytes,
        },
        &shape,
    )?)
}

fn key(layer: u32, role: WeightConsumer) -> WeightConsumerKey {
    WeightConsumerKey {
        layer: Some(layer as u64),
        role,
    }
}

fn to_dtype(dtype: TensorDType) -> Result<DType, QwenGraphError> {
    match dtype {
        TensorDType::Bf16 => Ok(DType::Bf16),
        TensorDType::F16 => Ok(DType::F16),
        TensorDType::F32 => Ok(DType::F32),
        TensorDType::I32 => Ok(DType::I32),
        TensorDType::I64 => Err(QwenGraphError::UnsupportedDType(dtype)),
        TensorDType::U8 => Ok(DType::U8),
    }
}

fn view(dtype: DType, shape: &[u64]) -> Result<TensorView, QwenGraphError> {
    let shape: Vec<usize> = shape
        .iter()
        .map(|&dimension| {
            usize::try_from(dimension).map_err(|_| QwenGraphError::Overflow("tensor shape"))
        })
        .collect::<Result<_, _>>()?;
    Ok(TensorView::contiguous(dtype, &shape)?)
}

fn checked_layout(
    dtype: DType,
    encoding: Encoding,
    shape: &[u64],
) -> Result<(Vec<u64>, u64), QwenGraphError> {
    let mut strides = vec![0_u64; shape.len()];
    let mut stride = 1_u64;
    for (dimension, current) in shape.iter().zip(strides.iter_mut()).rev() {
        *current = stride;
        stride = stride
            .checked_mul(*dimension)
            .ok_or(QwenGraphError::Overflow("state shape"))?;
    }
    let elements = shape.iter().try_fold(1_u64, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or(QwenGraphError::Overflow("state elements"))
    })?;
    let byte_size = encoding
        .storage_bytes(dtype, elements)
        .map_err(|_| QwenGraphError::Overflow("state bytes"))?;
    Ok((strides, byte_size))
}

/// Structural metadata fixture shared by D1 host-only tests. It contains a
/// canonical synthetic load plan only; it never opens or reads cache payloads.
#[cfg(test)]
pub(crate) fn qwen35_execution_fixture() -> (QwenGraph, WeightLoadPlan) {
    tests::execution_fixture()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_weight_load_plan, read_model_lock};
    use std::path::PathBuf;

    fn repository_path(relative: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative)
    }

    fn fixed_lock() -> ModelLock {
        read_model_lock(repository_path("docs/models/locks/qwen3.5-4b-bf16.json"))
            .expect("fixed Qwen lock parses")
    }

    fn fixed_dimensions() -> QwenGraphDimensions {
        QwenGraphDimensions::from_spec(reviewed_qwen35_spec(&fixed_lock()).unwrap()).unwrap()
    }

    fn tensor_byte_size(dtype: TensorDType, shape: &[u64]) -> u64 {
        let width = match dtype {
            TensorDType::Bf16 | TensorDType::F16 => 2,
            TensorDType::F32 | TensorDType::I32 => 4,
            TensorDType::I64 => 8,
            TensorDType::U8 => 1,
        };
        shape.iter().product::<u64>() * width
    }

    fn descriptor(
        name: String,
        source_file: &str,
        dtype: TensorDType,
        shape: Vec<u64>,
        source_start: u64,
    ) -> TensorDescriptor {
        let byte_size = tensor_byte_size(dtype, &shape);
        TensorDescriptor {
            tensor_name: name,
            source_file: source_file.to_owned(),
            dtype,
            shape,
            header_length_field_bytes: 8,
            header_length_bytes: 0,
            data_buffer_start: source_start,
            data_offset_basis: "safetensors-data-buffer".to_owned(),
            data_offsets: [0, byte_size],
            absolute_byte_range: [source_start, source_start + byte_size],
            byte_size,
        }
    }

    /// Build accepted synthetic metadata only. This is a canonical load-plan
    /// fixture, not an exact real-cache catalog and never reads payload bytes.
    fn synthetic_descriptors(
        lock: &ModelLock,
        vision_count: usize,
        mtp_count: usize,
    ) -> Vec<TensorDescriptor> {
        let sources: Vec<(String, u64)> = lock
            .model
            .files
            .iter()
            .filter(|file| file.path.ends_with(".safetensors"))
            .map(|file| (file.path.clone(), file.size_bytes))
            .collect();
        let mut specs = Vec::new();
        let dimensions = fixed_dimensions();
        for consumer in expected_consumers(&QWEN35_LAYER_TYPES, true) {
            let (name, dtype, shape) =
                expected_weight(consumer, &QWEN35_LAYER_TYPES, dimensions).expect("known consumer");
            specs.push((name, dtype, shape));
        }
        for index in 0..vision_count {
            specs.push((
                format!("model.visual.synthetic_{index}"),
                TensorDType::Bf16,
                vec![1],
            ));
        }
        for index in 0..mtp_count {
            specs.push((format!("mtp.synthetic_{index}"), TensorDType::Bf16, vec![1]));
        }

        let mut source_index = 0;
        let mut source_offset = 17_u64;
        let mut descriptors = Vec::with_capacity(specs.len());
        for (name, dtype, shape) in specs {
            let byte_size = tensor_byte_size(dtype, &shape);
            loop {
                let (source_file, source_size) = &sources[source_index];
                if source_offset + byte_size <= *source_size {
                    descriptors.push(descriptor(name, source_file, dtype, shape, source_offset));
                    source_offset += byte_size;
                    break;
                }
                source_index += 1;
                source_offset = 17;
                assert!(
                    source_index < sources.len(),
                    "synthetic plan exceeds shards"
                );
            }
        }
        descriptors
    }

    fn synthetic_canonical_load_plan(lock: &ModelLock) -> WeightLoadPlan {
        let descriptors = synthetic_descriptors(lock, 297, 15);
        build_weight_load_plan(lock, descriptors.iter()).expect("canonical metadata plan builds")
    }

    pub(super) fn execution_fixture() -> (QwenGraph, WeightLoadPlan) {
        let lock = fixed_lock();
        let plan = synthetic_canonical_load_plan(&lock);
        let graph = build_qwen35_graph(&lock, &plan, 3, 17).expect("fixture graph builds");
        (graph, plan)
    }

    fn fixture_bindings(schedule: &[LayerType]) -> Vec<QwenGraphWeightBinding> {
        let dimensions = fixed_dimensions();
        expected_consumers(schedule, true)
            .into_iter()
            .map(|consumer| {
                let (tensor_name, dtype, shape) =
                    expected_weight(consumer, schedule, dimensions).expect("fixture role");
                QwenGraphWeightBinding {
                    consumer,
                    tensor_name,
                    classification: WeightClassification::Required,
                    dtype,
                    shape,
                    source_range: [0, 1],
                    destination_start: 0,
                }
            })
            .collect()
    }

    fn tiny_fixture(schedule: &[LayerType]) -> QwenGraph {
        GraphBuilder::new(GraphBuilderConfig {
            layer_types: schedule.to_vec(),
            dimensions: fixed_dimensions(),
            token_count: 3,
            state_capacity: 17,
            bindings: fixture_bindings(schedule),
            known_unconsumed: BTreeSet::new(),
            model_fingerprint: "fixture".to_owned(),
            plan_digest: [7; 32],
            fp8_tensor_names: BTreeSet::new(),
            fp8_dtype: None,
            fp8_sidecar_fingerprint: None,
        })
        .expect("fixture bindings")
        .build()
        .expect("fixture graph")
    }

    fn node_id(graph: &QwenGraph, label: &str) -> usize {
        graph
            .nodes()
            .iter()
            .position(|node| node.label() == label)
            .unwrap_or_else(|| panic!("missing node {label}"))
    }

    fn tensor_id(graph: &QwenGraph, name: &str) -> usize {
        graph
            .tensor_metadata()
            .iter()
            .find(|tensor| tensor.name() == name)
            .map(QwenGraphTensor::id)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
    }

    #[test]
    fn public_fixed_graph_is_structural_and_preserves_boundaries() {
        let lock = fixed_lock();
        let plan = synthetic_canonical_load_plan(&lock);
        assert_eq!(plan.entries.len(), QWEN35_PLAN_ENTRY_COUNT);
        for token_count in [1, 3, 17, 255, 256, 257] {
            let graph = build_qwen35_graph(&lock, &plan, token_count, 257).expect("graph builds");
            assert_eq!(graph.layer_types(), QWEN35_LAYER_TYPES.as_slice());
            assert_eq!(graph.token_count(), token_count);
            assert_eq!(graph.nodes().len(), 484);
            assert_eq!(graph.states().len(), 64);
            assert_eq!(
                graph
                    .states()
                    .iter()
                    .filter(|state| matches!(
                        state.kind(),
                        QwenGraphStateKind::LinearConvolution | QwenGraphStateKind::LinearRecurrent
                    ))
                    .count(),
                48
            );
            assert_eq!(
                graph
                    .states()
                    .iter()
                    .filter(|state| matches!(
                        state.kind(),
                        QwenGraphStateKind::FullKey | QwenGraphStateKind::FullValue
                    ))
                    .count(),
                16
            );
            assert_eq!(graph.weight_bindings().len(), QWEN35_REQUIRED_WEIGHT_COUNT);
            assert_eq!(graph.nodes()[0].label(), "embedding");
            assert_eq!(graph.nodes()[483].label(), "argmax");
            assert_eq!(
                graph.nodes()[0].kind(),
                QwenGraphNodeKind::Semantic(SemanticOpKind::Embedding)
            );
            assert_eq!(
                graph.nodes()[483].kind(),
                QwenGraphNodeKind::Semantic(SemanticOpKind::Argmax)
            );
            let preprocess = graph
                .nodes()
                .iter()
                .find(|node| matches!(node.kind(), QwenGraphNodeKind::AttentionPreprocess { .. }))
                .expect("deferred full attention preprocess");
            assert!(preprocess.operation().is_none());
            assert!(
                graph
                    .nodes()
                    .iter()
                    .all(|node| node.label().starts_with("embedding")
                        || node.label().starts_with("layer.")
                        || node.label() == "final_rmsnorm"
                        || node.label() == "tied_lm_head_matmul"
                        || node.label() == "argmax")
            );
            let full = graph
                .states()
                .iter()
                .find(|state| state.kind() == QwenGraphStateKind::FullKey)
                .expect("full key state");
            assert_eq!(full.shape(), &[257, 4, 256]);
            assert_eq!(full.dtype(), DType::F16);
            assert_eq!(full.strides(), &[4 * 256, 256, 1]);
            assert!(graph.weight_binding("model.visual.synthetic_0").is_err());
            assert!(graph.weight_binding("not-a-weight").is_err());
        }
    }

    #[test]
    fn mixed_schedule_has_explicit_bindings_aliases_and_edges() {
        let graph = tiny_fixture(&[
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
        ]);
        assert_eq!(graph.nodes().len(), 50);
        assert_eq!(
            graph
                .nodes()
                .iter()
                .map(QwenGraphNode::label)
                .collect::<Vec<_>>(),
            vec![
                "embedding",
                "layer.0.input_rmsnorm",
                "layer.0.linear.qkv_matmul",
                "layer.0.linear.z_matmul",
                "layer.0.linear.b_matmul",
                "layer.0.linear.a_matmul",
                "layer.0.linear_attention_state",
                "layer.0.linear.out_matmul",
                "layer.0.attention_residual_add",
                "layer.0.post_attention_rmsnorm",
                "layer.0.mlp_gate_matmul",
                "layer.0.mlp_up_matmul",
                "layer.0.mlp_silu_mul",
                "layer.0.mlp_down_matmul",
                "layer.0.mlp_residual_add",
                "layer.1.input_rmsnorm",
                "layer.1.full.q_matmul",
                "layer.1.full.k_matmul",
                "layer.1.full.v_matmul",
                "layer.1.full.q_norm.broadcast",
                "layer.1.full.k_norm.broadcast",
                "layer.1.attention_preprocess",
                "layer.1.kv_append",
                "layer.1.causal_attention",
                "layer.1.sigmoid_gate",
                "layer.1.full.o_matmul",
                "layer.1.attention_residual_add",
                "layer.1.post_attention_rmsnorm",
                "layer.1.mlp_gate_matmul",
                "layer.1.mlp_up_matmul",
                "layer.1.mlp_silu_mul",
                "layer.1.mlp_down_matmul",
                "layer.1.mlp_residual_add",
                "layer.2.input_rmsnorm",
                "layer.2.linear.qkv_matmul",
                "layer.2.linear.z_matmul",
                "layer.2.linear.b_matmul",
                "layer.2.linear.a_matmul",
                "layer.2.linear_attention_state",
                "layer.2.linear.out_matmul",
                "layer.2.attention_residual_add",
                "layer.2.post_attention_rmsnorm",
                "layer.2.mlp_gate_matmul",
                "layer.2.mlp_up_matmul",
                "layer.2.mlp_silu_mul",
                "layer.2.mlp_down_matmul",
                "layer.2.mlp_residual_add",
                "final_rmsnorm",
                "tied_lm_head_matmul",
                "argmax",
            ]
        );

        let linear = &graph.nodes()[node_id(&graph, "layer.0.linear_attention_state")];
        let linear_inputs: Vec<_> = linear
            .inputs()
            .iter()
            .map(|&id| graph.tensor_metadata()[id].name())
            .collect();
        assert_eq!(
            linear_inputs,
            vec![
                "layer.0.linear.qkv.output",
                "layer.0.linear.z.output",
                "layer.0.linear.b.output",
                "layer.0.linear.a.output",
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                "model.language_model.layers.0.linear_attn.A_log",
                "model.language_model.layers.0.linear_attn.dt_bias",
                "model.language_model.layers.0.linear_attn.norm.weight",
            ]
        );
        assert_eq!(
            linear.weight_consumers(),
            &[
                key(0, WeightConsumer::GdnConv1d),
                key(0, WeightConsumer::GdnALog),
                key(0, WeightConsumer::GdnDtBias),
                key(0, WeightConsumer::GdnNorm),
            ]
        );
        assert!(matches!(
            linear.kind(),
            QwenGraphNodeKind::LinearAttentionState {
                output_shape: [3, 4096],
                ..
            }
        ));
        for (label, role) in [
            ("layer.0.linear.qkv_matmul", WeightConsumer::GdnInProjQkv),
            ("layer.0.linear.z_matmul", WeightConsumer::GdnInProjZ),
            ("layer.0.linear.b_matmul", WeightConsumer::GdnInProjB),
            ("layer.0.linear.a_matmul", WeightConsumer::GdnInProjA),
        ] {
            assert_eq!(
                graph.nodes()[node_id(&graph, label)].weight_consumers(),
                &[key(0, role)]
            );
        }

        for (alias_name, source_name, expected_shape) in [
            (
                "layer.1.full.q_gate.packed",
                "layer.1.full.q.output",
                vec![3, 16, 512],
            ),
            (
                "layer.1.full.k.reshaped",
                "layer.1.full.k.output",
                vec![3, 4, 256],
            ),
            (
                "layer.1.full.v.reshaped",
                "layer.1.full.v.output",
                vec![3, 4, 256],
            ),
            (
                "layer.1.full.o.input",
                "layer.1.full.sigmoid_mul.output",
                vec![3, 4096],
            ),
        ] {
            let alias = &graph.tensor_metadata()[tensor_id(&graph, alias_name)];
            let source = &graph.tensor_metadata()[tensor_id(&graph, source_name)];
            assert_eq!(alias.view().shape(), expected_shape.as_slice());
            assert_eq!(alias.view().dtype(), source.view().dtype());
            assert_eq!(alias.view().encoding(), source.view().encoding());
            assert_eq!(alias.view().payload_bytes(), source.view().payload_bytes());
            assert_eq!(
                alias.backing(),
                QwenGraphTensorBacking::Alias {
                    tensor_id: source.id()
                }
            );
        }

        let q_norm = tensor_id(
            &graph,
            "model.language_model.layers.1.self_attn.q_norm.weight",
        );
        let k_norm = tensor_id(
            &graph,
            "model.language_model.layers.1.self_attn.k_norm.weight",
        );
        assert_eq!(graph.tensor_metadata()[q_norm].view().shape(), &[256]);
        assert_eq!(graph.tensor_metadata()[k_norm].view().shape(), &[256]);
        for (label, source, expanded, heads, role) in [
            (
                "layer.1.full.q_norm.broadcast",
                q_norm,
                "layer.1.full.q_norm.expanded",
                16,
                WeightConsumer::AttentionQNorm,
            ),
            (
                "layer.1.full.k_norm.broadcast",
                k_norm,
                "layer.1.full.k_norm.expanded",
                4,
                WeightConsumer::AttentionKNorm,
            ),
        ] {
            let node = &graph.nodes()[node_id(&graph, label)];
            let expanded = &graph.tensor_metadata()[tensor_id(&graph, expanded)];
            assert_eq!(node.inputs(), &[source]);
            assert_eq!(node.outputs(), &[expanded.id()]);
            assert_eq!(node.weight_consumers(), &[key(1, role)]);
            assert_eq!(expanded.backing(), QwenGraphTensorBacking::Owned);
            assert_eq!(expanded.view().shape(), &[heads, 256]);
            assert_eq!(expanded.view().payload_bytes(), heads as u64 * 256 * 2);
        }
        let preprocess = &graph.nodes()[node_id(&graph, "layer.1.attention_preprocess")];
        assert!(matches!(
            preprocess.kind(),
            QwenGraphNodeKind::AttentionPreprocess {
                layer: 1,
                token_count: 3,
                q_heads: 16,
                kv_heads: 4,
                head_dim: 256,
            }
        ));
        assert!(preprocess.operation().is_none());
        assert_eq!(preprocess.weight_consumers(), &[]);
        assert_eq!(
            preprocess.inputs()[2],
            tensor_id(&graph, "layer.1.full.q_norm.expanded")
        );
        assert_eq!(
            preprocess.inputs()[3],
            tensor_id(&graph, "layer.1.full.k_norm.expanded")
        );
        assert!(
            preprocess
                .dependencies()
                .contains(&node_id(&graph, "layer.1.full.q_norm.broadcast"))
        );
        assert!(
            preprocess
                .dependencies()
                .contains(&node_id(&graph, "layer.1.full.k_norm.broadcast"))
        );
        assert!(
            preprocess
                .dependencies()
                .contains(&node_id(&graph, "layer.1.full.q_matmul"))
        );
        assert!(
            preprocess
                .dependencies()
                .contains(&node_id(&graph, "layer.1.full.k_matmul"))
        );

        let kv = &graph.nodes()[node_id(&graph, "layer.1.kv_append")];
        let kv_descriptor = match kv.kind() {
            QwenGraphNodeKind::FullKvAppend { state, .. } => state,
            kind => panic!("unexpected KV node kind: {kind:?}"),
        };
        assert_eq!(kv.inputs().len(), 2);
        assert!(kv.outputs().is_empty());
        assert!(
            kv.dependencies()
                .contains(&node_id(&graph, "layer.1.attention_preprocess"))
        );
        assert!(
            kv.dependencies()
                .contains(&node_id(&graph, "layer.1.full.v_matmul"))
        );
        let causal = &graph.nodes()[node_id(&graph, "layer.1.causal_attention")];
        assert!(matches!(
            causal.kind(),
            QwenGraphNodeKind::FullCausalAttention {
                query_shape: [3, 16, 256],
                output_shape: [3, 16, 256],
                ..
            }
        ));
        assert!(
            causal
                .dependencies()
                .contains(&node_id(&graph, "layer.1.kv_append"))
        );
        assert_eq!(causal.inputs().len(), 1);
        assert_eq!(causal.weight_consumers(), &[]);
        let sigmoid = node_id(&graph, "layer.1.sigmoid_gate");
        let o_input = tensor_id(&graph, "layer.1.full.o.input");
        assert_eq!(
            graph.nodes()[sigmoid].outputs(),
            &[tensor_id(&graph, "layer.1.full.sigmoid_mul.output")]
        );
        assert_eq!(
            graph.nodes()[node_id(&graph, "layer.1.full.o_matmul")].inputs()[0],
            o_input
        );
        assert!(
            graph.nodes()[node_id(&graph, "layer.1.full.o_matmul")]
                .dependencies()
                .contains(&sigmoid)
        );

        for label in [
            "layer.1.attention_residual_add",
            "layer.2.attention_residual_add",
        ] {
            let node = &graph.nodes()[node_id(&graph, label)];
            assert_eq!(node.inputs().len(), 2);
            assert!(node.dependencies().len() >= 2);
        }
        assert!(
            graph.nodes()[node_id(&graph, "layer.1.attention_residual_add")]
                .dependencies()
                .contains(&node_id(&graph, "layer.0.mlp_residual_add"))
        );
        assert_eq!(
            graph.nodes()[node_id(&graph, "tied_lm_head_matmul")].inputs()[1],
            graph.nodes()[node_id(&graph, "embedding")].inputs()[0]
        );
        assert_eq!(graph.weight_bindings().len(), 41);
        assert_eq!(kv_descriptor.layer_id(), 1);
    }

    #[test]
    fn state_rows_owner_and_total_bytes_are_explicit() {
        let graph = tiny_fixture(&[LayerType::LinearAttention, LayerType::FullAttention]);
        let full_states: Vec<_> = graph
            .states()
            .iter()
            .filter(|state| {
                matches!(
                    state.kind(),
                    QwenGraphStateKind::FullKey | QwenGraphStateKind::FullValue
                )
            })
            .collect();
        assert_eq!(full_states.len(), 2);
        assert_ne!(full_states[0].kind(), full_states[1].kind());
        assert_eq!(full_states[0].descriptor(), full_states[1].descriptor());
        assert_eq!(full_states[0].byte_size(), 4 * 17 * 256 * 2);
        assert_eq!(full_states[1].byte_size(), 4 * 17 * 256 * 2);
        assert_eq!(
            graph.total_state_bytes(),
            graph
                .states()
                .iter()
                .map(QwenGraphState::byte_size)
                .sum::<u64>()
        );
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), QwenGraphNodeKind::FullKvAppend { .. }))
                .count(),
            1
        );
        let embedding_weight = graph.nodes()[node_id(&graph, "embedding")].inputs()[0];
        let tied_weight = graph.nodes()[node_id(&graph, "tied_lm_head_matmul")].inputs()[1];
        assert_eq!(embedding_weight, tied_weight);
        assert_eq!(
            graph.nodes()[node_id(&graph, "embedding")].weight_consumers(),
            graph.nodes()[node_id(&graph, "tied_lm_head_matmul")].weight_consumers()
        );
    }

    #[test]
    fn digest_or_content_tamper_and_metadata_boundaries_fail_closed() {
        let lock = fixed_lock();
        let plan = synthetic_canonical_load_plan(&lock);
        assert!(matches!(
            build_qwen35_graph(&lock, &plan, 0, 1),
            Err(QwenGraphError::ZeroTokenCount)
        ));
        assert!(matches!(
            build_qwen35_graph(&lock, &plan, 2, 1),
            Err(QwenGraphError::TokenCountExceedsCapacity { .. })
        ));
        assert!(matches!(
            build_qwen35_graph(&lock, &plan, 1, QWEN35_MAX_POSITION_EMBEDDINGS + 1),
            Err(QwenGraphError::CapacityExceedsMax { .. })
        ));
        assert!(matches!(
            build_qwen35_graph(&lock, &plan, 1, 0),
            Err(QwenGraphError::ZeroStateCapacity)
        ));
        assert!(matches!(
            checked_layout(DType::F32, Encoding::Unquantized, &[u64::MAX, 2]),
            Err(QwenGraphError::Overflow(_))
        ));
        assert!(matches!(
            to_dtype(TensorDType::I64),
            Err(QwenGraphError::UnsupportedDType(TensorDType::I64))
        ));

        let mut wrong_lock = lock.clone();
        wrong_lock.model.architecture.text_config.layer_types[3] = LayerType::LinearAttention;
        assert!(build_qwen35_graph(&wrong_lock, &plan, 1, 1).is_err());
        let mut wrong_tie = plan.clone();
        wrong_tie.tied_embeddings = false;
        assert!(build_qwen35_graph(&lock, &wrong_tie, 1, 1).is_err());
        let mut wrong_shape = plan.clone();
        wrong_shape.entries[0].shape[0] += 1;
        assert!(build_qwen35_graph(&lock, &wrong_shape, 1, 1).is_err());
        let mut wrong_overlap = plan.clone();
        wrong_overlap.entries[1].source_range = wrong_overlap.entries[0].source_range;
        assert!(build_qwen35_graph(&lock, &wrong_overlap, 1, 1).is_err());
        let mut missing = plan.clone();
        missing.entries.pop();
        assert!(build_qwen35_graph(&lock, &missing, 1, 1).is_err());
        let mut duplicate = plan.clone();
        duplicate.entries[1].tensor_name = duplicate.entries[0].tensor_name.clone();
        assert!(build_qwen35_graph(&lock, &duplicate, 1, 1).is_err());
        let mut wrong_class = plan.clone();
        wrong_class.entries[0].classification = WeightClassification::KnownUnconsumed;
        assert!(build_qwen35_graph(&lock, &wrong_class, 1, 1).is_err());

        assert!(
            build_weight_load_plan(&lock, synthetic_descriptors(&lock, 296, 16).iter()).is_err()
        );
        assert!(
            build_weight_load_plan(&lock, synthetic_descriptors(&lock, 298, 14).iter()).is_err()
        );

        let graph = build_qwen35_graph(&lock, &plan, 1, 1).expect("valid metadata graph");
        assert!(matches!(
            graph.weight_binding("model.visual.synthetic_1"),
            Err(QwenGraphDispatchError::KnownUnconsumedWeight(_))
        ));
        assert!(matches!(
            graph.weight_binding("unknown.weight"),
            Err(QwenGraphDispatchError::UnknownWeight(_))
        ));
    }

    #[test]
    fn invalid_alias_payload_length_is_rejected_before_edges() {
        let schedule = [LayerType::LinearAttention];
        let mut builder = GraphBuilder::new(GraphBuilderConfig {
            layer_types: schedule.to_vec(),
            dimensions: fixed_dimensions(),
            token_count: 3,
            state_capacity: 17,
            bindings: fixture_bindings(&schedule),
            known_unconsumed: BTreeSet::new(),
            model_fingerprint: "fixture".to_owned(),
            plan_digest: [9; 32],
            fp8_tensor_names: BTreeSet::new(),
            fp8_dtype: None,
            fp8_sidecar_fingerprint: None,
        })
        .expect("fixture bindings");
        let source = builder.add_tensor("source", view(DType::Bf16, &[4]).unwrap());
        builder
            .add_typed(
                "source.producer",
                QwenGraphNodeKind::Semantic(SemanticOpKind::Add),
                vec![],
                vec![source],
                vec![],
                vec![],
            )
            .expect("producer metadata");
        assert!(
            builder
                .add_alias("bad.alias", source, view(DType::Bf16, &[3]).unwrap(), 0)
                .is_err()
        );
        assert_eq!(builder.tensors.len(), 1);
        assert_eq!(builder.producers[source], Some(0));
    }
}
